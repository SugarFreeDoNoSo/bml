use crate::*;
use bml_domain::number::StreamLimits;
use std::{
    collections::BTreeSet,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

fn format(kind: FormatKind) -> Error {
    Error::Format {
        kind,
        offset: None,
        tensor: None,
        block: None,
    }
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b)
        .ok_or_else(|| format(FormatKind::ArithmeticOverflow))
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b)
        .ok_or_else(|| format(FormatKind::ArithmeticOverflow))
}
fn size(n: u64) -> Result<usize, Error> {
    usize::try_from(n).map_err(|_| format(FormatKind::ArithmeticOverflow))
}
fn hash(bytes: &[u8], expected: Digest, scope: IntegrityScope) -> Result<(), Error> {
    let actual = Digest::of(bytes);
    if expected == actual {
        Ok(())
    } else {
        Err(Error::Integrity {
            scope,
            expected,
            actual,
        })
    }
}
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .at
            .checked_add(n)
            .ok_or_else(|| format(FormatKind::ArithmeticOverflow))?;
        let out = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| format(FormatKind::SectionLayout))?;
        self.at = end;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn digest(&mut self) -> Result<Digest, Error> {
        Ok(Digest(self.take(32)?.try_into().unwrap()))
    }
    fn done(&self) -> Result<(), Error> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(format(FormatKind::SectionLayout))
        }
    }
}

/// Owns a read-only file and a validated catalog, not a snapshot of mutable files.
/// Payload integrity is checked on every read. Keep the source file immutable.
pub struct WeightReader {
    file: File,
    path: PathBuf,
    limits: ReadLimits,
    summary: FileSummary,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    blocks: Vec<BlockInfo>,
}
impl WeightReader {
    pub fn open(path: impl AsRef<Path>, limits: ReadLimits) -> Result<Self, Error> {
        limits.validate()?;
        let path = path.as_ref().to_path_buf();
        let io = |operation, source| Error::Io {
            operation,
            path: path.clone(),
            source,
        };
        let mut file = File::open(&path).map_err(|e| io(IoOperation::Open, e))?;
        let length = file.metadata().map_err(|e| io(IoOperation::Read, e))?.len();
        check_limit(Resource::MaxFileBytes, limits.max_file_bytes, Some(length))?;
        if length < HEADER_BYTES {
            return Err(format(FormatKind::HeaderLength));
        }
        let mut header = [0u8; 256];
        file.read_exact(&mut header)
            .map_err(|e| io(IoOperation::Read, e))?;
        let mut c = Cursor::new(&header);
        if c.take(8)? != MAGIC {
            return Err(format(FormatKind::Magic));
        }
        let version = u16::from_le_bytes(c.take(2)?.try_into().unwrap());
        let endian = c.take(1)?[0];
        let basis = c.take(1)?[0];
        for (kind, value, wanted) in [
            (UnsupportedKind::Version, version as u64, 1),
            (UnsupportedKind::Endian, endian as u64, 1),
            (UnsupportedKind::Basis, basis as u64, 2),
        ] {
            if value != wanted {
                return Err(Error::Unsupported { kind, value });
            }
        }
        if c.u32()? != 256 {
            return Err(format(FormatKind::HeaderLength));
        }
        let flags = c.u64()?;
        if flags != 0 {
            return Err(Error::Unsupported {
                kind: UnsupportedKind::Flags,
                value: flags,
            });
        }
        let file_bytes = c.u64()?;
        if file_bytes != length {
            return Err(format(FormatKind::FileLength));
        }
        let tensor_count = c.u64()?;
        let block_count = c.u64()?;
        check_limit(Resource::MaxTensors, limits.max_tensors, Some(tensor_count))?;
        check_limit(Resource::MaxBlocks, limits.max_blocks, Some(block_count))?;
        if tensor_count == 0 || block_count == 0 {
            return Err(format(FormatKind::BlockCoverage));
        }
        let metadata_offset = c.u64()?;
        let metadata_bytes = c.u64()?;
        let tensors_offset = c.u64()?;
        let tensors_bytes = c.u64()?;
        let blocks_offset = c.u64()?;
        let blocks_bytes = c.u64()?;
        let payload_offset = c.u64()?;
        let payload_bytes = c.u64()?;
        let source_sha256 = c.digest()?;
        let catalog_sha256 = c.digest()?;
        let header_sha256 = c.digest()?;
        if c.take(48)?.iter().any(|&b| b != 0) {
            return Err(format(FormatKind::Reserved));
        }
        header[176..208].fill(0);
        hash(&header, header_sha256, IntegrityScope::Header)?;
        check_limit(
            Resource::MaxMetadataBytes,
            limits.max_metadata_bytes,
            Some(metadata_bytes),
        )?;
        check_limit(
            Resource::MaxIndexBytes,
            limits.max_index_bytes,
            Some(add(tensors_bytes, blocks_bytes)?),
        )?;
        if metadata_offset != HEADER_BYTES
            || tensors_offset != add(metadata_offset, metadata_bytes)?
            || blocks_offset != add(tensors_offset, tensors_bytes)?
            || blocks_bytes != mul(block_count, BLOCK_RECORD_BYTES)?
            || payload_offset != add(blocks_offset, blocks_bytes)?
            || file_bytes != add(payload_offset, payload_bytes)?
            || tensors_bytes < mul(tensor_count, 49)?
        {
            return Err(format(FormatKind::SectionLayout));
        }
        let catalog_len = payload_offset - HEADER_BYTES;
        let mut catalog = Vec::new();
        catalog
            .try_reserve_exact(size(catalog_len)?)
            .map_err(|_| Error::Limit {
                resource: Resource::MaxIndexBytes,
                limit: limits.max_index_bytes,
                required: Some(catalog_len),
                location: None,
            })?;
        catalog.resize(size(catalog_len)?, 0);
        file.read_exact(&mut catalog)
            .map_err(|e| io(IoOperation::Read, e))?;
        hash(&catalog, catalog_sha256, IntegrityScope::Catalog)?;
        let metadata_end = size(metadata_bytes)?;
        let tensor_end = size(add(metadata_bytes, tensors_bytes)?)?;
        let metadata = crate::metadata::decode(&catalog[..metadata_end], &limits)?;
        check_limit(
            Resource::MaxOutputBytes,
            metadata.limits.max_output_bytes,
            Some(file_bytes),
        )?;
        let mut tensors = Vec::new();
        let mut names = BTreeSet::new();
        let mut t = Cursor::new(&catalog[metadata_end..tensor_end]);
        let mut next_block = 0;
        let mut total_terms = 0;
        for id in 0..tensor_count {
            let name_bytes = t.u32()? as u64;
            let rank = t.u32()?;
            let source_dtype = SourceDtype::try_from(t.u32()?)?;
            if t.u32()? != 0 {
                return Err(format(FormatKind::Reserved));
            }
            let element_count = t.u64()?;
            let first_block = t.u64()?;
            let count = t.u64()?;
            check_limit(
                Resource::MaxNameBytes,
                limits.max_name_bytes,
                Some(name_bytes),
            )?;
            if !(1..=4).contains(&rank) || name_bytes == 0 {
                return Err(format(FormatKind::TensorShape));
            }
            let mut dimensions = Vec::new();
            let mut product = 1;
            for _ in 0..rank {
                let dim = t.u64()?;
                if dim == 0 {
                    return Err(format(FormatKind::TensorShape));
                }
                product = mul(product, dim)?;
                dimensions.push(dim);
            }
            if product != element_count {
                return Err(format(FormatKind::TensorShape));
            }
            let name = std::str::from_utf8(t.take(size(name_bytes)?)?)
                .map_err(|_| format(FormatKind::TensorShape))?
                .to_owned();
            if name.contains('\0') || !names.insert(name.clone()) {
                return Err(format(FormatKind::DuplicateTensor));
            }
            if count == 0 || first_block != next_block {
                return Err(format(FormatKind::BlockCoverage));
            }
            next_block = add(next_block, count)?;
            if next_block > block_count {
                return Err(format(FormatKind::BlockCoverage));
            }
            total_terms = add(total_terms, element_count)?;
            check_limit(
                Resource::MaxTotalTerms,
                limits.max_total_terms,
                Some(total_terms),
            )?;
            tensors.push(TensorInfo {
                id: TensorId(id),
                name,
                ggml_dimensions: dimensions,
                source_dtype,
                element_count,
                first_block: BlockId(first_block),
                block_count: count,
            });
        }
        t.done()?;
        if next_block != block_count {
            return Err(format(FormatKind::BlockCoverage));
        }
        let mut b = Cursor::new(&catalog[tensor_end..]);
        let mut blocks = Vec::new();
        let mut data_offset = payload_offset;
        for tensor in &tensors {
            let mut first_element = 0;
            for _ in 0..tensor.block_count {
                let id = BlockId(blocks.len() as u64);
                let tensor_id = TensorId(b.u64()?);
                let first = b.u64()?;
                let term_count = b.u64()?;
                let offset = b.u64()?;
                let data_bytes = b.u64()?;
                let useful_bits = b.u64()?;
                let digest = b.digest()?;
                check_limit(
                    Resource::MaxBlockTerms,
                    limits.max_block_terms.min(metadata.limits.max_block_terms),
                    Some(term_count),
                )?;
                check_limit(
                    Resource::MaxBlockBytes,
                    limits.max_block_bytes.min(metadata.limits.max_block_bytes),
                    Some(data_bytes),
                )?;
                let expected_bytes =
                    add(24, add(useful_bits / 8, u64::from(useful_bits % 8 != 0))?)?;
                if tensor_id != tensor.id
                    || first != first_element
                    || term_count == 0
                    || term_count > useful_bits
                    || offset != data_offset
                    || data_bytes != expected_bytes
                {
                    return Err(format(FormatKind::BlockCoverage));
                }
                first_element = add(first_element, term_count)?;
                data_offset = add(data_offset, data_bytes)?;
                if first_element > tensor.element_count || data_offset > file_bytes {
                    return Err(format(FormatKind::BlockCoverage));
                }
                blocks.push(BlockInfo {
                    id,
                    tensor_id,
                    first_element: first,
                    term_count,
                    data_offset: offset,
                    data_bytes,
                    useful_bits,
                    digest,
                });
            }
            if first_element != tensor.element_count {
                return Err(format(FormatKind::BlockCoverage));
            }
        }
        b.done()?;
        if data_offset != file_bytes {
            return Err(format(FormatKind::BlockCoverage));
        }
        let summary = FileSummary {
            version,
            basis,
            file_bytes,
            metadata_bytes,
            tensors_bytes,
            blocks_bytes,
            payload_bytes,
            tensor_count,
            block_count,
            source_sha256,
            catalog_sha256,
        };
        Ok(Self {
            file,
            path,
            limits,
            summary,
            metadata,
            tensors,
            blocks,
        })
    }
    pub fn summary(&self) -> &FileSummary {
        &self.summary
    }
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }
    pub fn tensor(&self, id: TensorId) -> Result<&TensorInfo, Error> {
        usize::try_from(id.0)
            .ok()
            .and_then(|i| self.tensors.get(i))
            .ok_or(Error::Index {
                kind: IndexKind::Tensor,
                index: id.0,
                count: self.summary.tensor_count,
            })
    }
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn block_info(&self, id: BlockId) -> Result<&BlockInfo, Error> {
        usize::try_from(id.0)
            .ok()
            .and_then(|i| self.blocks.get(i))
            .ok_or(Error::Index {
                kind: IndexKind::Block,
                index: id.0,
                count: self.summary.block_count,
            })
    }
    pub fn read_block(&mut self, id: BlockId) -> Result<DecodedBlock, Error> {
        let info = self.block_info(id)?.clone();
        let io = |operation, source| Error::Io {
            operation,
            path: self.path.clone(),
            source,
        };
        if self
            .file
            .metadata()
            .map_err(|e| io(IoOperation::Read, e))?
            .len()
            != self.summary.file_bytes
        {
            return Err(format(FormatKind::FileLength));
        }
        self.file
            .seek(SeekFrom::Start(info.data_offset))
            .map_err(|e| io(IoOperation::Seek, e))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size(info.data_bytes)?)
            .map_err(|_| Error::Limit {
                resource: Resource::MaxBlockBytes,
                limit: self.limits.max_block_bytes,
                required: Some(info.data_bytes),
                location: None,
            })?;
        bytes.resize(size(info.data_bytes)?, 0);
        self.file
            .read_exact(&mut bytes)
            .map_err(|e| io(IoOperation::Read, e))?;
        hash(&bytes, info.digest, IntegrityScope::Block(id))?;
        let mut c = Cursor::new(&bytes);
        if c.take(8)? != b"BMLS\x01\x02\0\0"
            || c.u64()? != info.term_count
            || c.u64()? != info.useful_bits
        {
            return Err(format(FormatKind::BlockHeader));
        }
        // Bound each expanded tree before the domain decoder allocates its words.
        let max_bits = self
            .limits
            .max_weight_bits
            .min(self.metadata.limits.max_weight_bits);
        let mut pending = 1u64;
        let mut tree_bits = 0u64;
        let mut terms = 0u64;
        for i in 0..info.useful_bits {
            let one = bytes[24 + size(i / 8)?] & (1 << (7 - i % 8)) != 0;
            tree_bits += 1;
            check_limit(Resource::MaxWeightBits, max_bits, Some(tree_bits))?;
            if one {
                pending = add(pending, 1)?;
            } else {
                pending -= 1;
            }
            if pending == 0 {
                terms += 1;
                pending = 1;
                tree_bits = 0;
            }
        }
        if tree_bits != 0 || terms != info.term_count {
            return Err(Error::Stream {
                block: id,
                source: StreamError::Length,
            });
        }
        let terms = Bml::decode_stream(
            &bytes,
            StreamLimits {
                max_bits: size(info.useful_bits)?,
                max_terms: size(info.term_count)?,
                max_depth: size(
                    self.limits
                        .max_weight_depth
                        .min(self.metadata.limits.max_weight_depth),
                )?,
            },
        )
        .map_err(|source| Error::Stream { block: id, source })?;
        Ok(DecodedBlock {
            id,
            tensor_id: info.tensor_id,
            first_element: info.first_element,
            terms,
        })
    }
    pub fn verify_all(&mut self) -> Result<VerificationReport, Error> {
        let mut report = VerificationReport {
            blocks_checked: 0,
            terms_checked: 0,
            bytes_checked: 0,
            all_payload_blocks_checked: false,
        };
        for index in 0..self.summary.block_count {
            let block = self.read_block(BlockId(index))?;
            report.blocks_checked += 1;
            report.terms_checked = add(report.terms_checked, block.terms.len() as u64)?;
            report.bytes_checked = add(report.bytes_checked, self.blocks[size(index)?].data_bytes)?;
        }
        report.all_payload_blocks_checked = true;
        Ok(report)
    }
}
