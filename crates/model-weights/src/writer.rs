use crate::*;
use bml_domain::number::{ExactStep, StreamLimits};
use sha2::{Digest as _, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

fn io(operation: IoOperation, path: &Path, source: std::io::Error) -> Error {
    Error::Io {
        operation,
        path: path.into(),
        source,
    }
}
fn overflow() -> Error {
    Error::Format {
        kind: FormatKind::ArithmeticOverflow,
        offset: None,
        tensor: None,
        block: None,
    }
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b).ok_or_else(overflow)
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn size(n: u64) -> Result<usize, Error> {
    usize::try_from(n).map_err(|_| overflow())
}
fn invalid(kind: FormatKind) -> Error {
    Error::Format {
        kind,
        offset: None,
        tensor: None,
        block: None,
    }
}

struct Temporary {
    path: PathBuf,
    file: Option<File>,
}
impl Temporary {
    fn create(destination: &Path) -> Result<Self, Error> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        for _ in 0..128 {
            let path = parent.join(format!(
                ".bmlgguf-{}-{}.tmp",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file: Some(file),
                    })
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io(IoOperation::CreateTemporary, &path, e)),
            }
        }
        Err(io(
            IoOperation::CreateTemporary,
            destination,
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "temporary name attempts exhausted",
            ),
        ))
    }
    fn file(&mut self) -> &mut File {
        self.file
            .as_mut()
            .expect("temporary handle is open until cleanup")
    }
    fn cleanup(&mut self) -> Result<(), Error> {
        self.file.take();
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(io(IoOperation::Cleanup, &self.path, e)),
        }
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

/// Transactional BMLGGUF v1 writer. Every failed mutating operation poisons it.
/// Publication uses a same-directory hard link, which never replaces a destination.
pub struct WeightWriter {
    destination: PathBuf,
    metadata: Vec<u8>,
    source: Digest,
    options: WriteOptions,
    tensors: Vec<TensorInfo>,
    blocks: Vec<BlockInfo>,
    state: WriterState,
    written: u64,
    total_terms: u64,
    tensor_bytes: u64,
    payload_bytes: u64,
    replay_steps: u64,
    memory: std::sync::Arc<crate::memory::MemoryBudget>,
    spool: Temporary,
}
impl WeightWriter {
    pub fn create(
        destination: impl AsRef<Path>,
        metadata: Metadata,
        source_sha256: Digest,
        options: WriteOptions,
    ) -> Result<Self, Error> {
        Self::create_accounted(
            destination,
            metadata,
            source_sha256,
            options,
            Default::default(),
        )
    }
    pub(crate) fn create_accounted(
        destination: impl AsRef<Path>,
        mut metadata: Metadata,
        source_sha256: Digest,
        options: WriteOptions,
        memory: std::sync::Arc<crate::memory::MemoryBudget>,
    ) -> Result<Self, Error> {
        options.validate()?;
        let destination = destination.as_ref().to_path_buf();
        match fs::symlink_metadata(&destination) {
            Ok(_) => return Err(Error::DestinationExists { path: destination }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(io(IoOperation::Open, &destination, e)),
        }
        // The actual requested options, not caller-invented provenance, are stored.
        metadata.limits = options.conversion;
        metadata.gguf.sort_by(|a, b| a.key.cmp(&b.key));
        memory.check(0, options.conversion.max_resident_bytes)?;
        let cap = (options.conversion.max_resident_bytes - memory.external()) / 8;
        let metadata = crate::metadata_encode::encode(&metadata, &options.limits, cap)?;
        memory.check(
            add(mul(metadata.len() as u64, 8)?, 65_536)?,
            options.conversion.max_resident_bytes,
        )?;
        let spool = Temporary::create(&destination)?;
        Ok(Self {
            destination,
            metadata,
            source: source_sha256,
            options,
            tensors: Vec::new(),
            blocks: Vec::new(),
            state: WriterState::Ready,
            written: 0,
            total_terms: 0,
            tensor_bytes: 0,
            payload_bytes: 0,
            replay_steps: 0,
            memory,
            spool,
        })
    }
    fn require(&self, operation: WriterOperation, state: WriterState) -> Result<(), Error> {
        if self.state == state {
            Ok(())
        } else {
            Err(Error::State {
                operation,
                state: self.state,
            })
        }
    }
    fn poison<T>(&mut self, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_err() {
            self.state = WriterState::Aborted;
        }
        result
    }
    fn check_size(&self, tensor_bytes: u64, blocks: u64, payload: u64) -> Result<u64, Error> {
        let index = add(tensor_bytes, mul(blocks, BLOCK_RECORD_BYTES)?)?;
        check_limit(
            Resource::MaxIndexBytes,
            self.options.limits.max_index_bytes,
            Some(index),
        )?;
        let bytes = add(
            add(add(HEADER_BYTES, self.metadata.len() as u64)?, index)?,
            payload,
        )?;
        check_limit(
            Resource::MaxOutputBytes,
            self.options
                .conversion
                .max_output_bytes
                .min(self.options.limits.max_file_bytes),
            Some(bytes),
        )?;
        Ok(bytes)
    }
    // Conservative accounting for retained catalog structures, codec buffers and
    // replay's bounded term/rational working sets. This is not measured RSS.
    fn resident(&self, index: u64, block: u64, proof: u64) -> Result<(), Error> {
        let catalog = add(self.metadata.len() as u64, index)?;
        let required = add(add(mul(catalog, 8)?, mul(block, 8)?)?, add(proof, 65_536)?)?;
        self.memory
            .check(required, self.options.conversion.max_resident_bytes)
    }
    #[cfg(feature = "gguf-import")]
    pub(crate) fn check_external_memory(&self, bytes: u64) -> Result<(), Error> {
        self.memory.set_external(bytes);
        self.resident(
            add(self.tensor_bytes, mul(self.blocks.len() as u64, 80)?)?,
            0,
            0,
        )
    }
    pub fn begin_tensor(
        &mut self,
        name: &str,
        ggml_dimensions: &[u64],
        source_dtype: SourceDtype,
    ) -> Result<TensorId, Error> {
        let result = self.begin(name, ggml_dimensions, source_dtype);
        self.poison(result)
    }
    fn begin(&mut self, name: &str, dims: &[u64], dtype: SourceDtype) -> Result<TensorId, Error> {
        self.require(WriterOperation::BeginTensor, WriterState::Ready)?;
        check_limit(
            Resource::MaxTensors,
            self.options.limits.max_tensors,
            Some(add(self.tensors.len() as u64, 1)?),
        )?;
        check_limit(
            Resource::MaxNameBytes,
            self.options.limits.max_name_bytes,
            Some(name.len() as u64),
        )?;
        if name.is_empty() || name.contains('\0') || self.tensors.iter().any(|t| t.name == name) {
            return Err(invalid(FormatKind::DuplicateTensor));
        }
        if !(1..=4).contains(&dims.len()) || dims.contains(&0) {
            return Err(invalid(FormatKind::TensorShape));
        }
        let elements = dims.iter().try_fold(1u64, |n, &dim| mul(n, dim))?;
        check_limit(
            Resource::MaxTotalTerms,
            self.options.limits.max_total_terms,
            Some(add(self.total_terms, elements)?),
        )?;
        u32::try_from(name.len()).map_err(|_| overflow())?;
        let tensor_bytes = add(
            self.tensor_bytes,
            add(40, add(mul(dims.len() as u64, 8)?, name.len() as u64)?)?,
        )?;
        self.check_size(tensor_bytes, self.blocks.len() as u64, self.payload_bytes)?;
        self.resident(add(tensor_bytes, mul(self.blocks.len() as u64, 80)?)?, 0, 0)?;
        let id = TensorId(self.tensors.len() as u64);
        self.tensors.push(TensorInfo {
            id,
            name: name.into(),
            ggml_dimensions: dims.into(),
            source_dtype: dtype,
            element_count: elements,
            first_block: BlockId(self.blocks.len() as u64),
            block_count: 0,
        });
        self.tensor_bytes = tensor_bytes;
        self.written = 0;
        self.state = WriterState::TensorOpen;
        Ok(id)
    }
    pub fn write_block(&mut self, weights: &[CertifiedBml]) -> Result<BlockId, Error> {
        let result = self.block(weights);
        self.poison(result)
    }
    fn block(&mut self, weights: &[CertifiedBml]) -> Result<BlockId, Error> {
        self.require(WriterOperation::WriteBlock, WriterState::TensorOpen)?;
        let tensor = self.tensors.last().unwrap();
        if weights.is_empty() {
            return Err(invalid(FormatKind::BlockCoverage));
        }
        let count = weights.len() as u64;
        check_limit(
            Resource::MaxBlockTerms,
            self.options
                .limits
                .max_block_terms
                .min(self.options.conversion.max_block_terms),
            Some(count),
        )?;
        if add(self.written, count)? > tensor.element_count {
            return Err(invalid(FormatKind::BlockCoverage));
        }
        let next_blocks = add(self.blocks.len() as u64, 1)?;
        check_limit(
            Resource::MaxBlocks,
            self.options.limits.max_blocks,
            Some(next_blocks),
        )?;
        let bits = weights
            .iter()
            .try_fold(0u64, |n, w| add(n, w.term().bit_len() as u64))?;
        let block_bytes = add(24, add(bits / 8, u64::from(bits % 8 != 0))?)?;
        check_limit(
            Resource::MaxBlockBytes,
            self.options
                .limits
                .max_block_bytes
                .min(self.options.conversion.max_block_bytes),
            Some(block_bytes),
        )?;
        let next_payload = add(self.payload_bytes, block_bytes)?;
        self.check_size(self.tensor_bytes, next_blocks, next_payload)?;
        let synth = ConversionOptions {
            write: self.options,
            ..Default::default()
        }
        .effective_synthesis()?;
        let mut steps = self.replay_steps;
        for (i, weight) in weights.iter().enumerate() {
            let location = WeightLocation {
                tensor_id: tensor.id,
                tensor_name: tensor.name.clone(),
                flat_index: add(self.written, i as u64)?,
            };
            check_limit(
                Resource::MaxWeightBits,
                synth.max_bits as u64,
                Some(weight.term().bit_len() as u64),
            )
            .map_err(|e| locate(e, &location))?;
            let step_count = weight.derivation().steps.len() as u64;
            check_limit(
                Resource::MaxSynthesisStepsPerWeight,
                synth.max_steps as u64,
                Some(step_count),
            )
            .map_err(|e| locate(e, &location))?;
            steps = add(steps, step_count)?;
            check_limit(
                Resource::MaxSynthesisStepsTotal,
                self.options.conversion.max_synthesis_steps_total,
                Some(steps),
            )
            .map_err(|e| locate(e, &location))?;
            let proof = add(
                synth.max_total_bits as u64 / 8,
                mul(
                    step_count,
                    add(
                        std::mem::size_of::<ExactStep>() as u64,
                        add(synth.rational.max_bits as u64 / 2, 256)?,
                    )?,
                )?,
            )?;
            self.resident(
                add(self.tensor_bytes, mul(next_blocks, 80)?)?,
                block_bytes,
                proof,
            )?;
            weight
                .derivation()
                .verify(weight.term(), weight.target(), synth)
                .map_err(|source| Error::Synthesis { location, source })?;
        }
        let id = BlockId(self.blocks.len() as u64);
        let terms: Vec<_> = weights.iter().map(|w| w.term().clone()).collect();
        let bytes = Bml::encode_stream(
            &terms,
            StreamLimits {
                max_bits: size(bits)?,
                max_terms: weights.len(),
                max_depth: size(
                    self.options
                        .limits
                        .max_weight_depth
                        .min(self.options.conversion.max_weight_depth),
                )?,
            },
        )
        .map_err(|source| Error::Stream { block: id, source })?;
        let path = self.spool.path.clone();
        self.spool
            .file()
            .write_all(&bytes)
            .map_err(|e| io(IoOperation::Write, &path, e))?;
        self.blocks.push(BlockInfo {
            id,
            tensor_id: tensor.id,
            first_element: self.written,
            term_count: count,
            data_offset: self.payload_bytes,
            data_bytes: bytes.len() as u64,
            useful_bits: bits,
            digest: Digest::of(&bytes),
        });
        self.tensors.last_mut().unwrap().block_count += 1;
        self.written += count;
        self.payload_bytes = next_payload;
        self.replay_steps = steps;
        Ok(id)
    }
    pub fn end_tensor(&mut self) -> Result<(), Error> {
        let result = (|| {
            self.require(WriterOperation::EndTensor, WriterState::TensorOpen)?;
            if self.written != self.tensors.last().unwrap().element_count {
                return Err(invalid(FormatKind::BlockCoverage));
            }
            self.total_terms = add(self.total_terms, self.written)?;
            self.state = WriterState::Ready;
            Ok(())
        })();
        self.poison(result)
    }
    pub fn abort(mut self) -> Result<(), Error> {
        self.state = WriterState::Aborted;
        self.spool.cleanup()
    }
    pub fn finish(mut self) -> Result<WriteReport, Error> {
        self.require(WriterOperation::Finish, WriterState::Ready)?;
        if self.tensors.is_empty() {
            return Err(invalid(FormatKind::BlockCoverage));
        }
        let file_bytes = self.check_size(
            self.tensor_bytes,
            self.blocks.len() as u64,
            self.payload_bytes,
        )?;
        let peak_temporary_bytes = add(file_bytes, self.payload_bytes)?;
        let block_bytes = mul(self.blocks.len() as u64, 80)?;
        self.resident(
            add(self.tensor_bytes, block_bytes)?,
            self.blocks.iter().map(|b| b.data_bytes).max().unwrap_or(0),
            0,
        )?;
        let payload_offset = file_bytes - self.payload_bytes;
        let mut final_file = Temporary::create(&self.destination)?;
        let final_path = final_file.path.clone();
        let output = final_file.file();
        output
            .write_all(&[0u8; 256])
            .map_err(|e| io(IoOperation::Write, &final_path, e))?;
        let mut catalog = Sha256::new();
        write_catalog(output, &final_path, &mut catalog, &self.metadata)?;
        for tensor in &self.tensors {
            let mut record = Vec::with_capacity(size(add(
                40,
                add(
                    mul(tensor.ggml_dimensions.len() as u64, 8)?,
                    tensor.name.len() as u64,
                )?,
            )?)?);
            record.extend((tensor.name.len() as u32).to_le_bytes());
            record.extend((tensor.ggml_dimensions.len() as u32).to_le_bytes());
            record.extend(u32::from(tensor.source_dtype).to_le_bytes());
            record.extend(0u32.to_le_bytes());
            for n in [
                tensor.element_count,
                tensor.first_block.0,
                tensor.block_count,
            ] {
                record.extend(n.to_le_bytes());
            }
            for dim in &tensor.ggml_dimensions {
                record.extend(dim.to_le_bytes());
            }
            record.extend(tensor.name.as_bytes());
            write_catalog(output, &final_path, &mut catalog, &record)?;
        }
        for block in &self.blocks {
            let mut record = Vec::with_capacity(80);
            for n in [
                block.tensor_id.0,
                block.first_element,
                block.term_count,
                add(payload_offset, block.data_offset)?,
                block.data_bytes,
                block.useful_bits,
            ] {
                record.extend(n.to_le_bytes());
            }
            record.extend(block.digest.0);
            write_catalog(output, &final_path, &mut catalog, &record)?;
        }
        let spool_path = self.spool.path.clone();
        self.spool
            .file()
            .seek(SeekFrom::Start(0))
            .map_err(|e| io(IoOperation::Seek, &spool_path, e))?;
        let copied = std::io::copy(self.spool.file(), output)
            .map_err(|e| io(IoOperation::Write, &final_path, e))?;
        if copied != self.payload_bytes {
            return Err(invalid(FormatKind::FileLength));
        }
        let summary = FileSummary {
            version: 1,
            basis: 2,
            file_bytes,
            metadata_bytes: self.metadata.len() as u64,
            tensors_bytes: self.tensor_bytes,
            blocks_bytes: block_bytes,
            payload_bytes: self.payload_bytes,
            tensor_count: self.tensors.len() as u64,
            block_count: self.blocks.len() as u64,
            source_sha256: self.source,
            catalog_sha256: Digest(catalog.finalize().into()),
        };
        let header = header(&summary)?;
        output
            .seek(SeekFrom::Start(0))
            .map_err(|e| io(IoOperation::Seek, &final_path, e))?;
        output
            .write_all(&header)
            .map_err(|e| io(IoOperation::Write, &final_path, e))?;
        output
            .flush()
            .map_err(|e| io(IoOperation::Flush, &final_path, e))?;
        let mut reader = WeightReader::open(&final_path, self.options.limits)?;
        reader.verify_all()?;
        drop(reader);
        output
            .seek(SeekFrom::Start(0))
            .map_err(|e| io(IoOperation::Seek, &final_path, e))?;
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let count = output
                .read(&mut buffer)
                .map_err(|e| io(IoOperation::Read, &final_path, e))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        let output_sha256 = Digest(digest.finalize().into());
        // Remove the spool before the commit point so cleanup errors cannot
        // masquerade as a failed publication after a valid destination exists.
        self.spool.cleanup()?;
        final_file.file.take();
        fs::hard_link(&final_path, &self.destination).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                Error::DestinationExists {
                    path: self.destination.clone(),
                }
            } else {
                io(IoOperation::Publish, &self.destination, e)
            }
        })?;
        // Removing the extra hard link is best-effort after commit. Drop retries.
        let _ = final_file.cleanup();
        Ok(WriteReport {
            path: self.destination.clone(),
            summary,
            output_sha256,
            terms_written: self.total_terms,
            peak_temporary_bytes,
        })
    }
}
fn locate(error: Error, location: &WeightLocation) -> Error {
    match error {
        Error::Limit {
            resource,
            limit,
            required,
            ..
        } => Error::Limit {
            resource,
            limit,
            required,
            location: Some(location.clone()),
        },
        e => e,
    }
}
fn write_catalog(
    file: &mut File,
    path: &Path,
    digest: &mut Sha256,
    bytes: &[u8],
) -> Result<(), Error> {
    file.write_all(bytes)
        .map_err(|e| io(IoOperation::Write, path, e))?;
    digest.update(bytes);
    Ok(())
}

fn header(s: &FileSummary) -> Result<[u8; 256], Error> {
    let mut h = [0u8; 256];
    h[..8].copy_from_slice(&MAGIC);
    h[8..10].copy_from_slice(&1u16.to_le_bytes());
    h[10] = 1;
    h[11] = 2;
    h[12..16].copy_from_slice(&256u32.to_le_bytes());
    let tensors_offset = add(256, s.metadata_bytes)?;
    let blocks_offset = add(tensors_offset, s.tensors_bytes)?;
    for (at, n) in [
        (24, s.file_bytes),
        (32, s.tensor_count),
        (40, s.block_count),
        (48, 256),
        (56, s.metadata_bytes),
        (64, tensors_offset),
        (72, s.tensors_bytes),
        (80, blocks_offset),
        (88, s.blocks_bytes),
        (96, add(blocks_offset, s.blocks_bytes)?),
        (104, s.payload_bytes),
    ] {
        h[at..at + 8].copy_from_slice(&n.to_le_bytes());
    }
    h[112..144].copy_from_slice(&s.source_sha256.0);
    h[144..176].copy_from_slice(&s.catalog_sha256.0);
    let digest = Digest::of(&h);
    h[176..208].copy_from_slice(&digest.0);
    Ok(h)
}

#[cfg(test)]
mod accounted_creation_tests {
    use super::*;

    #[test]
    fn importer_reservation_bounds_metadata_encoding_before_allocation() {
        let directory =
            std::env::temp_dir().join(format!("bml-shared-encoding-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(directory.clone());
        let memory = std::sync::Arc::new(crate::memory::MemoryBudget::default());
        memory.set_external(79_500);
        let mut options = WriteOptions::default();
        options.conversion.max_resident_bytes = 80_000;
        let tool = ToolIdentity {
            name: "test".into(),
            version: "1".into(),
            revision: "fixture".into(),
        };
        let metadata = Metadata {
            schema: 1,
            profile: "bml-exact-decoded-f32-v1".into(),
            source_gguf_version: 3,
            converter: tool.clone(),
            codec: tool,
            limits: options.conversion,
            gguf: vec![],
        };
        let result = WeightWriter::create_accounted(
            directory.join("out"),
            metadata,
            Digest([0; 32]),
            options,
            memory,
        );
        // The encoder may use only floor((80000 - 79500) / 8) = 62 bytes.
        // A later resident-memory failure would mean it already over-allocated.
        assert!(matches!(
            result,
            Err(Error::Limit {
                resource: Resource::MaxMetadataBytes,
                limit: 62,
                ..
            })
        ));
        assert_eq!(std::fs::read_dir(directory).unwrap().count(), 0);
    }
}
