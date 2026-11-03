//! Bound source metadata and descriptor allocations before the mmap reader parses them.
use crate::*;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

struct Scan<'a> {
    file: File,
    path: &'a Path,
    length: u64,
    at: u64,
    limits: &'a ReadLimits,
    leaves: u64,
    arrays: u64,
}
fn format() -> Error {
    Error::Format {
        kind: FormatKind::ArithmeticOverflow,
        offset: None,
        tensor: None,
        block: None,
    }
}
impl Scan<'_> {
    fn read<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let mut bytes = [0; N];
        self.file
            .read_exact(&mut bytes)
            .map_err(|source| Error::Io {
                operation: IoOperation::Read,
                path: self.path.into(),
                source,
            })?;
        self.at = self.at.checked_add(N as u64).ok_or_else(format)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.read()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.read()?))
    }
    fn skip(&mut self, n: u64) -> Result<(), Error> {
        let end = self.at.checked_add(n).ok_or_else(format)?;
        if end > self.length {
            return Err(Error::Format {
                kind: FormatKind::FileLength,
                offset: Some(self.at),
                tensor: None,
                block: None,
            });
        }
        self.file
            .seek(SeekFrom::Start(end))
            .map_err(|source| Error::Io {
                operation: IoOperation::Seek,
                path: self.path.into(),
                source,
            })?;
        self.at = end;
        Ok(())
    }
    fn string(&mut self, name: bool) -> Result<(), Error> {
        let n = self.u64()?;
        let (resource, cap) = if name {
            (Resource::MaxNameBytes, self.limits.max_name_bytes)
        } else {
            (Resource::MaxStringBytes, self.limits.max_string_bytes)
        };
        check_limit(resource, cap, Some(n))?;
        if name && n == 0 {
            return Err(Error::Metadata {
                key: None,
                reason: MetadataReason::InvalidValue,
            });
        }
        self.skip(n)
    }
    fn value(&mut self, kind: u32) -> Result<(), Error> {
        if kind == 9 {
            let element = self.u32()?;
            let count = self.u64()?;
            if element == 9 {
                return Err(Error::Metadata {
                    key: None,
                    reason: MetadataReason::NestedArray,
                });
            }
            self.arrays = self.arrays.checked_add(count).ok_or_else(format)?;
            check_limit(
                Resource::MaxMetadataArrayElements,
                self.limits.max_metadata_array_elements,
                Some(self.arrays),
            )?;
            for _ in 0..count {
                self.value(element)?;
                self.metadata_bytes()?;
            }
        } else {
            self.leaves = self.leaves.checked_add(1).ok_or_else(format)?;
            match kind {
                0 | 1 | 7 => self.skip(1)?,
                2 | 3 => self.skip(2)?,
                4..=6 => self.skip(4)?,
                8 => self.string(false)?,
                10..=12 => self.skip(8)?,
                _ => {
                    return Err(Error::Metadata {
                        key: None,
                        reason: MetadataReason::InvalidType,
                    })
                }
            }
        }
        Ok(())
    }
    fn metadata_bytes(&self) -> Result<(), Error> {
        check_limit(
            Resource::MaxMetadataBytes,
            self.limits.max_metadata_bytes,
            Some(self.at - 24),
        )
    }
}

/// Conservative owned metadata/descriptor working-set estimate, excluding mmap RSS.
pub(crate) fn inspect(path: &Path, options: &ConversionOptions) -> Result<u64, Error> {
    let file = File::open(path).map_err(|source| Error::Io {
        operation: IoOperation::Open,
        path: path.into(),
        source,
    })?;
    let length = file
        .metadata()
        .map_err(|source| Error::Io {
            operation: IoOperation::Read,
            path: path.into(),
            source,
        })?
        .len();
    let limits = &options.write.limits;
    check_limit(Resource::MaxFileBytes, limits.max_file_bytes, Some(length))?;
    let mut s = Scan {
        file,
        path,
        length,
        at: 0,
        limits,
        leaves: 0,
        arrays: 0,
    };
    if s.read::<4>()? != *b"GGUF" {
        return Err(Error::Gguf {
            tensor: None,
            source: bml_gguf::GgufError::InvalidMagic(0),
        });
    }
    let version = s.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(Error::Unsupported {
            kind: UnsupportedKind::GgufVersion,
            value: version.into(),
        });
    }
    let tensors = s.u64()?;
    let metadata = s.u64()?;
    check_limit(Resource::MaxTensors, limits.max_tensors, Some(tensors))?;
    check_limit(
        Resource::MaxMetadataEntries,
        limits.max_metadata_entries,
        Some(metadata),
    )?;
    for _ in 0..metadata {
        s.string(true)?;
        let kind = s.u32()?;
        s.value(kind)?;
        s.metadata_bytes()?;
    }
    let metadata_bytes = s.at - 24;
    let tensor_start = s.at;
    let mut total = 0u64;
    for _ in 0..tensors {
        s.string(true)?;
        let rank = s.u32()?;
        if !(1..=4).contains(&rank) {
            return Err(Error::Format {
                kind: FormatKind::TensorShape,
                offset: Some(s.at - 4),
                tensor: None,
                block: None,
            });
        }
        let mut elements = 1u64;
        for _ in 0..rank {
            let dim = s.u64()?;
            if dim == 0 {
                return Err(format());
            }
            elements = elements.checked_mul(dim).ok_or_else(format)?;
        }
        total = total.checked_add(elements).ok_or_else(format)?;
        check_limit(Resource::MaxTotalTerms, limits.max_total_terms, Some(total))?;
        s.skip(12)?;
        check_limit(
            Resource::MaxIndexBytes,
            limits.max_index_bytes,
            Some(s.at - tensor_start),
        )?;
    }
    let accounted = metadata_bytes
        .checked_add(s.at - tensor_start)
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| {
            s.leaves
                .checked_add(metadata)?
                .checked_add(tensors)?
                .checked_mul(256)?
                .checked_add(n)
        })
        .ok_or_else(format)?;
    check_limit(
        Resource::MaxResidentBytes,
        options.write.conversion.max_resident_bytes,
        Some(accounted),
    )?;
    Ok(accounted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_only_preflight_accepts_exact_eof_and_accounts_catalog() {
        // Preflight scans descriptors, not tensor payloads. A 24-byte header,
        // one 14-byte U8 metadata record, and one 33-byte tensor descriptor
        // must fit exactly at EOF. The public GGUF reader later checks data.
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.push(b'k');
        bytes.extend(0u32.to_le_bytes());
        bytes.push(7);
        bytes.extend(1u64.to_le_bytes());
        bytes.push(b'w');
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        assert_eq!(bytes.len(), 71);
        let path =
            std::env::temp_dir().join(format!("bml-preflight-budget-{}", std::process::id()));
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let cleanup = Cleanup(path.clone());
        std::fs::write(&path, bytes).unwrap();
        // Catalog bytes carry factor eight; one scalar leaf, one metadata
        // entry and one tensor each reserve 256 bytes of owned bookkeeping.
        let expected = 47 * 8 + 3 * 256;
        let mut options = ConversionOptions::default();
        options.write.conversion.max_resident_bytes = expected;
        assert_eq!(inspect(&path, &options).unwrap(), expected);
        options.write.limits.max_metadata_bytes = 13;
        assert!(matches!(
            inspect(&path, &options),
            Err(Error::Limit {
                resource: Resource::MaxMetadataBytes,
                required: Some(14),
                ..
            })
        ));
        options.write.limits.max_metadata_bytes = 14;
        assert_eq!(inspect(&path, &options).unwrap(), expected);
        options.write.conversion.max_resident_bytes = expected - 1;
        assert!(matches!(inspect(&path, &options), Err(Error::Limit {
            resource: Resource::MaxResidentBytes, required: Some(n), ..
        }) if n == expected));
        drop(cleanup);
    }
}
