//! Explicit, bounded conversion of decoded GGUF weights to certified BML trees.
use crate::*;
use bml_domain::number::ExactStep;
use bml_gguf::{GgufDataType as D, GgufMetadataValue as V, GgufReader};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    time::{Duration, Instant},
};

fn malformed() -> Error {
    Error::Format {
        kind: FormatKind::ArithmeticOverflow,
        offset: None,
        tensor: None,
        block: None,
    }
}
fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b).ok_or_else(malformed)
}
fn mul(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b).ok_or_else(malformed)
}
// Keep elapsed-time accounting independent of the clock so it can be checked
// using exact durations, including phases already containing earlier work.
fn accumulate_time(total: &mut Duration, elapsed: Duration) {
    *total += elapsed;
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
        other => other,
    }
}
fn metadata_error() -> Error {
    Error::Metadata {
        key: None,
        reason: MetadataReason::InvalidType,
    }
}

fn metadata_value(value: &V) -> Result<MetadataValue, Error> {
    Ok(match value {
        V::U8(x) => MetadataValue::U8(*x),
        V::I8(x) => MetadataValue::I8(*x),
        V::U16(x) => MetadataValue::U16(*x),
        V::I16(x) => MetadataValue::I16(*x),
        V::U32(x) => MetadataValue::U32(*x),
        V::I32(x) => MetadataValue::I32(*x),
        V::U64(x) => MetadataValue::U64(*x),
        V::I64(x) => MetadataValue::I64(*x),
        V::F32(x) => MetadataValue::F32Bits(x.to_bits()),
        V::F64(x) => MetadataValue::F64Bits(x.to_bits()),
        V::Bool(x) => MetadataValue::Bool(*x),
        V::String(x) => MetadataValue::String(x.clone()),
        V::Array(kind, values) => {
            macro_rules! values {
                ($variant:ident,$convert:expr) => {{
                    let mut out = Vec::new();
                    for value in values {
                        if let V::$variant(v) = value {
                            out.push($convert(v));
                        } else {
                            return Err(metadata_error());
                        }
                    }
                    out
                }};
            }
            MetadataValue::Array(match kind {
                D::U8 => MetadataArray::U8(values!(U8, |v: &u8| *v)),
                D::I8 => MetadataArray::I8(values!(I8, |v: &i8| *v)),
                D::U16 => MetadataArray::U16(values!(U16, |v: &u16| *v)),
                D::I16 => MetadataArray::I16(values!(I16, |v: &i16| *v)),
                D::U32 => MetadataArray::U32(values!(U32, |v: &u32| *v)),
                D::I32 => MetadataArray::I32(values!(I32, |v: &i32| *v)),
                D::U64 => MetadataArray::U64(values!(U64, |v: &u64| *v)),
                D::I64 => MetadataArray::I64(values!(I64, |v: &i64| *v)),
                D::F32 => MetadataArray::F32Bits(values!(F32, |v: &f32| v.to_bits())),
                D::F64 => MetadataArray::F64Bits(values!(F64, |v: &f64| v.to_bits())),
                D::Bool => MetadataArray::Bool(values!(Bool, |v: &bool| *v)),
                D::String => MetadataArray::String(values!(String, |v: &String| v.clone())),
                D::Array => {
                    return Err(Error::Metadata {
                        key: None,
                        reason: MetadataReason::NestedArray,
                    })
                }
                _ => return Err(metadata_error()),
            })
        }
    })
}
fn identity(name: &str, revision: &str) -> ToolIdentity {
    ToolIdentity {
        name: name.into(),
        version: env!("CARGO_PKG_VERSION").into(),
        revision: revision.into(),
    }
}
fn certificate_bytes(value: &CertifiedBml) -> Result<u64, Error> {
    // Account expanded term, proof steps and rational words, plus owned headers.
    let rational =
        value.target().numerator_words_le().len() + value.target().denominator_words_le().len();
    add(
        add(
            (value.term().bit_len() as u64).div_ceil(64) * 8,
            mul(
                value.derivation().steps.len() as u64,
                std::mem::size_of::<ExactStep>() as u64,
            )?,
        )?,
        add(
            mul(rational as u64, 8)?,
            std::mem::size_of::<CertifiedBml>() as u64,
        )?,
    )
}

/// Convert supported GGUF tensor encodings to an autonomous BMLGGUF file.
///
/// Input must remain immutable while mapped. A successful report counts actual
/// emitted synthesis proof steps, not FLOPs or wall-clock instructions. Optional
/// cache entries are keyed by exact decoded F32 bits (both zeros normalize to zero).
pub fn from_gguf(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    options: ConversionOptions,
) -> Result<ConversionReport, Error> {
    options.validate()?;
    let source = source.as_ref();
    let destination = destination.as_ref();
    if source == destination
        || (destination.exists()
            && fs::canonicalize(source).ok() == fs::canonicalize(destination).ok())
    {
        return Err(Error::SourceDestinationConflict {
            source: source.into(),
            destination: destination.into(),
        });
    }
    match fs::symlink_metadata(destination) {
        Ok(_) => {
            return Err(Error::DestinationExists {
                path: destination.into(),
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(Error::Io {
                operation: IoOperation::Open,
                path: destination.into(),
                source,
            })
        }
    }
    let started = Instant::now();
    let base_memory = crate::gguf_preflight::inspect(source, &options)?;
    let reader = GgufReader::open(source).map_err(|source| Error::Gguf {
        tensor: None,
        source,
    })?;
    let source_sha256 = Digest::of(reader.bytes());
    let source_bytes = reader.len() as u64;
    let mut entries = Vec::new();
    for (key, value) in reader.metadata() {
        entries.push(GgufMetadataEntry {
            key: key.clone(),
            value: metadata_value(value)?,
        });
    }
    let metadata = Metadata {
        schema: 1,
        profile: "bml-exact-decoded-f32-v1".into(),
        source_gguf_version: reader.header().version,
        converter: identity("bml-model-weights", "exact-rational-synthesis-v1"),
        codec: identity("bml-gguf", "f32-f16-q4_0-q8_0-q6_k-v1"),
        limits: options.write.conversion,
        gguf: entries,
    };
    let mut read_decode_time = started.elapsed();
    let start = Instant::now();
    let memory = std::sync::Arc::new(crate::memory::MemoryBudget::default());
    memory.set_external(base_memory);
    let mut writer = WeightWriter::create_accounted(
        destination,
        metadata,
        source_sha256,
        options.write,
        memory.clone(),
    )?;
    let mut write_verify_time = start.elapsed();
    let mut synthesis_certification_time = Duration::ZERO;
    let mut weights_synthesized = 0;
    let mut weights_reused = 0;
    let mut synthesis_steps = 0;
    let mut tree_bits = 0;
    let mut tree_bytes = 0;
    let mut cache: BTreeMap<u32, (CertifiedBml, u64)> = BTreeMap::new();
    let mut cache_bytes = 0u64;
    let mut peak_cache_bytes = 0;
    let max_terms = options
        .write
        .limits
        .max_block_terms
        .min(options.write.conversion.max_block_terms);
    let max_bytes = options
        .write
        .limits
        .max_block_bytes
        .min(options.write.conversion.max_block_bytes);
    let synthesis = options.effective_synthesis()?;
    let synthesis_memory = add(
        synthesis.max_total_bits as u64 / 8,
        mul(
            synthesis.max_steps as u64,
            add(
                std::mem::size_of::<ExactStep>() as u64,
                add(synthesis.rational.max_bits as u64 / 2, 256)?,
            )?,
        )?,
    )?;
    for (index, tensor) in reader.tensor_infos().iter().enumerate() {
        let dtype = SourceDtype::try_from(tensor.data_type as u32)?;
        let view = reader
            .tensor_view(&tensor.name)
            .map_err(|source| Error::Gguf {
                tensor: Some(tensor.name.clone()),
                source,
            })?;
        // Fixed codec geometry: at most 256 decoded floats per chunk.
        // Scalars batch 256 elements; quantized formats decode one block.
        let (encoded_block, decoded_block, decode_chunk) = match dtype {
            SourceDtype::F32 => (4, 1, 1024),
            SourceDtype::F16 => (2, 1, 512),
            SourceDtype::Q4_0 => (18, 32, 18),
            SourceDtype::Q8_0 => (34, 32, 34),
            SourceDtype::Q6K => (210, 256, 210),
        };
        let start = Instant::now();
        writer.begin_tensor(&tensor.name, &tensor.dims, dtype)?;
        accumulate_time(&mut write_verify_time, start.elapsed());
        let mut flat_index = 0u64;
        let mut pending: Vec<CertifiedBml> = Vec::new();
        let mut pending_bits = 0u64;
        let mut pending_memory = 0u64;
        for bytes in view.bytes().chunks(decode_chunk) {
            let start = Instant::now();
            let elements = bytes.len() / encoded_block * decoded_block;
            let decoded =
                bml_gguf::decode_f32(tensor.data_type, bytes, elements).map_err(|source| {
                    Error::Gguf {
                        tensor: Some(tensor.name.clone()),
                        source,
                    }
                })?;
            accumulate_time(&mut read_decode_time, start.elapsed());
            for value in decoded {
                let location = WeightLocation {
                    tensor_id: TensorId(index as u64),
                    tensor_name: tensor.name.clone(),
                    flat_index,
                };
                if !value.is_finite() {
                    return Err(Error::NonFiniteWeight {
                        location,
                        decoded_f32_bits: value.to_bits(),
                    });
                }
                // For nonempty pending, the previous iteration accepted the
                // same base/cache/scratch plus twice pending_memory. This
                // smaller reservation cannot newly exceed the cap. Flushes
                // are governed below by the term and encoded-byte limits.
                let working = add(
                    add(base_memory, cache_bytes)?,
                    add(pending_memory, add(synthesis_memory, 65536)?)?,
                )?;
                writer
                    .check_external_memory(working)
                    .map_err(|e| locate(e, &location))?;
                let start = Instant::now();
                let key = if value == 0.0 { 0 } else { value.to_bits() };
                let certified = if let Some((certified, _)) = cache.get(&key) {
                    weights_reused += 1;
                    certified.clone()
                } else {
                    let remaining = options
                        .write
                        .conversion
                        .max_synthesis_steps_total
                        .checked_sub(synthesis_steps)
                        .ok_or_else(malformed)?;
                    check_limit(Resource::MaxSynthesisStepsTotal, remaining, Some(1))
                        .map_err(|e| locate(e, &location))?;
                    let mut bounded = synthesis;
                    bounded.max_steps = bounded
                        .max_steps
                        .min(usize::try_from(remaining).unwrap_or(usize::MAX));
                    let certified = Bml::from_f64_exact_with_options(f64::from(value), bounded)
                        .map_err(|source| Error::Synthesis {
                            location: location.clone(),
                            source,
                        })?;
                    synthesis_steps =
                        add(synthesis_steps, certified.derivation().steps.len() as u64)?;
                    weights_synthesized += 1;
                    let charge = add(certificate_bytes(&certified)?, 64)?;
                    let previous_cache = cache_bytes;
                    let next_cache = add(cache_bytes, charge)?;
                    if next_cache <= options.write.conversion.max_cache_bytes
                        && writer.check_external_memory(add(working, charge)?).is_ok()
                    {
                        cache.insert(key, (certified.clone(), charge));
                        cache_bytes = next_cache;
                        peak_cache_bytes = peak_cache_bytes.max(cache_bytes);
                    }
                    writer
                        .check_external_memory(add(working, cache_bytes - previous_cache)?)
                        .map_err(|e| locate(e, &location))?;
                    certified
                };
                accumulate_time(&mut synthesis_certification_time, start.elapsed());
                let bits = certified.term().bit_len() as u64;
                check_limit(
                    Resource::MaxBlockBytes,
                    max_bytes,
                    Some(add(24, bits.div_ceil(8))?),
                )
                .map_err(|e| locate(e, &location))?;
                let next_bits = add(pending_bits, bits)?;
                if !pending.is_empty()
                    && (pending.len() as u64 >= max_terms
                        || add(24, next_bits.div_ceil(8))? > max_bytes)
                {
                    let start = Instant::now();
                    writer.write_block(&pending)?;
                    accumulate_time(&mut write_verify_time, start.elapsed());
                    pending = Vec::new();
                    pending_bits = 0;
                    pending_memory = 0;
                }
                pending_bits = add(pending_bits, bits)?;
                pending_memory = add(pending_memory, certificate_bytes(&certified)?)?;
                // Vec growth may retain spare slots; charge a conservative second
                // copy of the owned certificate storage as well as synthesis scratch.
                let retained = add(
                    add(base_memory, cache_bytes)?,
                    add(mul(pending_memory, 2)?, add(synthesis_memory, 65536)?)?,
                )?;
                writer
                    .check_external_memory(retained)
                    .map_err(|e| locate(e, &location))?;
                tree_bits = add(tree_bits, bits)?;
                tree_bytes = add(tree_bytes, bits.div_ceil(64) * 8)?;
                pending.push(certified);
                flat_index += 1;
            }
        }
        let start = Instant::now();
        if !pending.is_empty() {
            writer.write_block(&pending)?;
        }
        writer.end_tensor()?;
        accumulate_time(&mut write_verify_time, start.elapsed());
    }
    // Release importer-owned buffers before the writer reopens the final catalog.
    drop(cache);
    drop(reader);
    writer.check_external_memory(0)?;
    let start = Instant::now();
    let write = writer.finish()?;
    accumulate_time(&mut write_verify_time, start.elapsed());
    Ok(ConversionReport {
        write,
        source_sha256,
        source_bytes,
        weights_synthesized,
        weights_reused,
        synthesis_steps,
        tree_bytes,
        tree_bits,
        peak_resident_bytes: memory.peak(),
        peak_cache_bytes,
        read_decode_time,
        synthesis_certification_time,
        write_verify_time,
    })
}

#[cfg(test)]
mod metadata_contracts {
    use super::*;

    #[test]
    fn measured_phases_accumulate_instead_of_replacing_or_subtracting() {
        let mut total = Duration::from_nanos(50);
        accumulate_time(&mut total, Duration::from_nanos(7));
        accumulate_time(&mut total, Duration::from_nanos(19));
        assert_eq!(total, Duration::from_nanos(76));
    }

    #[test]
    fn adapter_rejects_nested_and_mistyped_arrays_without_coercion() {
        // Exercise the adapter independently of GGUF preflight, which also
        // rejects nested arrays before constructing an imported model.
        assert!(matches!(
            metadata_value(&V::Array(D::Array, vec![])),
            Err(Error::Metadata {
                reason: MetadataReason::NestedArray,
                ..
            })
        ));
        assert!(matches!(
            metadata_value(&V::Array(D::U8, vec![V::I8(1)])),
            Err(Error::Metadata {
                reason: MetadataReason::InvalidType,
                ..
            })
        ));
    }
}
