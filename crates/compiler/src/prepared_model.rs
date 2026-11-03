//! Explicit numeric boundary from persisted BML terms into symbolic causal RPN.
use super::{InferenceCompiler, ModelConfig, Vocabulary};
use bml_domain::number::{ExactInputLimits, ExactRational};
use bml_model_weights::{MetadataArray as A, MetadataValue as V, WeightReader};
use bml_parser::{GgufDataType as D, GgufMetadataValue as G};
use std::collections::HashMap;

/// Evaluation is opt-in and never a fallback from an unsuccessful symbolic load.
#[derive(Clone, Copy, Debug)]
pub enum WeightEvaluation {
    /// Evaluate each tree in F64, cast to F32, then require the exact rational
    /// of that F32 to equal the recognized target of the original tree.
    /// Non-finite, unrecognized, or differently rounded weights are rejected.
    CheckedExactF32,
}

fn checked_f32(term: &bml_domain::number::Bml, location: &str) -> Result<f32, String> {
    let certificate = term
        .certify_exact(Default::default())
        .map_err(|e| format!("{location}: certificate recognition: {e}"))?;
    let number = term
        .to_f64()
        .map_err(|e| format!("{location}: BML evaluation: {e}"))? as f32;
    if !number.is_finite() {
        return Err(format!("{location}: non-finite F32 evaluation"));
    }
    let target = ExactRational::from_f64(f64::from(number), ExactInputLimits::default())
        .map_err(|e| format!("{location}: {e}"))?;
    if &target != certificate.target() {
        return Err(format!(
            "{location}: F32 evaluation differs from exact BML target"
        ));
    }
    Ok(number)
}

fn value(v: &V) -> G {
    match v {
        V::U8(x) => G::U8(*x),
        V::I8(x) => G::I8(*x),
        V::U16(x) => G::U16(*x),
        V::I16(x) => G::I16(*x),
        V::U32(x) => G::U32(*x),
        V::I32(x) => G::I32(*x),
        V::U64(x) => G::U64(*x),
        V::I64(x) => G::I64(*x),
        V::F32Bits(x) => G::F32(f32::from_bits(*x)),
        V::F64Bits(x) => G::F64(f64::from_bits(*x)),
        V::Bool(x) => G::Bool(*x),
        V::String(x) => G::String(x.clone()),
        V::Array(a) => {
            macro_rules! array {
                ($kind:ident,$xs:expr,$map:expr) => {
                    G::Array(D::$kind, $xs.iter().map($map).collect())
                };
            }
            match a {
                A::U8(x) => array!(U8, x, |v| G::U8(*v)),
                A::I8(x) => array!(I8, x, |v| G::I8(*v)),
                A::U16(x) => array!(U16, x, |v| G::U16(*v)),
                A::I16(x) => array!(I16, x, |v| G::I16(*v)),
                A::U32(x) => array!(U32, x, |v| G::U32(*v)),
                A::I32(x) => array!(I32, x, |v| G::I32(*v)),
                A::U64(x) => array!(U64, x, |v| G::U64(*v)),
                A::I64(x) => array!(I64, x, |v| G::I64(*v)),
                A::F32Bits(x) => array!(F32, x, |v| G::F32(f32::from_bits(*v))),
                A::F64Bits(x) => array!(F64, x, |v| G::F64(f64::from_bits(*v))),
                A::Bool(x) => array!(Bool, x, |v| G::Bool(*v)),
                A::String(x) => array!(String, x, |v: &String| G::String(v.clone())),
            }
        }
    }
}

pub(super) fn load(
    reader: &mut WeightReader,
    cap: usize,
    policy: WeightEvaluation,
) -> Result<InferenceCompiler, String> {
    let metadata: HashMap<_, _> = reader
        .metadata()
        .gguf
        .iter()
        .map(|entry| (entry.key.clone(), value(&entry.value)))
        .collect();
    let (config, tokenizer) =
        bml_inference::model::Model::profile_metadata(&metadata).map_err(|e| e.to_string())?;
    let total = reader
        .tensors()
        .iter()
        .try_fold(0usize, |sum, t| {
            sum.checked_add(usize::try_from(t.element_count).ok()?)
        })
        .ok_or("prepared weight count overflow")?;
    if total > cap || total > u32::MAX as usize {
        return Err("prepared weight materialization limit".into());
    }
    let infos = reader.tensors().to_vec();
    let mut payloads = Vec::new();
    for info in &infos {
        let count = usize::try_from(info.element_count).map_err(|_| "prepared tensor overflow")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(count.checked_mul(4).ok_or("prepared byte overflow")?)
            .map_err(|_| "prepared allocation")?;
        for id in info.first_block.0
            ..info
                .first_block
                .0
                .checked_add(info.block_count)
                .ok_or("prepared block overflow")?
        {
            let block = reader
                .read_block(bml_model_weights::BlockId(id))
                .map_err(|e| e.to_string())?;
            for (offset, term) in block.terms.iter().enumerate() {
                let location = format!("{}[{}]", info.name, block.first_element + offset as u64);
                let evaluated = match policy {
                    WeightEvaluation::CheckedExactF32 => checked_f32(term, &location)?,
                };
                bytes.extend(evaluated.to_le_bytes());
            }
        }
        payloads.push(bytes);
    }
    let tensors = infos
        .iter()
        .zip(&payloads)
        .map(|(info, bytes)| bml_inference::Tensor {
            name: info.name.clone(),
            dims: info.ggml_dimensions.clone(),
            dtype: D::F32,
            data: bytes,
        })
        .collect();
    // Reuse the full inference model validator for required roles, shapes,
    // tokenizer identity, unsupported variants and finite tensor values.
    bml_inference::model::Model::from_tensors(config.clone(), tokenizer.clone(), tensors)
        .map_err(|e| e.to_string())?;
    let mut pool = Vec::new();
    pool.try_reserve_exact(total)
        .map_err(|_| "prepared pool allocation")?;
    let mut offsets = HashMap::new();
    let mut dims = HashMap::new();
    for (info, bytes) in infos.iter().zip(payloads) {
        offsets.insert(
            info.name.clone(),
            u32::try_from(pool.len()).map_err(|_| "prepared offset overflow")?,
        );
        if !(1..=2).contains(&info.ggml_dimensions.len()) {
            return Err("unsupported prepared tensor rank".into());
        }
        dims.insert(
            info.name.clone(),
            (
                usize::try_from(info.ggml_dimensions[0]).map_err(|_| "dimension overflow")?,
                usize::try_from(*info.ggml_dimensions.get(1).unwrap_or(&1))
                    .map_err(|_| "dimension overflow")?,
            ),
        );
        pool.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x)),
        );
    }
    let u32dim =
        |n| u32::try_from(n).map_err(|_| "prepared model dimension exceeds u32".to_string());
    Ok(InferenceCompiler {
        config: ModelConfig {
            architecture: "llama".into(),
            n_layers: u32dim(config.block_count)?,
            n_heads: u32dim(config.head_count)?,
            n_embd: u32dim(config.embedding_length)?,
            context_length: u32dim(config.context_length)?,
            vocab_size: u32dim(config.vocab_size)?,
        },
        vocab: Vocabulary::from_spec(tokenizer)?,
        weight_pool: pool,
        mapped_weights: None,
        weight_offsets: offsets,
        tensor_dims: dims,
        head_dim: u32dim(config.head_dimension())?,
        n_kv_heads: u32dim(config.head_count_kv)?,
        rms_norm_eps: f64::from(config.rms_norm_eps),
        rope_freq_base: f64::from(config.rope_freq_base),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bml_domain::number::Bml;
    #[test]
    fn explicit_f32_boundary_rejects_loss_instead_of_silently_rounding() {
        let exact = Bml::from_decimal_str_exact("0.5").unwrap();
        assert_eq!(checked_f32(exact.term(), "w[0]").unwrap(), 0.5);
        let decimal = Bml::from_decimal_str_exact("0.1").unwrap();
        let error = checked_f32(decimal.term(), "w[7]").unwrap_err();
        assert!(error.contains("w[7]"));
        assert!(error.contains("differs from exact BML target"));
    }
}
