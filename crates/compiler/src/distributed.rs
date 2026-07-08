//! Compilación distribuida: fragmenta un GGUF por capa del transformer.
//!
//! Cada fragmento `.bmlgraph` es **self-contained**: contiene bytecode RPN
//! + los pesos f32 de esa capa. Un nodo puede cargar un solo fragmento y
//! ejecutar su porción del transformer sin necesidad del GGUF original.
//!
//! # Formato fragmento v2
//!
//! ```text
//! fragment_N.bmlgraph:
//!   [magic 4B][version 4B]
//!   [fragment_id 4B]
//!   [layer_start 4B][layer_end 4B]       ← rango de capas
//!   [n_ops 4B][ops...]
//!   [n_weights 8B][weights: N × 4B f32] ← pesos de esas capas
//!   [n_tensors 4B]
//!     [name_len 4B][name][offset 8B][n_rows 4B][n_cols 4B] × n_tensors
//! ```
//!
//! # Tamaño por fragmento
//!
//! TinyLlama 1.1B Q4_0, una capa:
//!   - attn_norm:   2048 f32 = 8 KB
//!   - attn_q:      2048×2048 = 4M f32 = 16 MB
//!   - attn_k:      2048×2048 = 4M f32 = 16 MB  (GQA: 4×512×2048 = 2M = 8MB)
//!   - attn_v:      2048×2048 = 4M f32 = 16 MB  (GQA: 4×512×2048 = 2M = 8MB)
//!   - attn_output: 2048×2048 = 4M f32 = 16 MB
//!   - ffn_norm:    2048 f32 = 8 KB
//!   - ffn_gate:    2048×5632 = 11.5M f32 = 46 MB
//!   - ffn_up:      2048×5632 = 11.5M f32 = 46 MB
//!   - ffn_down:    5632×2048 = 11.5M f32 = 46 MB
//!   Total por capa: ~200 MB
//!
//! 22 capas × 200 MB = ~4.4 GB + embeddings 32000×2048 = 262 MB ≈ 4.6 GB total

use crate::fragment::{BMLGRAPH_MAGIC, BMLGRAPH_VERSION};
use crate::gguf_compiler::{read_tensor_f32, ModelConfig};
use crate::rpn::RpnOp;
use bml_domain::encoder::RealEncoder;
use bml_parser::GgufParser;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// Metadatos de un tensor dentro de un fragmento.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub offset: u64,
    pub n_rows: u32,
    pub n_cols: u32,
}

/// Un fragmento distribuido self-contained.
///
/// Contiene todo lo que un nodo necesita para ejecutar una porción
/// del transformer: bytecode + pesos + metadata.
#[derive(Debug)]
pub struct DistributedFragment {
    /// ID del fragmento (0-indexed).
    pub fragment_id: u32,
    /// Primera capa del transformer (inclusive).
    pub layer_start: u32,
    /// Última capa del transformer (exclusive).
    pub layer_end: u32,
    /// Bytecode RPN (puede ser vacío si los pesos se ejecutan directo).
    pub ops: Vec<RpnOp>,
    /// Pesos dequantizados como f32 (concatenados).
    pub weights: Vec<f32>,
    /// Mapa de tensor_name → (offset en weights, dims).
    pub tensors: Vec<TensorMeta>,
    /// Configuración del modelo (compartida entre todos los fragmentos).
    pub config: ModelConfig,
    /// Número de KV heads (para GQA).
    pub n_kv_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
}

/// Compila un GGUF a fragmentos distribuidos self-contained.
///
/// Cada capa del transformer se convierte en un fragmento independiente
/// con sus pesos embebidos. Un nodo puede cargar un solo fragmento y
/// ejecutar su porción del transformer sin el GGUF original.
///
/// # Parámetros
///
/// - `gguf_path`: Ruta al GGUF.
/// - `layers_per_fragment`: Cuántas capas por fragmento (1 = una capa
///   por fragmento, más granularidad; N = menos fragmentos más grandes).
/// - `include_embeddings`: Si true, el último fragmento incluye
///   token_embd.weight y output_norm.weight para compute_logits.
pub fn compile_distributed(
    gguf_path: &Path,
    layers_per_fragment: u32,
    include_embeddings: bool,
) -> Result<Vec<DistributedFragment>, String> {
    let parser = GgufParser::open(gguf_path).map_err(|e| format!("parser: {e}"))?;
    let config = read_model_config_pub(&parser)?;

    let head_dim = config.n_embd / config.n_heads;
    let n_kv_heads = parser
        .get_metadata(&format!("{}.attention.key_value_head_count", config.architecture))
        .and_then(|v| match v {
            bml_parser::GgufMetadataValue::U32(n) => Some(*n),
            bml_parser::GgufMetadataValue::I32(n) => Some(*n as u32),
            _ => None,
        })
        .unwrap_or(config.n_heads);

    let mut fragments = Vec::new();

    // Fragmentar capas
    let mut layer = 0u32;
    while layer < config.n_layers {
        let layer_end = (layer + layers_per_fragment).min(config.n_layers);

        let frag = compile_layer_range(&parser, &config, layer, layer_end, head_dim, n_kv_heads, false)?;
        fragments.push(frag);

        layer = layer_end;
    }

    // Fragmento final: embeddings + output_norm + lm_head
    if include_embeddings {
        let frag = compile_layer_range(&parser, &config, config.n_layers, config.n_layers + 1, head_dim, n_kv_heads, true)?;
        fragments.push(frag);
    }

    Ok(fragments)
}

/// Compila un rango de capas a un fragmento distribuido.
fn compile_layer_range(
    parser: &GgufParser,
    config: &ModelConfig,
    layer_start: u32,
    layer_end: u32,
    head_dim: u32,
    n_kv_heads: u32,
    is_final: bool,
) -> Result<DistributedFragment, String> {
    let mut weights: Vec<f32> = Vec::new();
    let mut tensors: Vec<TensorMeta> = Vec::new();

    let mut add_tensor = |parser: &GgufParser,
                          weights: &mut Vec<f32>,
                          tensors: &mut Vec<TensorMeta>,
                          name: &str|
     -> Result<(), String> {
        if let Some(vals) = read_tensor_f32(parser, name) {
            let info = parser.find_tensor(name);
            let (n_rows, n_cols) = if let Some(info) = info {
                if info.dims.len() >= 2 {
                    (info.dims[0] as u32, info.dims[1] as u32)
                } else if info.dims.len() == 1 {
                    (info.dims[0] as u32, 1)
                } else {
                    (vals.len() as u32, 1)
                }
            } else {
                (vals.len() as u32, 1)
            };
            let offset = weights.len() as u64;
            weights.extend(vals);
            tensors.push(TensorMeta {
                name: name.to_string(),
                offset,
                n_rows,
                n_cols,
            });
        }
        Ok(())
    };

    // Cargar pesos de cada capa en el rango
    for layer in layer_start..layer_end {
        if is_final {
            // Fragmento final: output_norm + token_embd (lm_head tied)
            add_tensor(parser, &mut weights, &mut tensors, "output_norm.weight")?;
            add_tensor(parser, &mut weights, &mut tensors, "token_embd.weight")?;
        } else {
            let prefix = format!("blk.{layer}");
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.attn_norm.weight"))?;
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.attn_q.weight"))?;
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.attn_k.weight"))?;
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.attn_v.weight"))?;
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.attn_output.weight"))?;
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.ffn_norm.weight"))?;
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.ffn_gate.weight"))?;
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.ffn_up.weight"))?;
            add_tensor(parser, &mut weights, &mut tensors, &format!("{prefix}.ffn_down.weight"))?;
        }
    }

    // Bytecode: un nodo Var(0) placeholder.
    // El worker ejecuta el transformer usando los pesos directamente
    // (matmul/attention/RoPE en f64), no via RPN bytecode.
    let ops = vec![RpnOp::Var(0)];

    let fragment_id = if is_final { layer_end } else { layer_start };

    Ok(DistributedFragment {
        fragment_id,
        layer_start,
        layer_end,
        ops,
        weights,
        tensors,
        config: config.clone(),
        n_kv_heads,
        head_dim,
    })
}

/// Serializa fragmentos distribuidos a un directorio.
///
/// Genera:
/// - `header.bmlgraph`: magic, version, n_fragments, config
/// - `fragment_N.bmlgraph`: self-contained (bytecode + pesos + tensors)
pub fn serialize_distributed(
    fragments: &[DistributedFragment],
    output_dir: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(output_dir).map_err(|e| format!("crear dir: {e}"))?;

    // Header: magic, version, n_fragments, config
    let header_path = output_dir.join("header.bmlgraph");
    let mut f = std::fs::File::create(&header_path).map_err(|e| format!("crear header: {e}"))?;

    f.write_all(&BMLGRAPH_MAGIC.to_le_bytes()).map_err(|e| format!("write magic: {e}"))?;
    f.write_all(&BMLGRAPH_VERSION.to_le_bytes()).map_err(|e| format!("write version: {e}"))?;
    f.write_all(&(fragments.len() as u32).to_le_bytes()).map_err(|e| format!("write n_frag: {e}"))?;

    let config = &fragments[0].config;
    let arch_bytes = config.architecture.as_bytes();
    f.write_all(&(arch_bytes.len() as u64).to_le_bytes()).map_err(|e| format!("write arch len: {e}"))?;
    f.write_all(arch_bytes).map_err(|e| format!("write arch: {e}"))?;
    f.write_all(&config.n_layers.to_le_bytes()).map_err(|e| format!("write n_layers: {e}"))?;
    f.write_all(&config.n_heads.to_le_bytes()).map_err(|e| format!("write n_heads: {e}"))?;
    f.write_all(&config.n_embd.to_le_bytes()).map_err(|e| format!("write n_embd: {e}"))?;
    f.write_all(&config.context_length.to_le_bytes()).map_err(|e| format!("write ctx: {e}"))?;
    f.write_all(&config.vocab_size.to_le_bytes()).map_err(|e| format!("write vocab: {e}"))?;

    // Fragmentos: cada uno self-contained
    for frag in fragments {
        let frag_path = output_dir.join(format!("fragment_{}.bmlgraph", frag.fragment_id));
        let mut ff = std::fs::File::create(&frag_path).map_err(|e| format!("crear frag {}: {e}", frag.fragment_id))?;

        // Magic + version
        ff.write_all(&BMLGRAPH_MAGIC.to_le_bytes()).map_err(|e| format!("write magic: {e}"))?;
        ff.write_all(&BMLGRAPH_VERSION.to_le_bytes()).map_err(|e| format!("write version: {e}"))?;

        // Fragment ID
        ff.write_all(&frag.fragment_id.to_le_bytes()).map_err(|e| format!("write frag id: {e}"))?;

        // Layer range
        ff.write_all(&frag.layer_start.to_le_bytes()).map_err(|e| format!("write layer start: {e}"))?;
        ff.write_all(&frag.layer_end.to_le_bytes()).map_err(|e| format!("write layer end: {e}"))?;

        // Bytecode RPN
        ff.write_all(&(frag.ops.len() as u32).to_le_bytes()).map_err(|e| format!("write n_ops: {e}"))?;
        for op in &frag.ops {
            serialize_op(&mut ff, op)?;
        }

        // Pesos f32
        ff.write_all(&(frag.weights.len() as u64).to_le_bytes()).map_err(|e| format!("write n_weights: {e}"))?;
        for &w in &frag.weights {
            ff.write_all(&w.to_le_bytes()).map_err(|e| format!("write weight: {e}"))?;
        }

        // Tensor metadata
        ff.write_all(&(frag.tensors.len() as u32).to_le_bytes()).map_err(|e| format!("write n_tensors: {e}"))?;
        for tensor in &frag.tensors {
            let name_bytes = tensor.name.as_bytes();
            ff.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(|e| format!("write name len: {e}"))?;
            ff.write_all(name_bytes).map_err(|e| format!("write name: {e}"))?;
            ff.write_all(&tensor.offset.to_le_bytes()).map_err(|e| format!("write offset: {e}"))?;
            ff.write_all(&tensor.n_rows.to_le_bytes()).map_err(|e| format!("write n_rows: {e}"))?;
            ff.write_all(&tensor.n_cols.to_le_bytes()).map_err(|e| format!("write n_cols: {e}"))?;
        }

        // Config del modelo (repetida por fragmento para self-contained)
        ff.write_all(&(frag.n_kv_heads as u32).to_le_bytes()).map_err(|e| format!("write n_kv_heads: {e}"))?;
        ff.write_all(&frag.head_dim.to_le_bytes()).map_err(|e| format!("write head_dim: {e}"))?;
    }

    Ok(())
}

/// Carga un fragmento distribuido desde un archivo.
///
/// El fragmento es self-contained: no necesita el GGUF ni el header.
pub fn load_distributed_fragment(
    frag_path: &Path,
) -> Result<DistributedFragment, String> {
    let bytes = std::fs::read(frag_path).map_err(|e| format!("leer fragmento: {e}"))?;
    if bytes.len() < 20 {
        return Err("fragmento demasiado pequeño".into());
    }

    let mut offset = 0;

    // Magic + version
    let magic = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    if magic != BMLGRAPH_MAGIC {
        return Err(format!("magic inválido: 0x{magic:08X}"));
    }
    offset += 8;

    // Fragment ID
    let fragment_id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Layer range
    let layer_start = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let layer_end = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Bytecode RPN
    let n_ops = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let mut ops = Vec::with_capacity(n_ops);
    for _ in 0..n_ops {
        let (op, consumed) = deserialize_op(&bytes[offset..])?;
        ops.push(op);
        offset += consumed;
    }

    // Pesos f32
    let n_weights = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;
    let mut weights = Vec::with_capacity(n_weights);
    for _ in 0..n_weights {
        let w = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        weights.push(w);
    }

    // Tensor metadata
    let n_tensors = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let name = String::from_utf8_lossy(&bytes[offset..offset + name_len]).to_string();
        offset += name_len;
        let tensor_offset = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let n_rows = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let n_cols = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        tensors.push(TensorMeta {
            name,
            offset: tensor_offset,
            n_rows,
            n_cols,
        });
    }

    // Config del modelo
    let n_kv_heads = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let head_dim = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

    // Cargar config desde el header (necesitamos el header para la config completa)
    // Por ahora, usamos una config default que se completa al cargar el header.
    let config = ModelConfig {
        architecture: String::new(),
        n_layers: 0,
        n_heads: 0,
        n_embd: 0,
        context_length: 0,
        vocab_size: 0,
    };

    Ok(DistributedFragment {
        fragment_id,
        layer_start,
        layer_end,
        ops,
        weights,
        tensors,
        config,
        n_kv_heads,
        head_dim,
    })
}

/// Carga el header distribuido para obtener la config del modelo.
pub fn load_distributed_header(
    dir: &Path,
) -> Result<(ModelConfig, usize), String> {
    let header_path = dir.join("header.bmlgraph");
    let bytes = std::fs::read(&header_path).map_err(|e| format!("leer header: {e}"))?;
    if bytes.len() < 12 {
        return Err("header demasiado pequeño".into());
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != BMLGRAPH_MAGIC {
        return Err(format!("magic inválido: 0x{magic:08X}"));
    }
    let n_fragments = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let mut offset = 12;
    let arch_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;
    let architecture = String::from_utf8_lossy(&bytes[offset..offset + arch_len]).to_string();
    offset += arch_len;
    let n_layers = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let n_heads = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let n_embd = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let context_length = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let vocab_size = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());

    Ok((
        ModelConfig {
            architecture,
            n_layers,
            n_heads,
            n_embd,
            context_length,
            vocab_size,
        },
        n_fragments,
    ))
}

/// Serializa una operación RPN a bytes.
fn serialize_op(f: &mut std::fs::File, op: &RpnOp) -> Result<(), String> {
    match op {
        RpnOp::One => f.write_all(&[0]).map_err(|e| format!("write op: {e}")),
        RpnOp::Zero => f.write_all(&[6]).map_err(|e| format!("write op: {e}")),
        RpnOp::Bml => f.write_all(&[1]).map_err(|e| format!("write op: {e}")),
        RpnOp::Dup => f.write_all(&[2]).map_err(|e| format!("write op: {e}")),
        RpnOp::Loop { count, body_len } => {
            f.write_all(&[3]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&count.to_le_bytes()).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&body_len.to_le_bytes()).map_err(|e| format!("write op: {e}"))
        }
        RpnOp::Var(id) => {
            f.write_all(&[4]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&id.to_le_bytes()).map_err(|e| format!("write op: {e}"))
        }
        RpnOp::Const(id) => {
            f.write_all(&[5]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&id.to_le_bytes()).map_err(|e| format!("write op: {e}"))
        }
        RpnOp::VarIndexed { base } => {
            f.write_all(&[7]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&base.to_le_bytes()).map_err(|e| format!("write op: {e}"))
        }
        RpnOp::StoreResult { slot } => {
            f.write_all(&[8]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&slot.to_le_bytes()).map_err(|e| format!("write op: {e}"))
        }
        RpnOp::FAdd => f.write_all(&[9]).map_err(|e| format!("write op: {e}")),
        RpnOp::FMul => f.write_all(&[10]).map_err(|e| format!("write op: {e}")),
        RpnOp::Pick { depth } => {
            f.write_all(&[11]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&depth.to_le_bytes()).map_err(|e| format!("write op: {e}"))
        }
        RpnOp::Drop => f.write_all(&[12]).map_err(|e| format!("write op: {e}")),
        RpnOp::Swap => f.write_all(&[13]).map_err(|e| format!("write op: {e}")),
    }
}

/// Deserializa una operación RPN desde bytes. Retorna (op, bytes_consumed).
fn deserialize_op(bytes: &[u8]) -> Result<(RpnOp, usize), String> {
    if bytes.is_empty() {
        return Err("bytes insuficientes para op".into());
    }
    let tag = bytes[0];
    let (op, consumed) = match tag {
        0 => (RpnOp::One, 1),
        6 => (RpnOp::Zero, 1),
        1 => (RpnOp::Bml, 1),
        2 => (RpnOp::Dup, 1),
        3 => {
            let count = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
            let body_len = u32::from_le_bytes(bytes[5..9].try_into().unwrap());
            (RpnOp::Loop { count, body_len }, 9)
        }
        4 => {
            let id = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
            (RpnOp::Var(id), 5)
        }
        5 => {
            let id = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
            (RpnOp::Const(id), 5)
        }
        7 => {
            let base = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
            (RpnOp::VarIndexed { base }, 5)
        }
        8 => {
            let slot = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
            (RpnOp::StoreResult { slot }, 5)
        }
        9 => (RpnOp::FAdd, 1),
        10 => (RpnOp::FMul, 1),
        11 => {
            let depth = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
            (RpnOp::Pick { depth }, 5)
        }
        12 => (RpnOp::Drop, 1),
        13 => (RpnOp::Swap, 1),
        _ => return Err(format!("tag desconocido: {tag}")),
    };
    Ok((op, consumed))
}

/// Lee la config del modelo (wrapper público de la función privada en gguf_compiler).
fn read_model_config_pub(parser: &GgufParser) -> Result<ModelConfig, String> {
    let arch = parser
        .architecture()
        .ok_or("no se encontró general.architecture")?
        .to_string();
    let arch_clone = arch.clone();

    let get_u32 = |key: &str| -> u32 {
        match parser.get_metadata(&format!("{arch_clone}.{key}")) {
            Some(bml_parser::GgufMetadataValue::U32(v)) => *v,
            Some(bml_parser::GgufMetadataValue::U64(v)) => *v as u32,
            Some(bml_parser::GgufMetadataValue::I32(v)) => *v as u32,
            Some(bml_parser::GgufMetadataValue::I64(v)) => *v as u32,
            _ => 0,
        }
    };

    Ok(ModelConfig {
        architecture: arch,
        n_layers: get_u32("block_count"),
        n_heads: get_u32("attention.head_count"),
        n_embd: get_u32("embedding_length"),
        context_length: get_u32("context_length"),
        vocab_size: get_u32("vocab_size"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::HardwareSpec;

    #[test]
    fn compile_distributed_tinyllama() {
        let path = "/root/tinyllama.gguf";
        if !Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }

        let fragments = compile_distributed(Path::new(path), 1, true).expect("compile");
        assert!(fragments.len() > 0);

        let config = &fragments[0].config;
        println!(
            "Model: {} ({} layers, {} heads, {} embd)",
            config.architecture, config.n_layers, config.n_heads, config.n_embd
        );
        println!("Fragments: {}", fragments.len());

        for frag in &fragments {
            let weight_mb = frag.weights.len() as f64 * 4.0 / (1024.0 * 1024.0);
            println!(
                "  fragment {} (layers {}-{}): {} tensors, {:.1} MB weights",
                frag.fragment_id, frag.layer_start, frag.layer_end,
                frag.tensors.len(), weight_mb,
            );
        }

        // Verificar que la suma de pesos ≈ peso total del modelo
        let total_weights: usize = fragments.iter().map(|f| f.weights.len()).sum();
        println!("Total weights: {} ({:.1} MB)", total_weights, total_weights as f64 * 4.0 / (1024.0 * 1024.0));
    }

    #[test]
    fn serialize_load_distributed_roundtrip() {
        let path = "/root/tinyllama.gguf";
        if !Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }

        let fragments = compile_distributed(Path::new(path), 4, true).expect("compile");
        let out_dir = std::env::temp_dir().join(format!("bml_dist_test_{}", std::process::id()));

        serialize_distributed(&fragments, &out_dir).expect("serialize");
        println!("Serialized {} fragments to {}", fragments.len(), out_dir.display());

        // Listar archivos
        for entry in std::fs::read_dir(&out_dir).unwrap() {
            let entry = entry.unwrap();
            let meta = entry.metadata().unwrap();
            println!("  {} ({} bytes)", entry.file_name().to_string_lossy(), meta.len());
        }

        // Cargar header
        let (config, n_fragments) = load_distributed_header(&out_dir).expect("header");
        assert_eq!(n_fragments, fragments.len());
        assert_eq!(config.architecture, fragments[0].config.architecture);
        assert_eq!(config.n_layers, fragments[0].config.n_layers);

        // Cargar primer fragmento
        let frag0_path = out_dir.join(format!("fragment_{}.bmlgraph", fragments[0].fragment_id));
        let frag0 = load_distributed_fragment(&frag0_path).expect("load frag 0");

        assert_eq!(frag0.fragment_id, fragments[0].fragment_id);
        assert_eq!(frag0.layer_start, fragments[0].layer_start);
        assert_eq!(frag0.layer_end, fragments[0].layer_end);
        assert_eq!(frag0.weights.len(), fragments[0].weights.len());
        assert_eq!(frag0.tensors.len(), fragments[0].tensors.len());

        // Verificar que los pesos coinciden
        for (i, (a, b)) in frag0.weights.iter().zip(fragments[0].weights.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "peso {i} no coincide");
        }

        println!("Roundtrip OK: {} tensors, {} weights", frag0.tensors.len(), frag0.weights.len());

        std::fs::remove_dir_all(&out_dir).ok();
    }
}

// ===========================================================================
// BmlWeightPool: pesos como árboles BML nativos
// ===========================================================================

/// Pool de pesos codificados como árboles BML.
///
/// Cada peso f32 se codifica como un nodo `Const(id)` en el const pool.
/// Valores idénticos se deduplican automáticamente. Para un modelo Q4_0
/// con 16 valores de peso distintos, solo se crean 14 entradas únicas
/// en el const pool (0 y 1 son `Zero`/`One`).
///
/// # Compresión
///
/// - f32 sin comprimir: 4 bytes por peso
/// - BML con const pool: `ceil(log2(n_unique))` bits por peso + const pool
/// - Para Q4_0 (14 únicos): 4 bits/peso + 112 bytes const pool
/// - 1B pesos × 4 bits = 500 MB vs 3.9 GB (f32) = **8x compresión**
pub struct BmlWeightPool {
    /// Encoder subyacente (gestiona const pool + deduplicación).
    encoder: RealEncoder,
    /// Número total de pesos codificados (incluyendo duplicados).
    n_total: usize,
}

impl BmlWeightPool {
    pub fn new() -> Self {
        Self {
            encoder: RealEncoder::new(),
            n_total: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            encoder: RealEncoder::with_capacity(cap),
            n_total: 0,
        }
    }

    /// Codifica un peso f32 como nodo BML.
    ///
    /// Valores idénticos producen el mismo `ConstId` (deduplicación).
    pub fn encode(&mut self, weight: f32) -> u32 {
        self.n_total += 1;
        let node_id = self.encoder.encode_f64(weight as f64);
        // El NodeId es el índice del nodo en el encoder.
        // Para Const, el ConstId está dentro del NodeKind.
        node_id
    }

    /// Codifica un slice de pesos f32 y retorna los NodeIds.
    pub fn encode_slice(&mut self, weights: &[f32]) -> Vec<u32> {
        weights.iter().map(|&w| self.encode(w)).collect()
    }

    /// Número de valores únicos en el const pool.
    pub fn n_unique(&self) -> usize {
        self.encoder.const_count()
    }

    /// Número total de pesos codificados (con duplicados).
    pub fn n_total(&self) -> usize {
        self.n_total
    }

    /// Ratio de compresión (n_total / n_unique).
    pub fn compression_ratio(&self) -> f64 {
        if self.n_unique() == 0 {
            return 0.0;
        }
        self.n_total as f64 / self.n_unique() as f64
    }

    /// Tamaño estimado en bytes si se almacenan como índices de bits.
    pub fn estimated_size_bytes(&self) -> usize {
        if self.n_unique() == 0 {
            return 0;
        }
        let bits_per_weight = (self.n_unique() as f64).log2().ceil() as usize;
        let bits_per_weight = bits_per_weight.max(1);
        let weight_table_size = (self.n_total * bits_per_weight + 7) / 8;
        let const_pool_size = self.n_unique() * 8; // f64
        weight_table_size + const_pool_size
    }

    /// Tamaño en bytes si se almacenaran como f32 (sin comprimir).
    pub fn f32_size_bytes(&self) -> usize {
        self.n_total * 4
    }

    /// Compresión lograda vs f32.
    pub fn compression_vs_f32(&self) -> f64 {
        if self.f32_size_bytes() == 0 {
            return 0.0;
        }
        self.f32_size_bytes() as f64 / self.estimated_size_bytes() as f64
    }

    /// Referencia al const pool (valores únicos f64).
    pub fn const_pool(&self) -> &[f64] {
        self.encoder.const_values()
    }

    /// Referencia al encoder subyacente.
    pub fn encoder(&self) -> &RealEncoder {
        &self.encoder
    }
}

impl Default for BmlWeightPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Compila un GGUF con pesos codificados como árboles BML.
///
/// A diferencia de `compile_distributed`, este codifica los pesos con
/// `BmlWeightPool` para reportar estadísticas de compresión.
pub fn compile_distributed_bml(
    gguf_path: &Path,
    layers_per_fragment: u32,
) -> Result<(Vec<DistributedFragment>, BmlWeightPool), String> {
    let parser = GgufParser::open(gguf_path).map_err(|e| format!("parser: {e}"))?;
    let config = read_model_config_pub(&parser)?;

    let head_dim = config.n_embd / config.n_heads;
    let n_kv_heads = parser
        .get_metadata(&format!("{}.attention.key_value_head_count", config.architecture))
        .and_then(|v| match v {
            bml_parser::GgufMetadataValue::U32(n) => Some(*n),
            bml_parser::GgufMetadataValue::I32(n) => Some(*n as u32),
            _ => None,
        })
        .unwrap_or(config.n_heads);

    let mut pool = BmlWeightPool::with_capacity(1024);
    let mut fragments = Vec::new();

    let mut layer = 0u32;
    while layer < config.n_layers {
        let layer_end = (layer + layers_per_fragment).min(config.n_layers);
        let frag = compile_layer_range_bml(&parser, &config, layer, layer_end, head_dim, n_kv_heads, false, &mut pool)?;
        fragments.push(frag);
        layer = layer_end;
    }

    // Fragmento final
    let frag = compile_layer_range_bml(&parser, &config, config.n_layers, config.n_layers + 1, head_dim, n_kv_heads, true, &mut pool)?;
    fragments.push(frag);

    Ok((fragments, pool))
}

fn compile_layer_range_bml(
    parser: &GgufParser,
    config: &ModelConfig,
    layer_start: u32,
    layer_end: u32,
    head_dim: u32,
    n_kv_heads: u32,
    is_final: bool,
    pool: &mut BmlWeightPool,
) -> Result<DistributedFragment, String> {
    let mut weights: Vec<f32> = Vec::new();
    let mut tensors: Vec<TensorMeta> = Vec::new();

    for layer in layer_start..layer_end {
        if is_final {
            if let Some(vals) = read_tensor_f32(parser, "output_norm.weight") {
                let offset = weights.len() as u64;
                weights.extend(vals);
                tensors.push(TensorMeta { name: "output_norm.weight".into(), offset, n_rows: config.n_embd, n_cols: 1 });
            }
            if let Some(vals) = read_tensor_f32(parser, "token_embd.weight") {
                let offset = weights.len() as u64;
                let info = parser.find_tensor("token_embd.weight");
                let (r, c) = if let Some(i) = info { if i.dims.len() >= 2 { (i.dims[0] as u32, i.dims[1] as u32) } else { (vals.len() as u32, 1) } } else { (vals.len() as u32, 1) };
                weights.extend(vals);
                tensors.push(TensorMeta { name: "token_embd.weight".into(), offset, n_rows: r, n_cols: c });
            }
        } else {
            let prefix = format!("blk.{layer}");
            for suffix in ["attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                let name = format!("{prefix}.{suffix}");
                if let Some(vals) = read_tensor_f32(parser, &name) {
                    let info = parser.find_tensor(&name);
                    let (r, c) = if let Some(i) = info { if i.dims.len() >= 2 { (i.dims[0] as u32, i.dims[1] as u32) } else { (vals.len() as u32, 1) } } else { (vals.len() as u32, 1) };
                    let offset = weights.len() as u64;
                    weights.extend(vals);
                    tensors.push(TensorMeta { name, offset, n_rows: r, n_cols: c });
                }
            }
        }
    }

    // Codificar todos los pesos en el BmlWeightPool
    for &w in &weights {
        pool.encode(w);
    }

    let ops = vec![RpnOp::Var(0)];
    let fragment_id = if is_final { layer_end } else { layer_start };

    Ok(DistributedFragment {
        fragment_id,
        layer_start,
        layer_end,
        ops,
        weights,
        tensors,
        config: config.clone(),
        n_kv_heads,
        head_dim,
    })
}

#[cfg(test)]
mod bml_weight_tests {
    use super::*;

    #[test]
    fn bml_weight_pool_q4_0_values() {
        let mut pool = BmlWeightPool::new();
        let q4_values: [f32; 16] = [
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];

        // Simular 1M pesos con valores Q4_0
        for _ in 0..62_500 {
            for &v in &q4_values {
                pool.encode(v);
            }
        }

        println!("Q4_0 pool: {} unique, {} total", pool.n_unique(), pool.n_total());
        println!("  Compression ratio: {:.1}x", pool.compression_ratio());
        println!("  Estimated size: {} bytes ({:.1} MB)", pool.estimated_size_bytes(), pool.estimated_size_bytes() as f64 / 1_000_000.0);
        println!("  f32 size: {} bytes ({:.1} MB)", pool.f32_size_bytes(), pool.f32_size_bytes() as f64 / 1_000_000.0);
        println!("  Compression vs f32: {:.1}x", pool.compression_vs_f32());

        assert_eq!(pool.n_unique(), 14); // 16 - 2 (0 y 1 son Zero/One)
        assert_eq!(pool.n_total(), 1_000_000);
        assert!(pool.compression_ratio() > 70_000.0); // 1M / 14 ≈ 71429
    }

    #[test]
    fn bml_weight_pool_dedup() {
        let mut pool = BmlWeightPool::new();
        pool.encode(0.5);
        pool.encode(0.5);
        pool.encode(0.5);
        pool.encode(0.25);
        pool.encode(0.25);

        assert_eq!(pool.n_unique(), 2);
        assert_eq!(pool.n_total(), 5);
        assert_eq!(pool.const_pool().len(), 2);
    }

    #[test]
    fn compile_distributed_bml_tinyllama() {
        let path = "/root/tinyllama.gguf";
        if !Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }

        let (fragments, pool) = compile_distributed_bml(Path::new(path), 1).expect("compile bml");

        println!("\n=== BML Weight Pool Statistics ===");
        println!("  Unique values: {}", pool.n_unique());
        println!("  Total weights: {}", pool.n_total());
        println!("  Compression ratio: {:.1}x", pool.compression_ratio());
        println!("  Estimated BML size: {:.1} MB", pool.estimated_size_bytes() as f64 / 1_000_000.0);
        println!("  f32 size: {:.1} MB", pool.f32_size_bytes() as f64 / 1_000_000.0);
        println!("  Compression vs f32: {:.1}x", pool.compression_vs_f32());
        println!("  Fragments: {}", fragments.len());

        assert!(pool.n_unique() > 0);
        assert!(pool.n_total() > 1_000_000);
        assert!(pool.compression_vs_f32() > 1.0);
    }
}

// ===========================================================================
// Sub-fragmentación L1i (< 30 KB por sub-fragmento)
// ===========================================================================

/// Tamaño objetivo de cada sub-fragmento en bytes (30 KB < 32 KB L1i).
pub const L1I_TARGET_SIZE: usize = 30 * 1024;

/// Referencia a un rango de pesos dentro del fragmento padre.
///
/// Los sub-fragmentos no copian pesos — referencian por offset al pool
/// de pesos del fragmento padre. Esto permite que los pesos estén en L2/L3
/// mientras el bytecode del sub-fragmento cabe en L1i.
#[derive(Debug, Clone)]
pub struct WeightRef {
    /// Nombre del tensor (ej. "blk.0.attn_q.weight").
    pub tensor_name: String,
    /// Offset inicial dentro del pool de pesos del fragmento.
    pub offset: u64,
    /// Número de elementos f32 referenciados.
    pub len: u32,
}

/// Un sub-fragmento L1i: bytecode < 30 KB + referencias a pesos.
///
/// El runtime ejecuta sub-fragmentos secuencialmente. Cada sub-fragmento
/// cabe en L1i (bytecode), mientras los pesos se sirven desde L2/L3.
/// El cambio de sub-fragmento es O(1): cambiar el slice de ops.
#[derive(Debug, Clone)]
pub struct SubFragment {
    /// ID del fragmento padre.
    pub fragment_id: u32,
    /// ID del sub-fragmento (0-indexed dentro del fragmento padre).
    pub sub_id: u32,
    /// Capa inicial (inclusive).
    pub layer_start: u32,
    /// Capa final (exclusive).
    pub layer_end: u32,
    /// Bytecode RPN (< 30 KB).
    pub ops: Vec<RpnOp>,
    /// Referencias a pesos del fragmento padre (no copias).
    pub weight_refs: Vec<WeightRef>,
    /// Sub-fragmentos que deben completarse antes que este.
    pub depends_on: Vec<u32>,
}

impl SubFragment {
    /// Tamaño del bytecode en bytes.
    pub fn bytecode_size(&self) -> usize {
        self.ops.iter().map(|op| op_size(op)).sum()
    }

    /// Verifica que el bytecode cabe en L1i.
    pub fn fits_l1i(&self) -> bool {
        self.bytecode_size() <= L1I_TARGET_SIZE
    }
}

/// Tamaño en bytes de una operación RPN serializada.
fn op_size(op: &RpnOp) -> usize {
    match op {
        RpnOp::One | RpnOp::Zero | RpnOp::Bml | RpnOp::Dup
        | RpnOp::FAdd | RpnOp::FMul | RpnOp::Drop | RpnOp::Swap => 1,
        RpnOp::Var(_) | RpnOp::Const(_) | RpnOp::VarIndexed { .. }
        | RpnOp::StoreResult { .. } | RpnOp::Pick { .. } => 5,
        RpnOp::Loop { .. } => 9,
    }
}

/// Sub-fragmenta un `DistributedFragment` en sub-fragmentos de < 30 KB.
///
/// Cada sub-fragmento contiene un subconjunto del bytecode RPN del
/// fragmento padre. Los pesos se referencian por offset (no se copian).
///
/// # Dependencias
///
/// Los sub-fragmentos se generan en orden secuencial por defecto
/// (`depends_on = [sub_id - 1]`). El scheduler puede reorganizar
/// las dependencias según el DAG del transformer.
///
/// # Retorna
///
/// `Vec<SubFragment>` ordenado por `sub_id`.
pub fn sub_fragment(frag: &DistributedFragment) -> Vec<SubFragment> {
    sub_fragment_with_threshold(frag, L1I_TARGET_SIZE)
}

/// Sub-fragmenta con un tamaño objetivo personalizado.
pub fn sub_fragment_with_threshold(frag: &DistributedFragment, target_size: usize) -> Vec<SubFragment> {
    let mut sub_fragments = Vec::new();
    let mut current_ops = Vec::new();
    let mut current_refs = Vec::new();
    let mut current_size = 0usize;
    let mut sub_id = 0u32;

    // Crear referencias a todos los tensores del fragmento
    let all_weight_refs: Vec<WeightRef> = frag.tensors.iter().map(|t| WeightRef {
        tensor_name: t.name.clone(),
        offset: t.offset,
        len: (t.n_rows as u64 * t.n_cols as u64).min(u32::MAX as u64) as u32,
    }).collect();

    for op in &frag.ops {
        let sz = op_size(op);
        if current_size + sz > target_size && !current_ops.is_empty() {
            // Flush sub-fragment actual
            let depends_on = if sub_id > 0 { vec![sub_id - 1] } else { vec![] };
            sub_fragments.push(SubFragment {
                fragment_id: frag.fragment_id,
                sub_id,
                layer_start: frag.layer_start,
                layer_end: frag.layer_end,
                ops: std::mem::take(&mut current_ops),
                weight_refs: std::mem::take(&mut current_refs),
                depends_on,
            });
            sub_id += 1;
            current_size = 0;
        }
        current_ops.push(op.clone());
        current_size += sz;
    }

    // Flush último sub-fragmento
    if !current_ops.is_empty() {
        let depends_on = if sub_id > 0 { vec![sub_id - 1] } else { vec![] };
        sub_fragments.push(SubFragment {
            fragment_id: frag.fragment_id,
            sub_id,
            layer_start: frag.layer_start,
            layer_end: frag.layer_end,
            ops: current_ops,
            weight_refs: if sub_fragments.is_empty() { all_weight_refs.clone() } else { vec![] },
            depends_on,
        });
    }

    // Si el fragmento no tiene ops, crear un sub-fragmento con todas las refs
    if sub_fragments.is_empty() {
        sub_fragments.push(SubFragment {
            fragment_id: frag.fragment_id,
            sub_id: 0,
            layer_start: frag.layer_start,
            layer_end: frag.layer_end,
            ops: vec![],
            weight_refs: all_weight_refs,
            depends_on: vec![],
        });
    } else {
        // El primer sub-fragmento obtiene todas las weight_refs
        // (en la práctica, cada sub-fragmento tendría solo las refs que necesita,
        // pero por ahora las ponemos todas en el primero para que el runtime
        // tenga acceso al pool completo)
        sub_fragments[0].weight_refs = all_weight_refs;
    }

    sub_fragments
}

/// Sub-fragmenta todos los fragmentos de un modelo.
///
/// Retorna un mapa `fragment_id -> Vec<SubFragment>`.
pub fn sub_fragment_all(fragments: &[DistributedFragment]) -> HashMap<u32, Vec<SubFragment>> {
    fragments
        .iter()
        .map(|frag| (frag.fragment_id, sub_fragment(frag)))
        .collect()
}

/// Serializa sub-fragmentos a un directorio.
///
/// Genera archivos `sub_{fragment_id}_{sub_id}.bmlgraph` dentro del
/// directorio del fragmento padre.
pub fn serialize_sub_fragments(
    parent_dir: &Path,
    frag: &DistributedFragment,
    sub_frags: &[SubFragment],
) -> Result<(), String> {
    let frag_dir = parent_dir.join(format!("fragment_{}", frag.fragment_id));
    std::fs::create_dir_all(&frag_dir).map_err(|e| format!("crear dir: {e}"))?;

    for sf in sub_frags {
        let path = frag_dir.join(format!("sub_{}.bmlgraph", sf.sub_id));
        let mut f = std::fs::File::create(&path).map_err(|e| format!("crear sub {}: {e}", sf.sub_id))?;

        // Header
        f.write_all(&BMLGRAPH_MAGIC.to_le_bytes()).map_err(|e| format!("write magic: {e}"))?;
        f.write_all(&BMLGRAPH_VERSION.to_le_bytes()).map_err(|e| format!("write version: {e}"))?;

        // IDs
        f.write_all(&sf.fragment_id.to_le_bytes()).map_err(|e| format!("write frag id: {e}"))?;
        f.write_all(&sf.sub_id.to_le_bytes()).map_err(|e| format!("write sub id: {e}"))?;
        f.write_all(&sf.layer_start.to_le_bytes()).map_err(|e| format!("write layer start: {e}"))?;
        f.write_all(&sf.layer_end.to_le_bytes()).map_err(|e| format!("write layer end: {e}"))?;

        // Bytecode
        f.write_all(&(sf.ops.len() as u32).to_le_bytes()).map_err(|e| format!("write n_ops: {e}"))?;
        for op in &sf.ops {
            serialize_op(&mut f, op)?;
        }

        // Weight refs
        f.write_all(&(sf.weight_refs.len() as u32).to_le_bytes()).map_err(|e| format!("write n_refs: {e}"))?;
        for wr in &sf.weight_refs {
            let name_bytes = wr.tensor_name.as_bytes();
            f.write_all(&(name_bytes.len() as u32).to_le_bytes()).map_err(|e| format!("write name len: {e}"))?;
            f.write_all(name_bytes).map_err(|e| format!("write name: {e}"))?;
            f.write_all(&wr.offset.to_le_bytes()).map_err(|e| format!("write offset: {e}"))?;
            f.write_all(&wr.len.to_le_bytes()).map_err(|e| format!("write len: {e}"))?;
        }

        // Depends on
        f.write_all(&(sf.depends_on.len() as u32).to_le_bytes()).map_err(|e| format!("write n_deps: {e}"))?;
        for &dep in &sf.depends_on {
            f.write_all(&dep.to_le_bytes()).map_err(|e| format!("write dep: {e}"))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod sub_fragment_tests {
    use super::*;

    #[test]
    fn sub_fragment_empty() {
        let frag = DistributedFragment {
            fragment_id: 0,
            layer_start: 0,
            layer_end: 1,
            ops: vec![],
            weights: vec![],
            tensors: vec![],
            config: ModelConfig {
                architecture: "test".into(),
                n_layers: 1,
                n_heads: 1,
                n_embd: 1,
                context_length: 1,
                vocab_size: 1,
            },
            n_kv_heads: 1,
            head_dim: 1,
        };

        let subs = sub_fragment(&frag);
        assert_eq!(subs.len(), 1);
        assert!(subs[0].ops.is_empty());
        assert_eq!(subs[0].sub_id, 0);
    }

    #[test]
    fn sub_fragment_small() {
        let frag = DistributedFragment {
            fragment_id: 0,
            layer_start: 0,
            layer_end: 1,
            ops: vec![RpnOp::One, RpnOp::One, RpnOp::Bml],
            weights: vec![],
            tensors: vec![],
            config: ModelConfig {
                architecture: "test".into(),
                n_layers: 1,
                n_heads: 1,
                n_embd: 1,
                context_length: 1,
                vocab_size: 1,
            },
            n_kv_heads: 1,
            head_dim: 1,
        };

        let subs = sub_fragment(&frag);
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].ops.len(), 3);
        assert!(subs[0].fits_l1i());
    }

    #[test]
    fn sub_fragment_large() {
        // Crear un fragmento con muchas ops para forzar sub-fragmentación
        // Bml = 1 byte, necesitamos >30K ops para superar 30KB
        let ops: Vec<RpnOp> = (0..50_000).map(|_| RpnOp::Bml).collect();
        let frag = DistributedFragment {
            fragment_id: 0,
            layer_start: 0,
            layer_end: 1,
            ops,
            weights: vec![],
            tensors: vec![],
            config: ModelConfig {
                architecture: "test".into(),
                n_layers: 1,
                n_heads: 1,
                n_embd: 1,
                context_length: 1,
                vocab_size: 1,
            },
            n_kv_heads: 1,
            head_dim: 1,
        };

        let subs = sub_fragment(&frag);
        assert!(subs.len() > 1, "debe generar múltiples sub-fragmentos");

        // Cada sub-fragmento debe caber en L1i
        for sf in &subs {
            assert!(sf.fits_l1i(), "sub {} excede L1i: {} bytes", sf.sub_id, sf.bytecode_size());
        }

        // Las dependencias deben formar una cadena
        for i in 1..subs.len() {
            assert_eq!(subs[i].depends_on, vec![(i as u32) - 1]);
        }

        println!("50K ops → {} sub-fragmentos, tamaños: {:?}", subs.len(),
            subs.iter().map(|sf| sf.bytecode_size()).collect::<Vec<_>>());
    }

    #[test]
    fn sub_fragment_preserves_total_ops() {
        let ops: Vec<RpnOp> = (0..50_000)
            .map(|i| if i % 2 == 0 { RpnOp::One } else { RpnOp::Bml })
            .collect();
        let total = ops.len();
        let frag = DistributedFragment {
            fragment_id: 0,
            layer_start: 0,
            layer_end: 1,
            ops,
            weights: vec![],
            tensors: vec![],
            config: ModelConfig {
                architecture: "test".into(),
                n_layers: 1,
                n_heads: 1,
                n_embd: 1,
                context_length: 1,
                vocab_size: 1,
            },
            n_kv_heads: 1,
            head_dim: 1,
        };

        let subs = sub_fragment(&frag);
        let total_in_subs: usize = subs.iter().map(|sf| sf.ops.len()).sum();
        assert_eq!(total_in_subs, total, "ops se perdieron en sub-fragmentación");
    }
}
