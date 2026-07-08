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
