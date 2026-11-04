//! Compilación distribuida: fragmenta un GGUF por capa del transformer.
//!
//! Cada fragmento `.bmlgraph` contiene bytecode RPN y pesos f32. Su configuración
//! se recupera del archivo `header.bmlgraph` del mismo directorio; ambos son
//! necesarios para cargar un shard, sin necesidad del GGUF original.
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
//!   [n_kv_heads 4B][head_dim 4B]
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

use crate::fragment::{BMLGRAPH_MAGIC, BMLGRAPH_SELECTION_VERSION, BMLGRAPH_VERSION};
use crate::gguf_compiler::{read_model_config, read_tensor_f32, ModelConfig};
use crate::rpn::RpnOp;
use bml_domain::encoder::RealEncoder;
use bml_parser::GgufParser;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// Metadatos de un tensor dentro de un fragmento.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// Nombre del tensor en el archivo GGUF.
    pub name: String,
    /// Índice del primer elemento del tensor en el vector de pesos del fragmento.
    pub offset: u64,
    /// Primera dimensión GGUF, o número de valores si no hay dimensiones.
    pub n_rows: u32,
    /// Segunda dimensión GGUF, o uno si el tensor tiene menos de dos dimensiones.
    pub n_cols: u32,
}

/// Un fragmento distribuido con configuración restaurada.
///
/// En memoria contiene bytecode, pesos, metadata y configuración; en disco
/// la configuración compartida se guarda en el header del directorio.
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

/// Compila un GGUF a fragmentos distribuidos con pesos embebidos.
///
/// Cada capa del transformer se convierte en un fragmento independiente
/// con sus pesos embebidos. Un nodo carga el fragmento y su header compartido
/// para recuperar la configuración sin el GGUF original.
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
    if layers_per_fragment == 0 {
        return Err("layers_per_fragment must be greater than zero".into());
    }
    let parser = GgufParser::open(gguf_path).map_err(|e| format!("parser: {e}"))?;
    let config = read_model_config(&parser)?;

    let head_dim = config.n_embd / config.n_heads;
    let n_kv_heads = parser
        .get_metadata(&format!("{}.attention.head_count_kv", config.architecture))
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
        let layer_end = layer
            .saturating_add(layers_per_fragment)
            .min(config.n_layers);

        let frag = compile_layer_range(
            &parser, &config, layer, layer_end, head_dim, n_kv_heads, false,
        )?;
        fragments.push(frag);

        layer = layer_end;
    }

    // Fragmento final: embeddings + output_norm + lm_head
    if include_embeddings {
        let frag = compile_layer_range(
            &parser,
            &config,
            config.n_layers,
            config.n_layers + 1,
            head_dim,
            n_kv_heads,
            true,
        )?;
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

    let add_tensor = |parser: &GgufParser,
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
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.attn_norm.weight"),
            )?;
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.attn_q.weight"),
            )?;
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.attn_k.weight"),
            )?;
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.attn_v.weight"),
            )?;
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.attn_output.weight"),
            )?;
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.ffn_norm.weight"),
            )?;
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.ffn_gate.weight"),
            )?;
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.ffn_up.weight"),
            )?;
            add_tensor(
                parser,
                &mut weights,
                &mut tensors,
                &format!("{prefix}.ffn_down.weight"),
            )?;
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
/// - `fragment_N.bmlgraph`: bytecode + pesos + tensors; requiere el header hermano
pub fn serialize_distributed(
    fragments: &[DistributedFragment],
    output_dir: &Path,
) -> Result<(), String> {
    if fragments.iter().flat_map(|f| &f.ops).any(|op| {
        matches!(
            op,
            RpnOp::ContextVarIndexed { .. } | RpnOp::ContextConstIndexed { .. }
        )
    }) {
        return Err("indexed context loads require the single-file v3 graph format".into());
    }
    let version = if fragments
        .iter()
        .flat_map(|f| &f.ops)
        .any(|op| matches!(op, RpnOp::FMax))
    {
        BMLGRAPH_SELECTION_VERSION
    } else {
        BMLGRAPH_VERSION
    };
    std::fs::create_dir_all(output_dir).map_err(|e| format!("crear dir: {e}"))?;

    // Header: magic, version, n_fragments, config
    let header_path = output_dir.join("header.bmlgraph");
    let mut f = std::fs::File::create(&header_path).map_err(|e| format!("crear header: {e}"))?;

    f.write_all(&BMLGRAPH_MAGIC.to_le_bytes())
        .map_err(|e| format!("write magic: {e}"))?;
    f.write_all(&version.to_le_bytes())
        .map_err(|e| format!("write version: {e}"))?;
    f.write_all(&(fragments.len() as u32).to_le_bytes())
        .map_err(|e| format!("write n_frag: {e}"))?;

    let config = &fragments[0].config;
    let arch_bytes = config.architecture.as_bytes();
    f.write_all(&(arch_bytes.len() as u64).to_le_bytes())
        .map_err(|e| format!("write arch len: {e}"))?;
    f.write_all(arch_bytes)
        .map_err(|e| format!("write arch: {e}"))?;
    f.write_all(&config.n_layers.to_le_bytes())
        .map_err(|e| format!("write n_layers: {e}"))?;
    f.write_all(&config.n_heads.to_le_bytes())
        .map_err(|e| format!("write n_heads: {e}"))?;
    f.write_all(&config.n_embd.to_le_bytes())
        .map_err(|e| format!("write n_embd: {e}"))?;
    f.write_all(&config.context_length.to_le_bytes())
        .map_err(|e| format!("write ctx: {e}"))?;
    f.write_all(&config.vocab_size.to_le_bytes())
        .map_err(|e| format!("write vocab: {e}"))?;

    // Fragmentos: la configuración compartida está en el header.
    for frag in fragments {
        let frag_path = output_dir.join(format!("fragment_{}.bmlgraph", frag.fragment_id));
        let mut ff = std::fs::File::create(&frag_path)
            .map_err(|e| format!("crear frag {}: {e}", frag.fragment_id))?;

        // Magic + version
        ff.write_all(&BMLGRAPH_MAGIC.to_le_bytes())
            .map_err(|e| format!("write magic: {e}"))?;
        ff.write_all(&version.to_le_bytes())
            .map_err(|e| format!("write version: {e}"))?;

        // Fragment ID
        ff.write_all(&frag.fragment_id.to_le_bytes())
            .map_err(|e| format!("write frag id: {e}"))?;

        // Layer range
        ff.write_all(&frag.layer_start.to_le_bytes())
            .map_err(|e| format!("write layer start: {e}"))?;
        ff.write_all(&frag.layer_end.to_le_bytes())
            .map_err(|e| format!("write layer end: {e}"))?;

        // Bytecode RPN
        ff.write_all(&(frag.ops.len() as u32).to_le_bytes())
            .map_err(|e| format!("write n_ops: {e}"))?;
        for op in &frag.ops {
            serialize_op(&mut ff, op)?;
        }

        // Pesos f32
        ff.write_all(&(frag.weights.len() as u64).to_le_bytes())
            .map_err(|e| format!("write n_weights: {e}"))?;
        for &w in &frag.weights {
            ff.write_all(&w.to_le_bytes())
                .map_err(|e| format!("write weight: {e}"))?;
        }

        // Tensor metadata
        ff.write_all(&(frag.tensors.len() as u32).to_le_bytes())
            .map_err(|e| format!("write n_tensors: {e}"))?;
        for tensor in &frag.tensors {
            let name_bytes = tensor.name.as_bytes();
            ff.write_all(&(name_bytes.len() as u32).to_le_bytes())
                .map_err(|e| format!("write name len: {e}"))?;
            ff.write_all(name_bytes)
                .map_err(|e| format!("write name: {e}"))?;
            ff.write_all(&tensor.offset.to_le_bytes())
                .map_err(|e| format!("write offset: {e}"))?;
            ff.write_all(&tensor.n_rows.to_le_bytes())
                .map_err(|e| format!("write n_rows: {e}"))?;
            ff.write_all(&tensor.n_cols.to_le_bytes())
                .map_err(|e| format!("write n_cols: {e}"))?;
        }

        // Config del modelo (repetida por fragmento para self-contained)
        ff.write_all(&(frag.n_kv_heads as u32).to_le_bytes())
            .map_err(|e| format!("write n_kv_heads: {e}"))?;
        ff.write_all(&frag.head_dim.to_le_bytes())
            .map_err(|e| format!("write head_dim: {e}"))?;
    }

    Ok(())
}

/// Maximum accepted sizes for the legacy distributed container. These bounds
/// apply before allocations driven by encoded counts; they do not validate RPN
/// execution semantics.
const MAX_DISTRIBUTED_FRAGMENT_BYTES: usize = 512 * 1024 * 1024;
const MAX_DISTRIBUTED_HEADER_BYTES: usize = 1024 * 1024;
const MAX_DISTRIBUTED_OPS: usize = 1_000_000;
const MAX_DISTRIBUTED_TENSORS: usize = 65_536;

struct DistributedReader<'a> {
    bytes: &'a [u8],
}

impl<'a> DistributedReader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let (value, rest) = self
            .bytes
            .split_at_checked(count)
            .ok_or("truncated distributed file")?;
        self.bytes = rest;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn text(&mut self, count: usize) -> Result<String, String> {
        String::from_utf8(self.take(count)?.to_vec())
            .map_err(|_| "invalid distributed UTF-8".into())
    }
    fn prefix(&mut self) -> Result<u32, String> {
        if self.u32()? != BMLGRAPH_MAGIC {
            return Err("invalid distributed magic".into());
        }
        let version = self.u32()?;
        if !matches!(version, BMLGRAPH_VERSION | BMLGRAPH_SELECTION_VERSION) {
            return Err("unsupported distributed version".into());
        }
        Ok(version)
    }
    fn finish(self) -> Result<(), String> {
        if self.bytes.is_empty() {
            Ok(())
        } else {
            Err("trailing distributed bytes".into())
        }
    }
}

/// Load a legacy shard together with the configuration in its sibling header.
///
/// The fragment is bounded to 512 MiB, one million opcodes and 65,536 tensors;
/// the header is bounded to 1 MiB. Counts, ranges, UTF-8 and complete consumption
/// are checked before count-driven allocations. RPN execution needs its own
/// semantic validation, and model execution must validate operation shapes.
pub fn load_distributed_fragment(frag_path: &Path) -> Result<DistributedFragment, String> {
    let directory = frag_path.parent().unwrap_or_else(|| Path::new("."));
    let (config, _) = load_distributed_header(directory)?;
    let bytes = crate::causal_artifact::read_bounded(frag_path, MAX_DISTRIBUTED_FRAGMENT_BYTES)?;
    let mut reader = DistributedReader { bytes: &bytes };
    let version = reader.prefix()?;
    let fragment_id = reader.u32()?;
    let layer_start = reader.u32()?;
    let layer_end = reader.u32()?;
    if layer_start > layer_end || layer_end > config.n_layers {
        return Err("invalid distributed layer range".into());
    }
    let n_ops = reader.u32()? as usize;
    if n_ops > MAX_DISTRIBUTED_OPS || n_ops > reader.bytes.len() {
        return Err("distributed opcode count exceeds budget or available bytes".into());
    }
    let mut ops = Vec::with_capacity(n_ops);
    for _ in 0..n_ops {
        let (op, consumed) = deserialize_op(reader.bytes)?;
        if op == RpnOp::FMax && version < BMLGRAPH_SELECTION_VERSION {
            return Err("maximum selection requires distributed v4".into());
        }
        reader.take(consumed)?;
        ops.push(op);
    }
    let n_weights =
        usize::try_from(reader.u64()?).map_err(|_| "distributed weight count overflow")?;
    let weight_bytes = n_weights
        .checked_mul(4)
        .ok_or("distributed weight byte overflow")?;
    let raw_weights = reader.take(weight_bytes)?;
    let weights: Vec<f32> = raw_weights
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect();
    let n_tensors = reader.u32()? as usize;
    // The final two u32 fields are not tensor metadata.
    if n_tensors > MAX_DISTRIBUTED_TENSORS || n_tensors > reader.bytes.len().saturating_sub(8) / 20
    {
        return Err("distributed tensor count exceeds budget or available bytes".into());
    }
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name_len = reader.u32()? as usize;
        let name = reader.text(name_len)?;
        let offset = reader.u64()?;
        let n_rows = reader.u32()?;
        let n_cols = reader.u32()?;
        let end = u64::from(n_rows)
            .checked_mul(u64::from(n_cols))
            .and_then(|size| offset.checked_add(size))
            .ok_or("distributed tensor range overflow")?;
        if end > n_weights as u64 {
            return Err("distributed tensor exceeds weight storage".into());
        }
        tensors.push(TensorMeta {
            name,
            offset,
            n_rows,
            n_cols,
        });
    }
    let n_kv_heads = reader.u32()?;
    let head_dim = reader.u32()?;
    reader.finish()?;
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

/// Load the shared distributed configuration, checking the complete v2 header.
pub fn load_distributed_header(dir: &Path) -> Result<(ModelConfig, usize), String> {
    let bytes = crate::causal_artifact::read_bounded(
        &dir.join("header.bmlgraph"),
        MAX_DISTRIBUTED_HEADER_BYTES,
    )?;
    let mut reader = DistributedReader { bytes: &bytes };
    reader.prefix()?;
    let n_fragments = reader.u32()? as usize;
    if n_fragments == 0 || n_fragments > MAX_DISTRIBUTED_TENSORS {
        return Err("invalid distributed fragment count".into());
    }
    let arch_len =
        usize::try_from(reader.u64()?).map_err(|_| "distributed architecture length overflow")?;
    let architecture = reader.text(arch_len)?;
    let config = ModelConfig {
        architecture,
        n_layers: reader.u32()?,
        n_heads: reader.u32()?,
        n_embd: reader.u32()?,
        context_length: reader.u32()?,
        vocab_size: reader.u32()?,
    };
    reader.finish()?;
    Ok((config, n_fragments))
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
            f.write_all(&count.to_le_bytes())
                .map_err(|e| format!("write op: {e}"))?;
            f.write_all(&body_len.to_le_bytes())
                .map_err(|e| format!("write op: {e}"))
        }
        RpnOp::Var(id) => {
            f.write_all(&[4]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&id.to_le_bytes())
                .map_err(|e| format!("write op: {e}"))
        }
        RpnOp::Const(id) => {
            f.write_all(&[5]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&id.to_le_bytes())
                .map_err(|e| format!("write op: {e}"))
        }
        RpnOp::VarIndexed { base } => {
            f.write_all(&[7]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&base.to_le_bytes())
                .map_err(|e| format!("write op: {e}"))
        }
        RpnOp::StoreResult { slot } => {
            f.write_all(&[8]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&slot.to_le_bytes())
                .map_err(|e| format!("write op: {e}"))
        }
        RpnOp::FAdd => f.write_all(&[9]).map_err(|e| format!("write op: {e}")),
        RpnOp::FMul => f.write_all(&[10]).map_err(|e| format!("write op: {e}")),
        RpnOp::FMax => f.write_all(&[16]).map_err(|e| format!("write op: {e}")),
        RpnOp::Pick { depth } => {
            f.write_all(&[11]).map_err(|e| format!("write op: {e}"))?;
            f.write_all(&depth.to_le_bytes())
                .map_err(|e| format!("write op: {e}"))
        }
        RpnOp::Drop => f.write_all(&[12]).map_err(|e| format!("write op: {e}")),
        RpnOp::Swap => f.write_all(&[13]).map_err(|e| format!("write op: {e}")),
        RpnOp::ContextVarIndexed { .. } | RpnOp::ContextConstIndexed { .. } => {
            Err("indexed context loads require the single-file v3 graph format".into())
        }
    }
}

/// Deserializa una operación RPN desde bytes. Retorna (op, bytes_consumed).
fn deserialize_op(bytes: &[u8]) -> Result<(RpnOp, usize), String> {
    if bytes.is_empty() {
        return Err("bytes insuficientes para op".into());
    }
    let tag = bytes[0];
    let width = match tag {
        0 | 1 | 2 | 6 | 9 | 10 | 12 | 13 | 16 => 1,
        3 => 9,
        4 | 5 | 7 | 8 | 11 => 5,
        _ => return Err(format!("tag desconocido: {tag}")),
    };
    if bytes.len() < width {
        return Err("truncated distributed opcode".into());
    }
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
        16 => (RpnOp::FMax, 1),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::HardwareSpec;

    #[test]
    fn opcode_decoder_accepts_exact_width_and_rejects_each_truncated_prefix() {
        let cases = [
            (vec![0], RpnOp::One),
            (vec![1], RpnOp::Bml),
            (vec![2], RpnOp::Dup),
            (
                vec![3, 2, 0, 0, 0, 1, 0, 0, 0],
                RpnOp::Loop {
                    count: 2,
                    body_len: 1,
                },
            ),
            (vec![4, 7, 0, 0, 0], RpnOp::Var(7)),
            (vec![5, 7, 0, 0, 0], RpnOp::Const(7)),
            (vec![6], RpnOp::Zero),
            (vec![7, 7, 0, 0, 0], RpnOp::VarIndexed { base: 7 }),
            (vec![8, 7, 0, 0, 0], RpnOp::StoreResult { slot: 7 }),
            (vec![9], RpnOp::FAdd),
            (vec![10], RpnOp::FMul),
            (vec![11, 7, 0, 0, 0], RpnOp::Pick { depth: 7 }),
            (vec![12], RpnOp::Drop),
            (vec![13], RpnOp::Swap),
        ];
        for (bytes, expected) in cases {
            assert_eq!(deserialize_op(&bytes).unwrap(), (expected, bytes.len()));
            for cut in 0..bytes.len() {
                assert!(
                    deserialize_op(&bytes[..cut]).is_err(),
                    "tag={} cut={cut}",
                    bytes[0]
                );
            }
            let mut following = bytes.clone();
            following.push(255);
            assert_eq!(deserialize_op(&following).unwrap(), (expected, bytes.len()));
        }
    }

    #[test]
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn compile_distributed_tinyllama() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();

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
                frag.fragment_id,
                frag.layer_start,
                frag.layer_end,
                frag.tensors.len(),
                weight_mb,
            );
        }

        // Verificar que la suma de pesos ≈ peso total del modelo
        let total_weights: usize = fragments.iter().map(|f| f.weights.len()).sum();
        println!(
            "Total weights: {} ({:.1} MB)",
            total_weights,
            total_weights as f64 * 4.0 / (1024.0 * 1024.0)
        );
    }

    #[test]
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn serialize_load_distributed_roundtrip() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();

        let fragments = compile_distributed(Path::new(path), 4, true).expect("compile");
        let out_dir = std::env::temp_dir().join(format!("bml_dist_test_{}", std::process::id()));

        serialize_distributed(&fragments, &out_dir).expect("serialize");
        println!(
            "Serialized {} fragments to {}",
            fragments.len(),
            out_dir.display()
        );

        // Listar archivos
        for entry in std::fs::read_dir(&out_dir).unwrap() {
            let entry = entry.unwrap();
            let meta = entry.metadata().unwrap();
            println!(
                "  {} ({} bytes)",
                entry.file_name().to_string_lossy(),
                meta.len()
            );
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
        for (i, (a, b)) in frag0
            .weights
            .iter()
            .zip(fragments[0].weights.iter())
            .enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "peso {i} no coincide");
        }

        println!(
            "Roundtrip OK: {} tensors, {} weights",
            frag0.tensors.len(),
            frag0.weights.len()
        );

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
    /// Crea un pool vacío, sin pesos codificados.
    pub fn new() -> Self {
        Self {
            encoder: RealEncoder::new(),
            n_total: 0,
        }
    }

    /// Crea un pool vacío con capacidad inicial para codificar `cap` pesos.
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
    if layers_per_fragment == 0 {
        return Err("layers_per_fragment must be greater than zero".into());
    }
    let parser = GgufParser::open(gguf_path).map_err(|e| format!("parser: {e}"))?;
    let config = read_model_config(&parser)?;

    let head_dim = config.n_embd / config.n_heads;
    let n_kv_heads = parser
        .get_metadata(&format!("{}.attention.head_count_kv", config.architecture))
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
        let layer_end = layer
            .saturating_add(layers_per_fragment)
            .min(config.n_layers);
        let frag = compile_layer_range_bml(
            &parser, &config, layer, layer_end, head_dim, n_kv_heads, false, &mut pool,
        )?;
        fragments.push(frag);
        layer = layer_end;
    }

    // Fragmento final
    let frag = compile_layer_range_bml(
        &parser,
        &config,
        config.n_layers,
        config.n_layers + 1,
        head_dim,
        n_kv_heads,
        true,
        &mut pool,
    )?;
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
                tensors.push(TensorMeta {
                    name: "output_norm.weight".into(),
                    offset,
                    n_rows: config.n_embd,
                    n_cols: 1,
                });
            }
            if let Some(vals) = read_tensor_f32(parser, "token_embd.weight") {
                let offset = weights.len() as u64;
                let info = parser.find_tensor("token_embd.weight");
                let (r, c) = if let Some(i) = info {
                    if i.dims.len() >= 2 {
                        (i.dims[0] as u32, i.dims[1] as u32)
                    } else {
                        (vals.len() as u32, 1)
                    }
                } else {
                    (vals.len() as u32, 1)
                };
                weights.extend(vals);
                tensors.push(TensorMeta {
                    name: "token_embd.weight".into(),
                    offset,
                    n_rows: r,
                    n_cols: c,
                });
            }
        } else {
            let prefix = format!("blk.{layer}");
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                let name = format!("{prefix}.{suffix}");
                if let Some(vals) = read_tensor_f32(parser, &name) {
                    let info = parser.find_tensor(&name);
                    let (r, c) = if let Some(i) = info {
                        if i.dims.len() >= 2 {
                            (i.dims[0] as u32, i.dims[1] as u32)
                        } else {
                            (vals.len() as u32, 1)
                        }
                    } else {
                        (vals.len() as u32, 1)
                    };
                    let offset = weights.len() as u64;
                    weights.extend(vals);
                    tensors.push(TensorMeta {
                        name,
                        offset,
                        n_rows: r,
                        n_cols: c,
                    });
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
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];

        // Simular 1M pesos con valores Q4_0
        for _ in 0..62_500 {
            for &v in &q4_values {
                pool.encode(v);
            }
        }

        println!(
            "Q4_0 pool: {} unique, {} total",
            pool.n_unique(),
            pool.n_total()
        );
        println!("  Compression ratio: {:.1}x", pool.compression_ratio());
        println!(
            "  Estimated size: {} bytes ({:.1} MB)",
            pool.estimated_size_bytes(),
            pool.estimated_size_bytes() as f64 / 1_000_000.0
        );
        println!(
            "  f32 size: {} bytes ({:.1} MB)",
            pool.f32_size_bytes(),
            pool.f32_size_bytes() as f64 / 1_000_000.0
        );
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
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn compile_distributed_bml_tinyllama() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();

        let (fragments, pool) = compile_distributed_bml(Path::new(path), 1).expect("compile bml");

        println!("\n=== BML Weight Pool Statistics ===");
        println!("  Unique values: {}", pool.n_unique());
        println!("  Total weights: {}", pool.n_total());
        println!("  Compression ratio: {:.1}x", pool.compression_ratio());
        println!(
            "  Estimated BML size: {:.1} MB",
            pool.estimated_size_bytes() as f64 / 1_000_000.0
        );
        println!(
            "  f32 size: {:.1} MB",
            pool.f32_size_bytes() as f64 / 1_000_000.0
        );
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
        RpnOp::One
        | RpnOp::Zero
        | RpnOp::Bml
        | RpnOp::Dup
        | RpnOp::FAdd
        | RpnOp::FMul
        | RpnOp::FMax
        | RpnOp::Drop
        | RpnOp::Swap => 1,
        RpnOp::Var(_)
        | RpnOp::Const(_)
        | RpnOp::VarIndexed { .. }
        | RpnOp::ContextVarIndexed { .. }
        | RpnOp::ContextConstIndexed { .. }
        | RpnOp::StoreResult { .. }
        | RpnOp::Pick { .. } => 5,
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
pub fn sub_fragment_with_threshold(
    frag: &DistributedFragment,
    target_size: usize,
) -> Vec<SubFragment> {
    let mut sub_fragments = Vec::new();
    let mut current_ops = Vec::new();
    let mut current_refs = Vec::new();
    let mut current_size = 0usize;
    let mut sub_id = 0u32;

    // Crear referencias a todos los tensores del fragmento
    let all_weight_refs: Vec<WeightRef> = frag
        .tensors
        .iter()
        .map(|t| WeightRef {
            tensor_name: t.name.clone(),
            offset: t.offset,
            len: (t.n_rows as u64 * t.n_cols as u64).min(u32::MAX as u64) as u32,
        })
        .collect();

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
            weight_refs: if sub_fragments.is_empty() {
                all_weight_refs.clone()
            } else {
                vec![]
            },
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
    if sub_frags.iter().flat_map(|f| &f.ops).any(|op| {
        matches!(
            op,
            RpnOp::ContextVarIndexed { .. } | RpnOp::ContextConstIndexed { .. }
        )
    }) {
        return Err("indexed context loads require the single-file v3 graph format".into());
    }
    let frag_dir = parent_dir.join(format!("fragment_{}", frag.fragment_id));
    std::fs::create_dir_all(&frag_dir).map_err(|e| format!("crear dir: {e}"))?;

    for sf in sub_frags {
        let version = if sf.ops.iter().any(|op| matches!(op, RpnOp::FMax)) {
            BMLGRAPH_SELECTION_VERSION
        } else {
            BMLGRAPH_VERSION
        };
        let path = frag_dir.join(format!("sub_{}.bmlgraph", sf.sub_id));
        let mut f =
            std::fs::File::create(&path).map_err(|e| format!("crear sub {}: {e}", sf.sub_id))?;

        // Header
        f.write_all(&BMLGRAPH_MAGIC.to_le_bytes())
            .map_err(|e| format!("write magic: {e}"))?;
        f.write_all(&version.to_le_bytes())
            .map_err(|e| format!("write version: {e}"))?;

        // IDs
        f.write_all(&sf.fragment_id.to_le_bytes())
            .map_err(|e| format!("write frag id: {e}"))?;
        f.write_all(&sf.sub_id.to_le_bytes())
            .map_err(|e| format!("write sub id: {e}"))?;
        f.write_all(&sf.layer_start.to_le_bytes())
            .map_err(|e| format!("write layer start: {e}"))?;
        f.write_all(&sf.layer_end.to_le_bytes())
            .map_err(|e| format!("write layer end: {e}"))?;

        // Bytecode
        f.write_all(&(sf.ops.len() as u32).to_le_bytes())
            .map_err(|e| format!("write n_ops: {e}"))?;
        for op in &sf.ops {
            serialize_op(&mut f, op)?;
        }

        // Weight refs
        f.write_all(&(sf.weight_refs.len() as u32).to_le_bytes())
            .map_err(|e| format!("write n_refs: {e}"))?;
        for wr in &sf.weight_refs {
            let name_bytes = wr.tensor_name.as_bytes();
            f.write_all(&(name_bytes.len() as u32).to_le_bytes())
                .map_err(|e| format!("write name len: {e}"))?;
            f.write_all(name_bytes)
                .map_err(|e| format!("write name: {e}"))?;
            f.write_all(&wr.offset.to_le_bytes())
                .map_err(|e| format!("write offset: {e}"))?;
            f.write_all(&wr.len.to_le_bytes())
                .map_err(|e| format!("write len: {e}"))?;
        }

        // Depends on
        f.write_all(&(sf.depends_on.len() as u32).to_le_bytes())
            .map_err(|e| format!("write n_deps: {e}"))?;
        for &dep in &sf.depends_on {
            f.write_all(&dep.to_le_bytes())
                .map_err(|e| format!("write dep: {e}"))?;
        }
    }

    Ok(())
}

// ===========================================================================
// VectorFragment: distribución de columnas de matmul
// ===========================================================================

/// Un fragmento de vector para distribución de matmul por columnas.
///
/// El matmul `y = x · W` se parte por columnas de W. Cada `VectorFragment`
/// contiene:
/// - El vector de entrada `x` (compartido entre todos los fragments)
/// - Una columna (o slice de columnas) de pesos `W[:, j_start..j_end]`
/// - El slot de salida para `y[j_start..j_end]`
///
/// Multiple fragments → multiple nodos en paralelo (columnas independientes).
#[derive(Debug, Clone)]
pub struct VectorFragment {
    /// Identificador del fragmento.
    pub fragment_id: u32,
    /// Vector de entrada compartido por las particiones del matmul.
    pub x: Vec<f64>,
    /// Pesos de las columnas seleccionadas, almacenados por filas.
    pub w_columns: Vec<f64>,
    /// Número de filas de la matriz de pesos y dimensión de entrada.
    pub n_in: u32,
    /// Número de columnas de la matriz de pesos completa.
    pub n_out_total: u32,
    /// Índice de la primera columna de esta partición en la matriz completa.
    pub j_start: u32,
    /// Número de columnas incluidas en esta partición.
    pub n_cols: u32,
    /// Slot de destino del vector de salida completo.
    pub output_slot: u32,
}

impl VectorFragment {
    /// Copia la entrada y extrae `n_cols` columnas desde `j_start`.
    ///
    /// `weight_offset` es un índice de elementos en `weights`. La matriz fuente
    /// se interpreta por filas con `n_out_total` columnas; los índices que
    /// exceden `weights` se rellenan con cero.
    pub fn new(
        fragment_id: u32,
        x: &[f64],
        weights: &[f32],
        weight_offset: u64,
        n_in: u32,
        n_out_total: u32,
        j_start: u32,
        n_cols: u32,
        output_slot: u32,
    ) -> Self {
        let n_in_us = n_in as usize;
        let n_cols_us = n_cols as usize;
        let n_out_us = n_out_total as usize;

        let mut w_columns = Vec::with_capacity(n_in_us * n_cols_us);
        for i in 0..n_in_us {
            for jj in 0..n_cols_us {
                let j = j_start as usize + jj;
                let idx = weight_offset as usize + i * n_out_us + j;
                w_columns.push(if idx < weights.len() {
                    weights[idx] as f64
                } else {
                    0.0
                });
            }
        }

        Self {
            fragment_id,
            x: x.to_vec(),
            w_columns,
            n_in,
            n_out_total,
            j_start,
            n_cols,
            output_slot,
        }
    }

    /// Ejecuta el matmul de este fragmento: y = x · W_columns.
    ///
    /// Retorna `y[j_start..j_start+n_cols]` como Vec<f64>.
    pub fn execute(&self) -> Vec<f64> {
        let n_in = self.n_in as usize;
        let n_cols = self.n_cols as usize;
        let mut y = vec![0.0_f64; n_cols];

        for j in 0..n_cols {
            let mut dot = 0.0_f64;
            for i in 0..n_in.min(self.x.len()) {
                let w_idx = i * n_cols + j;
                if w_idx < self.w_columns.len() {
                    dot += self.x[i] * self.w_columns[w_idx];
                }
            }
            y[j] = dot;
        }

        y
    }

    /// Tamaño de los pesos en bytes.
    pub fn weights_size_bytes(&self) -> usize {
        self.w_columns.len() * 8
    }

    /// Tamaño del vector x en bytes.
    pub fn x_size_bytes(&self) -> usize {
        self.x.len() * 8
    }

    /// Tamaño total del fragmento en bytes.
    pub fn total_size_bytes(&self) -> usize {
        self.weights_size_bytes() + self.x_size_bytes()
    }
}

/// Parte un matmul en `n_fragments` VectorFragments por columnas.
///
/// Cada fragmento contiene un slice de columnas de W.
/// Los fragments son independientes → ejecución paralela.
pub fn split_matmul_columns(
    x: &[f64],
    weights: &[f32],
    weight_offset: u64,
    n_in: u32,
    n_out: u32,
    n_fragments: u32,
    output_slot: u32,
) -> Vec<VectorFragment> {
    let cols_per_frag = (n_out + n_fragments - 1) / n_fragments;
    let mut fragments = Vec::with_capacity(n_fragments as usize);

    for frag_id in 0..n_fragments {
        let j_start = frag_id * cols_per_frag;
        if j_start >= n_out {
            break;
        }
        let n_cols = cols_per_frag.min(n_out - j_start);

        // El offset de este fragmento es weight_offset + j_start
        // (cada columna empieza en j_start dentro del pool de pesos)
        fragments.push(VectorFragment::new(
            frag_id,
            x,
            weights,
            weight_offset,
            n_in,
            n_out,
            j_start,
            n_cols,
            output_slot,
        ));
    }

    fragments
}

/// Serializa un VectorFragment a bytes (para envío via TCP).
pub fn serialize_vector_fragment(vf: &VectorFragment) -> Vec<u8> {
    let mut bytes = Vec::new();

    // fragment_id
    bytes.extend_from_slice(&vf.fragment_id.to_le_bytes());
    // n_in, n_out_total, j_start, n_cols, output_slot
    bytes.extend_from_slice(&vf.n_in.to_le_bytes());
    bytes.extend_from_slice(&vf.n_out_total.to_le_bytes());
    bytes.extend_from_slice(&vf.j_start.to_le_bytes());
    bytes.extend_from_slice(&vf.n_cols.to_le_bytes());
    bytes.extend_from_slice(&vf.output_slot.to_le_bytes());

    // x vector
    bytes.extend_from_slice(&(vf.x.len() as u64).to_le_bytes());
    for &v in &vf.x {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    // w_columns
    bytes.extend_from_slice(&(vf.w_columns.len() as u64).to_le_bytes());
    for &v in &vf.w_columns {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    bytes
}

/// Deserializa un VectorFragment desde bytes (recibido via TCP).
pub fn deserialize_vector_fragment(bytes: &[u8]) -> Result<VectorFragment, String> {
    let mut reader = DistributedReader { bytes };
    let fragment_id = reader.u32()?;
    let n_in = reader.u32()?;
    let n_out_total = reader.u32()?;
    let j_start = reader.u32()?;
    let n_cols = reader.u32()?;
    let output_slot = reader.u32()?;
    if j_start
        .checked_add(n_cols)
        .filter(|end| *end <= n_out_total)
        .is_none()
    {
        return Err("invalid vector fragment column range".into());
    }
    let x_len = usize::try_from(reader.u64()?).map_err(|_| "vector input length overflow")?;
    if x_len != n_in as usize {
        return Err("vector input length does not match n_in".into());
    }
    let x_bytes = reader.take(
        x_len
            .checked_mul(8)
            .ok_or("vector input byte length overflow")?,
    )?;
    let x = x_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    let w_len = usize::try_from(reader.u64()?).map_err(|_| "vector weight length overflow")?;
    let expected = (n_in as usize)
        .checked_mul(n_cols as usize)
        .ok_or("vector weight count overflow")?;
    if w_len != expected {
        return Err("vector weight length does not match dimensions".into());
    }
    let w_bytes = reader.take(
        w_len
            .checked_mul(8)
            .ok_or("vector weight byte length overflow")?,
    )?;
    let w_columns = w_bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    reader.finish()?;
    Ok(VectorFragment {
        fragment_id,
        x,
        w_columns,
        n_in,
        n_out_total,
        j_start,
        n_cols,
        output_slot,
    })
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
            assert!(
                sf.fits_l1i(),
                "sub {} excede L1i: {} bytes",
                sf.sub_id,
                sf.bytecode_size()
            );
        }

        // Las dependencias deben formar una cadena
        for i in 1..subs.len() {
            assert_eq!(subs[i].depends_on, vec![(i as u32) - 1]);
        }

        println!(
            "50K ops → {} sub-fragmentos, tamaños: {:?}",
            subs.len(),
            subs.iter().map(|sf| sf.bytecode_size()).collect::<Vec<_>>()
        );
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
        assert_eq!(
            total_in_subs, total,
            "ops se perdieron en sub-fragmentación"
        );
    }
}

#[cfg(test)]
mod vector_fragment_tests {
    use super::*;

    #[test]
    fn vector_fragment_matmul() {
        // x = [1, 2, 3], W = [[1, 0], [0, 1], [1, 1]]
        // y = x · W = [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        let x = vec![1.0, 2.0, 3.0];
        let weights: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2] row-major
        let vf = VectorFragment::new(0, &x, &weights, 0, 3, 2, 0, 2, 0);
        let y = vf.execute();
        assert_eq!(y.len(), 2);
        assert!((y[0] - 4.0).abs() < 1e-9, "y[0] = {}", y[0]);
        assert!((y[1] - 5.0).abs() < 1e-9, "y[1] = {}", y[1]);
    }

    #[test]
    fn split_matmul_columns_parallel() {
        // x = [1, 2], W = [[1, 2, 3, 4]] (1×2 × 2×4)
        // y = x · W = [1*1+2*1, 1*2+2*2, 1*3+2*3, 1*4+2*4] = [3, 6, 9, 12]
        // W row-major [n_in=2, n_out=4]: [1, 2, 3, 4, 1, 2, 3, 4]
        let x = vec![1.0, 2.0];
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        // n_in=2, n_out=4, split en 2 fragments de 2 columnas cada uno
        let fragments = split_matmul_columns(&x, &weights, 0, 2, 4, 2, 0);

        assert_eq!(fragments.len(), 2);
        // Fragment 0: columnas 0-1 → y[0]=3, y[1]=6
        // Fragment 1: columnas 2-3 → y[2]=9, y[3]=12
        let y0 = fragments[0].execute();
        let y1 = fragments[1].execute();
        assert_eq!(y0.len(), 2);
        assert_eq!(y1.len(), 2);
        assert!((y0[0] - 3.0).abs() < 1e-9, "y0[0] = {}", y0[0]);
        assert!((y0[1] - 6.0).abs() < 1e-9, "y0[1] = {}", y0[1]);
        assert!((y1[0] - 9.0).abs() < 1e-9, "y1[0] = {}", y1[0]);
        assert!((y1[1] - 12.0).abs() < 1e-9, "y1[1] = {}", y1[1]);
    }

    #[test]
    fn vector_fragment_serialize_roundtrip() {
        let x = vec![1.0, 2.0, 3.0];
        let weights: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let vf = VectorFragment::new(42, &x, &weights, 0, 3, 2, 0, 2, 5);

        let bytes = serialize_vector_fragment(&vf);
        let vf2 = deserialize_vector_fragment(&bytes).unwrap();

        assert_eq!(vf2.fragment_id, 42);
        assert_eq!(vf2.n_in, 3);
        assert_eq!(vf2.j_start, 0);
        assert_eq!(vf2.n_cols, 2);
        assert_eq!(vf2.output_slot, 5);
        assert_eq!(vf2.x.len(), 3);
        assert_eq!(vf2.w_columns.len(), 6);

        let y = vf2.execute();
        assert!((y[0] - 4.0).abs() < 1e-9);
        assert!((y[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn split_matmul_large() {
        // Matmul 4×8 split en 4 fragments
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weights: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let fragments = split_matmul_columns(&x, &weights, 0, 4, 8, 4, 0);

        assert_eq!(fragments.len(), 4);
        for frag in &fragments {
            assert_eq!(frag.n_cols, 2);
            assert_eq!(frag.n_in, 4);
        }

        // Verificar que la unión de resultados = matmul completo
        let mut y_full = Vec::new();
        for frag in &fragments {
            y_full.extend(frag.execute());
        }
        assert_eq!(y_full.len(), 8);
    }
}
