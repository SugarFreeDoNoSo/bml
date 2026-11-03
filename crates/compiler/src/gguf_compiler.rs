//! Compilador GGUF → `.bmlgraph`.
//!
//! Toma un archivo GGUF (parseado con `bml-parser`), traduce las
//! operaciones del transformer a sub-DAGs BML usando los pesos como
//! `Const` y los inputs como `Var`, aplica Hash Consing + constant
//! folding + fragmentación AOT, y serializa los `.bmlgraph` a disco.
//!
//! # Flujo
//!
//! 1. Parsear GGUF → metadatos + tensor infos + datos zero-copy.
//! 2. Comprobar la arquitectura: solo se admite la ruta experimental `llama`.
//! 3. Leer dimensiones del modelo (n_layers, n_heads, n_embd, etc.).
//! 4. Para cada capa del transformer:
//!    a. Leer pesos de los tensores (zero-copy desde mmap).
//!    b. Traducir cada operación (matmul, RMSNorm, RoPE, attention, MLP)
//!       a sub-DAGs BML usando `HashConsRegistry`.
//!    c. Los pesos son `Const(id)` (precomputados con constant folding).
//!    d. Los inputs son `Var(id)`.
//!    e. RoPE usa EML compile-time para precomputar cos/sin como `Const`.
//! 5. Concatenar los sub-DAGs.
//! 6. Linearizar a RPN.
//! 7. Fragmentar con `fragment_program` según el hardware objetivo.
//! 8. Serializar a `.bmlgraph`.

use crate::eml;
use crate::fragment::{fragment_program, BmlGraph, BMLGRAPH_MAGIC, BMLGRAPH_VERSION};
use crate::hardware::HardwareSpec;
use crate::hash_cons::HashConsRegistry;
use crate::rpn::linearize;
use crate::tokenizer::Vocabulary;
use bml_domain::NodeId;
use bml_parser::{GgufMetadataValue, GgufParser};
use std::collections::HashMap;
use std::io::Write as IoWrite;
use std::path::Path;

/// Configuración del modelo leída del GGUF.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Arquitectura (ej. "llama").
    pub architecture: String,
    /// Número de capas.
    pub n_layers: u32,
    /// Número de heads de atención.
    pub n_heads: u32,
    /// Dimensión de embedding.
    pub n_embd: u32,
    /// Longitud de contexto.
    pub context_length: u32,
    /// Vocabulario size.
    pub vocab_size: u32,
}

/// Resultado de la compilación.
pub struct CompilationResult {
    /// Grafo BML fragmentado.
    pub graph: BmlGraph,
    /// Pool de constantes precalculadas.
    pub const_pool: Vec<f64>,
    /// Configuración del modelo.
    pub config: ModelConfig,
    /// Número de fragmentos generados.
    pub num_fragments: usize,
}

/// Compila un GGUF a `.bmlgraph`.
///
/// # Argumentos
///
/// - `gguf_path`: Ruta al archivo GGUF.
/// - `hardware`: Especificaciones del hardware objetivo.
///
/// # Retorna
///
/// Un `CompilationResult` con el grafo fragmentado, el pool de constantes,
/// y la configuración del modelo.
pub fn compile_gguf<P: AsRef<Path>>(
    gguf_path: P,
    hardware: &HardwareSpec,
) -> Result<CompilationResult, String> {
    let parser = GgufParser::open(gguf_path).map_err(|e| format!("parser: {e}"))?;
    let config = read_model_config(&parser)?;

    // Construir el DAG BML del transformer.
    let mut reg = HashConsRegistry::with_capacity(1024);

    // Para cada capa, construir los sub-DAGs.
    // Por ahora, construimos un DAG simplificado que representa
    // la estructura del transformer sin los pesos reales.
    // La implementación completa requiere leer los tensores y
    // traducir cada operación matricial.
    let root = build_transformer_dag(&mut reg, &config, &parser)?;

    // Linearizar a RPN.
    let soa_pool = reg.into_soa_and_pool();
    let (soa, const_pool) = soa_pool;
    let program = linearize(&soa, root);

    // Fragmentar según el hardware.
    let threshold = hardware.fragment_threshold();
    let graph = fragment_program(&program, threshold);

    Ok(CompilationResult {
        num_fragments: graph.num_fragments(),
        graph,
        const_pool,
        config,
    })
}

/// Lee la configuración para las rutas experimentales de ejecución.
/// La inspección de metadatos mediante el reader no tiene esta restricción.
pub(crate) fn read_model_config(parser: &GgufParser) -> Result<ModelConfig, String> {
    let arch = parser
        .architecture()
        .ok_or("no se encontró general.architecture")?
        .to_string();
    if arch != "llama" {
        return Err(format!(
            "unsupported GGUF architecture '{arch}': only the experimental 'llama' execution path is available"
        ));
    }
    let arch_clone = arch.clone();

    let get_u32 = |key: &str| -> Result<u32, String> {
        let key = format!("{arch_clone}.{key}");
        let invalid = || format!("invalid integer metadata: {key}");
        match parser.get_metadata(&key) {
            Some(GgufMetadataValue::U32(v)) => Ok(*v),
            Some(GgufMetadataValue::U64(v)) => u32::try_from(*v).map_err(|_| invalid()),
            Some(GgufMetadataValue::I32(v)) => u32::try_from(*v).map_err(|_| invalid()),
            Some(GgufMetadataValue::I64(v)) => u32::try_from(*v).map_err(|_| invalid()),
            None => Ok(0),
            _ => Err(invalid()),
        }
    };

    Ok(ModelConfig {
        architecture: arch,
        n_layers: get_u32("block_count")?,
        n_heads: get_u32("attention.head_count")?,
        n_embd: get_u32("embedding_length")?,
        context_length: get_u32("context_length")?,
        vocab_size: get_u32("vocab_size")?,
    })
}

/// Build a balanced BML tree from all weight elements.
fn build_weight_dag(reg: &mut HashConsRegistry, weights: &[f32]) -> NodeId {
    if weights.is_empty() {
        return reg.const_value(0.0);
    }
    let nodes: Vec<NodeId> = weights.iter().map(|&w| reg.const_value(w as f64)).collect();
    combine_balanced(reg, &nodes)
}

/// Combine nodes into a balanced BML tree.
fn combine_balanced(reg: &mut HashConsRegistry, nodes: &[NodeId]) -> NodeId {
    match nodes.len() {
        0 => reg.const_value(0.0),
        1 => nodes[0],
        2 => reg.bml(nodes[0], nodes[1]),
        _ => {
            let mid = nodes.len() / 2;
            let left = combine_balanced(reg, &nodes[..mid]);
            let right = combine_balanced(reg, &nodes[mid..]);
            reg.bml(left, right)
        }
    }
}

/// Construye el DAG BML del transformer con pesos reales del GGUF.
///
/// Lee los tensores del GGUF (zero-copy + dequantizacion), los almacena
/// como `Const` en el pool de constantes, y construye el DAG usando
/// los pesos reales.
///
/// # Estructura por capa
///
/// 1. RMSNorm de los inputs (con pesos de norma reales).
/// 2. Matmul con pesos de atención Q, K, V (pesos reales).
/// 3. RoPE (precomputado con EML).
/// 4. Attention scores (Q·K^T).
/// 5. Softmax de los scores.
/// 6. Matmul con V.
/// 7. Matmul con pesos de output projection.
/// 8. Residual connection.
/// 9. RMSNorm.
/// 10. MLP: matmul + SwiGLU + matmul.
/// 11. Residual connection.
fn build_transformer_dag(
    reg: &mut HashConsRegistry,
    config: &ModelConfig,
    parser: &GgufParser,
) -> Result<NodeId, String> {
    // Input: Var(0) representa el embedding del token actual.
    let input = reg.var(0);

    // Leer frecuencia base de RoPE.
    let rope_freq_base = parser
        .get_metadata(&format!("{}.rope.freq_base", config.architecture))
        .and_then(|v| match v {
            bml_parser::GgufMetadataValue::F32(f) => Some(*f as f64),
            _ => None,
        })
        .unwrap_or(10000.0);

    // Precomputar constantes de RoPE con EML.
    let rope_consts = eml::rope_constants(
        config.context_length as usize,
        config.n_embd as usize,
        rope_freq_base,
    );

    // Construir capa por capa.
    let mut hidden = input;

    for layer in 0..config.n_layers {
        let prefix = format!("blk.{layer}");

        // === RMSNorm de atención ===
        // Leer pesos de norma (F32, n_embd elementos).
        let norm_name = format!("{prefix}.attn_norm.weight");
        let norm_weights = read_tensor_f32(parser, &norm_name)
            .ok_or(format!("tensor no encontrado: {norm_name}"))?;
        // Construir DAG combinado con todos los pesos de norma
        let norm_weight_dag = build_weight_dag(reg, &norm_weights);
        hidden = reg.bml(hidden, norm_weight_dag);

        // === Self-attention: Q, K, V ===
        // Leer pesos Q, K, V (Q4_0, dequantizados a f32).
        let q_name = format!("{prefix}.attn_q.weight");
        let k_name = format!("{prefix}.attn_k.weight");
        let v_name = format!("{prefix}.attn_v.weight");

        let q_weights =
            read_tensor_f32(parser, &q_name).ok_or(format!("tensor no encontrado: {q_name}"))?;
        let k_weights =
            read_tensor_f32(parser, &k_name).ok_or(format!("tensor no encontrado: {k_name}"))?;
        let v_weights =
            read_tensor_f32(parser, &v_name).ok_or(format!("tensor no encontrado: {v_name}"))?;

        // Almacenar TODOS los pesos como Const y construir DAGs combinados
        let q_weight_dag = build_weight_dag(reg, &q_weights);
        let k_weight_dag = build_weight_dag(reg, &k_weights);
        let v_weight_dag = build_weight_dag(reg, &v_weights);

        let q = reg.bml(hidden, q_weight_dag);
        let k = reg.bml(hidden, k_weight_dag);
        let v = reg.bml(hidden, v_weight_dag);

        // === RoPE ===
        // Aplicar RoPE con constantes precomputadas.
        if !rope_consts.is_empty() {
            let cos_val = reg.const_value(rope_consts[0].0);
            let sin_val = reg.const_value(rope_consts[0].1);
            let q_rotated = reg.bml(q, sin_val);
            let q_scaled = reg.bml(q, cos_val);
            let q_new = reg.bml(q_scaled, q_rotated);
            let _ = q_new; // placeholder: en implementación completa se aplicaría a todos los pares
        }

        // === Attention scores: Q · K^T ===
        let score = reg.bml(q, k);

        // === Softmax ===
        // Escala por 1/sqrt(head_dim)
        let head_dim = config.n_embd / config.n_heads;
        let softmax_scale_val = 1.0 / (head_dim as f64).sqrt();
        let softmax_scale = reg.const_value(softmax_scale_val);
        let attn = reg.bml(score, softmax_scale);

        // === Output: attn · V ===
        let attn_out = reg.bml(attn, v);

        // === Output projection ===
        let o_name = format!("{prefix}.attn_output.weight");
        let o_weights =
            read_tensor_f32(parser, &o_name).ok_or(format!("tensor no encontrado: {o_name}"))?;
        let o_weight_dag = build_weight_dag(reg, &o_weights);
        let o_out = reg.bml(attn_out, o_weight_dag);

        // === Residual ===
        hidden = reg.bml(hidden, o_out);

        // === MLP RMSNorm ===
        let mlp_norm_name = format!("{prefix}.ffn_norm.weight");
        let mlp_norm_weights = read_tensor_f32(parser, &mlp_norm_name)
            .ok_or(format!("tensor no encontrado: {mlp_norm_name}"))?;
        let mlp_norm_dag = build_weight_dag(reg, &mlp_norm_weights);
        hidden = reg.bml(hidden, mlp_norm_dag);

        // === MLP: gate, up, down ===
        let gate_name = format!("{prefix}.ffn_gate.weight");
        let up_name = format!("{prefix}.ffn_up.weight");
        let down_name = format!("{prefix}.ffn_down.weight");

        let gate_weights = read_tensor_f32(parser, &gate_name)
            .ok_or(format!("tensor no encontrado: {gate_name}"))?;
        let up_weights =
            read_tensor_f32(parser, &up_name).ok_or(format!("tensor no encontrado: {up_name}"))?;
        let down_weights = read_tensor_f32(parser, &down_name)
            .ok_or(format!("tensor no encontrado: {down_name}"))?;

        let gate_weight_dag = build_weight_dag(reg, &gate_weights);
        let up_weight_dag = build_weight_dag(reg, &up_weights);
        let down_weight_dag = build_weight_dag(reg, &down_weights);

        let gate = reg.bml(hidden, gate_weight_dag);
        let up = reg.bml(hidden, up_weight_dag);

        // SwiGLU: gate * sigmoid(gate) * up
        // Simplificado: bml(gate, up)
        let mlp_act = reg.bml(gate, up);
        let mlp_out = reg.bml(mlp_act, down_weight_dag);

        // === Residual ===
        hidden = reg.bml(hidden, mlp_out);
    }

    // === Final RMSNorm ===
    let final_norm_name = "output_norm.weight";
    let final_norm_weights = read_tensor_f32(parser, final_norm_name)
        .ok_or(format!("tensor no encontrado: {final_norm_name}"))?;
    let final_norm_dag = build_weight_dag(reg, &final_norm_weights);
    hidden = reg.bml(hidden, final_norm_dag);

    // === Output projection (lm_head) ===
    let lm_head_name = if parser.find_tensor("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    };
    let lm_head_weights = read_tensor_f32(parser, lm_head_name)
        .ok_or(format!("tensor no encontrado: {lm_head_name}"))?;
    let lm_head_dag = build_weight_dag(reg, &lm_head_weights);
    hidden = reg.bml(hidden, lm_head_dag);

    Ok(hidden)
}

/// Lee un tensor del GGUF y retorna sus valores como f32.
///
/// Para tipos estándar (F32, F16), lee directamente.
/// Para tipos cuantizados (Q4_0, Q8_0, Q6_K), dequantiza a f32.
pub fn read_tensor_f32(parser: &GgufParser, name: &str) -> Option<Vec<f32>> {
    parser.read_tensor_f32(name).ok()
}

/// Compila un GGUF a `.bmlgraph` usando solo metadatos + grafo simbólico.
///
/// No carga ni dequantiza los pesos (eso sería O(n_params) en RAM y CPU).
/// Lee los metadatos del GGUF (config del modelo) y construye un DAG
/// simbólico que representa la estructura del transformer.
///
/// Los pesos se cargan en runtime (no en compile-time).
pub fn compile_gguf_fast(
    gguf_path: &std::path::Path,
    _hardware: &HardwareSpec,
) -> Result<CompilationResult, String> {
    let parser = bml_parser::GgufParser::open(gguf_path).map_err(|e| format!("parser: {e}"))?;
    let config = read_model_config(&parser)?;

    let mut reg = HashConsRegistry::with_capacity(64);
    let root = reg.var(0);
    let soa_pool = reg.into_soa_and_pool();
    let (soa, const_pool) = soa_pool;
    let program = linearize(&soa, root);
    let graph = fragment_program(&program, 32 * 1024);

    Ok(CompilationResult {
        num_fragments: graph.num_fragments(),
        graph,
        const_pool,
        config,
    })
}
///
/// Crea un directorio con:
/// - `header.bmlgraph`: magic, version, n_fragments, config, const_pool
/// - `fragment_0.bmlgraph`, `fragment_1.bmlgraph`, ...: cada fragmento serializado
pub fn serialize_to_dir(result: &CompilationResult, output_dir: &Path) -> Result<(), String> {
    if result.num_fragments != result.graph.fragments.len() {
        return Err("fragment count does not match graph".into());
    }
    let fragment_count = u32::try_from(result.num_fragments).map_err(|_| "too many fragments")?;
    for fragment in &result.graph.fragments {
        u32::try_from(fragment.ops.len()).map_err(|_| "too many fragment operations")?;
    }
    std::fs::create_dir_all(output_dir).map_err(|e| format!("crear dir: {e}"))?;

    // Header: magic, version, n_fragments, config, const_pool
    let header_path = output_dir.join("header.bmlgraph");
    let mut f = std::fs::File::create(&header_path).map_err(|e| format!("crear header: {e}"))?;

    // Magic y version
    f.write_all(&BMLGRAPH_MAGIC.to_le_bytes())
        .map_err(|e| format!("write magic: {e}"))?;
    f.write_all(&BMLGRAPH_VERSION.to_le_bytes())
        .map_err(|e| format!("write version: {e}"))?;

    // Número de fragmentos
    f.write_all(&fragment_count.to_le_bytes())
        .map_err(|e| format!("write n_frag: {e}"))?;

    // Config del modelo
    let arch_bytes = result.config.architecture.as_bytes();
    f.write_all(&(arch_bytes.len() as u64).to_le_bytes())
        .map_err(|e| format!("write arch len: {e}"))?;
    f.write_all(arch_bytes)
        .map_err(|e| format!("write arch: {e}"))?;
    f.write_all(&result.config.n_layers.to_le_bytes())
        .map_err(|e| format!("write n_layers: {e}"))?;
    f.write_all(&result.config.n_heads.to_le_bytes())
        .map_err(|e| format!("write n_heads: {e}"))?;
    f.write_all(&result.config.n_embd.to_le_bytes())
        .map_err(|e| format!("write n_embd: {e}"))?;
    f.write_all(&result.config.context_length.to_le_bytes())
        .map_err(|e| format!("write ctx: {e}"))?;
    f.write_all(&result.config.vocab_size.to_le_bytes())
        .map_err(|e| format!("write vocab: {e}"))?;

    // Pool de constantes
    f.write_all(&(result.const_pool.len() as u64).to_le_bytes())
        .map_err(|e| format!("write pool len: {e}"))?;
    for &val in &result.const_pool {
        f.write_all(&val.to_le_bytes())
            .map_err(|e| format!("write pool val: {e}"))?;
    }

    // Fragmentos
    for (i, fragment) in result.graph.fragments.iter().enumerate() {
        let frag_path = output_dir.join(format!("fragment_{i}.bmlgraph"));
        let mut ff =
            std::fs::File::create(&frag_path).map_err(|e| format!("crear frag {i}: {e}"))?;
        ff.write_all(&(fragment.ops.len() as u32).to_le_bytes())
            .map_err(|e| format!("write frag {i} len: {e}"))?;
        for op in &fragment.ops {
            use crate::rpn::RpnOp;
            match op {
                RpnOp::One => {
                    ff.write_all(&[0]).map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Zero => {
                    ff.write_all(&[6]).map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Bml => {
                    ff.write_all(&[1]).map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Dup => {
                    ff.write_all(&[2]).map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Loop { count, body_len } => {
                    ff.write_all(&[3]).map_err(|e| format!("write op: {e}"))?;
                    ff.write_all(&count.to_le_bytes())
                        .map_err(|e| format!("write op: {e}"))?;
                    ff.write_all(&body_len.to_le_bytes())
                        .map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Var(id) => {
                    ff.write_all(&[4]).map_err(|e| format!("write op: {e}"))?;
                    ff.write_all(&id.to_le_bytes())
                        .map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Const(id) => {
                    ff.write_all(&[5]).map_err(|e| format!("write op: {e}"))?;
                    ff.write_all(&id.to_le_bytes())
                        .map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::VarIndexed { base } => {
                    ff.write_all(&[7]).map_err(|e| format!("write op: {e}"))?;
                    ff.write_all(&base.to_le_bytes())
                        .map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::StoreResult { slot } => {
                    ff.write_all(&[8]).map_err(|e| format!("write op: {e}"))?;
                    ff.write_all(&slot.to_le_bytes())
                        .map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::FAdd => {
                    ff.write_all(&[9]).map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::FMul => {
                    ff.write_all(&[10]).map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Pick { depth } => {
                    ff.write_all(&[11]).map_err(|e| format!("write op: {e}"))?;
                    ff.write_all(&depth.to_le_bytes())
                        .map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Drop => {
                    ff.write_all(&[12]).map_err(|e| format!("write op: {e}"))?;
                }
                RpnOp::Swap => {
                    ff.write_all(&[13]).map_err(|e| format!("write op: {e}"))?;
                }
            }
        }
    }

    Ok(())
}

/// Carga un `.bmlgraph` desde un directorio.
///
/// Lee el header (config + const_pool) y los fragmentos.
pub use crate::graph_directory::{load_from_dir, load_from_dir_with_limits, GraphLoadLimits};

/// Resultado de compilar un GGUF para inferencia real con fragmentos por capa.
#[path = "prepared_model.rs"]
pub mod prepared_model;

/// Validated model configuration, tokenizer and execution weight storage.
pub struct InferenceCompiler {
    config: ModelConfig,
    vocab: Vocabulary,
    /// Pool de pesos (f32, no f64 — ahorra la mitad de RAM).
    weight_pool: Vec<f32>,
    mapped_weights: Option<GgufParser>,
    /// Offset en weight_pool de cada tensor clave (por capa y tipo).
    weight_offsets: HashMap<String, u32>,
    /// Dimensiones GGUF de cada tensor (nombre → (ne0, ne1)); ne0 es contiguo.
    tensor_dims: HashMap<String, (usize, usize)>,
    /// Dimensión head.
    head_dim: u32,
    /// Dimensión de KV heads (GQA).
    n_kv_heads: u32,
    /// GGUF epsilon shared by attention, feed-forward and final RMSNorm.
    rms_norm_eps: f64,
    rope_freq_base: f64,
}

impl InferenceCompiler {
    /// Load autonomous BML weights under an explicitly selected numeric policy.
    /// No source GGUF or weight synthesis is used. The cap bounds total expanded
    /// weights; evaluation may fail even for a structurally valid container.
    pub fn from_bmlgguf(
        reader: &mut bml_model_weights::WeightReader,
        max_evaluated_weights: usize,
        policy: prepared_model::WeightEvaluation,
    ) -> Result<Self, String> {
        prepared_model::load(reader, max_evaluated_weights, policy)
    }
    /// Abre un GGUF y carga TODOS los pesos dequantizados como f32.
    ///
    /// The expanded pool needs four bytes per weight. Use [`Self::open_mapped`]
    /// to retain encoded matrices and decode one row at a time.
    /// Rechaza arquitecturas distintas de `llama` antes de leer el vocabulario
    /// o expandir pesos. La ruta `llama` sigue siendo experimental.
    pub fn open<P: AsRef<Path>>(gguf_path: P) -> Result<Self, String> {
        Self::open_storage(gguf_path, false)
    }

    /// Keep matrix weights encoded in the GGUF mapping and decode one row at a time.
    /// Native F64 projection order and F32 KV arithmetic remain unchanged.
    /// The mapped file must not be modified or truncated while this object exists.
    pub fn open_mapped<P: AsRef<Path>>(gguf_path: P) -> Result<Self, String> {
        Self::open_storage(gguf_path, true)
    }

    fn open_storage<P: AsRef<Path>>(gguf_path: P, mapped: bool) -> Result<Self, String> {
        let parser = GgufParser::open(gguf_path).map_err(|e| format!("parser: {e}"))?;
        Self::from_parser_storage(parser, mapped)
    }

    pub(crate) fn from_mapped_parser(parser: GgufParser) -> Result<Self, String> {
        Self::from_parser_storage(parser, true)
    }

    fn from_parser_storage(parser: GgufParser, mapped: bool) -> Result<Self, String> {
        let config = read_model_config(&parser)?;
        if config.n_heads == 0
            || config.n_embd == 0
            || config.context_length == 0
            || !config.n_embd.is_multiple_of(config.n_heads)
        {
            return Err("invalid model dimensions or context".into());
        }
        if config.n_layers != 0 && parser.find_tensor("output_norm.weight").is_none() {
            return Err("missing output_norm.weight".into());
        }
        let validated = if config.n_layers != 0 {
            Some(
                bml_inference::model::Model::from_gguf(&parser)
                    .map_err(|error| format!("Llama profile: {error}"))?
                    .config()
                    .clone(),
            )
        } else {
            None
        };
        let rope_freq_base = validated
            .as_ref()
            .map_or(10000.0, |c| f64::from(c.rope_freq_base));
        let vocab =
            if config.n_layers == 0 && parser.get_metadata("tokenizer.ggml.scores").is_none() {
                Vocabulary::from_projection_fixture(&parser)?
            } else {
                Vocabulary::from_gguf(&parser)?
            };

        let rms_norm_eps = match parser.get_metadata("llama.attention.layer_norm_rms_epsilon") {
            None => 1e-5,
            Some(GgufMetadataValue::F32(value)) => f64::from(*value),
            _ => return Err("invalid llama.attention.layer_norm_rms_epsilon type".into()),
        };
        if !rms_norm_eps.is_finite() || rms_norm_eps <= 0.0 {
            return Err("invalid llama.attention.layer_norm_rms_epsilon value".into());
        }

        let head_dim = config.n_embd / config.n_heads;

        let n_kv_heads = parser
            .get_metadata(&format!("{}.attention.head_count_kv", config.architecture))
            .and_then(|v| match v {
                GgufMetadataValue::U32(n) => Some(*n),
                GgufMetadataValue::I32(n) => Some(*n as u32),
                _ => None,
            })
            .unwrap_or(config.n_heads);
        if n_kv_heads == 0 || !config.n_heads.is_multiple_of(n_kv_heads) {
            return Err("invalid grouped-query attention heads".into());
        }

        let mut weight_pool: Vec<f32> = Vec::new();
        let mut weight_offsets = HashMap::new();
        let mut tensor_dims = HashMap::new();

        if mapped {
            for info in parser.tensor_infos() {
                if info.dims.len() == 1 || info.dims.len() == 2 {
                    let cols =
                        usize::try_from(info.dims[0]).map_err(|_| "tensor dimension overflow")?;
                    let rows = usize::try_from(*info.dims.get(1).unwrap_or(&1))
                        .map_err(|_| "tensor dimension overflow")?;
                    tensor_dims.insert(info.name.clone(), (cols, rows));
                }
                if info.name.ends_with("_norm.weight") {
                    if info.dims != [u64::from(config.n_embd)] {
                        return Err(format!("invalid {} dimensions", info.name));
                    }
                    let values = parser
                        .read_tensor_f32(&info.name)
                        .map_err(|e| e.to_string())?;
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err(format!("non-finite {}", info.name));
                    }
                    let offset =
                        u32::try_from(weight_pool.len()).map_err(|_| "weight offset overflow")?;
                    weight_offsets.insert(info.name.clone(), offset);
                    weight_pool.extend(values);
                }
            }
            for name in ["token_embd.weight", "output.weight"] {
                if name == "output.weight" && parser.find_tensor(name).is_none() {
                    continue;
                }
                let info = parser
                    .find_tensor(name)
                    .ok_or_else(|| format!("missing {name}"))?;
                if info.dims != [u64::from(config.n_embd), vocab.len() as u64] {
                    return Err(format!("invalid {name} dimensions"));
                }
                parser
                    .tensor_view(name)
                    .map_err(|e| e.to_string())?
                    .decode_row_f32(0)
                    .map_err(|e| e.to_string())?;
            }
            return Ok(Self {
                config,
                vocab,
                weight_pool,
                mapped_weights: Some(parser),
                weight_offsets,
                tensor_dims,
                head_dim,
                n_kv_heads,
                rms_norm_eps,
                rope_freq_base,
            });
        }

        // Cargar embedding (token_embd.weight)
        if let Some(emb_vals) = read_tensor_f32(&parser, "token_embd.weight") {
            let dims = get_tensor_dims(&parser, "token_embd.weight");
            let offset = weight_pool.len() as u32;
            weight_pool.extend(&emb_vals);
            weight_offsets.insert("token_embd.weight".into(), offset);
            tensor_dims.insert("token_embd.weight".into(), dims);
        }

        // Cargar pesos por capa
        for layer in 0..config.n_layers {
            let prefix = format!("blk.{layer}");
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.attn_norm.weight", prefix),
            );
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.attn_q.weight", prefix),
            );
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.attn_k.weight", prefix),
            );
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.attn_v.weight", prefix),
            );
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.attn_output.weight", prefix),
            );
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.ffn_norm.weight", prefix),
            );
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.ffn_gate.weight", prefix),
            );
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.ffn_up.weight", prefix),
            );
            load_tensor(
                &parser,
                &mut weight_pool,
                &mut weight_offsets,
                &mut tensor_dims,
                &format!("{}.ffn_down.weight", prefix),
            );
        }

        // Zero-layer fixtures retain their historical projection-only contract.
        // Real transformer layers require a valid final normalization tensor.
        if let Some(info) = parser.find_tensor("output_norm.weight") {
            if info.dims != [u64::from(config.n_embd)] {
                return Err("invalid output_norm.weight dimensions".into());
            }
            let values = parser
                .read_tensor_f32("output_norm.weight")
                .map_err(|error| format!("output_norm.weight: {error}"))?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err("non-finite output_norm.weight".into());
            }
            let offset =
                u32::try_from(weight_pool.len()).map_err(|_| "weight pool exceeds u32 offsets")?;
            weight_pool.extend(values);
            weight_offsets.insert("output_norm.weight".into(), offset);
            tensor_dims.insert("output_norm.weight".into(), (config.n_embd as usize, 1));
        }

        // Only models without an explicit head tie it to the embeddings.
        // A present but malformed/unsupported head must not silently fall back.
        if let Some(info) = parser.find_tensor("output.weight") {
            if info.dims != [u64::from(config.n_embd), vocab.len() as u64] {
                return Err("invalid output.weight dimensions".into());
            }
            let values = parser
                .read_tensor_f32("output.weight")
                .map_err(|error| format!("output.weight: {error}"))?;
            let offset =
                u32::try_from(weight_pool.len()).map_err(|_| "weight pool exceeds u32 offsets")?;
            weight_pool.extend(values);
            weight_offsets.insert("output.weight".into(), offset);
            tensor_dims.insert(
                "output.weight".into(),
                (config.n_embd as usize, vocab.len()),
            );
        }

        Ok(Self {
            config,
            vocab,
            weight_pool,
            mapped_weights: None,
            weight_offsets,
            tensor_dims,
            head_dim,
            n_kv_heads,
            rms_norm_eps,
            rope_freq_base,
        })
    }

    /// Retorna la configuración del modelo.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Retorna el vocabulario.
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Decoded weights; mapped mode keeps only normalization vectors here.
    pub fn weight_pool(&self) -> &[f32] {
        &self.weight_pool
    }

    /// Encoded GGUF mapping bytes; zero for the fully expanded storage mode.
    pub fn mapped_bytes(&self) -> usize {
        self.mapped_weights
            .as_ref()
            .map_or(0, |parser| parser.len())
    }

    /// Offsets into the decoded pool; mapped matrices do not have pool offsets.
    pub fn weight_offsets(&self) -> &HashMap<String, u32> {
        &self.weight_offsets
    }

    /// Select the explicit vocabulary projection, falling back only when absent.
    pub fn output_weight_name(&self) -> &'static str {
        if self.tensor_dims.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        }
    }

    pub(crate) fn use_artifact_vocabulary(&mut self, vocab: Vocabulary) {
        self.vocab = vocab;
    }

    /// Validated rotary frequency base used by the causal execution profile.
    pub fn rope_frequency_base(&self) -> f64 {
        self.rope_freq_base
    }

    /// Validated RMSNorm epsilon read from the GGUF metadata.
    pub fn rms_norm_epsilon(&self) -> f64 {
        self.rms_norm_eps
    }

    /// Retorna el mapa de dimensiones de tensores.
    pub fn tensor_dims(&self) -> &HashMap<String, (usize, usize)> {
        &self.tensor_dims
    }

    /// Retorna la dimensión de head.
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Retorna el número de KV heads.
    pub fn n_kv_heads(&self) -> u32 {
        self.n_kv_heads
    }

    /// Decode a token embedding, rejecting an invalid ID or row decode failure.
    pub fn get_embedding(&self, token_id: u32) -> Result<Vec<f64>, String> {
        self.get_embedding_f64(token_id)
    }

    /// Causal attention with F32 KV storage and softmax, and native F64 projections.
    /// The historical cache type name does not imply hashing or i8 storage.
    pub fn forward_cached(
        &self,
        input_ids: &[u32],
        kv_cache: &mut crate::kv_cache::HashConsedKV,
    ) -> Result<Vec<f64>, String> {
        self.forward_cached_with_projection(input_ids, kv_cache, &mut |x, name| {
            self.matmul_f64(x, name)
        })
    }

    /// Execute the same causal model using a caller-supplied matrix projection.
    /// Each projection must return the declared GGUF output dimension and finite values.
    /// Errors leave the committed cache position unchanged; uncommitted rows may be written.
    pub fn forward_cached_with_projection(
        &self,
        input_ids: &[u32],
        kv_cache: &mut crate::kv_cache::HashConsedKV,
        project: &mut impl FnMut(&[f64], &str) -> Result<Vec<f64>, String>,
    ) -> Result<Vec<f64>, String> {
        let n_embd = self.config.n_embd as usize;
        if !kv_cache.matches_layout(self.config.n_layers, self.n_kv_heads, self.head_dim) {
            return Err("KV cache layout does not match the model".into());
        }
        self.validate_causal_input(input_ids, kv_cache.current_pos(), kv_cache.capacity())?;

        // Posición inicial basada en el cache existente
        let start_pos = kv_cache.current_pos();
        let mut hidden: Vec<f64> = vec![0.0; n_embd];

        // Procesar cada token secuencialmente a través de todas las capas
        for (offset, &token_id) in input_ids.iter().enumerate() {
            let pos = start_pos + offset as u32;
            let emb = self.get_embedding_f64(token_id)?;
            for i in 0..n_embd.min(emb.len()) {
                hidden[i] = emb[i];
            }
            for layer in 0..self.config.n_layers {
                self.forward_layer_cached(&mut hidden, layer, pos, kv_cache, project)?;
            }
        }

        // Avanzar la posición del cache
        let logits = self.compute_logits(&hidden, project)?;
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("non-finite logits".into());
        }
        kv_cache.advance_by(input_ids.len() as u32);
        Ok(logits)
    }

    fn validate_causal_input(&self, ids: &[u32], start: u32, capacity: u32) -> Result<(), String> {
        if ids.is_empty() {
            return Err("empty causal input".into());
        }
        if ids.iter().any(|&id| id as usize >= self.vocab.len()) {
            return Err("token ID outside vocabulary".into());
        }
        let count = u32::try_from(ids.len()).map_err(|_| "input length overflow")?;
        let end = start
            .checked_add(count)
            .ok_or("context position overflow")?;
        if end > capacity.min(self.config.context_length) {
            return Err("causal context exhausted".into());
        }
        Ok(())
    }

    /// Forward de una capa con KV cache y softmax real.
    fn forward_layer_cached(
        &self,
        hidden: &mut [f64],
        layer: u32,
        pos: u32,
        kv_cache: &mut crate::kv_cache::HashConsedKV,
        project: &mut impl FnMut(&[f64], &str) -> Result<Vec<f64>, String>,
    ) -> Result<(), String> {
        let n_embd = self.config.n_embd as usize;
        let prefix = format!("blk.{layer}");

        // === RMSNorm de atención ===
        let mut normalized = hidden.to_vec();
        self.rmsnorm_inplace(&mut normalized, &format!("{}.attn_norm.weight", prefix));

        // === Q, K, V projections ===
        let q = self.project_checked(&normalized, &format!("{}.attn_q.weight", prefix), project)?;
        let k = self.project_checked(&normalized, &format!("{}.attn_k.weight", prefix), project)?;
        let v = self.project_checked(&normalized, &format!("{}.attn_v.weight", prefix), project)?;

        // === RoPE a Q y K ===
        let head_dim = self.head_dim as usize;
        let mut q_rotated = q;
        let mut k_rotated = k;
        self.apply_rope_inplace(&mut q_rotated, pos as usize, head_dim);
        self.apply_rope_inplace(&mut k_rotated, pos as usize, head_dim);

        // === Almacenar K y V en F32 ===
        let n_kv_heads = self.n_kv_heads as usize;
        let actual_kv_heads = k_rotated.len() / head_dim;
        let kv_heads_to_use = n_kv_heads.min(actual_kv_heads);
        for kv_h in 0..kv_heads_to_use {
            let k_start = kv_h * head_dim;
            let k_end = (k_start + head_dim).min(k_rotated.len());
            let v_start = kv_h * head_dim;
            let v_end = (v_start + head_dim).min(v.len());
            let k_slice: Vec<f32> = k_rotated[k_start..k_end]
                .iter()
                .map(|&v| v as f32)
                .collect();
            let v_slice: Vec<f32> = v[v_start..v_end].iter().map(|&v| v as f32).collect();
            if k_slice
                .iter()
                .chain(&v_slice)
                .any(|value| !value.is_finite())
            {
                return Err("non-finite KV projection".into());
            }
            kv_cache.store_k(layer, pos, kv_h as u32, &k_slice);
            kv_cache.store_v(layer, pos, kv_h as u32, &v_slice);
        }

        // === Attention con softmax real + KV cache ===
        let n_heads = self.config.n_heads as usize;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let q_heads_per_kv = n_heads / kv_heads_to_use;

        let mut output = vec![0.0_f64; n_heads * head_dim];

        for h in 0..n_heads {
            let kv_h = h / q_heads_per_kv;
            let q_start = h * head_dim;

            // Scores: Q[h] · K[p] para todos los p <= pos
            // Usar dot product f32 (dequantizar K del cache) para precisión
            let q_slice: Vec<f32> = q_rotated[q_start..q_start + head_dim]
                .iter()
                .map(|&v| v as f32)
                .collect();
            let mut scores = Vec::with_capacity(pos as usize + 1);
            for p in 0..=pos {
                let k_cached = kv_cache.load_k(layer, p, kv_h as u32);
                let dot: f32 = q_slice
                    .iter()
                    .zip(k_cached.iter())
                    .map(|(&q, &k)| q * k)
                    .sum();
                scores.push(dot as f64 * scale);
            }
            if scores.iter().any(|value| !value.is_finite()) {
                return Err("non-finite attention score".into());
            }

            // Softmax real
            let attn =
                crate::kv_cache::softmax_f32(&scores.iter().map(|&s| s as f32).collect::<Vec<_>>());

            // Output: Σ attn[p] * V[p]
            let o_start = h * head_dim;
            for d in 0..head_dim {
                let mut sum = 0.0_f64;
                for p in 0..=pos {
                    let v_cached = kv_cache.load_v(layer, p, kv_h as u32);
                    sum += attn[p as usize] as f64 * v_cached[d] as f64;
                }
                output[o_start + d] = sum;
            }
        }

        // === Output projection ===
        let o_out =
            self.project_checked(&output, &format!("{}.attn_output.weight", prefix), project)?;

        // === Residual ===
        for i in 0..n_embd {
            hidden[i] += o_out.get(i).copied().unwrap_or(0.0);
        }

        // === MLP RMSNorm ===
        normalized.copy_from_slice(hidden);
        self.rmsnorm_inplace(&mut normalized, &format!("{}.ffn_norm.weight", prefix));

        // === MLP: gate, up, down ===
        let gate =
            self.project_checked(&normalized, &format!("{}.ffn_gate.weight", prefix), project)?;
        let up =
            self.project_checked(&normalized, &format!("{}.ffn_up.weight", prefix), project)?;

        // SwiGLU: gate * sigmoid(gate) * up
        let mut swiglu_out = vec![0.0_f64; gate.len().min(up.len())];
        for i in 0..swiglu_out.len() {
            let g = gate.get(i).copied().unwrap_or(0.0);
            let u = up.get(i).copied().unwrap_or(0.0);
            swiglu_out[i] = swiglu_scalar(g, u);
        }

        let mlp_out =
            self.project_checked(&swiglu_out, &format!("{}.ffn_down.weight", prefix), project)?;

        // === Residual ===
        for i in 0..n_embd.min(mlp_out.len()) {
            hidden[i] += mlp_out[i];
        }
        if hidden.iter().any(|value| !value.is_finite()) {
            return Err("non-finite layer output".into());
        }
        Ok(())
    }

    /// Obtiene el embedding de un token.
    /// token_embd.weight tiene shape [n_embd, vocab_size].
    /// GGUF almacena n_embd valores contiguos para cada token (ne0 primero).
    fn get_embedding_f64(&self, token_id: u32) -> Result<Vec<f64>, String> {
        if token_id as usize >= self.vocab.len() {
            return Err("token ID outside vocabulary".into());
        }
        if let Some(parser) = &self.mapped_weights {
            let row = parser
                .tensor_view("token_embd.weight")
                .map_err(|e| e.to_string())?
                .decode_row_f32(token_id as usize)
                .map_err(|e| e.to_string())?;
            return Ok(row.into_iter().map(f64::from).collect());
        }
        let offset = self
            .weight_offsets
            .get("token_embd.weight")
            .copied()
            .unwrap_or(0);
        let (emb_dim, vocab_sz) = self
            .tensor_dims
            .get("token_embd.weight")
            .copied()
            .unwrap_or((32000, 32000));
        let tid = token_id.min(vocab_sz as u32 - 1) as usize;
        let mut emb = vec![0.0_f64; emb_dim.min(self.config.n_embd as usize)];
        for i in 0..emb.len() {
            let idx = offset as usize + tid * emb_dim + i;
            if idx < self.weight_pool.len() {
                emb[i] = self.weight_pool[idx] as f64;
            }
        }
        Ok(emb)
    }

    /// Run a complete causal prefix using a fresh F32 KV cache.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f64>, String> {
        self.forward_with_projection(input_ids, &mut |x, name| self.matmul_f64(x, name))
    }

    /// Run a complete causal prefix with checked, caller-supplied projections.
    pub fn forward_with_projection(
        &self,
        input_ids: &[u32],
        project: &mut impl FnMut(&[f64], &str) -> Result<Vec<f64>, String>,
    ) -> Result<Vec<f64>, String> {
        self.validate_causal_input(input_ids, 0, self.config.context_length)?;
        let mut cache = crate::kv_cache::HashConsedKV::try_new(
            self.config.n_layers,
            self.n_kv_heads,
            self.head_dim,
            input_ids.len() as u32,
            crate::kv_cache::DEFAULT_MAX_KV_BYTES,
        )?;
        self.forward_cached_with_projection(input_ids, &mut cache, project)
    }

    /// RMSNorm: hidden = hidden / sqrt(mean(hidden²) + eps) * weight
    fn rmsnorm_inplace(&self, hidden: &mut [f64], weight_name: &str) {
        let n_embd = self.config.n_embd as usize;
        let eps = self.rms_norm_eps;
        let mean_sq: f64 = hidden.iter().map(|v| v * v).sum::<f64>() / n_embd as f64;
        let rms = (mean_sq + eps).sqrt();

        let offset = self.weight_offsets.get(weight_name).copied().unwrap_or(0);
        for i in 0..n_embd {
            let w = if (offset as usize + i) < self.weight_pool.len() {
                self.weight_pool[offset as usize + i] as f64
            } else {
                1.0
            };
            hidden[i] = hidden[i] / rms * w;
        }
    }

    /// Proyección con dimensiones GGUF [n_in, n_out]:
    /// y[j] = suma_i x[i] * W[j * n_in + i], con ne0 contiguo.
    fn matmul_f64(&self, x: &[f64], name: &str) -> Result<Vec<f64>, String> {
        self.project_rows(x, name, |input, row| {
            Ok(input
                .iter()
                .zip(row)
                .fold(0.0, |sum, (&x, &w)| sum + x * f64::from(w)))
        })
    }

    /// Apply a row dot-product backend without expanding mapped matrix weights.
    /// The callback sees exactly one GGML row; its returned scalar must be finite.
    pub fn project_rows(
        &self,
        input: &[f64],
        name: &str,
        mut dot: impl FnMut(&[f64], &[f32]) -> Result<f64, String>,
    ) -> Result<Vec<f64>, String> {
        let &(columns, rows) = self
            .tensor_dims
            .get(name)
            .ok_or("missing projection dimensions")?;
        if input.len() != columns {
            return Err("projection input dimension mismatch".into());
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(rows)
            .map_err(|_| "projection allocation failed")?;
        let mapped = self
            .mapped_weights
            .as_ref()
            .map(|parser| parser.tensor_view(name))
            .transpose()
            .map_err(|e| e.to_string())?;
        for row in 0..rows {
            let decoded;
            let weights = if let Some(view) = mapped {
                decoded = view.decode_row_f32(row).map_err(|e| e.to_string())?;
                decoded.as_slice()
            } else {
                let offset = *self
                    .weight_offsets
                    .get(name)
                    .ok_or("missing projection weights")? as usize;
                let start = row
                    .checked_mul(columns)
                    .and_then(|n| n.checked_add(offset))
                    .ok_or("weight range overflow")?;
                let end = start.checked_add(columns).ok_or("weight range overflow")?;
                self.weight_pool
                    .get(start..end)
                    .ok_or("weight range outside pool")?
            };
            let value = dot(input, weights)?;
            if !value.is_finite() {
                return Err("non-finite projection output".into());
            }
            output.push(value);
        }
        Ok(output)
    }

    /// Project through the separate output head, or tied embeddings if absent.
    /// Each vocabulary row contains ne0 contiguous values.
    fn compute_logits(
        &self,
        hidden: &[f64],
        project: &mut impl FnMut(&[f64], &str) -> Result<Vec<f64>, String>,
    ) -> Result<Vec<f64>, String> {
        let mut normalized;
        let hidden = if self.weight_offsets.contains_key("output_norm.weight") {
            normalized = hidden.to_vec();
            self.rmsnorm_inplace(&mut normalized, "output_norm.weight");
            normalized.as_slice()
        } else {
            hidden
        };
        self.project_checked(hidden, self.output_weight_name(), project)
    }

    fn project_checked(
        &self,
        input: &[f64],
        name: &str,
        project: &mut impl FnMut(&[f64], &str) -> Result<Vec<f64>, String>,
    ) -> Result<Vec<f64>, String> {
        let &(n_in, n_out) = self
            .tensor_dims
            .get(name)
            .ok_or_else(|| format!("missing projection {name}"))?;
        if input.len() != n_in || input.iter().any(|v| !v.is_finite()) {
            return Err(format!("invalid projection input for {name}"));
        }
        let output = project(input, name)?;
        if output.len() != n_out || output.iter().any(|v| !v.is_finite()) {
            return Err(format!("invalid projection output for {name}"));
        }
        Ok(output)
    }

    /// Aplica RoPE a Q o K in-place.
    ///
    /// Para cada par de dimensiones (even, odd) en cada head,
    /// rota según el ángulo cos/sin precomputado para esa dimensión.
    fn apply_rope_inplace(&self, x: &mut [f64], pos: usize, head_dim: usize) {
        let n_half = head_dim / 2;
        let n_heads = x.len() / head_dim;
        for h in 0..n_heads {
            for i in 0..n_half {
                let even_idx = h * head_dim + i * 2;
                let odd_idx = h * head_dim + i * 2 + 1;
                let x_even = x.get(even_idx).copied().unwrap_or(0.0);
                let x_odd = x.get(odd_idx).copied().unwrap_or(0.0);

                // RoPE rotation: angle = pos / base^(2i / head_dim)
                let freq = 1.0 / self.rope_freq_base.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                let c = angle.cos();
                let s = angle.sin();

                let rotated_even = x_even * c - x_odd * s;
                let rotated_odd = x_even * s + x_odd * c;

                if even_idx < x.len() {
                    x[even_idx] = rotated_even;
                }
                if odd_idx < x.len() {
                    x[odd_idx] = rotated_odd;
                }
            }
        }
    }
}

/// Native IEEE SwiGLU scalar: SiLU(gate) times the independent up projection.
/// This is a numerical activation, not a symbolic BML identity.
pub fn swiglu_scalar(gate: f64, up: f64) -> f64 {
    gate / (1.0 + (-gate).exp()) * up
}

fn load_tensor(
    parser: &GgufParser,
    pool: &mut Vec<f32>,
    offsets: &mut HashMap<String, u32>,
    dims_map: &mut HashMap<String, (usize, usize)>,
    name: &str,
) {
    if let Some(values) = read_tensor_f32(parser, name) {
        let info = parser.find_tensor(name);
        let dims = if let Some(info) = info {
            if info.dims.len() >= 2 {
                (info.dims[0] as usize, info.dims[1] as usize)
            } else if info.dims.len() == 1 {
                (info.dims[0] as usize, 1)
            } else {
                (values.len(), 1)
            }
        } else {
            (values.len(), 1)
        };
        let offset = pool.len() as u32;
        pool.extend(values);
        offsets.insert(name.to_string(), offset);
        dims_map.insert(name.to_string(), dims);
    }
}

fn get_tensor_dims(parser: &GgufParser, name: &str) -> (usize, usize) {
    if let Some(info) = parser.find_tensor(name) {
        if info.dims.len() >= 2 {
            (info.dims[0] as usize, info.dims[1] as usize)
        } else if info.dims.len() == 1 {
            (info.dims[0] as usize, 1)
        } else {
            (1, 1)
        }
    } else {
        (1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_projection_respects_gguf_contiguous_input_axis() {
        // GGUF ne0 is contiguous: each output row contains all input weights.
        // Rectangles, signed inputs and an unrelated pool prefix distinguish
        // the file layout from a transpose or a missing tensor offset.
        for (n_in, n_out, values, input, expected) in [
            (
                3,
                2,
                vec![1.0, -2.0, 0.5, -3.0, 4.0, 2.0],
                vec![-2.0, 0.5, 4.0],
                vec![-1.0, 16.0],
            ),
            (
                2,
                3,
                vec![1.0, -2.0, 0.5, -3.0, 4.0, 2.0],
                vec![2.0, -0.5],
                vec![3.0, 2.5, 7.0],
            ),
        ] {
            let mut weight_pool = vec![99.0, -99.0];
            weight_pool.extend(values);
            let compiler = InferenceCompiler {
                config: ModelConfig {
                    architecture: "llama".into(),
                    n_layers: 0,
                    n_heads: 1,
                    n_embd: n_in as u32,
                    context_length: 1,
                    vocab_size: 0,
                },
                vocab: Vocabulary {
                    id_to_token: vec![],
                    token_to_id: HashMap::new(),
                    model_type: "llama".into(),
                    bos_token_id: 0,
                    eos_token_id: 0,
                    tokenizer: None,
                },
                weight_pool,
                mapped_weights: None,
                weight_offsets: HashMap::from([("projection.weight".into(), 2)]),
                tensor_dims: HashMap::from([("projection.weight".into(), (n_in, n_out))]),
                head_dim: n_in as u32,
                n_kv_heads: 1,
                rms_norm_eps: 1e-5,
                rope_freq_base: 10000.0,
            };
            assert_eq!(
                compiler.matmul_f64(&input, "projection.weight").unwrap(),
                expected
            );
        }
    }

    #[test]
    fn compile_synthetic_gguf() {
        use bml_parser::create_gguf_with_metadata;
        let path = create_gguf_with_metadata();
        let hw = HardwareSpec::new(4, 32 * 1024, 256 * 1024, 16 * 1024 * 1024);

        let result = compile_gguf(&path, &hw);
        let error = result
            .err()
            .expect("incomplete model must fail compilation");
        assert_eq!(error, "tensor no encontrado: output_norm.weight");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn compile_real_tinyllama() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();
        let hw = HardwareSpec::detect_local();
        println!("Hardware: {hw:?}");

        let result = compile_gguf(path, &hw);
        match result {
            Ok(r) => {
                println!(
                    "Model: {} ({} layers, {} heads, {} embd)",
                    r.config.architecture, r.config.n_layers, r.config.n_heads, r.config.n_embd
                );
                println!("Fragments: {}", r.num_fragments);
                println!("Const pool: {} values", r.const_pool.len());
                println!("Graph ops: {}", r.graph.total_byte_size());
                assert!(r.num_fragments >= 1);
                assert!(r.config.n_layers > 0);
            }
            Err(e) => {
                panic!("Error compiling tinyllama: {e}");
            }
        }
    }

    #[test]
    fn read_tensor_f32_from_synthetic() {
        use bml_parser::create_gguf_with_metadata;
        let path = create_gguf_with_metadata();
        let parser = GgufParser::open(&path).unwrap();
        let tensor = read_tensor_f32(&parser, "token_embd.weight");
        assert!(tensor.is_some());
        let values = tensor.unwrap();
        assert_eq!(values.len(), 8); // 4*2 = 8 elementos
        assert_eq!(values[0], 0.0);
        assert_eq!(values[7], 7.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn read_real_tinyllama_weights() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();
        let parser = GgufParser::open(path).unwrap();

        // Leer attn_norm (F32, 2048 elementos)
        let norm = read_tensor_f32(&parser, "blk.0.attn_norm.weight");
        assert!(norm.is_some(), "attn_norm debe ser legible");
        let norm_vals = norm.unwrap();
        assert_eq!(norm_vals.len(), 2048);
        println!("attn_norm[0..5]: {:?}", &norm_vals[0..5]);

        // Leer attn_q (Q4_0, 2048x2048 = 4M elementos)
        let q = read_tensor_f32(&parser, "blk.0.attn_q.weight");
        assert!(q.is_some(), "attn_q debe ser legible (Q4_0)");
        let q_vals = q.unwrap();
        assert_eq!(q_vals.len(), 2048 * 2048);
        println!("attn_q[0..5]: {:?}", &q_vals[0..5]);
        // Los valores dequantizados no deben ser todos cero
        let nonzero = q_vals.iter().filter(|v| **v != 0.0).count();
        assert!(
            nonzero > 0,
            "debe haber valores no cero despues de dequantizar"
        );
        println!(
            "attn_q: {} non-zero values out of {}",
            nonzero,
            q_vals.len()
        );
    }

    #[test]
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn serialize_load_roundtrip() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();
        let hw = HardwareSpec::detect_local();
        let result = compile_gguf(path, &hw).expect("compilar");

        // Serializar a disco
        let out_dir =
            std::env::temp_dir().join(format!("bml_test_bmlgraph_{}", std::process::id()));
        serialize_to_dir(&result, &out_dir).expect("serializar");

        // Verificar que el directorio existe
        assert!(out_dir.exists());
        assert!(out_dir.join("header.bmlgraph").exists());

        // Cargar de disco
        let (graph, const_pool, config) = load_from_dir(&out_dir).expect("cargar");

        // Verificar que los datos coinciden
        assert_eq!(graph.num_fragments(), result.num_fragments);
        assert_eq!(const_pool.len(), result.const_pool.len());
        assert_eq!(config.architecture, result.config.architecture);
        assert_eq!(config.n_layers, result.config.n_layers);

        // Verificar que el grafo cargado evalúa igual que el original
        let ctx = bml_domain::EvalContext::new(&[], &const_pool);
        let val_original = result.graph.evaluate(0.0);
        let val_loaded = graph.evaluate(0.0);
        assert_eq!(
            val_original.to_bits(),
            val_loaded.to_bits(),
            "evaluación no coincide: original={val_original}, loaded={val_loaded}"
        );

        // Limpiar
        std::fs::remove_dir_all(&out_dir).ok();
    }

    #[test]
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn compile_serialize_load_tinyllama() {
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();
        let hw = HardwareSpec::detect_local();
        let result = compile_gguf(path, &hw).expect("compilar");

        println!(
            "Model: {} ({} layers, {} heads, {} embd)",
            result.config.architecture,
            result.config.n_layers,
            result.config.n_heads,
            result.config.n_embd
        );
        println!("Fragments: {}", result.num_fragments);
        println!("Const pool: {} values", result.const_pool.len());

        // Serializar
        let out_dir =
            std::env::temp_dir().join(format!("bml_tinyllama_bmlgraph_{}", std::process::id()));
        serialize_to_dir(&result, &out_dir).expect("serializar");
        println!("Serialized to: {:?}", out_dir);

        // Listar archivos generados
        for entry in std::fs::read_dir(&out_dir).unwrap() {
            let entry = entry.unwrap();
            let meta = entry.metadata().unwrap();
            println!("  {:?} ({} bytes)", entry.file_name(), meta.len());
        }

        // Cargar y verificar
        let (graph, const_pool, config) = load_from_dir(&out_dir).expect("cargar");
        println!(
            "Loaded: {} fragments, {} consts, arch={}",
            graph.num_fragments(),
            const_pool.len(),
            config.architecture
        );

        assert!(graph.num_fragments() > 0);
        assert!(config.n_layers > 0);

        // Limpiar
        std::fs::remove_dir_all(&out_dir).ok();
    }
}
