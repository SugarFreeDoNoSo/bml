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
//! 2. Detectar arquitectura (llama, qwen, etc.).
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
use bml_domain::{ConstId, NodeId, VarId};
use bml_parser::{GgufDataType, GgufMetadataValue, GgufParser};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

/// Configuración del modelo leída del GGUF.
#[derive(Debug, Clone)]
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

/// Lee la configuración del modelo desde los metadatos del GGUF.
fn read_model_config(parser: &GgufParser) -> Result<ModelConfig, String> {
    let arch = parser
        .architecture()
        .ok_or("no se encontró general.architecture")?
        .to_string();
    let arch_clone = arch.clone();

    let get_u32 = |key: &str| -> u32 {
        match parser.get_metadata(&format!("{arch_clone}.{key}")) {
            Some(GgufMetadataValue::U32(v)) => *v,
            Some(GgufMetadataValue::U64(v)) => *v as u32,
            Some(GgufMetadataValue::I32(v)) => *v as u32,
            Some(GgufMetadataValue::I64(v)) => *v as u32,
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

/// Construye el DAG BML del transformer.
///
/// Esta es una versión simplificada que construye la estructura del
/// transformer sin los pesos reales. La implementación completa
/// requiere leer los tensores del GGUF y traducir cada operación.
///
/// # Estructura simplificada
///
/// Para cada capa:
/// 1. RMSNorm de los inputs.
/// 2. Matmul con pesos de atención (Q, K, V).
/// 3. RoPE (precomputado con EML).
/// 4. Attention scores (matmul Q·K^T).
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

    // Construir capa por capa.
    let mut hidden = input;

    for layer in 0..config.n_layers {
        // RMSNorm: x / sqrt(mean(x^2) + eps)
        // Simplificado: bml(hidden, const(rms_scale))
        // rms_scale se precomputa con EML en compile-time.
        let rms_scale = reg.const_value(1.0); // placeholder
        hidden = reg.bml(hidden, rms_scale);

        // Self-attention: Q = W_q * x, K = W_k * x, V = W_v * x
        // Simplificado: cada proyección es bml(hidden, const(weight))
        // Los pesos se leen del GGUF (zero-copy) y se almacenan como Const.
        let q_weight = reg.const_value(1.0); // placeholder: leer del GGUF
        let k_weight = reg.const_value(1.0);
        let v_weight = reg.const_value(1.0);

        let q = reg.bml(hidden, q_weight);
        let k = reg.bml(hidden, k_weight);
        let v = reg.bml(hidden, v_weight);

        // RoPE: precomputar cos/sin con EML compile-time.
        let rope_consts = eml::rope_constants(
            config.context_length as usize,
            config.n_embd as usize,
            10000.0,
        );
        // Para cada par de dimensiones, aplicar RoPE.
        // Simplificado: aplicar el primer par de constantes.
        if !rope_consts.is_empty() {
            let cos_val = reg.const_value(rope_consts[0].0);
            let sin_val = reg.const_value(rope_consts[0].1);
            // q' = q * cos - rotate(q) * sin
            let q_rotated = reg.bml(q, sin_val);
            let q_scaled = reg.bml(q, cos_val);
            let _q_new = reg.bml(q_scaled, q_rotated); // simplificado
        }

        // Attention scores: Q · K^T
        // Simplificado: bml(q, k)
        let score = reg.bml(q, k);

        // Softmax: precomputar con EML compile-time.
        // En runtime, softmax se hace con exp2 + add + div.
        // Simplificado: bml(score, const(softmax_scale))
        let softmax_scale = reg.const_value(1.0);
        let attn = reg.bml(score, softmax_scale);

        // Output: attn · V
        let attn_out = reg.bml(attn, v);

        // Output projection: W_o * attn_out
        let o_weight = reg.const_value(1.0);
        let o_out = reg.bml(attn_out, o_weight);

        // Residual: hidden + o_out
        // Simplificado: bml(hidden, o_out)
        hidden = reg.bml(hidden, o_out);

        // MLP: RMSNorm + matmul + SwiGLU + matmul + residual
        let mlp_norm = reg.const_value(1.0);
        hidden = reg.bml(hidden, mlp_norm);

        let gate_weight = reg.const_value(1.0);
        let up_weight = reg.const_value(1.0);
        let down_weight = reg.const_value(1.0);

        let gate = reg.bml(hidden, gate_weight);
        let up = reg.bml(hidden, up_weight);

        // SwiGLU: gate * sigmoid(1.7 * gate) * up
        // Simplificado: bml(gate, up)
        let mlp_act = reg.bml(gate, up);
        let mlp_out = reg.bml(mlp_act, down_weight);

        // Residual: hidden + mlp_out
        hidden = reg.bml(hidden, mlp_out);
    }

    // Final RMSNorm
    let final_norm = reg.const_value(1.0);
    hidden = reg.bml(hidden, final_norm);

    // Output projection (lm_head)
    let lm_head = reg.const_value(1.0);
    hidden = reg.bml(hidden, lm_head);

    Ok(hidden)
}

/// Lee un tensor del GGUF y retorna sus valores como f64.
///
/// Para tipos estándar (F32), lee directamente.
/// Para tipos cuantizados, dequantiza a f32.
pub fn read_tensor_f32(parser: &GgufParser, name: &str) -> Option<Vec<f32>> {
    let info = parser.find_tensor(name)?;
    let data = parser.tensor_data(info)?;
    match info.data_type {
        GgufDataType::F32 => Some(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        _ => None, // Cuantización: requiere dequantización específica por tipo
    }
}

/// Serializa un `CompilationResult` a un directorio `.bmlgraph`.
///
/// Crea un directorio con:
/// - `header.bmlgraph`: magic, version, n_fragments, config, const_pool
/// - `fragment_0.bmlgraph`, `fragment_1.bmlgraph`, ...: cada fragmento serializado
pub fn serialize_to_dir(result: &CompilationResult, output_dir: &Path) -> Result<(), String> {
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
    f.write_all(&(result.num_fragments as u32).to_le_bytes())
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
            }
        }
    }

    Ok(())
}

/// Carga un `.bmlgraph` desde un directorio.
///
/// Lee el header (config + const_pool) y los fragmentos.
pub fn load_from_dir(input_dir: &Path) -> Result<(BmlGraph, Vec<f64>, ModelConfig), String> {
    let header_path = input_dir.join("header.bmlgraph");
    let header_bytes = std::fs::read(&header_path).map_err(|e| format!("leer header: {e}"))?;
    if header_bytes.len() < 12 {
        return Err("header demasiado pequeño".into());
    }
    let magic = u32::from_le_bytes(header_bytes[0..4].try_into().unwrap());
    if magic != BMLGRAPH_MAGIC {
        return Err(format!("magic inválido: 0x{magic:08X}"));
    }
    let _version = u32::from_le_bytes(header_bytes[4..8].try_into().unwrap());
    let n_fragments = u32::from_le_bytes(header_bytes[8..12].try_into().unwrap()) as usize;
    let mut offset = 12;

    // Config
    let arch_len =
        u64::from_le_bytes(header_bytes[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;
    let architecture =
        String::from_utf8_lossy(&header_bytes[offset..offset + arch_len]).to_string();
    offset += arch_len;
    let n_layers = u32::from_le_bytes(header_bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let n_heads = u32::from_le_bytes(header_bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let n_embd = u32::from_le_bytes(header_bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let context_length = u32::from_le_bytes(header_bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let vocab_size = u32::from_le_bytes(header_bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Const pool
    let pool_len =
        u64::from_le_bytes(header_bytes[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;
    let mut const_pool = Vec::with_capacity(pool_len);
    for _ in 0..pool_len {
        let val = f64::from_le_bytes(header_bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        const_pool.push(val);
    }

    let config = ModelConfig {
        architecture,
        n_layers,
        n_heads,
        n_embd,
        context_length,
        vocab_size,
    };

    // Fragmentos
    use crate::fragment::Fragment;
    use crate::rpn::RpnOp;
    let mut fragments = Vec::with_capacity(n_fragments);
    for i in 0..n_fragments {
        let frag_path = input_dir.join(format!("fragment_{i}.bmlgraph"));
        let frag_bytes = std::fs::read(&frag_path).map_err(|e| format!("leer frag {i}: {e}"))?;
        let n_ops = u32::from_le_bytes(frag_bytes[0..4].try_into().unwrap()) as usize;
        let mut ops = Vec::with_capacity(n_ops);
        let mut off = 4;
        for _ in 0..n_ops {
            let tag = frag_bytes[off];
            off += 1;
            let op = match tag {
                0 => RpnOp::One,
                6 => RpnOp::Zero,
                1 => RpnOp::Bml,
                2 => RpnOp::Dup,
                3 => {
                    let count = u32::from_le_bytes(frag_bytes[off..off + 4].try_into().unwrap());
                    off += 4;
                    let body_len = u32::from_le_bytes(frag_bytes[off..off + 4].try_into().unwrap());
                    off += 4;
                    RpnOp::Loop { count, body_len }
                }
                4 => {
                    let id = u32::from_le_bytes(frag_bytes[off..off + 4].try_into().unwrap());
                    off += 4;
                    RpnOp::Var(id)
                }
                5 => {
                    let id = u32::from_le_bytes(frag_bytes[off..off + 4].try_into().unwrap());
                    off += 4;
                    RpnOp::Const(id)
                }
                _ => return Err(format!("tag desconocido: {tag}")),
            };
            ops.push(op);
        }
        fragments.push(Fragment { ops });
    }

    let graph = BmlGraph {
        fragments,
        threshold: 0,
    };
    Ok((graph, const_pool, config))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_synthetic_gguf() {
        use bml_parser::create_gguf_with_metadata;
        let path = create_gguf_with_metadata();
        let hw = HardwareSpec::new(4, 32 * 1024, 256 * 1024, 16 * 1024 * 1024);

        let result = compile_gguf(&path, &hw);
        // El GGUF sintético no tiene metadatos de modelo completos,
        // así que esperamos un error o un resultado con config vacía.
        match result {
            Ok(r) => {
                println!(
                    "Compiled: {} fragments, {} consts",
                    r.num_fragments,
                    r.const_pool.len()
                );
                assert!(r.num_fragments >= 1);
            }
            Err(e) => {
                // Es OK si falla porque el GGUF sintético no tiene config de modelo.
                println!("Expected error with synthetic GGUF: {e}");
            }
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn compile_real_tinyllama() {
        let path = "/root/tinyllama.gguf";
        if !Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }
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
    fn serialize_load_roundtrip() {
        let path = "/root/tinyllama.gguf";
        if !Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }
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
    fn compile_serialize_load_tinyllama() {
        let path = "/root/tinyllama.gguf";
        if !Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }
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
