//! # bml-cli
//!
//! CLI para compilar modelos GGUF a `.bmlgraph` y ejecutar inferencia BML.
//!
//! # Subcomandos
//!
//! ```sh
//! # Compilar GGUF → .bmlgraph
//! bml-cli compile -m modelo.gguf -o modelo.bmlgraph
//!
//! # Ejecutar desde .bmlgraph (pre-compilado) o GGUF (compila en caliente)
//! bml-cli run -m modelo.bmlgraph/ -p "Hello" -n 64
//! bml-cli run -m modelo.gguf -p "Hello" -n 64
//! ```

use bml_compiler::distributed::{
    compile_distributed, load_distributed_header, serialize_distributed,
};
use bml_compiler::gguf_compiler::InferenceCompiler;
use bml_compiler::sampler;
use bml_runtime::net::{recv_msg, send_msg, Message, MsgType, NodeHandle};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
mod weights;

#[derive(Parser, Debug)]
#[command(
    name = "bml-cli",
    version,
    about = "BML inference CLI — compila y ejecuta modelos GGUF"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Convierte GGUF a pesos BML exactos en un archivo .bmlgguf autónomo
    ConvertWeights(weights::ConvertArgs),
    /// Inspecciona catálogo BMLGGUF; --verify-all comprueba todos los bloques
    InspectWeights(weights::InspectArgs),
    /// Compila un GGUF a formato .bmlgraph
    Compile {
        /// Ruta al archivo GGUF de entrada
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        /// Directorio de salida para .bmlgraph
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Tamaño máximo de fragmento L1 en bytes (default: 32768 = 32 KB)
        #[arg(long = "l1-threshold", default_value_t = 32768)]
        l1_threshold: usize,

        /// Tamaño máximo de fragmento L3 en bytes (default: 8388608 = 8 MB)
        #[arg(long = "l3-threshold", default_value_t = 8388608)]
        l3_threshold: usize,

        /// Compila en modo distribuido: un fragmento self-contained por capa
        #[arg(long = "distributed")]
        distributed: bool,

        /// Compilar aritmética BML completa (experimental); requiere contexto explícito.
        #[arg(long, conflicts_with = "distributed", requires = "context_size")]
        symbolic: bool,

        /// Contexto máximo que cubrirá el artefacto AOT --symbolic.
        #[arg(long = "context-size", requires = "symbolic")]
        context_size: Option<u32>,

        /// Perfil RMS acotado experimental: máximo absoluto de activaciones.
        #[arg(long, requires_all = ["symbolic", "rms_weight_bound"])]
        rms_input_bound: Option<f64>,

        /// Perfil RMS acotado experimental: máximo absoluto de pesos RMS.
        #[arg(long, requires_all = ["symbolic", "rms_input_bound"])]
        rms_weight_bound: Option<f64>,

        /// Capas por fragmento (solo --distributed, default: 1 = una capa por fragmento)
        #[arg(long = "layers-per-fragment", default_value_t = 1)]
        layers_per_fragment: u32,
    },

    /// Ejecuta inferencia desde .bmlgraph, GGUF o BMLGGUF con evaluación explícita
    Run {
        /// Ruta al modelo: directorio .bmlgraph/, GGUF o contenedor BMLGGUF
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        /// Evaluación explícita de pesos BMLGGUF; no hay selección automática
        #[arg(long, value_enum, requires = "max_evaluated_weights", conflicts_with_all = ["weights", "bml"])]
        weight_evaluation: Option<weights::WeightEvaluation>,

        /// Máximo de pesos F32 materializados; requiere --weight-evaluation
        #[arg(long, requires = "weight_evaluation", value_parser = clap::value_parser!(u64).range(1..))]
        max_evaluated_weights: Option<u64>,

        /// Relocated GGUF weights for a compiled causal artifact (hash must match).
        #[arg(long = "weights")]
        weights: Option<PathBuf>,

        /// Texto de entrada (prompt)
        #[arg(short = 'p', long = "prompt")]
        prompt: String,

        /// Número de tokens a generar
        #[arg(short = 'n', long = "num-tokens", default_value_t = 128)]
        num_tokens: u32,

        /// Número de threads
        #[arg(short = 't', long = "threads", default_value_t = 4)]
        threads: u32,

        /// Temperatura de sampling (0 = greedy)
        #[arg(long = "temp", default_value_t = 0.8)]
        temp: f64,

        /// Tamaño máximo de contexto
        #[arg(short = 'c', long = "context-size", default_value_t = 2048)]
        context_size: u32,

        /// Usar proyecciones RPN IEEE en la ruta GGUF experimental
        #[arg(long = "bml")]
        bml: bool,
    },

    /// Distribuye fragmentos a nodos workers via TCP
    Distribute {
        /// Directorio .bmlgraph distribuido (con fragmentos self-contained)
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        /// Lista de nodos workers (host:port,host:port,...)
        #[arg(long = "nodes")]
        nodes: String,

        /// Prompt para generar
        #[arg(short = 'p', long = "prompt")]
        prompt: String,

        /// Número de tokens a generar
        #[arg(short = 'n', long = "num-tokens", default_value_t = 16)]
        num_tokens: u32,

        /// Temperatura de sampling
        #[arg(long = "temp", default_value_t = 0.8)]
        temp: f64,
    },
}

/// Determina si un path es un directorio .bmlgraph o un archivo GGUF.
fn is_bmlgraph_dir(path: &std::path::Path) -> bool {
    path.is_dir()
        && (path.join("header.bmlgraph").exists()
            || path.join(bml_compiler::causal_artifact::MANIFEST).exists()
            || path
                .join(bml_compiler::symbolic_artifact::MANIFEST)
                .exists())
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::ConvertWeights(args) => weights::exit_on_error(weights::convert(args)),
        Command::InspectWeights(args) => weights::exit_on_error(weights::inspect(args)),
        Command::Compile {
            model,
            output,
            l1_threshold,
            l3_threshold: _,
            distributed,
            symbolic,
            context_size,
            rms_input_bound,
            rms_weight_bound,
            layers_per_fragment,
        } => {
            if !model.exists() {
                eprintln!("Error: el archivo GGUF no existe: {}", model.display());
                std::process::exit(1);
            }

            println!("Compilando {} → {}", model.display(), output.display());

            if distributed {
                println!(
                    "Modo distribuido: {} capas por fragmento",
                    layers_per_fragment
                );
                println!("Leyendo pesos del GGUF (dequantizando)...");

                let fragments = match compile_distributed(&model, layers_per_fragment, true) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("Error compilando distribuido: {e}");
                        std::process::exit(1);
                    }
                };

                let config = &fragments[0].config;
                println!(
                    "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
                    config.architecture,
                    config.n_layers,
                    config.n_heads,
                    config.n_embd,
                    config.vocab_size,
                );
                println!("Fragmentos: {}", fragments.len());

                let total_weights: usize = fragments.iter().map(|f| f.weights.len()).sum();
                println!(
                    "Total pesos: {} ({:.1} MB)",
                    total_weights,
                    total_weights as f64 * 4.0 / (1024.0 * 1024.0),
                );

                for frag in &fragments {
                    let weight_mb = frag.weights.len() as f64 * 4.0 / (1024.0 * 1024.0);
                    println!(
                        "  fragmento {} (capas {}-{}): {} tensores, {:.1} MB",
                        frag.fragment_id,
                        frag.layer_start,
                        frag.layer_end,
                        frag.tensors.len(),
                        weight_mb,
                    );
                }

                if let Err(e) = serialize_distributed(&fragments, &output) {
                    eprintln!("Error serializando: {e}");
                    std::process::exit(1);
                }
            } else if symbolic {
                let limits = bml_compiler::symbolic_kernels::KernelLimits {
                    fragment_bytes: l1_threshold,
                    ..Default::default()
                };
                let context = context_size.expect("clap requires symbolic context");
                let manifest = inference_result(if let Some(input) = rms_input_bound {
                    bml_compiler::symbolic_artifact::compile_directory_with_rms_bounds(
                        &model,
                        &output,
                        context,
                        bml_compiler::symbolic_kernels::RmsBounds {
                            input,
                            weight: rms_weight_bound.expect("clap requires paired RMS bounds"),
                        },
                        limits,
                    )
                } else {
                    bml_compiler::symbolic_artifact::compile_directory(
                        &model, &output, context, limits,
                    )
                });
                println!("Perfil: {}", manifest.profile);
                println!(
                    "Kernels BML: {}; contexto compilado: {}",
                    manifest.kernels.len(),
                    manifest.context_length
                );
                println!("Serializado a: {}", output.display());
            } else {
                let manifest = inference_result(bml_compiler::causal_artifact::compile_directory(
                    &model,
                    &output,
                    l1_threshold,
                ));
                println!("Perfil: {}", manifest.profile);
                println!(
                    "Programas RPN: {}; proyecciones: {}",
                    manifest.programs.len(),
                    manifest.projections.len()
                );
            }

            println!("Serializado a: {}", output.display());
            if let Ok(entries) = std::fs::read_dir(&output) {
                for entry in entries.flatten() {
                    if let Ok(m) = entry.metadata() {
                        println!(
                            "  {} ({} bytes)",
                            entry.file_name().to_string_lossy(),
                            m.len()
                        );
                    }
                }
            }
        }

        Command::Run {
            model,
            weight_evaluation,
            max_evaluated_weights,
            weights,
            prompt,
            num_tokens,
            threads: _,
            temp,
            context_size,
            bml,
        } => {
            if weights.is_some()
                && !model
                    .join(bml_compiler::causal_artifact::MANIFEST)
                    .is_file()
                && !model
                    .join(bml_compiler::symbolic_artifact::MANIFEST)
                    .is_file()
            {
                eprintln!("--weights solo se admite con un artefacto causal compilado");
                std::process::exit(1);
            }
            if !model.exists() {
                eprintln!("Error: el modelo no existe: {}", model.display());
                std::process::exit(1);
            }

            if let Some(evaluation) = weight_evaluation {
                let count = inference_result(
                    usize::try_from(
                        max_evaluated_weights.expect("clap requires materialization cap"),
                    )
                    .map_err(|_| "materialization cap exceeds host usize".to_string()),
                );
                let loaded = inference_result(
                    bml_runtime::symbolic_causal::SymbolicCausalModel::open_bmlgguf(
                        &model,
                        Default::default(),
                        count,
                        evaluation.into(),
                        Default::default(),
                    ),
                );
                println!("BMLGGUF evaluation: checked-exact-f32; max_evaluated_weights: {count}");
                run_text_model(
                    TextArtifact::Symbolic(Box::new(loaded)),
                    &prompt,
                    num_tokens,
                    temp,
                    context_size,
                    "BMLGGUF",
                );
            } else if model.is_file()
                && inference_result(weights::is_container(&model).map_err(|e| e.to_string()))
            {
                eprintln!("BMLGGUF requiere --weight-evaluation checked-exact-f32 y --max-evaluated-weights N");
                std::process::exit(1);
            } else if is_bmlgraph_dir(&model) {
                if model.join(bml_compiler::causal_artifact::MANIFEST).exists()
                    || model
                        .join(bml_compiler::symbolic_artifact::MANIFEST)
                        .exists()
                {
                    run_from_causal_artifact(
                        &model,
                        weights.as_deref(),
                        &prompt,
                        num_tokens,
                        temp,
                        context_size,
                    );
                } else {
                    eprintln!("El contenedor escalar heredado no contiene un modelo causal ejecutable. Recompila el GGUF con `bml-cli compile` en un directorio nuevo.");
                    std::process::exit(1);
                }
            } else if bml {
                run_from_gguf_bml(&model, &prompt, num_tokens, temp, context_size);
            } else {
                run_from_gguf(&model, &prompt, num_tokens, temp, context_size);
            }
        }

        Command::Distribute {
            model,
            nodes,
            prompt,
            num_tokens,
            temp,
        } => {
            distribute(&model, &nodes, &prompt, num_tokens, temp);
        }
    }
}

enum TextArtifact {
    Native(Box<bml_runtime::causal_artifact::CompiledCausalModel>),
    Symbolic(Box<bml_runtime::symbolic_causal::SymbolicCausalModel>),
}
impl TextArtifact {
    fn open(directory: &std::path::Path, source: Option<&std::path::Path>) -> Result<Self, String> {
        let native = directory
            .join(bml_compiler::causal_artifact::MANIFEST)
            .is_file();
        let symbolic = directory
            .join(bml_compiler::symbolic_artifact::MANIFEST)
            .is_file();
        match (native, symbolic) {
            (true, false) => Ok(Self::Native(Box::new(
                bml_runtime::causal_artifact::CompiledCausalModel::open(directory, source)?,
            ))),
            (false, true) => Ok(Self::Symbolic(Box::new(
                bml_runtime::symbolic_causal::SymbolicCausalModel::open_artifact(
                    directory,
                    source,
                    Default::default(),
                )?,
            ))),
            _ => Err("artefacto causal ausente o ambiguo".into()),
        }
    }
    fn compiler(&self) -> &InferenceCompiler {
        match self {
            Self::Native(m) => m.compiler(),
            Self::Symbolic(m) => m.compiler(),
        }
    }
    fn context_limit(&self) -> u32 {
        match self {
            Self::Native(m) => m.compiler().config().context_length,
            Self::Symbolic(m) => m.context_limit(),
        }
    }
    fn profile(&self) -> &str {
        match self {
            Self::Native(m) => &m.manifest().profile,
            Self::Symbolic(m) => m.profile(),
        }
    }
    fn forward_cached(
        &mut self,
        ids: &[u32],
        cache: &mut bml_compiler::HashConsedKV,
    ) -> Result<Vec<f64>, String> {
        match self {
            Self::Native(m) => m.forward_cached(ids, cache),
            Self::Symbolic(m) => m.forward_cached(ids, cache),
        }
    }
}

/// Ejecuta inferencia desde un directorio .bmlgraph pre-compilado.
fn run_from_causal_artifact(
    directory: &std::path::Path,
    source: Option<&std::path::Path>,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
    context_size: u32,
) {
    let model = inference_result(TextArtifact::open(directory, source));
    run_text_model(
        model,
        prompt,
        num_tokens,
        temp,
        context_size,
        "artefacto compilado",
    );
}

fn run_text_model(
    mut model: TextArtifact,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
    context_size: u32,
    source_label: &str,
) {
    let compiler = model.compiler();
    let ids = encode_prompt(compiler.vocab(), prompt);
    let eos_token_id = compiler.vocab().eos_token_id;
    let capacity = context_size.min(model.context_limit());
    let mut cache = inference_result(bml_compiler::HashConsedKV::try_new(
        compiler.config().n_layers,
        compiler.n_kv_heads(),
        compiler.head_dim(),
        capacity,
        bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES,
    ));
    println!("Perfil compilado: {}", model.profile());
    println!("Tokenizado: {} tokens", ids.len());
    let mut logits = inference_result(model.forward_cached(&ids, &mut cache));
    let mut generated = 0u32;
    for step in 0..num_tokens {
        if ids.len() + generated as usize >= capacity as usize {
            break;
        }
        let next = sampler::sample(&logits, temp, step as u64).unwrap_or(eos_token_id);
        generated += 1;
        if next == eos_token_id {
            println!("\n[EOS]");
            break;
        }
        emit_token(model.compiler().vocab(), next);
        if step + 1 < num_tokens && ids.len() + (generated as usize) < capacity as usize {
            logits = inference_result(model.forward_cached(&[next], &mut cache));
        }
    }
    println!("\n\nGenerados {generated} tokens desde {source_label}");
}

fn run_from_gguf(
    gguf_path: &std::path::Path,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
    context_size: u32,
) {
    println!("Cargando modelo desde {}...", gguf_path.display());
    println!("Validando el modelo y mapeando los pesos...");

    let compiler = match InferenceCompiler::open_mapped(gguf_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error cargando modelo: {e}");
            std::process::exit(1);
        }
    };

    let config = compiler.config();
    let vocab = compiler.vocab();
    println!(
        "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
        config.architecture,
        config.n_layers,
        config.n_heads,
        config.n_embd,
        vocab.len(),
    );
    println!(
        "Decoded normalization pool: {} valores ({:.1} MiB)",
        compiler.weight_pool().len(),
        compiler.weight_pool().len() as f64 * std::mem::size_of::<f32>() as f64 / (1024.0 * 1024.0),
    );
    println!(
        "Mapped GGUF: {:.1} MiB",
        compiler.mapped_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!("Prompt: {}", prompt);
    println!("Generando {} tokens (temp={})...\n", num_tokens, temp);

    let prompt_ids = encode_prompt(vocab, prompt);
    println!("Tokenizado: {} tokens", prompt_ids.len());

    let max_ctx = context_size.min(config.context_length) as usize;
    if max_ctx == 0 {
        eprintln!("Error: contexto cero");
        std::process::exit(1);
    }

    // Causal F32 KV cache; never wrap positions during generation.
    let mut kv = inference_result(bml_compiler::HashConsedKV::try_new(
        config.n_layers,
        compiler.n_kv_heads(),
        config.n_embd / config.n_heads,
        max_ctx as u32,
        bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES,
    ));

    let mut sequence = prompt_ids.clone();
    // Primer paso: procesar todo el prompt (construye el KV cache)
    let logits = inference_result(compiler.forward_cached(&sequence, &mut kv));
    let mut last_logits = logits;

    for step in 0..num_tokens {
        if sequence.len() >= max_ctx {
            println!("\n[context limit]");
            break;
        }
        // Sampling del token anterior
        let next_id =
            sampler::sample(&last_logits, temp, step as u64).unwrap_or(vocab.eos_token_id);

        sequence.push(next_id);
        if next_id != vocab.eos_token_id {
            emit_token(vocab, next_id);
        }
        if next_id == vocab.eos_token_id {
            println!("\n[EOS]");
            break;
        }

        if step + 1 < num_tokens && sequence.len() < max_ctx {
            last_logits = inference_result(compiler.forward_cached(&[next_id], &mut kv));
        }
    }

    println!("\n");
    println!(
        "Generados {} tokens desde prompt '{}'",
        sequence.len() - prompt_ids.len(),
        prompt,
    );
}

/// Distribuye fragmentos a nodos workers y coordina la inferencia.
///
/// # Flujo
///
/// 1. Carga el header del .bmlgraph distribuido (config + n_fragments)
/// 2. Conecta a cada nodo worker via TCP
/// 3. Envía cada fragmento a un nodo (ExecuteFragment)
/// 4. Para cada token a generar:
///    a. Tokeniza el prompt → token_ids
///    b. Embedding lookup → hidden state
///    c. Envía hidden al primer nodo (BatchRequest)
///    d. Cada nodo ejecuta sus capas y pasa hidden al siguiente
///    e. El último nodo produce logits
///    f. Sampling → next token
/// 5. Repite hasta EOS o num_tokens
fn distribute(
    bmlgraph_dir: &std::path::Path,
    nodes_str: &str,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
) {
    if !bmlgraph_dir.is_dir() {
        eprintln!(
            "Error: {} no es un directorio .bmlgraph",
            bmlgraph_dir.display()
        );
        std::process::exit(1);
    }

    let (config, n_fragments) = match load_distributed_header(bmlgraph_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error cargando header: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Modelo: {} ({} capas, {} heads, {} embd)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd,
    );
    println!("Fragmentos: {}", n_fragments);

    let nodes: Vec<&str> = nodes_str.split(',').map(|s| s.trim()).collect();
    println!("Nodos: {:?}", nodes);

    if nodes.len() != n_fragments {
        eprintln!(
            "Warning: {} nodos para {} fragmentos — algunos nodos recibiran multiples fragmentos",
            nodes.len(),
            n_fragments,
        );
    }

    // Conectar a cada nodo y enviar fragmentos
    let mut node_handles: Vec<NodeHandle> = Vec::new();

    for (i, node_addr) in nodes.iter().enumerate() {
        let frag_id = if i < n_fragments {
            i as u32
        } else {
            (i % n_fragments) as u32
        };
        let frag_path = bmlgraph_dir.join(format!("fragment_{}.bmlgraph", frag_id));

        if !frag_path.exists() {
            eprintln!("Error: fragmento {} no existe", frag_path.display());
            std::process::exit(1);
        }

        let frag_bytes = match std::fs::read(&frag_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Error leyendo {}: {e}", frag_path.display());
                std::process::exit(1);
            }
        };

        let weight_mb = frag_bytes.len() as f64 / (1024.0 * 1024.0);
        println!("Conectando a {node_addr} (fragmento {frag_id}, {weight_mb:.1} MB)...");

        let mut handle = match NodeHandle::connect(node_addr) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Error conectando a {node_addr}: {e}");
                std::process::exit(1);
            }
        };

        // Health check
        match handle.health_check() {
            Ok(true) => {}
            Ok(false) => {
                eprintln!("Error: {node_addr} no está healthy");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error health check {node_addr}: {e}");
                std::process::exit(1);
            }
        }

        // Enviar fragmento via TCP raw (ExecuteFragment)
        match send_msg(
            &mut handle.stream_mut(),
            &Message::new(MsgType::ExecuteFragment, frag_bytes),
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error enviando fragmento a {node_addr}: {e}");
                std::process::exit(1);
            }
        }

        // Esperar confirmación
        match recv_msg(handle.stream_mut()) {
            Ok(msg) => {
                if msg.msg_type == MsgType::ReportResult {
                    let result = f64::from_le_bytes(
                        msg.payload
                            .get(0..8)
                            .unwrap_or(&[0; 8])
                            .try_into()
                            .unwrap_or([0; 8]),
                    );
                    if result.is_finite() && result > 0.0 {
                        println!("  {node_addr}: fragmento cargado OK");
                    } else {
                        eprintln!("  {node_addr}: error cargando fragmento");
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error recibiendo confirmación de {node_addr}: {e}");
                std::process::exit(1);
            }
        }

        node_handles.push(handle);
    }

    println!(
        "Todos los fragmentos distribuidos. {} nodos conectados.",
        node_handles.len()
    );
    println!("Prompt: {}", prompt);
    println!("Generando {} tokens (temp={})...\n", num_tokens, temp);

    // Necesitamos el vocabulario para tokenizar/sampling.
    // Lo cargamos desde el GGUF original (el .bmlgraph distribuido no lo tiene).
    // En una implementación completa, el vocab iría en el header.
    // Por ahora, usamos el InferenceCompiler solo para el vocab.
    let gguf_candidates = [
        bmlgraph_dir.with_extension("gguf"),
        bmlgraph_dir
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(format!("{}.gguf", config.architecture)),
    ];
    let vocab = if let Some(gguf_path) = gguf_candidates.iter().find(|p| p.exists()) {
        match InferenceCompiler::open_mapped(gguf_path) {
            Ok(c) => Some(c.vocab().clone()),
            Err(_) => None,
        }
    } else {
        None
    };

    let vocab = match vocab {
        Some(v) => v,
        None => {
            eprintln!("Warning: no se pudo cargar vocabulario. Output sin decode.");
            eprintln!("  Coloca el GGUF junto al .bmlgraph/ para enable decode.");
            // Sin vocab, solo podemos ejecutar el forward pass sin generar texto.
            eprintln!("Ejecutando forward pass distribuido (sin decode)...");
            let n_embd = config.n_embd as usize;
            let mut hidden = vec![0.0_f64; n_embd];

            // Pasar hidden por cada nodo en secuencia
            for (i, handle) in node_handles.iter_mut().enumerate() {
                let mut payload = Vec::with_capacity(4 + n_embd * 8);
                payload.extend_from_slice(&(n_embd as u32).to_le_bytes());
                for v in &hidden {
                    payload.extend_from_slice(&v.to_le_bytes());
                }

                send_msg(
                    &mut handle.stream_mut(),
                    &Message::new(MsgType::BatchRequest, payload),
                )
                .ok();
                let response = recv_msg(handle.stream_mut()).ok();

                if let Some(msg) = response {
                    if msg.msg_type == MsgType::BatchResult && msg.payload.len() >= 4 {
                        let n = u32::from_le_bytes(msg.payload[0..4].try_into().unwrap()) as usize;
                        hidden.clear();
                        for j in 0..n.min(msg.payload.len() / 8 - 1) {
                            let offset = 4 + j * 8;
                            if offset + 8 <= msg.payload.len() {
                                hidden.push(f64::from_le_bytes(
                                    msg.payload[offset..offset + 8].try_into().unwrap(),
                                ));
                            }
                        }
                        println!(
                            "  Nodo {i}: hidden[0..3] = {:?}",
                            &hidden[..hidden.len().min(3)]
                        );
                    }
                }
            }

            println!("\nForward pass distribuido completo.");
            println!("Hidden state final: {} dimensiones", hidden.len());
            return;
        }
    };

    // Pipeline autoregresivo con vocabulario
    let prompt_ids = encode_prompt(&vocab, prompt);
    println!("Tokenizado: {} tokens", prompt_ids.len());

    let mut sequence = prompt_ids.clone();
    let n_embd = config.n_embd as usize;

    for step in 0..num_tokens {
        // Embedding lookup (en el coordinador, liviano)
        let mut hidden = vec![0.0_f64; n_embd];
        for &tid in &sequence {
            // Placeholder: en implementación completa, usar embedding del último fragmento
            for (i, v) in hidden.iter_mut().enumerate() {
                *v += (tid as f64) * 0.001 / (i as f64 + 1.0);
            }
        }
        let scale = 1.0 / (sequence.len() as f64).sqrt();
        for v in &mut hidden {
            *v *= scale;
        }

        // Pasar hidden por cada nodo en secuencia
        for (_i, handle) in node_handles.iter_mut().enumerate() {
            let mut payload = Vec::with_capacity(4 + n_embd * 8);
            payload.extend_from_slice(&(n_embd as u32).to_le_bytes());
            for v in &hidden {
                payload.extend_from_slice(&v.to_le_bytes());
            }

            send_msg(
                &mut handle.stream_mut(),
                &Message::new(MsgType::BatchRequest, payload),
            )
            .ok();
            let response = recv_msg(handle.stream_mut()).ok();

            if let Some(msg) = response {
                if msg.msg_type == MsgType::BatchResult && msg.payload.len() >= 4 {
                    let n = u32::from_le_bytes(msg.payload[0..4].try_into().unwrap()) as usize;
                    hidden.clear();
                    for j in 0..n.min((msg.payload.len() - 4) / 8) {
                        let offset = 4 + j * 8;
                        if offset + 8 <= msg.payload.len() {
                            hidden.push(f64::from_le_bytes(
                                msg.payload[offset..offset + 8].try_into().unwrap(),
                            ));
                        }
                    }
                }
            }
        }

        // El último hidden son los logits (simplificado)
        let logits = &hidden;
        let next_id = sampler::sample(logits, temp, step as u64).unwrap_or(vocab.eos_token_id);

        sequence.push(next_id);
        if next_id != vocab.eos_token_id {
            emit_token(&vocab, next_id);
        }
        if next_id == vocab.eos_token_id {
            println!("\n[EOS]");
            break;
        }
    }

    println!(
        "\n\nDistribuido: {} tokens generados desde {} nodos",
        sequence.len() - prompt_ids.len(),
        node_handles.len(),
    );
}

/// Ejecuta inferencia desde GGUF usando building blocks BML.
///
/// Shared causal inference with RPN matrix projections.
fn run_from_gguf_bml(
    gguf_path: &std::path::Path,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
    context_size: u32,
) {
    use bml_cli::bml_inference;

    println!(
        "Cargando modelo desde {} (modo BML)...",
        gguf_path.display()
    );
    println!("Validando el modelo y mapeando los pesos...");

    let compiler = match InferenceCompiler::open_mapped(gguf_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error cargando modelo: {e}");
            std::process::exit(1);
        }
    };

    let config = compiler.config();
    let vocab = compiler.vocab();
    println!(
        "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
        config.architecture,
        config.n_layers,
        config.n_heads,
        config.n_embd,
        vocab.len(),
    );
    println!("Prompt: {}", prompt);
    println!(
        "Generando {} tokens (temp={}) con BML building blocks...\n",
        num_tokens, temp
    );

    let prompt_ids = encode_prompt(vocab, prompt);
    println!("Tokenizado: {} tokens", prompt_ids.len());

    let mut hot = bml_runtime::HotLoop::with_capacity(8192);
    let mut sequence = prompt_ids.clone();
    let max_ctx = context_size.min(config.context_length) as usize;

    let mut cache = inference_result(bml_compiler::HashConsedKV::try_new(
        config.n_layers,
        compiler.n_kv_heads(),
        compiler.head_dim(),
        max_ctx as u32,
        bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES,
    ));
    let mut logits = inference_result(bml_inference::forward_bml_cached(
        &mut hot,
        &compiler,
        &prompt_ids,
        &mut cache,
    ));
    for step in 0..num_tokens {
        if sequence.len() >= max_ctx {
            break;
        }
        let next_id = sampler::sample(&logits, temp, step as u64).unwrap_or(vocab.eos_token_id);

        sequence.push(next_id);
        if next_id != vocab.eos_token_id {
            emit_token(vocab, next_id);
        }
        if next_id == vocab.eos_token_id {
            println!("\n[EOS]");
            break;
        }
        if step + 1 < num_tokens && sequence.len() < max_ctx {
            logits = inference_result(bml_inference::forward_bml_cached(
                &mut hot,
                &compiler,
                &[next_id],
                &mut cache,
            ));
        }
    }

    println!(
        "\n\nGenerados {} tokens (modo BML) desde prompt '{}'",
        sequence.len() - prompt_ids.len(),
        prompt,
    );
}

fn encode_prompt(vocab: &bml_compiler::tokenizer::Vocabulary, prompt: &str) -> Vec<u32> {
    match vocab.encode(prompt) {
        Ok(ids) if !ids.is_empty() => ids,
        Ok(_) => {
            eprintln!("Error: el prompt no produce tokens");
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("Error tokenizando: {error}");
            std::process::exit(1);
        }
    }
}

fn emit_token(vocab: &bml_compiler::tokenizer::Vocabulary, id: u32) {
    use std::io::Write;
    let bytes = vocab.token_piece_bytes(id).unwrap_or_else(|error| {
        eprintln!("Error decodificando: {error}");
        std::process::exit(1);
    });
    let mut output = std::io::stdout().lock();
    if let Err(error) = output.write_all(&bytes).and_then(|_| output.flush()) {
        eprintln!("Error escribiendo salida: {error}");
        std::process::exit(1);
    }
}

fn inference_result<T>(result: Result<T, String>) -> T {
    result.unwrap_or_else(|error| {
        eprintln!("Error de inferencia: {error}");
        std::process::exit(1);
    })
}

#[cfg(test)]
mod artifact_detection_tests {
    use super::is_bmlgraph_dir;

    #[test]
    fn a_directory_requires_a_recognized_artifact_marker() {
        let root =
            std::env::temp_dir().join(format!("bml-artifact-detection-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
        assert!(!is_bmlgraph_dir(&root));
        assert!(!is_bmlgraph_dir(&root.join("missing")));
        let ordinary = root.join("model.gguf");
        std::fs::write(&ordinary, b"GGUF").unwrap();
        assert!(!is_bmlgraph_dir(&ordinary));
        for marker in [
            "header.bmlgraph",
            bml_compiler::causal_artifact::MANIFEST,
            bml_compiler::symbolic_artifact::MANIFEST,
        ] {
            let path = root.join(marker);
            std::fs::write(&path, b"{}").unwrap();
            assert!(is_bmlgraph_dir(&root), "marker: {marker}");
            std::fs::remove_file(path).unwrap();
            assert!(!is_bmlgraph_dir(&root));
        }
    }
}
