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

use bml_compiler::gguf_compiler::InferenceCompiler;
use bml_compiler::sampler;
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

        /// Compila un artefacto BML acotado para sesiones distribuidas
        #[arg(long = "distributed")]
        distributed: bool,

        /// Compilar aritmética BML completa (experimental); requiere contexto explícito.
        #[arg(long, conflicts_with = "distributed", requires = "context_size")]
        symbolic: bool,

        /// Contexto máximo que cubrirá el artefacto AOT --symbolic.
        #[arg(long = "context-size")]
        context_size: Option<u32>,

        /// Perfil RMS acotado experimental: máximo absoluto de activaciones.
        #[arg(long, requires_all = ["symbolic", "rms_weight_bound"])]
        rms_input_bound: Option<f64>,

        /// Perfil RMS acotado experimental: máximo absoluto de pesos RMS.
        #[arg(long, requires_all = ["symbolic", "rms_input_bound"])]
        rms_weight_bound: Option<f64>,

        /// Perfil acotado completo v4 con comprobación de cada operación BML.
        #[arg(long, requires = "symbolic", conflicts_with_all = ["rms_input_bound", "rms_weight_bound"])]
        bounded: bool,

        /// Presupuesto diagnóstico explícito: 2M nodos/ops por programa, 4M ops totales.
        #[arg(long)]
        symbolic_large_budget: bool,

        /// Parámetro histórico; los rangos se asignan al arrancar los workers
        #[arg(long = "layers-per-fragment", default_value_t = 1)]
        layers_per_fragment: u32,

        /// Inclusive RMS input bound for the distributed bounded profile.
        #[arg(long, requires_all = ["distributed", "distributed_rms_weight_bound"])]
        distributed_rms_input_bound: Option<f64>,
        /// Inclusive learned RMS weight bound for the distributed bounded profile.
        #[arg(long, requires_all = ["distributed", "distributed_rms_input_bound"])]
        distributed_rms_weight_bound: Option<f64>,
        /// Inclusive projection input bound for the distributed profile.
        #[arg(long, requires_all = ["distributed", "distributed_projection_weight_bound"])]
        distributed_projection_input_bound: Option<f64>,
        /// Inclusive projection weight bound for the distributed profile.
        #[arg(long, requires_all = ["distributed", "distributed_projection_input_bound"])]
        distributed_projection_weight_bound: Option<f64>,
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

        /// Presupuesto simbólico diagnóstico: 2M ops por programa y 4M pasos/valores.
        #[arg(long)]
        symbolic_large_budget: bool,
        /// Límite explícito de pasos del grafo para modelos grandes.
        #[arg(long, requires = "symbolic_large_budget", value_parser = clap::value_parser!(u64).range(1..))]
        symbolic_max_steps: Option<u64>,
    },

    /// Exporta los pesos codificados de un rango para workers sin GGUF.
    ExportShard {
        /// Directorio del artefacto simbólico compilado.
        #[arg(short = 'm', long = "model")]
        model: PathBuf,
        /// Directorio nuevo para el paquete de pesos.
        #[arg(short = 'o', long = "output")]
        output: PathBuf,
        /// GGUF reubicado cuya identidad debe coincidir con el manifiesto.
        #[arg(long = "weights")]
        weights: Option<PathBuf>,
        /// Primera capa incluida.
        #[arg(long = "layer-start")]
        layer_start: u32,
        /// Límite exclusivo de las capas.
        #[arg(long = "layer-end")]
        layer_end: u32,
        /// Máximo de bytes codificados en el paquete.
        #[arg(long = "max-bytes", default_value_t = 536_870_912, value_parser = clap::value_parser!(u64).range(1..))]
        max_bytes: u64,
    },
    /// Distribuye fragmentos a nodos workers via TCP
    Distribute {
        /// Directorio AOT simbólico acotado con pesos GGUF enlazados
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        /// Lista de nodos workers (host:port,host:port,...)
        #[arg(long = "nodes")]
        nodes: String,
        /// Relocated GGUF weights; its hash must match the artifact.
        #[arg(long = "weights")]
        weights: Option<PathBuf>,
        /// Use the explicit larger symbolic execution budgets on the coordinator.
        #[arg(long)]
        symbolic_large_budget: bool,
        /// Límite explícito de pasos del grafo del coordinador.
        #[arg(long, requires = "symbolic_large_budget", value_parser = clap::value_parser!(u64).range(1..))]
        symbolic_max_steps: Option<u64>,
        /// Read/write timeout for every worker connection in seconds.
        #[arg(long = "timeout-seconds", default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
        timeout_seconds: u64,

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
        Command::ExportShard {
            model,
            output,
            weights,
            layer_start,
            layer_end,
            max_bytes,
        } => {
            let before = inference_result(bml_compiler::symbolic_artifact::checked_identity(
                &model,
                weights.as_deref(),
            ));
            let loaded = inference_result(bml_compiler::symbolic_artifact::load_directory(
                &model,
                weights.as_deref(),
            ));
            if loaded.manifest.profile != bml_compiler::symbolic_artifact::BOUNDED_PROFILE {
                eprintln!("export-shard requires the bounded symbolic v4 profile");
                std::process::exit(1);
            }

            let shard =
                inference_result(bml_compiler::symbolic_weight_shard::export_layer_weights(
                    &model,
                    weights.as_deref(),
                    layer_start..layer_end,
                    &output,
                    max_bytes,
                ));
            let after = inference_result(bml_compiler::symbolic_artifact::checked_identity(
                &model,
                weights.as_deref(),
            ));
            if before != after {
                eprintln!("artifact changed while exporting shard; discard the output");
                std::process::exit(1);
            }
            println!("artifact_sha256={before}");
            println!("shard_sha256={shard}");
            println!("layers={layer_start}..{layer_end}");
        }

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
            bounded,
            symbolic_large_budget,
            layers_per_fragment,
            distributed_rms_input_bound,
            distributed_rms_weight_bound,
            distributed_projection_input_bound,
            distributed_projection_weight_bound,
        } => {
            if !model.exists() {
                eprintln!("Error: el archivo GGUF no existe: {}", model.display());
                std::process::exit(1);
            }

            println!("Compilando {} → {}", model.display(), output.display());

            if context_size.is_some() && !distributed && !symbolic {
                eprintln!("--context-size requiere --symbolic o --distributed");
                std::process::exit(2);
            }
            if symbolic_large_budget && !distributed && !symbolic {
                eprintln!("--symbolic-large-budget requiere --symbolic o --distributed");
                std::process::exit(2);
            }
            if distributed {
                let Some(context) = context_size else {
                    eprintln!("--distributed requiere --context-size explícito");
                    std::process::exit(2);
                };
                if layers_per_fragment != 1 {
                    eprintln!("--layers-per-fragment es histórico; configure rangos al arrancar los workers");
                    std::process::exit(2);
                }
                let mut limits = bml_compiler::symbolic_kernels::KernelLimits {
                    fragment_bytes: l1_threshold,
                    ..Default::default()
                };
                if symbolic_large_budget {
                    limits.max_nodes = 2_000_000;
                    limits.max_ops = 4_000_000;
                    limits.max_output_ops = 2_000_000;
                    limits.max_programs = 4096;
                }
                let mut profile = bml_compiler::symbolic_bounded::BoundedProfile::default();
                if let Some(input) = distributed_rms_input_bound {
                    profile.rms.input = input;
                    profile.rms.weight =
                        distributed_rms_weight_bound.expect("clap requires both RMS bounds");
                }
                if let Some(input) = distributed_projection_input_bound {
                    profile.projection_input = input;
                    profile.projection_weight = distributed_projection_weight_bound
                        .expect("clap requires both projection bounds");
                }
                let manifest =
                    inference_result(bml_compiler::symbolic_artifact::compile_directory_bounded(
                        &model, &output, context, profile, limits,
                    ));
                println!(
                    "Artefacto distribuido simbólico: {} (contexto {})",
                    manifest.profile, context
                );
            } else if symbolic {
                let mut limits = bml_compiler::symbolic_kernels::KernelLimits {
                    fragment_bytes: l1_threshold,
                    ..Default::default()
                };
                if symbolic_large_budget {
                    limits.max_nodes = 2_000_000;
                    limits.max_ops = 4_000_000;
                    limits.max_output_ops = 2_000_000;
                    limits.max_programs = 4096;
                }
                let context = context_size.expect("clap requires symbolic context");
                let manifest = inference_result(if bounded {
                    bml_compiler::symbolic_artifact::compile_directory_bounded(
                        &model,
                        &output,
                        context,
                        Default::default(),
                        limits,
                    )
                } else if let Some(input) = rms_input_bound {
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
            symbolic_large_budget,
            symbolic_max_steps,
        } => {
            if symbolic_large_budget
                && weight_evaluation.is_none()
                && !model
                    .join(bml_compiler::symbolic_artifact::MANIFEST)
                    .is_file()
            {
                eprintln!(
                    "--symbolic-large-budget requiere un artefacto simbólico o evaluación BMLGGUF"
                );
                std::process::exit(1);
            }
            let symbolic_limits = symbolic_limits(symbolic_large_budget, symbolic_max_steps);
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
                        symbolic_limits,
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
                        symbolic_limits,
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
            weights,
            symbolic_large_budget,
            symbolic_max_steps,
            timeout_seconds,
            prompt,
            num_tokens,
            temp,
        } => {
            if let Err(error) = distribute(
                &model,
                weights.as_deref(),
                &nodes,
                &prompt,
                num_tokens,
                temp,
                symbolic_large_budget,
                symbolic_max_steps,
                timeout_seconds,
            ) {
                eprintln!("Error de inferencia distribuida: {error}");
                std::process::exit(1);
            }
        }
    }
}

enum TextArtifact {
    Native(Box<bml_runtime::causal_artifact::CompiledCausalModel>),
    Symbolic(Box<bml_runtime::symbolic_causal::SymbolicCausalModel>),
}
fn symbolic_limits(
    large: bool,
    max_steps: Option<u64>,
) -> bml_runtime::symbolic_causal::SymbolicLimits {
    let mut limits = if large {
        bml_runtime::symbolic_causal::SymbolicLimits {
            max_nodes: 2_000_000,
            max_cached_ops: 4_000_000,
            max_programs: 4096,
            graph: bml_runtime::graph::GraphLimits {
                max_ops: 2_000_000,
                max_steps: 4_000_000,
                max_values: 4_000_000,
            },
        }
    } else {
        Default::default()
    };
    if let Some(steps) = max_steps {
        limits.graph.max_steps = usize::try_from(steps).unwrap_or_else(|_| {
            eprintln!("symbolic step budget overflow");
            std::process::exit(2);
        });
    }
    limits
}
impl TextArtifact {
    fn open(
        directory: &std::path::Path,
        source: Option<&std::path::Path>,
        limits: bml_runtime::symbolic_causal::SymbolicLimits,
    ) -> Result<Self, String> {
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
                    directory, source, limits,
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
    limits: bml_runtime::symbolic_causal::SymbolicLimits,
) {
    let model = inference_result(TextArtifact::open(directory, source, limits));
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

/// Execute a persisted bounded symbolic artifact over configured workers.
/// Legacy distributed fragments lack complete causal programs and are rejected.
#[allow(clippy::too_many_arguments)]
fn distribute(
    directory: &std::path::Path,
    source: Option<&std::path::Path>,
    nodes: &str,
    prompt: &str,
    num_tokens: u32,
    temperature: f64,
    large_budget: bool,
    max_steps: Option<u64>,
    timeout_seconds: u64,
) -> Result<(), String> {
    use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
    use bml_runtime::symbolic_transport::{
        SymbolicLayerClient, SymbolicPipeline, SymbolicTransportLimits,
    };
    use std::io::Write;
    if !temperature.is_finite() || temperature < 0.0 {
        return Err("temperatura inválida".into());
    }
    if !directory
        .join(bml_compiler::symbolic_artifact::MANIFEST)
        .is_file()
    {
        return Err("se requiere un artefacto simbólico persistido; los fragmentos distribuidos históricos están incompletos".into());
    }
    let nodes: Vec<_> = nodes.split(',').map(str::trim).collect();
    if nodes.is_empty() || nodes.iter().any(|node| node.is_empty()) {
        return Err("lista de nodos inválida".into());
    }
    let limits = SymbolicLimits {
        max_nodes: 0,
        ..symbolic_limits(large_budget, max_steps)
    };
    let identity = bml_compiler::symbolic_artifact::checked_identity(directory, source)?;
    let model = SymbolicCausalModel::open_artifact(directory, source, limits)?;
    let vocab = model.compiler().vocab().clone();
    let ids = vocab
        .encode(prompt)
        .map_err(|error| format!("tokenización: {error}"))?;
    if ids.is_empty() {
        return Err("el prompt no produce tokens".into());
    }
    let required = ids
        .len()
        .checked_add(num_tokens.saturating_sub(1) as usize)
        .ok_or("contexto desbordado")?;
    let context = u32::try_from(required).map_err(|_| "contexto desbordado")?;
    if context > model.context_limit() {
        return Err("el prompt y la generación exceden el contexto compilado".into());
    }
    let layers = model.compiler().config().n_layers;
    let active = nodes.len().min(layers as usize);
    let mut clients = Vec::with_capacity(active);
    let transport_limits = SymbolicTransportLimits {
        timeout: std::time::Duration::from_secs(timeout_seconds),
        ..Default::default()
    };
    for (index, node) in nodes.iter().take(active).enumerate() {
        let start = (index as u64 * u64::from(layers) / active as u64) as u32;
        let end = ((index as u64 + 1) * u64::from(layers) / active as u64) as u32;
        clients.push(
            SymbolicLayerClient::connect(*node, &identity, start..end, context, transport_limits)
                .map_err(|error| format!("worker {node} capas {start}..{end}: {error}"))?,
        );
    }
    drop(model);
    let mut pipeline = SymbolicPipeline::open(directory, source, context, clients, limits)?;
    if num_tokens == 0 {
        pipeline.close()?;
        return Ok(());
    }
    let mut logits = Vec::new();
    for id in ids {
        logits = pipeline.forward(id)?;
    }
    let mut output = std::io::stdout().lock();
    for step in 0..num_tokens {
        let token =
            sampler::sample(&logits, temperature, u64::from(step)).ok_or("sampling inválido")?;
        if token == vocab.eos_token_id {
            break;
        }
        let piece = vocab
            .token_piece_bytes(token)
            .map_err(|error| format!("detokenización: {error}"))?;
        output
            .write_all(&piece)
            .and_then(|_| output.flush())
            .map_err(|error| error.to_string())?;
        if step + 1 < num_tokens {
            logits = pipeline.forward(token)?;
        }
    }
    pipeline.close()?;
    Ok(())
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
