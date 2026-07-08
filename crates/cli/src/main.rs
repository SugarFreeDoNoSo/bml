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

use bml_compiler::gguf_compiler::{
    compile_gguf_fast, load_from_dir, serialize_to_dir, InferenceCompiler,
};
use bml_compiler::hardware::HardwareSpec;
use bml_compiler::sampler;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    },

    /// Ejecuta inferencia desde .bmlgraph o GGUF
    Run {
        /// Ruta al modelo: directorio .bmlgraph/ o archivo .gguf
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

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
    },
}

/// Determina si un path es un directorio .bmlgraph o un archivo GGUF.
fn is_bmlgraph_dir(path: &std::path::Path) -> bool {
    path.is_dir() && path.join("header.bmlgraph").exists()
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Compile {
            model,
            output,
            l1_threshold,
            l3_threshold,
        } => {
            if !model.exists() {
                eprintln!("Error: el archivo GGUF no existe: {}", model.display());
                std::process::exit(1);
            }

            println!("Compilando {} → {}", model.display(), output.display());
            let hw = HardwareSpec::new(4, l1_threshold, l1_threshold * 8, l1_threshold);
            println!("Hardware target: {:?}", hw);

            // compile_gguf_fast solo lee metadatos del GGUF, no carga pesos.
            let result = match compile_gguf_fast(&model, &hw) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error compilando: {e}");
                    std::process::exit(1);
                }
            };

            let config = &result.config;
            println!(
                "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
                config.architecture, config.n_layers, config.n_heads, config.n_embd, config.vocab_size,
            );
            println!("Fragmentos: {}", result.num_fragments);
            println!("Const pool: {} valores", result.const_pool.len());

            if let Err(e) = serialize_to_dir(&result, &output) {
                eprintln!("Error serializando: {e}");
                std::process::exit(1);
            }

            println!("Serializado a: {}", output.display());

            if let Ok(entries) = std::fs::read_dir(&output) {
                for entry in entries.flatten() {
                    if let Ok(m) = entry.metadata() {
                        println!("  {} ({} bytes)", entry.file_name().to_string_lossy(), m.len());
                    }
                }
            }

            println!("Serializado a: {}", output.display());

            if let Ok(entries) = std::fs::read_dir(&output) {
                for entry in entries.flatten() {
                    if let Ok(m) = entry.metadata() {
                        println!("  {} ({} bytes)", entry.file_name().to_string_lossy(), m.len());
                    }
                }
            }
        }

        Command::Run {
            model,
            prompt,
            num_tokens,
            threads: _,
            temp,
            context_size,
        } => {
            if !model.exists() {
                eprintln!("Error: el modelo no existe: {}", model.display());
                std::process::exit(1);
            }

            if is_bmlgraph_dir(&model) {
                run_from_bmlgraph(&model, &prompt, num_tokens, temp, context_size);
            } else {
                run_from_gguf(&model, &prompt, num_tokens, temp, context_size);
            }
        }
    }
}

/// Ejecuta inferencia desde un directorio .bmlgraph pre-compilado.
fn run_from_bmlgraph(
    bmlgraph_dir: &std::path::Path,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
    context_size: u32,
) {
    println!("Cargando .bmlgraph desde {}...", bmlgraph_dir.display());

    let (graph, const_pool, config) = match load_from_dir(bmlgraph_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error cargando .bmlgraph: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd, config.vocab_size,
    );
    println!("Fragmentos: {}", graph.num_fragments());
    println!("Const pool: {} valores", const_pool.len());
    println!("Prompt: {}", prompt);
    println!("Generando {} tokens (temp={})...\n", num_tokens, temp);

    // El camino .bmlgraph ejecuta el grafo BML compilado.
    // Esto ejecuta el hot loop RPN sobre los fragmentos.
    let mut runtime = bml_runtime::Runtime::new(8192, 64);

    let ctx = bml_domain::EvalContext::new(&[], &const_pool);

    // Warmup
    runtime.execute_graph_with_ctx(&graph, &ctx);

    // Medir tiempo de ejecución del grafo
    let start = std::time::Instant::now();
    let result = runtime.execute_graph_with_ctx(&graph, &ctx);
    let elapsed = start.elapsed();

    println!("Ejecución del grafo BML: {:.3?}", elapsed);
    println!("Resultado (f64): {:.6}", result);
    println!(
        "Fragmentos ejecutados: {} ({} ops total)",
        graph.num_fragments(),
        graph.total_byte_size(),
    );

    // Nota: el camino .bmlgraph ejecuta el DAG compilado del transformer
    // completo. La inferencia autoregresiva token-a-token requiere el
    // InferenceCompiler que tiene matmul, attention, etc. implementados
    // en f64 directo. El .bmlgraph es el grafo BML puro.
    //
    // Para generar texto real, usar `bml-cli run -m modelo.gguf`.
    let _ = num_tokens;
    let _ = temp;
    let _ = context_size;

    println!("\nNota: La generación autoregresiva de texto requiere el modo GGUF");
    println!("      (que usa InferenceCompiler con matmul/attention/RoPE en f64).");
    println!("      El modo .bmlgraph ejecuta el DAG BML compilado (operador puro).");
}

/// Ejecuta inferencia desde un archivo GGUF (compila en caliente).
fn run_from_gguf(
    gguf_path: &std::path::Path,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
    context_size: u32,
) {
    println!("Cargando modelo desde {}...", gguf_path.display());
    println!("Esto puede tardar para modelos grandes (dequantizando pesos)...");

    let compiler = match InferenceCompiler::open(gguf_path) {
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
        config.architecture, config.n_layers, config.n_heads, config.n_embd, vocab.len(),
    );
    println!(
        "Weight pool: {} valores ({:.1} MB)",
        compiler.weight_pool().len(),
        compiler.weight_pool().len() as f64 * 8.0 / (1024.0 * 1024.0),
    );
    println!("Prompt: {}", prompt);
    println!("Generando {} tokens (temp={})...\n", num_tokens, temp);

    let prompt_ids = vocab.encode(prompt);
    println!("Tokenizado: {} tokens", prompt_ids.len());

    let mut sequence = prompt_ids.clone();
    let max_ctx = context_size.min(config.context_length) as usize;

    for step in 0..num_tokens {
        let ctx_start = if sequence.len() > max_ctx {
            sequence.len() - max_ctx
        } else {
            0
        };
        let context = &sequence[ctx_start..];

        let logits = compiler.forward(context);

        let next_id = sampler::sample(&logits, temp, step as u64)
            .unwrap_or(vocab.eos_token_id);

        let token_text = vocab
            .decode_single(next_id)
            .strip_prefix('▁')
            .unwrap_or("<unk>");

        if token_text == "<|eot_id|>" || token_text == "</s>" {
            println!();
            break;
        }
        if token_text == "<0x0A>" {
            println!();
        } else {
            print!("{}", token_text);
        }
        use std::io::Write;
        std::io::stdout().flush().ok();

        sequence.push(next_id);

        if next_id == vocab.eos_token_id {
            println!("\n[EOS]");
            break;
        }
    }

    println!("\n");
    println!(
        "Generados {} tokens desde prompt '{}'",
        sequence.len() - prompt_ids.len(),
        prompt,
    );
}
