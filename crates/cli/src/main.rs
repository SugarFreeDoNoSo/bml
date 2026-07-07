//! # bml-cli
//!
//! CLI compatible con llama.cpp para ejecutar inferencia BML local.
//!
//! # Uso
//!
//! ```sh
//! bml-cli -m model.bmlgraph/ -p "Hello" -n 10
//! ```

use bml_compiler::gguf_compiler::load_from_dir;
use bml_runtime::Runtime;
use clap::Parser;

/// BML CLI — ejecuta inferencia local con un modelo .bmlgraph.
#[derive(Parser, Debug)]
#[command(
    name = "bml-cli",
    version,
    about = "BML inference CLI (llama.cpp compatible)"
)]
struct Args {
    /// Ruta al directorio .bmlgraph.
    #[arg(short = 'm', long = "model")]
    model: String,

    /// Prompt de entrada.
    #[arg(short = 'p', long = "prompt")]
    prompt: String,

    /// Número de tokens a generar.
    #[arg(short = 'n', long = "num-tokens", default_value_t = 128)]
    num_tokens: u32,

    /// Número de threads.
    #[arg(short = 't', long = "threads", default_value_t = 4)]
    threads: u32,

    /// Temperatura de sampling (0 = greedy).
    #[arg(long = "temp", default_value_t = 0.8)]
    temp: f64,

    /// Tamaño de contexto.
    #[arg(short = 'c', long = "context-size", default_value_t = 2048)]
    context_size: u32,
}

fn main() {
    let args = Args::parse();

    println!("Loading model from {}...", args.model);
    let (graph, const_pool, config) = match load_from_dir(std::path::Path::new(&args.model)) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Model: {} ({} layers, {} heads, {} embd)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd
    );
    println!("Fragments: {}", graph.num_fragments());
    println!("Const pool: {} values", const_pool.len());
    println!();

    // Crear runtime
    let mut runtime = Runtime::new(8192, 64);

    // Ejecutar inferencia (placeholder)
    println!("Prompt: {}", args.prompt);
    println!(
        "Generating {} tokens (temp={})...",
        args.num_tokens, args.temp
    );

    // Ejecutar el grafo con el prompt como input
    let inputs = vec![args.prompt.len() as f64];
    let ctx = bml_domain::EvalContext::new(&inputs, &const_pool);
    let result = runtime.execute_graph_with_ctx(&graph, &ctx);

    println!();
    println!("Result (raw): {result}");
    println!();
    println!(
        "{} [BML placeholder: {} tokens generated]",
        args.prompt, args.num_tokens
    );
}
