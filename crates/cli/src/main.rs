//! # bml-cli
//!
//! CLI compatible con llama.cpp para ejecutar inferencia BML local.
//!
//! # Uso
//!
//! ```sh
//! bml-cli -m /root/tinyllama.gguf -p "Hello" -n 10
//! ```

use bml_compiler::gguf_compiler::InferenceCompiler;
use bml_compiler::sampler;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "bml-cli",
    version,
    about = "BML inference CLI (llama.cpp compatible)"
)]
struct Args {
    #[arg(short = 'm', long = "model")]
    model: String,
    #[arg(short = 'p', long = "prompt")]
    prompt: String,
    #[arg(short = 'n', long = "num-tokens", default_value_t = 128)]
    num_tokens: u32,
    #[arg(short = 't', long = "threads", default_value_t = 4)]
    threads: u32,
    #[arg(long = "temp", default_value_t = 0.8)]
    temp: f64,
    #[arg(short = 'c', long = "context-size", default_value_t = 2048)]
    context_size: u32,
}

fn main() {
    let args = Args::parse();

    println!("Loading model from {}...", args.model);
    println!("This may take a while for large models (dequantizing weights)...");

    let compiler = match InferenceCompiler::open(&args.model) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error loading model: {e}");
            std::process::exit(1);
        }
    };

    let config = compiler.config();
    let vocab = compiler.vocab();
    println!(
        "Model: {} ({} layers, {} heads, {} embd, {} vocab)",
        config.architecture,
        config.n_layers,
        config.n_heads,
        config.n_embd,
        vocab.len(),
    );
    println!(
        "Weight pool: {} values ({:.1} MB)",
        compiler.weight_pool().len(),
        compiler.weight_pool().len() as f64 * 8.0 / (1024.0 * 1024.0)
    );
    println!("Prompt: {}", args.prompt);
    println!(
        "Generating {} tokens (temp={})...",
        args.num_tokens, args.temp
    );
    println!();

    let prompt_ids = vocab.encode(&args.prompt);
    println!("Tokenized: {} tokens", prompt_ids.len());

    let mut sequence = prompt_ids.clone();
    let max_ctx = args.context_size.min(512) as usize;

    for step in 0..args.num_tokens {
        // Truncar al context size
        let ctx_start = if sequence.len() > max_ctx {
            sequence.len() - max_ctx
        } else {
            0
        };
        let context = &sequence[ctx_start..];

        // Forward pass
        let logits = compiler.forward(context);

        // Sample next token
        let next_id = sampler::sample(&logits, args.temp, step as u64)
            .unwrap_or(vocab.eos_token_id);

        // Print token text
        let token_text = vocab.decode_single(next_id)
            .strip_prefix('▁')
            .unwrap_or("<unk>");

        if token_text == "<|eot_id|>" || token_text == "<|endoftext|>" {
            print!("\n");
            break;
        }
        if token_text == "<0x0A>" {
            println!();
        } else {
            print!("{}", token_text);
        }
        std::io::Write::flush(&mut std::io::stdout()).ok();

        sequence.push(next_id);

        // Stop on EOS
        if next_id == vocab.eos_token_id {
            println!("\n[EOS]");
            break;
        }
    }

    println!();
    println!();
    println!(
        "Generated {} tokens from prompt '{}'",
        sequence.len() - prompt_ids.len(),
        args.prompt
    );
}