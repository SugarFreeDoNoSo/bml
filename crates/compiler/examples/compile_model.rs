use bml_compiler::gguf_compiler::compile_gguf_fast;
use bml_compiler::hardware::HardwareSpec;
use bml_compiler::gguf_compiler::serialize_to_dir;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gguf_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        eprintln!("Uso: compile_model <modelo.gguf> [output_dir]");
        std::process::exit(1);
    };
    let output_dir = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        gguf_path.with_extension("bmlgraph")
    };

    if !gguf_path.exists() {
        eprintln!("Error: {} no existe", gguf_path.display());
        std::process::exit(1);
    }

    println!("Compiling {} -> {}", gguf_path.display(), output_dir.display());
    let hw = HardwareSpec::detect_local();
    println!("Hardware: {:?}", hw);

    let result = compile_gguf_fast(&gguf_path, &hw).expect("compile failed");
    println!("Model: {} ({} layers, {} heads, {} embd)",
        result.config.architecture, result.config.n_layers, result.config.n_heads, result.config.n_embd);
    println!("Fragments: {}", result.num_fragments);
    println!("Const pool: {} values", result.const_pool.len());

    serialize_to_dir(&result, &output_dir).expect("serialize failed");
    println!("Serialized to {}", output_dir.display());

    for entry in std::fs::read_dir(&output_dir).unwrap() {
        let entry = entry.unwrap();
        let meta = entry.metadata().unwrap();
        println!("  {} ({} bytes)", entry.file_name().to_string_lossy(), meta.len());
    }
}
