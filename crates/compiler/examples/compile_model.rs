use bml_compiler::gguf_compiler::compile_gguf_with_limits;
use bml_compiler::hardware::HardwareSpec;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let gguf_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        eprintln!("Uso: compile_model <modelo.gguf> <context_size> [output_dir]");
        std::process::exit(1);
    };
    let context = args
        .get(2)
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            eprintln!("context_size debe ser un entero positivo explícito");
            std::process::exit(1);
        });
    let output_dir = if args.len() > 3 {
        PathBuf::from(&args[3])
    } else {
        gguf_path.with_extension("bmlgraph")
    };

    if !gguf_path.exists() {
        eprintln!("Error: {} no existe", gguf_path.display());
        std::process::exit(1);
    }

    println!(
        "Compiling {} -> {}",
        gguf_path.display(),
        output_dir.display()
    );
    let hw = HardwareSpec::detect_local();
    println!("Hardware: {:?}", hw);

    let result = compile_gguf_with_limits(&gguf_path, &hw, context, Default::default())
        .expect("compile failed");
    let manifest = result.manifest();
    println!(
        "Model: {} ({} layers, {} heads, {} embd)",
        manifest.config.architecture,
        manifest.config.n_layers,
        manifest.config.n_heads,
        manifest.config.n_embd
    );
    println!(
        "Profile: {}; kernels: {}; context: {}",
        manifest.profile,
        manifest.kernels.len(),
        manifest.context_length
    );

    result.publish(&output_dir).expect("serialize failed");
    println!("Serialized to {}", output_dir.display());

    for entry in std::fs::read_dir(&output_dir).unwrap() {
        let entry = entry.unwrap();
        let meta = entry.metadata().unwrap();
        println!(
            "  {} ({} bytes)",
            entry.file_name().to_string_lossy(),
            meta.len()
        );
    }
}
