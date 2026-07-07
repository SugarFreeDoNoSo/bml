use bml_compiler::gguf_compiler::compile_gguf;
use bml_compiler::hardware::HardwareSpec;
use bml_compiler::gguf_compiler::serialize_to_dir;
use std::path::Path;

fn main() {
    let gguf_path = "/root/tinyllama.gguf";
    let output_dir = Path::new("/root/tinyllama.bmlgraph");
    
    println!("Compiling {} -> {:?}", gguf_path, output_dir);
    let hw = HardwareSpec::detect_local();
    println!("Hardware: {:?}", hw);
    
    let result = compile_gguf(gguf_path, &hw).expect("compile failed");
    println!("Model: {} ({} layers, {} heads, {} embd)",
        result.config.architecture, result.config.n_layers, result.config.n_heads, result.config.n_embd);
    println!("Fragments: {}", result.num_fragments);
    println!("Const pool: {} values", result.const_pool.len());
    
    serialize_to_dir(&result, output_dir).expect("serialize failed");
    println!("Serialized to {:?}", output_dir);
    
    // Listar archivos
    for entry in std::fs::read_dir(output_dir).unwrap() {
        let entry = entry.unwrap();
        let meta = entry.metadata().unwrap();
        println!("  {:?} ({} bytes)", entry.file_name(), meta.len());
    }
}
