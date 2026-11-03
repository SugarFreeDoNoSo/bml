use bml::gguf::GgufReader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fixture sintética GGUF v3: un tensor F32 GGML [3,2], sin modelo externo.
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes()); // Un tensor.
    bytes.extend(0_u64.to_le_bytes()); // Sin metadatos.
    bytes.extend(5_u64.to_le_bytes());
    bytes.extend(b"pesos");
    bytes.extend(2_u32.to_le_bytes());
    bytes.extend(3_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    bytes.extend(0_u32.to_le_bytes()); // F32.
    bytes.extend(0_u64.to_le_bytes()); // Offset relativo.
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(
        [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .flat_map(f32::to_le_bytes),
    );
    let path = std::path::PathBuf::from(format!("fixture-{}.gguf", std::process::id()));
    std::fs::write(&path, bytes)?;
    {
        let reader = GgufReader::open(&path)?;
        let view = reader.tensor_view("pesos")?;
        println!("Tensor GGML: {:?}", reader.tensor_infos()[0].dims);
        let tensor = view.to_tensor_f32()?;
        println!("Forma row-major: {:?}", tensor.shape().dims());
        println!("Valores planos sin reordenar: {:?}", tensor.as_slice());
        assert_eq!(tensor.shape().dims(), &[2, 3]);
    } // El lector y sus vistas dejan de usar el mmap antes de borrar el archivo.
    std::fs::remove_file(path)?;
    Ok(())
}
