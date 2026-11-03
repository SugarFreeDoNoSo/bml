use bml::gguf::{decode_f32, GgufDataType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let bytes: Vec<u8> = [1.5_f32, -2.25, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    println!("F32: {:?}", decode_f32(GgufDataType::F32, &bytes, 3)?);
    let half = [0x00, 0x3e, 0x80, 0xc0]; // 1.5 y -2.25 en binary16.
    println!("F16: {:?}", decode_f32(GgufDataType::F16, &half, 2)?);
    let mut q4 = vec![0x00, 0x3c]; // Escala F16 = 1.0; bloque de 32 elementos.
    q4.extend([0x98; 16]); // Mitad baja: 8-8=0; mitad alta: 9-8=1.
    println!("Q4_0: {:?}", decode_f32(GgufDataType::Q4_0, &q4, 32)?);
    let mut q8 = vec![0x00, 0x38]; // Escala F16 = 0.5.
    q8.extend((-16_i8..16).map(|value| value as u8));
    println!("Q8_0: {:?}", decode_f32(GgufDataType::Q8_0, &q8, 32)?);
    println!(
        "Bloque truncado: {:?}",
        decode_f32(GgufDataType::Q8_0, &q8[..33], 32)
    );
    Ok(())
}
