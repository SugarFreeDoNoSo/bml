use bml::Bml;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 0 es el terminal uno; 1 introduce una operación con dos operandos.
    // El entero codifica el árbol, NO su valor numérico.
    for bits in ["0", "100", "11000", "10100"] {
        let number = Bml::from_binary(bits)?;
        println!(
            "bits={bits:<5} entero={:?} longitud={} valor={}",
            number.to_u64(),
            number.bit_len(),
            number.to_f64()?
        );
        assert_eq!(Bml::from_words_le(number.to_words_le())?, number);
    }
    let two = Bml::from_u64(4)?; // 4 = 0b100, cuya evaluación es 2.
    println!("Bml::from_u64(4) → {} → {}", two.to_binary(), two.to_f64()?);
    println!("Código incompleto '10': {:?}", Bml::from_binary("10"));
    Ok(())
}
