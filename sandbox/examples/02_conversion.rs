use bml::{Bml, SynthesisOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Cambia los valores: la síntesis es acotada y puede devolver un error.
    for input in [0.0, 1.0, -1.0, 0.5, 1.5, 2.0] {
        let number = Bml::from_f64(input)?;
        let actual = number.to_f64()?;
        println!(
            "entrada={input:>5} salida={actual:>12.8} error={:.3e} bits={}",
            (input - actual).abs(),
            number.bit_len()
        );
    }
    let options = SynthesisOptions {
        max_bits: 1,
        ..SynthesisOptions::default()
    };
    println!(
        "Intentar representar 2 con un solo bit: {:?}",
        Bml::from_f64_with_options(2.0, options)
    );
    Ok(())
}
