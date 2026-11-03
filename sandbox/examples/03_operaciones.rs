use bml::Bml;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Bml::from_f64(1.5)?;
    let y = Bml::from_f64(2.0)?;
    // Las operaciones construyen expresiones. to_f64 las evalúa después.
    // mul/div/recip/pow/sqrt requieren operandos positivos.
    for (label, result) in [
        ("x + y", x.add(&y)?),
        ("x - y", x.sub(&y)?),
        ("-x", x.neg()?),
        ("x * y", x.mul(&y)?),
        ("x / y", x.div(&y)?),
        ("1 / x", x.recip()?),
        ("x ^ y", x.pow(&y)?),
        ("sqrt(x)", x.sqrt()?),
        ("2 ^ x", x.exp2()?),
        ("log2(x)", x.log2()?),
    ] {
        println!(
            "{label:<8} = {:>12.8} ({} bits)",
            result.to_f64()?,
            result.bit_len()
        );
    }
    Ok(())
}
