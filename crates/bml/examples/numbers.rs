use bml::Bml;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Bml::from_f64(1.5)?;
    let y = Bml::from_f64(2.0)?;
    let sum = x.add(&y)?;
    println!("{x} + {y} = {sum}");
    println!("Numerical value: {}", sum.to_f64()?);
    Ok(())
}
