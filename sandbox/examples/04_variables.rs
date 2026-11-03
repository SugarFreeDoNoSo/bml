use bml_domain::symbolic::{Basis, ClosedTerm, OpenTerm};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let expression = OpenTerm::op(OpenTerm::Var("x".into()), OpenTerm::Var("y".into()));
    println!("Variables libres: {:?}", expression.free_variables());
    let x = HashMap::from([("x".into(), ClosedTerm::One)]);
    let partial = expression.bind(&x);
    println!("Tras asignar x: {:?}", partial.free_variables());
    println!("Cerrar sin asignar y: {:?}", partial.clone().collapse());
    let y = HashMap::from([("y".into(), ClosedTerm::One)]);
    let closed = partial.bind(&y).collapse()?;
    println!(
        "Mismo árbol bajo BML: {}",
        closed.evaluate_real(Basis::Bml)?
    );
    println!(
        "Mismo árbol bajo EML: {}",
        closed.evaluate_real(Basis::Eml)?
    );
    println!("Original intacto: {:?}", expression.free_variables());
    Ok(())
}
