use bml_domain::symbolic::{Basis, ClosedTerm};

fn main() {
    // BML(x,y) = 2^x - log2(y), con comprobación del dominio real.
    for (x, y) in [
        (3.0, 4.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (2048.0, 1.0),
        (-2048.0, 1.0),
    ] {
        println!("BML({x}, {y}) = {:?}", Basis::Bml.apply_real(x, y));
    }
    let two = ClosedTerm::op(ClosedTerm::One, ClosedTerm::One);
    let four = ClosedTerm::op(two, ClosedTerm::One);
    let zero = ClosedTerm::op(ClosedTerm::One, four);
    let invalid = ClosedTerm::op(ClosedTerm::One, zero);
    println!(
        "Un árbol cerrado puede ser inválido al evaluar: {:?}",
        invalid.evaluate_real(Basis::Bml)
    );
}
