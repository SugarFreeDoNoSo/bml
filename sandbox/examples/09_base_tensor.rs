use bml::tensor::{apply_basis, Shape, Tensor};
use bml_domain::symbolic::Basis;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Tensor::from_vec(Shape::new([3])?, vec![1.0_f64, 2.0, 3.0])?;
    let y = Tensor::from_vec(Shape::new([3])?, vec![1.0_f64, 2.0, 4.0])?;
    for basis in [Basis::Bml, Basis::Eml] {
        println!(
            "{basis:?}: {:?}",
            apply_basis(&x.view(), &y.view(), basis)?.as_slice()
        );
    }
    let invalid = Tensor::from_vec(Shape::new([3])?, vec![1.0_f64, 0.0, 4.0])?;
    println!(
        "Error en el índice plano 1: {:?}",
        apply_basis(&x.view(), &invalid.view(), Basis::Bml)
    );
    println!("apply_basis aplica la primitiva; no sustituye add/mul/matmul.");
    Ok(())
}
