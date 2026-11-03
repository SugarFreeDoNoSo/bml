use bml::tensor::{add, dot, mul, Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_vec(Shape::new([3])?, vec![1.0_f64, 2.0, 3.0])?;
    let b = Tensor::from_vec(Shape::new([3])?, vec![4.0_f64, 5.0, 6.0])?;
    println!("a = {:?}; b = {:?}", a.as_slice(), b.as_slice());
    println!(
        "Suma elemento a elemento: {:?}",
        add(&a.view(), &b.view())?.as_slice()
    );
    println!(
        "Producto elemento a elemento: {:?}",
        mul(&a.view(), &b.view())?.as_slice()
    );
    println!("Producto escalar: {}", dot(&a.view(), &b.view())?);
    // Estos kernels usan IEEE nativo, sin las comprobaciones de symbolic.
    let huge = Tensor::from_vec(Shape::new([1])?, vec![f64::MAX])?;
    println!(
        "Overflow IEEE: {:?}",
        add(&huge.view(), &huge.view())?.as_slice()
    );
    Ok(())
}
