use bml::tensor::{matmul, Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_vec(Shape::new([2, 3])?, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    let b = Tensor::from_vec(
        Shape::new([3, 2])?,
        vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    )?;
    let result = matmul(&a.view(), &b.view())?;
    println!("[2,3] × [3,2] → {:?}", result.shape().dims());
    for row in result.as_slice().chunks(2) {
        println!("{row:?}");
    }
    assert_eq!(result.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
    println!(
        "Dimensiones incompatibles: {:?}",
        matmul(&a.view(), &a.view())
    );
    Ok(())
}
