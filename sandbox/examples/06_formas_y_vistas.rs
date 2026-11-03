use bml::tensor::{Shape, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tensor = Tensor::from_vec(Shape::new([2, 3])?, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    println!(
        "Forma: {:?}; pasos: {:?}",
        tensor.shape().dims(),
        tensor.shape().strides()
    );
    println!("Valores por filas: {:?}", tensor.as_slice());
    let view = tensor.view();
    println!("Elemento [1,2]: {}", view.get(&[1, 2])?);
    let reshaped = view.reshape(Shape::new([3, 2])?)?;
    println!("Vista [3,2]: {:?}", reshaped.as_slice());
    println!(
        "Comparte memoria: {}",
        view.as_slice().as_ptr() == reshaped.as_slice().as_ptr()
    );
    println!("Índice fuera de rango: {:?}", view.get(&[2, 0]));
    println!("Cambio a 7 elementos: {:?}", view.reshape(Shape::new([7])?));
    Ok(())
}
