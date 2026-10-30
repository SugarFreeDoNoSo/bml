//! Import only a bounded F32 synthetic matrix fixture, not an inference graph.
use bml_gguf::{GgufDataType, GgufReader};
use std::io::Write;
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: synthetic_export FIXTURE.gguf WEIGHTS.bin".into());
    }
    let r = GgufReader::open(&a[1])?;
    if r.architecture() != Some("bml-synthetic-matrix") {
        return Err("not a synthetic matrix fixture".into());
    }
    let t = r.find_tensor("matrix.weight").ok_or("matrix missing")?;
    if t.data_type != GgufDataType::F32
        || t.dims.len() != 2
        || t.dims[0] != t.dims[1]
        || !(1..=128).contains(&t.dims[0])
    {
        return Err("only square F32 matrices of dimension 1..128 are supported".into());
    }
    let v = r.read_tensor_f32("matrix.weight")?;
    if v.iter().any(|x| !x.is_finite()) {
        return Err("non-finite matrix".into());
    }
    let mut f = std::fs::File::create(&a[2])?;
    for x in v {
        f.write_all(&x.to_le_bytes())?;
    }
    println!("{}", t.dims[0]);
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
