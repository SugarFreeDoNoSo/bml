//! Metadata and bounded F32 samples without expanding a synthetic GGUF tensor.
use bml_gguf::{GgufDataType, GgufReader};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 2 {
        return Err("usage: stream_fixture_probe FIXTURE.gguf".into());
    }
    let r = GgufReader::open(&a[1])?;
    if r.len() > 3_000_000_000
        || r.header().tensor_count != 1
        || r.architecture() != Some("bml-synthetic-matrix")
    {
        return Err("unexpected fixture/size".into());
    }
    let t = r.find_tensor("matrix.weight").ok_or("matrix missing")?;
    if t.data_type != GgufDataType::F32
        || t.dims.len() != 2
        || t.dims[0] != t.dims[1]
        || !(1..=20480).contains(&t.dims[0])
    {
        return Err("expected bounded square F32 matrix".into());
    }
    let raw = r.tensor_data(t).ok_or("invalid range")?;
    let offset = raw.as_ptr() as usize - r.bytes().as_ptr() as usize;
    let n = t.dims[0] as usize;
    let indices = [0, n - 1, n * n / 2, n * n - 1];
    let mut samples = Vec::new();
    for i in indices {
        let v = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into()?);
        if !v.is_finite() {
            return Err("non-finite sample".into());
        }
        samples.push(v.to_string());
    }
    println!(
        "{{\"n\":{n},\"offset\":{offset},\"bytes\":{},\"samples\":[{}]}}",
        raw.len(),
        samples.join(",")
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}
