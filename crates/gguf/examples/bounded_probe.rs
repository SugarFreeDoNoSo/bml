//! Bounded Q4_0 byte/decode/projection probe; no model inference.
use bml_gguf::{decode_f32, GgufDataType, GgufReader};
fn hex(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: bounded_probe MODEL TENSOR ROW_START ROWS BLOCKS".into());
    }
    let start: usize = a[3].parse()?;
    let rows: usize = a[4].parse()?;
    let blocks: usize = a[5].parse()?;
    if !(1..=8).contains(&rows) || !(1..=128).contains(&blocks) {
        return Err("rows must be 1..8 and blocks 1..128".into());
    }
    let reader = GgufReader::open(&a[1])?;
    let info = reader.find_tensor(&a[2]).ok_or("tensor not found")?;
    if info.dims.len() != 2 || info.data_type != GgufDataType::Q4_0 {
        return Err("probe requires a rank-2 Q4_0 tensor".into());
    }
    let width = usize::try_from(info.dims[0])?;
    let height = usize::try_from(info.dims[1])?;
    let elements = blocks.checked_mul(32).ok_or("overflow")?;
    if !width.is_multiple_of(32)
        || elements > width
        || start.checked_add(rows).ok_or("overflow")? > height
    {
        return Err("sample exceeds tensor dimensions".into());
    }
    let bytes = reader.tensor_data(info).ok_or("invalid tensor range")?;
    let stride = (width / 32).checked_mul(18).ok_or("overflow")?;
    println!(
        "metadata\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        hex(reader.architecture().unwrap_or("unknown").as_bytes()),
        reader.header().version,
        reader.header().metadata_kv_count,
        reader.header().tensor_count,
        width,
        height,
        reader.len()
    );
    for row in start..start + rows {
        let begin = row.checked_mul(stride).ok_or("overflow")?;
        let raw = bytes
            .get(begin..begin + blocks * 18)
            .ok_or("sample out of range")?;
        let decoded = decode_f32(GgufDataType::Q4_0, raw, elements)?;
        if decoded.iter().any(|v| !v.is_finite()) {
            return Err("non-finite decoded value".into());
        }
        let dot: f64 = decoded
            .iter()
            .enumerate()
            .map(|(i, v)| f64::from(*v) * ((i % 17) as f64 - 8.0) / 8.0)
            .sum();
        println!(
            "row\t{row}\t{}\t{}\t{dot}",
            hex(raw),
            decoded
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("bounded_probe: {e}");
        std::process::exit(2);
    }
}
