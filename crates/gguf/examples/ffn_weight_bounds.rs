//! Scan a local Q4_0 model's down-projection weights; this does not execute inference.
use bml_gguf::{decode_f32, GgufDataType, GgufReader};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: ffn_weight_bounds MODEL LAYER_COUNT")?;
    let layers: usize = args.next().ok_or("missing layer count")?.parse()?;
    if layers == 0 || args.next().is_some() {
        return Err("expected one positive layer count".into());
    }
    let reader = GgufReader::open(path)?;
    for layer in 0..layers {
        let name = format!("blk.{layer}.ffn_down.weight");
        let info = reader.find_tensor(&name).ok_or("missing tensor")?;
        if info.data_type != GgufDataType::Q4_0 || info.dims.len() != 2 {
            return Err(format!("expected a Q4_0 matrix: {name}").into());
        }
        let bytes = reader.tensor_data(info).ok_or("missing tensor data")?;
        let count: usize = info.dims.iter().try_fold(1usize, |a, &b| {
            let dimension = usize::try_from(b).map_err(|_| "dimension overflow")?;
            a.checked_mul(dimension).ok_or("element count overflow")
        })?;
        let data = decode_f32(GgufDataType::Q4_0, bytes, count)?;
        if data.iter().any(|value| !value.is_finite()) {
            return Err(format!("non-finite decoded weight in {name}").into());
        }
        let max = data.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        println!("{name} max_abs={max}");
    }
    Ok(())
}
