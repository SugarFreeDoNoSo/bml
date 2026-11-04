//! Scan a local Llama model's normalization weights; this does not execute inference.
use bml_gguf::{decode_f32, GgufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: rms_weight_bounds MODEL LAYER_COUNT")?;
    let layers: usize = args.next().ok_or("missing layer count")?.parse()?;
    if layers == 0 || args.next().is_some() {
        return Err("expected one positive layer count".into());
    }
    let reader = GgufReader::open(path)?;
    let mut names = Vec::new();
    for layer in 0..layers {
        names.push(format!("blk.{layer}.attn_norm.weight"));
        names.push(format!("blk.{layer}.ffn_norm.weight"));
    }
    names.push("output_norm.weight".into());
    for name in names {
        let info = reader
            .find_tensor(&name)
            .ok_or("missing normalization tensor")?;
        if info.dims.len() != 1 {
            return Err(format!("expected normalization vector: {name}").into());
        }
        let count = usize::try_from(info.dims[0])?;
        let data = decode_f32(
            info.data_type,
            reader.tensor_data(info).ok_or("missing tensor data")?,
            count,
        )?;
        if data.iter().any(|value| !value.is_finite()) {
            return Err(format!("non-finite decoded weight in {name}").into());
        }
        let (index, max) =
            data.iter()
                .enumerate()
                .fold((0usize, 0.0_f32), |(index, max), (i, value)| {
                    if value.abs() > max {
                        (i, value.abs())
                    } else {
                        (index, max)
                    }
                });
        println!("{name} max_abs={max} index={index}");
    }
    Ok(())
}
