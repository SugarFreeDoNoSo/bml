use bml_model_weights::{convert, ConversionOptions, Error, ReadLimits, WeightReader};
use clap::Args;
use std::path::PathBuf;

/// Explicit numeric boundary for autonomous BML weights.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum WeightEvaluation {
    /// Require exact rational agreement between evaluated F32 and persisted BML
    CheckedExactF32,
}
impl From<WeightEvaluation> for bml_compiler::gguf_compiler::prepared_model::WeightEvaluation {
    fn from(value: WeightEvaluation) -> Self {
        match value {
            WeightEvaluation::CheckedExactF32 => Self::CheckedExactF32,
        }
    }
}

pub fn is_container(path: &std::path::Path) -> std::io::Result<bool> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0; 8];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(magic == bml_model_weights::MAGIC),
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(error) => Err(error),
    }
}

/// Independent reader/input ceilings. Defaults match the BMLGGUF v1 profile.
#[derive(Args, Debug)]
pub struct ReadArgs {
    #[arg(long, default_value_t = 68_719_476_736u64)]
    max_file_bytes: u64,
    #[arg(long, default_value_t = 67_108_864u64)]
    max_metadata_bytes: u64,
    #[arg(long, default_value_t = 65_536u64)]
    max_metadata_entries: u64,
    #[arg(long, default_value_t = 4_194_304u64)]
    max_metadata_array_elements: u64,
    #[arg(long, default_value_t = 4096u64)]
    max_name_bytes: u64,
    #[arg(long, default_value_t = 1_048_576u64)]
    max_string_bytes: u64,
    #[arg(long, default_value_t = 65_536u64)]
    max_tensors: u64,
    #[arg(long, default_value_t = 1_048_576u64)]
    max_blocks: u64,
    #[arg(long, default_value_t = 134_217_728u64)]
    max_index_bytes: u64,
    #[arg(long, default_value_t = 65_536u64)]
    max_block_terms: u64,
    #[arg(long, default_value_t = 8_388_608u64)]
    max_block_bytes: u64,
    #[arg(long, default_value_t = 1_048_576u64)]
    max_weight_bits: u64,
    #[arg(long, default_value_t = 4096u64)]
    max_weight_depth: u64,
    #[arg(long, default_value_t = 4_294_967_296u64)]
    max_total_terms: u64,
}
impl From<ReadArgs> for ReadLimits {
    fn from(a: ReadArgs) -> Self {
        Self {
            max_file_bytes: a.max_file_bytes,
            max_metadata_bytes: a.max_metadata_bytes,
            max_metadata_entries: a.max_metadata_entries,
            max_metadata_array_elements: a.max_metadata_array_elements,
            max_name_bytes: a.max_name_bytes,
            max_string_bytes: a.max_string_bytes,
            max_tensors: a.max_tensors,
            max_blocks: a.max_blocks,
            max_index_bytes: a.max_index_bytes,
            max_block_terms: a.max_block_terms,
            max_block_bytes: a.max_block_bytes,
            max_weight_bits: a.max_weight_bits,
            max_weight_depth: a.max_weight_depth,
            max_total_terms: a.max_total_terms,
        }
    }
}

#[derive(Args, Debug)]
pub struct ConvertArgs {
    #[arg(short = 'm', long = "model")]
    model: PathBuf,
    /// Destino nuevo; nunca se sobrescribe
    #[arg(short, long)]
    output: PathBuf,
    #[command(flatten)]
    read: ReadArgs,
    /// Pasos por peso; limita síntesis y replay independientemente
    #[arg(long, default_value_t = 4096u64)]
    max_synthesis_steps_per_weight: u64,
    /// Pasos acumulados; caché ahorra síntesis, pero no replay
    #[arg(long, default_value_t = 1_000_000u64)]
    max_synthesis_steps_total: u64,
    #[arg(long, default_value_t = 68_719_476_736u64)]
    max_output_bytes: u64,
    /// Memoria propia contabilizada conjuntamente; no es RSS ni incluye mmap
    #[arg(long, default_value_t = 268_435_456u64)]
    max_resident_bytes: u64,
    /// Cero desactiva caché de certificados
    #[arg(long, default_value_t = 0u64)]
    max_cache_bytes: u64,
    /// Bits retenidos durante síntesis por certificado
    #[arg(long, default_value_t = 16_777_216usize)]
    max_synthesis_total_bits: usize,
    #[arg(long, default_value_t = 128usize)]
    max_synthesis_recursion: usize,
}

#[derive(Args, Debug)]
pub struct InspectArgs {
    #[arg(short = 'm', long = "model")]
    model: PathBuf,
    /// Verifica hashes y gramática BMLS de todo el payload
    #[arg(long)]
    verify_all: bool,
    #[command(flatten)]
    read: ReadArgs,
}

pub fn exit_on_error(result: Result<(), Error>) {
    if let Err(error) = result {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

pub fn convert(args: ConvertArgs) -> Result<(), Error> {
    let mut options = ConversionOptions::default();
    options.write.limits = args.read.into();
    let limits = &mut options.write.conversion;
    limits.max_weight_bits = options.write.limits.max_weight_bits;
    limits.max_weight_depth = options.write.limits.max_weight_depth;
    limits.max_block_terms = options.write.limits.max_block_terms;
    limits.max_block_bytes = options.write.limits.max_block_bytes;
    limits.max_synthesis_steps_per_weight = args.max_synthesis_steps_per_weight;
    limits.max_synthesis_steps_total = args.max_synthesis_steps_total;
    limits.max_output_bytes = args.max_output_bytes;
    limits.max_resident_bytes = args.max_resident_bytes;
    limits.max_cache_bytes = args.max_cache_bytes;
    options.synthesis.max_bits = usize::try_from(limits.max_weight_bits).unwrap_or(usize::MAX);
    options.synthesis.max_steps =
        usize::try_from(limits.max_synthesis_steps_per_weight).unwrap_or(usize::MAX);
    options.synthesis.max_total_bits = args.max_synthesis_total_bits;
    options.synthesis.max_recursion = args.max_synthesis_recursion;
    println!(
        "limits: {:?}; synthesis: {:?}",
        options.write, options.synthesis
    );
    let report = convert::from_gguf(&args.model, &args.output, options)?;
    println!("published: {}", report.write.path.display());
    println!(
        "source_sha256: {}\noutput_sha256: {}",
        report.source_sha256, report.write.output_sha256
    );
    println!(
        "tensors: {}\nweights: {}\noutput_bytes: {}",
        report.write.summary.tensor_count,
        report.write.terms_written,
        report.write.summary.file_bytes
    );
    println!(
        "synthesis_steps: {}\nweights_reused: {}\npeak_accounted_bytes: {}\npeak_cache_bytes: {}",
        report.synthesis_steps,
        report.weights_reused,
        report.peak_resident_bytes,
        report.peak_cache_bytes
    );
    println!(
        "read_decode: {:?}\nsynthesis_certification: {:?}\nwrite_verify: {:?}",
        report.read_decode_time, report.synthesis_certification_time, report.write_verify_time
    );
    Ok(())
}

pub fn inspect(args: InspectArgs) -> Result<(), Error> {
    let limits: ReadLimits = args.read.into();
    println!("limits: {limits:?}");
    let mut reader = WeightReader::open(&args.model, limits)?;
    let summary = reader.summary();
    println!(
        "format: BMLGGUF v{}\ntensors: {}\nblocks: {}\nfile_bytes: {}\nsource_sha256: {}",
        summary.version,
        summary.tensor_count,
        summary.block_count,
        summary.file_bytes,
        summary.source_sha256
    );
    println!(
        "profile: {}\nmetadata_entries: {}",
        reader.metadata().profile,
        reader.metadata().gguf.len()
    );
    for tensor in reader.tensors() {
        println!(
            "tensor: {:?} ggml_dimensions={:?} source_dtype={:?} terms={} blocks={}",
            tensor.name,
            tensor.ggml_dimensions,
            tensor.source_dtype,
            tensor.element_count,
            tensor.block_count
        );
    }
    if args.verify_all {
        let report = reader.verify_all()?;
        println!(
            "payload: verified-all; blocks={} terms={} bytes={}",
            report.blocks_checked, report.terms_checked, report.bytes_checked
        );
    } else {
        println!("payload: not-verified (catalog validated; use --verify-all)");
    }
    Ok(())
}

#[cfg(test)]
mod detection_tests {
    use super::is_container;

    #[test]
    fn container_detection_distinguishes_magic_short_files_and_io_errors() {
        let root =
            std::env::temp_dir().join(format!("bml-container-detection-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        struct Cleanup(std::path::PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let _cleanup = Cleanup(root.clone());
        let file = root.join("weights");
        for size in 0..bml_model_weights::MAGIC.len() {
            std::fs::write(&file, &bml_model_weights::MAGIC[..size]).unwrap();
            assert!(!is_container(&file).unwrap(), "prefix length {size}");
        }
        std::fs::write(&file, bml_model_weights::MAGIC).unwrap();
        assert!(is_container(&file).unwrap());
        std::fs::write(&file, [0u8; 8]).unwrap();
        assert!(!is_container(&file).unwrap());
        assert_eq!(
            is_container(&root.join("missing")).unwrap_err().kind(),
            std::io::ErrorKind::NotFound
        );
        #[cfg(unix)]
        assert!(
            is_container(&root).is_err(),
            "a directory cannot be read as a weights file"
        );
    }
}
