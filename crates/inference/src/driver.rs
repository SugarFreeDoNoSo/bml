//! Command-line entrypoints shared by the compiler and native model artifacts.

use crate::model::{Model, Operation, Session};
use crate::resources::{append_to_executable, write_model, write_model_file, Artifact};
use crate::{Error, Result};
use bml_gguf::GgufReader;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

const USAGE: &str = "bml-model <inspect|run|export|compile> --model FILE [options]\n\
run: --prompt TEXT | --tokens ID,ID  --predict N --context N --logits\n\
export/compile: --output FILE (must not exist)\n\
compile: requires this BML source distribution and the pinned Rust toolchain;\n\
the resulting Linux x86_64 executable requires neither at runtime.";

const MAX_DIAGNOSTIC_BYTES: usize = 64 * 1024 * 1024;

fn process_arguments() -> Result<std::vec::IntoIter<String>> {
    std::env::args_os()
        .skip(1)
        .map(|argument| {
            argument
                .into_string()
                .map_err(|_| Error("arguments must be valid UTF-8".into()))
        })
        .collect::<Result<Vec<_>>>()
        .map(Vec::into_iter)
}

#[derive(Default)]
struct Arguments {
    values: BTreeMap<String, String>,
    logits: bool,
}

impl Arguments {
    fn parse(args: impl Iterator<Item = String>, allowed: &[&str]) -> Result<Self> {
        let mut result = Self::default();
        let mut args = args.peekable();
        while let Some(key) = args.next() {
            if key == "--logits" && allowed.contains(&"--logits") {
                if result.logits {
                    return Err(Error("repeated argument: --logits".into()));
                }
                result.logits = true;
                continue;
            }
            if !allowed.contains(&key.as_str()) || result.values.contains_key(&key) {
                return Err(Error(format!("unknown or repeated argument: {key}")));
            }
            let value = args
                .next()
                .ok_or_else(|| Error(format!("missing value for {key}")))?;
            result.values.insert(key, value);
        }
        Ok(result)
    }

    fn required(&self, key: &str) -> Result<&str> {
        self.values
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| Error(format!("required argument: {key}")))
    }

    fn number(&self, key: &str, default: usize) -> Result<usize> {
        self.values.get(key).map_or(Ok(default), |value| {
            value
                .parse()
                .map_err(|_| Error(format!("invalid nonnegative integer for {key}")))
        })
    }
}

#[derive(Serialize)]
struct RunOutput {
    prompt_tokens: Vec<u32>,
    tokens: Vec<u32>,
    eos_id: u32,
    text: String,
    bytes: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logits: Option<Vec<Vec<f32>>>,
    context: usize,
    kv_bytes: usize,
    prefill_ms: f64,
    decode_ms: f64,
}

/// A model-specific sequence of native token operations emitted by AOT.
pub type NativeToken = fn(&mut Session<'_>, u32) -> Result<Vec<f32>>;

fn execute(model: &Model<'_>, args: &Arguments, native: Option<NativeToken>) -> Result<RunOutput> {
    if args.values.contains_key("--tokens") && args.values.contains_key("--prompt") {
        return Err(Error("use either --prompt or --tokens".into()));
    }
    let context = args.number("--context", model.config().context_length.min(512))?;
    let predict = args.number("--predict", 16)?;
    if args.logits
        && predict
            .checked_add(1)
            .and_then(|steps| steps.checked_mul(model.config().vocab_size))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .is_none_or(|bytes| bytes > MAX_DIAGNOSTIC_BYTES)
    {
        return Err(Error(
            "requested logits exceed the 64 MiB diagnostic limit".into(),
        ));
    }
    let prompt_tokens = if let Some(tokens) = args.values.get("--tokens") {
        tokens
            .split(',')
            .map(|id| {
                id.parse()
                    .map_err(|_| Error("invalid --tokens list".into()))
            })
            .collect::<Result<Vec<u32>>>()?
    } else {
        model.tokenizer().encode(args.required("--prompt")?, true)?
    };
    if prompt_tokens.is_empty()
        || prompt_tokens
            .len()
            .checked_add(predict)
            .is_none_or(|length| length > context)
    {
        return Err(Error(
            "prompt plus requested generation exceeds context or is empty".into(),
        ));
    }
    let mut session = model.session(context)?;
    let start = Instant::now();
    let mut scores = if let Some(evaluate) = native {
        let mut scores = Vec::new();
        for &token in &prompt_tokens {
            drop(scores);
            scores = evaluate(&mut session, token)?;
        }
        scores
    } else {
        session.prefill(&prompt_tokens)?
    };
    let prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut logits = args.logits.then(|| vec![scores.clone()]);
    let mut tokens = Vec::new();
    let mut bytes = Vec::new();
    let eos_id = model.tokenizer().spec().eos_id;
    let start = Instant::now();
    for _ in 0..predict {
        let mut selected = 0;
        for i in 1..scores.len() {
            if scores[i] > scores[selected] {
                selected = i;
            }
        }
        let token =
            u32::try_from(selected).map_err(|_| Error("selected token does not fit u32".into()))?;
        tokens.push(token);
        bytes.extend(model.tokenizer().token_piece_bytes(token, true)?);
        drop(scores);
        scores = if let Some(evaluate) = native {
            evaluate(&mut session, token)?
        } else {
            session.decode(token)?
        };
        if let Some(logits) = &mut logits {
            logits.push(scores.clone());
        }
        if token == eos_id {
            break;
        }
    }
    let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(RunOutput {
        prompt_tokens,
        tokens,
        eos_id,
        text: String::from_utf8_lossy(&bytes).into_owned(),
        bytes,
        logits,
        context,
        kv_bytes: session.kv_bytes(),
        prefill_ms,
        decode_ms,
    })
}

fn print_json(value: &impl Serialize) -> Result<()> {
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, value)?;
    writeln!(output)?;
    Ok(())
}

/// Execute the model tools using command-line arguments from the process.
pub fn main_cli() -> Result<()> {
    let mut args = process_arguments()?;
    let Some(command) = args.next() else {
        println!("{USAGE}");
        return Ok(());
    };
    if command == "--help" || command == "-h" {
        println!("{USAGE}");
        return Ok(());
    }
    let allowed: &[&str] = match command.as_str() {
        "inspect" => &["--model"],
        "run" => &[
            "--model",
            "--prompt",
            "--tokens",
            "--predict",
            "--context",
            "--logits",
        ],
        "export" | "compile" => &["--model", "--output"],
        _ => return Err(Error(format!("unknown command: {command}\n{USAGE}"))),
    };
    let args = Arguments::parse(args, allowed)?;
    let path = Path::new(args.required("--model")?);
    if command == "export" || command == "compile" {
        let reader = GgufReader::open(path)?;
        let output = Path::new(args.required("--output")?);
        if command == "export" {
            let manifest = write_model_file(&reader, output)?;
            return print_json(&manifest);
        }
        return compile(&reader, output);
    }
    let mut prefix = [0; 4];
    File::open(path)?.read_exact(&mut prefix)?;
    if &prefix == b"GGUF" {
        let reader = GgufReader::open(path)?;
        handle_model(&Model::from_gguf(&reader)?, &command, &args)
    } else {
        let artifact = Artifact::open(path)?;
        handle_model(&artifact.model()?, &command, &args)
    }
}

fn handle_model(model: &Model<'_>, command: &str, args: &Arguments) -> Result<()> {
    if command == "inspect" {
        print_json(&serde_json::json!({
            "config": model.config(),
            "plan": model.plan(),
            "memory": model.memory_plan(model.config().context_length)?,
        }))
    } else {
        print_json(&execute(model, args, None)?)
    }
}

/// Entry point linked into generated native model executables.
///
/// Only the current executable's normalized resources are opened. Their full
/// digest and typed execution plan must match those bound at build time.
/// No subprocess, compiler, GGUF parser or network access is used in this path.
pub fn embedded_main(
    expected_resource_sha256: &str,
    expected_plan: &[Operation],
    evaluate: NativeToken,
) {
    let result = process_arguments().and_then(|args| {
        run_embedded_with_arguments(
            expected_resource_sha256,
            expected_plan,
            evaluate,
            &args.collect::<Vec<_>>(),
        )
    });
    if let Err(error) = result {
        eprintln!("bml-model: {error}");
        std::process::exit(1);
    }
}

/// Run the current executable's embedded model with explicitly supplied options.
///
/// This has the same hash, plan and context checks as [`embedded_main`], but
/// returns errors to the caller rather than terminating the process.
pub fn run_embedded_with_arguments(
    expected_resource_sha256: &str,
    expected_plan: &[Operation],
    evaluate: NativeToken,
    arguments: &[String],
) -> Result<()> {
    let args = Arguments::parse(
        arguments.iter().cloned(),
        &["--prompt", "--tokens", "--predict", "--context", "--logits"],
    )?;
    let artifact = Artifact::open("/proc/self/exe")?;
    if artifact.resource_sha256() != expected_resource_sha256 {
        return Err(Error(
            "embedded resource identity differs from compiled model".into(),
        ));
    }
    let model = artifact.model()?;
    if model.plan() != expected_plan {
        return Err(Error(
            "embedded execution plan differs from compiled model".into(),
        ));
    }
    print_json(&execute(&model, &args, Some(evaluate))?)
}

fn file_hash(path: &Path) -> Result<String> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn compile(reader: &GgufReader, output: &Path) -> Result<()> {
    let build_started = Instant::now();
    if !cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
        return Err(Error(
            "AOT target currently requires Linux x86_64 GNU".into(),
        ));
    }
    if output.exists() {
        return Err(Error("output already exists".into()));
    }
    let model = Model::from_gguf(reader)?;
    let library = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = library
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| Error("BML source workspace is unavailable".into()))?;
    let build_root = workspace.join("target/bml-aot");
    fs::create_dir_all(&build_root)?;
    let build = build_root.join(format!(
        "build-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| Error(e.to_string()))?
            .as_nanos()
    ));
    fs::create_dir(&build)?;
    fs::create_dir(build.join("src"))?;
    let resource_path = build.join("model.bmlres");
    let mut resource = File::create(&resource_path)?;
    let manifest = write_model(reader, &mut resource)?;
    resource.sync_all()?;
    let plan = model
        .plan()
        .iter()
        .map(|op| format!("Operation::{op:?}"))
        .collect::<Vec<_>>()
        .join(",\n");
    let layers = (0..model.config().block_count)
        .map(|index| format!("let step = step.layer({index})?;\n"))
        .collect::<String>();
    let source = format!(
        "use bml_inference::model::{{Operation,Session}};\nconst PLAN: &[Operation] = &[{plan}];\n\
         fn evaluate(session: &mut Session<'_>, token:u32)->bml_inference::Result<Vec<f32>> {{\n\
         let step = session.begin_token(token)?;\n{layers}step.finish()\n}}\n\
         fn main() {{ bml_inference::driver::embedded_main({:?}, PLAN, evaluate); }}\n",
        file_hash(&resource_path)?,
    );
    fs::write(build.join("src/main.rs"), &source)?;
    let dependency_path = serde_json::to_string(&library.to_string_lossy())?;
    let binary_name = format!(
        "bml-native-{}",
        build.file_name().unwrap_or_default().to_string_lossy()
    );
    let cargo_manifest = format!(
        "[package]\nname = {binary_name:?}\nversion = \"1.0.0\"\nedition = \"2021\"\n[workspace]\n[dependencies]\nbml-inference = {{ path = {dependency_path} }}\n[profile.release]\nopt-level = 3\nstrip = true\npanic = \"abort\"\ncodegen-units = 1\n"
    );
    fs::write(build.join("Cargo.toml"), cargo_manifest)?;
    fs::copy(workspace.join("Cargo.lock"), build.join("Cargo.lock"))?;
    fs::copy(
        workspace.join("rust-toolchain.toml"),
        build.join("rust-toolchain.toml"),
    )?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let generated_manifest = build.join("Cargo.toml");
    let target = build_root.join("target");
    for arguments in [
        vec!["metadata", "--offline", "--format-version", "1"],
        vec!["build", "--release", "--locked", "--offline"],
    ] {
        let status = Command::new(&cargo)
            .args(arguments)
            .arg("--manifest-path")
            .arg(&generated_manifest)
            .env("CARGO_TARGET_DIR", &target)
            .env("RUSTFLAGS", "")
            .env("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=x86-64")
            .stdout(Stdio::null())
            .status()?;
        if !status.success() {
            return Err(Error(format!(
                "native compilation failed; build files: {}",
                build.display()
            )));
        }
    }
    let launcher = target.join("release").join(binary_name);
    let rustc = Command::new("rustc")
        .arg("--version")
        .arg("--verbose")
        .current_dir(&build)
        .output()?;
    if !rustc.status.success() {
        return Err(Error("cannot record the native compiler version".into()));
    }
    append_to_executable(&launcher, &resource_path, output)?;
    print_json(&serde_json::json!({
        "output": output,
        "sha256": file_hash(output)?,
        "bytes": fs::metadata(output)?.len(),
        "target": manifest.target,
        "cpu": "x86-64",
        "rustc_version": String::from_utf8_lossy(&rustc.stdout),
        "elapsed_ms": build_started.elapsed().as_secs_f64() * 1000.0,
        "build_flags": ["--release", "--locked", "--offline", "-Ctarget-cpu=x86-64"],
        "source_sha256": manifest.source_sha256,
        "generated_source_sha256": format!("{:x}", Sha256::digest(source.as_bytes())),
        "generated_lock_sha256": file_hash(&build.join("Cargo.lock"))?,
        "build_directory": build,
    }))
}
