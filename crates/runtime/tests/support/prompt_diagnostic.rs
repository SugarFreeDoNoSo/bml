use super::{argmax, prompt_profile::Probe, save_vector};
use bml_compiler::{
    gguf_compiler::InferenceCompiler,
    symbolic_artifact::{self, Manifest},
    symbolic_bounded::{BoundedProfile, ProjectionBounds},
    symbolic_kernels::RmsBounds,
    HashConsedKV,
};
use bml_model_weights::Digest;
use bml_runtime::symbolic_causal::{SymbolicCausalModel, SymbolicLimits};
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

pub const CASES: [(&str, [u32; 2]); 2] = [("Hello", [1, 15043]), ("Hi", [1, 6324])];
const SOURCE_SHA: &str = "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556";
const FILES: [(&str, &str); 13] = [
    (
        "kernel_0_0.bmlgraph",
        "6a8ff193ab925becd5530c3c4428a153ec1ce1513b803a0205ecf4c9f98e855d",
    ),
    (
        "kernel_10_0.bmlgraph",
        "fda06a88e3ed98f883d46784ddced30db761b9967e1d28baef8b3599c11fc947",
    ),
    (
        "kernel_11_0.bmlgraph",
        "25703baee8d38c660fa1eb3e4470fccf6c6c421f606410fe2a131fd1c9f8ec41",
    ),
    (
        "kernel_1_0.bmlgraph",
        "7902cf2fe4d290c590bd4108e7a3797db77bba15858cea69db456337a90e4ca7",
    ),
    (
        "kernel_2_0.bmlgraph",
        "8e183190e2c41d17d6fb635fd644d23bdd7232e01e591797f09cc23d58c34d8c",
    ),
    (
        "kernel_3_0.bmlgraph",
        "b1d0f80ab59cef487dc05ad3862d8654a4e4342ffb7cecc9dd1ea920b8f19502",
    ),
    (
        "kernel_4_0.bmlgraph",
        "df9807c0eb31ebcb5a99f10d8b840e05d7ef1f8f7c507904fdef29da74267bc9",
    ),
    (
        "kernel_5_0.bmlgraph",
        "cb83228027dc778bddc0bfbcbd695e5fe4c856131f221f4a2358df5bc6b8b848",
    ),
    (
        "kernel_6_0.bmlgraph",
        "0ea01dde5e0d6e1caf51aa1872011d6ec0cd32c8f09e31dff7cf38182f518763",
    ),
    (
        "kernel_7_0.bmlgraph",
        "5567b8a8cb312a0dedd951d6a1b6a5885ce53c16d4f1a5968d801be6aceaf809",
    ),
    (
        "kernel_8_0.bmlgraph",
        "066b1892452faee72e3e2bacb4cd3822d946f93746e93854411c1b1a886ccda0",
    ),
    (
        "kernel_9_0.bmlgraph",
        "8ff3a8fef4891e26845f4dd88929e05ded1f82af70a351de9756d9e4504c8a01",
    ),
    (
        "symbolic.bmlgraph.json",
        "0b6e9bc33693e9581676fad805fec2cf2173410c1ba112ddce99f490709bbd97",
    ),
];
pub fn profile() -> BoundedProfile {
    BoundedProfile {
        rms: RmsBounds {
            input: 160.0,
            weight: 4.0,
        },
        projection_input: 64.0,
        projection_weight: 4.0,
        down_projection: Some(ProjectionBounds {
            input: 256.0,
            weight: 1.5,
        }),
        attention: 20.0,
        value: 16.0,
        rope: 32.0,
        gate: 18.0,
        up: 16.0,
        residual: 160.0,
        scale_input: 256.0,
        score: 16.0,
    }
}
pub fn validate_prompt(prompt: &str, ids: &[u32], context: u32) -> Result<[u32; 2], String> {
    let expected = CASES
        .iter()
        .find(|(text, _)| *text == prompt)
        .ok_or("prompt must be exactly Hello or Hi")?
        .1;
    if ids != expected || ids.len() != 2 || context != 2 {
        return Err(format!("predeclared prompt IDs/context differ: prompt={prompt:?} ids={ids:?} context={context}"));
    }
    Ok(expected)
}
fn limits() -> SymbolicLimits {
    SymbolicLimits {
        max_nodes: 0,
        max_cached_ops: 4_000_000,
        max_programs: 4096,
        graph: bml_runtime::graph::GraphLimits {
            max_ops: 2_000_000,
            max_steps: 100_000_000,
            max_values: 4_000_000,
        },
    }
}
fn write_json(directory: &Path, name: &str, value: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(name))
        .map_err(|error| error.to_string())?;
    file.write_all(&bytes)
        .and_then(|()| file.flush())
        .map_err(|error| error.to_string())?;
    Ok(Digest::of(&bytes).to_string())
}
struct Setup {
    native: InferenceCompiler,
    source: PathBuf,
    artifact: PathBuf,
    output: PathBuf,
    prompt: String,
    ids: [u32; 2],
    protocol_sha256: String,
}
impl Setup {
    fn from_environment(mode: &str) -> Result<Self, String> {
        let prompt = std::env::var("BML_TEST_PROMPT").map_err(|_| "BML_TEST_PROMPT is required")?;
        if !CASES.iter().any(|(text, _)| *text == prompt) {
            return Err("prompt must be exactly Hello or Hi".into());
        }
        let path = |name: &str| {
            std::env::var_os(name)
                .map(PathBuf::from)
                .ok_or_else(|| format!("{name} is required"))
        };
        let source = path("BML_TEST_GGUF")?
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let artifact = path("BML_TEST_SYMBOLIC_ARTIFACT")?
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let output = path("BML_TEST_LOGITS_DIR")?;
        let revision = std::env::var("BML_TEST_SOURCE_REVISION")
            .map_err(|_| "BML_TEST_SOURCE_REVISION is required")?;
        if revision.len() != 40
            || !revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("BML_TEST_SOURCE_REVISION must be a full lowercase Git SHA; root must authenticate it against build inputs".into());
        }
        let mut actual_names = Vec::new();
        for entry in fs::read_dir(&artifact).map_err(|error| error.to_string())? {
            let entry = entry.map_err(|error| error.to_string())?;
            if !entry
                .file_type()
                .map_err(|error| error.to_string())?
                .is_file()
            {
                return Err("artifact must contain exactly 13 regular files".into());
            }
            actual_names.push(
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| "non-UTF8 artifact name")?,
            );
        }
        actual_names.sort();
        let mut expected_names: Vec<_> = FILES.iter().map(|(name, _)| name.to_string()).collect();
        expected_names.sort();
        if actual_names != expected_names {
            return Err("artifact must contain the 13 pinned files".into());
        }
        let mut files = Vec::new();
        for (name, expected) in FILES {
            let bytes = fs::read(artifact.join(name)).map_err(|error| error.to_string())?;
            if Digest::of(&bytes).to_string() != expected {
                return Err(format!("artifact digest differs: {name}"));
            }
            files.push(json!({"name":name,"bytes":bytes.len(),"sha256":expected}));
        }
        // This loader authenticates the mapped GGUF bytes and stored programs.
        // It parses bytecode but neither compiles nor evaluates BML kernels.
        let loaded = symbolic_artifact::load_directory(&artifact, Some(&source))?;
        let manifest: Manifest = loaded.manifest;
        let native = loaded.compiler;
        drop(loaded.kernels);
        if manifest.version != 4
            || manifest.context_length != 2
            || manifest.profile != symbolic_artifact::BOUNDED_PROFILE
            || manifest.source_sha256 != SOURCE_SHA
            || manifest.bounded_profile != Some(profile())
        {
            return Err("pinned complete bounded context-2 profile differs".into());
        }
        let ids = validate_prompt(
            &prompt,
            &native.vocab().encode(&prompt)?,
            manifest.context_length,
        )?;
        if (
            native.config().n_layers,
            native.config().n_heads,
            native.config().n_embd,
            native.n_kv_heads(),
            native.head_dim(),
            native.vocab().len(),
        ) != (22, 32, 2048, 4, 64, 32000)
        {
            return Err("expected the pinned TinyLlama dimensions".into());
        }
        let binary = std::env::current_exe().map_err(|error| error.to_string())?;
        let binary_sha256 =
            Digest::of(&fs::read(&binary).map_err(|error| error.to_string())?).to_string();
        fs::create_dir(&output)
            .map_err(|error| format!("output must be a new directory: {error}"))?;
        let protocol_sha256 = write_json(
            &output,
            "protocol.json",
            &json!({
                "format_version":1,"mode":mode,"status":"prepared_before_any_forward",
                "predeclared_cases":[{"prompt":"Hello","input_ids":[1,15043]},{"prompt":"Hi","input_ids":[1,6324]}],
                "selected_prompt":prompt,"rust_encoded_input_ids":ids,"context":2,
                "source":source,"source_sha256":SOURCE_SHA,"artifact":artifact,"artifact_files":files,"profile":profile(),
                "binary":binary,"binary_sha256":binary_sha256,"source_revision_declared":revision,
                "provenance_qualification":"The revision label is declared; the separately captured immutable build/source receipt must authenticate it.",
                "symbolic_limits":{"max_nodes":0,"max_cached_ops":4_000_000,"max_programs":4096,
                    "graph":{"max_ops":2_000_000,"max_steps":100_000_000,"max_values":4_000_000}},
                "cache_capacity_per_backend":2,"kv_bytes_per_backend":22*4*64*2*2*4,
                "qualification":"One text per process; diagnostic only; no preset numerical tolerance or parity/performance claim. Native-only never qualifies as a full BML run."
            }),
        )?;
        Ok(Self {
            native,
            source,
            artifact,
            output,
            prompt,
            ids,
            protocol_sha256,
        })
    }
    fn cache(&self) -> Result<HashConsedKV, String> {
        HashConsedKV::try_new(
            self.native.config().n_layers,
            self.native.n_kv_heads(),
            self.native.head_dim(),
            2,
            bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES,
        )
    }
}
pub fn run_native_only() -> Result<(), String> {
    let setup = Setup::from_environment("native_only_profile_preflight")?;
    let mut cache = setup.cache()?;
    let mut probe = Probe::new(&setup.native, profile());
    let mut positions = Vec::new();
    for (position, &id) in setup.ids.iter().enumerate() {
        let started = Instant::now();
        let result = probe.position(&setup.native, id, &mut cache);
        let error = result.as_ref().err().cloned();
        let mut report = json!({"mode":"native_only_profile_preflight","bml_executed":false,
            "status":if error.is_some(){"failed"}else{"native_position_pass"},
            "prompt":setup.prompt,"input_token":id,"position":position,"native_cache_position":cache.current_pos(),
            "protocol_sha256":setup.protocol_sha256,"elapsed_seconds":started.elapsed().as_secs_f64(),
            "ranges":probe.report(),"error":error});
        if let Ok(logits) = &result {
            if cache.current_pos() != position as u32 + 1 {
                return Err("native cache position differs".into());
            }
            let name = format!("native-{position}.f64le");
            save_vector(&setup.output, &name, logits);
            report["native_vector"] = name.into();
            report["logits"] = logits.len().into();
            report["native_argmax"] = argmax(logits).into();
        }
        write_json(
            &setup.output,
            &format!("native-position-{position}.json"),
            &report,
        )?;
        eprintln!("{report}");
        result?;
        positions.push(report);
    }
    write_json(
        &setup.output,
        "native-preflight-summary.json",
        &json!({
            "status":"native_only_preflight_pass","bml_executed":false,"prompt":setup.prompt,
            "input_ids":setup.ids,"protocol_sha256":setup.protocol_sha256,"positions":positions,
            "qualification":"Necessary native-derived operand checks only; does not establish BML admission, global parity, or full execution."
        }),
    )?;
    Ok(())
}
pub fn run_full() -> Result<(), String> {
    let setup = Setup::from_environment("full_symbolic_prompt_diagnostic")?;
    let mut symbolic =
        SymbolicCausalModel::open_artifact(&setup.artifact, Some(&setup.source), limits())?;
    if symbolic.context_limit() != 2 || symbolic.profile() != symbolic_artifact::BOUNDED_PROFILE {
        return Err("symbolic artifact profile differs".into());
    }
    let mut native_cache = setup.cache()?;
    let mut symbolic_cache = setup.cache()?;
    let mut positions = Vec::new();
    for (position, &id) in setup.ids.iter().enumerate() {
        let reference = setup.native.forward_cached(&[id], &mut native_cache)?;
        valid_logits(&reference)?;
        let native_file = format!("native-{position}.f64le");
        save_vector(&setup.output, &native_file, &reference);
        let started = Instant::now();
        let result = symbolic.forward_cached(&[id], &mut symbolic_cache);
        let actual = match result {
            Ok(actual) => actual,
            Err(error) => {
                write_json(
                    &setup.output,
                    &format!("failure-{position}.json"),
                    &json!({"status":"partial_full_run_failed",
                    "prompt":setup.prompt,"input_token":id,"position":position,"native_vector":native_file,
                    "native_cache_position":native_cache.current_pos(),"symbolic_cache_position":symbolic_cache.current_pos(),
                    "protocol_sha256":setup.protocol_sha256,"error":error}),
                )?;
                return Err(error);
            }
        };
        let elapsed = started.elapsed().as_secs_f64();
        let symbolic_file = format!("symbolic-{position}.f64le");
        save_vector(&setup.output, &symbolic_file, &actual);
        valid_logits(&actual)?;
        if native_cache.current_pos() != position as u32 + 1
            || symbolic_cache.current_pos() != position as u32 + 1
        {
            return Err("cache position differs".into());
        }
        let differences: Vec<_> = actual
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .collect();
        if differences.iter().any(|value| !value.is_finite()) {
            return Err("nonfinite differences".into());
        }
        let maximum = differences.iter().copied().fold(0.0, f64::max);
        let count = differences.len() as f64;
        let mean = differences.iter().map(|value| value / count).sum::<f64>();
        let rms = if maximum == 0.0 {
            0.0
        } else {
            maximum
                * (differences
                    .iter()
                    .map(|value| (value / maximum).powi(2))
                    .sum::<f64>()
                    / count)
                    .sqrt()
        };
        if !mean.is_finite() || !rms.is_finite() {
            return Err("nonfinite aggregate statistics".into());
        }
        let report = json!({"status":"full_position_completed_diagnostic","prompt":setup.prompt,"input_token":id,"position":position,
            "logits":actual.len(),"native_vector":native_file,"symbolic_vector":symbolic_file,
            "native_argmax":argmax(&reference),"symbolic_argmax":argmax(&actual),"max_absolute_error":maximum,
            "mean_absolute_error":mean,"rms_error":rms,"native_cache_position":native_cache.current_pos(),
            "symbolic_cache_position":symbolic_cache.current_pos(),"symbolic_elapsed_seconds":elapsed,"protocol_sha256":setup.protocol_sha256});
        write_json(&setup.output, &format!("position-{position}.json"), &report)?;
        eprintln!("{report}");
        positions.push(report);
    }
    let families = symbolic.executed_operations();
    if families
        != [
            "attention_dot",
            "dot",
            "down_dot",
            "residual_add",
            "rms_norm",
            "rope",
            "scale",
            "softmax",
            "swiglu",
            "weighted_sum",
        ]
    {
        return Err("incomplete executed BML families".into());
    }
    write_json(
        &setup.output,
        "summary.json",
        &json!({"status":"full_completed_diagnostic","prompt":setup.prompt,
        "input_ids":setup.ids,"positions":positions,"executed_operations":families,"protocol_sha256":setup.protocol_sha256,
        "qualification":"Both fixed positions completed; diagnostic only, no preset numerical tolerance, numerical-parity or performance claim."}),
    )?;
    Ok(())
}
fn valid_logits(values: &[f64]) -> Result<(), String> {
    if values.len() != 32000 || values.iter().any(|value| !value.is_finite()) {
        return Err("expected 32000 finite logits".into());
    }
    Ok(())
}
