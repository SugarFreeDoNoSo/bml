//! The bounded numerical profile requires explicit paired compile options.
use std::{path::PathBuf, process::Command};

#[path = "support/token_fixture.rs"]
#[allow(dead_code)] // This suite uses only the normalized variant of the shared fixture.
mod token_fixture;

struct Artifact(PathBuf);
impl Drop for Artifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn bounded_rms_cli_persists_explicit_profile_and_runs_with_relocated_weights() {
    let source = token_fixture::fixture_with_output_norm();
    let directory = Artifact(source.0.with_extension("bounded.bmlgraph"));
    let compile = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args([
            "compile",
            "--symbolic",
            "--context-size",
            "6",
            "--l1-threshold",
            "256",
            "--rms-input-bound",
            "16",
            "--rms-weight-bound",
            "16",
            "--model",
        ])
        .arg(&source.0)
        .arg("--output")
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let path = directory.0.join("symbolic.bmlgraph.json");
    let bytes = std::fs::read(&path).unwrap();
    let manifest = bml_compiler::symbolic_artifact::load_directory(&directory.0, None)
        .unwrap()
        .manifest;
    assert_eq!(manifest.version, 3);
    assert_eq!(
        manifest.profile,
        "llama-symbolic-bml-bounded-rms-f64-f32kv-v3"
    );
    assert_eq!(manifest.rms_bounds.unwrap().input, 16.0);
    assert_eq!(manifest.rms_bounds.unwrap().weight, 16.0);
    assert!(manifest
        .kernels
        .iter()
        .any(|kernel| kernel.operation == bml_compiler::symbolic_kernels::Operation::Norm));
    let relocated = directory.0.join("weights.gguf");
    std::fs::rename(&source.0, &relocated).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["run", "--model"])
        .arg(&directory.0)
        .arg("--weights")
        .arg(&relocated)
        .args([
            "--prompt",
            "x",
            "--num-tokens",
            "6",
            "--temp",
            "0",
            "--context-size",
            "8",
        ])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let output = String::from_utf8(run.stdout).unwrap();
    assert!(
        output.contains("llama-symbolic-bml-bounded-rms-f64-f32kv-v3"),
        "{output}"
    );
    assert!(output.contains("\né\n[EOS]"), "{output}");
    assert!(
        output.contains("Generados 3 tokens desde artefacto compilado"),
        "{output}"
    );
    assert_eq!(std::fs::read(path).unwrap(), bytes);
}

#[test]
fn bounded_rms_cli_rejects_unpaired_options_bad_bounds_and_actual_input_violations() {
    let source = token_fixture::fixture_with_output_norm();
    let directory = Artifact(source.0.with_extension("rejected.bmlgraph"));
    for arguments in [
        vec![
            "--symbolic",
            "--context-size",
            "6",
            "--rms-input-bound",
            "16",
        ],
        vec![
            "--symbolic",
            "--context-size",
            "6",
            "--rms-weight-bound",
            "16",
        ],
        vec!["--rms-input-bound", "16", "--rms-weight-bound", "16"],
        vec![
            "--symbolic",
            "--context-size",
            "6",
            "--rms-input-bound",
            "0",
            "--rms-weight-bound",
            "16",
        ],
        vec![
            "--symbolic",
            "--context-size",
            "6",
            "--rms-input-bound",
            "16",
            "--rms-weight-bound",
            "NaN",
        ],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .arg("compile")
            .args(&arguments)
            .arg("--model")
            .arg(&source.0)
            .arg("--output")
            .arg(&directory.0)
            .output()
            .unwrap();
        assert!(!result.status.success(), "accepted {arguments:?}");
        assert!(
            !directory.0.exists(),
            "partial publication for {arguments:?}"
        );
    }
    let compile = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args([
            "compile",
            "--symbolic",
            "--context-size",
            "6",
            "--rms-input-bound",
            "0.5",
            "--rms-weight-bound",
            "1",
            "--model",
        ])
        .arg(&source.0)
        .arg("--output")
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["run", "--model"])
        .arg(&directory.0)
        .args(["--prompt", "x", "--num-tokens", "1", "--temp", "0"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    let error = String::from_utf8_lossy(&result.stderr);
    assert!(error.contains("rms_norm:"), "{error}");
    assert!(error.contains("input 0"), "{error}");
}
