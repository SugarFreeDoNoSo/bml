//! CLI integration for the explicitly selected complete bounded profile.
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};

#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;
#[path = "support/token_fixture.rs"]
#[allow(dead_code)]
mod token_fixture;

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn compile(source: &Path, output: &Path, options: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .arg("compile")
        .args(options)
        .arg("--model")
        .arg(source)
        .arg("--output")
        .arg(output)
        .output()
        .unwrap()
}
fn successful(output: Output) -> String {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn full_bounded_cli_persists_profile_and_runs_utf8_with_relocated_weights() {
    let source = token_fixture::fixture_with_output_norm();
    let directory = Directory(source.0.with_extension("full-bounded.bmlgraph"));
    let output = successful(compile(
        &source.0,
        &directory.0,
        &[
            "--symbolic",
            "--bounded",
            "--context-size",
            "6",
            "--symbolic-large-budget",
        ],
    ));
    assert!(
        output.contains("llama-symbolic-bml-bounded-f64-f32kv-v4"),
        "{output}"
    );
    let manifest_path = directory.0.join("symbolic.bmlgraph.json");
    let original = std::fs::read(&manifest_path).unwrap();
    let artifact = bml_compiler::symbolic_artifact::load_directory(&directory.0, None).unwrap();
    assert_eq!(artifact.manifest.version, 4);
    assert!(artifact.manifest.rms_bounds.is_none());
    assert!(artifact.manifest.bounded_profile.is_some());
    let dot = artifact
        .kernels
        .iter()
        .find(|k| k.operation == bml_compiler::symbolic_kernels::Operation::Dot)
        .unwrap();
    let ops: Vec<_> = dot
        .graphs
        .iter()
        .flat_map(|g| &g.fragments)
        .flat_map(|f| &f.ops)
        .collect();
    assert!(ops
        .iter()
        .any(|op| matches!(op, bml_compiler::RpnOp::Loop { .. })));
    assert!(ops
        .iter()
        .any(|op| matches!(op, bml_compiler::RpnOp::ContextVarIndexed { .. })));
    assert!(ops
        .iter()
        .any(|op| matches!(op, bml_compiler::RpnOp::ContextConstIndexed { .. })));
    drop(artifact);
    let relocated = directory.0.join("relocated.gguf");
    std::fs::rename(&source.0, &relocated).unwrap();
    for large_budget in [false, true] {
        let mut run = Command::new(env!("CARGO_BIN_EXE_bml-cli"));
        run.args(["run", "--model"])
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
            ]);
        if large_budget {
            run.arg("--symbolic-large-budget");
        }
        let output = successful(run.output().unwrap());
        assert!(
            output.contains("llama-symbolic-bml-bounded-f64-f32kv-v4"),
            "{output}"
        );
        assert!(output.contains("\né\n[EOS]"), "{output}");
        assert!(
            output.contains("Generados 3 tokens desde artefacto compilado"),
            "{output}"
        );
        assert_eq!(std::fs::read(&manifest_path).unwrap(), original);
    }
}

#[test]
fn full_bounded_cli_runs_nonempty_causal_prompt_with_all_operation_families() {
    use bml_compiler::symbolic_kernels::Operation;
    let source = symbolic_fixture::fixture(0.25);
    let directory = Directory(source.0.with_extension("causal-full-bounded.bmlgraph"));
    successful(compile(
        &source.0,
        &directory.0,
        &["--symbolic", "--bounded", "--context-size", "6"],
    ));
    let loaded = bml_compiler::symbolic_artifact::load_directory(&directory.0, None).unwrap();
    assert_eq!(loaded.manifest.config.n_layers, 1);
    for operation in [
        Operation::Dot,
        Operation::AttentionDot,
        Operation::WeightedSum,
        Operation::Norm,
        Operation::Rope,
        Operation::Softmax,
        Operation::Swiglu,
        Operation::Add,
        Operation::Scale,
    ] {
        assert!(
            loaded
                .manifest
                .kernels
                .iter()
                .any(|k| k.operation == operation),
            "missing {operation:?}"
        );
    }
    drop(loaded);
    let output = successful(
        Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["run", "--model"])
            .arg(&directory.0)
            .args([
                "--prompt",
                "AB",
                "--num-tokens",
                "1",
                "--temp",
                "0",
                "--symbolic-large-budget",
            ])
            .output()
            .unwrap(),
    );
    assert!(output.contains("Tokenizado: 4 tokens"), "{output}");
    assert!(
        output.contains("Generados 1 tokens desde artefacto compilado"),
        "{output}"
    );
}

#[test]
fn full_bounded_cli_rejects_mixed_profiles_wrong_budget_modes_and_short_fragments() {
    let source = token_fixture::fixture_with_output_norm();
    let rejected = Directory(source.0.with_extension("rejected-full-bounded.bmlgraph"));
    for options in [
        vec!["--bounded"],
        vec!["--symbolic-large-budget"],
        vec![
            "--symbolic",
            "--bounded",
            "--context-size",
            "6",
            "--rms-input-bound",
            "16",
            "--rms-weight-bound",
            "16",
        ],
        vec![
            "--symbolic",
            "--bounded",
            "--context-size",
            "6",
            "--l1-threshold",
            "128",
        ],
    ] {
        let result = compile(&source.0, &rejected.0, &options);
        assert!(!result.status.success(), "accepted {options:?}");
        assert!(!rejected.0.exists(), "partial publication for {options:?}");
    }
    let native = Directory(
        source
            .0
            .with_extension("native-no-symbolic-budget.bmlgraph"),
    );
    successful(compile(&source.0, &native.0, &[]));
    for path in [&source.0, &native.0] {
        let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["run", "--model"])
            .arg(path)
            .args([
                "--prompt",
                "x",
                "--num-tokens",
                "1",
                "--symbolic-large-budget",
            ])
            .output()
            .unwrap();
        assert!(!result.status.success());
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            error.contains("requiere un artefacto simbólico o evaluación BMLGGUF"),
            "{error}"
        );
    }
}
