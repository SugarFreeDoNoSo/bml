//! Real symbolic AOT compile/run commands: persisted tokenizer, byte pieces,
//! relocated source and compiled context are exercised in separate processes.
use std::{path::PathBuf, process::Command};
#[path = "support/token_fixture.rs"]
mod token_fixture;
struct Artifact(PathBuf);
impl Drop for Artifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn symbolic_cli_compiles_aot_and_generates_utf8_eos_with_relocated_weights() {
    let source = token_fixture::fixture();
    let directory = Artifact(source.0.with_extension("symbolic.bmlgraph"));
    let missing = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["compile", "--symbolic", "--model"])
        .arg(&source.0)
        .arg("--output")
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("--context-size"));
    assert!(!directory.0.exists());
    let compile = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args([
            "compile",
            "--symbolic",
            "--context-size",
            "6",
            "--l1-threshold",
            "256",
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
    assert!(directory.0.join("symbolic.bmlgraph.json").is_file());
    let mut program_count = 0;
    for entry in std::fs::read_dir(&directory.0).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "bmlgraph") {
            let bytes = std::fs::read(path).unwrap();
            let graph = bml_compiler::BmlGraph::deserialize(&bytes, 256).unwrap();
            assert_eq!(graph.threshold, 256);
            assert!(graph
                .fragments
                .iter()
                .all(|fragment| fragment.byte_size() <= 256));
            program_count += 1;
        }
    }
    assert!(program_count > 0);
    assert!(!directory.0.join("model.bmlgraph.json").exists());
    let relocated = directory.0.join("weights.gguf");
    std::fs::rename(&source.0, &relocated).unwrap();
    for (capacity, expected) in [(8, Some(3)), (3, Some(0)), (2, None)] {
        let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
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
            ])
            .arg(capacity.to_string())
            .output()
            .unwrap();
        assert_eq!(
            result.status.success(),
            expected.is_some(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        if let Some(count) = expected {
            let text = String::from_utf8(result.stdout).unwrap();
            assert!(text.contains("llama-symbolic-bml-f64-f32kv-v2"), "{text}");
            assert!(
                text.contains(&format!(
                    "Generados {count} tokens desde artefacto compilado"
                )),
                "{text}"
            );
            if count == 3 {
                assert!(text.contains("\né\n[EOS]"), "{text}");
            }
        }
    }
    // Mixed profile markers must not silently choose one set of programs.
    std::fs::write(directory.0.join("model.bmlgraph.json"), b"{}").unwrap();
    let ambiguous = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["run", "--model"])
        .arg(&directory.0)
        .arg("--weights")
        .arg(&relocated)
        .args(["--prompt", "x", "--num-tokens", "1"])
        .output()
        .unwrap();
    assert!(!ambiguous.status.success());
    assert!(String::from_utf8_lossy(&ambiguous.stderr).contains("ambiguo"));
}
