//! Actual compile/run commands; zero-layer byte fixture isolates persistence,
//! tokenizer metadata, stored RPN execution, context limits and output IO.
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
fn compiled_cli_generates_utf8_and_eos_using_saved_metadata_and_relocated_weights() {
    let source = token_fixture::fixture();
    let directory = Artifact(source.0.with_extension("compiled.bmlgraph"));
    let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .arg("compile")
        .arg("--model")
        .arg(&source.0)
        .arg("--output")
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(directory.0.join("model.bmlgraph.json").exists());
    assert!(directory.0.join("program_0.bmlgraph").exists());
    let relocated = directory.0.join("relocated.gguf");
    std::fs::rename(&source.0, &relocated).unwrap();
    for (capacity, expected) in [(8, Some(3)), (3, Some(0)), (2, None)] {
        let output = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .arg("run")
            .arg("--model")
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
            output.status.success(),
            expected.is_some(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Some(count) = expected {
            let text = String::from_utf8(output.stdout).unwrap();
            assert!(text.contains("Tokenizado: 3 tokens"), "{text}");
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
}
