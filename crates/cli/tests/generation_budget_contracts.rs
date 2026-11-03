//! Generation limits apply to emitted tokens; an unrequested next inference
//! must not turn a successful final prediction into a numerical failure.
use std::{path::Path, process::Command};

#[path = "../../compiler/tests/support/causal_fixture.rs"]
mod causal_fixture;
#[path = "support/token_fixture.rs"]
mod token_fixture;

fn edit_output(path: &Path, edit: impl FnOnce(&mut [f32])) {
    let (mut bytes, start, mut values) = {
        let parser = bml_parser::GgufParser::open(path).unwrap();
        let info = parser.find_tensor("output.weight").unwrap();
        let data = parser.tensor_data(info).unwrap();
        let start = data.as_ptr() as usize - parser.bytes().as_ptr() as usize;
        (
            parser.bytes().to_vec(),
            start,
            parser.read_tensor_f32("output.weight").unwrap(),
        )
    }; // Release the mapping before rewriting the fixture.
    edit(&mut values);
    for (i, value) in values.into_iter().enumerate() {
        bytes[start + i * 4..start + (i + 1) * 4].copy_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn final_requested_token_does_not_evaluate_the_next_causal_position() {
    let source = causal_fixture::fixture(true);
    edit_output(&source.0, |values| {
        values.fill(0.0);
        values[69 * 4 + 1] = 1.0; // Prompt B predicts A, whose attention overflows.
    });
    let model = bml_compiler::gguf_compiler::InferenceCompiler::open(&source.0).unwrap();
    let logits = model.forward(&[1, 3, 70]).unwrap();
    assert!(logits[69] > 0.0);
    assert!(model
        .forward(&[1, 3, 70, 69])
        .unwrap_err()
        .contains("non-finite"));
    for rpn in [false, true] {
        for count in [1, 2] {
            let mut command = Command::new(env!("CARGO_BIN_EXE_bml-cli"));
            command
                .args(["run", "--model"])
                .arg(&source.0)
                .args([
                    "--prompt",
                    "B",
                    "--temp",
                    "0",
                    "--context-size",
                    "8",
                    "--num-tokens",
                ])
                .arg(count.to_string());
            if rpn {
                command.arg("--bml");
            }
            let output = command.output().unwrap();
            assert_eq!(
                output.status.success(),
                count == 1,
                "rpn={rpn} count={count}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            if count == 1 {
                let text = String::from_utf8(output.stdout).unwrap();
                assert!(text.contains("Generados 1 tokens"), "{text}");
            } else {
                assert!(String::from_utf8_lossy(&output.stderr).contains("non-finite"));
            }
        }
    }
}

#[test]
fn compiled_generation_can_outgrow_the_prompt_until_the_context_limit() {
    let source = token_fixture::fixture();
    // Remove the EOS row to produce a repeating C3/A9/unknown sequence.
    edit_output(&source.0, |values| values[4..6].fill(0.0));
    let directory = source.0.with_extension("generation.bmlgraph");
    bml_compiler::causal_artifact::compile_directory(&source.0, &directory, 32768).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["run", "--model"])
        .arg(&directory)
        .args([
            "--prompt",
            "x",
            "--temp",
            "0",
            "--context-size",
            "8",
            "--num-tokens",
            "5",
        ])
        .output()
        .unwrap();
    std::fs::remove_dir_all(&directory).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("Tokenizado: 3 tokens"), "{text}");
    assert!(
        text.contains("Generados 5 tokens desde artefacto compilado"),
        "{text}"
    );
    assert!(!text.contains("[EOS]"), "{text}");
}
