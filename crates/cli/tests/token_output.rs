//! Real CLI IO with a zero-layer projection fixture, not full-model validation.
use std::process::Command;
#[path = "support/token_fixture.rs"]
mod token_fixture;
use token_fixture::fixture;

#[test]
fn cli_preserves_multibyte_token_pieces_and_stops_at_eos() {
    let model = fixture();
    for rpn in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_bml-cli"));
        command.arg("run").arg("--model").arg(&model.0).args([
            "--prompt",
            "x",
            "--num-tokens",
            "6",
            "--temp",
            "0",
            "--context-size",
            "8",
        ]);
        if rpn {
            command.arg("--bml");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8(output.stdout).unwrap();
        assert!(text.contains("Tokenizado: 3 tokens"), "{text}"); // BOS, space, x
        assert!(text.contains("\né\n[EOS]"), "{text}");
        assert!(text.contains("Generados 3 tokens"), "{text}");
        assert!(!text.contains("<unk>"), "{text}");
    }
}

#[test]
fn cli_rpn_enforces_context_without_truncating_prompt_history() {
    let model = fixture();
    for (capacity, success, generated) in [(0, false, 0), (2, false, 0), (3, true, 0), (4, true, 1)]
    {
        let output = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .arg("run")
            .arg("--model")
            .arg(&model.0)
            .args([
                "--bml",
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
            success,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if success {
            // A one-token boundary may legitimately end with an incomplete UTF-8 byte.
            assert!(String::from_utf8_lossy(&output.stdout)
                .contains(&format!("Generados {generated} tokens")));
        }
    }
}
