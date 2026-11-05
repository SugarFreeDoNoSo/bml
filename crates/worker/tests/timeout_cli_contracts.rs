use std::process::Command;

#[test]
fn symbolic_timeout_must_be_positive_before_help_can_complete() {
    // Help prevents every control from binding a port or loading a model,
    // including if the timeout arm incorrectly falls through to a KV budget.
    let invalid = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
        .args(["--symbolic-timeout-seconds", "0", "--help"])
        .output()
        .unwrap();
    assert_eq!(invalid.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("timeout must be positive"));
    assert!(invalid.stdout.is_empty());
    for seconds in ["1", "600"] {
        let valid = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
            .args(["--symbolic-timeout-seconds", seconds, "--help"])
            .output()
            .unwrap();
        assert!(
            valid.status.success(),
            "{}",
            String::from_utf8_lossy(&valid.stderr)
        );
        assert!(String::from_utf8_lossy(&valid.stdout).contains("--symbolic-timeout-seconds"));
    }
}
