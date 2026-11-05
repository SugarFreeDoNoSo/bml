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

#[test]
fn symbolic_integer_options_validate_before_help() {
    for (flag, invalid, message) in [
        ("--layer-start", "4294967296", "layer start overflow"),
        ("--layer-end", "4294967296", "layer end overflow"),
        ("--symbolic-context", "4294967296", "context overflow"),
        ("--symbolic-max-steps", "0", "max steps must be positive"),
    ] {
        let rejected = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
            .args([flag, invalid, "--help"])
            .output()
            .unwrap();
        assert_eq!(rejected.status.code(), Some(2), "{flag}");
        assert!(String::from_utf8_lossy(&rejected.stderr).contains(message));
        let accepted = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
            .args([flag, "1", "--help"])
            .output()
            .unwrap();
        assert!(accepted.status.success(), "{flag}");
    }
}

#[test]
fn prepared_plan_rejects_each_explicit_artifact_or_range_option_before_io() {
    let absent = std::env::temp_dir().join(format!(
        "bml-cli-conflicting-plan-absent-{}",
        std::process::id()
    ));
    assert!(!absent.exists());
    let missing = absent.to_str().unwrap();
    // The nonexistent plan makes falling through the conflict guard fail
    // promptly at file I/O, before any listener or model can be created.
    let control = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
        .args(["--prepared", missing, "--worker-index", "0"])
        .output()
        .unwrap();
    assert_eq!(control.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&control.stderr).contains("invalid prepared plan"));
    for (flag, value) in [
        ("--symbolic-artifact", missing),
        ("--symbolic-source", missing),
        ("--symbolic-shard", missing),
        ("--symbolic-shard-sha256", "abc"),
        ("--symbolic-artifact-sha256", "abc"),
        ("--layer-start", "0"),
        ("--layer-end", "1"),
        ("--fragment", missing),
    ] {
        let rejected = Command::new(env!("CARGO_BIN_EXE_bml-worker"))
            .args(["--prepared", missing, "--worker-index", "0", flag, value])
            .output()
            .unwrap();
        assert_eq!(rejected.status.code(), Some(2), "{flag}");
        assert!(String::from_utf8_lossy(&rejected.stderr)
            .contains("prepared plan conflicts with explicit artifact, shard or range options"));
        assert!(rejected.stdout.is_empty());
    }
    assert!(!absent.exists());
}
