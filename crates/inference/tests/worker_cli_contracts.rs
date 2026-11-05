use std::process::Command;

#[test]
fn causal_worker_accepts_each_resource_option_before_help() {
    // Help ends before model loading or listening; every resource switch must
    // be recognized even when a worker is not launched.
    for flag in ["--memory-bytes", "--frame-bytes", "--timeout-ms"] {
        let output = Command::new(env!("CARGO_BIN_EXE_bml-layer-worker"))
            .args([flag, "1", "--help"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{flag}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains(flag));
    }
}

#[test]
fn causal_worker_rejects_each_zero_resource_before_opening_a_model() {
    let absent = std::env::temp_dir().join(format!(
        "bml-causal-worker-absent-{}.gguf",
        std::process::id()
    ));
    assert!(!absent.exists());
    for flag in [
        "--max-sessions",
        "--context",
        "--memory-bytes",
        "--frame-bytes",
        "--timeout-ms",
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_bml-layer-worker"))
            .args([
                "--model",
                absent.to_str().unwrap(),
                "--layers",
                "0:1",
                flag,
                "0",
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(1), "{flag}");
        assert!(String::from_utf8_lossy(&output.stderr)
            .contains("worker limits must be positive and frames fit u32"));
        assert!(output.stdout.is_empty());
    }
    assert!(!absent.exists());
}
