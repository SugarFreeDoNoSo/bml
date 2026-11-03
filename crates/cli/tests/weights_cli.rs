use std::{
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml-weights-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn source(&self) -> PathBuf {
        self.0.join("source.gguf")
    }
    fn output(&self) -> PathBuf {
        self.0.join("output.bmlgguf")
    }
    fn write(&self, values: &[f32]) {
        // Independent GGUF v3: no metadata, one F32 vector, axis 0 contiguous.
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.push(b'w');
        bytes.extend(1u32.to_le_bytes());
        bytes.extend((values.len() as u64).to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        for value in values {
            bytes.extend(value.to_le_bytes());
        }
        std::fs::write(self.source(), bytes).unwrap();
    }
    fn convert(&self, extra: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["convert-weights", "-m"])
            .arg(self.source())
            .arg("-o")
            .arg(self.output())
            .args(extra)
            .output()
            .unwrap()
    }
    fn inspect(&self, extra: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args(["inspect-weights", "-m"])
            .arg(self.output())
            .args(extra)
            .output()
            .unwrap()
    }
    fn clean_failure(&self) {
        assert!(!self.output().exists());
        assert_eq!(std::fs::read_dir(&self.0).unwrap().count(), 1);
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn stdout(result: &Output) -> String {
    String::from_utf8_lossy(&result.stdout).into_owned()
}
fn stderr(result: &Output) -> String {
    String::from_utf8_lossy(&result.stderr).into_owned()
}
fn success(result: &Output) {
    assert!(
        result.status.success(),
        "{}\n{}",
        stdout(result),
        stderr(result)
    );
}

#[test]
fn converts_and_inspects_without_source_with_explicit_budgets() {
    let f = Fixture::new();
    f.write(&[1., 2., 1.]);
    let result = f.convert(&["--max-block-terms", "1", "--max-cache-bytes", "1048576"]);
    success(&result);
    assert!(stdout(&result).contains("weights: 3"));
    assert!(stdout(&result).contains("weights_reused: 1"));
    assert!(stdout(&result).contains("output_sha256:"));
    std::fs::remove_file(f.source()).unwrap();
    let result = f.inspect(&[]);
    success(&result);
    assert!(stdout(&result).contains("payload: not-verified"));
    assert!(stdout(&result).contains("ggml_dimensions=[3]"));
    let result = f.inspect(&["--verify-all"]);
    success(&result);
    assert!(stdout(&result).contains("payload: verified-all; blocks=3 terms=3"));
}

#[test]
fn corrupt_payload_is_distinguished_from_valid_catalog() {
    let f = Fixture::new();
    f.write(&[1.]);
    success(&f.convert(&[]));
    let mut bytes = std::fs::read(f.output()).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(f.output(), bytes).unwrap();
    success(&f.inspect(&[]));
    let result = f.inspect(&["--verify-all"]);
    assert_eq!(result.status.code(), Some(1));
    assert!(stderr(&result).contains("integrity failure"));
    assert!(!stdout(&result).contains("payload: verified-all"));
}

#[test]
fn failed_conversion_preserves_files_and_reports_localized_errors() {
    let f = Fixture::new();
    f.write(&[1., f32::INFINITY]);
    let original = std::fs::read(f.source()).unwrap();
    let result = f.convert(&["--max-block-terms", "1"]);
    assert_eq!(result.status.code(), Some(1));
    assert!(stderr(&result).contains("nonfinite weight"));
    assert!(stderr(&result).contains("flat_index: 1"));
    f.clean_failure();
    assert_eq!(std::fs::read(f.source()).unwrap(), original);
    std::fs::write(f.output(), b"user data").unwrap();
    let result = f.convert(&[]);
    assert_eq!(result.status.code(), Some(1));
    assert!(stderr(&result).contains("destination already exists"));
    assert_eq!(std::fs::read(f.output()).unwrap(), b"user data");
}

#[test]
fn cli_limits_and_argument_errors_fail_without_publication() {
    let f = Fixture::new();
    f.write(&[1., 1., 1.]);
    let result = f.convert(&[
        "--max-synthesis-steps-total",
        "2",
        "--max-cache-bytes",
        "1048576",
        "--max-block-terms",
        "1",
    ]);
    assert_eq!(result.status.code(), Some(1));
    assert!(stderr(&result).contains("MaxSynthesisStepsTotal"));
    assert!(stderr(&result).contains("flat_index: 2"));
    f.clean_failure();
    for args in [["--max-resident-bytes", "1"], ["--max-block-terms", "0"]] {
        assert_eq!(f.convert(&args).status.code(), Some(1));
        f.clean_failure();
    }
    assert_eq!(
        f.convert(&["--max-cache-bytes", "invalid"]).status.code(),
        Some(2)
    );
    f.clean_failure();
    success(&f.convert(&[]));
    let result = f.inspect(&["--max-total-terms", "2"]);
    assert_eq!(result.status.code(), Some(1));
    assert!(stderr(&result).contains("MaxTotalTerms"));
}

#[test]
fn help_exposes_limits_and_verification_semantics() {
    for (command, expected) in [
        (
            "convert-weights",
            vec![
                "--max-cache-bytes",
                "--max-synthesis-steps-total",
                "--max-output-bytes",
                "--max-resident-bytes",
            ],
        ),
        (
            "inspect-weights",
            vec!["--verify-all", "--max-metadata-bytes", "--max-total-terms"],
        ),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
            .args([command, "--help"])
            .output()
            .unwrap();
        success(&result);
        for flag in expected {
            assert!(stdout(&result).contains(flag));
        }
    }
}
