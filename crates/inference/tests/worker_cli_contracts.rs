#[allow(dead_code)]
#[path = "support/gguf_fixture.rs"]
mod gguf_fixture;

use serde_json::Value;
use std::{
    io::{BufRead, BufReader, Read},
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Child, Command, Output, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct ChildGuard(Option<Child>);
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn run_expecting_exit(command: &mut Command, timeout: Duration) -> Output {
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut out = child.stdout.take().unwrap();
    let mut err = child.stderr.take().unwrap();
    let stdout_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = out.read_to_end(&mut buf);
        buf
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = err.read_to_end(&mut buf);
        buf
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            let stdout = stdout_thread.join().unwrap_or_default();
            let stderr = stderr_thread.join().unwrap_or_default();
            panic!(
                "bml-layer-worker did not exit within {:?}; stdout={}; stderr={}",
                timeout,
                String::from_utf8_lossy(&stdout),
                String::from_utf8_lossy(&stderr),
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    Output {
        status,
        stdout: stdout_thread.join().unwrap(),
        stderr: stderr_thread.join().unwrap(),
    }
}

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

#[test]
fn causal_worker_enforces_u32_frame_budget_and_layer_range_bounds() {
    let absent = std::env::temp_dir().join(format!(
        "bml-causal-worker-absent-frame-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    assert!(!absent.exists());
    #[cfg(target_pointer_width = "64")]
    {
        for too_large in [u64::from(u32::MAX) + 1, u64::from(u32::MAX) + 2] {
            let output = Command::new(env!("CARGO_BIN_EXE_bml-layer-worker"))
                .args([
                    "--model",
                    absent.to_str().unwrap(),
                    "--layers",
                    "0:1",
                    "--frame-bytes",
                    &too_large.to_string(),
                ])
                .output()
                .unwrap();
            assert_eq!(output.status.code(), Some(1), "frame-bytes={too_large}");
            assert!(String::from_utf8_lossy(&output.stderr)
                .contains("worker limits must be positive and frames fit u32"));
        }
    }
    let exact_max = Command::new(env!("CARGO_BIN_EXE_bml-layer-worker"))
        .args([
            "--model",
            absent.to_str().unwrap(),
            "--layers",
            "0:1",
            "--frame-bytes",
            &u32::MAX.to_string(),
        ])
        .output()
        .unwrap();
    assert_eq!(exact_max.status.code(), Some(1));
    assert!(!String::from_utf8_lossy(&exact_max.stderr)
        .contains("worker limits must be positive and frames fit u32"));

    let model = TempFile(std::env::temp_dir().join(format!(
        "bml-causal-worker-fixture-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )));
    std::fs::write(&model.0, gguf_fixture::fixture()).unwrap();
    for bad_layers in ["1:1", "0:2"] {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_bml-layer-worker"));
        cmd.args([
            "--model",
            model.0.to_str().unwrap(),
            "--layers",
            bad_layers,
            "--listen",
            "127.0.0.1:0",
            "--once",
        ]);
        let output = run_expecting_exit(&mut cmd, Duration::from_secs(10));
        assert_eq!(output.status.code(), Some(1), "layers={bad_layers}");
        assert!(String::from_utf8_lossy(&output.stderr).contains("layer range outside model"));
        assert!(output.stdout.is_empty());
    }

    let mut child = ChildGuard(Some(
        Command::new(env!("CARGO_BIN_EXE_bml-layer-worker"))
            .args([
                "--model",
                model.0.to_str().unwrap(),
                "--layers",
                "0:1",
                "--listen",
                "127.0.0.1:0",
                "--once",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ));
    let mut line = String::new();
    let stdout = child.0.as_mut().unwrap().stdout.take().unwrap();
    BufReader::new(stdout).read_line(&mut line).unwrap();
    let info: Value = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(info["layers"], serde_json::json!([0, 1]));
    let address: SocketAddr = info["listening"].as_str().unwrap().parse().unwrap();
    let stream = TcpStream::connect(address).unwrap();
    drop(stream);
    let status = child.0.take().unwrap().wait().unwrap();
    assert!(status.success());
}
