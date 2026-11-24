//! Real HTTP/SSE requests to the server executable using a persisted artifact.
use bml_compiler::{
    causal_artifact::{compile_directory, MANIFEST},
    BmlGraph, Fragment, RpnOp,
};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Lines},
    net::TcpStream,
    process::{Child, ChildStdout, Command},
};
#[allow(dead_code)]
#[path = "../../inference/tests/support/gguf_fixture.rs"]
mod gguf_fixture;
#[path = "../../cli/tests/support/token_fixture.rs"]
mod token_fixture;
struct Artifact(PathBuf);
impl Drop for Artifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn start(directory: &Path, weights: &Path) -> (Child, Lines<BufReader<ChildStdout>>, u16) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_bml-server"))
        .arg("--model")
        .arg(directory)
        .arg("--weights")
        .arg(weights)
        .args(["--port", "0"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let port = tokio::time::timeout(Duration::from_secs(15), async {
        while let Some(line) = lines.next_line().await.unwrap() {
            if let Some(address) = line.strip_prefix("Server listening on http://") {
                return address.parse::<std::net::SocketAddr>().unwrap().port();
            }
        }
        panic!("server exited before binding its listener");
    })
    .await
    .expect("server startup deadline");
    (child, lines, port)
}

async fn request_json(port: u16, payload: serde_json::Value) -> (String, Vec<u8>) {
    tokio::time::timeout(Duration::from_secs(15), async {
        let mut socket = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
        let body = serde_json::to_vec(&payload).unwrap();
        let head = format!("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
        socket.write_all(head.as_bytes()).await.unwrap();
        socket.write_all(&body).await.unwrap();
        let mut response = Vec::new();
        socket.read_to_end(&mut response).await.unwrap();
        let split = response.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = String::from_utf8(response[..split].to_vec()).unwrap();
        let mut body = &response[split + 4..];
        if !head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
            return (head, body.to_vec());
        }
        let mut decoded = Vec::new();
        loop {
            let end = body.windows(2).position(|w| w == b"\r\n").unwrap();
            let count = usize::from_str_radix(
                std::str::from_utf8(&body[..end])
                    .unwrap()
                    .split(';')
                    .next()
                    .unwrap(),
                16,
            )
            .unwrap();
            body = &body[end + 2..];
            if count == 0 {
                break;
            }
            decoded.extend_from_slice(&body[..count]);
            assert_eq!(&body[count..count + 2], b"\r\n");
            body = &body[count + 2..];
        }
        (head, decoded)
    })
    .await
    .expect("HTTP response deadline")
}

async fn request(port: u16, stream: bool, prompt: &str) -> (String, Vec<u8>) {
    request_json(
        port,
        serde_json::json!({"prompt": prompt, "max_tokens": 6, "temperature": 0, "stream": stream}),
    )
    .await
}

#[tokio::test]
async fn compiled_http_and_sse_use_persisted_programs_and_do_not_echo_the_prompt() {
    let source = token_fixture::fixture();
    let directory = Artifact(source.0.with_extension("http.bmlgraph"));
    compile_directory(&source.0, &directory.0, 32768).unwrap();
    let weights = directory.0.join("relocated.gguf");
    std::fs::rename(&source.0, &weights).unwrap();
    let (mut child, _stdout, port) = start(&directory.0, &weights).await;
    let (headers, body) = request(port, false, "x").await;
    assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["choices"][0]["text"], "é");
    let (headers, body) = request(port, true, "x").await;
    assert!(headers
        .to_ascii_lowercase()
        .contains("content-type: text/event-stream"));
    let text = String::from_utf8(body).unwrap();
    let mut pieces = Vec::new();
    let mut done = 0;
    let mut stopped = 0;
    for line in text.lines().filter_map(|s| s.strip_prefix("data: ")) {
        if line == "[DONE]" {
            done += 1;
            continue;
        }
        let event: serde_json::Value = serde_json::from_str(line).unwrap();
        pieces.push(event["choices"][0]["text"].as_str().unwrap().to_owned());
        if event["choices"][0]["finish_reason"] == "stop" {
            stopped += 1;
        }
    }
    assert_eq!(pieces, ["é", ""]);
    assert_eq!(done, 1);
    assert_eq!(stopped, 1);
    let (headers, _) = request(port, false, &"x".repeat(20)).await;
    assert!(headers.starts_with("HTTP/1.1 500"), "{headers}");
    child.kill().await.unwrap();
    child.wait().await.unwrap();

    // A deliberate program change with an updated digest must affect the server.
    // Zero logits select the unknown token, which emits no bytes in this fixture.
    let graph = BmlGraph {
        fragments: vec![Fragment {
            ops: vec![RpnOp::Zero],
        }],
        const_pool: vec![],
        threshold: 32768,
    };
    let bytes = graph.serialize();
    let path = directory.0.join(MANIFEST);
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["programs"][0]["sha256"] = format!("{:x}", Sha256::digest(&bytes)).into();
    std::fs::write(directory.0.join("program_0.bmlgraph"), bytes).unwrap();
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let (mut child, _stdout, port) = start(&directory.0, &weights).await;
    let (headers, body) = request(port, false, "x").await;
    assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["choices"][0]["text"], "");
    child.kill().await.unwrap();
    child.wait().await.unwrap();
}

#[tokio::test]
async fn symbolic_http_and_sse_execute_saved_programs_with_compiled_context_limits() {
    for capacity in [6, 3, 2] {
        let source = token_fixture::fixture();
        let directory = Artifact(source.0.with_extension("symbolic-http.bmlgraph"));
        bml_compiler::symbolic_artifact::compile_directory(
            &source.0,
            &directory.0,
            capacity,
            Default::default(),
        )
        .unwrap();
        let weights = directory.0.join("relocated.gguf");
        std::fs::rename(&source.0, &weights).unwrap();
        let (mut child, _stdout, port) = start(&directory.0, &weights).await;
        let (headers, body) = request(port, false, "x").await;
        if capacity == 2 {
            assert!(headers.starts_with("HTTP/1.1 500"), "{headers}");
            assert!(String::from_utf8(body)
                .unwrap()
                .contains("exceeds model context"));
        } else {
            assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
            let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                json["choices"][0]["text"],
                if capacity == 6 { "é" } else { "" }
            );
        }
        let (headers, body) = request(port, true, "x").await;
        assert!(headers
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"));
        let text = String::from_utf8(body).unwrap();
        let (mut pieces, mut stops, mut errors, mut done) = (Vec::new(), 0, 0, 0);
        for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
            if data == "[DONE]" {
                done += 1;
                continue;
            }
            let event: serde_json::Value = serde_json::from_str(data).unwrap();
            pieces.push(event["choices"][0]["text"].as_str().unwrap().to_owned());
            if let Some(reason) = event["choices"][0]["finish_reason"].as_str() {
                if reason == "stop" {
                    stops += 1;
                }
                if reason.starts_with("error:") {
                    errors += 1;
                }
            }
        }
        assert_eq!(pieces.concat(), if capacity == 6 { "é" } else { "" });
        assert_eq!(pieces.len(), if capacity == 6 { 2 } else { 1 });
        assert_eq!(
            (stops, errors, done),
            if capacity == 2 { (0, 1, 1) } else { (1, 0, 1) }
        );
        child.kill().await.unwrap();
        child.wait().await.unwrap();
        if capacity == 6 {
            // A valid edited program must be executed, not silently regenerated.
            let graph = BmlGraph {
                fragments: vec![Fragment {
                    ops: vec![RpnOp::One],
                }],
                const_pool: vec![],
                threshold: 32768,
            };
            let bytes = graph.serialize();
            let manifest_path = directory.0.join(bml_compiler::symbolic_artifact::MANIFEST);
            let mut manifest: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
            manifest["kernels"][0]["programs"][0] = format!("{:x}", Sha256::digest(&bytes)).into();
            std::fs::write(directory.0.join("kernel_0_0.bmlgraph"), bytes).unwrap();
            std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            let (mut child, _stdout, port) = start(&directory.0, &weights).await;
            let (headers, body) = request(port, false, "x").await;
            assert!(headers.starts_with("HTTP/1.1 200"));
            let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(json["choices"][0]["text"], "");
            child.kill().await.unwrap();
            child.wait().await.unwrap();
            std::fs::write(directory.0.join(MANIFEST), b"{}").unwrap();
            let output = tokio::time::timeout(
                Duration::from_secs(15),
                Command::new(env!("CARGO_BIN_EXE_bml-server"))
                    .arg("--model")
                    .arg(&directory.0)
                    .arg("--weights")
                    .arg(&weights)
                    .args(["--port", "0"])
                    .kill_on_drop(true)
                    .output(),
            )
            .await
            .unwrap()
            .unwrap();
            assert!(!output.status.success());
            assert!(String::from_utf8_lossy(&output.stderr).contains("ambiguous"));
        }
    }
}

#[tokio::test]
async fn server_cli_parses_layer_workers_before_reporting_missing_model() {
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        Command::new(env!("CARGO_BIN_EXE_bml-server"))
            .args(["--layer-workers", "127.0.0.1:9999"])
            .kill_on_drop(true)
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--model <path> es obligatorio"), "{stderr}");
}

#[tokio::test]
async fn remote_layer_worker_http_and_sse_enforce_context_temperature_and_decode_steps() {
    use bml_gguf::GgufReader;
    use bml_inference::{
        model::Model,
        remote::{serve_layer_connection, RemoteLimits},
    };
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    let dir =
        Artifact(std::env::temp_dir().join(format!("bml_remote_server_{}", std::process::id())));
    let _ = std::fs::remove_dir_all(&dir.0);
    std::fs::create_dir(&dir.0).unwrap();
    let model_path = dir.0.join("remote.gguf");
    let mut raw = gguf_fixture::fixture();
    std::fs::write(&model_path, &raw).unwrap();
    let embd_offset = {
        let reader = GgufReader::open(&model_path).unwrap();
        let info = reader.find_tensor("token_embd.weight").unwrap();
        let data = reader.tensor_data(info).unwrap();
        data.as_ptr() as usize - reader.bytes().as_ptr() as usize
    };
    // Row 69 is `<0x41>` ('A'). Align its Q4_0 weights with the fixture embedding sign pattern
    // so greedy decoding deterministically emits 'A' without early EOS.
    let row = embd_offset + 69 * 18;
    raw[row..row + 2].copy_from_slice(&0x4000u16.to_le_bytes());
    raw[row + 2..row + 18].fill(0xf0);
    std::fs::write(&model_path, &raw).unwrap();

    let prompt = "x";
    let (prompt_len, context_len) = {
        let reader = GgufReader::open(&model_path).unwrap();
        let model = Model::from_gguf(&reader).unwrap();
        (
            model.tokenizer().encode(prompt, true).unwrap().len(),
            model.config().context_length,
        )
    };
    let exact_max_tokens = context_len - prompt_len;
    assert!(exact_max_tokens >= 3);
    let expected = "A".repeat(exact_max_tokens);

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let worker_addr = listener.local_addr().unwrap();
    listener.set_nonblocking(true).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let worker_model = model_path.clone();
    let worker = std::thread::spawn(move || {
        let reader = GgufReader::open(&worker_model).unwrap();
        let model = Model::from_gguf(&reader).unwrap();
        while !worker_stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    let _ = serve_layer_connection(&model, 0..1, stream, RemoteLimits::default());
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => break,
            }
        }
    });

    let mut child = Command::new(env!("CARGO_BIN_EXE_bml-server"))
        .arg("--model")
        .arg(&model_path)
        .args(["--port", "0", "--layer-workers", &worker_addr.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let port = tokio::time::timeout(Duration::from_secs(15), async {
        while let Some(line) = lines.next_line().await.unwrap() {
            if let Some(address) = line.strip_prefix("Server listening on http://") {
                return address.parse::<std::net::SocketAddr>().unwrap().port();
            }
        }
        panic!("remote server exited before binding its listener");
    })
    .await
    .expect("remote server startup deadline");

    // Negative temperature must be rejected with 400 Bad Request before scheduling.
    let (headers, _) = request_json(
        port,
        serde_json::json!({"prompt": prompt, "max_tokens": 1, "temperature": -1.0, "stream": false}),
    )
    .await;
    assert!(headers.starts_with("HTTP/1.1 400"), "{headers}");

    // Zero max_tokens must return an empty completion immediately without failing.
    let (headers, body) = request_json(
        port,
        serde_json::json!({"prompt": prompt, "max_tokens": 0, "temperature": 0.0, "stream": false}),
    )
    .await;
    assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["choices"][0]["text"], "");

    // Context strictly above model context_length must be rejected.
    let (headers, body) = request_json(
        port,
        serde_json::json!({"prompt": prompt, "max_tokens": exact_max_tokens + 1, "temperature": 0.0, "stream": false}),
    )
    .await;
    assert!(headers.starts_with("HTTP/1.1 502"), "{headers}");
    assert!(String::from_utf8_lossy(&body).contains("exceeds model context"));

    // Exact context boundary must succeed without issuing an out-of-range extra decode step.
    let (headers, body) = request_json(
        port,
        serde_json::json!({"prompt": prompt, "max_tokens": exact_max_tokens, "temperature": 0.0, "stream": false}),
    )
    .await;
    assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["choices"][0]["text"], expected);

    // Streaming SSE over remote workers must emit every generated token and finish cleanly.
    let (headers, body) = request_json(
        port,
        serde_json::json!({"prompt": prompt, "max_tokens": exact_max_tokens, "temperature": 0.0, "stream": true}),
    )
    .await;
    assert!(headers.starts_with("HTTP/1.1 200"), "{headers}");
    assert!(headers
        .to_ascii_lowercase()
        .contains("content-type: text/event-stream"));
    let text = String::from_utf8(body).unwrap();
    let (mut pieces, mut stops, mut done) = (Vec::new(), 0, 0);
    for data in text.lines().filter_map(|line| line.strip_prefix("data: ")) {
        if data == "[DONE]" {
            done += 1;
            continue;
        }
        let event: serde_json::Value = serde_json::from_str(data).unwrap();
        pieces.push(event["choices"][0]["text"].as_str().unwrap().to_owned());
        if event["choices"][0]["finish_reason"] == "stop" {
            stops += 1;
        }
    }
    assert_eq!(pieces.concat(), expected);
    assert_eq!((stops, done), (1, 1));

    child.kill().await.unwrap();
    child.wait().await.unwrap();
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap();
}
