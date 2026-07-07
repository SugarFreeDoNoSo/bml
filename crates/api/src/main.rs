//! # bml-server
//!
//! API HTTP OpenAI-compatible con streaming SSE, batching y backpressure.
//!
//! Endpoints:
//! - `POST /v1/completions` — genera texto a partir de un prompt.
//! - `GET /health` — health check.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use bml_compiler::gguf_compiler::InferenceCompiler;
use bml_compiler::sampler::sample;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

const MAX_PENDING_REQUESTS: usize = 64;

struct ServerState {
    compiler: Mutex<InferenceCompiler>,
    pending_requests: AtomicUsize,
}

#[derive(Debug, Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temp")]
    temperature: f64,
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> u32 {
    128
}
fn default_temp() -> f64 {
    0.8
}

#[derive(Debug, Serialize)]
struct CompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Serialize)]
struct Choice {
    text: String,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct StreamEvent {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    text: String,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = PathBuf::from("model.gguf");
    let mut port: u16 = 8080;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                model_path = PathBuf::from(&args[i]);
            }
            "--port" | "-p" => {
                i += 1;
                port = args[i].parse().unwrap_or(8080);
            }
            _ => {}
        }
        i += 1;
    }

    println!("Loading model from {:?}...", model_path);
    let compiler = InferenceCompiler::open(&model_path)
        .map_err(|e| format!("Error loading model: {e}"))?;

    let config = compiler.config();
    println!(
        "Model: {} ({} layers, {} heads, {} embd)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd
    );
    println!("Max pending requests: {MAX_PENDING_REQUESTS}");

    let state = Arc::new(ServerState {
        compiler: Mutex::new(compiler),
        pending_requests: AtomicUsize::new(0),
    });

    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/health", get(health))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("Server listening on http://0.0.0.0:{port}");
    println!("  POST /v1/completions");
    println!("  GET  /health");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let pending = state.pending_requests.fetch_add(1, Ordering::SeqCst);
    if pending >= MAX_PENDING_REQUESTS {
        state.pending_requests.fetch_sub(1, Ordering::SeqCst);
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "Server is at capacity. Please retry later.".to_string(),
                    error_type: "rate_limit_error".to_string(),
                },
            }),
        )
            .into_response();
    }

    if req.stream {
        let (tx, rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(256);

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            let result = generate_streaming(&state_clone, &req, tx.clone());
            if let Err(e) = result {
                let event = StreamEvent {
                    choices: vec![StreamChoice {
                        text: String::new(),
                        finish_reason: Some(format!("error: {e}")),
                    }],
                };
                let json = serde_json::to_string(&event).unwrap();
                let _ = tx.send(Ok(format!("data: {json}\n\n"))).await;
            }
            let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;
            state_clone.pending_requests.fetch_sub(1, Ordering::SeqCst);
        });

        let stream = ReceiverStream::new(rx);
        Sse::new(
            stream.map(|item| item.map(|data| axum::response::sse::Event::default().data(data))),
        )
        .into_response()
    } else {
        let text = generate_text(&state, &req);
        state.pending_requests.fetch_sub(1, Ordering::SeqCst);

        let response = CompletionResponse {
            choices: vec![Choice {
                text,
                finish_reason: "stop".to_string(),
            }],
        };
        Json(response).into_response()
    }
}

async fn health(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let pending = state.pending_requests.load(Ordering::SeqCst);
    let config = state.compiler.lock().unwrap().config().clone();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "model": config.architecture,
            "layers": config.n_layers,
            "fragments": 1,
            "pending_requests": pending,
            "capacity": MAX_PENDING_REQUESTS,
        })),
    )
}

fn generate_text(state: &ServerState, req: &CompletionRequest) -> String {
    let compiler = state.compiler.lock().unwrap();
    let vocab = compiler.vocab();
    let input_ids = vocab.encode(&req.prompt);
    let mut tokens = input_ids.clone();
    let mut output_parts: Vec<String> = Vec::new();
    let max = req.max_tokens as usize;

    for _ in 0..max {
        let logits = compiler.forward(&tokens);
        if logits.is_empty() {
            break;
        }
        let token_id = match sample(&logits, req.temperature, 0) {
            Some(id) => id,
            None => break,
        };
        let piece = vocab.decode_single(token_id);
        output_parts.push(piece.to_string());
        tokens.push(token_id);
        if tokens.len() > 4096 {
            tokens.drain(0..(tokens.len() - 2048));
        }
    }
    output_parts.join("")
}

fn generate_streaming(
    state: &ServerState,
    req: &CompletionRequest,
    tx: mpsc::Sender<Result<String, std::convert::Infallible>>,
) -> Result<(), String> {
    let compiler = state.compiler.lock().map_err(|e| format!("lock: {e}"))?;
    let vocab = compiler.vocab();
    let input_ids = vocab.encode(&req.prompt);
    let mut tokens = input_ids.clone();
    let max = req.max_tokens as usize;

    let first = StreamEvent {
        choices: vec![StreamChoice {
            text: req.prompt.clone(),
            finish_reason: None,
        }],
    };
    let json = serde_json::to_string(&first).unwrap();
    let _ = tx.blocking_send(Ok(format!("data: {json}\n\n")));

    for _ in 0..max {
        let logits = compiler.forward(&tokens);
        if logits.is_empty() {
            break;
        }
        let token_id = match sample(&logits, req.temperature, 0) {
            Some(id) => id,
            None => break,
        };
        let piece = vocab.decode_single(token_id);
        let event = StreamEvent {
            choices: vec![StreamChoice {
                text: piece.to_string(),
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&event).unwrap();
        if tx.blocking_send(Ok(format!("data: {json}\n\n"))).is_err() {
            break;
        }
        tokens.push(token_id);
        if tokens.len() > 4096 {
            tokens.drain(0..(tokens.len() - 2048));
        }
    }

    let done = StreamEvent {
        choices: vec![StreamChoice {
            text: String::new(),
            finish_reason: Some("stop".to_string()),
        }],
    };
    let json = serde_json::to_string(&done).unwrap();
    let _ = tx.blocking_send(Ok(format!("data: {json}\n\n")));
    Ok(())
}