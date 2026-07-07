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
use bml_compiler::gguf_compiler::{load_from_dir, ModelConfig};
use bml_runtime::Runtime;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

/// Capacidad máxima de la cola de requests pendientes.
const MAX_PENDING_REQUESTS: usize = 64;

struct ServerState {
    graph: bml_compiler::BmlGraph,
    const_pool: Vec<f64>,
    config: ModelConfig,
    runtime: Mutex<Runtime>,
    /// Número de requests pendientes (para backpressure).
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
    let mut model_path = PathBuf::from("model.bmlgraph");
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
    let (graph, const_pool, config) =
        load_from_dir(&model_path).map_err(|e| format!("Error loading model: {e}"))?;

    println!(
        "Model: {} ({} layers, {} heads, {} embd)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd
    );
    println!("Fragments: {}", graph.num_fragments());
    println!("Max pending requests: {MAX_PENDING_REQUESTS}");

    let state = Arc::new(ServerState {
        graph,
        const_pool,
        config,
        runtime: Mutex::new(Runtime::new(8192, 64)),
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
    // Backpressure: si hay demasiados requests pendientes, rechazar.
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
        let (tx, rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(32);

        let state_clone = Arc::clone(&state);
        tokio::spawn(async move {
            // Generar tokens (placeholder)
            let tokens = vec!["B", "M", "L"];
            for token in tokens {
                let event = StreamEvent {
                    choices: vec![StreamChoice {
                        text: token.to_string(),
                        finish_reason: None,
                    }],
                };
                let json = serde_json::to_string(&event).unwrap();
                let _ = tx.send(Ok(format!("data: {json}\n\n"))).await;
            }
            let done = StreamEvent {
                choices: vec![StreamChoice {
                    text: String::new(),
                    finish_reason: Some("stop".to_string()),
                }],
            };
            let json = serde_json::to_string(&done).unwrap();
            let _ = tx.send(Ok(format!("data: {json}\n\n"))).await;
            let _ = tx.send(Ok("data: [DONE]\n\n".to_string())).await;

            // Liberar slot de backpressure
            state_clone.pending_requests.fetch_sub(1, Ordering::SeqCst);
        });

        let stream = ReceiverStream::new(rx);
        Sse::new(
            stream.map(|item| item.map(|data| axum::response::sse::Event::default().data(data))),
        )
        .into_response()
    } else {
        let result = generate_tokens(&state, &req);
        state.pending_requests.fetch_sub(1, Ordering::SeqCst);

        let response = CompletionResponse {
            choices: vec![Choice {
                text: result,
                finish_reason: "stop".to_string(),
            }],
        };
        Json(response).into_response()
    }
}

async fn health(State(state): State<Arc<ServerState>>) -> impl IntoResponse {
    let pending = state.pending_requests.load(Ordering::SeqCst);
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ok",
            "model": state.config.architecture,
            "layers": state.config.n_layers,
            "fragments": state.graph.num_fragments(),
            "pending_requests": pending,
            "capacity": MAX_PENDING_REQUESTS,
        })),
    )
}

fn generate_tokens(state: &ServerState, req: &CompletionRequest) -> String {
    let inputs = vec![req.prompt.len() as f64];
    let ctx = bml_domain::EvalContext::new(&inputs, &state.const_pool);
    let _result = state
        .runtime
        .lock()
        .unwrap()
        .execute_graph_with_ctx(&state.graph, &ctx);
    format!(
        "{} [BML placeholder: {} tokens]",
        req.prompt, req.max_tokens
    )
}
