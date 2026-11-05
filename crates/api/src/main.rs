//! # bml-server
//!
//! API HTTP OpenAI-compatible con streaming SSE, batching y backpressure.
//!
//! Soporta dos modos de carga del modelo:
//! - `.bmlgraph/` — programas persistidos y pesos GGUF verificados por SHA-256
//! - `.gguf` — inferencia causal nativa con pesos mapeados
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
use bml_compiler::gguf_compiler::{InferenceCompiler, ModelConfig};
use bml_compiler::sampler::sample;
use bml_runtime::causal_artifact::CompiledCausalModel;
use bml_runtime::scheduler::{
    AdmissionError, RequestBatchConfig, RequestBatcher, RequestSubmitter,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

const MAX_PENDING_REQUESTS: usize = 64;

enum ModelBackend {
    Local(Mutex<LocalModel>),
    #[cfg(test)]
    Unavailable,
    Remote {
        reader: bml_gguf::GgufReader,
        replicas: Vec<Vec<SocketAddr>>,
    },
}

enum LocalModel {
    Native(Box<InferenceCompiler>),
    Compiled(Box<CompiledCausalModel>),
    Symbolic(Box<bml_runtime::symbolic_causal::SymbolicCausalModel>),
}
impl LocalModel {
    fn compiler(&self) -> &InferenceCompiler {
        match self {
            Self::Native(model) => model,
            Self::Compiled(model) => model.compiler(),
            Self::Symbolic(model) => model.compiler(),
        }
    }
    fn context_limit(&self) -> u32 {
        match self {
            Self::Symbolic(model) => model.context_limit(),
            _ => self.compiler().config().context_length,
        }
    }
    fn forward_cached(
        &mut self,
        ids: &[u32],
        cache: &mut bml_compiler::HashConsedKV,
    ) -> Result<Vec<f64>, String> {
        match self {
            Self::Native(model) => model.forward_cached(ids, cache),
            Self::Compiled(model) => model.forward_cached(ids, cache),
            Self::Symbolic(model) => model.forward_cached(ids, cache),
        }
    }
}

fn load_local_backend(
    path: &Path,
    weights: Option<&Path>,
) -> Result<(ModelBackend, ModelConfig), String> {
    let local = if path.is_dir() {
        let native = path.join(bml_compiler::causal_artifact::MANIFEST).is_file();
        let symbolic = path
            .join(bml_compiler::symbolic_artifact::MANIFEST)
            .is_file();
        match (native, symbolic) {
            (true, false) => LocalModel::Compiled(Box::new(CompiledCausalModel::open(path, weights)?)),
            (false, true) => LocalModel::Symbolic(Box::new(bml_runtime::symbolic_causal::SymbolicCausalModel::open_artifact(path, weights, Default::default())?)),
            (true, true) => return Err("ambiguous native/symbolic artifact profiles".into()),
            _ => return Err("legacy scalar directory is not a causal model; recompile the GGUF into a new directory".into()),
        }
    } else {
        if weights.is_some() {
            return Err("--weights requires a compiled causal artifact".into());
        }
        LocalModel::Native(Box::new(InferenceCompiler::open_mapped(path)?))
    };
    let mut config = local.compiler().config().clone();
    config.context_length = local.context_limit();
    Ok((ModelBackend::Local(Mutex::new(local)), config))
}

struct ServerState {
    backend: ModelBackend,
    config: ModelConfig,
    pending_requests: Arc<AtomicUsize>,
    dispatched_batches: AtomicUsize,
    requests: RequestSubmitter<QueuedRequest>,
}

// Counts queued and executing requests, including streams until completion.
// Ownership follows the request, so rejection/cancellation/unwind releases it.
struct PendingRequest(Arc<AtomicUsize>);
impl Drop for PendingRequest {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}
struct QueuedRequest {
    request: CompletionRequest,
    response: oneshot::Sender<Response>,
    _pending: PendingRequest,
}

fn start_scheduler(state: &Arc<ServerState>, mut batches: RequestBatcher<QueuedRequest>) {
    let state = Arc::downgrade(state);
    std::thread::spawn(move || {
        while let Some(batch) = batches.next_batch() {
            let Some(state) = state.upgrade() else {
                break;
            };
            state.dispatched_batches.fetch_add(1, Ordering::SeqCst);
            // Every request in a batch uses its selected replica. Numerical
            // forward remains per request; this is not fused tensor batching.
            for job in batch.requests {
                if job.response.is_closed() {
                    continue;
                }
                execute_request(&state, job, batch.node);
            }
        }
    });
}

fn execute_request(state: &ServerState, job: QueuedRequest, node: usize) {
    let req = &job.request;
    match &state.backend {
        ModelBackend::Local(_) if req.stream => {
            let (tx, rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(256);
            let stream = ReceiverStream::new(rx);
            let response = Sse::new(
                stream
                    .map(|item| item.map(|data| axum::response::sse::Event::default().data(data))),
            )
            .into_response();
            if job.response.send(response).is_err() {
                return;
            }
            if let Err(error) = generate_streaming_local(state, req, tx.clone()) {
                let event = StreamEvent {
                    choices: vec![StreamChoice {
                        text: String::new(),
                        finish_reason: Some(format!("error: {error}")),
                    }],
                };
                let _ = tx.blocking_send(Ok(serde_json::to_string(&event).unwrap()));
            }
            let _ = tx.blocking_send(Ok("[DONE]".into()));
        }
        ModelBackend::Local(_) => {
            let response = match generate_text_local(state, req) {
                Ok(text) => Json(CompletionResponse {
                    choices: vec![Choice {
                        text,
                        finish_reason: "stop".into(),
                    }],
                })
                .into_response(),
                Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error).into_response(),
            };
            let _ = job.response.send(response);
        }
        ModelBackend::Remote { reader, replicas } => {
            let Some(endpoints) = replicas.get(node) else {
                let _ = job
                    .response
                    .send(StatusCode::SERVICE_UNAVAILABLE.into_response());
                return;
            };
            if req.stream {
                let (tx, rx) = mpsc::channel::<Result<String, std::convert::Infallible>>(256);
                let response =
                    Sse::new(ReceiverStream::new(rx).map(|item| {
                        item.map(|data| axum::response::sse::Event::default().data(data))
                    }))
                    .into_response();
                if job.response.send(response).is_err() {
                    return;
                }
                let result = generate_remote(reader, endpoints, req, |piece| {
                    let event = StreamEvent {
                        choices: vec![StreamChoice {
                            text: piece.into(),
                            finish_reason: None,
                        }],
                    };
                    tx.blocking_send(Ok(serde_json::to_string(&event).unwrap()))
                        .is_ok()
                });
                let reason = match result {
                    Ok(_) => "stop".into(),
                    Err(error) => format!("error: {error}"),
                };
                let event = StreamEvent {
                    choices: vec![StreamChoice {
                        text: String::new(),
                        finish_reason: Some(reason),
                    }],
                };
                let _ = tx.blocking_send(Ok(serde_json::to_string(&event).unwrap()));
                let _ = tx.blocking_send(Ok("[DONE]".into()));
            } else {
                let response = match generate_remote(reader, endpoints, req, |_| true) {
                    Ok(text) => Json(CompletionResponse {
                        choices: vec![Choice {
                            text,
                            finish_reason: "stop".into(),
                        }],
                    })
                    .into_response(),
                    Err(error) => (
                        StatusCode::BAD_GATEWAY,
                        Json(ErrorResponse {
                            error: ErrorDetail {
                                message: error.to_string(),
                                error_type: "remote_inference_error".into(),
                            },
                        }),
                    )
                        .into_response(),
                };
                let _ = job.response.send(response);
            }
        }
        #[cfg(test)]
        ModelBackend::Unavailable => {
            let response = (
                StatusCode::NOT_IMPLEMENTED,
                Json(serde_json::json!({
                    "error": "test backend is unavailable",
                })),
            )
                .into_response();
            let _ = job.response.send(response);
        }
    }
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
    let mut model_path: Option<PathBuf> = None;
    let mut weights_path: Option<PathBuf> = None;
    let mut port: u16 = 8080;
    let mut replicas: Vec<Vec<SocketAddr>> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                i += 1;
                if i < args.len() {
                    model_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--weights" => {
                i += 1;
                weights_path = Some(PathBuf::from(
                    args.get(i).ok_or("--weights needs a GGUF path")?,
                ));
            }
            "--layer-workers" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or("--layer-workers needs comma-separated IP:PORT endpoints")?;
                replicas.push(
                    value
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
            "--port" | "-p" => {
                i += 1;
                port = args[i].parse().unwrap_or(8080);
            }
            _ => {}
        }
        i += 1;
    }

    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("Error: --model <path> es obligatorio");
            eprintln!("Uso: bml-server -m <modelo.bmlgraph/ o modelo.gguf> [--port 8080] [--weights modelo.gguf] [--layer-workers IP:PORT,IP:PORT]");
            std::process::exit(1);
        }
    };

    if !model_path.exists() {
        eprintln!("Error: el modelo no existe: {}", model_path.display());
        std::process::exit(1);
    }

    let (backend, config) = if !replicas.is_empty() {
        if weights_path.is_some() {
            return Err("--weights is only supported by local compiled artifacts".into());
        }
        let reader = bml_gguf::GgufReader::open(&model_path)?;
        let model = bml_inference::model::Model::from_gguf(&reader)?;
        let c = model.config();
        let config = ModelConfig {
            architecture: "llama".into(),
            n_layers: c.block_count.try_into()?,
            n_heads: c.head_count.try_into()?,
            n_embd: c.embedding_length.try_into()?,
            context_length: c.context_length.try_into()?,
            vocab_size: c.vocab_size.try_into()?,
        };
        (ModelBackend::Remote { reader, replicas }, config)
    } else {
        println!("Cargando modelo local desde {}...", model_path.display());
        load_local_backend(&model_path, weights_path.as_deref())?
    };

    println!(
        "Modelo: {} ({} capas, {} heads, {} embd)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd
    );
    println!("Max pending requests: {MAX_PENDING_REQUESTS}");

    let nodes = match &backend {
        ModelBackend::Remote { replicas, .. } => replicas.len(),
        _ => 1,
    };
    let (requests, batches) = RequestBatcher::new(RequestBatchConfig {
        nodes,
        ..Default::default()
    })
    .unwrap();
    let state = Arc::new(ServerState {
        backend,
        config,
        pending_requests: Arc::new(AtomicUsize::new(0)),
        dispatched_batches: AtomicUsize::new(0),
        requests,
    });

    start_scheduler(&state, batches);
    let app = Router::new()
        .route("/v1/completions", post(completions))
        .route("/health", get(health))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    println!("Server listening on http://{}", listener.local_addr()?);
    println!("  POST /v1/completions");
    println!("  GET  /health");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn completions(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if !req.temperature.is_finite() || req.temperature < 0.0 {
        return (
            StatusCode::BAD_REQUEST,
            "temperature must be finite and nonnegative",
        )
            .into_response();
    }
    let (response, result) = oneshot::channel();
    state.pending_requests.fetch_add(1, Ordering::SeqCst);
    let job = QueuedRequest {
        request: req,
        response,
        _pending: PendingRequest(Arc::clone(&state.pending_requests)),
    };
    match state.requests.try_submit(job) {
        Ok(()) => result
            .await
            .unwrap_or_else(|_| StatusCode::SERVICE_UNAVAILABLE.into_response()),
        Err(AdmissionError::Full(_)) => (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "Request queue is full. Please retry later.".into(),
                    error_type: "rate_limit_error".into(),
                },
            }),
        )
            .into_response(),
        Err(AdmissionError::Closed(_)) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
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
            "heads": state.config.n_heads,
            "embd": state.config.n_embd,
            "pending_requests": pending,
            "dispatched_batches": state.dispatched_batches.load(Ordering::SeqCst),
            "capacity": MAX_PENDING_REQUESTS,
            "capacity_scope": "queued_requests_excluding_active_batch",
        })),
    )
}

// Keep a trailing incomplete UTF-8 sequence buffered across token boundaries.
fn complete_utf8_prefix(bytes: &[u8]) -> usize {
    let mut offset = 0;
    // Each complete error consumes at least one input byte.
    for _ in bytes {
        match std::str::from_utf8(&bytes[offset..]) {
            Ok(_) => return bytes.len(),
            Err(error) => match error.error_len() {
                Some(length) => offset += error.valid_up_to() + length,
                None => return offset + error.valid_up_to(),
            },
        }
    }
    bytes.len()
}
fn generate_remote(
    reader: &bml_gguf::GgufReader,
    endpoints: &[SocketAddr],
    req: &CompletionRequest,
    mut emit: impl FnMut(&str) -> bool,
) -> bml_inference::Result<String> {
    use bml_inference::{
        model::Model,
        remote::{RemoteLimits, RemotePipeline},
        Error,
    };
    let model = Model::from_gguf(reader)?;
    let tokens = model.tokenizer().encode(&req.prompt, true)?;
    let context = tokens
        .len()
        .checked_add(req.max_tokens as usize)
        .ok_or_else(|| Error("request context overflow".into()))?;
    if context > model.config().context_length || tokens.is_empty() {
        return Err(Error("prompt/generation exceeds model context".into()));
    }
    if req.max_tokens == 0 {
        return Ok(String::new());
    }
    let mut pipeline =
        RemotePipeline::connect(&model, endpoints, context, RemoteLimits::default())?;
    let mut logits = pipeline.prefill(&tokens)?;
    let mut generated = Vec::new();
    let mut text = String::new();
    for step in 0..req.max_tokens {
        let values: Vec<_> = logits.iter().map(|&x| f64::from(x)).collect();
        let token = sample(&values, req.temperature, u64::from(step))
            .ok_or_else(|| Error("invalid remote logits".into()))?;
        if token == model.tokenizer().spec().eos_id {
            break;
        }
        generated.push(token);
        let bytes = model.tokenizer().decode_bytes(&generated, true)?;
        let next = String::from_utf8_lossy(&bytes[..complete_utf8_prefix(&bytes)]).into_owned();
        if !next.starts_with(&text) {
            return Err(Error("non-prefix token decoding".into()));
        }
        if next.len() > text.len() && !emit(&next[text.len()..]) {
            return Ok(text);
        }
        text = next;
        if step + 1 < req.max_tokens {
            logits = pipeline.decode(token)?;
        }
    }
    let final_text = model.tokenizer().decode(&generated, true)?;
    if !final_text.starts_with(&text) {
        return Err(Error("non-prefix final decoding".into()));
    }
    if final_text.len() > text.len() {
        let _ = emit(&final_text[text.len()..]);
    }
    pipeline.close()?;
    Ok(final_text)
}

fn generate_text_local(state: &ServerState, req: &CompletionRequest) -> Result<String, String> {
    let mut compiler = match &state.backend {
        ModelBackend::Local(c) => c.lock().map_err(|error| error.to_string())?,
        _ => return Err("backend does not support local causal inference".into()),
    };
    generate_local(&mut compiler, req, |_| true)
}

fn generate_streaming_local(
    state: &ServerState,
    req: &CompletionRequest,
    tx: mpsc::Sender<Result<String, std::convert::Infallible>>,
) -> Result<(), String> {
    let mut compiler = match &state.backend {
        ModelBackend::Local(c) => c.lock().map_err(|error| error.to_string())?,
        _ => return Err("backend does not support local causal inference".into()),
    };
    generate_local(&mut compiler, req, |piece| {
        let event = StreamEvent {
            choices: vec![StreamChoice {
                text: piece.into(),
                finish_reason: None,
            }],
        };
        tx.blocking_send(Ok(serde_json::to_string(&event).unwrap()))
            .is_ok()
    })?;
    let done = StreamEvent {
        choices: vec![StreamChoice {
            text: String::new(),
            finish_reason: Some("stop".into()),
        }],
    };
    let _ = tx.blocking_send(Ok(serde_json::to_string(&done).unwrap()));
    Ok(())
}

fn generate_local(
    model: &mut LocalModel,
    req: &CompletionRequest,
    mut emit: impl FnMut(&str) -> bool,
) -> Result<String, String> {
    let compiler = model.compiler();
    let ids = compiler.vocab().encode(&req.prompt)?;
    let eos_token_id = compiler.vocab().eos_token_id;
    let context_limit = model.context_limit();
    let config = compiler.config();
    if ids.is_empty() || ids.len() > context_limit as usize {
        return Err("prompt is empty or exceeds model context".into());
    }
    let maximum = (req.max_tokens as usize).min(context_limit as usize - ids.len());
    if maximum == 0 {
        return Ok(String::new());
    }
    let capacity = u32::try_from(ids.len() + maximum).map_err(|_| "context size overflow")?;
    let mut cache = bml_compiler::HashConsedKV::try_new(
        config.n_layers,
        compiler.n_kv_heads(),
        config.n_embd / config.n_heads,
        capacity,
        bml_compiler::kv_cache::DEFAULT_MAX_KV_BYTES,
    )?;
    let mut logits = model.forward_cached(&ids, &mut cache)?;
    let mut pending = Vec::new();
    let mut text = String::new();
    for step in 0..maximum {
        let id = sample(&logits, req.temperature, step as u64).ok_or("invalid sampling input")?;
        if id == eos_token_id {
            break;
        }
        pending.extend(model.compiler().vocab().token_piece_bytes(id)?);
        let piece = take_decoded_text(&mut pending, false);
        text.push_str(&piece);
        if !piece.is_empty() && !emit(&piece) {
            return Ok(text);
        }
        if step + 1 < maximum {
            logits = model.forward_cached(&[id], &mut cache)?;
        }
    }
    let tail = take_decoded_text(&mut pending, true);
    text.push_str(&tail);
    if !tail.is_empty() {
        let _ = emit(&tail);
    }
    Ok(text)
}

// Keep incomplete UTF-8 across generated pieces; malformed or unfinished bytes
// are replaced only when their status is known, preserving the SSE text boundary.
fn take_decoded_text(pending: &mut Vec<u8>, finish: bool) -> String {
    let end = if finish {
        pending.len()
    } else {
        complete_utf8_prefix(pending)
    };
    let text = String::from_utf8_lossy(&pending[..end]).into_owned();
    pending.drain(..end);
    text
}

#[cfg(test)]
#[path = "../../inference/tests/support/gguf_fixture.rs"]
mod gguf_fixture;

#[cfg(test)]
#[path = "../../cli/tests/support/token_fixture.rs"]
mod token_fixture;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_stream_and_completion_share_causal_decode_without_prompt_echo() {
        let fixture = token_fixture::fixture();
        let compiler = InferenceCompiler::open(&fixture.0).unwrap();
        let (mut state, _batches) = state(1);
        let inner = Arc::get_mut(&mut state).unwrap();
        inner.config = compiler.config().clone();
        inner.backend = ModelBackend::Local(Mutex::new(LocalModel::Native(Box::new(compiler))));
        let request = CompletionRequest {
            prompt: "x".into(),
            max_tokens: 6,
            temperature: 0.0,
            stream: false,
        };
        assert_eq!(generate_text_local(&state, &request).unwrap(), "é");
        let (tx, mut rx) = mpsc::channel(8);
        generate_streaming_local(&state, &request, tx).unwrap();
        let mut pieces = Vec::new();
        while let Ok(event) = rx.try_recv() {
            let json: serde_json::Value = serde_json::from_str(&event.unwrap()).unwrap();
            pieces.push(json["choices"][0]["text"].as_str().unwrap().to_owned());
        }
        assert_eq!(pieces, ["é", ""]);
    }

    fn state(capacity: usize) -> (Arc<ServerState>, RequestBatcher<QueuedRequest>) {
        let (requests, batches) = RequestBatcher::new(RequestBatchConfig {
            queue_capacity: capacity,
            ..Default::default()
        })
        .unwrap();
        (
            Arc::new(ServerState {
                backend: ModelBackend::Unavailable,
                config: ModelConfig {
                    architecture: "test".into(),
                    n_layers: 0,
                    n_heads: 0,
                    n_embd: 0,
                    context_length: 0,
                    vocab_size: 0,
                },
                pending_requests: Arc::new(AtomicUsize::new(0)),
                dispatched_batches: AtomicUsize::new(0),
                requests,
            }),
            batches,
        )
    }
    fn request() -> CompletionRequest {
        CompletionRequest {
            prompt: "test".into(),
            max_tokens: 0,
            temperature: 0.,
            stream: false,
        }
    }

    #[tokio::test]
    async fn full_queue_returns_429_and_preserves_admitted_request() {
        let (state, mut batches) = state(1);
        let (response, result) = oneshot::channel();
        state.pending_requests.fetch_add(1, Ordering::SeqCst);
        assert!(state
            .requests
            .try_submit(QueuedRequest {
                request: request(),
                response,
                _pending: PendingRequest(state.pending_requests.clone())
            })
            .is_ok());
        let rejected = completions(State(state.clone()), Json(request())).await;
        assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(state.pending_requests.load(Ordering::SeqCst), 1);
        let body = axum::body::to_bytes(rejected.into_body(), 4096)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["error"]["type"], "rate_limit_error");
        let batch = batches.next_batch().unwrap();
        assert_eq!(batch.requests.len(), 1);
        execute_request(&state, batch.requests.into_iter().next().unwrap(), 0);
        assert_eq!(result.await.unwrap().status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(state.pending_requests.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn closed_scheduler_returns_503_without_leaking_pending_count() {
        let (state, batches) = state(1);
        drop(batches);
        let response = completions(State(state.clone()), Json(request())).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(state.pending_requests.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn scheduled_requests_finish_and_release_admission() {
        let (state, batches) = state(16);
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let state = state.clone();
                tokio::spawn(
                    async move { completions(State(state), Json(request())).await.status() },
                )
            })
            .collect();
        // Admit all ten before the first collection window starts. This tests
        // batching deterministically without assuming an OS wakeup deadline.
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while state.pending_requests.load(Ordering::SeqCst) != 10 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        start_scheduler(&state, batches);
        for handle in handles {
            assert_eq!(handle.await.unwrap(), StatusCode::NOT_IMPLEMENTED);
        }
        // Response notification can precede the worker dropping the last guard.
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while state.pending_requests.load(Ordering::SeqCst) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(state.dispatched_batches.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn streaming_utf8_prefix_does_not_split_a_multibyte_token() {
        assert_eq!(complete_utf8_prefix(&[b'a', 0xe2, 0x82]), 1);
        assert_eq!(complete_utf8_prefix(&[b'a', 0xe2, 0x82, 0xac]), 4);
        assert_eq!(complete_utf8_prefix(&[0xff, 0xe2]), 1);
    }

    #[test]
    fn historical_streaming_keeps_partial_utf8_until_complete_or_finished() {
        let mut pending = Vec::new();
        let mut text = String::new();
        for byte in " Hello 🙂\n".bytes() {
            pending.push(byte);
            text.push_str(&take_decoded_text(&mut pending, false));
            assert!(!text.contains('\u{fffd}'));
            assert!(pending.len() < 4);
        }
        assert_eq!(text, " Hello 🙂\n");
        assert!(pending.is_empty());
        pending.extend([0xf0, 0x9f]);
        assert_eq!(take_decoded_text(&mut pending, false), "");
        assert_eq!(take_decoded_text(&mut pending, true), "\u{fffd}");
        assert!(pending.is_empty());
        pending.extend([0xff, b'A']);
        assert_eq!(take_decoded_text(&mut pending, false), "\u{fffd}A");
    }

    #[tokio::test]
    async fn remote_http_and_sse_match_local_generation() {
        use bml_inference::{
            model::Model,
            remote::{serve_layer_connection, RemoteLimits},
        };
        use std::net::TcpListener;
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "bml_http_{}_{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::write(&path, crate::gguf_fixture::fixture()).unwrap();
        let listeners = [
            TcpListener::bind("127.0.0.1:0").unwrap(),
            TcpListener::bind("127.0.0.1:0").unwrap(),
        ];
        let addresses = listeners
            .each_ref()
            .map(|listener| listener.local_addr().unwrap());
        let workers: Vec<_> = listeners
            .into_iter()
            .map(|listener| {
                let worker_path = path.clone();
                std::thread::spawn(move || {
                    let reader = bml_gguf::GgufReader::open(worker_path).unwrap();
                    let model = Model::from_gguf(&reader).unwrap();
                    // Each replica accepts exactly one batch/request. Sending both
                    // to replica zero makes the second request fail this test.
                    serve_layer_connection(
                        &model,
                        0..1,
                        listener.accept().unwrap().0,
                        RemoteLimits::default(),
                    )
                    .unwrap();
                })
            })
            .collect();
        let reader = bml_gguf::GgufReader::open(&path).unwrap();
        let model = Model::from_gguf(&reader).unwrap();
        let tokens = model.tokenizer().encode("a", true).unwrap();
        let mut session = model.session(tokens.len() + 3).unwrap();
        let mut logits = session.prefill(&tokens).unwrap();
        let mut generated = Vec::new();
        for step in 0..3 {
            let values: Vec<_> = logits.iter().map(|&v| f64::from(v)).collect();
            let token = sample(&values, 0., step).unwrap();
            if token == model.tokenizer().spec().eos_id {
                break;
            }
            generated.push(token);
            if step < 2 {
                logits = session.decode(token).unwrap();
            }
        }
        let expected = model.tokenizer().decode(&generated, true).unwrap();
        drop(model);
        let (mut state, old_batches) = state(16);
        let (requests, batches) = RequestBatcher::new(RequestBatchConfig {
            nodes: 2,
            max_batch_size: 1,
            ..Default::default()
        })
        .unwrap();
        let mutable = Arc::get_mut(&mut state).unwrap();
        mutable.requests = requests;
        mutable.backend = ModelBackend::Remote {
            reader,
            replicas: addresses.iter().map(|&a| vec![a]).collect(),
        };
        drop(old_batches);
        start_scheduler(&state, batches);
        let make_request = |stream| CompletionRequest {
            prompt: "a".into(),
            max_tokens: 3,
            temperature: 0.,
            stream,
        };
        let response = completions(State(state.clone()), Json(make_request(false))).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let response: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(response["choices"][0]["text"], expected);
        let response = completions(State(state.clone()), Json(make_request(true))).await;
        assert_eq!(response.headers()["content-type"], "text/event-stream");
        let body = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        let mut streamed = String::new();
        let mut done = 0;
        for line in body.lines().filter_map(|line| line.strip_prefix("data: ")) {
            if line == "[DONE]" {
                done += 1;
                continue;
            }
            let event: serde_json::Value = serde_json::from_str(line).unwrap();
            streamed.push_str(event["choices"][0]["text"].as_str().unwrap());
        }
        assert_eq!(streamed, expected);
        assert_eq!(done, 1);
        for worker in workers {
            worker.join().unwrap();
        }
        std::fs::remove_file(path).unwrap();
    }
}
