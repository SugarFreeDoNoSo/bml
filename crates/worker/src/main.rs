//! # bml-worker
//!
//! Daemon que escucha TCP y ejecuta fragmentos `.bmlgraph` distribuidos.
//!
//! # Uso
//!
//! ```sh
//! # Iniciar worker en puerto 9999
//! bml-worker --port 9999
//!
//! # Con un fragmento pre-cargado
//! bml-worker --port 9999 --fragment /path/to/fragment_0.bmlgraph
//! ```
//!
//! # Protocolo
//!
//! Usa el protocolo TCP raw de `bml_runtime::net`:
//!
//! | msg_type | Acción |
//! |-----------|--------|
//! | ExecuteFragment | Carga fragmento, responde ReportResult (ack, no inferencia) |
//! | ExecuteGraph | Ejecuta .bmlgraph validado con inputs/buffer; devuelve resultado y buffer |
//! | HealthCheck | Responde alive |
//! | StealWork | Retira un fragmento pendiente y lo transfiere al requester |
//! | EnqueueFragment | Valida y admite trabajo pendiente en una cola acotada |
//! | BatchRequest | Recibe dimensión + hidden state, ejecuta las operaciones legadas, responde BatchResult |

use bml_compiler::distributed::{
    deserialize_vector_fragment, load_distributed_fragment, DistributedFragment,
};
use bml_runtime::net::{
    recv_msg_with_limit, send_msg, Message, MsgType, DEFAULT_MAX_PAYLOAD_BYTES,
};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

struct WorkerState {
    /// Fragmento cargado (si --fragment fue especificado).
    fragment: Option<DistributedFragment>,
    pending: Arc<bml_runtime::queue::PendingWork<Vec<u8>>>,
    max_payload_bytes: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut port: u16 = 9999;
    let mut fragment_path: Option<PathBuf> = None;
    let mut max_payload_bytes = DEFAULT_MAX_PAYLOAD_BYTES;
    let mut symbolic_path: Option<PathBuf> = None;
    let mut symbolic_source: Option<PathBuf> = None;
    let mut shard_path: Option<PathBuf> = None;
    let mut shard_hash: Option<String> = None;
    let mut shard_bytes: u64 = 512 * 1024 * 1024;
    let mut symbolic_start: Option<u32> = None;
    let mut symbolic_end: Option<u32> = None;
    let mut symbolic_limits = bml_runtime::symbolic_transport::SymbolicTransportLimits::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().unwrap_or(9999);
                }
            }
            "--fragment" | "-f" => {
                i += 1;
                if i < args.len() {
                    fragment_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--max-message-bytes" => {
                i += 1;
                max_payload_bytes = args
                    .get(i)
                    .and_then(|s| s.parse::<u32>().ok())
                    .filter(|size| *size > 0)
                    .unwrap_or_else(|| {
                        eprintln!("--max-message-bytes requires a positive u32 byte budget");
                        std::process::exit(2);
                    }) as usize;
            }
            "--symbolic-artifact" | "--symbolic-source" => {
                let option = args[i].clone();
                i += 1;
                let path = args.get(i).map(PathBuf::from).unwrap_or_else(|| {
                    eprintln!("{option} requires a path");
                    std::process::exit(2);
                });
                if option == "--symbolic-artifact" {
                    symbolic_path = Some(path);
                } else {
                    symbolic_source = Some(path);
                }
            }
            "--symbolic-shard" | "--symbolic-shard-sha256" | "--symbolic-shard-bytes" => {
                let option = args[i].clone();
                i += 1;
                let value = args.get(i).unwrap_or_else(|| {
                    eprintln!("{option} requires a value");
                    std::process::exit(2);
                });
                match option.as_str() {
                    "--symbolic-shard" => shard_path = Some(PathBuf::from(value)),
                    "--symbolic-shard-sha256" => shard_hash = Some(value.clone()),
                    _ => {
                        shard_bytes = value.parse().unwrap_or_else(|_| {
                            eprintln!("invalid shard byte budget");
                            std::process::exit(2);
                        })
                    }
                }
            }
            "--layer-start"
            | "--layer-end"
            | "--symbolic-context"
            | "--symbolic-kv-bytes"
            | "--symbolic-timeout-seconds" => {
                let option = args[i].clone();
                i += 1;
                let value = args
                    .get(i)
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or_else(|| {
                        eprintln!("{option} requires a nonnegative integer");
                        std::process::exit(2);
                    });
                match option.as_str() {
                    "--layer-start" => {
                        symbolic_start = Some(u32::try_from(value).unwrap_or_else(|_| {
                            eprintln!("layer start overflow");
                            std::process::exit(2);
                        }))
                    }
                    "--layer-end" => {
                        symbolic_end = Some(u32::try_from(value).unwrap_or_else(|_| {
                            eprintln!("layer end overflow");
                            std::process::exit(2);
                        }))
                    }
                    "--symbolic-context" => {
                        symbolic_limits.context = u32::try_from(value).unwrap_or_else(|_| {
                            eprintln!("context overflow");
                            std::process::exit(2);
                        })
                    }
                    "--symbolic-timeout-seconds" => {
                        if value == 0 {
                            eprintln!("timeout must be positive");
                            std::process::exit(2);
                        }
                        symbolic_limits.timeout = std::time::Duration::from_secs(value as u64);
                    }
                    _ => symbolic_limits.kv_bytes = value,
                }
            }
            "--symbolic-large-budget" => {
                symbolic_limits.model.max_cached_ops = 4_000_000;
                symbolic_limits.model.max_programs = 4096;
                symbolic_limits.model.graph = bml_runtime::graph::GraphLimits {
                    max_ops: 2_000_000,
                    max_steps: 4_000_000,
                    max_values: 4_000_000,
                };
            }
            "-h" | "--help" => {
                println!("bml-worker [--port 9999] [--fragment /path/to/fragment_N.bmlgraph]");
                println!("  --port N        Puerto TCP (default: 9999)");
                println!("  --fragment PATH Fragmento .bmlgraph pre-cargado");
                println!("  --max-message-bytes N Maximum incoming payload (default: 67108864)");
                println!("  --symbolic-artifact PATH --layer-start N --layer-end N: serve persisted symbolic layer sessions");
                println!("  --symbolic-source PATH --symbolic-context N --symbolic-kv-bytes N: source override and per-session budgets");
                println!("  --symbolic-shard PATH --symbolic-shard-sha256 HASH --symbolic-shard-bytes N: authenticated layer weights");
                println!("  --symbolic-large-budget: allow larger persisted symbolic programs");
                println!("  --symbolic-timeout-seconds N: per-connection read/write timeout");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    if symbolic_path.is_none() && (shard_path.is_some() || shard_hash.is_some()) {
        eprintln!("symbolic shard requires --symbolic-artifact");
        std::process::exit(2);
    }
    if let Some(path) = symbolic_path {
        if fragment_path.is_some() {
            eprintln!("--fragment and --symbolic-artifact are mutually exclusive");
            std::process::exit(2);
        }
        let (Some(start), Some(end)) = (symbolic_start, symbolic_end) else {
            eprintln!("symbolic worker requires --layer-start and --layer-end");
            std::process::exit(2);
        };
        symbolic_limits.frame_bytes = max_payload_bytes;
        let server = bml_runtime::symbolic_transport::SymbolicLayerServer::new(
            &path,
            symbolic_source.as_deref(),
            start..end,
            symbolic_limits,
        )
        .and_then(|service| match (&shard_path, &shard_hash) {
            (Some(path), Some(hash)) => service.with_weight_shard(path, hash, shard_bytes),
            (None, None) => Ok(service),
            _ => Err("shard path and trusted hash must be supplied together".into()),
        });
        let service = match server {
            Ok(service) => Arc::new(service),
            Err(error) => {
                eprintln!("Error loading symbolic artifact: {error}");
                std::process::exit(1);
            }
        };
        let listener = TcpListener::bind(("0.0.0.0", port)).unwrap_or_else(|error| {
            eprintln!("Error binding symbolic worker: {error}");
            std::process::exit(1);
        });
        println!(
            "symbolic worker listening on {} for layers {}..{} identity {}",
            listener.local_addr().unwrap(),
            start,
            end,
            service.identity()
        );
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let service = Arc::clone(&service);
                    std::thread::spawn(move || {
                        if let Err(error) = service.serve_connection(stream) {
                            eprintln!("symbolic connection: {error}");
                        }
                    });
                }
                Err(error) => eprintln!("symbolic accept: {error}"),
            }
        }
        return;
    }

    // Cargar fragmento si se especificó
    let state = Arc::new(Mutex::new(WorkerState {
        max_payload_bytes,
        pending: Arc::new(bml_runtime::queue::PendingWork::new(64).unwrap()),
        fragment: if let Some(ref path) = fragment_path {
            match load_legacy_fragment(path) {
                Ok(f) => {
                    let weight_mb = f.weights.len() as f64 * 4.0 / (1024.0 * 1024.0);
                    println!(
                        "Fragmento cargado: id={} capas {}-{} {} tensores {:.1} MB",
                        f.fragment_id,
                        f.layer_start,
                        f.layer_end,
                        f.tensors.len(),
                        weight_mb,
                    );
                    Some(f)
                }
                Err(e) => {
                    eprintln!("Error cargando fragmento: {e}");
                    std::process::exit(1);
                }
            }
        } else {
            None
        },
    }));

    let listener = match TcpListener::bind(("0.0.0.0", port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error bindeando puerto {port}: {e}");
            std::process::exit(1);
        }
    };

    println!("bml-worker escuchando en 0.0.0.0:{port}");
    if fragment_path.is_some() {
        println!("  Fragmento pre-cargado");
    } else {
        println!("  Sin fragmento pre-cargado (esperando ExecuteFragment)");
    }

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error aceptando conexión: {e}");
                continue;
            }
        };

        let state = Arc::clone(&state);
        std::thread::spawn(move || {
            handle_connection(stream, state);
        });
    }
}

fn handle_connection(mut stream: TcpStream, state: Arc<Mutex<WorkerState>>) {
    let peer = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_default();
    let (pending, max_payload_bytes) = {
        let state = state.lock().unwrap();
        (state.pending.clone(), state.max_payload_bytes)
    };
    loop {
        let msg = match recv_msg_with_limit(&mut stream, max_payload_bytes) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[{peer}] Error recv: {e}");
                return;
            }
        };

        match msg.msg_type {
            MsgType::ExecuteGraph => {
                let payload = bml_runtime::graph_rpc::handle_request(&msg.payload);
                if send_msg(&mut stream, &Message::new(MsgType::GraphResult, payload)).is_err() {
                    return;
                }
            }
            MsgType::HealthCheck => {
                let mut payload = vec![1];
                payload.extend_from_slice(&(pending.len() as u64).to_le_bytes());
                let response = Message::new(MsgType::HealthCheck, payload);
                let _ = send_msg(&mut stream, &response);
            }

            MsgType::ExecuteFragment => {
                // Deserializar fragmento desde el payload
                let frag = match deserialize_fragment_from_bytes(&msg.payload) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("[{peer}] Error deserializando fragmento: {e}");
                        let _ = send_msg(
                            &mut stream,
                            &Message::new(MsgType::ReportResult, f64::NAN.to_le_bytes().to_vec()),
                        );
                        return;
                    }
                };

                let weight_mb = frag.weights.len() as f64 * 4.0 / (1024.0 * 1024.0);
                println!(
                    "[{peer}] ExecuteFragment: id={} capas {}-{} {} tensores {:.1} MB",
                    frag.fragment_id,
                    frag.layer_start,
                    frag.layer_end,
                    frag.tensors.len(),
                    weight_mb,
                );

                // Cargar el fragmento en el estado del worker
                {
                    let mut s = state.lock().unwrap();
                    s.fragment = Some(frag);
                }

                // Responder OK
                let response = Message::new(MsgType::ReportResult, 1.0_f64.to_le_bytes().to_vec());
                let _ = send_msg(&mut stream, &response);
            }

            MsgType::BatchRequest => {
                // Recibe: hidden state (n_embd × f64) + ejecuta sus capas
                // Formato payload: [n_embd 4B][hidden: n_embd × 8B f64]
                let s = state.lock().unwrap();
                if let Some(ref frag) = s.fragment {
                    let n_embd = frag.config.n_embd as usize;

                    let expected_len = n_embd.checked_mul(8).and_then(|n| n.checked_add(4));
                    let declared_dimension = msg
                        .payload
                        .get(..4)
                        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()));
                    if expected_len != Some(msg.payload.len())
                        || declared_dimension != Some(frag.config.n_embd)
                    {
                        let _ = send_msg(&mut stream, &Message::new(MsgType::BatchResult, vec![]));
                        return;
                    }

                    // Leer hidden state
                    let mut hidden = vec![0.0_f64; n_embd];
                    for i in 0..n_embd {
                        let offset = 4 + i * 8;
                        hidden[i] =
                            f64::from_le_bytes(msg.payload[offset..offset + 8].try_into().unwrap());
                    }

                    // Ejecutar las capas de este fragmento
                    let result_hidden = match execute_layers(frag, &hidden) {
                        Ok(result) => result,
                        Err(error) => {
                            eprintln!("[{peer}] Invalid legacy shard: {error}");
                            let _ =
                                send_msg(&mut stream, &Message::new(MsgType::BatchResult, vec![]));
                            return;
                        }
                    };

                    // Responder con el nuevo hidden state
                    let mut response_payload = Vec::with_capacity(4 + result_hidden.len() * 8);
                    response_payload.extend_from_slice(&(result_hidden.len() as u32).to_le_bytes());
                    for v in &result_hidden {
                        response_payload.extend_from_slice(&v.to_le_bytes());
                    }

                    let _ = send_msg(
                        &mut stream,
                        &Message::new(MsgType::BatchResult, response_payload),
                    );
                } else {
                    eprintln!("[{peer}] BatchRequest sin fragmento cargado");
                    let _ = send_msg(&mut stream, &Message::new(MsgType::BatchResult, vec![]));
                }
            }

            MsgType::EnqueueFragment => {
                let status = match deserialize_fragment_from_bytes(&msg.payload) {
                    Ok(fragment) => {
                        if pending.push(serialize_fragment_to_bytes(&fragment)).is_ok() {
                            1
                        } else {
                            0
                        }
                    }
                    Err(_) => 2,
                };
                if send_msg(
                    &mut stream,
                    &Message::new(MsgType::EnqueueResult, vec![status]),
                )
                .is_err()
                {
                    return;
                }
            }
            MsgType::StealWork => {
                let payload = pending.pop().unwrap_or_default();
                if send_msg(&mut stream, &Message::new(MsgType::StealWork, payload)).is_err() {
                    return;
                }
            }
            MsgType::VectorMatmul => {
                // Deserializar VectorFragment desde el payload
                let vf = match deserialize_vector_fragment(&msg.payload) {
                    Ok(vf) => vf,
                    Err(e) => {
                        eprintln!("[{peer}] Error deserializando VectorFragment: {e}");
                        let _ = send_msg(&mut stream, &Message::new(MsgType::VectorResult, vec![]));
                        return;
                    }
                };

                let weight_mb = vf.weights_size_bytes() as f64 / (1024.0 * 1024.0);
                println!(
                    "[{peer}] VectorMatmul: frag={} n_in={} n_cols={} {:.1} MB",
                    vf.fragment_id, vf.n_in, vf.n_cols, weight_mb,
                );

                // Ejecutar el matmul del fragmento
                let y = vf.execute();

                // Serializar y responder
                let mut payload = Vec::with_capacity(4 + y.len() * 8);
                payload.extend_from_slice(&(y.len() as u32).to_le_bytes());
                for &v in &y {
                    payload.extend_from_slice(&v.to_le_bytes());
                }
                let _ = send_msg(&mut stream, &Message::new(MsgType::VectorResult, payload));
            }

            _ => {
                eprintln!("[{peer}] Mensaje no soportado: {:?}", msg.msg_type);
            }
        }
    }
}

/// Validate the shapes consumed by the legacy layer executor. Inspect actual
/// tensor metadata, not every declared layer, so admission work is bounded by
/// the received metadata even when a shard declares a very large layer range.
fn validate_legacy_fragment(fragment: &DistributedFragment) -> Result<(), String> {
    let n_embd = fragment.config.n_embd;
    if n_embd == 0 {
        return Err("legacy embedding dimension must be positive".into());
    }
    if fragment.layer_start > fragment.layer_end || fragment.layer_end > fragment.config.n_layers {
        return Err("invalid layer range".into());
    }
    for tensor in &fragment.tensors {
        let end = u64::from(tensor.n_rows)
            .checked_mul(u64::from(tensor.n_cols))
            .and_then(|size| tensor.offset.checked_add(size))
            .ok_or("tensor range overflow")?;
        if end > fragment.weights.len() as u64 {
            return Err("tensor exceeds weight storage".into());
        }
        let Some((layer, operation)) = legacy_tensor_layer(&tensor.name) else {
            continue;
        };
        if !(fragment.layer_start..fragment.layer_end).contains(&layer) {
            continue;
        }
        let expected = match operation {
            "attn_norm.weight" => (n_embd, 1),
            "attn_q.weight" => (n_embd, n_embd),
            _ => continue,
        };
        if (tensor.n_rows, tensor.n_cols) != expected {
            return Err(format!("invalid legacy tensor shape: {}", tensor.name));
        }
    }
    Ok(())
}

fn legacy_tensor_layer(name: &str) -> Option<(u32, &str)> {
    let (text, operation) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer = text.parse::<u32>().ok()?;
    // Preserve the exact canonical names matched by the historical executor.
    (text == layer.to_string()).then_some((layer, operation))
}

fn load_legacy_fragment(path: &std::path::Path) -> Result<DistributedFragment, String> {
    let fragment = load_distributed_fragment(path)?;
    validate_legacy_fragment(&fragment)?;
    Ok(fragment)
}

/// Execute the historical RMSNorm/Q-only path after validating all dimensions.
/// This is not a complete causal transformer implementation.
fn execute_layers(frag: &DistributedFragment, hidden: &[f64]) -> Result<Vec<f64>, String> {
    validate_legacy_fragment(frag)?;
    let n_embd = frag.config.n_embd as usize;
    if hidden.len() != n_embd {
        return Err("legacy hidden-state dimension mismatch".into());
    }
    // Missing operations are no-ops in this legacy path. Build only the layers
    // represented by executable metadata, rather than walking a potentially
    // enormous declared range. Keep ascending layers and the first tensor with
    // each canonical name, exactly as the former per-layer searches did.
    let mut layers = std::collections::BTreeMap::new();
    for tensor in &frag.tensors {
        let Some((layer, operation)) = legacy_tensor_layer(&tensor.name) else {
            continue;
        };
        if !(frag.layer_start..frag.layer_end).contains(&layer) {
            continue;
        }
        if operation == "attn_norm.weight" || operation == "attn_q.weight" {
            let entry = layers.entry(layer).or_insert((None, None));
            let slot = if operation == "attn_norm.weight" {
                &mut entry.0
            } else {
                &mut entry.1
            };
            slot.get_or_insert(tensor);
        }
    }
    let mut h = hidden.to_vec();
    for (norm, projection) in layers.into_values() {
        if let Some(meta) = norm {
            let offset = meta.offset as usize;
            let mean_sq: f64 = h.iter().map(|v| v * v).sum::<f64>() / n_embd as f64;
            let rms = (mean_sq + 1e-5).sqrt();
            for i in 0..n_embd {
                h[i] = h[i] / rms * f64::from(frag.weights[offset + i]);
            }
        }
        if let Some(meta) = projection {
            let offset = meta.offset as usize;
            let mut q = vec![0.0_f64; n_embd];
            for j in 0..n_embd {
                let mut dot = 0.0;
                for i in 0..n_embd {
                    let idx = offset + i * n_embd + j;
                    dot += h[i] * f64::from(frag.weights[idx]);
                }
                q[j] = dot;
            }
            h = q;
        }
    }
    Ok(h)
}

/// Serializa un DistributedFragment a bytes (para envío via TCP).
fn serialize_fragment_to_bytes(frag: &DistributedFragment) -> Vec<u8> {
    let mut bytes = Vec::new();

    // fragment_id, layer_start, layer_end
    bytes.extend_from_slice(&frag.fragment_id.to_le_bytes());
    bytes.extend_from_slice(&frag.layer_start.to_le_bytes());
    bytes.extend_from_slice(&frag.layer_end.to_le_bytes());

    // Config
    let arch_bytes = frag.config.architecture.as_bytes();
    bytes.extend_from_slice(&(arch_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(arch_bytes);
    bytes.extend_from_slice(&frag.config.n_layers.to_le_bytes());
    bytes.extend_from_slice(&frag.config.n_heads.to_le_bytes());
    bytes.extend_from_slice(&frag.config.n_embd.to_le_bytes());
    bytes.extend_from_slice(&frag.config.context_length.to_le_bytes());
    bytes.extend_from_slice(&frag.config.vocab_size.to_le_bytes());

    // n_kv_heads, head_dim
    bytes.extend_from_slice(&frag.n_kv_heads.to_le_bytes());
    bytes.extend_from_slice(&frag.head_dim.to_le_bytes());

    // Weights
    bytes.extend_from_slice(&(frag.weights.len() as u64).to_le_bytes());
    for &w in &frag.weights {
        bytes.extend_from_slice(&w.to_le_bytes());
    }

    // Tensors
    bytes.extend_from_slice(&(frag.tensors.len() as u32).to_le_bytes());
    for tensor in &frag.tensors {
        let name_bytes = tensor.name.as_bytes();
        bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&tensor.offset.to_le_bytes());
        bytes.extend_from_slice(&tensor.n_rows.to_le_bytes());
        bytes.extend_from_slice(&tensor.n_cols.to_le_bytes());
    }

    bytes
}

/// Deserializa un DistributedFragment desde bytes (recibido via TCP).
fn deserialize_fragment_from_bytes(bytes: &[u8]) -> Result<DistributedFragment, String> {
    fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
        let (value, rest) = bytes.split_at_checked(n).ok_or("truncated fragment")?;
        *bytes = rest;
        Ok(value)
    }
    fn u32_value(bytes: &mut &[u8]) -> Result<u32, String> {
        Ok(u32::from_le_bytes(take(bytes, 4)?.try_into().unwrap()))
    }
    fn u64_value(bytes: &mut &[u8]) -> Result<u64, String> {
        Ok(u64::from_le_bytes(take(bytes, 8)?.try_into().unwrap()))
    }
    fn text(bytes: &mut &[u8]) -> Result<String, String> {
        let n = u32_value(bytes)? as usize;
        String::from_utf8(take(bytes, n)?.to_vec()).map_err(|_| "invalid UTF-8".into())
    }
    let mut bytes = bytes;
    let fragment_id = u32_value(&mut bytes)?;
    let layer_start = u32_value(&mut bytes)?;
    let layer_end = u32_value(&mut bytes)?;
    let architecture = text(&mut bytes)?;
    let config = bml_compiler::gguf_compiler::ModelConfig {
        architecture,
        n_layers: u32_value(&mut bytes)?,
        n_heads: u32_value(&mut bytes)?,
        n_embd: u32_value(&mut bytes)?,
        context_length: u32_value(&mut bytes)?,
        vocab_size: u32_value(&mut bytes)?,
    };
    if layer_start > layer_end || layer_end > config.n_layers {
        return Err("invalid layer range".into());
    }
    let n_kv_heads = u32_value(&mut bytes)?;
    let head_dim = u32_value(&mut bytes)?;
    let n_weights = usize::try_from(u64_value(&mut bytes)?).map_err(|_| "weight count overflow")?;
    let weight_bytes = n_weights.checked_mul(4).ok_or("weight size overflow")?;
    let raw_weights = take(&mut bytes, weight_bytes)?;
    let weights: Vec<_> = raw_weights
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let n_tensors = u32_value(&mut bytes)? as usize;
    // Even an empty tensor name needs 20 bytes of metadata. Check before
    // reserving so a tiny malicious payload cannot request a huge allocation.
    if n_tensors > bytes.len() / 20 {
        return Err("truncated tensor metadata".into());
    }
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = text(&mut bytes)?;
        let offset = u64_value(&mut bytes)?;
        let n_rows = u32_value(&mut bytes)?;
        let n_cols = u32_value(&mut bytes)?;
        let end = u64::from(n_rows)
            .checked_mul(u64::from(n_cols))
            .and_then(|size| offset.checked_add(size))
            .ok_or("tensor range overflow")?;
        if end > n_weights as u64 {
            return Err("tensor exceeds weight storage".into());
        }
        tensors.push(bml_compiler::distributed::TensorMeta {
            name,
            offset,
            n_rows,
            n_cols,
        });
    }
    if !bytes.is_empty() {
        return Err("trailing fragment bytes".into());
    }
    let fragment = DistributedFragment {
        fragment_id,
        layer_start,
        layer_end,
        ops: vec![],
        weights,
        tensors,
        config,
        n_kv_heads,
        head_dim,
    };
    validate_legacy_fragment(&fragment)?;
    Ok(fragment)
}

#[cfg(test)]
mod connection_tests {
    use super::*;
    use bml_compiler::distributed::{serialize_vector_fragment, VectorFragment};
    use bml_runtime::net::{recv_msg, NodeHandle};

    #[test]
    fn graph_rpc_two_worker_stages_match_complete_local_graph() {
        use bml_compiler::{BmlGraph, Fragment, RpnOp};
        use bml_runtime::{
            buffer::ResultBuffer,
            graph::{GraphLimits, ValidatedGraph},
        };
        let mut nodes = Vec::new();
        let mut servers = Vec::new();
        for _ in 0..2 {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            servers.push(std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                handle_connection(
                    stream,
                    Arc::new(Mutex::new(WorkerState {
                        fragment: None,
                        pending: Arc::new(bml_runtime::queue::PendingWork::new(1).unwrap()),
                        max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
                    })),
                );
            }));
            let mut node = NodeHandle::connect(&address.to_string()).unwrap();
            node.stream_mut()
                .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                .unwrap();
            nodes.push(node);
        }
        let first = BmlGraph {
            fragments: vec![
                Fragment {
                    ops: vec![
                        RpnOp::Var(0),
                        RpnOp::Const(0),
                        RpnOp::FMul,
                        RpnOp::Zero,
                        RpnOp::StoreResult { slot: 0 },
                    ],
                },
                Fragment {
                    ops: vec![RpnOp::Zero, RpnOp::VarIndexed { base: 0 }],
                },
            ],
            const_pool: vec![3.0],
            threshold: 32768,
        };
        let second = BmlGraph {
            fragments: vec![Fragment {
                ops: vec![
                    RpnOp::Zero,
                    RpnOp::VarIndexed { base: 0 },
                    RpnOp::Const(0),
                    RpnOp::FMul,
                    RpnOp::One,
                    RpnOp::FAdd,
                ],
            }],
            const_pool: vec![3.0],
            threshold: 32768,
        };
        let mut complete = first.clone();
        complete.fragments.push(Fragment {
            ops: vec![RpnOp::Drop],
        });
        complete.fragments.extend(second.fragments.clone());
        let local = ValidatedGraph::new(complete, 1, 1, 1, GraphLimits::default()).unwrap();
        let published = bml_runtime::append_only::AppendOnlyResults::new(12).unwrap();
        let mut previous = Vec::new();
        for input in [2.0, -2.0, 0.5] {
            let initial = ResultBuffer::new(1, 1);
            let a = nodes[0]
                .execute_graph_append_only(&first, &[input], &initial, &published)
                .unwrap();
            assert_eq!(published.get(a.value_index), Some(input * 3.0));
            let transferred = ResultBuffer::from_vec(
                1,
                1,
                a.buffer_indices
                    .map(|i| published.get(i).unwrap())
                    .collect(),
            )
            .unwrap();
            let b = nodes[1]
                .execute_graph_append_only(&second, &[input], &transferred, &published)
                .unwrap();
            let value = published.get(b.value_index).unwrap();
            assert_eq!(value, input * 9.0 + 1.0);
            let mut local_buffer = ResultBuffer::new(1, 1);
            assert_eq!(value, local.execute(&[input], &mut local_buffer).unwrap());
            assert_eq!(
                b.buffer_indices
                    .map(|i| published.get(i).unwrap())
                    .collect::<Vec<_>>(),
                local_buffer.data()
            );
            for (index, bits) in &previous {
                assert_eq!(published.get(*index).unwrap().to_bits(), *bits);
            }
            previous = (0..published.reserved_count())
                .map(|i| (i, published.get(i).unwrap().to_bits()))
                .collect();
            assert!(nodes[0].health_check().unwrap());
            assert!(nodes[1].health_check().unwrap());
        }
        let error = nodes[0]
            .execute_graph_append_only(&first, &[2.0], &ResultBuffer::new(1, 1), &published)
            .unwrap_err();
        let bml_runtime::graph_rpc::GraphPublishError::StorageFull(retained) = error else {
            panic!("expected completed but unpublished result");
        };
        assert_eq!(retained.value, 6.0);
        assert_eq!(retained.buffer, [6.0]);
        assert_eq!(published.reserved_count(), 12);
        let retry = bml_runtime::append_only::AppendOnlyResults::new(2).unwrap();
        let position = retained.publish(&retry).unwrap();
        assert_eq!(retry.get(position.value_index), Some(6.0));
        let invalid = BmlGraph {
            fragments: vec![Fragment {
                ops: vec![RpnOp::Dup],
            }],
            const_pool: vec![],
            threshold: 32,
        };
        assert!(nodes[0]
            .execute_graph(&invalid, &[], &ResultBuffer::new(0, 0))
            .is_err());
        assert!(nodes[0].health_check().unwrap());
        drop(nodes);
        for server in servers {
            server.join().unwrap();
        }
    }

    #[test]
    fn configured_frame_limit_closes_oversized_session_before_reading_body() {
        use std::io::{Read, Write};
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(
                stream,
                Arc::new(Mutex::new(WorkerState {
                    fragment: None,
                    pending: Arc::new(bml_runtime::queue::PendingWork::new(1).unwrap()),
                    max_payload_bytes: 3,
                })),
            );
        });
        let mut client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(std::time::Duration::from_secs(2)))
            .unwrap();
        let mut header = (MsgType::HealthCheck as u32).to_le_bytes().to_vec();
        header.extend_from_slice(&4u32.to_le_bytes());
        client.write_all(&header).unwrap();
        assert_eq!(client.read(&mut [0u8; 1]).unwrap(), 0);
        server.join().unwrap();
    }

    #[test]
    fn legacy_batch_tcp_validates_dimensions_and_exact_payload_length() {
        let mut fragment = fragment_fixture();
        fragment.layer_end = 0;
        fragment.config.n_embd = 3;
        fragment.weights.clear();
        fragment.tensors.clear();
        let mut valid = 3_u32.to_le_bytes().to_vec();
        for value in [1.25_f64, -2.5, 7.0] {
            valid.extend_from_slice(&value.to_le_bytes());
        }
        let mut wrong_dimension = valid.clone();
        wrong_dimension[..4].copy_from_slice(&2_u32.to_le_bytes());
        let mut trailing = valid.clone();
        trailing.push(0);
        let mut cases = vec![
            (valid.clone(), true),
            (wrong_dimension, false),
            (trailing, false),
        ];
        for end in 0..valid.len() {
            cases.push((valid[..end].to_vec(), false));
        }
        for (payload, accepted) in cases {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            let server = std::thread::spawn(move || {
                let (stream, _) = listener.accept().unwrap();
                handle_connection(
                    stream,
                    Arc::new(Mutex::new(WorkerState {
                        fragment: None,
                        pending: Arc::new(bml_runtime::queue::PendingWork::new(1).unwrap()),
                        max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
                    })),
                );
            });
            let mut client = TcpStream::connect(address).unwrap();
            client
                .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                .unwrap();
            client
                .set_write_timeout(Some(std::time::Duration::from_secs(2)))
                .unwrap();
            send_msg(
                &mut client,
                &Message::new(
                    MsgType::ExecuteFragment,
                    serialize_fragment_to_bytes(&fragment),
                ),
            )
            .unwrap();
            let ack = recv_msg(&mut client).unwrap();
            assert_eq!(ack.msg_type, MsgType::ReportResult);
            assert_eq!(ack.payload, 1.0_f64.to_le_bytes());
            send_msg(
                &mut client,
                &Message::new(MsgType::BatchRequest, payload.clone()),
            )
            .unwrap();
            let response = recv_msg(&mut client).unwrap();
            assert_eq!(response.msg_type, MsgType::BatchResult);
            if accepted {
                // No layers: the independently encoded vector is preserved byte for byte.
                assert_eq!(response.payload, valid);
                send_msg(&mut client, &Message::new(MsgType::HealthCheck, vec![])).unwrap();
                let health = recv_msg(&mut client).unwrap();
                assert_eq!(health.msg_type, MsgType::HealthCheck);
                assert_eq!(health.payload[0], 1);
            } else {
                assert!(response.payload.is_empty(), "payload={payload:?}");
                use std::io::Read;
                assert_eq!(
                    client.read(&mut [0_u8; 1]).unwrap(),
                    0,
                    "invalid session must close"
                );
            }
            drop(client);
            server.join().unwrap();
        }
    }

    #[test]
    fn persistent_session_runs_multiple_commands_and_real_vector_work() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle_connection(
                stream,
                Arc::new(Mutex::new(WorkerState {
                    max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
                    fragment: None,
                    pending: Arc::new(bml_runtime::queue::PendingWork::new(64).unwrap()),
                })),
            );
        });
        let mut node = NodeHandle::connect(&address.to_string()).unwrap();
        node.stream_mut()
            .set_read_timeout(Some(std::time::Duration::from_secs(3)))
            .unwrap();
        assert!(node.health_check().unwrap());
        assert!(node.steal_work().unwrap().is_none());
        let fragment = VectorFragment::new(0, &[2., 3.], &[1., 2., 3., 4.], 0, 2, 2, 0, 2, 0);
        for _ in 0..2 {
            send_msg(
                node.stream_mut(),
                &Message::new(MsgType::VectorMatmul, serialize_vector_fragment(&fragment)),
            )
            .unwrap();
            let result = recv_msg(node.stream_mut()).unwrap();
            assert_eq!(result.msg_type, MsgType::VectorResult);
            assert_eq!(result.payload.len(), 20);
            assert_eq!(
                u32::from_le_bytes(result.payload[..4].try_into().unwrap()),
                2
            );
            let values: Vec<_> = result.payload[4..]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b))
                .collect();
            assert_eq!(values, [11., 16.]); // Independent two-column dot products.
            assert!(node.health_check().unwrap());
        }
        drop(node);
        server.join().unwrap();
    }
    fn malformed_legacy_shapes() -> Vec<DistributedFragment> {
        use bml_compiler::distributed::TensorMeta;
        let mut oversized = fragment_fixture();
        oversized.weights.clear();
        oversized.tensors = vec![TensorMeta {
            name: "blk.0.attn_q.weight".into(),
            offset: 0,
            n_rows: 0,
            n_cols: u32::MAX,
        }];
        let mut shrinking = fragment_fixture();
        shrinking.config.n_layers = 2;
        shrinking.layer_end = 2;
        shrinking.config.n_embd = 2;
        shrinking.weights = vec![1.0, 2.0, 1.0, 1.0];
        shrinking.tensors = vec![
            TensorMeta {
                name: "blk.0.attn_q.weight".into(),
                offset: 0,
                n_rows: 2,
                n_cols: 1,
            },
            TensorMeta {
                name: "blk.1.attn_norm.weight".into(),
                offset: 2,
                n_rows: 2,
                n_cols: 1,
            },
        ];
        let mut empty = fragment_fixture();
        empty.config.n_embd = 0;
        empty.tensors.clear();
        empty.weights.clear();
        vec![oversized, shrinking, empty]
    }

    #[test]
    fn legacy_shard_preflight_rejects_unsafe_shapes_before_execution() {
        for fragment in malformed_legacy_shapes() {
            assert!(validate_legacy_fragment(&fragment).is_err());
            assert!(
                deserialize_fragment_from_bytes(&serialize_fragment_to_bytes(&fragment)).is_err()
            );
        }
        let mut bad = malformed_legacy_shapes();
        // This small malformed model previously panicked in its second layer.
        assert!(execute_layers(&bad.remove(1), &[1.0, 2.0]).is_err());
        let mut valid = fragment_fixture();
        valid.config.n_embd = 2;
        valid.weights = vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        valid.tensors = vec![
            bml_compiler::distributed::TensorMeta {
                name: "blk.0.attn_norm.weight".into(),
                offset: 0,
                n_rows: 2,
                n_cols: 1,
            },
            bml_compiler::distributed::TensorMeta {
                name: "blk.0.attn_q.weight".into(),
                offset: 2,
                n_rows: 2,
                n_cols: 2,
            },
        ];
        let result = execute_layers(&valid, &[1.0, 2.0]).unwrap();
        let rms = (2.5_f64 + 1e-5).sqrt();
        for (actual, expected) in result.iter().zip([7.0 / rms, 10.0 / rms]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!(execute_layers(&valid, &[1.0]).is_err());
        // Admission traverses metadata only, not this enormous but legal range.
        valid.tensors.clear();
        valid.weights.clear();
        valid.config.n_layers = u32::MAX;
        valid.layer_end = u32::MAX;
        validate_legacy_fragment(&valid).unwrap();
    }

    #[test]
    fn invalid_legacy_shard_upload_preserves_loaded_model_and_other_connections() {
        use std::io::Read;
        let invalid = malformed_legacy_shapes();
        let connections = invalid.len() + 1;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let shared = Arc::new(Mutex::new(WorkerState {
            fragment: None,
            pending: Arc::new(bml_runtime::queue::PendingWork::new(1).unwrap()),
            max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
        }));
        let server = std::thread::spawn(move || {
            let mut handlers = Vec::new();
            for _ in 0..connections {
                let (stream, _) = listener.accept().unwrap();
                let state = shared.clone();
                handlers.push(std::thread::spawn(move || handle_connection(stream, state)));
            }
            for handler in handlers {
                handler.join().unwrap();
            }
        });
        let connect = || {
            let stream = TcpStream::connect(address).unwrap();
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(std::time::Duration::from_secs(2)))
                .unwrap();
            stream
        };
        let mut healthy = connect();
        let mut valid = fragment_fixture();
        valid.layer_end = 0;
        send_msg(
            &mut healthy,
            &Message::new(
                MsgType::ExecuteFragment,
                serialize_fragment_to_bytes(&valid),
            ),
        )
        .unwrap();
        assert_eq!(
            recv_msg(&mut healthy).unwrap().payload,
            1.0_f64.to_le_bytes()
        );
        let mut batch = 1_u32.to_le_bytes().to_vec();
        batch.extend_from_slice(&2.5_f64.to_le_bytes());
        for fragment in invalid {
            let mut rejected = connect();
            send_msg(
                &mut rejected,
                &Message::new(
                    MsgType::ExecuteFragment,
                    serialize_fragment_to_bytes(&fragment),
                ),
            )
            .unwrap();
            let response = recv_msg(&mut rejected).unwrap();
            assert_eq!(response.msg_type, MsgType::ReportResult);
            assert!(f64::from_le_bytes(response.payload.try_into().unwrap()).is_nan());
            assert_eq!(rejected.read(&mut [0; 1]).unwrap(), 0);
            send_msg(&mut healthy, &Message::new(MsgType::HealthCheck, vec![])).unwrap();
            let health = recv_msg(&mut healthy).unwrap();
            assert_eq!(health.msg_type, MsgType::HealthCheck);
            assert_eq!(health.payload[0], 1);
            send_msg(
                &mut healthy,
                &Message::new(MsgType::BatchRequest, batch.clone()),
            )
            .unwrap();
            let output = recv_msg(&mut healthy).unwrap();
            assert_eq!(output.msg_type, MsgType::BatchResult);
            assert_eq!(output.payload, batch);
        }
        drop(healthy);
        server.join().unwrap();
    }

    #[test]
    fn legacy_sparse_layers_preserve_order_without_visiting_empty_layer_ranges() {
        use bml_compiler::distributed::TensorMeta;
        let mut fragment = fragment_fixture();
        fragment.config.n_layers = u32::MAX;
        fragment.layer_end = u32::MAX;
        fragment.weights = vec![2.0, 3.0, 99.0];
        let last_projection = format!("blk.{}.attn_q.weight", u32::MAX - 1);
        // Deliberately reversed metadata order, with a duplicate: execute norm
        // before projection by layer number and use the first projection only.
        fragment.tensors = vec![
            TensorMeta {
                name: last_projection.clone(),
                offset: 1,
                n_rows: 1,
                n_cols: 1,
            },
            TensorMeta {
                name: "blk.0.attn_norm.weight".into(),
                offset: 0,
                n_rows: 1,
                n_cols: 1,
            },
            TensorMeta {
                name: last_projection,
                offset: 2,
                n_rows: 1,
                n_cols: 1,
            },
        ];
        let expected = 3.0_f64 / (9.0_f64 + 1e-5).sqrt() * 2.0 * 3.0;
        assert_eq!(execute_layers(&fragment, &[3.0]).unwrap(), [expected]);
        fragment.tensors.clear();
        assert_eq!(execute_layers(&fragment, &[3.0]).unwrap(), [3.0]);
    }

    #[test]
    fn legacy_file_loading_restores_config_and_applies_shape_preflight() {
        use bml_compiler::distributed::serialize_distributed;
        let directory =
            std::env::temp_dir().join(format!("bml-worker-shapes-{}", std::process::id()));
        let valid = fragment_fixture();
        serialize_distributed(std::slice::from_ref(&valid), &directory).unwrap();
        let path = directory.join("fragment_9.bmlgraph");
        let loaded = load_legacy_fragment(&path).unwrap();
        assert_eq!(loaded.config, valid.config);
        assert_eq!(execute_layers(&loaded, &[2.0]).unwrap(), [2.0]);
        for invalid in malformed_legacy_shapes() {
            serialize_distributed(&[invalid], &directory).unwrap();
            assert!(load_legacy_fragment(&path).is_err());
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn fragment_fixture() -> DistributedFragment {
        DistributedFragment {
            fragment_id: 9,
            layer_start: 0,
            layer_end: 1,
            ops: vec![],
            weights: vec![1.25],
            tensors: vec![bml_compiler::distributed::TensorMeta {
                name: "w".into(),
                offset: 0,
                n_rows: 1,
                n_cols: 1,
            }],
            config: bml_compiler::gguf_compiler::ModelConfig {
                architecture: "llama".into(),
                n_layers: 1,
                n_heads: 1,
                n_embd: 1,
                context_length: 8,
                vocab_size: 2,
            },
            n_kv_heads: 1,
            head_dim: 1,
        }
    }

    #[test]
    fn fragment_decoder_rejects_every_truncation_and_invalid_lengths() {
        let bytes = serialize_fragment_to_bytes(&fragment_fixture());
        let decoded = deserialize_fragment_from_bytes(&bytes).unwrap();
        assert_eq!(decoded.fragment_id, 9);
        assert_eq!(decoded.weights, [1.25]);
        assert_eq!(decoded.tensors[0].name, "w");
        for cut in 0..bytes.len() {
            assert!(
                deserialize_fragment_from_bytes(&bytes[..cut]).is_err(),
                "cut={cut}"
            );
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(deserialize_fragment_from_bytes(&trailing).is_err());
        let mut name = bytes.clone();
        name[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(deserialize_fragment_from_bytes(&name).is_err());
        let mut utf8 = bytes.clone();
        utf8[16] = 255;
        assert!(deserialize_fragment_from_bytes(&utf8).is_err());
        // Three IDs + length-prefixed five-byte architecture + seven u32s.
        let weights_at = 12 + 4 + 5 + 7 * 4;
        let mut count = bytes.clone();
        count[weights_at..weights_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(deserialize_fragment_from_bytes(&count).is_err());
        let mut tensor_count = bytes;
        let tensors_at = weights_at + 8 + 4;
        tensor_count[tensors_at..tensors_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(deserialize_fragment_from_bytes(&tensor_count).is_err());
    }

    #[test]
    fn fragment_decoder_checks_layer_and_tensor_storage_ranges() {
        let mut fragment = fragment_fixture();
        fragment.layer_end = 2;
        assert!(deserialize_fragment_from_bytes(&serialize_fragment_to_bytes(&fragment)).is_err());
        fragment.layer_end = 1;
        fragment.tensors[0].offset = 1;
        assert!(deserialize_fragment_from_bytes(&serialize_fragment_to_bytes(&fragment)).is_err());
        fragment.tensors[0].offset = u64::MAX;
        assert!(deserialize_fragment_from_bytes(&serialize_fragment_to_bytes(&fragment)).is_err());
    }

    #[test]
    fn fragment_decoder_accepts_empty_layer_ranges_and_minimal_tensor_metadata() {
        let mut fragment = fragment_fixture();
        // A zero-layer partition is valid at either end of the model.
        for boundary in [0, fragment.config.n_layers] {
            fragment.layer_start = boundary;
            fragment.layer_end = boundary;
            // Empty names still have exactly 20 bytes of tensor metadata.
            fragment.tensors[0].name.clear();
            let bytes = serialize_fragment_to_bytes(&fragment);
            let decoded = deserialize_fragment_from_bytes(&bytes).unwrap();
            assert_eq!(
                (decoded.layer_start, decoded.layer_end),
                (boundary, boundary)
            );
            assert_eq!(decoded.tensors.len(), 1);
            assert_eq!(decoded.tensors[0].name, "");
            assert_eq!(decoded.weights, [1.25]);
        }
        fragment.layer_start = 1;
        fragment.layer_end = 0;
        assert_eq!(
            deserialize_fragment_from_bytes(&serialize_fragment_to_bytes(&fragment))
                .err()
                .as_deref(),
            Some("invalid layer range")
        );
    }

    #[test]
    fn fragment_decoder_rejects_impossible_tensor_count_before_parsing_entries() {
        let mut fragment = fragment_fixture();
        fragment.tensors[0].name.clear();
        let mut bytes = serialize_fragment_to_bytes(&fragment);
        // The last entry occupies the minimum 20 bytes. Advertising two entries
        // cannot fit, even before inspecting their contents or reserving slots.
        let count_at = bytes.len() - 20 - 4;
        bytes[count_at..count_at + 4].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(
            deserialize_fragment_from_bytes(&bytes).err().as_deref(),
            Some("truncated tensor metadata")
        );
        // Even an invalid first name must not mask this envelope-level error.
        bytes[count_at + 4..count_at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            deserialize_fragment_from_bytes(&bytes).err().as_deref(),
            Some("truncated tensor metadata")
        );
    }
    #[test]
    fn concurrent_remote_steals_transfer_pending_work_once() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let state = Arc::new(Mutex::new(WorkerState {
            max_payload_bytes: DEFAULT_MAX_PAYLOAD_BYTES,
            fragment: Some(fragment_fixture()),
            pending: Arc::new(bml_runtime::queue::PendingWork::new(8).unwrap()),
        }));
        let server_state = state.clone();
        let server = std::thread::spawn(move || {
            let mut connections = Vec::new();
            for _ in 0..5 {
                let (stream, _) = listener.accept().unwrap();
                let state = server_state.clone();
                connections.push(std::thread::spawn(move || handle_connection(stream, state)));
            }
            for connection in connections {
                connection.join().unwrap();
            }
        });
        let mut producer = NodeHandle::connect(&address.to_string()).unwrap();
        assert!(
            producer.steal_work().unwrap().is_none(),
            "loaded model is not pending work"
        );
        assert_eq!(producer.health_status().unwrap().pending, Some(0));
        assert!(producer.enqueue_fragment(&[0]).is_err());
        for id in 0..8 {
            let mut fragment = fragment_fixture();
            fragment.fragment_id = id;
            assert!(producer
                .enqueue_fragment(&serialize_fragment_to_bytes(&fragment))
                .unwrap());
        }
        assert!(!producer
            .enqueue_fragment(&serialize_fragment_to_bytes(&fragment_fixture()))
            .unwrap());
        assert_eq!(producer.health_status().unwrap().pending, Some(8));
        drop(producer);
        let barrier = Arc::new(std::sync::Barrier::new(4));
        let thieves: Vec<_> = (0..4)
            .map(|_| {
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    let mut node = NodeHandle::connect(&address.to_string()).unwrap();
                    node.stream_mut()
                        .set_read_timeout(Some(std::time::Duration::from_secs(3)))
                        .unwrap();
                    barrier.wait();
                    let mut ids = Vec::new();
                    while let Some(bytes) = node.steal_work().unwrap() {
                        ids.push(deserialize_fragment_from_bytes(&bytes).unwrap().fragment_id);
                    }
                    ids
                })
            })
            .collect();
        let mut ids: Vec<_> = thieves
            .into_iter()
            .flat_map(|t| t.join().unwrap())
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..8).collect::<Vec<_>>());
        server.join().unwrap();
        let state = state.lock().unwrap();
        assert!(state.pending.is_empty());
        assert_eq!(state.fragment.as_ref().unwrap().fragment_id, 9);
    }
}
