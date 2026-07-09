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
//! | ExecuteFragment | Deserializa fragmento, ejecuta forward, responde ReportResult |
//! | HealthCheck | Responde alive |
//! | StealWork | Si tiene trabajo pendiente, envía un fragmento al requester |
//! | BatchRequest | Recibe hidden state + token IDs, ejecuta sus capas, responde BatchResult |

use bml_compiler::distributed::{
    deserialize_vector_fragment, load_distributed_fragment, DistributedFragment,
};
use bml_runtime::net::{recv_msg, send_msg, MsgType, Message};
use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

struct WorkerState {
    /// Fragmento cargado (si --fragment fue especificado).
    fragment: Option<DistributedFragment>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut port: u16 = 9999;
    let mut fragment_path: Option<PathBuf> = None;

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
            "-h" | "--help" => {
                println!("bml-worker [--port 9999] [--fragment /path/to/fragment_N.bmlgraph]");
                println!("  --port N        Puerto TCP (default: 9999)");
                println!("  --fragment PATH Fragmento .bmlgraph pre-cargado");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    // Cargar fragmento si se especificó
    let state = Arc::new(Mutex::new(WorkerState {
        fragment: if let Some(ref path) = fragment_path {
            match load_distributed_fragment(path) {
                Ok(f) => {
                    let weight_mb = f.weights.len() as f64 * 4.0 / (1024.0 * 1024.0);
                    println!(
                        "Fragmento cargado: id={} capas {}-{} {} tensores {:.1} MB",
                        f.fragment_id, f.layer_start, f.layer_end, f.tensors.len(), weight_mb,
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
    let peer = stream.peer_addr().map(|a| a.to_string()).unwrap_or_default();
    let msg = match recv_msg(&mut stream) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[{peer}] Error recv: {e}");
            return;
        }
    };

    match msg.msg_type {
        MsgType::HealthCheck => {
            let response = Message::new(MsgType::HealthCheck, vec![1]);
            let _ = send_msg(&mut stream, &response);
        }

        MsgType::ExecuteFragment => {
            // Deserializar fragmento desde el payload
            let frag = match deserialize_fragment_from_bytes(&msg.payload) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("[{peer}] Error deserializando fragmento: {e}");
                    let _ = send_msg(&mut stream, &Message::new(
                        MsgType::ReportResult,
                        f64::NAN.to_le_bytes().to_vec(),
                    ));
                    return;
                }
            };

            let weight_mb = frag.weights.len() as f64 * 4.0 / (1024.0 * 1024.0);
            println!(
                "[{peer}] ExecuteFragment: id={} capas {}-{} {} tensores {:.1} MB",
                frag.fragment_id, frag.layer_start, frag.layer_end, frag.tensors.len(), weight_mb,
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

                if msg.payload.len() < 4 + n_embd * 8 {
                    let _ = send_msg(&mut stream, &Message::new(
                        MsgType::BatchResult,
                        vec![],
                    ));
                    return;
                }

                // Leer hidden state
                let mut hidden = vec![0.0_f64; n_embd];
                for i in 0..n_embd {
                    let offset = 4 + i * 8;
                    hidden[i] = f64::from_le_bytes(
                        msg.payload[offset..offset + 8].try_into().unwrap(),
                    );
                }

                // Ejecutar las capas de este fragmento
                let result_hidden = execute_layers(frag, &hidden);

                // Responder con el nuevo hidden state
                let mut response_payload = Vec::with_capacity(4 + result_hidden.len() * 8);
                response_payload.extend_from_slice(&(result_hidden.len() as u32).to_le_bytes());
                for v in &result_hidden {
                    response_payload.extend_from_slice(&v.to_le_bytes());
                }

                let _ = send_msg(&mut stream, &Message::new(
                    MsgType::BatchResult,
                    response_payload,
                ));
            } else {
                eprintln!("[{peer}] BatchRequest sin fragmento cargado");
                let _ = send_msg(&mut stream, &Message::new(
                    MsgType::BatchResult,
                    vec![],
                ));
            }
        }

        MsgType::StealWork => {
            // Si tenemos un fragmento, lo enviamos al requester
            let s = state.lock().unwrap();
            if let Some(ref frag) = s.fragment {
                let frag_bytes = serialize_fragment_to_bytes(frag);
                let _ = send_msg(&mut stream, &Message::new(
                    MsgType::StealWork,
                    frag_bytes,
                ));
            } else {
                let _ = send_msg(&mut stream, &Message::new(
                    MsgType::StealWork,
                    vec![],
                ));
            }
        }

        MsgType::VectorMatmul => {
            // Deserializar VectorFragment desde el payload
            let vf = match deserialize_vector_fragment(&msg.payload) {
                Ok(vf) => vf,
                Err(e) => {
                    eprintln!("[{peer}] Error deserializando VectorFragment: {e}");
                    let _ = send_msg(&mut stream, &Message::new(
                        MsgType::VectorResult,
                        vec![],
                    ));
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

/// Ejecuta las capas del fragmento sobre el hidden state.
///
/// Por ahora, esto es un placeholder: aplica RMSNorm + matmul simple
/// usando los pesos del fragmento. La implementación completa reusa
/// la lógica de InferenceCompiler::forward_layer().
fn execute_layers(frag: &DistributedFragment, hidden: &[f64]) -> Vec<f64> {
    let n_embd = frag.config.n_embd as usize;
    let mut h = hidden.to_vec();

    for layer in frag.layer_start..frag.layer_end {
        if layer >= frag.config.n_layers {
            continue;
        }
        let prefix = format!("blk.{layer}");

        // RMSNorm
        let norm_name = format!("{prefix}.attn_norm.weight");
        if let Some(meta) = frag.tensors.iter().find(|t| t.name == norm_name) {
            let offset = meta.offset as usize;
            let mean_sq: f64 = h.iter().map(|v| v * v).sum::<f64>() / n_embd as f64;
            let rms = (mean_sq + 1e-5).sqrt();
            for i in 0..n_embd.min(meta.n_rows as usize) {
                let w = frag.weights.get(offset + i).copied().unwrap_or(1.0) as f64;
                h[i] = h[i] / rms * w;
            }
        }

        // Q matmul (simplificado)
        let q_name = format!("{prefix}.attn_q.weight");
        if let Some(meta) = frag.tensors.iter().find(|t| t.name == q_name) {
            let offset = meta.offset as usize;
            let n_in = meta.n_rows as usize;
            let n_out = meta.n_cols as usize;
            let mut q = vec![0.0_f64; n_out];
            for j in 0..n_out {
                let mut dot = 0.0;
                for i in 0..n_in.min(h.len()) {
                    let idx = offset + i * n_out + j;
                    if idx < frag.weights.len() {
                        dot += h[i] * frag.weights[idx] as f64;
                    }
                }
                q[j] = dot;
            }
            h = q;
        }

        // Residual (simplificado)
        for i in 0..h.len().min(n_embd) {
            h[i] = h.get(i).copied().unwrap_or(0.0);
        }
    }

    h
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
    if bytes.len() < 16 {
        return Err("payload demasiado pequeño".into());
    }

    let mut offset = 0;

    let fragment_id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let layer_start = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let layer_end = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;

    // Config
    let arch_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let architecture = String::from_utf8_lossy(&bytes[offset..offset + arch_len]).to_string();
    offset += arch_len;
    let n_layers = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let n_heads = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let n_embd = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let context_length = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let vocab_size = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let n_kv_heads = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;
    let head_dim = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    offset += 4;

    let config = bml_compiler::gguf_compiler::ModelConfig {
        architecture,
        n_layers,
        n_heads,
        n_embd,
        context_length,
        vocab_size,
    };

    // Weights
    let n_weights = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;
    let mut weights = Vec::with_capacity(n_weights);
    for _ in 0..n_weights {
        let w = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        weights.push(w);
    }

    // Tensors
    let n_tensors = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let name = String::from_utf8_lossy(&bytes[offset..offset + name_len]).to_string();
        offset += name_len;
        let tensor_offset = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        offset += 8;
        let n_rows = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let n_cols = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        tensors.push(bml_compiler::distributed::TensorMeta {
            name,
            offset: tensor_offset,
            n_rows,
            n_cols,
        });
    }

    Ok(DistributedFragment {
        fragment_id,
        layer_start,
        layer_end,
        ops: vec![],
        weights,
        tensors,
        config,
        n_kv_heads,
        head_dim,
    })
}
