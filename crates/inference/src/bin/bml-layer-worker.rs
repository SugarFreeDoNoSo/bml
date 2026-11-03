//! Dedicated bounded TCP worker for validated causal layer sessions.
use bml_gguf::GgufReader;
use bml_inference::model::Model;
use bml_inference::remote::{serve_layer_connection, RemoteLimits};
use std::io::Write;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path: Option<PathBuf> = None;
    let mut layers = None;
    let mut address: SocketAddr = "127.0.0.1:9998".parse()?;
    let mut limits = RemoteLimits::default();
    let mut max_sessions = 4usize;
    let mut once = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("bml-layer-worker --model FILE.gguf --layers START:END [--listen IP:PORT] [--max-sessions N] [--context N] [--memory-bytes N] [--frame-bytes N] [--timeout-ms N] [--once]\nLayer end is exclusive. Defaults: loopback:9998, 4 sessions, 4096 context, 512 MiB per-session numerical memory, 1 MiB frames, 30s socket timeout. --once serves one connection then exits.");
            return Ok(());
        }
        if arg == "--once" {
            once = true;
            continue;
        }
        let value = args.next().ok_or("missing argument value")?;
        match arg.as_str() {
            "--model" => model_path = Some(value.into()),
            "--layers" => {
                let (start, end) = value.split_once(':').ok_or("layers must be START:END")?;
                layers = Some(start.parse::<usize>()?..end.parse::<usize>()?);
            }
            "--listen" => address = value.parse()?,
            "--max-sessions" => max_sessions = value.parse()?,
            "--context" => limits.context = value.parse()?,
            "--memory-bytes" => limits.memory_bytes = value.parse()?,
            "--frame-bytes" => limits.frame_bytes = value.parse()?,
            "--timeout-ms" => limits.timeout = std::time::Duration::from_millis(value.parse()?),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let path = model_path.ok_or("--model is required")?;
    let layers = layers.ok_or("--layers is required")?;
    if max_sessions == 0
        || limits.context == 0
        || limits.memory_bytes == 0
        || limits.frame_bytes == 0
        || limits.frame_bytes > u32::MAX as usize
        || limits.timeout.is_zero()
    {
        return Err("worker limits must be positive and frames fit u32".into());
    }
    let reader = GgufReader::open(path)?;
    let model = Model::from_gguf(&reader)?;
    if layers.start >= layers.end || layers.end > model.config().block_count {
        return Err("layer range outside model".into());
    }
    let fingerprint = model.fingerprint();
    let listener = TcpListener::bind(address)?;
    println!(
        "{}",
        serde_json::json!({"listening": listener.local_addr()?.to_string(),
        "layers": [layers.start, layers.end], "fingerprint": fingerprint,
        "max_sessions": max_sessions, "protocol": 2})
    );
    std::io::stdout().flush()?;
    std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        let mut handles: Vec<std::thread::ScopedJoinHandle<'_, ()>> = Vec::new();
        loop {
            let (stream, _) = listener.accept()?;
            let mut index = 0;
            while index < handles.len() {
                if handles[index].is_finished() {
                    let _ = handles.swap_remove(index).join();
                } else {
                    index += 1;
                }
            }
            if handles.len() < max_sessions {
                let model = &model;
                let layers = layers.clone();
                handles.push(scope.spawn(move || {
                    if let Err(error) = serve_layer_connection(model, layers, stream, limits) {
                        eprintln!("layer connection ended: {error}");
                    }
                }));
            } // Dropping a connection when saturated does not allocate session KV.
            if once {
                break;
            }
        }
        for handle in handles {
            let _ = handle.join();
        }
        Ok(())
    })
}
fn main() {
    if let Err(error) = run() {
        eprintln!("bml-layer-worker: {error}");
        std::process::exit(1);
    }
}
