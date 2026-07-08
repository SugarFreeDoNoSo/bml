//! # bml-cli
//!
//! CLI para compilar modelos GGUF a `.bmlgraph` y ejecutar inferencia BML.
//!
//! # Subcomandos
//!
//! ```sh
//! # Compilar GGUF → .bmlgraph
//! bml-cli compile -m modelo.gguf -o modelo.bmlgraph
//!
//! # Ejecutar desde .bmlgraph (pre-compilado) o GGUF (compila en caliente)
//! bml-cli run -m modelo.bmlgraph/ -p "Hello" -n 64
//! bml-cli run -m modelo.gguf -p "Hello" -n 64
//! ```

use bml_compiler::gguf_compiler::{
    compile_gguf_fast, load_from_dir, serialize_to_dir, InferenceCompiler,
};
use bml_compiler::distributed::{
    compile_distributed, load_distributed_header, load_distributed_fragment, serialize_distributed,
};
use bml_compiler::hardware::HardwareSpec;
use bml_compiler::sampler;
use bml_runtime::net::{MsgType, NodeHandle, Message, send_msg, recv_msg};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "bml-cli",
    version,
    about = "BML inference CLI — compila y ejecuta modelos GGUF"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compila un GGUF a formato .bmlgraph
    Compile {
        /// Ruta al archivo GGUF de entrada
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        /// Directorio de salida para .bmlgraph
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Tamaño máximo de fragmento L1 en bytes (default: 32768 = 32 KB)
        #[arg(long = "l1-threshold", default_value_t = 32768)]
        l1_threshold: usize,

        /// Tamaño máximo de fragmento L3 en bytes (default: 8388608 = 8 MB)
        #[arg(long = "l3-threshold", default_value_t = 8388608)]
        l3_threshold: usize,

        /// Compila en modo distribuido: un fragmento self-contained por capa
        #[arg(long = "distributed")]
        distributed: bool,

        /// Capas por fragmento (solo --distributed, default: 1 = una capa por fragmento)
        #[arg(long = "layers-per-fragment", default_value_t = 1)]
        layers_per_fragment: u32,
    },

    /// Ejecuta inferencia desde .bmlgraph o GGUF
    Run {
        /// Ruta al modelo: directorio .bmlgraph/ o archivo .gguf
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        /// Texto de entrada (prompt)
        #[arg(short = 'p', long = "prompt")]
        prompt: String,

        /// Número de tokens a generar
        #[arg(short = 'n', long = "num-tokens", default_value_t = 128)]
        num_tokens: u32,

        /// Número de threads
        #[arg(short = 't', long = "threads", default_value_t = 4)]
        threads: u32,

        /// Temperatura de sampling (0 = greedy)
        #[arg(long = "temp", default_value_t = 0.8)]
        temp: f64,

        /// Tamaño máximo de contexto
        #[arg(short = 'c', long = "context-size", default_value_t = 2048)]
        context_size: u32,
    },

    /// Distribuye fragmentos a nodos workers via TCP
    Distribute {
        /// Directorio .bmlgraph distribuido (con fragmentos self-contained)
        #[arg(short = 'm', long = "model")]
        model: PathBuf,

        /// Lista de nodos workers (host:port,host:port,...)
        #[arg(long = "nodes")]
        nodes: String,

        /// Prompt para generar
        #[arg(short = 'p', long = "prompt")]
        prompt: String,

        /// Número de tokens a generar
        #[arg(short = 'n', long = "num-tokens", default_value_t = 16)]
        num_tokens: u32,

        /// Temperatura de sampling
        #[arg(long = "temp", default_value_t = 0.8)]
        temp: f64,
    },
}

/// Determina si un path es un directorio .bmlgraph o un archivo GGUF.
fn is_bmlgraph_dir(path: &std::path::Path) -> bool {
    path.is_dir() && path.join("header.bmlgraph").exists()
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::Compile {
            model,
            output,
            l1_threshold,
            l3_threshold: _,
            distributed,
            layers_per_fragment,
        } => {
            if !model.exists() {
                eprintln!("Error: el archivo GGUF no existe: {}", model.display());
                std::process::exit(1);
            }

            println!("Compilando {} → {}", model.display(), output.display());

            if distributed {
                println!("Modo distribuido: {} capas por fragmento", layers_per_fragment);
                println!("Leyendo pesos del GGUF (dequantizando)...");

                let fragments = match compile_distributed(&model, layers_per_fragment, true) {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("Error compilando distribuido: {e}");
                        std::process::exit(1);
                    }
                };

                let config = &fragments[0].config;
                println!(
                    "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
                    config.architecture, config.n_layers, config.n_heads, config.n_embd, config.vocab_size,
                );
                println!("Fragmentos: {}", fragments.len());

                let total_weights: usize = fragments.iter().map(|f| f.weights.len()).sum();
                println!(
                    "Total pesos: {} ({:.1} MB)",
                    total_weights,
                    total_weights as f64 * 4.0 / (1024.0 * 1024.0),
                );

                for frag in &fragments {
                    let weight_mb = frag.weights.len() as f64 * 4.0 / (1024.0 * 1024.0);
                    println!(
                        "  fragmento {} (capas {}-{}): {} tensores, {:.1} MB",
                        frag.fragment_id, frag.layer_start, frag.layer_end,
                        frag.tensors.len(), weight_mb,
                    );
                }

                if let Err(e) = serialize_distributed(&fragments, &output) {
                    eprintln!("Error serializando: {e}");
                    std::process::exit(1);
                }
            } else {
                let hw = HardwareSpec::new(4, l1_threshold, l1_threshold * 8, l1_threshold);
                println!("Hardware target: {:?}", hw);

                let result = match compile_gguf_fast(&model, &hw) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Error compilando: {e}");
                        std::process::exit(1);
                    }
                };

                let config = &result.config;
                println!(
                    "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
                    config.architecture, config.n_layers, config.n_heads, config.n_embd, config.vocab_size,
                );
                println!("Fragmentos: {}", result.num_fragments);

                if let Err(e) = serialize_to_dir(&result, &output) {
                    eprintln!("Error serializando: {e}");
                    std::process::exit(1);
                }
            }

            println!("Serializado a: {}", output.display());
            if let Ok(entries) = std::fs::read_dir(&output) {
                for entry in entries.flatten() {
                    if let Ok(m) = entry.metadata() {
                        println!("  {} ({} bytes)", entry.file_name().to_string_lossy(), m.len());
                    }
                }
            }
        }

        Command::Run {
            model,
            prompt,
            num_tokens,
            threads: _,
            temp,
            context_size,
        } => {
            if !model.exists() {
                eprintln!("Error: el modelo no existe: {}", model.display());
                std::process::exit(1);
            }

            if is_bmlgraph_dir(&model) {
                run_from_bmlgraph(&model, &prompt, num_tokens, temp, context_size);
            } else {
                run_from_gguf(&model, &prompt, num_tokens, temp, context_size);
            }
        }

        Command::Distribute {
            model,
            nodes,
            prompt,
            num_tokens,
            temp,
        } => {
            distribute(&model, &nodes, &prompt, num_tokens, temp);
        }
    }
}

/// Ejecuta inferencia desde un directorio .bmlgraph pre-compilado.
fn run_from_bmlgraph(
    bmlgraph_dir: &std::path::Path,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
    context_size: u32,
) {
    println!("Cargando .bmlgraph desde {}...", bmlgraph_dir.display());

    let (graph, const_pool, config) = match load_from_dir(bmlgraph_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error cargando .bmlgraph: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd, config.vocab_size,
    );
    println!("Fragmentos: {}", graph.num_fragments());
    println!("Const pool: {} valores", const_pool.len());
    println!("Prompt: {}", prompt);
    println!("Generando {} tokens (temp={})...\n", num_tokens, temp);

    // El camino .bmlgraph ejecuta el grafo BML compilado.
    // Esto ejecuta el hot loop RPN sobre los fragmentos.
    let mut runtime = bml_runtime::Runtime::new(8192, 64);

    let ctx = bml_domain::EvalContext::new(&[], &const_pool);

    // Warmup
    runtime.execute_graph_with_ctx(&graph, &ctx);

    // Medir tiempo de ejecución del grafo
    let start = std::time::Instant::now();
    let result = runtime.execute_graph_with_ctx(&graph, &ctx);
    let elapsed = start.elapsed();

    println!("Ejecución del grafo BML: {:.3?}", elapsed);
    println!("Resultado (f64): {:.6}", result);
    println!(
        "Fragmentos ejecutados: {} ({} ops total)",
        graph.num_fragments(),
        graph.total_byte_size(),
    );

    // Nota: el camino .bmlgraph ejecuta el DAG compilado del transformer
    // completo. La inferencia autoregresiva token-a-token requiere el
    // InferenceCompiler que tiene matmul, attention, etc. implementados
    // en f64 directo. El .bmlgraph es el grafo BML puro.
    //
    // Para generar texto real, usar `bml-cli run -m modelo.gguf`.
    let _ = num_tokens;
    let _ = temp;
    let _ = context_size;

    println!("\nNota: La generación autoregresiva de texto requiere el modo GGUF");
    println!("      (que usa InferenceCompiler con matmul/attention/RoPE en f64).");
    println!("      El modo .bmlgraph ejecuta el DAG BML compilado (operador puro).");
}

/// Ejecuta inferencia desde un archivo GGUF (compila en caliente).
fn run_from_gguf(
    gguf_path: &std::path::Path,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
    context_size: u32,
) {
    println!("Cargando modelo desde {}...", gguf_path.display());
    println!("Esto puede tardar para modelos grandes (dequantizando pesos)...");

    let compiler = match InferenceCompiler::open(gguf_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error cargando modelo: {e}");
            std::process::exit(1);
        }
    };

    let config = compiler.config();
    let vocab = compiler.vocab();
    println!(
        "Modelo: {} ({} capas, {} heads, {} embd, {} vocab)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd, vocab.len(),
    );
    println!(
        "Weight pool: {} valores ({:.1} MB)",
        compiler.weight_pool().len(),
        compiler.weight_pool().len() as f64 * 8.0 / (1024.0 * 1024.0),
    );
    println!("Prompt: {}", prompt);
    println!("Generando {} tokens (temp={})...\n", num_tokens, temp);

    let prompt_ids = vocab.encode(prompt);
    println!("Tokenizado: {} tokens", prompt_ids.len());

    let mut sequence = prompt_ids.clone();
    let max_ctx = context_size.min(config.context_length) as usize;

    for step in 0..num_tokens {
        let ctx_start = if sequence.len() > max_ctx {
            sequence.len() - max_ctx
        } else {
            0
        };
        let context = &sequence[ctx_start..];

        let logits = compiler.forward(context);

        let next_id = sampler::sample(&logits, temp, step as u64)
            .unwrap_or(vocab.eos_token_id);

        let token_text = vocab
            .decode_single(next_id)
            .strip_prefix('▁')
            .unwrap_or("<unk>");

        if token_text == "<|eot_id|>" || token_text == "</s>" {
            println!();
            break;
        }
        if token_text == "<0x0A>" {
            println!();
        } else {
            print!("{}", token_text);
        }
        use std::io::Write;
        std::io::stdout().flush().ok();

        sequence.push(next_id);

        if next_id == vocab.eos_token_id {
            println!("\n[EOS]");
            break;
        }
    }

    println!("\n");
    println!(
        "Generados {} tokens desde prompt '{}'",
        sequence.len() - prompt_ids.len(),
        prompt,
    );
}

/// Distribuye fragmentos a nodos workers y coordina la inferencia.
///
/// # Flujo
///
/// 1. Carga el header del .bmlgraph distribuido (config + n_fragments)
/// 2. Conecta a cada nodo worker via TCP
/// 3. Envía cada fragmento a un nodo (ExecuteFragment)
/// 4. Para cada token a generar:
///    a. Tokeniza el prompt → token_ids
///    b. Embedding lookup → hidden state
///    c. Envía hidden al primer nodo (BatchRequest)
///    d. Cada nodo ejecuta sus capas y pasa hidden al siguiente
///    e. El último nodo produce logits
///    f. Sampling → next token
/// 5. Repite hasta EOS o num_tokens
fn distribute(
    bmlgraph_dir: &std::path::Path,
    nodes_str: &str,
    prompt: &str,
    num_tokens: u32,
    temp: f64,
) {
    if !bmlgraph_dir.is_dir() {
        eprintln!("Error: {} no es un directorio .bmlgraph", bmlgraph_dir.display());
        std::process::exit(1);
    }

    let (config, n_fragments) = match load_distributed_header(bmlgraph_dir) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error cargando header: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "Modelo: {} ({} capas, {} heads, {} embd)",
        config.architecture, config.n_layers, config.n_heads, config.n_embd,
    );
    println!("Fragmentos: {}", n_fragments);

    let nodes: Vec<&str> = nodes_str.split(',').map(|s| s.trim()).collect();
    println!("Nodos: {:?}", nodes);

    if nodes.len() != n_fragments {
        eprintln!(
            "Warning: {} nodos para {} fragmentos — algunos nodos recibiran multiples fragmentos",
            nodes.len(), n_fragments,
        );
    }

    // Conectar a cada nodo y enviar fragmentos
    let mut node_handles: Vec<NodeHandle> = Vec::new();

    for (i, node_addr) in nodes.iter().enumerate() {
        let frag_id = if i < n_fragments { i as u32 } else { (i % n_fragments) as u32 };
        let frag_path = bmlgraph_dir.join(format!("fragment_{}.bmlgraph", frag_id));

        if !frag_path.exists() {
            eprintln!("Error: fragmento {} no existe", frag_path.display());
            std::process::exit(1);
        }

        let frag_bytes = match std::fs::read(&frag_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Error leyendo {}: {e}", frag_path.display());
                std::process::exit(1);
            }
        };

        let weight_mb = frag_bytes.len() as f64 / (1024.0 * 1024.0);
        println!("Conectando a {node_addr} (fragmento {frag_id}, {weight_mb:.1} MB)...");

        let mut handle = match NodeHandle::connect(node_addr) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("Error conectando a {node_addr}: {e}");
                std::process::exit(1);
            }
        };

        // Health check
        match handle.health_check() {
            Ok(true) => {}
            Ok(false) => {
                eprintln!("Error: {node_addr} no está healthy");
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("Error health check {node_addr}: {e}");
                std::process::exit(1);
            }
        }

        // Enviar fragmento via TCP raw (ExecuteFragment)
        match send_msg(
            &mut handle.stream_mut(),
            &Message::new(MsgType::ExecuteFragment, frag_bytes),
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error enviando fragmento a {node_addr}: {e}");
                std::process::exit(1);
            }
        }

        // Esperar confirmación
        match recv_msg(handle.stream_mut()) {
            Ok(msg) => {
                if msg.msg_type == MsgType::ReportResult {
                    let result = f64::from_le_bytes(
                        msg.payload.get(0..8).unwrap_or(&[0; 8]).try_into().unwrap_or([0; 8]),
                    );
                    if result.is_finite() && result > 0.0 {
                        println!("  {node_addr}: fragmento cargado OK");
                    } else {
                        eprintln!("  {node_addr}: error cargando fragmento");
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error recibiendo confirmación de {node_addr}: {e}");
                std::process::exit(1);
            }
        }

        node_handles.push(handle);
    }

    println!("Todos los fragmentos distribuidos. {} nodos conectados.", node_handles.len());
    println!("Prompt: {}", prompt);
    println!("Generando {} tokens (temp={})...\n", num_tokens, temp);

    // Necesitamos el vocabulario para tokenizar/sampling.
    // Lo cargamos desde el GGUF original (el .bmlgraph distribuido no lo tiene).
    // En una implementación completa, el vocab iría en el header.
    // Por ahora, usamos el InferenceCompiler solo para el vocab.
    let gguf_candidates = [
        bmlgraph_dir.with_extension("gguf"),
        bmlgraph_dir.parent().unwrap_or(std::path::Path::new("."))
            .join(format!("{}.gguf", config.architecture)),
    ];
    let vocab = if let Some(gguf_path) = gguf_candidates.iter().find(|p| p.exists()) {
        match InferenceCompiler::open(gguf_path) {
            Ok(c) => Some(c.vocab().clone()),
            Err(_) => None,
        }
    } else {
        None
    };

    let vocab = match vocab {
        Some(v) => v,
        None => {
            eprintln!("Warning: no se pudo cargar vocabulario. Output sin decode.");
            eprintln!("  Coloca el GGUF junto al .bmlgraph/ para enable decode.");
            // Sin vocab, solo podemos ejecutar el forward pass sin generar texto.
            eprintln!("Ejecutando forward pass distribuido (sin decode)...");
            let n_embd = config.n_embd as usize;
            let mut hidden = vec![0.0_f64; n_embd];

            // Pasar hidden por cada nodo en secuencia
            for (i, handle) in node_handles.iter_mut().enumerate() {
                let mut payload = Vec::with_capacity(4 + n_embd * 8);
                payload.extend_from_slice(&(n_embd as u32).to_le_bytes());
                for v in &hidden {
                    payload.extend_from_slice(&v.to_le_bytes());
                }

                send_msg(&mut handle.stream_mut(), &Message::new(MsgType::BatchRequest, payload)).ok();
                let response = recv_msg(handle.stream_mut()).ok();

                if let Some(msg) = response {
                    if msg.msg_type == MsgType::BatchResult && msg.payload.len() >= 4 {
                        let n = u32::from_le_bytes(msg.payload[0..4].try_into().unwrap()) as usize;
                        hidden.clear();
                        for j in 0..n.min(msg.payload.len() / 8 - 1) {
                            let offset = 4 + j * 8;
                            if offset + 8 <= msg.payload.len() {
                                hidden.push(f64::from_le_bytes(msg.payload[offset..offset + 8].try_into().unwrap()));
                            }
                        }
                        println!("  Nodo {i}: hidden[0..3] = {:?}", &hidden[..hidden.len().min(3)]);
                    }
                }
            }

            println!("\nForward pass distribuido completo.");
            println!("Hidden state final: {} dimensiones", hidden.len());
            return;
        }
    };

    // Pipeline autoregresivo con vocabulario
    let prompt_ids = vocab.encode(prompt);
    println!("Tokenizado: {} tokens", prompt_ids.len());

    let mut sequence = prompt_ids.clone();
    let n_embd = config.n_embd as usize;

    for step in 0..num_tokens {
        // Embedding lookup (en el coordinador, liviano)
        let mut hidden = vec![0.0_f64; n_embd];
        for &tid in &sequence {
            // Placeholder: en implementación completa, usar embedding del último fragmento
            for (i, v) in hidden.iter_mut().enumerate() {
                *v += (tid as f64) * 0.001 / (i as f64 + 1.0);
            }
        }
        let scale = 1.0 / (sequence.len() as f64).sqrt();
        for v in &mut hidden {
            *v *= scale;
        }

        // Pasar hidden por cada nodo en secuencia
        for (_i, handle) in node_handles.iter_mut().enumerate() {
            let mut payload = Vec::with_capacity(4 + n_embd * 8);
            payload.extend_from_slice(&(n_embd as u32).to_le_bytes());
            for v in &hidden {
                payload.extend_from_slice(&v.to_le_bytes());
            }

            send_msg(&mut handle.stream_mut(), &Message::new(MsgType::BatchRequest, payload)).ok();
            let response = recv_msg(handle.stream_mut()).ok();

            if let Some(msg) = response {
                if msg.msg_type == MsgType::BatchResult && msg.payload.len() >= 4 {
                    let n = u32::from_le_bytes(msg.payload[0..4].try_into().unwrap()) as usize;
                    hidden.clear();
                    for j in 0..n.min((msg.payload.len() - 4) / 8) {
                        let offset = 4 + j * 8;
                        if offset + 8 <= msg.payload.len() {
                            hidden.push(f64::from_le_bytes(msg.payload[offset..offset + 8].try_into().unwrap()));
                        }
                    }
                }
            }
        }

        // El último hidden son los logits (simplificado)
        let logits = &hidden;
        let next_id = sampler::sample(logits, temp, step as u64)
            .unwrap_or(vocab.eos_token_id);

        let token_text = vocab.decode_single(next_id)
            .strip_prefix('▁')
            .unwrap_or("<unk>");

        if token_text == "<|eot_id|>" || token_text == "</s>" {
            println!();
            break;
        }
        if token_text == "<0x0A>" {
            println!();
        } else {
            print!("{}", token_text);
        }
        use std::io::Write;
        std::io::stdout().flush().ok();

        sequence.push(next_id);
        if next_id == vocab.eos_token_id {
            println!("\n[EOS]");
            break;
        }
    }

    println!("\n\nDistribuido: {} tokens generados desde {} nodos",
        sequence.len() - prompt_ids.len(),
        node_handles.len(),
    );
}
