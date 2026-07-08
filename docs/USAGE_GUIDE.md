# Guía de uso completa de BML

## Arquitectura del flujo

```
GGUF (pesos cuantizados)
  │
  ├─ bml-cli compile -m modelo.gguf -o modelo.bmlgraph/
  │    │
  │    └─ InferenceCompiler::open(gguf)
  │         ├─ mmap zero-copy de pesos
  │         ├─ dequantiza Q4_0/Q8_0 → f32
  │         ├─ construye DAG BML por capa
  │         ├─ hash consing + constant folding
  │         ├─ lineariza a RPN
  │         └─ fragmenta según L1/L3 threshold
  │
  ├─ bml-cli run -m modelo.bmlgraph/ -p "Hello" -n 64
  │    │
  │    └─ load_from_dir(modelo.bmlgraph/)
  │         ├─ lee header.bmlgraph (config + const_pool)
  │         ├─ lee fragment_N.bmlgraph (ops RPN)
  │         └─ Runtime::execute_graph()
  │              ├─ tokenizer.encode(prompt)
  │              ├─ forward(token_ids) → logits
  │              ├─ sampler.sample(logits, temp)
  │              ├─ tokenizer.decode(token_id) → texto
  │              └─ loop autoregresivo
  │
  └─ bml-server -m modelo.bmlgraph/ --port 8080
       │
       └─ load_from_dir() + axum server
            ├─ POST /v1/completions  → JSON o SSE streaming
            └─ GET  /health          → status JSON
```

---

## Instalación

```sh
# Compilar todo el workspace en release
cargo build --release

# Los binarios quedan en:
#   target/release/bml-cli
#   target/release/bml-server
#   target/release/bml-bench
```

---

## 1. Compilar un modelo GGUF a .bmlgraph

### Sintaxis

```sh
bml-cli compile -m <gguf_path> -o <output_dir>
```

### Ejemplo

```sh
# Compilar TinyLlama-1.1B Q4_0
bml-cli compile -m /path/to/tinyllama-1.1b-q4_0.gguf -o /path/to/tinyllama.bmlgraph

# Compilar con hardware target personalizado
bml-cli compile -m modelo.gguf -o modelo.bmlgraph --l1-threshold 32768 --l3-threshold 8388608
```

### Salida

El comando genera un directorio `.bmlgraph/`:

```
modelo.bmlgraph/
├── header.bmlgraph          # magic, version, config, const_pool
├── fragment_0.bmlgraph      # RPN ops del fragmento 0
├── fragment_1.bmlgraph      # RPN ops del fragmento 1
└── ...
```

### Flags

| Flag | Descripción | Default |
|------|-------------|---------|
| `-m, --model` | Ruta al archivo GGUF de entrada | (requerido) |
| `-o, --output` | Directorio de salida `.bmlgraph` | `<modelo>.bmlgraph` |
| `--l1-threshold` | Tamaño máximo de fragmento para L1 (bytes) | 32768 (32 KB) |
| `--l3-threshold` | Tamaño máximo de fragmento para L3 (bytes) | 8388608 (8 MB) |

---

## 2. Ejecutar inferencia con bml-cli

### Sintaxis

```sh
bml-cli run -m <bmlgraph_dir> -p "<prompt>" -n <num_tokens> [opciones]
```

### Ejemplo

```sh
# Generar 64 tokens a partir de un prompt
bml-cli run -m /path/to/tinyllama.bmlgraph -p "The capital of France is" -n 64

# Con temperatura personalizada
bml-cli run -m /path/to/tinyllama.bmlgraph -p "Hello world" -n 128 --temp 0.5

# Con 2 threads
bml-cli run -m /path/to/tinyllama.bmlgraph -p "Hello" -n 32 -t 2
```

### Flags

| Flag | Descripción | Default |
|------|-------------|---------|
| `-m, --model` | Ruta al directorio `.bmlgraph` compilado | (requerido) |
| `-p, --prompt` | Texto de entrada | (requerido) |
| `-n, --num-tokens` | Número de tokens a generar | 128 |
| `-t, --threads` | Número de threads | 4 |
| `--temp` | Temperatura de sampling | 0.8 |
| `--context-size` | Tamaño máximo de contexto | 2048 |

---

## 3. Servidor HTTP (bml-server)

### Sintaxis

```sh
bml-server -m <bmlgraph_dir> --port <port>
```

### Ejemplo

```sh
# Iniciar servidor en puerto 8080
bml-server -m /path/to/tinyml.bmlgraph --port 8080

# Health check
curl http://localhost:8080/health

# Generar texto (non-streaming)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 64}'

# Generar texto (streaming SSE)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 64, "stream": true}'
```

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/v1/completions` | Genera texto a partir de un prompt |
| `GET` | `/health` | Health check con info del modelo |

### Request body

```json
{
  "prompt": "The capital of France is",
  "max_tokens": 128,
  "temperature": 0.8,
  "stream": false
}
```

### Response (non-streaming)

```json
{
  "choices": [
    {
      "text": " Paris, which is...",
      "finish_reason": "stop"
    }
  ]
}
```

### Response (streaming, SSE)

```
data: {"choices":[{"text":" Paris"}]}

data: {"choices":[{"text":","}]}

data: {"choices":[{"text":" which"}]}

data: [DONE]
```

---

## 4. Benchmark (bml-bench)

### Sintaxis

```sh
bml-bench [--json] [--md] [--multicore] [--reps N] [--threads N]
```

### Ejemplo

```sh
# Benchmark básico (markdown)
bml-bench

# Benchmark multicore (1/2/4 threads)
bml-bench --multicore

# JSON para integración
bml-bench --json --multicore --reps 10
```

### Salida (markdown)

```
## Hot loop (raw)
| Ops/seg | 626,279,877 |
| Tiempo/op | 1.597 ns |

## Multicore scaling
| Threads | Ops/seg | Speedup |
| 1 | 626M | 1.00x |
| 2 | 1,158M | 1.85x |
| 4 | 2,164M | 3.46x |
```

---

## 5. Flujo completo de ejemplo

```sh
# 1. Descargar un modelo GGUF
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# 2. Compilar a .bmlgraph
bml-cli compile -m tinyllama-1.1b-chat-v1.0.Q4_0.gguf -o tinyllama.bmlgraph

# 3. Ejecutar inferencia
bml-cli run -m tinyllama.bmlgraph -p "What is the capital of France?" -n 64 --temp 0.7

# 4. Iniciar servidor
bml-server -m tinyllama.bmlgraph --port 8080

# 5. Benchmark
bml-bench --multicore --reps 10
```

---

## Estado actual de implementación

### ✅ Implementado y funcional

| Componente | Estado | Notas |
|-----------|--------|-------|
| `bml-domain` | ✅ | Operador BML, AST, SoA, transformer |
| `bml-parser` | ✅ | Parser GGUF zero-copy con mmap2 |
| `bml-compiler` | ✅ | DAG, hash consing, RPN, fragmentación, InferenceCompiler, compile_gguf_fast |
| `bml-runtime` | ✅ | Hot loop RPN, ejecución secuencial/paralela, queue lock-free |
| `bml-cli compile` | ✅ | Compila GGUF → .bmlgraph (solo metadatos, instantáneo) |
| `bml-cli run` | ✅ | Dual-mode: .bmlgraph/ o .gguf (inferencia autoregresiva) |
| `bml-server` | ✅ | Dual-mode: .bmlgraph/ o .gguf, HTTP+SSE, backpressure |
| `bml-bench` | ✅ | Benchmark single + multicore |

### Sin hardcodeos

- `-m` es obligatorio en CLI y server (no hay defaults)
- `compile_model` example toma args en lugar de ruta fija
- `context_size` respeta `config.context_length` del modelo (sin `.min(512)`)

### Notas

- **Modo `.bmlgraph`**: ejecuta el DAG BML compilado (operador puro). No genera
  texto autoregresivo — eso requiere el modo GGUF con InferenceCompiler.
- **Modo `.gguf`**: carga pesos en RAM (~4GB para TinyLlama 1.1B Q4_0) y ejecuta
  inferencia completa con matmul, attention, RoPE, sampling en f64.
- **`compile`** es instantáneo (solo lee metadatos del GGUF, no carga pesos).
- **`run -m .gguf`** tarda ~10s en cargar (dequantiza 1B pesos a f32).
