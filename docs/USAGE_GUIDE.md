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
| `bml-compiler` | ✅ | DAG, hash consing, RPN, fragmentación, InferenceCompiler |
| `bml-runtime` | ✅ | Hot loop RPN, ejecución secuencial/paralela, queue lock-free |
| `bml-cli` | ⚠️ | Parcial — ver abajo |
| `bml-server` | ⚠️ | Parcial — ver abajo |
| `bml-bench` | ✅ | Benchmark funcional |

### ⚠️ Parcialmente implementado (lo que falta)

#### bml-cli

**Faltante:**

1. **Subcomando `compile`**: No existe. Actualmente el CLI solo tiene `run` que ejecuta
   directamente desde GGUF sin compilar a `.bmlgraph` primero.

2. **Subcomando `run` desde `.bmlgraph`**: El `run` actual llama a
   `InferenceCompiler::open(gguf_path)` que abre el GGUF directamente. Falta un
   camino que cargue un `.bmlgraph/` pre-compilado desde disco usando
   `load_from_dir()`.

3. **Flag `--l1-threshold` / `--l3-threshold`**: No existen en el CLI actual.

**Hardcodeos a eliminar:**

- `crates/cli/src/main.rs` línea 76: `let max_ctx = args.context_size.min(512)` —
  el `.min(512)` es un cap arbitrario que debería ser configurable.
- `crates/cli/src/main.rs` línea 99: `unwrap_or(vocab.eos_token_id)` — si el
  sampling falla, usa EOS. Esto es razonable pero debería loguear.

#### bml-server

**Hardcodeos a eliminar:**

- `crates/api/src/main.rs` línea 13: `let model_path = "model.gguf"` como default.
  Debería requerir `-m` o dar error.
- `crates/api/src/main.rs`: usa `InferenceCompiler::open()` que abre GGUF directo.
  Falta soporte para cargar `.bmlgraph/` compilado.

**Faltante:**

4. **Cargar `.bmlgraph/` en lugar de GGUF crudo**: El servidor debería poder
   cargar un directorio `.bmlgraph/` pre-compilado usando `load_from_dir()`.

#### Flujo de compilación → ejecución

5. **CLI `compile` subcomando**: Necesita:
   - Leer GGUF con `InferenceCompiler::open()`
   - Compilar con `compile_gguf()`
   - Serializar con `serialize_to_dir()`
   - Mensajes de progreso

6. **CLI `run` desde `.bmlgraph`**: Necesita:
   - Detectar si `-m` es un directorio `.bmlgraph/` o un archivo GGUF
   - Si es `.bmlgraph/`, usar `load_from_dir()` + `Runtime::execute_graph()`
   - Si es GGUF, usar `InferenceCompiler::open()` (camino actual)

7. **Server desde `.bmlgraph`**: Igual que #6 pero en el servidor.

---

## Plan para completar lo faltante

### Fase 1: Subcomando `compile` en bml-cli

```
bml-cli compile -m modelo.gguf -o modelo.bmlgraph [--l1-threshold N] [--l3-threshold N]
```

- Llama `InferenceCompiler::open()` + `compile_gguf()` + `serialize_to_dir()`
- Output: directorio `.bmlgraph/` con header + fragmentos

### Fase 2: Subcomando `run` dual-mode

```
bml-cli run -m modelo.bmlgraph/ -p "prompt" -n 64
```

- Detectar si `-m` es directorio (`.bmlgraph/`) o archivo (`.gguf`)
- Si directorio: `load_from_dir()` → `Runtime::execute_graph()`
- Si archivo: `InferenceCompiler::open()` (camino actual)
- Eliminar `.min(512)` hardcodeado

### Fase 3: Server dual-mode

- `bml-server -m modelo.bmlgraph/` carga pre-compilado
- `bml-server -m modelo.gguf` compila en caliente (camino actual)

### Fase 4: Limpieza

- Eliminar `model.gguf` default en server
- Hacer `-m` obligatorio en ambos
- Mensajes de error claros
