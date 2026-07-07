## 0. Bloqueadores: parser GGUF completo y AST con variables/constantes

- [x] 0.1 Extender parser GGUF: decodificar metadatos KV (pares clave-valor con tipos: string, int, float, array).
- [x] 0.2 Extender parser GGUF: decodificar tensor infos (nombre, n_dims, dims, data_type, offset de cada tensor).
- [x] 0.3 Extender parser GGUF: acceso a datos del tensor via slice sobre el mmap (zero-copy).
- [x] 0.4 Extender parser GGUF: leer `general.architecture` para saber si es llama, qwen, etc.
- [x] 0.5 Extender AST: añadir `Var(id)` para representar inputs variables (tokens del prompt).
- [x] 0.6 Extender AST: añadir `Const(f64)` para representar pesos del modelo (constantes arbitrarias).
- [x] 0.7 Actualizar gramática: `S -> 1 | Var(id) | Const(f64) | BML(S, S)`.
- [x] 0.8 Actualizar `evaluate()` para resolver `Var` desde un contexto de inputs y `Const` desde un pool de pesos.
- [x] 0.9 Actualizar `RpnOp` con `Var(u32)` y `Const(u32)` (índices al pool de inputs/pesos).
- [x] 0.10 Actualizar hot loop del runtime para resolver `Var` y `Const` desde buffers pre-asignados.
- [x] 0.11 Pruebas: construir un DAG con `Var` y `Const`, evaluarlo con inputs distintos, verificar resultados.

## 1. Operaciones del transformer (EML compile-time + BML runtime)

- [x] 1.1 Crear `crates/compiler/src/eml.rs` con funciones de compile-time: `eml(x,y)`, `exp(x)`, `ln(x)`, `sin(x)`, `cos(x)`, `pi()`, `i_unit()`, `rope(x, pos, freq)`.
- [x] 1.2 Derivar `softmax(x) = exp(x) / sum(exp(x))` en BML usando `exp2`, `add`, `div` (no necesita EML).
- [x] 1.3 Derivar `RMSNorm(x) = x / sqrt(mean(x^2) + eps)` en BML usando `mul`, `div`, `pow`, `add` (no necesita EML).
- [x] 1.4 Derivar `SwiGLU(x) = x * sigmoid(1.7 * x)` en BML usando `mul`, `add`, `div`, `neg` (no necesita EML).
- [x] 1.5 Derivar `RoPE(x, pos, freq)` usando EML en compile-time: precomputar `cos(pos*freq)` y `sin(pos*freq)` como `Const`, luego `x * Const + rotate(x) * Const` en BML runtime.
- [x] 1.6 Derivar `sin(x)` y `cos(x)` via Euler en EML compile-time (usa Complex<f64>). El resultado se almacena como `Const`.
- [x] 1.7 Pruebas de cada fórmula: verificar que BML runtime produce el mismo resultado que la referencia directa.

## 2. Compilador GGUF → .bmlgraph

- [x] 2.1 Crear `crates/compiler/src/gguf_compiler.rs` con la función `compile_gguf_to_bmlgraph(gguf_path, target) -> Vec<BmlGraph>`.
- [x] 2.2 Implementar la detección de hardware: `num_cpus`, `/sys/devices/system/cpu/...`, `lscpu` para obtener cores, L1, L2, L3.
- [x] 2.3 Implementar el cálculo del número mínimo de fragmentos: `max(1, ceil(total_ops / (L1_threshold * cores)))`.
- [x] 2.4 Implementar la traducción de cada capa del transformer (attention, MLP, norm) a sub-DAGs BML usando los pesos como `Const` y los inputs como `Var`.
- [x] 2.5 Concatenar los sub-DAGs y aplicar Hash Consing + fragmentación AOT existente.
- [x] 2.6 Serializar los `.bmlgraph` a disco (un archivo por fragmento o un directorio). Los pesos se referencian desde el GGUF mmap (zero-copy), no se copian.
- [x] 2.7 Pruebas: compilar un GGUF sintético y verificar que los `.bmlgraph` se generan correctamente.

## 3. Comunicación interna entre nodos (TCP raw + /dev/shm)

- [x] 3.1 Implementar `crates/runtime/src/net.rs` con protocolo TCP raw: framing `[u32 msg_type][u32 payload_len][payload]`.
- [x] 3.2 Implementar tipos de mensaje: `ExecuteFragment`, `ReportResult`, `StealWork`, `HealthCheck`, `BatchRequest`, `BatchResult`.
- [x] 3.3 Implementar `NodeHandle` que envuelve `TcpStream` con métodos `send_fragment()`, `recv_result()`.
- [x] 3.4 Implementar `/dev/shm` para same-machine: fragmentos en memoria compartida, cero copia, cero serialización.
- [x] 3.5 Implementar `crates/runtime/src/queue.rs` con cola lock-free (`crossbeam-deque` Chase-Lev).
- [x] 3.6 Implementar work-stealing: cuando un nodo vacía su cola, roba trabajo de otro vía TCP `StealWork`.
- [x] 3.7 Aislar el código de red del hot loop (módulo separado, no impactar D5: < 32 KB).
- [x] 3.8 Pruebas de integración: 2 nodos se comunican via TCP, uno envía un fragmento, el otro lo ejecuta y devuelve el resultado.
- [x] 3.9 Pruebas de `/dev/shm`: 4 workers leen un fragmento de memoria compartida, lo ejecutan y escriben resultados append-only.

## 4. API externa + CLI (HTTP + SSE, OpenAI-compatible)

- [ ] 4.1 Crear `crates/api/` como nuevo miembro del workspace con binario `bml-server`.
- [ ] 4.2 Añadir `axum` + `serde_json` como dependencias (ligero, no toca el hot loop).
- [ ] 4.3 Implementar endpoint `POST /v1/completions` compatible con OpenAI: `{"prompt": "...", "max_tokens": 10, "stream": true}`.
- [ ] 4.4 Implementar streaming via SSE (Server-Sent Events): `data: {"token": "..."}\n\n`.
- [ ] 4.5 Implementar batching: múltiples requests HTTP se encolan en el scheduler (Capa 2).
- [ ] 4.6 Implementar backpressure: si la cola está llena, retornar HTTP 429.
- [ ] 4.7 Crear `crates/cli/` con binario `bml-cli` (flags `-m`, `-p`, `-n`, `-t`, `--temp`, `-c`).
- [ ] 4.8 Implementar `bml-cli` que carga `.bmlgraph`, ejecuta inferencia local (sin servidor), produce texto.
- [ ] 4.9 Implementar sampling greedy + temperatura básica.
- [ ] 4.10 Pruebas: `bml-cli -m model.bmlgraph/ -p "Hello" -n 10` produce texto.
- [ ] 4.11 Pruebas: `curl -X POST http://localhost:8080/v1/completions -d '{"prompt":"Hello","stream":true}'` recibe tokens via SSE.

## 5. Scheduler con batching dinámico

- [ ] 5.1 Implementar `crates/runtime/src/scheduler.rs` con cola de requests (`crossbeam-channel`).
- [ ] 5.2 Implementar batching dinámico: agrupa N prompts en una ventana de 10ms y los procesa juntos.
- [ ] 5.3 Implementar distribución de batches a nodos: round-robin o least-loaded.
- [ ] 5.4 Implementar el rol de coordinador: recibe batches, particiona el trabajo, agrega resultados.
- [ ] 5.5 Implementar el rol de worker: ejecuta fragmentos, reporta resultados, roba trabajo.
- [ ] 5.6 El coordinador también puede ejecutar fragmentos localmente.
- [ ] 5.7 Pruebas: 1 coordinador + 3 workers ejecutan un `.bmlgraph` y producen el resultado correcto.
- [ ] 5.8 Pruebas de batching: 10 requests concurrentes se agrupan en 1-2 batches.

## 6. Pruebas de concurrencia

- [ ] 6.1 Pruebas con `loom` de la cola lock-free.
- [ ] 6.2 Pruebas con `loom` del work-stealing.
- [ ] 6.3 Pruebas de append-only bajo estrés multicore.
- [ ] 6.4 Pruebas de que el hot loop no toca el código de red (aislamiento).
- [ ] 6.5 Pruebas de backpressure: saturar la cola y verificar HTTP 429.

## 7. Cierre

- [ ] 7.1 `openspec validate bml-inference-pipeline` pasa sin errores.
- [ ] 7.2 `cargo test --workspace` pasa.
- [ ] 7.3 Commit y push.
