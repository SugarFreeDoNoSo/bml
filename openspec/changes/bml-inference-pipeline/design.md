## Context

El runtime BML actual ejecuta programas RPN sobre DAGs fragmentados con Hash Consing, hot loop < 32 KB, y patrón append-only. El parser GGUF decodifica cabeceras y mapea tensores zero-copy. Sin embargo, no existe el compilador que traduce las operaciones de un transformer (attention, MLP, RMSNorm, RoPE, softmax) a la gramática BML, ni el runtime distribuido que ejecuta fragmentos entre nodos, ni la API compatible con `llama.cpp`.

`llama.cpp` define el estándar de facto para inferencia LLM en CPU: binario `llama-cli` con flags `-m`, `-p`, `-n`, `-t`, `--temp`, y soporte para modelos GGUF cuantizados. Para que BML sea un drop-in replacement, debe exponer la misma API y producir texto generado a partir de un GGUF.

## Goals / Non-Goals

**Goals:**

- G1. Compilar un modelo GGUF a un conjunto de archivos `.bmlgraph` fragmentados, traduciendo cada operación del transformer a la gramática BML.
- G2. Optimizar el número de fragmentos según la máquina objetivo (cores, L1/L2/L3).
- G3. Implementar un runtime distribuido con gRPC donde los nodos ejecutan fragmentos y se coordinan vía colas lock-free con work-stealing.
- G4. Garantizar ausencia de condiciones de carrera (append-only + colas lock-free + verificación con `loom`).
- G5. Exponer una API compatible con `llama.cpp` (`bml-cli` con flags `-m`, `-p`, `-n`, `-t`, `--temp`).
- G6. Permitir que el compilador genere el número mínimo de fragmentos para la máquina objetivo.

**Non-Goals:**

- NG1. No se implementa GPU/SIMD en este change (se mide el estado actual en CPU).
- NG2. No se implementa sampling avanzado (top-k, top-p, nucleus) — solo greedy + temperatura básica.
- NG3. No se implementa fine-tuning ni entrenamiento.
- NG4. No se reemplaza el hot loop existente (D5: < 32 KB se mantiene).
- NG5. No se compara rendimiento con llama.cpp en este change (eso es el change `bml-vs-llamacpp-bench`).

## Decisions

- **D1 — Compilador GGUF → BML por capa.** El transformer se compila capa por capa: cada capa (attention + MLP + norm) se traduce a un sub-DAG BML, y el compilador concatena los sub-DAGs. *Racional:* permite fragmentación natural por capa y paralelismo entre capas independientes.
- **D2 — Traducción de operaciones estándar a BML.** Cada operación del transformer (matmul, RMSNorm, RoPE, softmax, SwiGLU) se traduce a la gramática BML usando el `BMLTransformer` del Hito 1. Las fórmulas exactas de `+`, `-`, `*`, `/`, `pow` en base 2 se derivan del Supplementary Information del paper. *Racional:* preserva la completitud funcional del operador BML.
- **D3 — Número mínimo de fragmentos por máquina.** El compilador detecta el hardware objetivo (`num_cpus`, `/sys/devices/system/cpu/...`, `lscpu`) y calcula el número mínimo de fragmentos: `max(1, ceil(total_ops / (L1_threshold * cores)))`. *Racional:* cada core ejecuta un fragmento que cabe en su L1i.
- **D4 — TCP raw + /dev/shm para comunicación interna.** El runtime distribuido usa TCP raw con formato `.bmlgraph` nativo para cross-machine, y `/dev/shm` para same-machine. *Racional:* gRPC (tonic + tokio + prost) añade ~2-3MB al binario y overhead de protobuf. TCP raw + formato nativo es 10x más rápido y cero deps. El hot loop queda puro.
- **D5 — Cola lock-free con work-stealing.** Cada nodo mantiene una cola local (lock-free, tipo Chase-Lev o `crossbeam-deque`) de fragmentos pendientes. Cuando un nodo vacía su cola, "roba" trabajo de otro nodo vía TCP `StealWork`. *Racional:* balancea carga sin locks centrales, evita cuellos de botella.
- **D6 — Append-only para resultados.** Cada nodo escribe resultados a un buffer pre-asignado (patrón append-only del Hito 5). Los resultados se propagan al nodo coordinador vía TCP `ReportResult`. *Racional:* evita condiciones de carrera en escritura; cada nodo escribe solo a su buffer.
- **D7 — Protocolo TCP raw.** Framing: `[u32 msg_type][u32 payload_len][payload]`. Mensajes: `ExecuteFragment`, `ReportResult`, `StealWork`, `HealthCheck`, `BatchRequest`, `BatchResult`. *Racional:* mínimo overhead, sin serialización protobuf.
- **D8 — API externa HTTP + SSE.** `axum` + `serde_json` para endpoint OpenAI-compatible `POST /v1/completions` con streaming SSE. *Racional:* compatible con cualquier cliente HTTP, streaming nativo, sin gRPC.
- **D9 — Scheduler con batching dinámico.** `crossbeam-channel` para cola de requests. Agrupa N prompts en ventana de 10ms. *Racional:* maximiza throughput sin sacrificar latencia.
- **D10 — Backpressure.** Si la cola del scheduler está llena, la API retorna HTTP 429. *Racional:* evita OOM bajo carga.
- **D11 — CLI local.** `bml-cli` con `clap` ejecuta inferencia local sin servidor. *Racional:* drop-in replacement de `llama-cli` para uso interactivo.
- **D12 — Arquitectura de 3 capas.** Capa 1: API HTTP (axum). Capa 2: Scheduler con batching (crossbeam). Capa 3: Nodos BML con TCP raw + hot loop puro. *Racional:* separa concerns, el hot loop no tiene deps de red.

## Risks / Trade-offs

- **R1 — Fórmulas BML de `+`/`-`/`*`/`/`/`pow` no derivadas.** Las fórmulas exactas en base 2 no están en el paper fuente. *Mitigación:* derivarlas del Supplementary Information o usar aproximaciones polinomiales. Si no se derivan, el transformer no puede compilarse a BML puro.
- **R2 — Overhead del intérprete RPN.** El benchmark mostró ~64x overhead vs naive matmul. *Mitigación:* el hot loop nativo (futuro) y SIMD reducirán esto. El pipeline funciona aunque sea lento.
- **R3 — gRPC añade dependencias pesadas.** `tonic` + `tokio` + `prost` aumentan el tamaño del binario y pueden impactar D5 (hot loop < 32 KB). *Mitigación:* aislar el código gRPC en un módulo separado; el hot loop no toca el stack gRPC.
- **R4 — Work-stealing puede introducir bugs de concurrencia.** *Mitigación:* usar `crossbeam-deque` (verificado) y tests con `loom`.
- **R5 — Cuantización GGUF.** Los modelos GGUF usan cuantización (Q4_0, Q4_K, Q8_0). El compilador debe dequantizar o operar directamente sobre los tensores cuantizados. *Mitigación:* dequantizar a F32 en el compilador (simple pero pierde el beneficio de cuantización) o implementar ops cuantizadas en BML (complejo).
- **R6 — Memoria de modelos grandes.** Un modelo 7B en F32 ocupa ~28GB. *Mitigación:* mantener mmap zero-copy del GGUF; solo compilar los metadatos y referencias a tensores, no copiar los datos.
- **R7 — Latencia gRPC.** La comunicación entre nodos añade latencia. *Mitigación:* usar streaming bidireccional y batching de fragmentos.
- **R8 — API compatible con llama.cpp es extensa.** `llama-cli` tiene muchos flags. *Mitigación:* implementar los flags core (`-m`, `-p`, `-n`, `-t`, `--temp`) primero; los avanzados (`--grammar`, `--json-schema`, etc.) después.
