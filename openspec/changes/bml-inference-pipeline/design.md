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
- **D13 — N hot loops especializados.** Cada operación del transformer (matmul, RMSNorm, attention, MLP) se compila como un fragmento separado con su propio hot loop < 32KB. Los fragmentos se ejecutan secuencialmente o en paralelo según los cores disponibles. *Racional:* preserva el operador BML puro (sin kernel nativo), permite paralelismo natural entre cores, y el cambio de hot loop es instantáneo (cargar < 32KB en L1i).
- **D14 — Buffer circular entre cores.** `ResultBuffer` con N slots pre-asignados, cada slot es un `Vec<f64>` de tamaño `n_embd`. Los hot loops escriben outputs con `StoreResult { slot, offset }` y leen inputs del siguiente con `VarIndexed { base }`. *Racional:* cero allocs, cero copias, comunicación directa entre cores via memoria compartida.
- **D15 — VarIndexed para indexación dinámica.** `RpnOp::VarIndexed { base }` lee `Var(base + offset)` donde `offset` viene del tope de la pila. *Racional:* permite que los loops accedan a pesos e inputs por índice dinámico (i*cols + j) sin expandir el DAG.
- **D16 — StoreResult para escritura de resultados.** `RpnOp::StoreResult { slot, offset }` escribe el tope de la pila al `ResultBuffer`. *Racional:* cada hot loop escribe su output al buffer para que el siguiente lo lea.
- **D17 — Cambio de hot loop.** Cuando hay más fragmentos que cores, un core ejecuta varios fragmentos secuencialmente. El cambio es cargar el siguiente fragmento (< 32KB) en L1i. *Racional:* sin context switching del OS, solo cambio de código en L1i.

## Risks / Trade-offs

### Brecha de ejecución distribuida verificada

El worker histórico solo aplica RMSNorm y una proyección Q; no implementa una
capa Llama completa. `bml-inference::Session` sí conserva claves/valores por capa,
posición causal y orden transaccional mediante `TokenStep`. Para reutilizarlo
entre nodos, el protocolo debe transportar identidad de sesión, posición de
token, rango de capas y configuración completa (incluidos epsilon y RoPE).
El mensaje histórico de hidden state y `ModelConfig` no contiene esos datos.
No se puede sustituir `execute_layers` por una llamada sin estado ni inferir
los parámetros ausentes sin romper el contrato. La próxima integración debe
comparar logits de varios tokens/sesiones contra la ruta local, cubrir rechazo
de posiciones fuera de orden y rollback/cierre; el matmul TCP actual no es esa
validación. La carga de pesos y su acknowledgement tampoco son inferencia.

La API `Model::layer_session(range, context, maximum_bytes)` ya ejecuta rangos
fijos mediante el mismo kernel causal validado. `embed_token` prepara la entrada
y `final_logits` aplica la salida del pipeline. Cada `LayerSession` comprueba
posición, ancho y finitud antes de ejecutar; solo avanza después de completar
su rango. Un fallo numérico permite reintentar la misma posición, sobrescribiendo
las entradas KV no confirmadas. Las pruebas comparan tres tokens y dos sesiones
intercaladas contra el evaluador local, además de un fallo en la segunda capa.
Actualmente conserva el modelo completo y reserva KV para todas sus capas bajo
presupuesto; no acredita ahorro de memoria por partición. El commit es local a
cada etapa: el coordinador todavía debe gestionar fallos entre etapas, identidad
de sesión y reintentos sin ejecutar dos veces una posición confirmada.

`bml-inference::remote` añade transporte TCP versionado para estas sesiones,
distinto del protocolo histórico del worker. Una conexión dedicada identifica
una sesión y su KV; `Open` fija contexto, `Forward` lleva posición y hidden state,
y `Reset`/`Close` controlan el ciclo de vida. El servidor impone límites de frame,
contexto, memoria y timeouts. Un error de transporte invalida el cliente; no
reintenta automáticamente una operación que puede haber quedado confirmada.
Una prueba con dos listeners TCP y capas 0..1/1..2 compara logits contra la
ejecución local para varios tokens y después de reset. Hay pruebas de tamaño de
frame y pérdida de respuesta. `bml-layer-worker` implementa arranque/configuración por CLI, límites de sesiones
y política de saturación mediante cierre de conexión antes de reservar KV. Una
prueba de proceso con GGUF sintético verifica tres tokens y salida `--once`.
El endpoint HTTP se conecta mediante `--layer-workers`: cada request abre sus
sesiones causales, soporta JSON/SSE y propaga errores remotos. El handler tiene
pruebas contra generación local, incluidas fronteras UTF-8. Aún valida el modelo
por request y procesa el batch secuencialmente; no es batching numérico fusionado.
`RemotePipeline` coordina prefill/decode,
verifica cobertura contigua y única de capas, y solo avanza la posición después
de todas las etapas y proyección final. Un fallo parcial invalida y desconecta
todas las etapas; no permite reset/reintento sobre cachés divergentes. Las pruebas
comparan logits, rechazan planes incompletos e inyectan fallo en la segunda etapa.
La versión 2 de apertura exige una identidad SHA-256 de configuración numérica
y tensores (orden canónico por nombre, encoding, dimensiones y bytes), calculada
una sola vez por modelo. El worker rechaza otra revisión antes de reservar KV.
No incluye tokenizer ni layout físico del archivo y no autentica al servidor;
las pruebas cubren reordenación de recursos, cambios de pesos/configuración y
rechazo de identidad errónea por TCP.

- **R1 — Fórmulas BML de `+`/`-`/`*`/`/`/`pow` no derivadas.** Las fórmulas exactas en base 2 no están en el paper fuente. *Mitigación:* derivarlas del Supplementary Information o usar aproximaciones polinomiales. Si no se derivan, el transformer no puede compilarse a BML puro.
- **R2 — Overhead del intérprete RPN.** El benchmark mostró ~64x overhead vs naive matmul. *Mitigación:* el hot loop nativo (futuro) y SIMD reducirán esto. El pipeline funciona aunque sea lento.
- **R3 — gRPC añade dependencias pesadas.** `tonic` + `tokio` + `prost` aumentan el tamaño del binario y pueden impactar D5 (hot loop < 32 KB). *Mitigación:* aislar el código gRPC en un módulo separado; el hot loop no toca el stack gRPC.
- **R4 — Work-stealing puede introducir bugs de concurrencia.** *Mitigación:* WorkQueue migra a la FIFO segmentada `crossbeam-queue::SegQueue`, que reclama posición antes de leer el payload. Los modelos del deque circular anterior detectaron conflicto de lectura especulativa con reutilización. Se prueba el código real de SegQueue con slots/atómicos Loom; la reclamación y liveness no se presuponen verificadas (tarea 9.2).
- **R5 — Cuantización GGUF.** Los modelos GGUF usan cuantización (Q4_0, Q4_K, Q8_0). El compilador debe dequantizar o operar directamente sobre los tensores cuantizados. *Mitigación:* dequantizar a F32 en el compilador (simple pero pierde el beneficio de cuantización) o implementar ops cuantizadas en BML (complejo).
- **R6 — Memoria de modelos grandes.** Un modelo 7B en F32 ocupa ~28GB. *Mitigación:* mantener mmap zero-copy del GGUF; solo compilar los metadatos y referencias a tensores, no copiar los datos.
- **R7 — Latencia gRPC.** La comunicación entre nodos añade latencia. *Mitigación:* usar streaming bidireccional y batching de fragmentos.
- **R8 — API compatible con llama.cpp es extensa.** `llama-cli` tiene muchos flags. *Mitigación:* implementar los flags core (`-m`, `-p`, `-n`, `-t`, `--temp`) primero; los avanzados (`--grammar`, `--json-schema`, etc.) después.

### Reparto de batches entre réplicas

Cada repetición CLI de `--layer-workers` define una réplica completa del pipeline.
El número de réplicas configura `RequestBatchConfig::nodes`; el consumidor usa
`batch.node` sin recalcular la selección por request. Dos batches consecutivos
se verifican contra dos workers TCP reales. El contador de batches permite
comprobar que diez handlers preadmitidos forman un lote sin depender del despertar
de hilos dentro de una ventana temporal. El procesamiento interno sigue siendo
secuencial y no acredita fusión de operaciones tensoriales ni throughput mayor.
## Extensión de la ruta: pesos BML persistentes

La conversión GGUF → `.bmlgguf` y la carga de pesos preparados se especifican en
[la especificación de pesos BML](../../specs/bml-model-weights/spec.md). Esta etapa transforma
los pesos a términos BML persistentes y permite consumirlos sin el GGUF original.
Su implementación y validación están documentadas en las
[tareas archivadas](../archive/2026-11-03-gguf-to-bml-weights/tasks.md).
