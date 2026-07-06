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
- **D4 — gRPC con tonic + tokio.** El runtime distribuido usa `tonic` (gRPC) sobre `tokio` (async runtime). *Racional:* es el stack gRPC estándar en Rust, con soporte para streaming bidireccional.
- **D5 — Cola lock-free con work-stealing.** Cada nodo mantiene una cola local (lock-free, tipo Chase-Lev o `crossbeam-deque`) de fragmentos pendientes. Cuando un nodo vacía su cola, "roba" trabajo de otro nodo vía RPC `StealWork`. *Racional:* balancea carga sin locks centrales, evita cuellos de botella.
- **D6 — Append-only para resultados.** Cada nodo escribe resultados a un buffer pre-asignado (patrón append-only del Hito 5). Los resultados se propagan al nodo coordinador vía RPC `ReportResult`. *Racional:* evita condiciones de carrera en escritura; cada nodo escribe solo a su buffer.
- **D7 — Protocolo gRPC.** Se define `bml.proto` con servicios:
  - `ExecuteFragment(FragmentRequest) -> FragmentResult`: ejecuta un fragmento y devuelve el resultado.
  - `StealWork(StealRequest) -> StealResponse`: roba un fragmento de la cola del nodo remoto.
  - `ReportResult(ResultRequest) -> Ack`: reporta un resultado al coordinador.
  - `HealthCheck(Empty) -> HealthStatus`: verifica que un nodo está vivo.
- **D8 — CLI compatible con llama.cpp.** `bml-cli` usa `clap` con los mismos flags que `llama-cli`: `-m` (modelo), `-p` (prompt), `-n` (n tokens), `-t` (threads), `--temp` (temperatura), `-c` (context size), `--top-k`, `--top-p`. *Racional:* drop-in replacement.
- **D9 — Compilación AOT separada de ejecución.** El compilador genera los `.bmlgraph` en una fase separada (`bml-compile model.gguf --target local`), y el runtime los carga (`bml-cli -m model.bmlgraph/`). *Racional:* permite optimizar una vez y ejecutar muchas.
- **D10 — Coordinador vs worker.** Un nodo actúa como coordinador (recibe el prompt, particiona el trabajo, agrega resultados) y los demás como workers (ejecutan fragmentos). El coordinador también puede ejecutar fragmentos. *Racional:* simplifica la topología sin un scheduler externo.
- **D11 — Detección de hardware en compile-time.** El compilador acepta `--target local` (detecta hardware actual), `--target specs:<cores>:<L1>:<L2>:<L3>` (especifica manualmente), o `--target cloud-gcp-n2` (presets). *Racional:* flexibilidad para compilar en una máquina y ejecutar en otra.

## Risks / Trade-offs

- **R1 — Fórmulas BML de `+`/`-`/`*`/`/`/`pow` no derivadas.** Las fórmulas exactas en base 2 no están en el paper fuente. *Mitigación:* derivarlas del Supplementary Information o usar aproximaciones polinomiales. Si no se derivan, el transformer no puede compilarse a BML puro.
- **R2 — Overhead del intérprete RPN.** El benchmark mostró ~64x overhead vs naive matmul. *Mitigación:* el hot loop nativo (futuro) y SIMD reducirán esto. El pipeline funciona aunque sea lento.
- **R3 — gRPC añade dependencias pesadas.** `tonic` + `tokio` + `prost` aumentan el tamaño del binario y pueden impactar D5 (hot loop < 32 KB). *Mitigación:* aislar el código gRPC en un módulo separado; el hot loop no toca el stack gRPC.
- **R4 — Work-stealing puede introducir bugs de concurrencia.** *Mitigación:* usar `crossbeam-deque` (verificado) y tests con `loom`.
- **R5 — Cuantización GGUF.** Los modelos GGUF usan cuantización (Q4_0, Q4_K, Q8_0). El compilador debe dequantizar o operar directamente sobre los tensores cuantizados. *Mitigación:* dequantizar a F32 en el compilador (simple pero pierde el beneficio de cuantización) o implementar ops cuantizadas en BML (complejo).
- **R6 — Memoria de modelos grandes.** Un modelo 7B en F32 ocupa ~28GB. *Mitigación:* mantener mmap zero-copy del GGUF; solo compilar los metadatos y referencias a tensores, no copiar los datos.
- **R7 — Latencia gRPC.** La comunicación entre nodos añade latencia. *Mitigación:* usar streaming bidireccional y batching de fragmentos.
- **R8 — API compatible con llama.cpp es extensa.** `llama-cli` tiene muchos flags. *Mitigación:* implementar los flags core (`-m`, `-p`, `-n`, `-t`, `--temp`) primero; los avanzados (`--grammar`, `--json-schema`, etc.) después.
