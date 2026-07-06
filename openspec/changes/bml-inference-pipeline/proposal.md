## Why

El runtime BML actual (Hito 5) ejecuta programas RPN sobre un DAG fragmentado, pero no puede correr un modelo GGUF real: falta el compilador que traduce tensores GGUF a `.bmlgraph`, el runtime distribuido que ejecuta fragmentos entre nodos, y la API compatible con `llama.cpp` que permite usar BML como drop-in replacement. Sin este pipeline, BML es un motor matemático sin aplicación práctica.

## What Changes

- **Compilador GGUF → `.bmlgraph`**: nuevo módulo en `crates/compiler/` que toma un archivo GGUF (parseado con `bml-parser`) y lo compila a un conjunto de archivos `.bmlgraph` fragmentados, optimizados para la máquina de compilación (número de cores, tamaños de caché L1/L2/L3).
- **Runtime distribuido con gRPC**: nuevo módulo en `crates/runtime/` que expone una API gRPC para recibir fragmentos `.bmlgraph`, ejecutarlos, y coordinarse con otros nodos vía un sistema de colas (channel-based) que evita condiciones de carrera.
- **Sistema de colas interno**: cada nodo mantiene una cola de fragmentos pendientes; los nodos se "roban" trabajo entre sí (work-stealing) cuando su cola se vacía, sin locks (lock-free queue).
- **API compatible con `llama.cpp`**: binario `bml-cli` que expone los mismos flags que `llama-cli` (`-m`, `-p`, `-n`, `-t`, `--temp`, etc.) y produce texto generado, de forma que BML sea un drop-in replacement.
- **Empaquetado AOT**: el compilador genera el **número mínimo de fragmentos** optimizado para la máquina objetivo (detecta cores y caché en compile-time o con un flag `--target`).

## Capabilities

### New Capabilities

- `gguf-compiler`: Compila un modelo GGUF a un conjunto de archivos `.bmlgraph` fragmentados, traduciendo cada operación del transformer (attention, MLP, RMSNorm, RoPE, softmax) a la gramática BML. Optimiza el número de fragmentos según la máquina objetivo (cores, caché L1/L2/L3).
- `distributed-runtime`: Runtime distribuido que ejecuta fragmentos `.bmlgraph` entre múltiples nodos vía gRPC, con sistema de colas lock-free y work-stealing para balancear carga sin condiciones de carrera.
- `bml-cli`: Binario con API compatible con `llama.cpp` (`-m`, `-p`, `-n`, `-t`, `--temp`) que permite usar BML como drop-in replacement de `llama-cli`.

### Modified Capabilities

- `bml-runtime`: Se extiende para soportar ejecución distribuida (recibir fragmentos remotos vía gRPC, ejecutarlos, devolver resultados). El hot loop local se mantiene sin cambios (D5: < 32 KB).
- `bml-compiler`: Se extiende con el módulo `gguf_compiler` que traduce tensores GGUF a `.bmlgraph`. El Hash Consing y la fragmentación existente se reutilizan.

## Impact

- **Nuevos crates/módulos:**
  - `crates/compiler/src/gguf_compiler.rs`: traducción GGUF → BML.
  - `crates/runtime/src/distributed.rs`: runtime distribuido con gRPC.
  - `crates/runtime/src/queue.rs`: cola lock-free y work-stealing.
  - `crates/cli/`: nuevo crate con binario `bml-cli`.
- **Dependencias nuevas:**
  - `tonic` + `prost` para gRPC.
  - `tokio` para runtime async (gRPC requiere async).
  - `clap` para CLI compatible con `llama.cpp`.
  - `crossbeam` o `loom`-verified queue para cola lock-free.
- **Protocolo gRPC:** se define un `.proto` con servicios `ExecuteFragment`, `StealWork`, `ReportResult`.
- **Detección de hardware:** el compilador detecta cores (`num_cpus`), tamaños de caché (`std::fs::read("/sys/devices/system/cpu/...")` o `lscpu`), y genera el número mínimo de fragmentos.
- **Breaking:** no rompe APIs existentes; todo es aditivo.
- **Complejidad:** este change es grande. Se descompone en fases en `tasks.md` para ejecución incremental.
