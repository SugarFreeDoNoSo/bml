## Context

El proyecto BML parte de `draft.md`, que define un compilador y motor de ejecución en Rust basado en un único operador matemático (BML), análogo continuo del NAND lógico, con completitud funcional. El sistema debe ejecutarse sobre la jerarquía de memoria del procesador con énfasis en L1. No existe código previo; el repo contiene solo `draft.md` y `references/`. La hoja de ruta del draft define 5 hitos secuenciales (Cimientos → Compilación/Deduplicación → Parser GGUF → Micro-Fragmentación → Runtime L1).

## Goals / Non-Goals

**Goals:**

- G1. Establecer un workspace Cargo con 4 crates de responsabilidad aislada (`bml-domain`, `bml-parser`, `bml-compiler`, `bml-runtime`).
- G2. Implementar el operador BML y su transformador (mapper) que reduce operaciones estándar a la gramática BML recursiva usando solo el operador y la constante 1.
- G3. Garantizar mechanical sympathy: SoA, `#[repr(align(64))]`, hot loop < 32 KB de instrucciones, append-only durante la evaluación del DAG.
- G4. Implementar Hash Consing para deduplicar sub-árboles BML idénticos y demostrar reducción a tiempo sub-lineal para operaciones repetidas.
- G5. Implementar ingesta GGUF zero-copy vía `memmap2`.
- G6. Micro-fragmentación AOT del DAG exportado (`.bmlgraph`) bajo umbral de caché (32 KB L1, configurable a L3).
- G7. Implementar el runtime RPN con cero allocs en hot path y la interfaz RPC/binaria para distribución append-only.

**Non-Goals:**

- NG1. No se implementa OOP para la capa de ejecución (prohibido por el draft).
- NG2. No se sobrescribe estado previo durante la evaluación del DAG (append-only estricto).
- NG3. No se generan todas las specs del sistema upfront; se crean por capability conforme se avanza.
- NG4. No se optimiza para GPU ni SIMD explícita en este change (queda fuera del alcance del draft original).
- NG5. No se sustituye `draft.md`; se conserva como referencia histórica.

## Decisions

- **D1 — Workspace con 4 crates.** Se replica la topología del draft con 4 crates (`bml-domain`, `bml-parser`, `bml-compiler`, `bml-runtime`). Las **carpetas** del workspace no llevan prefijo `bml-` (es redundante dentro del workspace `bml`); los **nombres de paquete** sí lo conservan para el namespace en crates.io. `bml-domain` tiene cero dependencias para garantizar reutilización sin acoplamiento. *Racional:* separación de responsabilidades que permite compilar y testear cada capa de forma independiente y aplicar LTO por crate.
- **D2 — Perfil release extremo en el Cargo.toml raíz.** `lto = "fat"`, `panic = "abort"`, `codegen-units = 1`, `opt-level = 3`. *Racional:* maximizar inlining y reducir el tamaño del binario del hot loop; `panic = "abort"` elimina la tabla de unwinding.
- **D3 — Data-Oriented Design, prohibido OOP en la capa de ejecución.** *Racional:* el draft lo prohíbe explícitamente; el rendimiento depende del layout de memoria, no de abstracciones de objeto.
- **D4 — SoA + `#[repr(align(64))]` para nodos del grafo.** Las estructuras de nodos se almacenan como Struct of Arrays alineadas a 64 bytes (línea de caché típica). *Racional:* evita false sharing entre hilos y garantiza que la CPU solo cargue en caché los bytes estrictamente necesarios para la evaluación matemática.
- **D5 — Hot loop RPN < 32 KB de instrucciones.** El intérprete RPN del runtime debe compilar a un tamaño de código inferior a 32 KB para caber en L1i. *Racional:* si el hot loop es expulsado de L1i, el rendimiento colapsa. Se valida con `perf stat -e instructions` o `cargo asm` midiendo el tamaño del binario del loop.
- **D6 — Append-only durante la evaluación del DAG.** Un hilo lee `v_i` de un nodo, computa `v_o` y lo escribe a una dirección pre-asignada nueva. Nunca sobrescribe. *Racional:* habilita paralelismo lock-free entre trabajadores y consistencia para distribución.
- **D7 — Hash Consing en `bml-compiler`.** Registro global de sub-árboles BML que permite deduplicar sub-árboles matemáticamente idénticos en tiempo de compilación. *Racional:* reduce el DAG y demuestra la reducción a tiempo sub-lineal `O(n^k)` con `k < 1` para operaciones repetidas (objetivo del benchmark del Hito 2).
- **D8 — RPN como representación linealizada del DAG.** El compilador convierte el DAG deduplicado en un arreglo unidimensional en Notación Polaca Inversa. *Racional:* el runtime itera secuencialmente sobre RPN sin saltos ni recursión, ideal para L1i.
- **D9 — Zero-Copy mmap con `memmap2` en `bml-parser`.** Los tensores GGUF se referencian desde el disco directo al espacio de memoria de Rust. *Racional:* elimina copias a RAM para archivos grandes (modelos GGUF típicos).
- **D10 — Micro-fragmentación AOT y formato `.bmlgraph`.** El compilador empaqueta el DAG en fragmentos cuyo tamaño de memoria pre-asignada no supera el umbral de caché objetivo (32 KB L1 por defecto, configurable a L3). *Racional:* garantiza que cada fragmento ejecutable caiba enteramente en la caché objetivo.
- **D11 — Runtime con cero allocs en hot path.** Los buffers se inicializan una sola vez al arrancar. *Racional:* cualquier alloc en el hot loop rompe la predictibilidad de latencia.
- **D12 — Interfaz RPC/binaria (gRPC u otro) para distribución.** El runtime expone RPC para recibir y transmitir fragmentos `.bmlgraph` entre nodos. *Racional:* habilita el escalado horizontal append-only del Hito 2 y el runtime distribuido del Hito 5.
- **D13 — Benchmarks con `criterion`.** Se usa `criterion` como dev-dependency para los benchmarks comparativos FMA vs BML del Hito 2. *Racional:* es el estándar de facto en el ecosistema Rust y produce reportes estadísticamente significativos.
- **D14 — Pruebas de cache hit/miss con `perf`.** Se usan `perf stat` y `perf record` sobre los tests de estrés multicore para medir L1/L2 hit/miss. *Racional:* es la única forma de validar empíricamente D5 y D10.
- **D15 — BML = Binary-Minus-Log.** El operador se nombra BML (Binary-Minus-Log) como análogo de EML (Exp-Minus-Log) en base 2: `bml(x, y) = 2^x - log2(y)`. La base 2 se alinea con el formato IEEE 754 de `f64` y permite usar `exp2`/`log2` nativos de la FPU. *Racional:* preserva la completitud funcional del operador EML (paper ArXiv 2603.21852v2) adaptándolo a base 2, evitando `exp`/`ln` que son más costosos y no se alinean con el formato nativo de `f64`.

## Risks / Trade-offs

- **R1 — Tamaño del hot loop.** Si el intérprete RPN crece por encima de 32 KB (por ejemplo, por soporte de muchos opcodes), se viola D5. *Mitigación:* medir con `cargo asm`/`perf` en cada PR que toque el runtime; refactorizar a tablas de saltos compactas si se acerca al límite.
- **R2 — Hash Consing correctness.** Un bug en el hash de sub-árboles deduplicaría nodos no equivalentes, produciendo resultados matemáticamente incorrectos. *Mitigación:* pruebas de propiedad con `proptest` sobre la igualdad estructural antes de confiar en el hash.
- **R3 — `memmap2` y lifetime de los tensores.** Los tensores mapeados viven mientras el mmap esté abierto; un cierre prematuro provoca UB. *Mitigación:* encapsular el mmap en un RAII guard con lifetime explícito ligado al parser.
- **R4 — Append-only y memoria.** El patrón append-only consume memoria proporcional al número de evaluaciones. *Mitigación:* pool de buffers pre-asignados rotativos (siempre pre-asignados, sin alloc en hot path).
- **R5 — Distribución lock-free sobre `/dev/shm`.** La prueba distribuida del Hito 2 puede enmascarar bugs de concurrencia que solo aparezcan bajo estrés. *Mitigación:* usar `loom` para pruebas de concurrencia determinísticas además del test en `/dev/shm`.
- **R6 — gRPC en el runtime.** Añadir `tonic`/gRPC aumenta el tamaño del binario y puede impactar D5 indirectamente. *Mitigación:* aislar el código RPC en un módulo fuera del hot loop; el hot loop no debe tocar el stack gRPC.
- **R7 — Spec drift respecto al draft.** Si el draft evoluciona, las specs pueden desincronizarse. *Mitigación:* `draft.md` se referencia desde `design.md` como fuente histórica; cualquier cambio funcional se rutea por un nuevo change de OpenSpec.
