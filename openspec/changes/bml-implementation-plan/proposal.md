## Why

El `draft.md` define la arquitectura y los 5 hitos del proyecto BML (un compilador y motor de ejecución en Rust basado en el operador BML, optimizado para la jerarquía de caché del procesador), pero no existe un plan de ejecución accionable que un agente o un desarrollador pueda seguir sin reinterpretar el draft. Este change convierte el draft en un conjunto de specs vivas, decisiones de diseño y tareas verificables por hito, de forma que el desarrollo pueda ejecutarse de forma incremental y auditable.

## What Changes

- Se formalizan las 4 capacidades del sistema (dominio, parser, compilador, runtime) como specs vivas bajo `openspec/specs/`.
- Se descompone la hoja de ruta del draft (5 hitos) en fases con tareas granulares y verificables en `tasks.md`.
- Se registran las decisiones técnicas estrictas del draft (DOD, SoA, `#[repr(align(64))]`, hot loop < 32 KB, append-only, Hash Consing, Zero-Copy mmap) en `design.md` como decisiones explícitas con su racional.
- Se establecen los criterios de aceptación por hito (pruebas del operador, benchmarks FMA vs BML, pruebas de cache hit/miss con `perf`, prueba distribuida con `/dev/shm`, decodificación GGUF, fragmentación AOT < 32 KB, hot loop RPN).
- **BREAKING (organización del repo):** se introducirá un workspace Cargo con 4 crates (`bml-domain`, `bml-parser`, `bml-compiler`, `bml-runtime`) en hitos sucesivos. Las carpetas del workspace no llevan prefijo `bml-` (es redundante); los nombres de paquete sí lo conservan para el namespace en crates.io. No se elimina código existente; el cambio es aditivo.

## Capabilities

### New Capabilities

- `bml-domain`: Entidades matemáticas del núcleo BML — operador base, gramática del AST, estructuras SoA de Nodo con `#[repr(align(64))]`, y el `BMLTransformer` que traduce operaciones estándar (suma, resta, multiplicación, división, potencia) a la gramática recursiva de BML usando solo el operador y la constante 1. Cero dependencias.
- `bml-parser`: Ingesta de archivos GGUF mediante mapeo directo a memoria (`memmap2`) sin copias a RAM. Decodifica cabeceras mágicas, metadatos y referencias a tensores.
- `bml-compiler`: Transforma tensores parseados en un DAG estático. Aplica **Hash Consing** para deduplicar sub-árboles BML idénticos, linealiza el DAG en RPN, y aplica micro-fragmentación AOT para que el binario exportado (`.bmlgraph`) caiga bajo el umbral de caché objetivo (32 KB para L1).
- `bml-runtime`: Ejecuta el grafo linealizado en RPN. Inicializa buffers una sola vez al arrancar (cero allocs en runtime), expone el *hot loop* confinado a < 32 KB de instrucciones, y provee la interfaz RPC/binaria para transmisión append-only de fragmentos entre nodos distribuidos.

### Modified Capabilities

_(Ninguna — el repo arranca desde el draft, no hay specs previas en `openspec/specs/`.)_

## Impact

- **Repo:** se crea el workspace Cargo raíz con perfil release extremo (`lto = "fat"`, `panic = "abort"`, `codegen-units = 1`) y los 4 crates miembros.
- **Dependencias:** `memmap2` (parser), `criterion` (benchmarks, dev-dependency), `perf` (tooling externo, no crate), posiblemente `tonic`/`gRPC` para el runtime distribuido (Hito 5).
- **Tests:** se introducen pruebas unitarias puras del operador, benchmarks comparativos FMA vs BML, pruebas de estrés multicore con `perf`, y una prueba distribuida con `n` trabajadores leyendo `/dev/shm`.
- **Documentación:** `draft.md` queda como fuente de verdad histórica; las specs vivas bajo `openspec/specs/` son el contrato funcional actual.
- **No se rompe nada existente:** el único contenido previo del repo es `draft.md` y `references/`, que se conservan intactos.
