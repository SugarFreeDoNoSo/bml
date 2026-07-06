## 1. Hito 1 — Cimientos de Memoria, Dominio y Transformación Matemática

- [ ] 1.1 Crear `Cargo.toml` raíz del workspace con los 4 miembros (`bml-domain`, `bml-parser`, `bml-compiler`, `bml-runtime`) y perfil release extremo: `lto = "fat"`, `panic = "abort"`, `codegen-units = 1`, `opt-level = 3`.
- [ ] 1.2 Scaffold de los 4 crates con `cargo new --lib` y verificar `cargo build --release` del workspace vacío.
- [ ] 1.3 En `bml-domain`, definir el operador BML como función pura `fn bml(a: f64, b: f64) -> f64` según la definición del draft (referencia ArXiv 2603.21852v2 adaptada).
- [ ] 1.4 Documentar la propiedad algebraica (magma no asociativo, orden de operandos inmutable) como doc-comments y en `openspec/specs/bml-domain/spec.md`.
- [ ] 1.5 Definir la gramática del AST en `bml-domain`: un nodo es `BML(left, right)` o `Const(1)`, nada más.
- [ ] 1.6 Definir la estructura `Node` en layout SoA con `#[repr(align(64))]` para los campos del grafo (operandos, resultado, flags).
- [ ] 1.7 Escribir pruebas unitarias puras del operador BML: conmutatividad NO (orden importa), casos límite (0, 1, NaN, Inf), y verificación de completitud funcional.
- [ ] 1.8 Implementar el `BMLTransformer` (mapper) que traduce `+`, `-`, `*`, `/`, `pow` a la gramática recursiva BML usando solo el operador y la constante 1.
- [ ] 1.9 Pruebas del `BMLTransformer`: para cada operación estándar, verificar que la evaluación del AST BML resultante coincide con la operación original en un rango de valores (usar `proptest`).
- [ ] 1.10 Validar hito: `cargo test -p bml-domain` pasa y `cargo build --release` del workspace sin warnings.

## 2. Hito 2 — Compilación, Deduplicación y MVP de Rendimiento

- [ ] 2.1 En `bml-compiler`, definir el tipo `Dag` y el `HashConsRegistry` (registro global de sub-árboles BML).
- [ ] 2.2 Implementar Hash Consing: hash estructural de sub-árboles canónicos, deduplicación en tiempo de compilación.
- [ ] 2.3 Pruebas de propiedad del Hash Consing con `proptest`: sub-árboles estructuralmente idénticos se deduplican; sub-árboles distintos no colisionan.
- [ ] 2.4 Implementar el transformador DAG → RPN: linealiza el DAG deduplicado en un arreglo unidimensional en Notación Polaca Inversa.
- [ ] 2.5 Pruebas del linealizador: la evaluación de la RPN coincide con la evaluación del DAG original.
- [ ] 2.6 Añadir `criterion` como dev-dependency del workspace.
- [ ] 2.7 Construir benchmark comparativo: fórmula compleja en FMA tradicional vs. DAG BML deduplicado. Medir tiempo de ejecución.
- [ ] 2.8 Documentar en el reporte del benchmark la reducción a tiempo sub-lineal `O(n^k)` con `k < 1` para operaciones repetidas gracias al Hash Consing.
- [ ] 2.9 Prueba de estrés multicore (escalado vertical): generar DAGs por debajo y por encima de 32 KB y medir latencia.
- [ ] 2.10 Medir cache hit/miss en L1/L2 con `perf stat -e L1-dcache-load-misses,L1-icache-load-misses` sobre la prueba de estrés.
- [ ] 2.11 Prueba distribuida (escalado horizontal): `n` trabajadores leen un bloque de memoria compartida en `/dev/shm`, ejecutan su porción del DAG BML y escriben la salida lock-free.
- [ ] 2.12 Medir latencia de transferencia en la prueba distribuida y documentar resultados.
- [ ] 2.13 Validar hito: benchmarks corren, reporte de `perf` adjunto, prueba de `/dev/shm` pasa sin data races (`loom` o Miri donde aplique).

## 3. Hito 3 — El Ingestor (Parser GGUF Zero-Copy)

- [ ] 3.1 Añadir `memmap2` como dependencia de `bml-parser`.
- [ ] 3.2 Implementar la decodificación de la cabecera mágica GGUF y de los metadatos.
- [ ] 3.3 Implementar el mapeo zero-copy de los tensores desde el disco al espacio de memoria de Rust vía mmap, encapsulado en un RAII guard con lifetime explícito.
- [ ] 3.4 Pruebas del parser: usar un fixture GGUF pequeño (puede generarse sintéticamente) y verificar que los tensores mapeados coinciden con los bytes del archivo.
- [ ] 3.5 Pruebas de lifetime: verificar que un tensor mapeado no es accesible tras cerrar el guard (debe fallar en compile-time por el lifetime).
- [ ] 3.6 Validar hito: `cargo test -p bml-parser` pasa y se confirma con `strace` que no hay `read` de los tensores a buffers userspace (solo `mmap`).

## 4. Hito 4 — Micro-Fragmentación y L1 Routing

- [ ] 4.1 Añadir al `bml-compiler` la lógica de partición AOT (Tensor Parallelism AOT) del DAG.
- [ ] 4.2 Definir el formato binario `.bmlgraph` (header + fragmentos) y su serialización.
- [ ] 4.3 Implementar el cálculo del tamaño de cada fragmento y la garantía de que no supera el umbral de caché objetivo (32 KB L1 por defecto, configurable a L3).
- [ ] 4.4 Pruebas de fragmentación: para DAGs de distintos tamaños, verificar que cada fragmento exportado pesa ≤ 32 KB (o el umbral configurado).
- [ ] 4.5 Pruebas de routing: verificar que el orden de fragmentos preserva la semántica del DAG original al reconstruirse y ejecutarse.
- [ ] 4.6 Validar hito: `cargo test -p bml-compiler` pasa y un DAG grande se exporta como N fragmentos todos ≤ 32 KB.

## 5. Hito 5 — El Motor L1 (Runtime)

- [ ] 5.1 En `bml-runtime`, implementar el inicializador de buffers: reserva toda la memoria necesaria una sola vez al arrancar (cero allocs en hot path).
- [ ] 5.2 Implementar el intérprete del flujo RPN (el *hot loop*): iteración secuencial sobre el arreglo RPN sin saltos ni recursión.
- [ ] 5.3 Medir el tamaño del hot loop con `cargo asm` y `perf stat -e instructions`; verificar que es < 32 KB. Si excede, refactorizar a tablas de saltos compactas.
- [ ] 5.4 Pruebas del runtime: ejecutar un `.bmlgraph` de prueba y verificar que el resultado coincide con la evaluación de referencia del DAG.
- [ ] 5.5 Implementar el patrón append-only: cada evaluación escribe `v_o` a una dirección pre-asignada nueva, nunca sobrescribe.
- [ ] 5.6 Pruebas de append-only con `loom` para verificar ausencia de data races bajo estrés multicore.
- [ ] 5.7 Implementar la interfaz RPC/binaria (gRPC vía `tonic` u otra) para recibir y transmitir fragmentos `.bmlgraph` entre nodos.
- [ ] 5.8 Aislar el código RPC en un módulo separado del hot loop para no impactar D5.
- [ ] 5.9 Pruebas de integración del RPC: un nodo envía un fragmento, otro lo recibe, lo ejecuta y devuelve el resultado.
- [ ] 5.10 Pruebas de estrés finales en entorno cloud (ej. GCP): medir throughput y latencia p99 del runtime distribuido.
- [ ] 5.11 Validar hito: runtime ejecuta `.bmlgraph` correctamente, hot loop < 32 KB confirmado, RPC funcional, y reporte de cloud adjunto.

## 6. Cierre del Change

- [ ] 6.1 `openspec validate bml-implementation-plan` pasa sin errores.
- [ ] 6.2 `openspec archive bml-implementation-plan` para mover las specs delta a `openspec/specs/` principales.
- [ ] 6.3 Actualizar `draft.md` con una nota apuntando a `openspec/specs/` como contrato funcional actual.
