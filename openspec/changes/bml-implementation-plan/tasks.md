## 1. Hito 1 — Cimientos de Memoria, Dominio y Transformación Matemática

- [x] 1.1 Crear `Cargo.toml` raíz del workspace con los 4 miembros (`bml-domain`, `bml-parser`, `bml-compiler`, `bml-runtime`) y perfil release extremo: `lto = "fat"`, `panic = "abort"`, `codegen-units = 1`, `opt-level = 3`.
- [x] 1.2 Scaffold de los 4 crates con `cargo new --lib` y verificar `cargo build --release` del workspace vacío.
- [x] 1.3 En `bml-domain`, definir el operador BML como función pura `fn bml(a: f64, b: f64) -> f64` según la definición del draft (referencia ArXiv 2603.21852v2 adaptada a base 2: `bml(x,y) = 2^x - log2(y)`).
- [x] 1.4 Documentar la propiedad algebraica (magma no asociativo, orden de operandos inmutable) como doc-comments y en `openspec/specs/bml-domain/spec.md`.
- [x] 1.5 Definir la gramática del AST en `bml-domain`: un nodo es `BML(left, right)` o `Const(1)`, nada más.
- [x] 1.6 Definir la estructura `Node` en layout SoA con `#[repr(align(64))]` para los campos del grafo (operandos, resultado, flags).
- [x] 1.7 Escribir pruebas unitarias puras del operador BML: conmutatividad NO (orden importa), casos límite (0, 1, NaN, Inf), y verificación de completitud funcional.
- [x] 1.8 Implementar el `BMLTransformer` (mapper) que traduce `+`, `-`, `*`, `/`, `pow` a la gramática recursiva BML usando solo el operador y la constante 1. (Completado: fórmulas derivadas del Supplementary Information del paper ArXiv 2603.21852v2, Sect. 2.5. `sub`, `neg`, `add`, `recip`, `mul`, `div`, `pow` implementadas y verificadas.)
- [x] 1.9 Pruebas del `BMLTransformer`: para cada operación estándar, verificar que la evaluación del AST BML resultante coincide con la operación original en un rango de valores (usar `proptest`). (Completado: tests de `sub`, `neg`, `add`, `recip`, `mul`, `div`, `pow` verificadas.)
- [x] 1.10 Validar hito: `cargo test -p bml-domain` pasa y `cargo build --release` del workspace sin warnings.

## 2. Hito 2 — Compilación, Deduplicación y MVP de Rendimiento

- [x] 2.1 En `bml-compiler`, definir el tipo `Dag` y el `HashConsRegistry` (registro global de sub-árboles BML).
- [x] 2.2 Implementar Hash Consing: hash estructural de sub-árboles canónicos, deduplicación en tiempo de compilación.
- [x] 2.3 Pruebas de propiedad del Hash Consing con `proptest`: sub-árboles estructuralmente idénticos se deduplican; sub-árboles distintos no colisionan.
- [x] 2.4 Implementar el transformador DAG → RPN: linealiza el DAG deduplicado en un arreglo unidimensional en Notación Polaca Inversa.
- [x] 2.5 Pruebas del linealizador: la evaluación de la RPN coincide con la evaluación del DAG original.
- [x] 2.6 Añadir `criterion` como dev-dependency del workspace.
- [x] 2.7 Construir benchmark comparativo: fórmula compleja en FMA tradicional vs. DAG BML deduplicado. Medir tiempo de ejecución.
- [ ] 2.8 Documentar en el reporte del benchmark la reducción a tiempo sub-lineal `O(n^k)` con `k < 1` para operaciones repetidas gracias al Hash Consing. Estado: parcial, reabierta por evidencia insuficiente. La mejora constante histórica no demuestra el exponente; `docs/benchmarks/REPORT.md` declara que la comparación de trabajo equivalente no acredita ventajas asintóticas. Falta definir n y demostrar el criterio sobre una familia de repetición estructural con trabajo comparable. No se cambia la aceptación.
- [x] 2.9 Prueba de estrés multicore (escalado vertical): generar DAGs por debajo y por encima de 32 KB y medir latencia.
- [ ] 2.10 Medir cache hit/miss en L1/L2 con `perf stat -e L1-dcache-load-misses,L1-icache-load-misses` sobre la prueba de estrés. Estado: bloqueada por evidencia PMU, reabierta. `tests/stress/perf_cache.rs` pide eventos genéricos de caché, no demuestra L1/L2 ni un fallback software. Los dos eventos citados son de L1: también faltan los eventos/denominadores específicos de L2 y de hits, con carga, host, intervalo y soporte identificados. Reutilizar el acceso PMU requerido por 5.3 y bml-inference-pipeline/9.6, conservando sus cargas y métricas distintas; no crear otra obligación equivalente.
- [x] 2.11 Prueba distribuida (escalado horizontal): `n` trabajadores leen un bloque de memoria compartida en `/dev/shm`, ejecutan su porción del DAG BML y escriben la salida lock-free.
- [x] 2.12 Medir latencia de transferencia en la prueba distribuida y documentar resultados.
- [ ] 2.13 Validar hito: benchmarks corren, reporte de `perf` adjunto, prueba de `/dev/shm` pasa sin data races (`loom` o Miri donde aplique). Estado: parcial, reabierta; falta el informe de 2.10 para este cierre agregado. Conservar benchmarks y prueba funcional `/dev/shm`; los modelos Loom de append-only/cola no instrumentan mmap ni esa prueba. `quality/evidence/runtime-cache-probe-functional.json` acredita sólo funcionalidad. No se exige una segunda campaña PMU para cerrar la misma dependencia.

## 3. Hito 3 — El Ingestor (Parser GGUF Zero-Copy)

- [x] 3.1 Añadir `memmap2` como dependencia de `bml-parser`.
- [x] 3.2 Implementar la decodificación de la cabecera mágica GGUF y de los metadatos.
- [x] 3.3 Implementar el mapeo zero-copy de los tensores desde el disco al espacio de memoria de Rust vía mmap, encapsulado en un RAII guard con lifetime explícito.
- [x] 3.4 Pruebas del parser: usar un fixture GGUF pequeño (puede generarse sintéticamente) y verificar que los tensores mapeados coinciden con los bytes del archivo.
- [x] 3.5 Pruebas de lifetime: verificar que un tensor mapeado no es accesible tras cerrar el guard (debe fallar en compile-time por el lifetime). (Verificado: el `MmapFile` es RAII, los slices se invalidan al hacer drop.)
- [x] 3.6 Validar hito: `cargo test -p bml-parser` pasa y se confirma con `strace` que no hay `read` de los tensores a buffers userspace (solo `mmap`).

## 4. Hito 4 — Micro-Fragmentación y L1 Routing

- [x] 4.1 Añadir al `bml-compiler` la lógica de partición AOT (Tensor Parallelism AOT) del DAG.
- [x] 4.2 Definir el formato binario `.bmlgraph` (header + fragmentos) y su serialización.
- [x] 4.3 Implementar el cálculo del tamaño de cada fragmento y la garantía de que no supera el umbral de caché objetivo (32 KB L1 por defecto, configurable a L3).
- [x] 4.4 Pruebas de fragmentación: para DAGs de distintos tamaños, verificar que cada fragmento exportado pesa ≤ 32 KB (o el umbral configurado).
- [x] 4.5 Pruebas de routing: verificar que el orden de fragmentos preserva la semántica del DAG original al reconstruirse y ejecutarse.
- [x] 4.6 Validar hito: `cargo test -p bml-compiler` pasa y un DAG grande se exporta como N fragmentos todos ≤ 32 KB.

## 5. Hito 5 — El Motor L1 (Runtime)

- [x] 5.1 En `bml-runtime`, implementar el inicializador de buffers: reserva toda la memoria necesaria una sola vez al arrancar (cero asignaciones en tiempo de ejecución).
- [x] 5.2 Implementar el intérprete del flujo RPN (*The Hot Loop*): iteración secuencial sobre el arreglo RPN sin saltos ni recursión.
- [ ] 5.3 Medir el tamaño del código máquina del hot loop y su comportamiento L1. La antigua justificación por tamaño de `.rlib` y tres variantes RPN no prueba este requisito y no describe el código actual. Requiere medición de símbolos/disassembly; los contadores PMU son evidencia separada. La auditoría actual de símbolos/disassembly del worker release con FMax mide 2533 bytes del dispatcher y verifica un helper local de 74 bytes; `quality/evidence/rms-context-hot-loop.json` conserva binario, toolchain, comandos y hashes; las mediciones anteriores se conservan en `quality/evidence/distributed-symbolic-hot-loop.json` y `quality/evidence/operation-layer-hot-loop-2026-11-03.json`. La medición anterior de 2309 bytes sigue conservada como evidencia histórica. La parte de comportamiento L1 sigue pendiente: el tamaño por sí solo no demuestra residencia ni ausencia de misses. Estado: parcial y bloqueada por medición PMU; tamaño estático publicado, comportamiento L1 pendiente. Véase docs/OPENSPEC_RECONCILIATION.md.

  Medición estática implementada en `tools/measure_hot_loop.py`: ejecutable
  release enlazado, símbolo `dispatch_ops` de 1411 bytes en x86_64. Evidencia con
  hashes y límites en `quality/evidence/bml-hot-loop-code-size.json`. Queda
  pendiente alcance transitivo/inline, aislamiento binario y mediciones L1/PMU;
  el tamaño del símbolo no prueba ausencia de expulsiones de caché.
- [x] 5.4 Pruebas del runtime: ejecutar un `.bmlgraph` de prueba y verificar que el resultado coincide con la evaluación de referencia del DAG.
- [x] 5.5 Implementar el patrón append-only: cada evaluación escribe `v_o` a una dirección pre-asignada nueva, nunca sobrescribe.
- [x] 5.6 Pruebas de append-only con `loom`. `append_only::loom_tests` instrumenta los mismos atómicos del algoritmo de producción mediante `cfg(test)` y feature `loom`: publicación/lectura concurrente vía `Runtime::execute_append_only` y carrera de saturación. Pasan ambas pruebas; alcance acotado a esas ejecuciones, sin extrapolar a Crossbeam o memoria compartida mmap.
- [x] 5.7 Implementar la interfaz RPC/binaria. `ExecuteGraph`/`GraphResult` ejecutan `.bmlgraph` validado con inputs y buffer; dos workers intercambian resultados y coinciden con el DAG local. `NodeHandle::execute_graph_append_only` publica escalar y buffer en rangos nuevos de `AppendOnlyResults`; si no hay capacidad devuelve el resultado completo para reintentar solo su publicación local. Cuatro pruebas Loom instrumentan las reservas/publicaciones reales. Las capas causales con sesiones/KV usan `bml-inference::remote`. La carga legacy `ExecuteFragment` sigue siendo un ack explícito. Evidencia: `quality/evidence/graph-rpc-publication.json`.
- [x] 5.8 Aislar RPC del hot loop. `tools/audit_hot_loop_binary.py` inspecciona el worker x86-64 release con RPC enlazado: dispatcher de 1590 bytes, 13 llamadas resueltas y dos saltos por tabla con todos sus destinos internos. Las únicas llamadas son recursión, exp2/log2 y rutas de crecimiento/pánico; ninguna RPC/socket/syscall en el dispatcher. Falla ante control no resuelto o destinos nuevos. Evidencia: `quality/evidence/worker-hot-loop-isolation.json`. No verifica transitivamente hooks de pánico/allocators externos ni afirma residencia L1.
- [x] 5.9 Pruebas de integración RPC. Pasan sesiones TCP persistentes, matmul vectorial, admisión y robo concurrente de pendientes; `graph_rpc_two_worker_stages_match_complete_local_graph` contrasta dos workers TCP y el DAG completo local con tres inputs y buffers. `tcp_layer_pipeline_matches_local_logits_and_preserves_causal_order` y los tests del coordinador contrastan las capas causales remotas. Evidencia: `quality/evidence/graph-rpc.json`. La evidencia histórica `quality/evidence/worker-batch-protocol.json` conserva la antigua aceptación de vectores mediante `BatchRequest`; esa inferencia parcial RMS/Q está ahora retirada con rechazo explícito y migración a `--symbolic-artifact`/`--prepared`. La admisión/ACK y health siguen disponibles. La carga legacy ahora restaura config desde header y valida formas/rangos antes de aceptar shards: C68, trece pruebas del worker y evidencia `quality/evidence/legacy-shard-loading.json`.
- [ ] 5.10 Pruebas de estrés finales en entorno cloud. (**Deferred**: requires cloud environment and bare metal PMU. Current environment lacks both.) Dependencia: host autorizado con PMU accesible y ejecución de estrés autorizada; se conserva la postergación existente.
- [ ] 5.11 Validar hito completo: ejecución `.bmlgraph`, tamaño real del hot loop, RPC y pruebas cloud. Las pruebas actuales no acreditan todos estos contratos. Estado: parcial; cierre agregado dependiente de 5.3 y 5.10, sin duplicar sus obligaciones.

- [x] 5.12 Eliminar asignaciones durante `Runtime::execute_sub_fragments` con pila y buffers preasignados. La ruta usa un slice prestado compartido con `HotLoop::execute_fragment_full`; el contrato público comprueba cuatro subfragmentos, bucles anidados, pila, resultados y entradas vacías. Registra cero asignaciones; restaurar `ops.clone()` en una copia aislada provoca 128 asignaciones y hace fallar la prueba. Evidencia: `quality/evidence/subfragment-allocation-contract.json`. Esto no acredita residencia L1i ni throughput.
- [x] 5.13 Definir el protocolo experimental L1i aceptado antes de medir: código máquina menor de 32 KiB, siete repeticiones con afinidad/carga/calentamiento constantes y máximo un miss por mil instrucciones en cada repetición. `tools/validate_l1i_campaign.py` valida los datos y rechaza contadores ausentes, no soportados, no contados o parcialmente multiplexados; sus pruebas no sustituyen una campaña PMU. El driver funcional se prepara en 5.14; la medición y su procedencia siguen pendientes en 5.3 y 5.10, reservadas a la nueva VM.
- [x] 5.14 Preparar un driver reproducible de `Runtime::execute_sub_fragments` con cuatro fragmentos, bucles anidados, pila y buffers preasignados, oráculo literal y barreras READY/go/FINISHED/stop para operar contadores externos después del calentamiento. `subfragment_cache_probe --verify` y la prueba funcional de barreras pasan; se conserva identidad del binario y fuentes en `quality/evidence/runtime-cache-probe-functional.json`. El intervalo incluye sincronización, comprobaciones e I/O mínimo según `docs/QUALITY.md`; no acredita contadores PMU, residencia L1i ni throughput.

## 6. Cierre del Change

- [x] 6.1 `cargo test --workspace` pasa (bml-domain, bml-parser, bml-compiler, bml-runtime).
- [x] 6.2 `openspec validate bml-implementation-plan --strict` pasa. La validación estructural no mueve deltas ni acredita implementación; el archivo queda pendiente hasta completar contratos.
- [x] 6.3 Actualizar `draft.md` con una nota apuntando a `openspec/specs/` como contrato funcional actual, distinguiendo diseño histórico, cambios activos y evidencia.
