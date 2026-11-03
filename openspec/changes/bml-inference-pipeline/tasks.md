## 8. Scheduler con batching dinámico

- [x] 8.1 Implementar `crates/runtime/src/scheduler.rs` con cola de requests (`crossbeam-channel`). `scheduler::RequestBatcher` usa el canal acotado reexportado por crossbeam; productores clonables, admisión no bloqueante y devolución del request en saturación/cierre. Integración HTTP implementada en 8.4.
- [x] 8.2 Implementar batching dinámico: agrupa N prompts en una ventana de 10ms. Ventana fija desde el primer request recibido, máximo de lote configurable y drenaje parcial al cerrar productores; pruebas en `scheduler::requests::tests`.
- [x] 8.3 Implementar distribución de batches a nodos: round-robin. Repetir `--layer-workers` configura réplicas completas; `RequestBatcher` selecciona `batch.node` y todas sus solicitudes se ejecutan en esa réplica. Prueba real de dos workers: JSON en réplica 0 y SSE en réplica 1, ambos iguales a generación local.
- [x] 8.4 Implementar backpressure: si la cola está llena, retornar HTTP 429. El handler usa `RequestSubmitter`; canal lleno devuelve 429, cerrado 503. El guard RAII cuenta solicitudes en cola/en ejecución y libera en rechazo, cancelación o finalización.
- [x] 8.5 Pruebas de batching: diez handlers concurrentes preadmitidos se procesan en un batch; `scheduled_requests_finish_and_release_admission` comprueba el contador y liberación de admisión. La preadmisión evita suponer plazos de scheduling del SO; la ventana fija de 10 ms se prueba por separado en runtime. No es una medición de throughput ni latencia bajo carga.

El batcher tiene pruebas de diez productores concurrentes sin pérdidas, cola llena,
FIFO, cierre, ventana vencida y asignación round-robin. La prueba concurrente
encola antes de drenar para no depender del tiempo de planificación del SO;
la integración HTTP tiene pruebas del handler pero no una campaña de throughput. La selección de nodo
se entrega como índice al consumidor. El servidor ejecuta batches en un hilo local, con forward por solicitud; el modo `--layer-workers` distribuye capas por pipeline TCP con identidad y KV,
con reparto round-robin entre réplicas completas. Falta ejecución numérica conjunta. No se afirma aceleración por batching.

### Obligaciones del compilador pendientes de acreditar

- [ ] 8.6 Completar la traducción real de cada operación del transformer a sub-DAGs BML mediante `BMLTransformer`, como exige `specs/gguf-compiler/spec.md`. C59–C72 acreditan las primitivas, operaciones, pesos externos, ejecución causal, artefactos AOT y reparación de los entrypoints GGUF. C75–C78 añaden RMS acotado v3 y evaluación real finita por nodo. C79–C84 añaden el perfil completo v4: nueve familias con comprobación de cada BML, dots indexados con código de tamaño constante, máximo auxiliar explícito en softmax, artefactos y CLI compatibles. Stories15M completó dos tokens desde un AOT v4 publicado y recargado sin construcción: 32.000 logits finitos por token, argmax 9038/2501 igual a la ruta nativa, errores absolutos máximos 1.620e-8/2.945e-6. Evidencia: `quality/evidence/full-bounded-2026-11-03.json`.

  Avance compacto: C88–C94 conservan los bits de las fórmulas anteriores con stages y loops RMS/softmax, código y scratch independientes del ancho, y firmas `indexed_buffer` con 17/10 literales. El plan mantiene todos los prefijos; para contextos 2/4/8/16, instrucciones, literales y bytes de programas crecen linealmente. V4 `buffer` histórico sigue cargando, y v1–v3 rechazan la representación nueva. Permanecen los límites de programas, pasos, nodos, valores y archivos: esta mejora no promete contexto 2048 con presupuestos predeterminados. Evidencia: `quality/evidence/compact-bounded-2026-11-03.md`.

  Continúan pendientes la validación v4 en modelos mayores (incluido TinyLlama) y prompts adicionales. El éxito medido de Stories15M no garantiza evaluabilidad universal, equivalencia con llama.cpp ni rendimiento competitivo. Los fallos de perfiles anteriores se conservan en `quality/evidence/compiler-entrypoint-repair-2026-11-03.md` y `quality/evidence/bounded-rms-2026-11-03.json`.
- [x] 8.7 Acreditar pesos GGUF por referencia mmap y generación real desde `.bmlgraph`. C56/C57 verifican programas persistidos y metadatos/bindings; TinyLlama compilado completó diez tokens sin copiar pesos y con la misma salida textual que el nativo mapeado. Evidencia `quality/evidence/tinyllama-compiled-causal.json`; no acredita la traducción BML pendiente en 8.6 ni aceleración.

## 9. Pruebas de concurrencia

Diagnóstico adicional de 8.6: el caso de una sola componente `x=1`, `weight=4`,
`epsilon=1e-5` reproduce un desbordamiento interno aunque la referencia vale
`3.999980000149999`. La primera salida inválida de Stories15M BOS es la 167.
Fuentes, hashes y operaciones exactas en
`quality/evidence/symbolic-rms-range-diagnostic-2026-11-03.md`.

Auditoría del transporte: el worker cerraba la conexión tras un mensaje, aunque
`NodeHandle` reutiliza TCP. Ahora procesa una sesión completa hasta desconexión;
una prueba con sockets locales ejecuta health, steal vacío, dos matmuls vectoriales
con oráculo `[11,16]` y nuevos health sobre la misma conexión. El cliente rechaza
tipos de respuesta incorrectos en health/steal. `cargo test -p bml-worker -p
bml-runtime --locked` pasa. Esto no completa inferencia distribuida:
`ExecuteFragment` sigue cargando pesos y `execute_layers` sigue siendo parcial.
El decodificador de esos fragmentos comprueba todos los accesos, UTF-8, conteos
antes de reservar, rangos de capas/tensores y bytes sobrantes. Sus pruebas
recorren cada truncamiento del fixture y longitudes maliciosas; las tres pruebas
del worker pasan con Clippy sin avisos nuevos. `StealWork` ahora retira de una cola acotada `PendingWork` (ArrayQueue de
crossbeam) independiente del modelo cargado. `EnqueueFragment`/`EnqueueResult`
permiten admisión explícita con respuesta de saturación o formato inválido. La
prueba TCP con cuatro ladrones obtiene exactamente los ocho IDs admitidos y
conserva el modelo cargado. No acredita Loom, ausencia universal de starvation
ni entrega exactamente una vez ante fallos de red: un fallo tras pop puede
perder el trabajo y un reintento de admisión ambiguo puede duplicarlo. Faltan
recuperación/acknowledgement y la integración de consumo local con inferencia.
HealthCheck devuelve un byte de estado y un conteo `u64` little-endian de
pendientes. `NodeHandle::health_status` admite también el frame histórico de un
byte (conteo desconocido, no cero inventado) y rechaza formatos inválidos. La
prueba TCP observa 0, 8 y 0 pendientes antes de admisión, después de admisión y
tras robo concurrente. Son snapshots, no reservas ni garantías de estabilidad.

Corrección verificada durante la auditoría: `WaveScheduler` omitía fragmentos
cuando una wave superaba el número de workers. Ahora distribuye la wave completa
con cola final, elimina el warmup con buffer vacío y ordena los IDs de cada wave.
`wide_wave_executes_every_fragment_before_dependents` comprueba siete productores
y un consumidor posterior con 0, 1, 2, 3, 7 y 12 workers. Pasan las seis pruebas
del scheduler y el ratchet Clippy sin avisos nuevos. El mutex del buffer todavía
serializa el cálculo: esto no acredita aceleración paralela ni sustituye Loom.

- [x] 9.1 Pruebas con `loom` de la cola lock-free. `PendingWork` usa `bml-queue`, adaptación trazable de ArrayQueue 0.3.13: el mismo archivo del algoritmo se compila con primitivas nativas o Loom (incluidos UnsafeCell y backoff). Seis modelos pasan y la prueba sin sincronización por join detecta la relajación de Release/Acquire en copias aisladas. La API nativa funciona con todas las features y runtime conserva deny(unsafe_code). Evidencia `quality/evidence/pending-work-loom.json`; son escenarios pequeños, no prueba general de liveness ni de WorkQueue/deque (9.2).
- [x] 9.2 Pruebas con `loom` del work-stealing. Integrado en `a88be04`: FIFO SegQueue real instrumentada, cinco modelos a 3/31 slots, contabilidad de Drop/reservas/liberaciones y control negativo de retirada prematura detectado antes de liberar memoria. Gate aislado: 631 Rust, 63 Python, cero discrepancias y diagnósticos nuevos. Evidencia `quality/evidence/delivery-segmented-work.json`. Cierre de pruebas acotadas, no prueba general de liveness ni de validez de todos los punteros crudos. Los estados abiertos que siguen son historial anterior a esta integración.
  Avance local: copia trazable del código real con índices/fences Loom, 3/3 modelos FIFO de arbitraje aprobados y 3/3 controles negativos de CAS detectados. Slots/epoch siguen nativos; no acredita publicación, reutilización ni resize/reclamación. Criterios de cierre en `WORK_STEALING_LOOM.md`; evidencia `quality/evidence/work-stealing-loom-indices.json`. 9.2 sigue abierta.
  Avance posterior: slots instrumentados con UnsafeCell Loom; publicación y reutilización secuencial pasan, pero la reutilización con ladrón obsoleto falla por lectura/escritura concurrente (capacidad 2; reproducido también a capacidad 64, con aborto secundario durante limpieza). Controles negativos Release y Acquire detectados. Evidencia `quality/evidence/work-stealing-loom-slots.json`; hay que resolver el acceso especulativo antes de acreditar reutilización y después verificar resize/reclamación. 9.2 permanece abierta.
  Reemplazo verificado: SegQueue real con slots/atómicos/backoff instrumentados pasa 3/3 modelos a 3 y 31 slots por bloque; controles negativos Release y Acquire fallan como se esperaba. Runtime nativo: 82 pruebas; gate del árbol actual: 704 pruebas. Evidencia `quality/evidence/work-stealing-segmented-replacement.json`. Pendientes: ampliar destrucción exactamente una vez/reclamación y sus controles negativos, e integrar los cambios revisados; no se cierra 9.2.
  Destrucción ampliada: 5/5 modelos a ambos tamaños, incluidos Drop exactamente una vez al cruzar segmentos con consumidores y cola abandonada, y supervivencia del valor extraído tras destruir la cola. Omitir Drop de pendientes falla como se esperaba (`work-stealing-segmented-drop.json`). Pendientes: contabilidad y controles negativos de liberación de segmentos e integración aislada; 9.2 permanece abierta.
  Contabilidad posterior: 5/5 modelos con una liberación por reserva a ambos tamaños; cuatro controles negativos de liberación omitida detectados. Snapshot aislado sin PendingWork: 63 pruebas runtime all-features aprobadas (`work-stealing-segmented-reclamation.json`). Pendientes revisión de reclamación prematura y gate/commit aislado; 9.2 sigue abierta.
- [x] 9.3 Pruebas de append-only bajo estrés multicore. `AppendOnlyResults` recibe 4.000 evaluaciones desde cuatro runtimes/hilos con índices únicos, valores estables y error al llenarse. Dos pruebas Loom exploran publicación y saturación sobre los atómicos reales; `Runtime::execute` conserva el ring histórico y no acredita este contrato.
- [x] 9.4 Pruebas de que el hot loop no toca el código de red (aislamiento). Auditoría del dispatcher dentro del worker release con red enlazada en `tools/audit_hot_loop_binary.py`, resolviendo llamadas GOT y tablas de salto, más pruebas negativas que rechazan llamadas RPC/socket/syscall y destinos no resueltos. `quality/evidence/worker-hot-loop-isolation.json` conserva la medición histórica de 1590 bytes. La revisión actual con contexto indexado y vistas de buffer registra 2309 bytes de dispatcher, 18 llamadas resueltas y un helper local `trunc` de 74 bytes auditado sin llamadas; evidencia `quality/evidence/operation-schedule-hot-loop.json`. No afirma aislamiento transitivo de hooks externos ni comportamiento L1.
- [x] 9.5 Pruebas de backpressure: HTTP 429. `cargo test -p bml-server --locked` pasa tres pruebas del handler: cola llena conserva el request admitido y devuelve error JSON/429, cierre devuelve 503 sin fugas, diez requests concluyen mediante el scheduler. No requieren descargar modelos.
- [ ] 9.6 Pruebas de cambio de hot loop: L1i no se contamina entre fragmentos. (Requiere PMU/hardware counters.)
- [x] 9.7 Corregir el scheduler público de `OperationFragment`: análisis conservador de bytecode/layout, dependencias RAW/WAR/WAW y ejecución concurrente mediante slices disjuntos, sin copiar pesos o activaciones. C87 y el rendezvous interno con timeout verifican resultados y entrada simultánea; el plan puede asignar memoria. Evidencia `quality/evidence/operation-fragment-repair-2026-11-03.md`. Esta corrección no cambia el mutex del `WaveScheduler` histórico ni acredita aceleración o residencia L1.

Diagnóstico del entorno del 2026-11-03 registrado en
`quality/evidence/spec-environment-blockers-2026-11-03.json`: VM x86_64,
`perf` ausente, `perf_event_paranoid=3`, sin dispositivo CPU PMU en sysfs.
9.6 necesita un host con esos contadores accesibles; 8.6 sigue siendo trabajo
local de implementación/validación y no se atribuye a este bloqueo.

## 10. Cierre

- [x] 10.1 `openspec validate bml-inference-pipeline --strict` pasa sin errores. Validación estructural; no sustituye requisitos pendientes de ejecución/concurrencia.
- [x] 10.2 `cargo test --workspace` pasa. Gate completo actualizado en `target/quality/pipeline-development-gate.log`: pruebas de workspace, auditoría histórica con cero discrepancias, traza 39/39 y Clippy sin diagnósticos nuevos.
- [x] 10.3 Commit y push. Implementación y evidencias publicadas en `ca8d030` y `1ef5f5d`; `origin/master` verificado en `1ef5f5d39a1d7ad6b3a0065f98164d1121044d4b` el 2026-11-03. Evidencia: `quality/evidence/publication-spec-closure-2026-11-03.md`. Esta publicación no cierra 8.6 ni 9.6 ni autoriza atribuirles validación.
