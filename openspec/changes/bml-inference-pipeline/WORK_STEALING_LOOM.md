# 9.2: Loom del work-stealing real

## Estado actual

Pruebas acotadas e integración completadas en `a88be04`. El gate del snapshot
aislado pasa 631 pruebas Rust y 63 Python, sin discrepancias históricas ni
diagnósticos nuevos. La aserción previa a liberación exige READ en todos los
slots no finales; el control que elude diferir la retirada falla antes de
liberar memoria en ambos tamaños. Evidencia: `quality/evidence/delivery-segmented-work.json`.
9.2 se marca completada para este alcance de pruebas. No se afirma verificación
general de punteros crudos ni liveness. Las secciones siguientes conservan el
historial y sus estados pendientes en cada etapa anterior.

## Implementación y frontera

La implementación anterior de `crates/runtime/src/queue.rs` construía `crossbeam::deque::Worker::new_fifo()`.
El lockfile fija crossbeam-deque 0.8.7, epoch 0.9.20 y utils 0.8.22. El deque
importa índices/fences desde `std::sync::atomic`; activar `crossbeam_loom` en
epoch/utils no instrumenta esos índices. `PendingWork`/ArrayQueue es otro
algoritmo y sus modelos no cierran esta tarea.

## Primer modelo ejecutado

`tools/probe_work_stealing_loom.py` prepara una copia del **archivo upstream
completo**, comprobando primero su SHA-256. Cambia la importación de atómicos y
fences por Loom y adapta seis accesos exclusivos de destructores de `get_mut`
a `with_mut`. No reemplaza `push`, `pop` ni `steal` por una simulación. Conserva
licencias y emite el diff íntegro y la procedencia de la copia ejecutada.

Tres modelos exploran conservación y unicidad de propiedad con FIFO prellenado:
un elemento/propietario/un ladrón, un elemento/propietario/dos ladrones y dos
elementos/propietario/dos ladrones. Se permite `Retry`, se recogen resultados de
los hilos y se drena el propietario antes de comparar el multiconjunto exacto.
Límite de preempciones: 2; máximo de ramas por ejecución: 10.000. No se limita
el número de permutaciones dentro de ese alcance.

Resultado: **3/3 pasan**. Un control negativo altera únicamente el CAS de
`Stealer::steal`, manteniendo `front` en lugar de avanzarlo: **3/3 fallan** por
propiedad duplicada (`[0,0]` o `[0,1,1]`). Es sensibilidad al arbitraje, no un
score de mutación completo.

## Reproducción offline

Con el toolchain fijado y las dependencias en el caché Cargo:

```sh
python3 tools/probe_work_stealing_loom.py \
  --source "$HOME/.cargo/registry/src/REGISTRO/crossbeam-deque-0.8.7" \
  --output target/quality/work-stealing-NUEVO
cargo generate-lockfile --offline --manifest-path target/quality/work-stealing-NUEVO/Cargo.toml
CARGO_PROFILE_TEST_DEBUG=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  cargo test --offline --locked --lib \
  --manifest-path target/quality/work-stealing-NUEVO/Cargo.toml
```

Repetir en otro directorio nuevo con `--negative-control`; el test debe terminar
con código 101 y mostrar duplicados. La generación del lockfile es preparación;
la verificación usa `--locked` y versiones directas idénticas a las del proyecto.

## Criterios para cerrar 9.2

1. Mantener trazabilidad fuente→adaptación y ejecutar las operaciones reales que
   utiliza `WorkQueue`, no un algoritmo de cola distinto.
2. Instrumentar slots y publicación además de índices/fences; comprobar push
   concurrente con robo, lecturas de datos publicados y reutilización circular.
3. Instrumentar la protección/reclamación del buffer y explorar crecimiento y
   reducción con ladrones activos. Una capacidad reducida de modelado debe
   documentarse y preservar las mismas ramas del algoritmo.
4. Cubrir vacío, último elemento, múltiples ladrones, `Retry`, conservación de
   IDs y descarte exactamente una vez de valores no triviales. Diferenciar la
   transferencia local de propiedad de la entrega por transporte.
5. Exigir controles negativos de arbitraje **y** publicación: una relajación de
   sincronización relevante debe ser detectada por un modelo sensible a datos.
6. Registrar comandos, límites, resultados finales y hashes. Solo después,
   actualizar 9.2 y la documentación de alcance, sin afirmar liveness universal.

**9.2 permanece abierta.** El modelo inicial de índices conserva slots, `Arc`,
epoch y backoff nativos. Evidencia histórica: `quality/evidence/work-stealing-loom-indices.json`.

## Instrumentación posterior de slots: fallo de reutilización

La opción `--slots` mantiene cada payload en un `loom::cell::UnsafeCell` y ejecuta
los accesos volátiles upstream dentro de `with`/`with_mut`. La reserva/liberación
se adapta al contenedor instrumentado; los destructores destruyen el payload
inicializado y la copia en resize mueve sus bits, no el estado de instrumentación.
La protección/reclamación epoch sigue nativa y **no está acreditada**. Se modela
explícitamente el fence SeqCst que `steal` obtiene de `epoch::pin()` o de su rama
reentrante para no omitir esa sincronización en los nuevos modelos.

Con `--slots --slot-capacity 2`, pasan los tres modelos de índices, la publicación
concurrente y la reutilización secuencial: **5 pasan y 1 falla**. El modelo
`stale_thief_competes_with_wrapped_slot_reuse` falla por acceso concurrente
lectura/escritura. Con `--slots` y capacidad upstream **64**, reproduce el mismo
diagnóstico primario; la limpieza añade otro panic y termina en SIGABRT. Ese run
se registra como abortado, no como suite completa aprobada o fallo limpio único.

La disputa no requiere resize: el propietario mantiene como máximo un elemento
vivo, avanza el anillo y sobrescribe un slot que un ladrón puede leer usando un
`front` antiguo. `Stealer::steal` lee el payload antes de validar el CAS de
`front`; el posterior `Retry` no borra el acceso especulativo ya ocurrido.
Los comentarios upstream de `Buffer::read/write` ya advierten de esta limitación
de los accesos volátiles. El resultado describe la copia instrumentada, no una
prueba de crash en producción ni una verificación completa de epoch.

Controles negativos de ordenamiento, ejecutados por separado sobre el modelo de
publicación (sin join/flag previo que esconda la sincronización de la cola):

- `--weaken-publication`: retirar el fence Release de push causa violación de
  causalidad; 1 test falla como se esperaba.
- `--weaken-acquire`: relajar la lectura Acquire de back del ladrón también
  causa violación de causalidad; 1 test falla como se esperaba.

Para reproducir, añadir esas opciones a la generación anterior y filtrar con
`--lib slots::concurrent_push_publishes_payload_before_steal -- --exact`.
Los logs y hashes finales están en `quality/evidence/work-stealing-loom-slots.json`.

El siguiente trabajo local es resolver/reemplazar el protocolo de acceso
especulativo a slots y repetir reutilización, antes de extender la instrumentación
a resize/reclamación. Convertir solo los payloads del modelo en atómicos ocultaría
el fallo y no verificaría el algoritmo usado en producción. No hay bloqueo de
hardware para continuar esta investigación.

## Reemplazo local por FIFO segmentada

`WorkQueue` ahora comparte `crossbeam::queue::SegQueue<Fragment>` mediante `Arc`
con `WorkStealer`. Conserva FIFO y push sin capacidad fija; el handle público
deja de ser el tipo específico de crossbeam-deque. `steal()` entrega Success o
Empty; `steal_from` sigue devolviendo `Option<Fragment>`. El handle puede sobrevivir
al propietario. Se conserva por separado la cola acotada PendingWork.

SegQueue reclama la posición antes de leer el payload y espera su publicación
con el bit WRITE. Los segmentos se retiran en vez de sobrescribir slots
circulares mientras puede quedar un lector especulativo. No añade mutexes al
camino de cola, pero esta decisión no acredita mejoras de rendimiento.

`tools/probe_segmented_work_loom.py` instrumenta el archivo real fijado de
crossbeam-queue 0.3.13: atomics/cells/backoff Loom y reserva explícita de células
instrumentadas. Conserva push/pop y el protocolo de retirada; omite únicamente
APIs exclusivas no usadas por WorkQueue (push_mut/pop_mut/IntoIterator).

Tres modelos de publicación, disputa entre consumidores y cambio de segmento con
ladrón retrasado pasan a tamaño reducido (3 slots) y upstream (31 slots).
Relajar Release de WRITE o Acquire de su espera rompe el modelo de publicación.
Pruebas nativas: 82 pasan, incluida unicidad de 96 IDs y handles tras destruir
el propietario. Gate del árbol de trabajo: 704 pruebas, aprobado.

Reproducción: generar con `--source <caché>/crossbeam-queue-0.3.13 --output
<directorio-nuevo> --lap 32`, preparar lockfile offline y ejecutar `cargo test
--manifest-path <directorio>/Cargo.toml --offline --locked --lib`. `--lap 4`
reduce el bloque; `--weaken-publication` y `--weaken-acquire` son controles
negativos deliberados, filtrando el test de publicación.

Evidencia: `quality/evidence/work-stealing-segmented-replacement.json`.
El conflicto especulativo anterior se elimina por sustitución del protocolo;
9.2 sigue abierta para ampliar reclamación, contabilidad de Drop exactamente una
vez, controles negativos de retirada e integración revisada del cambio.

### Destrucción de payloads verificada

La ampliación posterior ejecuta cinco modelos a 3 y 31 slots por bloque:
**5/5 pasan en ambos tamaños**. Dos modelos nuevos cuentan Drop por ID al
consumir el último segmento en disputa y abandonar tres payloads pendientes,
y comprueban que un valor extraído sobrevive a la destrucción de la cola.
`--omit-payload-drop` omite deliberadamente la destrucción de datos pendientes;
el primer modelo nuevo falla con `payload was not destroyed`.

Los hashes de las fuentes positivas instrumentadas coinciden con los probes
anteriores. Evidencia: `quality/evidence/work-stealing-segmented-drop.json`.
La contabilidad de payloads no cuenta las reservas/liberaciones de segmentos:
sus controles negativos y la integración aislada siguen pendientes en 9.2.

### Contabilidad de segmentos e integración aislada

La instrumentación posterior asigna un ID a cada reserva de Block y cuenta su
Drop con atómicos Relaxed, sin añadir publicación Acquire/Release. Los cinco
modelos pasan con bloques de 3 y 31 slots, comprobando al final una liberación
por reserva. Omitir la liberación en retirada o en Drop final produce el
diagnóstico `segment allocation … was not released` en ambos tamaños (cuatro
controles negativos detectados). Esto detecta fugas; no demuestra por sí solo
ausencia general de liberación prematura con punteros crudos.

El árbol aislado `43e02539913e14ab25a196c1581446442f65d103` selecciona WorkQueue,
WorkStealer y los probes sin el bloque PendingWork pendiente: runtime all-features
pasa 63 pruebas en build nuevo. Evidencia y hashes:
`quality/evidence/work-stealing-segmented-reclamation.json`.
Quedan revisión de protección frente a reclamación prematura y gate/commit de
la integración aislada antes de cerrar 9.2.
