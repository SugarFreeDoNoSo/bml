# Reparación de operaciones públicas y ejecución entre cores — 2026-11-03

La auditoría de `9338c429c4e4ed4319f022fdbb224d850540acce` encontró
matmul y sampling incorrectos aunque las rutas causales modernas pasaban sus
pruebas. Esta entrega corrige esas APIs, el helper público SwiGLU y la ejecución
de fragmentos dependientes. **No cierra `fix-placeholders` ni el objetivo global**:
el inventario G1 de sus tareas conserva las APIs simplificadas pendientes.

## Comportamiento corregido

| API | Antes | Evidencia del cambio |
| --- | --- | --- |
| `compile_matmul_fragment` | BML literal usado como producto/suma, estado de bucle incorrecto; `[0, NaN]` donde se esperaban `[11, 17]` | C85 ejecuta la API histórica y el constructor configurable con matrices firmadas/rectangulares, bases no nulas, cero filas/columnas y pila compartida. Cada fila usa identidades BML de producto y suma. |
| `sampler::sample` | Toda temperatura menor que `1e-9` seleccionaba greedy; un uniforme cero podía elegir probabilidad cero | Quince pruebas unitarias del código productivo: oráculos probabilísticos, selección determinista, CDF, escala desde el menor subnormal hasta `f64::MAX`, empates, overflow e intervalos del generador. Dos regresiones fallan en el código anterior. |
| `bml_swiglu_program` | Reducía `gate*up`, sin sigmoid ni escrituras de salida | C86 verifica `gate*sigmoid(gate)*up`, casos cerrados ±ln(2), signo/cero, frontera ±16, bindings de constantes, buffers, stack y tamaño de código constante. Se usa BML para toda la activación. |
| `Runtime::execute_with_cores` / `execute_fragments_parallel` | Chunks bajo un mutex, sin orden entre productores/consumidores y sin cálculo simultáneo | C87 verifica RAW/WAR/WAW, índices que cruzan slots, índices desconocidos, estado entre iteraciones, reads de outputs propios, stacks compartidos y un oráculo secuencial independiente de programas generados. Una prueba interna exige que dos workers entren antes de que cualquiera salga, con timeout para detectar serialización. |

El nuevo plan concurrente deriva accesos del bytecode y del layout real, no de
`FragmentMeta`: `VarIndexed` usa una dirección absoluta y `StoreResult` un slot.
Los índices desconocidos incluyen todo el rango del cast saturado legado.
El análisis de bucles ensancha valores cambiantes hasta un punto fijo con
presupuesto de análisis. Si no demuestra independencia o los fragmentos
comparten la pila, conserva la ejecución secuencial. Las waves respetan el
orden original y no exceden los cores solicitados.

Cada worker toma prestados los slots de entrada y posee slices mutables de sus
salidas. No se clonan pesos, activaciones ni el `ResultBuffer`; los préstamos
disjuntos se comprueban por Rust sin código unsafe. La planificación, las vistas
y los hilos sí asignan memoria. El camino secuencial no necesita asignaciones
si la pila ya tiene capacidad suficiente. El dispatcher conserva un único
cuerpo con un enum privado de almacenamiento, sin llamadas virtuales abiertas.

## Contratos y límites numéricos

- El matmul histórico mantiene su firma y su única constante externa de ancho.
  Declara límites absolutos 16 por operando, acumulador menor que 256 antes de
  cada paso, direcciones válidas y salida disjunta. Los literales adicionales se
  construyen con términos BML cerrados. No se convierte en un contrato BLAS
  universal ni se recortan valores para hacerlos entrar en el rango.
- `compile_matmul_fragment_checked` acepta límites configurables, mantiene su
  pool y bytecode inmutables y comprueba formas, direcciones y cada BML real
  antes de la ejecución. El preflight recibe los valores efectivos que el
  llamador debe colocar en el buffer. No sustituye la salida por aritmética
  nativa y no valida automáticamente un buffer distinto proporcionado después.
- SwiGLU reutiliza la identidad acotada con gate/up de magnitud máxima 16.
  Conserva los bindings externos de uno y `log2(e)` y documenta que `1.7` es
  obsoleto. Los builders BML siguen teniendo dominio numérico parcial; no se
  atribuye rango IEEE universal ni preservación general de cero con signo.
- Sampling usa greedy exclusivamente para cero. Temperaturas negativas/no
  finitas y logits no finitos devuelven `None`. La semilla pública continúa
  ignorada: las pruebas deterministas inyectan un uniforme en la misma ruta
  interna productiva y no prometen reproducibilidad por esa semilla.

## Verificación reproducible

Toolchain fijado por `rust-toolchain.toml` (1.98.1), siempre `--locked`:

Gate final aprobado sobre la base `9338c42`: **808 pruebas Rust, 86 Python,
87/87 contratos trazados, cero discrepancias históricas y cero diagnósticos
Clippy nuevos** (61 presentes frente a 192 de baseline). Compilación de workspace
con todas las features y validación OpenSpec estricta de 11 elementos también
aprobadas. La compilación all-features no se presenta como pruebas de ejecución
de todas las combinaciones de features.

```sh
cargo test -p bml-compiler --lib sampler::tests --locked
cargo test -p bml-compiler --lib op_fragments::tests --locked
cargo test -p bml-compiler --lib bml_ops::tests --locked
cargo test -p bml-runtime --test operation_matmul_contracts --locked
cargo test -p bml-runtime --test operation_swiglu_contracts --locked
cargo test -p bml-runtime --test operation_schedule_contracts --locked
cargo test -p bml-runtime --lib operation_schedule::tests --locked
./scripts/check.sh origin/master
```

Los logs focalizados están en `target/quality/operation-{matmul,swiglu,schedule}-*`
y `target/quality/sampler-*`. El consumidor greedy de pesos autónomos también
pasó su oráculo independiente. Los resultados finales, fuentes y hashes de logs
se registran en `operation-fragment-repair-2026-11-03.json`. La auditoría binaria
del dispatcher actualizado se registra por separado en
`operation-schedule-hot-loop.json`; medir bytes no prueba comportamiento L1 ni
sustituye los contadores PMU pendientes.

## Trabajo que conserva el alcance original

Siguen pendientes los helpers RMS de ambos módulos, el helper RoPE, atención,
MLP/down y composición completa de capa con residuales/layout. El pipeline
distribuido histórico combina un compilador que emite `Var(0)`, un worker que
solo ejecuta RMS/Q y una CLI con embeddings fabricados y hidden usado como
logits. Debe corregirse como pipeline completo, incluidos estado causal y
cobertura exacta de fragmentos. La ruta nativa `RemotePipeline` existente no
prueba la ejecución de esos fragmentos BML.

El perfil causal v4, su validación en modelos mayores y su escalabilidad RMS/
softmax mantienen su seguimiento separado. Tampoco se cierran las campañas
globales de mutación, ARM/NEON ni las pruebas de hardware. No se usa un gate
verde para acreditar un requisito contradicho por una API aún simplificada.
