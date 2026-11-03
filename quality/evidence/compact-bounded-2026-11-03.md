# RMS/RoPE públicos y programas acotados compactos — 2026-11-03

Esta entrega completa las normalizaciones y rotaciones públicas pendientes y
reduce el almacenamiento RMS/softmax del perfil causal v4. Conserva los
constructores numéricos anteriores, los artefactos v1–v3 y la carga v4 histórica.
**El objetivo global sigue abierto**: faltan otras APIs, validación en modelos
mayores/prompts adicionales y los gates globales/de hardware.

## Fórmulas y programas

C88/C89 contrastan cada BML de los nuevos stages con el constructor anterior y
con oráculos independientes. Los métodos anteriores permanecen sin cambios.
RMS reduce los cuadrados normalizados, calcula la raíz y los factores compartidos
y emite cada salida ponderada. Softmax empieza con el primer exponencial, suma
los siguientes en el mismo orden y calcula un único inverso del total.
Recalcular un exponencial o una normalización por elemento no reasocia sumas.
La eliminación local de nodos repetidos reutiliza únicamente operaciones con
operandos idénticos; no guarda ni introduce valores de activaciones observadas.

| Kernel v4 | Instrucciones almacenadas | Bytes en memoria RPN | Literales fijos | Pico de pila | Pasos ejecutados para ancho `n` |
| --- | ---: | ---: | ---: | ---: | --- |
| RMS | 1934 | 23208 | 17 | 271 | `1936 + 1025*(n-1)` |
| Softmax | 1797 | 21564 | 10 | 33 para n=1; 34 para n>1 | `1202 + 1197*(n-1)` |

C90/C91 verifican anchos 1, 288 y 2048 contra los bits de las expansiones
históricas y contra oráculos (tolerancia absoluta `2e-7` o más estricta según
caso). También comprueban los límites exactos de nodos, opcodes, programas,
fragmentos indivisibles, pasos y valores. Código pequeño no significa trabajo
constante ni autoriza superar el presupuesto de ejecución.

## Persistencia y compatibilidad

El layout explícito `indexed_buffer` está restringido a Norm/Softmax v4. Usa
contadores de bucle para las escrituras y no añade un literal por índice de
salida. El cargador acepta también el layout `buffer` histórico v4 con sus
literales y bytecode antiguos. V1–v3 y las demás familias rechazan el layout nuevo.
No se modifica `OutputLayout::for_operation` para los perfiles anteriores.

C94 ejecuta ambos formatos v4 con `max_nodes=0`, incluidos RMS con pesos no
unitarios y softmax negativo/extremo. Los programas antiguos se construyen con
el RMS anterior y el softmax denso original, se persisten con sus hashes y se
cargan a través de la API real. Las pruebas rechazan prefijos ausentes, literales
alterados y perfiles/layouts incompatibles; una salida omitida no confirma KV,
y sustituir por bytecode válido cambia la salida sin recompilarla.

En un fixture causal de una capa, conservando todos los prefijos requeridos:

| Contexto | Kernels | Literales totales | Bytes de programas |
| ---: | ---: | ---: | ---: |
| 2 | 11 | 123 | 60582 |
| 4 | 15 | 163 | 78062 |
| 8 | 23 | 243 | 113022 |
| 16 | 39 | 403 | 182942 |

Los conteos de instrucciones/literales y bytes de programas crecen linealmente.
El límite de 4096 programas y los demás presupuestos continúan vigentes. Esta
prueba no afirma que el plan de contexto 2048 quepa en esos presupuestos.

## APIs públicas y pools de constantes

- `compile_rmsnorm_fragment[_checked]` y `bml_rmsnorm_full_program` implementan
  RMS completo, incluido epsilon, media, raíz, pesos y escrituras por elemento.
  El constructor configurable añade preflight real de todos los stages.
- `OperationFragment` ahora tiene `const_pool: Option<Vec<f64>>`. El runtime y
  el análisis de dependencias usan el mismo `fragment.context(parent)` por
  operación. `None` mantiene el contexto externo histórico. Los llamadores que
  construyan el struct manualmente deben añadir ese campo; quienes ejecuten el
  fragmento crudo deben enlazar el contexto que expone la operación.
- `CheckedMatmulFragment` conserva su API de constantes y su aritmética; el pool
  reside ahora en la operación y el runtime lo enlaza automáticamente.
- `bml_rmsnorm_program` conserva su contrato de factor recíproco precomputado y
  dos multiplicaciones IEEE, corrigiendo las salidas y la pila. No calcula la
  media/raíz y no se presenta como una identidad BML. El escalado BML configurable
  y el RMS completo son APIs explícitas distintas.
- El RoPE de par conserva las coordenadas de `EvalContext`, elimina la negación
  mediante infinitos y escribe ambas componentes. El constructor vectorial usa
  bases absolutas, cabezas, posición y frecuencias explícitas. C93 comprueba las
  1024 parejas de 2048 coordenadas en posición cero y frecuencias distintas que
  se reinician por cabeza, además de signos, memoria, pila y errores de layout.

C92 contrasta RMS firmado/no unitario, epsilon no estándar, anchos hasta 2048,
buffers y stack preservados y workers con configuraciones diferentes. Las
pruebas históricas del runtime que solo afirmaban «no panic» ahora comprueban
resultados de cadenas RMS contra un oráculo. La ausencia de solapamiento y la
capacidad del buffer siguen siendo precondiciones explícitas del llamador.
Los perfiles BML conservan dominios parciales y no sustituyen contratos IEEE.

## Ejecución real

El diagnóstico Rust publicó doce kernels compactos de Stories15M, los recargó
con presupuesto de construcción cero y completó los inputs `[1,9038]`. Cada
posición produjo 32000 logits finitos y argmax `9038`/`2501`, iguales a la ruta
nativa. Los errores máximos absolutos fueron `1.6198153041102614e-8` y
`2.9445231559677154e-6`; las nueve familias se ejecutaron con comprobación de
cada BML. Evidencia y hashes: `compact-bounded-model-2026-11-03.json`.

| Artefacto Stories15M | V4 anterior | V4 compacto |
| --- | ---: | ---: |
| Bytes de programas | 1815059 | 64581 |
| Bytes manifest + programas | 3143970 | 1389267 |
| Literales fijos | 424 | 133 |
| Programas | 12 | 12 |

El modelo GGUF permanece externo e idéntico. Esto mide almacenamiento de
programas/manifest, no compresión del modelo. Los tiempos observados (~136 y
137 segundos por token en la VM) no son un benchmark controlado ni una promesa
de speedup. Los resúmenes numéricos coinciden con la campaña anterior; no se
atribuyen hashes de vectores completos que el diagnóstico no guardó.

Después del run se cambió únicamente la iteración de pares del preflight RoPE
para satisfacer Clippy (`chunks_exact` a `as_chunks`) y un comentario obsoleto
de la capa histórica, además de pruebas/documentación. La prueba estructural
RMS ahora busca el bucle después de la inicialización y exige el pool propio. El informe del modelo
conserva los hashes medidos y registra esos cambios posteriores, sin fingir
que se repitió la inferencia sobre ellos.

## Verificación y pendientes

El gate canónico `./scripts/check.sh origin/master`, con base
`c05df2e9dcbf7df559dd5edec9b607160351bd6e`, pasó: 845 pruebas Rust
(17 ignoradas), 86 Python, auditoría histórica con cero discrepancias,
94/94 escenarios registrados y cero diagnósticos Clippy nuevos (61 actuales
frente a una baseline de 192). También pasaron `cargo check --workspace
--all-features --locked` y OpenSpec estricto (11/11). El check con todas las
features acredita compilación, no la ejecución de pruebas con todas ellas.

Los comandos, resultados finales y hashes están en
`compact-bounded-2026-11-03.json`; logs focalizados en `target/quality/` con
prefijos `bounded-*-stage`, `compact-bounded-*`, `owned-rms-*`, `rms-rope-*` e
`indexed-buffer-*`. La auditoría binaria actual se registra por separado en
`compact-bounded-hot-loop-2026-11-03.json`; no acredita PMU/L1.

Quedan abiertas atención, MLP/down, composición de capa con residuales/layout y
el compilador/worker/CLI distribuido histórico. La nueva normalización no
corrige automáticamente la confusión de slots de esa capa. Tampoco se cierran
TinyLlama v4, prompts adicionales, ARM/NEON real, PMU/L1, estrés de hardware ni
el gate global de cobertura/mutaciones. No se archiva `fix-placeholders`.
