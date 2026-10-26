# Contratos y controles de calidad

Se adaptan escenarios trazables, evidencia TDD, revisión y mutación de [harness-sdd/uncle-bob-harness](https://github.com/betta-tech/harness-sdd/tree/bb80951622065a32057634d222cf7ed93a2a09cb). Clippy, cobertura, RustSec y hooks Git son controles adicionales implementados para Rust.

## Ejecución reproducible

```sh
./scripts/check.sh origin/master
./scripts/install-hooks.sh
cargo llvm-cov --workspace --locked --json --output-path target/quality/coverage.json
python3 tools/quality.py coverage --report target/quality/coverage.json
python3 tools/mutation_diff.py origin/master target/quality/changes.diff
cargo mutants --workspace --cargo-arg=--locked --test-workspace true --in-diff target/quality/changes.diff
python3 tools/quality.py mutations --report mutants.out/outcomes.json
```

Rust está fijado en `rust-toolchain.toml`; las versiones de las herramientas están fijadas en los workflows. El hook pre-commit comprueba el contenido staged, pre-push ejecuta los controles ordinarios y ambos conservan el código de error. Instalación solo local al repositorio; `--no-verify` permite omitir hooks. Las reglas de la rama deben exigir los checks de Actions para impedir merges cuando fallen; este PR no modifica esas reglas.

| Control | Criterio |
|---|---|
| Tests y doctests | La suite ordinaria pasa y los optativos figuran como ignored |
| Auditoría | Cero discrepancias en los seis contratos históricos |
| Trazabilidad | Trece escenarios con tests registrados por package y target de Cargo; la ejecución se comprueba aparte |
| Formato | Rustfmt en archivos Rust nuevos o modificados |
| Clippy | Ningún diagnóstico nuevo por archivo, lint y mensaje |
| Cobertura | Líneas y funciones sobre los mínimos versionados |
| Mutación | Baseline correcto, campaña completa, cero supervivientes y cero timeouts |
| Dependencias | Auditoría RustSec al cambiar manifests/lockfile y semanalmente |

Actions conserva informes incluso cuando falla un control. Los jobs usan permisos de lectura, acciones fijadas por SHA y ejecución de PR sin secretos ni `pull_request_target`.

La base de Clippy registra 192 diagnósticos preexistentes de `57c3bd7` con Rust 1.98.1. `--cap-lints warn` permite inventariar también lints denegados heredados: cualquier diagnóstico nuevo o error de compilación sigue haciendo fallar el script. Esta deuda debe reducirse por cambios revisados; no regenerar la base automáticamente.

La cobertura estable mide líneas y funciones del workspace ordinario. No incluye ramas ni doctests: estos últimos se ejecutan por separado; ramas requieren otra campaña de instrumentación. Los mínimos iniciales tienen margen cercano a un punto porcentual respecto de la medición que los originó. Cambiar el presupuesto requiere revisión.

La mutación compara el árbol base formateado con el código actual para excluir cambios exclusivamente de formato. Cada mutante viable debe ser detectado. El porcentaje es `caught / (caught + missed + timeout)`; los no compilables se reportan aparte. Una campaña vacía o sin mutantes viables es N/A, nunca 100 %. Una campaña incompleta falla. La opción manual `full_mutation` mide todo el workspace. No hay exclusiones por equivalencia preaprobadas.

## Instrumentación optativa

```sh
cargo test -p bml-parser --test zero_copy_strace -- --ignored --exact strace_shows_mmap_not_read
cargo test -p bml-tests --test perf_cache -- --ignored --exact perf_stat_cache_events_on_program
BML_TEST_GGUF=/ruta/modelo.gguf cargo test -p bml-gguf open_real_tinyllama -- --ignored
```

Strace traza el proceso del parser; perf mide un proceso que ejecuta el evaluador. Si se solicitan sin permisos o herramientas, fallan. No ejecutar `--ignored` indiscriminadamente, pues incluye las entradas auxiliares de esos hijos. Los tests `thread_concurrency` usan hilos del sistema, sin demostrar exploración exhaustiva de interleavings mediante Loom.

## Contratos y límites

- RPN conserva cada nodo alcanzable una vez, con bytecode y pila lineales en los nodos alcanzables. Inicializar el mapa de posiciones y copiar el pool cuesta además O(nodos totales + constantes). El mayor uso de pila obliga a medir de nuevo el rendimiento.
- `.bmlgraph` v2 guarda el pool de constantes; el lector sigue admitiendo v1. Consumidores antiguos necesitan actualización para leer v2. Una API `*_with_ctx` usa el pool del contexto explícito.
- En `BMLTransformer`, `add` admite signos en el dominio numérico probado; `sub` exige minuendo positivo; `mul`, `div`, `recip` operandos positivos; `pow` x > 1 e y > 0. Las expansiones no garantizan estabilidad f64 general. `neg` conserva infinitos intermedios del evaluador heredado, incompatible con el evaluador real estricto. Estas restricciones no corresponden a los kernels nativos de `tensor`.
- GGUF valida rangos exactos para escalares, Q4_0 y Q8_0. Otros cuantizadores solo permiten inspección de metadatos; `tensor_data` devuelve None. Se rechazan arrays anidados. El archivo mapeado debe permanecer sin modificaciones ni truncamiento mientras se usa.
- `tensor` ofrece buffers contiguos f32/f64, vistas prestadas y kernels CPU. EML/BML elementwise valida en f64 y comprueba la conversión de salida; broadcasting y strides arbitrarios siguen pendientes.
- Reader/codecs y vistas GGUF reconstruidos en `bml-gguf`, con puente tensorial optativo; fachada, fuzzing, validación completa de bytecode y concurrencia instrumentada siguen en el roadmap. Microbenchmarks no certifican inferencia ni tokens/s reales.

## Recuperación GGUF (2026-10-22)

El bloque B14/#18 se reconstruye desde `df146c2`; los resultados locales perdidos
no son evidencia. Por defecto `bml-gguf` solo depende de `memmap2`; `bml-parser` delega sin
cambiar su `io::Result` y compiler usa el codec compartido. Los fixtures verifican
F32/F16/Q4_0/Q8_0 y las 65.536 codificaciones binary16. Se corrige la identificación
GGML: 29 es IQ1_M, 30 BF16; ninguno tiene decoder f32 en este incremento. Los
valores auxiliares del enum para metadatos no son IDs serializables GGML.

Los errores tipados conservan la categoría y contexto del fallo. El decoder no
acepta colas/truncamientos ni devuelve salidas parciales. Los arrays anidados
siguen rechazados explícitamente. Esto no certifica todos los archivos GGUF ni
inferencia, rendimiento, fuzzing o una campaña completa de mutación.

## Reconstrucción de vistas GGUF (#19)

La pérdida del entorno local exige reconstruir y validar de nuevo este incremento.
`view.rs` y el consumidor externo se recuperan de texto completo del transcript;
`tensor_views.rs` se escribe de nuevo a partir de contratos y fragmentos, sin
atribuirle identidad byte a byte con el archivo perdido.

El feature `tensor` añade `bml-domain` temporalmente hasta extraer `bml-core`.
Las vistas están ligadas al reader; la conversión invierte ejes GGML sin reordenar
valores. Los tests cubren matrices rectangulares, rangos 1/3/4, F16/Q4_0/Q8_0,
ownership, errores tipados y una restricción de lifetime mediante doctest.
`bml-tests` activa el feature como dependencia de desarrollo y registra un
consumidor real, de modo que los comandos existentes del workspace lo incluyen.

Esta reconstrucción pasó `scripts/check.sh`, las suites aisladas sin features y
con `--all-features`, cobertura y mutación del delta #19 con tests del workspace. No se modifican
baselines ni workflows. El estado comprobado está en
`quality/evidence/gguf-tensor-views-reconstruction.json`; no usar cifras antiguas.
