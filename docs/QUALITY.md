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
| Trazabilidad | Seis escenarios con tests registrados en Cargo; la ejecución se comprueba aparte |
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
BML_TEST_GGUF=/ruta/modelo.gguf cargo test -p bml-parser open_real_tinyllama -- --ignored
```

Strace traza el proceso del parser; perf mide un proceso que ejecuta el evaluador. Si se solicitan sin permisos o herramientas, fallan. No ejecutar `--ignored` indiscriminadamente, pues incluye las entradas auxiliares de esos hijos. Los tests `thread_concurrency` usan hilos del sistema, sin demostrar exploración exhaustiva de interleavings mediante Loom.

## Contratos y límites

- RPN conserva cada nodo alcanzable una vez, con bytecode, tiempo y pila lineales. El mayor uso de pila obliga a medir de nuevo el rendimiento.
- `.bmlgraph` v2 guarda el pool de constantes; el lector sigue admitiendo v1. Consumidores antiguos necesitan actualización para leer v2. Una API `*_with_ctx` usa el pool del contexto explícito.
- `add` admite signos en el dominio numérico probado; `sub` exige minuendo positivo; `mul`, `div`, `recip` operandos positivos; `pow` x > 1 e y > 0. Las expansiones no garantizan estabilidad f64 general. `neg` conserva infinitos intermedios del evaluador heredado, incompatible con el evaluador real estricto.
- GGUF valida rangos exactos para escalares, Q4_0 y Q8_0. Otros cuantizadores solo permiten inspección de metadatos; `tensor_data` devuelve None. Se rechazan arrays anidados. El archivo mapeado debe permanecer sin modificaciones ni truncamiento mientras se usa.
- Fuzzing, validación completa de bytecode, tensores y concurrencia instrumentada siguen en el roadmap. Microbenchmarks no certifican inferencia ni tokens/s reales.
