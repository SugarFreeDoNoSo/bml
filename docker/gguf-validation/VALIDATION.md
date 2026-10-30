# Evidencia de desarrollo, 30 de octubre de 2026

- Base local previa: `97f832bb1d12d243f36aba6f704eb3b4b0accba4`.
- Docker daemon 28.4.0; plataforma comprobada Linux/x86_64 (imagen amd64).
- Build real con BuildKit, bases por digest y CA temporal manteniendo TLS.
- Smoke host y Docker: seis casos aprobados (Q4_0 sintético con arquitectura
  gemma4, archivo ausente, F32 no compatible con el probe, límites de filas,
  truncamiento y referencia incorrecta). El caso negativo de comparación exige
  explícitamente el error de comparación, no solo un fallo genérico del proceso.
- Docker smoke: usuario UID/GID del invocador, filesystem de imagen readonly,
  red deshabilitada, cap-drop ALL, no-new-privileges, 512 MiB y 2 CPU.
- `cargo test -p bml-gguf --locked --offline`: 41 aprobadas, cero fallidas,
  una ignorada que requiere modelo externo.
- `cargo clippy -p bml-gguf --examples --locked --offline -- -D warnings`: aprobado.
- `CARGO_NET_OFFLINE=true bash scripts/check.sh 97f832bb1d12d243f36aba6f704eb3b4b0accba4`: 312 Rust y cinco Python aprobadas; 15 opcionales ignoradas; cero discrepancias, trazabilidad 15/15, cero diagnósticos Clippy nuevos y formato correcto.
- Python: compilación de run.py y smoke.py; ejecución de ambas rutas mediante smoke.

La proyección es un dot con activación sintética: decodificación BML comparada
con la función upstream C original y acumulación independiente Python float64.
No certifica las capas ni el matmul de inferencia BML. El smoke no contiene ni
usa pesos de un modelo descargado. El informe de cada ejecución real queda en
result.json/run.log del volumen; el informe del build contiene commit, estado
del checkout y hashes de Cargo.lock y fuente Rust. La exportación lleva un
manifiesto externo con commit final, image ID, tamaños y SHA256 de artefactos.

No se probó Docker Desktop/Windows/macOS, ARM/multiarch, GPU ni GGUF real.
No se ejecutó inferencia Gemma4/TinyLlama ni benchmarks o pruebas WSL.
No se publicaron imágenes, commits remotos o PRs. Library no tiene un mecanismo
de subida expuesto en esta sesión: los archivos exportados se conservan localmente
para su transferencia, sin atribuirles Library ID ni persistencia remota.

Comandos principales desde la raíz:

```sh
source /workspace/.bml-env/env.sh  # solo este cloud; en otro host usar Rust 1.98.1
cargo build -p bml-gguf --example bounded_probe --release --locked --offline
cc -std=c11 -O2 docker/gguf-validation/reference/shim.c -lm -o /tmp/bml-q4-reference
BML_PROBE="$PWD/target/release/examples/bounded_probe" BML_REFERENCE=/tmp/bml-q4-reference python3 docker/gguf-validation/smoke.py
python3 docker/gguf-validation/smoke.py --image bml-gguf-validation:local
```
