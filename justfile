# ============================================================================
# BML — Justfile
# ============================================================================
# Comandos organizados por categoría:
#   build     → Compilación y verificación
#   test      → Tests unitarios, integración, concurrencia, estrés
#   bench     → Benchmarks (criterion)
#   deploy    → Despliegue de servicios (server, worker, cli)
#   dev       → Utilidades de desarrollo (doc, clean, watch, setup)
#
# Uso:
#   just <comando>              → ejecutar un comando
#   just --list                 → ver todos los comandos disponibles
#   just test-compiler filter=kv_cache  → filtrar tests específicos
#
# Para controlar paralelismo: export CARGO_BUILD_JOBS=N
# ============================================================================

# Filtro opcional para tests/benchmarks específicos
filter := ""

# ─── Build & Check ───────────────────────────────────────────────────────────

# Compilar todo el workspace (debug)
build:
    cargo build --workspace

# Compilar todo el workspace (release, optimizado)
build-release:
    cargo build --workspace --release

# Solo verificar que compila (más rápido que build)
check:
    cargo check --workspace

# Verificar + clippy (lints)
clippy:
    cargo clippy --workspace -- -D warnings

# Verificar formato
fmt-check:
    cargo fmt --all -- --check

# Aplicar formato automático
fmt:
    cargo fmt --all

# Build + clippy + fmt (CI rápido)
ci-check: check clippy fmt-check

# ─── Tests: Unitarios ────────────────────────────────────────────────────────

# Tests unitarios de bml-domain (sin dependencias, muy rápidos)
test-domain:
    cargo test -p bml-domain {{if filter == "" { "" } else { "-- " + filter }}}

# Tests unitarios de bml-parser (parser GGUF, mmap)
test-parser:
    cargo test -p bml-parser {{if filter == "" { "" } else { "-- " + filter }}}

# Tests unitarios de bml-compiler (compilador, DAG, KV cache, etc.)
# Sin proptest (que es muy pesado)
test-compiler:
    cargo test -p bml-compiler --lib -- --skip proptest {{if filter == "" { "" } else { filter }}}

# Tests unitarios de bml-runtime (hot loop, buffers, scheduler)
test-runtime:
    cargo test -p bml-runtime {{if filter == "" { "" } else { "-- " + filter }}}

# Todos los tests unitarios (domain + parser + compiler + runtime)
test-unit: test-domain test-parser test-compiler test-runtime

# Tests unitarios con proptest incluido (más lento, puede usar mucha RAM)
test-unit-full:
    cargo test -p bml-domain -p bml-parser -p bml-compiler -p bml-runtime {{if filter == "" { "" } else { "-- " + filter }}}

# ─── Tests: Integración ──────────────────────────────────────────────────────

# Test de integración: BMLGraph round-trip
test-integration:
    cargo test -p bml-tests --test bmlgraph_integration {{if filter == "" { "" } else { "-- " + filter }}}

# ─── Tests: Concurrencia ─────────────────────────────────────────────────────

# Tests de concurrencia con loom (verificación formal de sync primitives)
# Requiere LOOM_MAX_PREEMPTIONS (default: 3), single-threaded
test-loom:
    LOOM_MAX_PREEMPTIONS=3 cargo test -p bml-tests --test concurrency_loom -- --test-threads=1 --nocapture

# Test de concurrencia del runtime (crossbeam, sin loom)
test-concurrency:
    cargo test -p bml-tests --test runtime_concurrency {{if filter == "" { "" } else { "-- " + filter }}}

# Todos los tests de concurrencia (loom + runtime)
test-concurrency-all: test-loom test-concurrency

# ─── Tests: Estrés ───────────────────────────────────────────────────────────

# Test de estrés: multicore
test-stress-multicore:
    cargo test -p bml-tests --test stress_multicore -- --test-threads=1 --nocapture

# Test de estrés: shared memory distribuida
test-stress-shm:
    cargo test -p bml-tests --test distributed_shm -- --test-threads=1 --nocapture

# Test de estrés: caché L1/L3
test-stress-cache:
    cargo test -p bml-tests --test perf_cache -- --test-threads=1 --nocapture

# Todos los tests de estrés
test-stress: test-stress-multicore test-stress-shm test-stress-cache

# ─── Tests: Agrupados ────────────────────────────────────────────────────────

# Tests rápidos (unitarios sin proptest + integración) — ~10s
test-fast: test-unit test-integration

# Todos los tests excepto loom y estrés — ~2 min
test-normal: test-unit test-integration test-concurrency

# Absolutamente todos los tests — ~30 min
test-all: test-unit-full test-integration test-concurrency-all test-stress

# CI: tests rápidos que no requieren setup especial
test-ci: test-unit test-integration

# ─── Benchmarks ──────────────────────────────────────────────────────────────

# Benchmark: costo de operaciones individuales (BML vs FMA vs exp2 vs log2)
# + hot loop con programas de distintos tamaños
bench-ops:
    cargo bench -p bml-bench --bench bml_ops {{if filter == "" { "" } else { "-- " + filter }}}

# Benchmark: FMA vs BML (hash consing, deduplicación)
bench-fma:
    cargo bench -p bml-compiler --bench fma_vs_bml {{if filter == "" { "" } else { "-- " + filter }}}

# Benchmark: multiplicación de matrices (BML RPN vs naive vs ndarray)
bench-matmul:
    cargo bench -p bml-compiler --bench matrix_mul {{if filter == "" { "" } else { "-- " + filter }}}

# Benchmark: funciones complejas (BML pipeline completo)
bench-complex:
    cargo bench -p bml-compiler --bench complex_functions {{if filter == "" { "" } else { "-- " + filter }}}

# Benchmark: rendimiento final (tokens/seg, escalado cloud, costo/$)
bench-final:
    cargo bench -p bml-compiler --bench final_benchmark {{if filter == "" { "" } else { "-- " + filter }}}

# Todos los benchmarks del compilador
bench-compiler: bench-fma bench-matmul bench-complex bench-final

# Absolutamente todos los benchmarks
bench-all: bench-ops bench-compiler

# ─── Deploy ──────────────────────────────────────────────────────────────────

# Compilar y preparar CLI para release
deploy-cli: build-release
	@echo "✓ bml-cli compilado en target/release/bml-cli"
	@echo "  Uso: target/release/bml-cli --help"

# Compilar servidor HTTP (API REST con axum)
deploy-server: build-release
	@echo "✓ bml-server compilado en target/release/bml-server"
	@echo "  Uso: target/release/bml-server"
	@echo "  Variables de entorno:"
	@echo "    BML_HOST=0.0.0.0       (default: 127.0.0.1)"
	@echo "    BML_PORT=8080          (default: 8080)"
	@echo "    BML_MODEL=model.gguf   (ruta al modelo GGUF)"

# Compilar worker distribuido
deploy-worker: build-release
	@echo "✓ bml-worker compilado en target/release/bml-worker"
	@echo "  Uso: target/release/bml-worker"

# Compilar todos los binarios de release
deploy-all: deploy-cli deploy-server deploy-worker

# ─── Dev ──────────────────────────────────────────────────────────────────────

# Limpiar build artifacts
clean:
	cargo clean

# Generar documentación y abrir en navegador
doc:
	cargo doc --workspace --no-deps --open

# Generar documentación (sin abrir)
doc-build:
	cargo doc --workspace --no-deps

# Watch: recompilar y correr tests al cambiar archivos
watch:
	cargo watch -x check -x "test --lib -- --skip proptest"

# Watch solo tests de compiler (para desarrollo rápido)
watch-compiler:
	cargo watch -x "test -p bml-compiler --lib -- --skip proptest"

# Instalar/actualizar tooling
setup:
	rustup update stable
	rustup component add clippy rustfmt
	cargo install cargo-watch just

# Verificar que todo el tooling está instalado
doctor:
	@echo "=== Toolchain ==="
	@rustc --version
	@cargo --version
	@just --version
	@echo ""
	@echo "=== Components ==="
	@rustup component list --installed | grep -E "clippy|rustfmt|llvm-tools" || echo "  (run 'just setup' to install)"
	@echo ""
	@echo "=== Crates ==="
	@cargo metadata --format-version 1 --no-deps 2>/dev/null | grep '"name"' | head -9 || echo "  (run 'cargo check' first)"

# ─── CI Pipeline ─────────────────────────────────────────────────────────────

# Pipeline completo de CI (sin benchmarks)
ci: clean ci-check test-ci

# Pipeline completo de CI + benchmarks
ci-full: clean ci-check test-all bench-all

# ─── Atajos ───────────────────────────────────────────────────────────────────

# Alias: t = test-fast
alias t := test-fast

# Alias: ta = test-all
alias ta := test-all

# Alias: b = build
alias b := build

# Alias: br = build-release
alias br := build-release

# Alias: c = check
alias c := check
