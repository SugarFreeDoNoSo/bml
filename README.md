# BML — Binary-Minus-Log

Un compilador y motor de ejecución en Rust basado en un único operador matemático con completitud funcional, optimizado de forma extrema para la jerarquía de caché del procesador (específicamente L1).

## ¿Qué es BML?

**BML** = **Binary-Minus-Log**: `bml(x, y) = 2^x - log2(y)`.

Es el análogo de **EML** (Exp-Minus-Log, `eml(x, y) = exp(x) - ln(y)`) reescrito en **base 2** para alinearse con el formato IEEE 754 de `f64` y usar `exp2`/`log2` nativos de la FPU.

BML es un operador con **completitud funcional**: junto con la constante `1`, genera todo el repertorio de una calculadora científica (aritmética, exponencial, logaritmo, trigonométricas, etc.). Actúa como el **análogo continuo del NAND lógico** — un único operador binario del cual derivan todos los demás.

### Origen teórico

El paper ["All elementary functions from a single operator"](https://arxiv.org/abs/2603.21852v2) (ArXiv 2603.21852v2) demuestra que EML (`exp(x) - ln(y)`) con la constante `1` forma una base completa para las funciones elementales. BML adapta este resultado a base 2.

### Identidades fundamentales (base 2)

| Identidad | Fórmula | Análogo EML |
|---|---|---|
| Constante fundamental | `2 = bml(1, 1)` | `e = eml(1, 1)` |
| Exponencial | `2^x = bml(x, 1)` | `exp(x) = eml(x, 1)` |
| Logaritmo | `log2(x) = bml(1, bml(bml(1, x), 1))` | `ln(x) = eml(1, eml(eml(1, x), 1))` |

### Propiedades algebraicas

- **Magma no asociativo**: el orden de los operandos es estrictamente inmutable (`bml(a, b) ≠ bml(b, a)` en general).
- **Gramática del AST**: `S → 1 | BML(S, S)` — un nodo es la constante `1` o una aplicación de `BML` a dos sub-árboles. No existen otras operaciones primitivas.

## Arquitectura

Workspace Cargo con 4 crates de responsabilidad aislada:

```
bml/
├── crates/
│   ├── domain/     # Operador BML, gramática AST, layout SoA, BMLTransformer
│   ├── parser/     # Ingesta GGUF zero-copy (mmap)
│   ├── compiler/   # DAG, Hash Consing, linealización RPN, micro-fragmentación .bmlgraph
│   ├── runtime/    # Hot loop RPN, buffers pre-asignados, RPC distribuido
│   └── bench/      # Binario bml-bench (futuro)
├── tests/
│   ├── integration/  # Tests cross-crate (domain + compiler + runtime)
│   ├── stress/       # Multicore, perf, /dev/shm
│   └── concurrency/  # loom, data races
├── docs/
│   └── benchmarks/   # Reportes consolidados
├── references/       # Paper, draft
└── openspec/         # Specs vivas y plan de ejecución
```

| Crate | Responsabilidad | Dependencias |
|---|---|---|
| `bml-domain` | Operador base, gramática, SoA, transformer | cero |
| `bml-parser` | Ingesta GGUF via mmap | `memmap2` |
| `bml-compiler` | DAG, Hash Consing, RPN, fragmentación AOT | `bml-domain` |
| `bml-runtime` | Ejecución RPN, RPC distribuido | `bml-compiler` |

> **Nota:** Las carpetas no llevan prefijo `bml-` (es redundante dentro del workspace). Los nombres de paquete sí lo conservan para el namespace en crates.io.

## Reglas de ingeniería (Mechanical Sympathy)

- **Data-Oriented Design**: OOP prohibido en la capa de ejecución.
- **SoA + `#[repr(align(64))]`**: Struct of Arrays alineado a línea de caché; AoS prohibido.
- **Hot loop < 32 KB**: el intérprete RPN del runtime debe caber en L1i.
- **Append-only**: durante la evaluación del DAG, nunca se sobrescribe estado previo.
- **Cero allocs en hot path**: los buffers se inicializan una sola vez al arrancar.

## Estado del proyecto

Plan de ejecución gestionado con [OpenSpec](https://openspec.dev/). Ver `openspec/changes/bml-implementation-plan/` para el plan completo.

| Hito | Estado | Descripción |
|---|---|---|
| 1 — Cimientos | ✅ | Operador BML, AST, SoA, transformer (`exp2`/`log2` verificados) |
| 2 — Compilación | ⏳ | Hash Consing, RPN, benchmarks FMA vs BML, fórmulas `+`/`-`/`*`/`/`/`pow` |
| 3 — Parser | ⏳ | Ingesta GGUF zero-copy |
| 4 — Fragmentación | ⏳ | Micro-fragmentación AOT, formato `.bmlgraph` |
| 5 — Runtime | ⏳ | Hot loop RPN, RPC distribuido |

## Desarrollo

```sh
# Tests
cargo test --workspace

# Build release (perfil extremo: lto=fat, panic=abort, codegen-units=1)
cargo build --release

# Clippy
cargo clippy --workspace --all-targets
```

## Referencias

- ["All elementary functions from a single operator"](https://arxiv.org/abs/2603.21852v2) — paper original que define EML.
- `draft.md` — mapa de requerimientos y arquitectura del proyecto.
- `references/eml_paper.md` — texto del paper fuente.
- `openspec/` — specs vivas y plan de ejecución.
