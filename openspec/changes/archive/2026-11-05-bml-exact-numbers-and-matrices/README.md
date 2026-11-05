# BML exacto, operadores Rust y matrices precompiladas

**Estado:** especificaciones y plan de acción. Implementación pendiente.
Revisión: 2 de noviembre de 2026.

Esta entrega reúne las trece observaciones de la revisión completa de BML.
El índice de requisitos está en las seis specs; la correspondencia con cada
observación y sus tareas está en la [trazabilidad del diseño](design.md#trazabilidad-de-las-13-observaciones).

## Documentos

- [Propuesta y alcance](proposal.md).
- [Diseño, decisiones y dependencias](design.md).
- [Plan de acción y controles de cierre](tasks.md).

| Spec | Contenido |
|---|---|
| [bml-exact-synthesis](specs/bml-exact-synthesis/spec.md) | Magma, BML/EML, identidades, conversiones exactas, síntesis y evaluación. |
| [bml-rust-operators](specs/bml-rust-operators/spec.md) | `std::ops`, referencias, operandos mixtos y asignaciones fallibles. |
| [bml-code-storage](specs/bml-code-storage/spec.md) | Preorden, igualdad, capacidad, palabras y serialización. |
| [bml-symbolic-matrices](specs/bml-symbolic-matrices/spec.md) | Elementos precompilados, acceso, semántica matricial y reducción generalizada. |
| [bml-simd-backends](specs/bml-simd-backends/spec.md) | Escalar, AVX2/NEON, selección y equivalencia. |
| [bml-performance](specs/bml-performance/spec.md) | OPS, metodología, memoria y evidencia sobre hardware objetivo. |

## Decisiones confirmadas

- `Bml::op` construye la primitiva; la aritmética representa operaciones sobre valores.
- Los operadores aritméticos devuelven `Result`; las variantes prestadas permiten
  expresiones como `(&a + &b)?`.
- Las asignaciones usan `try_add_assign` y `try_mul_assign`, preservando el destino
  si fallan. `AddAssign`/`MulAssign` quedan diferidos explícitamente: sus métodos
  retornan `()` y no admiten propagar `Result`.
- Las matrices conservan elementos BML preparados antes de ejecutar el kernel.
  `A + B` es elemento a elemento, `A * B` es matricial y `hadamard` es explícito.
- La conversión decimal exacta y la conversión binaria exacta de `f64` son
  contratos distintos de la síntesis aproximada existente.
- Las mejoras AVX2/NEON necesitan equivalencia y mediciones propias.

## Secuencia de ejecución

1. Formalizar semántica, dominios y API de operadores.
2. Cerrar codificación, límites y conversiones exactas.
3. Implementar y validar síntesis y operaciones escalares comprobadas.
4. Integrar matrices BML y producto generalizado con orden fijo.
5. Integrar AVX2/NEON sobre kernels concretos.
6. Medir y seleccionar optimizaciones con corrección, memoria y costos registrados.

Los checks de documentación completados no significan que las APIs estén
implementadas. `openspec status` informa disponibilidad de artefactos;
`tasks.md` registra el progreso de implementación y la evidencia pendiente.

## Validación reproducible

Con OpenSpec 1.13.0 disponible mediante npm:

```sh
npx --yes --package @fission-ai/openspec@1.13.0 openspec validate bml-exact-numbers-and-matrices --strict
npx --yes --package @fission-ai/openspec@1.13.0 openspec status --change bml-exact-numbers-and-matrices
```

Para la implementación, usar Rust 1.98.1 según `rust-toolchain.toml` y los
controles de [QUALITY](../../../docs/QUALITY.md), empezando por:

```sh
cargo test -p bml-domain --test bml_encoding --test bml_operations --test bml_conversion --locked
cargo test -p bml --all-features --locked
./scripts/check.sh origin/master
```

La auditoría debe informar cero discrepancias. La evidencia de ejecución y
rendimiento en ARM y x86 se registra por separado de la compilación cruzada.
