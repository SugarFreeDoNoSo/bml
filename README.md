# BML — tensores, GGUF y álgebra EML/BML en Rust

Biblioteca experimental orientada a operar tensores y archivos GGUF, con un
núcleo simbólico para estudiar EML y su variante BML. La reorganización distingue
corrección matemática, representación de datos y rendimiento medido.

## Estado y entrada al proyecto

La primera etapa de la reorganización incorpora términos cerrados/abiertos,
sustitución parcial y evaluación real comprobada en `bml_domain::symbolic`.
El compilador, el parser y el runtime anteriores siguen disponibles y tienen
fallos conocidos; la migración tensorial está especificada, aún pendiente.

- [Arquitectura, contratos y migración](docs/architecture/TENSOR_LIBRARY.md).
- [Revisión reproducible: seis discrepancias funcionales](docs/architecture/REVIEW_2026-09-20.md).
- [Estrategia de tests y benchmarks](docs/TESTING.md).
- [Cambio OpenSpec y tareas pendientes](openspec/changes/tensor-library-foundation/proposal.md).

## Operadores y alcance matemático

```text
EML(x, y) = exp(x)  - ln(y)
BML(x, y) = exp2(x) - log2(y)
```

El artículo [All elementary functions from a single binary operator](https://arxiv.org/abs/2603.21852v2)
estudia EML con la constante `1` como base expresiva para funciones de calculadora.
Su evaluación completa contempla complejos y casos con reales extendidos.
La implementación real finita no cubre automáticamente esa semántica completa.
BML es una variante del proyecto: sus identidades y ventajas necesitan validación
propia. Expresividad no implica estabilidad numérica ni aceleración de tensores.

El árbol puro es `ClosedTerm = One | Op(ClosedTerm, ClosedTerm)`. La extensión
`OpenTerm` incorpora variables y términos cerrados. La interpretación se selecciona
con `Basis::Eml` o `Basis::Bml`; por ejemplo, `Op(One,One)` denota `e` o `2`.

```rust
use bml_domain::symbolic::{Basis, ClosedTerm, OpenTerm};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let formula = OpenTerm::op(
        OpenTerm::Var("x".into()),
        ClosedTerm::One.into(),
    );
    let env = HashMap::from([("x".to_owned(), ClosedTerm::One)]);
    let closed = formula.bind(&env).collapse()?;
    assert_eq!(closed.evaluate_real(Basis::Bml)?, 2.0);
    Ok(())
}
```

`collapse` comprueba ausencia de variables; `evaluate_real` comprueba dominio y
rango de los cálculos. Un término cerrado puede fallar al evaluarse. La búsqueda
de un árbol que represente un `f64` sería síntesis simbólica, no una conversión
inversa garantizada.

## Organización actual y destino

| Crate actual | Papel actual | Dirección de la reorganización |
|---|---|---|
| `domain` | AST anterior, SoA, transformer y nuevo módulo symbolic | Núcleo matemático y tipos tensoriales |
| `parser` | Lectura GGUF con mmap | Reader y codecs con rangos/errores precisos |
| `compiler` | DAG/RPN, dequantización, inferencia y distribución | Compilación; extraer lectura, modelo y servicios |
| `runtime` | Intérprete, buffers, scheduling y red | Ejecución experimental opcional |
| `cli`, `api`, `worker` | Aplicaciones | Consumidores de la biblioteca |
| `bench` | Microbenchmarks y extrapolaciones históricas | Comparaciones de trabajo equivalente |
| `tests` | Integración registrada mediante Cargo | Contratos entre crates y fixtures |

La arquitectura destino comienza con `bml-core`, `bml-gguf` y una fachada `bml`.
Los pesos permanecen como buffers o bloques cuantizados. Los árboles simbólicos
representan fórmulas; no se requiere un árbol por elemento del tensor. Los kernels
mantienen operaciones como matmul y reducciones a alto nivel.

## Desarrollo y validación

```sh
# Contratos del nuevo núcleo simbólico
cargo test -p bml-domain --test symbolic_contracts --locked
cargo test -p bml-domain --doc --locked

# Diagnóstico: salida no cero mientras persistan los fallos previos
cargo run -p bml-tests --example contract_audit --locked

# Suite completa; consultar limitaciones conocidas en la revisión
cargo test --workspace --locked

cargo build --release --locked
cargo clippy --workspace --all-targets --locked
```

Las pruebas instrumentadas requieren capacidades del entorno. La revisión
registra un fallo de ptrace/strace y un doctest previo; esta rama no declara que
la suite global esté completamente aprobada.

## Rendimiento

Los [reportes históricos](docs/benchmarks/REPORT.md) incluyen tokens/s extrapolados
de un programa sintético. No acreditan inferencia equivalente a llama.cpp.
El nuevo protocolo exige valores correctos, entradas y trabajo equivalentes,
contexto completo, error numérico, tiempo, memoria y asignaciones observadas.

La huella de instrucciones del intérprete, el bytecode y los datos tensoriales
se miden por separado. La uniformidad de un operador o el uso de `exp2`/`log2`
no garantiza una instrucción hardware única ni mayor velocidad.

## Referencias

- [Artículo EML, versión 2](https://arxiv.org/abs/2603.21852v2).
- [Especificación GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
- `references/eml_paper.md`: copia de referencia del artículo.
- `draft.md` y propuestas OpenSpec anteriores: historia del diseño experimental.
