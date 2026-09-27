# BML — tensores, GGUF y álgebra EML/BML en Rust

Biblioteca experimental orientada a operar tensores y archivos GGUF, con un
núcleo simbólico para estudiar EML y su variante BML. La reorganización distingue
corrección matemática, representación de datos y rendimiento medido.

## Estado y entrada al proyecto

La primera etapa de la reorganización incorpora términos cerrados/abiertos,
sustitución parcial y evaluación real comprobada en `bml_domain::symbolic`.
Esta etapa corrige los seis contratos auditados del compilador, parser y runtime.
`bml_domain::tensor` incorpora Shape, Tensor, TensorView y kernels CPU de referencia.
La extracción de GGUF y la fachada pública siguen pendientes; véanse los límites
y controles de calidad en [QUALITY.md](docs/QUALITY.md).

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

## Tensores y kernels CPU

```rust
use bml_domain::tensor::{matmul, Shape, Tensor, TensorError};

fn main() -> Result<(), TensorError> {
    let a = Tensor::from_vec(Shape::new([2, 3])?, vec![1., 2., 3., 4., 5., 6.])?;
    let b = Tensor::from_vec(Shape::new([3, 1])?, vec![2., -1., 0.5])?;
    let result = matmul(&a.view(), &b.view())?;
    assert_eq!(result.shape().dims(), &[2, 1]);
    assert_eq!(result.as_slice(), &[1.5, 6.0]);
    Ok(())
}
```

`add`, `mul`, `dot` y `matmul` conservan el dtype `f32`/`f64` y usan aritmética
IEEE nativa. `apply_basis` aplica la primitiva EML/BML elegida, valida en `f64`
y comprueba el rango al devolver `f32`. No convierte matmul en un árbol escalar.
Formas, límites y longitudes se validan; las vistas prestan los valores y clonan
solo metadatos. Broadcasting y strides no canónicos producen errores explícitos.

## Organización actual y destino

| Crate actual | Papel actual | Dirección de la reorganización |
|---|---|---|
| `domain` | AST anterior, SoA, transformer, symbolic y tensor | Núcleo independiente; futura extracción a bml-core |
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
cargo test -p bml-domain --test tensor_contracts --locked
cargo test -p bml-domain --doc --locked

# Regresión de los seis contratos: debe terminar sin discrepancias
cargo run -p bml-tests --example contract_audit --locked

# Suite ordinaria, con instrumentación y modelos externos optativos
cargo test --workspace --locked

cargo build --release --locked
./scripts/check.sh origin/master
```

Las pruebas instrumentadas requieren capacidades del entorno y se marcan como
ignored. La revisión del 20 de septiembre documenta el estado histórico; el
estado de esta etapa se registra en `quality/evidence` y los checks de Actions.

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

## Roadmap y calidad

[Roadmap](docs/ROADMAP.md) · [Controles y comandos](docs/QUALITY.md). Los resultados verificados de esta etapa se registran en `quality/evidence` y en los artifacts de Actions.
