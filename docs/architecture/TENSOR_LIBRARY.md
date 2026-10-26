# Biblioteca de tensores, GGUF y términos EML/BML

Estado: núcleo simbólico y tipos/kernels tensoriales CPU implementados en domain;
reader/codecs en bml-gguf; integración tensorial GGUF y fachada pendientes.
Actualización GGUF: 2026-10-22 UTC.
Fecha: 2026-09-20. Base revisada: `626f12f2d4a55817a31ec66562dff0eb3f81a2fa`.

## Objetivo y alcance inicial

La biblioteca debe leer tensores GGUF, conservar su forma y representación,
ejecutar operaciones numéricamente verificadas y permitir experimentar con
EML/BML. El primer entregable es una API local CPU; inferencia LLM, tokenización,
HTTP y distribución son consumidores opcionales con validación propia.

El artículo [All elementary functions from a single binary operator](https://arxiv.org/abs/2603.21852v2)
fundamenta una construcción expresiva para EML. Su evaluación completa contempla
complejos, ramas logarítmicas y casos con reales extendidos. Esto no establece
una aceleración de GEMM ni una garantía de estabilidad en `f64`.
La variante BML y cada reescritura necesitan contratos independientes.

## 1. Base algebraica propuesta por el usuario

Sean `T = One | Op(T,T)` y `T(X) = Closed(T) | Var(X) | Op(T(X),T(X))`.
Son estructuras sintácticas: `Op` siempre construye otro término. En ese sentido
`T` es el magma libre con un generador. La interpretación real de `Op` es parcial,
porque el logaritmo exige un segundo operando positivo. Cierre sintáctico y
existencia del resultado numérico son propiedades diferentes.

La API implementada en `crates/domain/src/symbolic.rs` es:

| Tipo/operación | Contrato |
|---|---|
| `ClosedTerm` | Únicamente `One` y `Op`; no contiene `f64`, variables o pesos |
| `OpenTerm` | Variables por nombre, subárboles cerrados y aplicaciones |
| `bind(&env)` | Sustitución parcial de variables por términos cerrados; conserva las desconocidas |
| `free_variables()` | Conjunto ordenado de nombres libres |
| `collapse()` | Cierra la estructura o devuelve el nombre de una variable libre |
| `evaluate_real(Basis)` | Interpreta todo el árbol como EML o BML, con errores explícitos |
| `Basis::apply_real(x,y)` | Primitiva numérica real comprobada, sin construir un árbol |

Las identidades de sustitución que deben mantenerse son:

- `bind(t, {}) = t`.
- `FV(bind(t, env)) = FV(t) \ keys(env)`.
- `bind(bind(t, env), env) = bind(t, env)` para entornos cerrados.
- Las sustituciones con dominios disjuntos conmutan.
- `collapse(Closed(t)) = Ok(t)`.
- `collapse(t)` tiene éxito si y solo si `FV(t)` está vacío.

Si dos entornos asignan un mismo nombre, la primera sustitución lo elimina;
no se debe afirmar conmutatividad en ese caso. `bind` preserva la construcción
binaria y el orden de operandos; `collapse` no simplifica ni evalúa.

`Basis` es una interpretación global y explícita. Por ejemplo, el árbol
`Op(One,One)` representa `e` con EML y `2` con BML. Un mismo árbol no denota
un mismo número bajo ambas bases. Un futuro artefacto compilado debe guardar
la base junto con la política numérica; una conversión de base necesita una
transformación demostrada, no solo cambiar una etiqueta.

Los términos de homeostasis/plasticidad pueden servir como analogía. Los contratos
implementables son sustitución, variables libres y cierre; no requieren una
hipótesis física o psicológica.

## 2. Evaluación y síntesis son operaciones diferentes

Se evita `From<ClosedTerm> for f64`: la evaluación puede fallar y no necesita
consumir el árbol. `evaluate_real(&self, basis) -> Result<f64, EvalError>` rechaza
entradas no finitas, argumentos logarítmicos no positivos, overflow y exponenciales
que se redondean a cero. Permite redondeo ordinario y subnormales no nulos; no
certifica una cota global del error o ausencia de cancelación.

Esta política **no reproduce** derivaciones que necesitan `log(0) = -inf`.
Un evaluador `ExtendedReal` o complejo futuro deberá tener una política explícita
y sus propios tests. No se usa para reemplazar silenciosamente el runtime actual.

`f64 -> ClosedTerm` tampoco es una inversa general: la evaluación no es inyectiva,
el redondeo pierde información y reconocer `1`/`e` no constituye un algoritmo de
síntesis. El conjunto de árboles finitos es numerable; no representa todos los
reales arbitrarios mediante términos finitos. Para `f64`, que es un conjunto
finito, aún hace falta una construcción y una política de equivalencia.

Si se desarrolla síntesis simbólica, su API será algo como
`synthesize(target, basis, budget) -> Candidate { term, error, cost }` con un
estado explícito de “sin candidato dentro del presupuesto”. Una búsqueda con
Adam no garantiza representación exacta ni terminación con éxito. La igualdad
estructural de árboles tampoco equivale a igualdad de funciones.

## 3. Separar tres representaciones

| Representación | Uso | Almacenamiento |
|---|---|---|
| Términos cerrados/abiertos | Construcción, sustitución y leyes simbólicas | `Box` para fórmulas pequeñas; arena para grafos grandes |
| Tensor | Datos numéricos, dimensiones, strides, dtype y operaciones | Buffer contiguo, vista prestada o bloques cuantizados |
| Plan de ejecución | Operaciones validadas, constantes, buffers y dependencias | IDs/arena, instrucciones de tensor y kernels |

Un peso de un GGUF permanece como dato; no se convierte en un árbol por elemento.
`Const(f64)` y los parámetros numéricos son extensiones del IR de ejecución,
no constructores del álgebra pura. Una fórmula abierta puede parametrizar una
operación aplicada sobre un tensor, con broadcasting y dtype definidos aparte.

El plan debe retener `MatMul`, reducciones y operaciones elementwise a alto nivel.
Expandir cada suma/producto a exponenciales y logaritmos elimina oportunidades de
usar kernels eficientes y puede introducir overflow, cancelación o dominios inválidos.
La expansión pura EML/BML es una modalidad experimental que se mide explícitamente.

## 4. Organización incremental, sin multiplicar crates prematuramente

| Destino propuesto | Contenido | Dependencias permitidas |
|---|---|---|
| `crates/core` (`bml-core`) | Términos, evaluación, `Tensor`, `TensorView`, `Shape`, errores, kernels CPU de referencia | Sin parser, HTTP o runtime distribuido |
| `crates/gguf` (`bml-gguf`) | Reader, metadata, rangos exactos, tipos GGML y codecs Q4_0/Q8_0 | `bml-core`, mmap opcional |
| `crates/bml` (`bml`) | Fachada pública y ejemplos tensor + GGUF | `bml-core`; `bml-gguf` mediante feature |
| `crates/compiler` / `crates/runtime` | Compilación y ejecución experimental | Núcleo/IR compartido, sin lector de modelos dentro del hot loop |
| `apps/cli`, `apps/api`, `apps/worker` | Adaptadores externos | Fachada y capacidades opcionales |
| `experiments/llm` | Modelo, tokenizer, KV cache, sampler y evaluación LLM | Fachada y plan experimental |
| `crates/tests` | Integración entre crates y soporte de fixtures | Solo dependencias de desarrollo |

El núcleo está en `bml-domain::symbolic` y `bml-domain::tensor` para mantener compatible
el workspace actual. La extracción de reader/codecs a `crates/gguf` ya está implementada; `parser`
conserva los imports y errores IO como adaptador. El resto sigue siendo el destino
de la migración, no rutas ya existentes. Extraer un backend o un crate de IR adicional
solo cuando exista un consumidor independiente que lo justifique.

Al migrar, `core::ir` puede alojar los tipos neutrales del plan para que compiler y
runtime dependan de ellos. Mover de `compiler` a `gguf` la decodificación, y a
`experiments/llm` la inferencia; retirar dependencias inversas de producción.
Las pruebas entre compiler/runtime viven en `crates/tests`.

En Rust, usar structs, enums, composición y traits pequeños en límites reales.
Un trait `Backend` tiene sentido al introducir una segunda implementación;
evitar interfaces dinámicas por cada escalar y jerarquías de clases en kernels.
La arena/SoA es una decisión de ejecución medible, independiente del AST público.

## 5. Contratos del MVP tensor/GGUF

- CPU, buffers poseídos `f32` y `f64`, vistas inmutables y forma validada.
- Rank cero representa un escalar (un elemento); una dimensión cero produce
  cero elementos. Productos de dimensiones y offsets usan aritmética comprobada.
- Primera versión de kernels: layout contiguo, `add`, `mul`, `dot`, `matmul`
  bidimensional y EML/BML elementwise de formas iguales. Broadcasting y vistas
  no contiguas devuelven un error explícito hasta tener implementación validada.
- `reshape` preserva el número de elementos. Para tensores strided futuros,
  transposición modifica shape/strides y no presupone contigüidad.
- `GgufTensorView<'a>` toma prestado el reader: los bytes no sobreviven al mmap.
  El mmap evita una copia completa a un buffer de usuario; no significa “sin RAM”.
- La primera compatibilidad garantizada es GGUF v3 little-endian con F32, F16,
  Q4_0 y Q8_0. Versión, endian y dtype no soportados devuelven errores distintos.
- El adaptador preserva las dimensiones GGML (`ne[0]` contigua) y explicita su
  conversión a shapes row-major de Rust. Una matriz rectangular asimétrica prueba
  el orden; una matriz cuadrada simétrica no detecta una transposición accidental.
- `general.alignment`, offsets, tamaños de bloques y longitud exacta por tensor
  se verifican antes de entregar vistas. Lecturas truncadas se rechazan.
- Cuantización, dequantización y conversiones de dtype son operaciones visibles;
  el uso de memoria y el error se reportan aparte.

## 6. Ruta de migración con criterios de salida

### API tensorial disponible

`Shape::new` valida el producto de dimensiones. Las formas vacías aceptan cualquier
posición de la dimensión cero y usan strides canónicos todos cero, sin elementos
direccionables. `Shape::offset` valida todos los ejes antes de calcular el offset
con aritmética comprobada; así una forma vacía con ejes enormes no desborda antes
de encontrar el eje vacío.

`Tensor::from_vec` toma posesión de un buffer de longitud exacta. `view` y
`TensorView::from_slice` prestan valores; las dimensiones/strides son metadatos
poseídos. `Tensor::reshape` conserva el buffer y deja el objeto intacto al fallar;
la variante de una vista devuelve nuevos metadatos sobre el mismo slice.
`from_strided` solo importa los strides canónicos (también en ejes singleton y
formas vacías); no implica soporte de vistas arbitrariamente strided.

Los kernels `add`/`mul` exigen formas idénticas, `dot` rank 1 y `matmul` rank 2
con dimensiones internas iguales. Dot y cada celda de matmul acumulan de izquierda
a derecha en el dtype original, con multiplicación y suma separadas. La reducción
vacía vale +0. Los kernels nativos preservan la semántica IEEE, incluidos NaN/inf;
no se les atribuye el contrato real estricto del evaluador simbólico.

`apply_basis` evalúa la primitiva seleccionada mediante `Basis::apply_real` en f64.
Con entradas f32 las ensancha exactamente y redondea una vez el resultado final;
devuelve `OutputRange` si este desborda f32 o un resultado no nulo se redondea a
cero. Se permiten resultados cero exactos y subnormales no nulos. Los errores
incluyen el primer índice plano inválido; no se entrega una salida parcial.
Esta política no promete una cota global de error ni expande kernels a EML/BML.

Los buffers de salida comprueban tamaño y reserva. La implementación no usa unsafe,
dependencias externas nuevas, paralelismo, SIMD ni un backend dinámico. Es la
referencia para futuras comparaciones, sin afirmaciones de aceleración.

### Etapas

1. **Contratos y evidencia (este cambio):** álgebra simbólica, diagnóstico
   reproducible, matriz de tests y clasificación correcta de benchmarks.
2. **Corrección P0:** reparar reutilización de nodos RPN, propagación del pool,
   Q4_0, alignment y truncamiento. Cada reproducción debe convertirse en un test
   que falle antes del arreglo y pase después. Definir dominios de las reescrituras.
3. **MVP local:** tipos tensor/core + reader/codec GGUF; operar matrices pequeñas
   con valores conocidos, incluidos ceros, negativos y fracciones.
4. **Optimización:** comparar kernels nativos, EML y BML con entradas idénticas;
   promover una ruta solo con error admisible y mejora medida. Fusión, tiling,
   SIMD y cálculo sobre bloques cuantizados son hipótesis independientes.
5. **Inferencia/distribución:** avanzar cuando el cálculo local y el formato sean
   correctos; medir logits, calidad y ejecución completa antes de publicar tokens/s.

Ver [tests](../TESTING.md) y el cambio
[`tensor-library-foundation`](../../openspec/changes/tensor-library-foundation/proposal.md).
