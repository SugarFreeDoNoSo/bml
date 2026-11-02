## Why

La base `bml-number-foundation` ya implementa árboles BML, palabras enteras y
síntesis acotada de `f64` con tolerancia. La revisión completa añade trece criterios:
magma, operadores Rust, AVX2/NEON, trabajo vectorizable, matrices precompiladas,
semántica matricial, codificación, capacidad, conversión exacta, estrategias de
síntesis, costo de evaluación, métricas OPS y validación. Este cambio los convierte
en contratos verificables y en un plan de acción dependiente de evidencia.

## What Changes

- Demostrar las identidades para `B(x,y) = 2^x - log2(y)`, con el dominio de cada
  intermedio; distinguir el magma libre de árboles de su interpretación real
  parcial y de EML. Toda representación final usa solamente `1` y `B`.
- Exponer `Add`, `Sub`, `Mul`, `Div` y `Neg` con `Output = Result`, variantes
  prestadas y operandos mixtos definidos. La decisión confirmada es usar
  `try_add_assign`/`try_mul_assign` para asignaciones fallibles y atómicas.
- Añadir contratos distintos `from_decimal_str_exact` y `from_f64_exact`, con
  racional objetivo exacto, política de valores especiales y síntesis acotada.
- Comparar factorización, constantes, potencias de dos, sumas/restas y resultados
  reutilizados por su costo después de expandirlos; registrar presupuesto y
  distinguir mejor candidato encontrado de mínimo demostrado.
- Precisar almacenamiento entero, exportaciones `u64`/`u128`, crecimiento y un
  flujo que preserve ceros iniciales. Conservar el almacenamiento multipalabra.
- Separar exactitud simbólica de evaluación flotante; documentar simplificaciones
  justificadas y errores de dominio/rango, sin certificar equivalencia por tolerancia.
- Añadir matrices de elementos `Bml`, operaciones fallibles y referencia escalar;
  fijar `+` elemento a elemento, `*` matricial, `hadamard`, operaciones con escalar
  y producto generalizado con orden e inicialización explícitos.
- Añadir contratos de backend escalar, AVX2 y NEON: detección de capacidades,
  trabajo vectorizado identificado, colas, errores y equivalencia comprobada.
- Definir construcciones/s, aplicaciones de B/s, expresiones/s, elementos/s,
  productos matriciales/s y la convención de FLOP/s; medir preparación, kernel,
  operación completa, memoria y crecimiento en hardware objetivo.

## Capabilities

### New Capabilities

- `bml-exact-synthesis`: semántica, conversiones exactas, identidades, búsqueda
  acotada y separación entre exactitud y evaluación numérica.
- `bml-rust-operators`: aritmética fallible, referencias, operandos mixtos y
  asignación comprobada, diferenciadas del operador primitivo.
- `bml-code-storage`: límites de palabras, codificación y flujo de árboles BML.
- `bml-symbolic-matrices`: preparación, almacenamiento, acceso, operaciones,
  reducciones, producto generalizado y presupuestos matriciales.
- `bml-simd-backends`: contratos escalares, AVX2/NEON y selección de kernels.
- `bml-performance`: unidades de trabajo, metodología y evidencia de rendimiento.

### Modified Capabilities

Ninguna especificación canónica se sustituye. El nuevo cambio amplía la base
numérica terminada; sus tareas y evidencia anteriores no acreditan estos contratos.

## Impact

Trabajo previsto en `crates/domain::number`, una capacidad explícita de matrices
simbólicas, fachada `crates/bml`, pruebas y benchmarks. Se conservan la API
aproximada `from_f64`, los tensores IEEE `f32/f64` y los pesos/bloques GGUF; no se
convierte automáticamente cada peso de inferencia en un árbol. AVX2 y NEON tendrán
evidencia propia y no amplían por sí solos el perfil AOT Linux x86_64 existente.

La entrega actual guarda las especificaciones y el plan; las tareas de código y
las campañas de hardware siguen pendientes. El orden será semántica/OPS →
representación/conversión → referencia escalar → matrices BML → SIMD → medición.
