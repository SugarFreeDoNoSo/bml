## ADDED Requirements

### Requirement: Primitiva BML distinta de aritmética derivada
`Bml::op` SHALL construir el término ordenado de la primitiva `B(x,y)` sin
evaluarlo. `Add`, `Sub`, `Mul`, `Div` y `Neg` SHALL representar respectivamente
las operaciones aritméticas derivadas, con reglas y dominios justificados en base
2; no SHALL operar aritméticamente sobre los bits codificados. Los métodos
históricos de construcción SHALL conservar su contrato estructural, salvo una
migración explícita y compatible documentada. Las nuevas APIs comprobadas SHALL
distinguir construcción, dominio no justificado y evaluación.

#### Scenario: Construcción de primitiva frente a suma
- **WHEN** se construyen `one.op(&one)` y `(&one + &one)`
- **THEN** ambos tienen semántica documentada aunque no tengan que compartir código
- **AND** la primera invocación construye `B(1,1)` y la segunda aplica la regla de suma.

### Requirement: Operadores fallibles y referencias
`Add`, `Sub`, `Mul`, `Div` y `Neg` SHALL devolver `Result` con errores tipados;
SHALL ofrecer variantes con operandos prestados sin requerir al usuario clonar
los originales. La documentación SHALL indicar los casos admitidos, presupuesto
y cómo usar `?`. Los operadores SHALL delegar en APIs comprobadas, sin ocultar
errores ni producir resultados parciales; cada regla SHALL justificar sus
dominios. No SHALL afirmarse que evaluar aproximadamente decide el signo exacto
de un árbol arbitrario. Si el dominio no puede justificarse SHALL informarse
explícitamente; la división por cero demostrada SHALL rechazarse.

#### Scenario: Suma prestada con resultado fallible
- **WHEN** se usa `(&a + &b)?` para valores admitidos dentro del presupuesto
- **THEN** el resultado es un `Bml` que representa la suma y `a` y `b` permanecen utilizables.

#### Scenario: División o dominio no admisible
- **WHEN** se intenta dividir por cero demostrado o no se puede justificar un dominio requerido
- **THEN** se devuelve el error tipado correspondiente sin asumir éxito por aproximación flotante.

### Requirement: Asignación fallible transaccional
La API SHALL ofrecer `try_add_assign` y `try_mul_assign` que devuelven `Result` y
solo reemplazan el destino tras construir el resultado completo. SHALL preservar
destino y operando si ocurre error. `AddAssign` y `MulAssign` sobre el tipo normal
SHALL quedar diferidos hasta acordar una política separada, porque sus métodos
retornan `()` y no pueden devolver `Result`; no SHALL introducir panic ni
descartar errores silenciosamente para simular estas asignaciones.

#### Scenario: Presupuesto insuficiente al asignar
- **WHEN** `try_add_assign` o `try_mul_assign` excede el presupuesto o encuentra dominio no justificado
- **THEN** devuelve error y el destino y el operando conservan exactamente su estructura original.

### Requirement: Operandos mixtos explícitos
La API SHALL definir la conversión numérica de operandos enteros y `f64` en las
operaciones mixtas admitidas: los enteros representan su valor matemático y los
`f64` su valor binario exacto bajo el contrato de síntesis exacta. Toda síntesis
fallida SHALL propagarse; `from_u64` SHALL seguir interpretando un código, no un
entero numérico. Las variantes simétricas admitidas SHALL documentarse y probarse.
En matrices, el escalar SHALL convertirse una sola vez antes del kernel, con su
costo de preparación contabilizado por separado.

#### Scenario: Entero como valor frente a código
- **WHEN** se suma el entero numérico `4` a un `Bml` mediante la API mixta
- **THEN** el operando numérico representa cuatro, no el código entero `4` que representa dos
- **AND** un fallo de síntesis se devuelve como error.
