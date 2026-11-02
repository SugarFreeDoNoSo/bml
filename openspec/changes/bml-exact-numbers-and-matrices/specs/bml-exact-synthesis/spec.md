## ADDED Requirements

### Requirement: Semántica BML y dominio de las identidades
La representación final SHALL usar exclusivamente `T ::= 1 | B(T,T)` con
`B(x,y)=2^x-log2(y)`. Los árboles finitos y el constructor binario ordenado SHALL
describirse como magma libre generado por `1`; ninguna identidad, asociatividad o
conmutatividad se presume. La interpretación real SHALL describirse como parcial,
no como magma total sobre los reales. Cada regla de construcción o simplificación SHALL incluir
una derivación para base 2 y las condiciones de dominio de todos sus intermedios.
Un segundo argumento no positivo SHALL ser inválido en evaluación real; la
validez de la gramática no SHALL certificar dominio ni evaluabilidad en `f64`.

#### Scenario: Árbol válido fuera del dominio real
- **WHEN** un árbol sintácticamente válido contiene una operación con segundo argumento cero o negativo
- **THEN** el análisis estructural puede aceptarlo, pero la evaluación real informa un error de dominio
- **AND** no se usa ese árbol para certificar una conversión exacta válida.

#### Scenario: El generador no es identidad del magma
- **WHEN** se compara estructuralmente `B(1,1)` con `1`
- **THEN** son árboles distintos y `1` no se declara identidad del constructor.

#### Scenario: Interpretación EML y BML separadas
- **WHEN** el mismo árbol se interpreta con EML `exp(x)-ln(y)` y BML `exp2(x)-log2(y)`
- **THEN** cada base conserva sus propias identidades, dominios e interpretación
- **AND** un formato compartido identifica la base elegida en lugar de inferirla de los bits.

### Requirement: Conversión decimal exacta
`from_decimal_str_exact` SHALL aceptar inicialmente `[+-]?[0-9]+(\.[0-9]+)?`,
contar los `d` dígitos fraccionarios incluyendo ceros iniciales y formar el
racional con denominador `10^d`, reducido por MCD. El parseo SHALL usar
aritmética entera exacta, sin pasar por `f64`. Sintaxis no admitida y límites
de entrada/recursos SHALL producir errores explícitos antes de una expansión
no acotada. Un éxito SHALL representar exactamente ese racional.

#### Scenario: Fracción decimal reducida
- **WHEN** se solicita convertir `"12.375"`
- **THEN** el objetivo exacto es `12375/1000 = 99/8`
- **AND** todo resultado exitoso tiene una justificación exacta de ese valor.

#### Scenario: Ceros iniciales y gramática
- **WHEN** se analizan `"0.0010"`, `"-0.125"` y `"-0.000"`
- **THEN** los objetivos reducidos son `1/1000`, `-1/8` y `0`
- **AND** `"12,375"`, `"1e-3"`, entradas vacías y entradas sobre el límite de dígitos se rechazan explícitamente.

### Requirement: Conversión exacta del valor almacenado en f64
`from_f64_exact` SHALL descomponer `to_bits()` en signo, significando y exponente,
obteniendo exactamente `(-1)^s m 2^q`, incluidos subnormales. SHALL normalizar
ambos ceros IEEE al cero matemático y rechazar NaN e infinitos. SHALL conservar
el contrato de la API aproximada `from_f64` existente y distinguirlo en la
documentación. Ningún éxito de las APIs exactas SHALL depender de tolerancias.

#### Scenario: Decimal y float con el mismo texto visible
- **WHEN** se comparan los objetivos de `from_decimal_str_exact("0.1")` y `from_f64_exact(0.1)`
- **THEN** son respectivamente `1/10` y `3602879701896397/36028797018963968`
- **AND** no se sustituyen entre sí aunque ambos se impriman como `0.1`.

#### Scenario: Subnormales, ceros y valores no finitos
- **WHEN** se convierte el menor subnormal positivo o negativo
- **THEN** el objetivo es respectivamente `1/2^1074` o `-1/2^1074`
- **AND** el parseo exacto no promete que la síntesis termine dentro del presupuesto disponible.

#### Scenario: Política de ceros y valores no finitos
- **WHEN** la entrada es `+0.0`, `-0.0`, NaN o un infinito
- **THEN** los dos ceros se normalizan y los valores no finitos se rechazan; no se promete preservar sus patrones IEEE.

### Requirement: Éxito exacto comprobable y fallos acotados
La síntesis exacta SHALL producir un término con una derivación comprobable
desde el racional objetivo y reglas probadas. Una comparación flotante o un
roundtrip no SHALL ser prueba de equivalencia. SHALL distinguir entrada inválida,
dominio no justificado, límite de recursos y búsqueda sin solución certificada;
no SHALL devolver un aproximado como éxito exacto ni prometer éxito universal.

#### Scenario: Presupuesto insuficiente
- **WHEN** el objetivo es un racional válido pero su síntesis excede los límites o no demuestra exactitud
- **THEN** devuelve un error específico, conserva la causa y no entrega una expresión aproximada como exacta.

### Requirement: Operaciones puras con signos y casos especiales
`add`, `mul`, `div` y `neg` SHALL expandirse completamente a `B` y `1`, sin
constantes externas ni resultados calculados mediante floats. Sus contratos
SHALL precisar signo, cero y dominio de los intermedios; división por cero y
dominios no justificados SHALL producir un error explícito en la API comprobada.
Una regla restringida a positivos no SHALL presentarse como válida para todo real.

#### Scenario: Multiplicación y división con signo
- **WHEN** se solicita multiplicar por cero, multiplicar operandos de signos distintos o dividir por cero
- **THEN** los casos admitidos usan expansiones con derivaciones y dominios documentados
- **AND** los casos inválidos se rechazan sin cambiar los operandos.

### Requirement: Búsqueda y criterio de optimalidad explícitos
El sintetizador SHALL comparar, cuando sean aplicables, factorización, catálogo
de constantes, potencias de dos y expresiones cercanas, sumas/restas y resultados
previamente compilados. SHALL limitar búsqueda y factorización y contar costos
después de expandir todas las macros y duplicaciones del árbol serializado.
El criterio principal SHALL minimizar operadores `B`; profundidad y orden
lexicográfico del código SHALL usarse para desempatar de forma determinista. Tiempo de
evaluación y estabilidad SHALL reportarse como objetivos distintos. Cada
resultado SHALL registrar estrategia, costo, presupuesto consumido y motivo de
terminación; minimalidad global SHALL requerir una prueba explícita.

#### Scenario: Constante 15 y costo expandido
- **WHEN** se considera `15=B(4,2)`, con `4=B(B(1,1),1)` y `2=B(1,1)`
- **THEN** su costo es cuatro operadores y nueve bits de árbol, incluyendo las constantes expandidas
- **AND** no se declara óptimo sin una prueba de minimalidad.

#### Scenario: Una heurística encuentra un candidato
- **WHEN** la búsqueda termina por su presupuesto después de hallar una construcción exacta
- **THEN** la identifica como mejor encontrada entre los candidatos explorados bajo ese presupuesto
- **AND** no afirma que factorización, menor profundidad o menor tiempo sean equivalentes a mínimo tamaño.

### Requirement: Exactitud simbólica independiente de evaluación numérica
La API SHALL distinguir estructura, dominio matemático, exactitud y evaluación
numérica. El evaluador directo `f64` SHALL conservar sus errores de dominio,
overflow y underflow. Simplificaciones simbólicas o evaluación con mayor
precisión SHALL declarar reglas, precisión y límites propios; igualdad
aproximada no SHALL autorizar una reescritura exacta.

#### Scenario: Valor pequeño con intermedio enorme
- **WHEN** se considera `2049=B(12,B(2047,1))`
- **THEN** la derivación real justifica `4096-log2(2^2047)=2049`
- **AND** la evaluación directa `f64` informa overflow en `2^2047`, sin refutar por ello la identidad simbólica
- **AND** una ruta simplificada solo evita ese intermedio mediante una identidad probada con dominio válido.
