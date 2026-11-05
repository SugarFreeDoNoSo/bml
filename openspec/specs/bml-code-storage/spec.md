# bml-code-storage Specification

## Purpose
Definir códigos canónicos de árboles BML y flujos versionados con límites explícitos, conservando estructura y bits útiles.

## Requirements

### Requirement: Código individual canónico de un árbol BML
El sistema SHALL codificar en preorden `C(1)=0` y
`C(B(a,b))=1 C(a) C(b)`, preservando el orden de los operandos.
El entero sin signo `0` SHALL representar únicamente la hoja `1`; cualquier
otro entero SHALL leerse desde su bit más significativo.

#### Scenario: Almacenamiento y validez estructural
- **WHEN** se almacena y valida un código individual
- **THEN** el almacenamiento
existente en palabras `u64` de menor peso primero SHALL conservarse sin
longitud individual persistida, normalizando las palabras altas nulas.
La validez sintáctica SHALL exigir exactamente un árbol completo; no SHALL
implicar que todos sus argumentos logarítmicos pertenezcan al dominio real.

#### Scenario: Ejemplos de valores y enteros codificados
- **WHEN** se codifican los árboles siguientes
- **THEN** se obtienen exactamente estas secuencias y enteros:

  | Valor | Árbol | Bits | Entero |
  |---:|---|---|---:|
  | 1 | `1` | `0` | 0 |
  | 2 | `B(1,1)` | `100` | 4 |
  | 4 | `B(B(1,1),1)` | `11000` | 24 |
  | 0 | `B(1,B(B(1,1),1))` | `1011000` | 88 |

#### Scenario: Árbol incompleto o símbolos sobrantes
- **WHEN** se decodifican las secuencias individuales `10`, `1000` o `00`
- **THEN** se rechazan por árbol incompleto o bits sobrantes, sin producir un `Bml`.

#### Scenario: Sintaxis válida fuera del dominio real
- **WHEN** se decodifica un árbol completo que aplica `B` a un segundo argumento de valor cero
- **THEN** la decodificación estructural puede tener éxito
- **AND** su evaluación real informa un error de dominio, sin atribuir validez semántica al código.

### Requirement: Igualdad estructural y base identificada
La igualdad y hash de `Bml` SHALL comparar estructura, no valores aproximados.
La equivalencia matemática SHALL tratarse por separado y no prometer una decisión
general. Si un contenedor admite BML y EML SHALL identificar la interpretación
en su tipo o metadatos y rechazar bases desconocidas.

#### Scenario: Árboles distintos de igual valor
- **WHEN** se comparan `1` y `B(1,B(1,1))`, que ambos valen uno bajo BML
- **THEN** la igualdad estructural devuelve falso aunque exista una derivación matemática de igualdad.

### Requirement: Tamaño del árbol y exportación a palabras fijas
El tamaño lógico SHALL ser `2n+1` bits para `n` operadores `B`, incluidas
las `n+1` hojas; la hoja única SHALL contar como un bit aunque su entero sea cero.
Las exportaciones a `u64` y `u128` SHALL admitir como máximo 31 y 63 operadores,
respectivamente, y conservar exactamente el código. Una exportación que exceda
el ancho SHALL devolver un error explícito, nunca truncar, envolver ni provocar
un panic.

#### Scenario: Contabilidad del almacenamiento
- **WHEN** se informa el tamaño de un código
- **THEN** el sistema SHALL distinguir tamaño lógico, palabras asignadas y
memoria observada; representar un entero no SHALL prometer una sola palabra.

#### Scenario: Frontera de u64
- **WHEN** se exportan árboles de 31 y 32 operadores a `u64`
- **THEN** el primero cabe con 63 bits y el segundo falla porque requiere 65 bits.

#### Scenario: Frontera de u128
- **WHEN** se exportan árboles de 63 y 64 operadores a `u128`
- **THEN** el primero cabe con 127 bits y el segundo falla porque requiere 129 bits.

### Requirement: Almacenamiento multipalabra con crecimiento acotado
El sistema SHALL permitir la representación multipalabra existente dentro de
un presupuesto explícito, sin exigir `BigUint` ni una dependencia nueva en
`bml-domain`. Las operaciones SHALL contabilizar el árbol totalmente expandido,
incluidas las copias de subárboles, con aritmética comprobada para tamaños y
reservas. Un resultado que exceda el presupuesto SHALL devolver error y
preservar los operandos; no SHALL sustituirse por un código parcial o aproximado.

#### Scenario: El resultado supera el ancho de sus operandos
- **WHEN** una operación sobre operandos exportables a `u64` produce más de 31 operadores
- **THEN** el resultado puede conservarse en multipalabra si respeta el presupuesto
- **AND** su exportación posterior a `u64` falla explícitamente.

#### Scenario: La expansión excede el presupuesto
- **WHEN** una operación o síntesis necesita más bits que el límite configurado
- **THEN** informa agotamiento de recursos y ambos operandos conservan sus códigos originales.

### Requirement: Flujo versionado que conserva todos los bits útiles
La versión 1 SHALL usar una cabecera de 24 bytes: magic ASCII `BMLS`, versión
`u8=1`, base `u8=2` (BML), dos bytes reservados nulos, `count` como `u64`
little-endian y `bit_length` como `u64` little-endian. El payload comienza en
el byte 24. Las otras bases, versiones y campos reservados no nulos SHALL
rechazarse. Los límites SHALL incluir bits totales, cantidad y profundidad
por árbol (hoja de profundidad cero), tanto al escribir como al leer.

#### Scenario: Contrato del contenedor versionado
- **WHEN** se serializan varios valores
- **THEN** la serialización SHALL usar un formato nuevo y versionado
con cantidad de elementos `count`, longitud útil total `bit_length` y payload.
El contrato de la versión SHALL fijar el ancho y endianness de los campos.
#### Scenario: Orden y relleno del payload
- **WHEN** se construye el payload del flujo
- **THEN** el payload SHALL concatenar los códigos en orden, del bit más significativo
al menos significativo de cada octeto, preservando las hojas iniciales `0`.
Su tamaño SHALL ser exactamente `ceil(bit_length/8)` bytes y los bits bajos
de relleno del último octeto SHALL ser cero.

#### Scenario: Consumo exacto y límites antes de reservar
- **WHEN** se lee un flujo versionado
- **THEN** el lector SHALL consumir exactamente
`count` árboles y `bit_length` bits útiles; SHALL rechazar versiones desconocidas,
truncamiento, bytes extra, campos inconsistentes, padding no canónico y tamaños
que desborden o excedan su presupuesto antes de reservar memoria.

#### Scenario: Hojas iniciales y finales
- **WHEN** se serializa la secuencia de valores `[1,2,1]`
- **THEN** `count=3`, `bit_length=5` y el payload es `01000000` (`0x40`)
- **AND** sus bits útiles `01000` se decodifican como `0`, `100`, `0`.

#### Scenario: Flujo vacío y ceros consecutivos
- **WHEN** se serializa una secuencia vacía o la secuencia `[1,1]`
- **THEN** la primera tiene `count=0`, `bit_length=0` y payload vacío
- **AND** la segunda tiene `count=2`, `bit_length=2` y payload `0x00`.

#### Scenario: Cabecera o relleno inconsistente
- **WHEN** el flujo `[1,2,1]` declara `count=2`, omite el payload o usa `0x41`
- **THEN** se rechaza por cantidad incompatible, truncamiento o padding no canónico.

### Requirement: Un DAG requiere un formato diferente
El formato de árbol SHALL duplicar subárboles repetidos y contabilizar cada
aparición. Una futura representación con referencias compartidas SHALL usar
un formato identificado por separado y reglas propias de validación; no SHALL
interpretarse como este entero ni atribuirle compresión sin mediciones.

#### Scenario: Subárbol repetido
- **WHEN** se codifica `B(t,t)` y `t` contiene `k` operadores
- **THEN** el árbol contiene `1+2k` operadores y `4k+3` bits, sin referencias compartidas.
