## ADDED Requirements

### Requirement: Operador BML base
El sistema SHALL definir un único operador matemático `bml(a, b)` que actúa como operador fundamental del dominio BML, análogo continuo del NAND lógico, con completitud funcional.

#### Scenario: Evaluación del operador
- **WHEN** se invoca `bml(a, b)` con dos valores `f64`
- **THEN** se retorna el resultado definido por la formulación del draft (referencia ArXiv 2603.21852v2 adaptada a la base BML)

#### Scenario: Inmutabilidad del orden de operandos
- **WHEN** se intercambia el orden de los operandos `bml(a, b)` vs `bml(b, a)` con `a != b`
- **THEN** los resultados son distintos, reflejando la propiedad de magma no asociativo

#### Scenario: Casos límite
- **WHEN** se evalúa `bml` con `0`, `1`, `NaN` o `Inf` como operandos
- **THEN** el resultado es determinista y consistente con la definición matemática (no paniquea)

### Requirement: Gramática del AST BML
El sistema SHALL construir el Árbol de Sintaxis Abstracta (AST) bajo una gramática estricta donde cada nodo es `BML(left, right)` o la constante `Const(1)`, y nada más.

#### Scenario: Nodo BML
- **WHEN** se construye un nodo del AST
- **THEN** es `BML(left, right)` con dos sub-árboles, o `Const(1)`

#### Scenario: Ausencia de operaciones primitivas
- **WHEN** se inspecciona el AST
- **THEN** no existen nodos `+`, `-`, `*`, `/`, `pow` ni otras operaciones estándar; solo `BML` y `Const(1)`

### Requirement: Layout de memoria SoA alineado a 64 bytes
El sistema SHALL almacenar los campos del grafo de nodos como Struct of Arrays (SoA) con `#[repr(align(64))]`, prohibiendo el patrón Array of Structures (AoS).

#### Scenario: Alineación de línea de caché
- **WHEN** se inspecciona el layout de memoria de los campos del grafo
- **THEN** cada campo relevante está alineado a 64 bytes (línea de caché típica)

#### Scenario: Acceso por campo
- **WHEN** el runtime solo necesita un campo (ej. operandos) para una evaluación
- **THEN** la CPU carga en caché solo los bytes de ese campo, no toda la estructura del nodo

### Requirement: BMLTransformer (mapper de operaciones estándar)
El sistema SHALL proveer un `BMLTransformer` capaz de tomar operaciones estándar (`+`, `-`, `*`, `/`, `pow`) y traducirlas puramente a la gramática recursiva BML usando solo el operador BML y la constante 1.

#### Scenario: Suma
- **WHEN** se transforma `a + b`
- **THEN** se produce un AST BML equivalente cuya evaluación coincide con `a + b` para todo `a, b` en el rango probado

#### Scenario: Resta, multiplicación, división, potencia
- **WHEN** se transforma `a - b`, `a * b`, `a / b`, o `a^b`
- **THEN** cada una se traduce a un AST BML equivalente cuya evaluación coincide con la operación original

#### Scenario: Composición de operaciones
- **WHEN** se transforma una fórmula compuesta (ej. `(a + b) * c`)
- **THEN** el AST BML resultante preserva la semántica de la fórmula original

### Requirement: Cero dependencias en bml-domain
El crate `bml-domain` SHALL tener cero dependencias externas; solo código Rust puro de la stdlib.

#### Scenario: Verificación de dependencias
- **WHEN** se ejecuta `cargo tree -p bml-domain`
- **THEN** no se listan dependencias externas más allá de la stdlib y dev-dependencies de test
