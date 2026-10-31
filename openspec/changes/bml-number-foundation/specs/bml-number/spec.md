## ADDED Requirements

### Requirement: Representación booleana canónica
El sistema SHALL representar un `Bml` cerrado por el recorrido prefijo de la
gramática `One = 0`, `Op(left,right) = 1 left right`, con operandos ordenados.
Las palabras de un entero sin signo SHALL almacenarse de menor peso primero y
sin longitud persistida. Las palabras altas nulas SHALL normalizarse; las
cadenas binarias no canónicas SHALL rechazarse.

#### Scenario: Constante y operadores
- **WHEN** se interpretan `0`, `100` y `11000`
- **THEN** representan respectivamente `1`, `Op(1,1)=2` y `Op(Op(1,1),1)=4`.

#### Scenario: Gramática inválida
- **WHEN** los bits no contienen exactamente un árbol completo
- **THEN** se informa un error y no se construye un `Bml` inválido.

### Requirement: Operaciones mediante reglas
Las operaciones SHALL construir términos usando reglas de BML base 2 sin calcular
el resultado con floats. Las identidades SHALL tener dominios documentados;
la igualdad de términos SHALL ser estructural.

#### Scenario: Suma como composición
- **WHEN** se suman dos términos aceptados por la regla de suma
- **THEN** se devuelve un término BML sin constantes numéricas externas.

### Requirement: Conversión comprobada y acotada
La evaluación SHALL comprobar dominio y rango. La síntesis desde `f64` SHALL
tener límites de recursos, comprobar la precisión solicitada e informar errores;
no SHALL depender de un pool de constantes ni garantizar éxito universal.

#### Scenario: Valor no sintetizable con el presupuesto disponible
- **WHEN** la síntesis agota su presupuesto o no alcanza la precisión requerida
- **THEN** devuelve un error explícito en lugar de un término no verificado.

### Requirement: Fachada pública
La fachada SHALL ofrecer el escalar por defecto y capacidades tensor y GGUF
mediante features optativas, preservando los contratos existentes.

#### Scenario: Uso escalar mínimo
- **WHEN** un consumidor usa la fachada sin features adicionales
- **THEN** puede acceder a `Bml` sin requerir las capacidades tensor o GGUF.
