## ADDED Requirements

### Requirement: Matrices con elementos Bml y formas comprobadas
La biblioteca SHALL proporcionar una capacidad explícita de matrices cuyos
elementos permanezcan como `Bml`, preparados durante la carga o antes del kernel,
sin volver a sintetizarlos por cada operación. La disposición inicial SHALL ser
row-major y estar documentada; SHALL evaluar localidad, bloques, caché, buffers,
memoria y asignaciones sin prometer que los payloads de `Vec<Bml>` sean contiguos.
Constantes y valores repetidos MAY reutilizarse sin ocultar costos de preparación.
SHALL validar dimensiones, productos de tamaños
y cantidad de elementos antes de asignar memoria. No SHALL convertir elementos
a floats implícitamente. Los tensores IEEE `f32/f64`, sus kernels y los bloques
GGUF existentes SHALL conservar sus contratos separados.

#### Scenario: Matriz simbólica distinta de un tensor IEEE
- **WHEN** se construye una matriz de forma `2x3` con seis elementos `Bml`
- **THEN** conserva los códigos BML y su orden, sin convertirlos a `f64`
- **AND** una longitud incompatible o un tamaño que desborda se rechazan explícitamente.

#### Scenario: Valores repetidos precompilados
- **WHEN** una matriz contiene el mismo valor en varias posiciones y se opera repetidamente
- **THEN** el kernel consume los códigos BML preparados sin sintetizar cada elemento durante el recorrido
- **AND** cualquier caché o copia y su costo de preparación se contabilizan por separado.

### Requirement: Operaciones matriciales simbólicas y fallibles
La API SHALL definir `A + B` como suma elemento a elemento, `A * B` como producto
matricial y `hadamard` como producto elemento a elemento. Las operaciones con
escalar SHALL definir su aplicación a cada elemento y preparar el escalar una
vez antes del kernel. SHALL validar formas y delegar la matriz o el bloque completo
en kernels que apliquen las operaciones simbólicas comprobadas.
Los códigos enteros no SHALL sumarse o multiplicarse como valores matemáticos.
Las operaciones SHALL retornar errores explícitos de dominio/recursos con la
posición del fallo, preservar operandos y no entregar resultados parciales.
`Add`/`Mul` SHALL devolver `Result` y preservar este contrato fallible.

#### Scenario: Operar valores y no sus códigos
- **WHEN** se suman dos elementos con código entero `4`, cada uno representando el valor `2`
- **THEN** se construye una expresión BML que representa `4`
- **AND** no se usa la suma entera de códigos `4+4=8` como implementación del resultado.

#### Scenario: Producto rectangular y crecimiento
- **WHEN** se multiplica una matriz BML `2x3` por una `3x4`
- **THEN** se obtiene una `2x4` con productos y sumas simbólicos verificados frente a un oráculo independiente
- **AND** si un resultado intermedio supera el presupuesto se informa su posición sin modificar las matrices de entrada.

### Requirement: Acceso e invariantes de matrices
La API SHALL exponer `Index` para `(fila,columna)` y un acceso comprobado que
informe índices fuera de rango. Si expone `IndexMut` SHALL documentar que el
índice inválido provoca panic y asegurar que la sustitución de elementos no elude
los límites de la matriz: la forma y longitud son invariantes, y los presupuestos
de bits totales SHALL revalidarse antes de toda operación o serialización que los
requiera. Una API de reemplazo comprobado SHALL preservar la matriz al fallar.

#### Scenario: Acceso inválido o sustitución excesiva
- **WHEN** se solicita un índice fuera de rango mediante el acceso comprobado o se reemplaza un elemento superando un presupuesto
- **THEN** se obtiene un error y la matriz no se modifica parcialmente.

### Requirement: Orden de reducción y presupuestos de matriz
La referencia escalar SHALL iniciar una reducción no vacía con el primer producto,
reducir los restantes de izquierda a derecha y devolver el cero simbólico en
reducciones vacías. SHALL aplicar límites por elemento y al conjunto
de la operación, incluyendo intermediarios y duplicación de subárboles. Cambiar
asociación u orden SHALL requerir una justificación explícita de equivalencia,
dominio y política de evaluación/recursos; la equivalencia matemática sola no
SHALL prometer códigos idénticos, igual memoria o igual comportamiento flotante.

#### Scenario: Dimensión interna vacía
- **WHEN** se multiplica una matriz `2x0` por una `0x3`
- **THEN** se obtiene una `2x3` de ceros simbólicos, sujetos al presupuesto de salida
- **AND** el código entero `0`, que representa `1`, no se usa como cero matemático.

### Requirement: Producto matricial generalizado
El producto generalizado SHALL aceptar o identificar explícitamente una operación
de combinación, una operación de acumulación y una política de inicialización.
SHALL especificar el orden y los paréntesis de la reducción; no SHALL suponer
asociatividad o identidad de la primitiva B. La política SHALL definir el caso de
dimensión interna cero mediante un valor inicial explícito o un error; iniciar
desde el primer término solo SHALL admitirse para reducciones no vacías.

#### Scenario: Acumulación no asociativa
- **WHEN** se usa una acumulación no asociativa con tres productos `p0`, `p1`, `p2` e inicio por primer término
- **THEN** el resultado estructural es `acumular(acumular(p0,p1),p2)` y no una reasociación.

#### Scenario: Producto generalizado vacío sin identidad
- **WHEN** la dimensión interna es cero y se eligió iniciar por primer término sin valor vacío
- **THEN** se devuelve un error explícito en vez de suponer que `1` es identidad de B.
