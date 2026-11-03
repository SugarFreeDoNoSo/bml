## ADDED Requirements

### Requirement: Parser GGUF completo
El sistema SHALL decodificar completamente un archivo GGUF: metadatos KV, tensor infos (nombre, dims, tipo, offset), y acceso zero-copy a los datos de los tensores via mmap.

#### Scenario: Decodificar metadatos KV
- **WHEN** se parsea un GGUF
- **THEN** los pares clave-valor de metadatos quedan accesibles con sus tipos (string, int, float, array)

#### Scenario: Decodificar tensor infos
- **WHEN** se parsea un GGUF
- **THEN** cada tensor tiene nombre, n_dims, dims, data_type y offset accesibles

#### Scenario: Acceso zero-copy a datos del tensor
- **WHEN** se accede a los datos de un tensor
- **THEN** se retorna un slice sobre el mmap sin copias a RAM

#### Scenario: Detectar arquitectura
- **WHEN** se parsea un GGUF
- **THEN** se lee `general.architecture` para determinar el tipo de modelo (llama, qwen, etc.)

### Requirement: AST con variables y constantes
El sistema SHALL extender la gramática del AST para soportar inputs variables (`Var(id)`) y constantes arbitrarias (`Const(f64)`), además de `1` y `BML(S, S)`.

#### Scenario: Nodo de variable
- **WHEN** se construye un nodo `Var(id)`
- **THEN** al evaluarlo, se resuelve desde el contexto de inputs

#### Scenario: Nodo de constante
- **WHEN** se construye un nodo `Const(value)`
- **THEN** al evaluarlo, retorna el valor constante

#### Scenario: Gramática extendida
- **WHEN** se inspecciona el AST
- **THEN** los nodos pueden ser `One`, `Var(id)`, `Const(value)`, o `BML(left, right)`

### Requirement: RPN con variables y constantes
El sistema SHALL extender `RpnOp` con `Var(u32)` y `Const(u32)` (índices al pool de inputs/pesos), y el hot loop las resolverá desde buffers pre-asignados.

#### Scenario: Ejecutar con inputs variables
- **WHEN** se ejecuta un programa RPN con `Var(0)` y se pasa un contexto de inputs
- **THEN** `Var(0)` se resuelve al primer input

#### Scenario: Ejecutar con pesos constantes
- **WHEN** se ejecuta un programa RPN con `Const(42)` y se pasa un pool de pesos
- **THEN** `Const(42)` se resuelve al peso en el índice 42

### Requirement: Compilación GGUF a .bmlgraph
El sistema SHALL compilar un modelo GGUF a un conjunto de archivos `.bmlgraph` fragmentados, traduciendo cada operación del transformer a la gramática BML usando los pesos como `Const` y los inputs como `Var`.

#### Scenario: Compilación de un GGUF
- **WHEN** se ejecuta `bml-compile model.gguf --target local`
- **THEN** se genera un directorio `model.bmlgraph/` con N archivos `.bmlgraph`, uno por fragmento

#### Scenario: Número mínimo de fragmentos
- **WHEN** se compila con `--target local` en una máquina de 4 cores con L1=32KB
- **THEN** el número de fragmentos es `max(1, ceil(total_ops / (32768 * 4)))`

#### Scenario: Target remoto
- **WHEN** se compila con `--target specs:8:32KB:256KB:16MB`
- **THEN** los fragmentos se optimizan para 8 cores con esos tamaños de caché

#### Scenario: Pesos zero-copy
- **WHEN** se compila un GGUF
- **THEN** los pesos se referencian desde el GGUF mmap, no se copian al `.bmlgraph`

### Requirement: Traducción de operaciones del transformer
El sistema SHALL traducir cada operación del transformer (matmul, RMSNorm, RoPE, softmax, SwiGLU) a la gramática BML usando el `BMLTransformer`.

#### Scenario: Matmul
- **WHEN** se traduce una operación matmul A·B
- **THEN** se produce un sub-DAG BML equivalente usando `mul`, `add`

#### Scenario: Softmax
- **WHEN** se traduce softmax(x)
- **THEN** se produce un sub-DAG BML usando `exp2`, `add`, `div`

#### Scenario: RMSNorm
- **WHEN** se traduce RMSNorm(x)
- **THEN** se produce un sub-DAG BML usando `mul`, `div`, `pow`, `add`

### Requirement: Perfil experimental de RMS con límites explícitos
El sistema SHALL permitir seleccionar una expansión RMS acotada con límites de
activaciones y pesos persistidos en un perfil numérico distinto. Los artefactos
existentes v1/v2 SHALL conservar sus identidades y resultados. Este perfil
parcial no sustituye la obligación de completar la inferencia del transformer.

#### Scenario: RMS acotado persistido
- **WHEN** se compila y carga un artefacto con el perfil `llama-symbolic-bml-bounded-rms-f64-f32kv-v3`
- **THEN** la configuración y los literales RMS se validan antes de ejecutar, los pesos siguen externos y los programas cargados no se regeneran

#### Scenario: Dominio de cada operación BML
- **WHEN** se ejecuta RMS acotado con operandos declarados válidos
- **THEN** cada operación BML rechaza argumentos no reales finitos, logaritmos no positivos y desbordamiento o subdesbordamiento exponencial, aunque operaciones posteriores pudieran ocultarlos

#### Scenario: Límites y tolerancia independientes
- **WHEN** se verifica el perfil con ceros, signos, extremos y datos reales de un RMS
- **THEN** se compara contra un oráculo numérico independiente con tolerancia explícita y se rechazan operandos fuera de los límites, sin atribuir equivalencia IEEE ni éxito de inferencia completa

### Requirement: Perfil completo de aritmética acotada
El sistema SHALL ofrecer un perfil explícito v4 que comprueba cada operación
BML de todas las familias ejecutadas. Los límites de operandos SHALL distinguir
proyecciones aprendidas, productos query/key y sumas probability/value. Los
perfiles anteriores SHALL conservar su representación y comportamiento.

#### Scenario: Proyección compacta con pesos externos
- **WHEN** se compila un dot de ancho creciente bajo el perfil v4
- **THEN** el cuerpo BML persistido mantiene tamaño constante y accede por índices comprobados a inputs y pesos externos, mientras el presupuesto de pasos refleja el trabajo lineal ejecutado

#### Scenario: Softmax con máximo auxiliar explícito
- **WHEN** el runtime prepara los scores de softmax
- **THEN** selecciona y copia su máximo como dato auxiliar validado y todas las restas, exponenciales, sumas y divisiones se ejecutan con BML, incluyendo los casos con todos los scores negativos

#### Scenario: Artefacto completo y compatible
- **WHEN** se carga un artefacto v4 con presupuesto de construcción cero
- **THEN** se ejecutan sus programas persistidos sin regenerarlos, se preservan exactamente los literales JSON finitos y se rechazan perfiles, roles, límites y referencias inconsistentes

#### Scenario: Bucle finito y verificable
- **WHEN** se ejecuta un programa con índices de contexto y bucles
- **THEN** cada índice debe ser entero, finito, no negativo y estar dentro del contexto, el presupuesto incluye todas las iteraciones y cada BML rechaza intermedios fuera de dominio/rango
