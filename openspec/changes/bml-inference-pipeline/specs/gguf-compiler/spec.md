## ADDED Requirements

### Requirement: Compilación GGUF a .bmlgraph
El sistema SHALL compilar un modelo GGUF a un conjunto de archivos `.bmlgraph` fragmentados, traduciendo cada operación del transformer a la gramática BML.

#### Scenario: Compilación de un GGUF
- **WHEN** se ejecuta `bml-compile model.gguf --target local`
- **THEN** se genera un directorio `model.bmlgraph/` con N archivos `.bmlgraph`, uno por fragmento

#### Scenario: Número mínimo de fragmentos
- **WHEN** se compila con `--target local` en una máquina de 4 cores con L1=32KB
- **THEN** el número de fragmentos es `max(1, ceil(total_ops / (32768 * 4)))`

#### Scenario: Target remoto
- **WHEN** se compila con `--target specs:8:32KB:256KB:16MB`
- **THEN** los fragmentos se optimizan para 8 cores con esos tamaños de caché

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
