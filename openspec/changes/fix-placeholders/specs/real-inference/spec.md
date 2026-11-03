## ADDED Requirements

### Requirement: Fragmentos públicos completos con layout explícito
Las APIs públicas de atención, MLP y composición de capa SHALL ejecutar sus operaciones completas con direcciones absolutas de entrada/pesos y slots de salida explícitos. Las declaraciones BML acotadas SHALL conservar sus dominios parciales, sin sustituirlas por aritmética nativa de datos ni afirmar rango IEEE universal. Los constructores configurables SHALL rechazar dimensiones y layouts incompatibles; las constantes por operación SHALL pertenecer al programa que las usa.

#### Scenario: Atención de una consulta sobre un prefijo KV
- **WHEN** se proporciona una cabeza de consulta, dimensiones explícitas y varias filas KV
- **THEN** el programa calcula cada dot, escala por la raíz del ancho, selecciona el máximo para softmax y suma todos los valores ponderados
- **AND** la variante con scratch declarado evita recalcular los scores por cada dimensión de salida
- **AND** la firma histórica sin parámetro de contexto representa un único token KV y no acredita GQA

#### Scenario: MLP rectangular y pesos down
- **WHEN** embedding y FFN tienen anchos distintos y pesos firmados no triviales
- **THEN** gate y up usan matrices completas, SwiGLU es `gate * sigmoid(gate) * up` y down produce todas las dimensiones de embedding
- **AND** el plan con scratch conserva cada bucle completo dentro del presupuesto de fragmento
- **AND** modificar cualquiera de las tres matrices cambia el resultado correspondiente

#### Scenario: Capa causal completa
- **WHEN** se ejecuta una capa de una cabeza en dos posiciones consecutivas con sus pesos y caché KV explícitos
- **THEN** cada posición aplica RMS, QKV, RoPE, atención al prefijo, proyección de salida, residual original, segundo RMS, MLP y segundo residual
- **AND** conserva las filas KV anteriores, las entradas y los pesos
- **AND** un oráculo independiente verifica la capa completa y las direcciones de slots se convierten con el tamaño real de slot
- **AND** esta API de operaciones no acredita por sí sola el compilador, worker ni CLI distribuidos históricos

### Requirement: Matmul vectorial completo
El sistema SHALL implementar matmul real con acumulador de suma de productos, donde cada peso W[i][j] se aplica a su dimensión correspondiente del hidden state.

#### Scenario: Matmul 2x2
- **WHEN** se ejecuta un matmul de una matriz 2x2 conocida por un vector conocido
- **THEN** el resultado coincide con el cálculo manual

#### Scenario: Acumulador
- **WHEN** se ejecuta un matmul de n_rows x n_cols
- **THEN** cada y[i] = sum_j(W[i][j] * x[j]) se computa correctamente

### Requirement: RoPE completo
El sistema SHALL aplicar RoPE a todos los pares de dimensiones del hidden state, no solo al primero.

#### Scenario: Posición 0
- **WHEN** se aplica RoPE en posición 0
- **THEN** los valores no cambian (cos=1, sin=0)

#### Scenario: Todos los pares
- **WHEN** se aplica RoPE con n_embd=2048
- **THEN** se aplican 1024 rotaciones (una por par de dimensiones)

### Requirement: SwiGLU real
El sistema SHALL implementar SwiGLU como `gate * sigmoid(gate) * up` con sigmoid real.

#### Scenario: SwiGLU(0)
- **WHEN** x=0
- **THEN** resultado = 0 (0 * sigmoid(0) * up = 0)

#### Scenario: SwiGLU(1)
- **WHEN** x=1
- **THEN** resultado = up / (1 + exp(-1)), sin un factor adicional en el exponente

#### Scenario: SwiGLU con signo y valor cerrado
- **WHEN** gate es ln(2) o -ln(2) y up es 3
- **THEN** el resultado es 2*ln(2) o -ln(2), respectivamente

### Requirement: Normalización final del modelo
La inferencia Llama SHALL aplicar `output_norm.weight` antes de la proyección de logits y usar el epsilon positivo y finito de `llama.attention.layer_norm_rms_epsilon` en las normalizaciones. Si el metadato no existe, el valor por defecto SHALL ser 1e-5. Un modelo con capas SHALL rechazar una normalización final ausente o inválida.

#### Scenario: Epsilon y pesos independientes
- **WHEN** el estado final es [3,4], los pesos de normalización son [2,-0.5], epsilon es 3.5 y la proyección es la identidad
- **THEN** los logits son [1.5,-0.5] tanto con caché como sin caché

#### Scenario: Metadatos inválidos
- **WHEN** epsilon es cero, negativo, no finito o de tipo incorrecto, o la normalización final tiene dimensiones o valores inválidos
- **THEN** la carga falla antes de ejecutar el modelo

### Requirement: Tokenización
El sistema SHALL tokenizar el prompt con el perfil SentencePiece BPE validado del GGUF, respetando scores, tipos de token, política BOS/EOS, espacios y fallback por bytes. La falta de metadatos necesarios SHALL producir un error, no una búsqueda por palabras ni un BOS usado como sustituto de texto desconocido.

#### Scenario: Tokenizar "Hello"
- **WHEN** se tokeniza "Hello"
- **THEN** se produce una lista de token IDs válidos

#### Scenario: Detokenizar
- **WHEN** se detokeniza un token ID
- **THEN** se producen sus bytes, incluidos espacios y saltos de línea, o un error para un ID inválido

#### Scenario: UTF-8 incremental
- **WHEN** varios tokens de bytes forman un carácter multibyte
- **THEN** la CLI conserva sus bytes en orden y el servidor SSE espera a tener texto completo antes de emitirlo
- **AND** los bytes inválidos o una secuencia incompleta al terminar se representan de forma explícita con el carácter de sustitución en la salida textual del servidor

### Requirement: Sampling
El sistema SHALL implementar sampling greedy (temp=0) y con temperatura (temp>0).

#### Scenario: Greedy
- **WHEN** temp=0
- **THEN** se selecciona el token con mayor logit

#### Scenario: Temperatura
- **WHEN** temp=0.8
- **THEN** se aplica softmax(logits/temp) antes de muestrear

### Requirement: Generación de tokens real
El sistema SHALL generar tokens reales ejecutando el transformer, no placeholders.

#### Scenario: Prefill causal e incremental
- **WHEN** se evalúa un prefijo completo o el mismo prefijo dividido en prefill y decode
- **THEN** cada token solo atiende a posiciones anteriores o a sí mismo, los logits coinciden y el orden de los tokens se conserva
- **AND** RMSNorm se aplica a una copia del estado residual antes de las proyecciones de atención y MLP

#### Scenario: GQA y límites de caché
- **WHEN** el modelo declara `attention.head_count_kv` distinto del número de cabezas de consulta
- **THEN** se comparte la cabeza KV según el agrupamiento declarado y se valida el layout de la caché
- **AND** IDs inválidos, dimensiones incompatibles, límites de contexto o presupuesto de caché se rechazan explícitamente sin sobrescribir el prefijo confirmado

#### Scenario: Error numérico y reintento
- **WHEN** un decode produce un resultado no finito
- **THEN** la posición confirmada no avanza, los datos de su prefijo se conservan y un reintento reemplaza las filas aún no confirmadas

#### Scenario: CLI
- **WHEN** se ejecuta `bml-cli run -m model.bmlgraph/ -p "Hello" -n 10`
- **THEN** se produce texto coherente (no placeholder ni NaN)

#### Scenario: Servidor
- **WHEN** se envía `POST /v1/completions` con `{"prompt":"Hello"}`
- **THEN** se reciben tokens reales via SSE

### Requirement: Artefacto causal persistido
La compilación local SHALL guardar programas RPN ejecutables, metadatos numéricos, el tokenizer BPE y bindings a pesos GGUF identificados por SHA-256. La carga SHALL ejecutar los programas guardados, sin regenerarlos ni sustituirlos por proyecciones directas. El perfil numérico SHALL identificar explícitamente las operaciones nativas compartidas; este perfil no acredita por sí mismo la traducción completa a identidades BML exigida por gguf-compiler.

#### Scenario: Pesos reubicados
- **WHEN** el mismo GGUF se mueve y se proporciona una ruta de pesos alternativa
- **THEN** se verifica su hash antes de ejecutar los programas del artefacto
- **AND** un archivo distinto, un programa corrupto o bindings incompatibles son rechazados

#### Scenario: Preflight de programas guardados
- **WHEN** un programa persistido excede el presupuesto de pasos o tiene referencias o efectos de stack inválidos
- **THEN** la carga falla antes de ejecutar inferencia o confirmar posiciones KV

#### Scenario: Perfil simbólico AOT con contexto explícito
- **WHEN** se compila con `bml-cli compile --symbolic --context-size N`
- **THEN** se persisten los kernels BML necesarios para todos los prefijos de hasta N tokens, con referencias a pesos externos y literales del perfil validados
- **AND** la ejecución carga los programas almacenados sin recompilarlos ni sustituirlos por aritmética nativa
- **AND** planes incompletos, perfiles ambiguos y posiciones fuera del contexto compilado se rechazan explícitamente

#### Scenario: Servidor con artefacto simbólico
- **WHEN** el servidor carga un artefacto simbólico AOT y recibe peticiones JSON o SSE
- **THEN** ejecuta el bytecode guardado y respeta el contexto compilado
- **AND** los bytes UTF-8 se emiten completos y cada stream termina exactamente una vez, también ante errores de contexto
- **AND** un directorio que mezcla perfiles nativo y simbólico se rechaza al arrancar

#### Scenario: Programas simbólicos vectoriales compartidos
- **WHEN** se compilan RMSNorm, softmax o RoPE para el perfil simbólico v2
- **THEN** las salidas comparten los sub-DAGs en un programa con escrituras a buffer y conservan los bits del lowering escalar
- **AND** el cargador conserva compatibilidad explícita con programas escalares v1 sin recompilarlos
- **AND** una salida no escrita o no finita falla antes de confirmar la posición KV
- **AND** los presupuestos de nodos, instrucciones y programas siguen siendo obligatorios
