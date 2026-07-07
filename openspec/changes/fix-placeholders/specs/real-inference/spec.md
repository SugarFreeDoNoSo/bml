## ADDED Requirements

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
El sistema SHALL implementar SwiGLU como `gate * sigmoid(1.7 * gate) * up` con sigmoid real.

#### Scenario: SwiGLU(0)
- **WHEN** x=0
- **THEN** resultado = 0 (0 * sigmoid(0) * up = 0)

#### Scenario: SwiGLU(1)
- **WHEN** x=1
- **THEN** resultado > 0 (sigmoid(1.7) > 0.5)

### Requirement: Tokenización
El sistema SHALL tokenizar el prompt a IDs de tokens usando el tokenizer del GGUF.

#### Scenario: Tokenizar "Hello"
- **WHEN** se tokeniza "Hello"
- **THEN** se produce una lista de token IDs válidos

#### Scenario: Detokenizar
- **WHEN** se detokeniza un token ID
- **THEN** se produce el string correspondiente

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

#### Scenario: CLI
- **WHEN** se ejecuta `bml-cli -m model.bmlgraph/ -p "Hello" -n 10`
- **THEN** se produce texto coherente (no placeholder ni NaN)

#### Scenario: Servidor
- **WHEN** se envía `POST /v1/completions` con `{"prompt":"Hello"}`
- **THEN** se reciben tokens reales via SSE
