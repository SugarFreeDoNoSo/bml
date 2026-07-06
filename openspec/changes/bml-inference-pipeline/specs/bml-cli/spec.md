## ADDED Requirements

### Requirement: CLI compatible con llama.cpp
El sistema SHALL proveer un binario `bml-cli` con flags compatibles con `llama-cli` para ser usado como drop-in replacement.

#### Scenario: Inferencia básica
- **WHEN** se ejecuta `bml-cli -m model.bmlgraph/ -p "Hello" -n 10`
- **THEN** se produce texto generado de 10 tokens

#### Scenario: Flags core
- **WHEN** se ejecuta `bml-cli --help`
- **THEN** se muestran los flags `-m`, `-p`, `-n`, `-t`, `--temp`, `-c`

#### Scenario: Drop-in replacement
- **WHEN** se reemplaza `llama-cli` por `bml-cli` en un script
- **THEN** los flags `-m`, `-p`, `-n`, `-t` funcionan igual

### Requirement: Sampling básico
El sistema SHALL implementar sampling greedy + temperatura básica.

#### Scenario: Greedy
- **WHEN** `--temp 0` (greedy)
- **THEN** se selecciona el token con mayor logit

#### Scenario: Temperatura
- **WHEN** `--temp 0.8`
- **THEN** se aplica softmax con temperatura 0.8 antes de muestrear
