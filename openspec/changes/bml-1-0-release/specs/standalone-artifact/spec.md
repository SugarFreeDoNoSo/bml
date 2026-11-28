## ADDED Requirements

### Requirement: Autonomous model executable
The compiler SHALL produce a native Linux x86_64 executable containing the
validated plan, tokenizer and normalized model resources, linked to BML's
own CPU execution kernels with declared system dependencies.

#### Scenario: Clean execution
- **WHEN** only the executable and declared system libraries are available
- **THEN** inference succeeds without a GGUF, BML installation, compiler,
  Python, llama.cpp or network and reproduces the validated outputs.

#### Scenario: Corrupt resources
- **WHEN** embedded resource hashes, ranges, alignment or format are invalid
- **THEN** startup reports an error before executing model operations.

### Requirement: Native standalone sublayer fragments
El compilador SHALL generar un ELF Linux nativo autónomo por cada fragmento de
subcapa seleccionado, con sus recursos incorporados y dependencias del sistema
declaradas. La tarea 4.6 SHALL permanecer abierta hasta verificar las etapas
local y entre hosts distintos; su cierre verificado SHALL ser el criterio de
cierre de la parte uno, sin otra casilla duplicada.

#### Scenario: Ejecución autónoma y composición
- **WHEN** se ejecutan los ELF de fragmentos y su coordinador con las dependencias declaradas
- **THEN** la ruta completa SHALL consumir sus recursos propios sin un GGUF externo durante la ejecución
- **AND** el coordinador SHALL identificar entradas, resultados y cobertura de cada fragmento, pasando activaciones y agregando resultados en orden determinista
- **AND** identidades, dimensiones o cobertura incompatibles SHALL fallar sin producir una generación declarada válida.

#### Scenario: Generación local frente al control monolítico
- **WHEN** se valida el conjunto de ELF con un modelo GGUF real de referencia fijado por revisión y SHA-256
- **THEN** la evidencia SHALL registrar revisión de fuente, identidades de ejecutables, prompts, BOS/plantilla, contexto e IDs de tokens
- **AND** la generación completa SHALL producir los mismos IDs que un proceso monolítico usando los mismos recursos y entradas; los logits prefill/decode SHALL satisfacer el criterio de paridad acordado frente a ese proceso
- **AND** el criterio de paridad de logits SHALL quedar acordado antes de medir; resultados y primera divergencia SHALL conservarse sin ampliar tolerancias para aprobar.

#### Scenario: Cierre de la parte uno y alcance distribuido
- **WHEN** la generación local por fragmentos ELF pasa la comparación
- **THEN** la parte uno y 4.6 SHALL continuar abiertas hasta acreditar la misma ejecución en hosts distintos con procedencia e identidad verificables
- **AND** el ELF monolítico y el prototipo híbrido de dos workers en un host SHALL contar sólo como prerrequisitos, no como ELF compilados por fragmento
- **AND** ni esos prerrequisitos ni la validación entre hosts SHALL presentarse por sí solos como ejecución Lambda o como cierre del contrato BMLX simbólico.

#### Scenario: Validaciones locales independientes
- **WHEN** la tarea final 4.6 todavía está pendiente
- **THEN** las validaciones locales CLI/ELF y el benchmark separado Docker/AVX empaquetado SHALL poder avanzar con sus propios criterios
- **AND** sus resultados SHALL conservar su alcance sin cerrar anticipadamente la parte uno ni exigir 4.6 para ejecutarlos.
