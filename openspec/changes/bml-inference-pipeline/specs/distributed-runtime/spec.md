## ADDED Requirements

### Requirement: Comunicación interna entre nodos
El sistema SHALL ejecutar fragmentos `.bmlgraph` entre múltiples nodos vía TCP raw y `/dev/shm`, con sistema de colas lock-free y work-stealing.

#### Scenario: Ejecución distribuida via TCP
- **WHEN** un coordinador envía un fragmento a un worker vía TCP `ExecuteFragment`
- **THEN** el worker lo ejecuta y devuelve el resultado vía TCP `ReportResult`

#### Scenario: Work-stealing
- **WHEN** un worker vacía su cola de fragmentos
- **THEN** roba trabajo de otro worker vía TCP `StealWork` sin locks

#### Scenario: Same-machine via /dev/shm
- **WHEN** múltiples workers están en la misma máquina
- **THEN** los fragmentos se comparten via `/dev/shm` sin serialización

#### Scenario: Health check
- **WHEN** se llama `HealthCheck` a un nodo
- **THEN** se retorna el estado del nodo y cuántos fragmentos tiene en cola

### Requirement: Cola lock-free
El sistema SHALL mantener una cola lock-free por nodo para fragmentos pendientes, verificada con `loom`.

#### Scenario: Ausencia de data races
- **WHEN** múltiples hilos acceden a la cola concurrentemente
- **THEN** no se observan data races (verificable con `loom`)

#### Scenario: Work-stealing sin deadlock
- **WHEN** un worker roba trabajo de otro
- **THEN** no se produce deadlock ni starvation

### Requirement: Aislamiento del hot loop
El sistema SHALL aislar el código de red del hot loop para no impactar el límite de 32 KB.

#### Scenario: Hot loop no toca red
- **WHEN** se mide el tamaño del hot loop
- **THEN** es < 32 KB, sin incluir código de red

### Requirement: API externa HTTP + SSE
El sistema SHALL exponer un endpoint HTTP OpenAI-compatible con streaming SSE para recibir prompts y enviar tokens.

#### Scenario: Completions endpoint
- **WHEN** se envía `POST /v1/completions` con `{"prompt": "Hello", "max_tokens": 10, "stream": true}`
- **THEN** se reciben tokens via SSE: `data: {"token": "..."}\n\n`

#### Scenario: Backpressure
- **WHEN** la cola del scheduler está llena
- **THEN** se retorna HTTP 429

### Requirement: Scheduler con batching dinámico
El sistema SHALL agrupar múltiples prompts en batches para maximizar throughput.

#### Scenario: Batching
- **WHEN** llegan 10 prompts en 10ms
- **THEN** se agrupan en 1-2 batches para procesamiento conjunto
