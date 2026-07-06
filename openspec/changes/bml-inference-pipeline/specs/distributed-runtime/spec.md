## ADDED Requirements

### Requirement: Runtime distribuido con gRPC
El sistema SHALL ejecutar fragmentos `.bmlgraph` entre múltiples nodos vía gRPC, con sistema de colas lock-free y work-stealing.

#### Scenario: Ejecución distribuida
- **WHEN** un coordinador envía un fragmento a un worker vía `ExecuteFragment`
- **THEN** el worker lo ejecuta y devuelve el resultado vía `ReportResult`

#### Scenario: Work-stealing
- **WHEN** un worker vacía su cola de fragmentos
- **THEN** roba trabajo de otro worker vía `StealWork` sin locks

#### Scenario: Health check
- **WHEN** se llama `HealthCheck` a un nodo
- **THEN** se retorna `HealthStatus` indicando si está vivo y cuántos fragmentos tiene en cola

### Requirement: Cola lock-free
El sistema SHALL mantener una cola lock-free por nodo para fragmentos pendientes, verificada con `loom`.

#### Scenario: Ausencia de data races
- **WHEN** múltiples hilos acceden a la cola concurrentemente
- **THEN** no se observan data races (verificable con `loom`)

#### Scenario: Work-stealing sin deadlock
- **WHEN** un worker roba trabajo de otro
- **THEN** no se produce deadlock ni starvation

### Requirement: Aislamiento del hot loop
El sistema SHALL aislar el código gRPC del hot loop para no impactar el límite de 32 KB.

#### Scenario: Hot loop no toca gRPC
- **WHEN** se mide el tamaño del hot loop
- **THEN** es < 32 KB, sin incluir código gRPC
