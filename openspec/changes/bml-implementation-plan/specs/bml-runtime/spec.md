## ADDED Requirements

### Requirement: Inicialización única de buffers
El sistema SHALL inicializar todos los buffers de memoria una sola vez al arrancar el runtime, con cero asignaciones de memoria en el hot path.

#### Scenario: Cero allocs en hot path
- **WHEN** se ejecuta el hot loop sobre un `.bmlgraph`
- **THEN** no se realizan llamadas al allocator (verificable con un allocator custom que paniquea en `alloc` después del setup)

### Requirement: Hot loop RPN confinado a 32 KB de instrucciones
El sistema SHALL implementar el intérprete del flujo RPN de forma que el código compilado del hot loop no supere 32 KB de instrucciones, garantizando que nunca sea expulsado de la caché L1 de instrucciones.

#### Scenario: Medición del tamaño del hot loop
- **WHEN** se mide el tamaño del binario del hot loop con `cargo asm` o `perf stat -e instructions`
- **THEN** el resultado es < 32 KB

#### Scenario: Hot loop bajo estrés
- **WHEN** se ejecuta el hot loop bajo carga sostenida
- **THEN** la tasa de L1-icache-load-misses medida con `perf` se mantiene baja (no hay expulsión del hot loop)

### Requirement: Evaluación append-only del DAG
El sistema SHALL evaluar el DAG en patrón append-only: un hilo lee `v_i` de un nodo, computa `v_o` y lo escribe a una dirección de memoria pre-asignada nueva, sin sobrescribir el estado previo.

#### Scenario: No sobrescritura
- **WHEN** se evalúa un nodo
- **THEN** `v_i` permanece inmutable y `v_o` se escribe a una dirección distinta pre-asignada

#### Scenario: Ausencia de data races
- **WHEN** múltiples hilos evalúan el DAG concurrentemente
- **THEN** no se observan data races (verificable con `loom` y/o Miri)

### Requirement: Ejecución de .bmlgraph
El sistema SHALL ejecutar un `.bmlgraph` y producir el mismo resultado que la evaluación de referencia del DAG original.

#### Scenario: Ejecución correcta
- **WHEN** se carga y ejecuta un `.bmlgraph` de prueba
- **THEN** el resultado coincide con la evaluación de referencia del DAG

### Requirement: Interfaz RPC/binaria para distribución append-only
El sistema SHALL exponer una interfaz RPC/binaria (gRPC u otro) para recibir y transmitir fragmentos `.bmlgraph` entre nodos de ejecución distribuida, en patrón append-only.

#### Scenario: Envío y recepción de fragmentos
- **WHEN** un nodo envía un fragmento `.bmlgraph` a otro nodo por RPC
- **THEN** el receptor lo recibe íntegro y puede ejecutarlo

#### Scenario: Aislamiento del hot loop
- **WHEN** se inspecciona el binario del runtime
- **THEN** el código RPC está aislado en un módulo separado y no forma parte del hot loop (no impacta el límite de 32 KB)

#### Scenario: Ejecución distribuida end-to-end
- **WHEN** se orquesta una ejecución distribuida entre N nodos
- **THEN** el resultado agregado coincide con la ejecución local del DAG completo
