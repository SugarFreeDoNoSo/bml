## ADDED Requirements

### Requirement: Inicialización única de buffers
El sistema SHALL inicializar todos los buffers de memoria una sola vez al arrancar el runtime, con cero asignaciones de memoria en el hot path.

#### Scenario: Cero allocs en hot path
- **WHEN** se ejecuta el hot loop sobre un `.bmlgraph`
- **THEN** no se realizan llamadas al allocator (verificable con un allocator custom que paniquea en `alloc` después del setup)

### Requirement: Hot loop RPN confinado a 32 KB de instrucciones
El sistema SHALL mantener el código máquina del hot loop por debajo de 32 KiB
(32.768 bytes) y acreditar su comportamiento L1i mediante el protocolo experimental
siguiente sobre la ruta real `Runtime::execute_sub_fragments` y su dispatcher.
La evidencia SHALL identificar revisión, binario, toolchain, CPU, sistema, cargas
y comandos, conservando disassembly y contadores originales. El tamaño del
bytecode RPN no SHALL contabilizarse como tamaño de instrucciones máquina ni
atribuirse a L1i. El resultado SHALL limitarse al hardware y carga medidos;
no SHALL presentarse como una garantía universal de ausencia de expulsiones.

#### Scenario: Medición del tamaño del hot loop
- **WHEN** se inspecciona el ejecutable release enlazado de la carga real
- **THEN** símbolos y disassembly SHALL identificar el código máquina del hot loop y su alcance, con tamaño inferior a 32.768 bytes
- **AND** el contador de instrucciones retiradas SHALL usarse para trabajo ejecutado, no como una medida de bytes del binario.

#### Scenario: Hot loop bajo estrés
- **WHEN** se mide la ruta real bajo carga sostenida con cambio entre varios subfragmentos
- **THEN** la campaña SHALL ejecutar siete repeticiones, con afinidad fijada al mismo core, calentamiento previo al intervalo medido y cantidades de trabajo declaradas e iguales
- **AND** en cada repetición SHALL cumplirse `L1I-misses * 1000 <= retired-instructions`, con al menos una instrucción retirada; no SHALL aceptarse solo la media de las repeticiones
- **AND** ambos eventos SHALL cubrir íntegramente el mismo intervalo y ámbito, con tiempo de ejecución positivo y 100 % de tiempo contado, sin multiplexación parcial.

#### Scenario: Contadores no disponibles o medición incompleta
- **WHEN** falta un contador de instrucciones retiradas o L1i, el evento no está soportado o no se contó, el intervalo tiene cero instrucciones, o se contaron parcialmente los eventos
- **THEN** la campaña SHALL fallar o quedar pendiente, sin reemplazar L1i por misses de caché genéricos ni convertir valores ausentes en cero
- **AND** una validación de datos sintéticos, una medición estática o una carga de `RpnProgram::evaluate` aislada no SHALL cerrar esta validación de la ruta Runtime.

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
