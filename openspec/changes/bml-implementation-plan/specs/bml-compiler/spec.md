## ADDED Requirements

### Requirement: Construcción del DAG estático
El sistema SHALL transformar los tensores parseados en un DAG estático de nodos BML.

#### Scenario: Construcción desde tensores
- **WHEN** se alimenta el compilador con tensores parseados
- **THEN** se produce un `Dag` estático de nodos `BML`/`Const(1)` sin referencias mutables posteriores

### Requirement: Hash Consing de sub-árboles BML
El sistema SHALL mantener un registro global de sub-árboles BML y deduplicar sub-árboles matemáticamente idénticos en tiempo de compilación.

#### Scenario: Sub-árboles idénticos
- **WHEN** dos sub-árboles BML son estructuralmente idénticos
- **THEN** se deduplican y se reutiliza el mismo identificador de nodo en el DAG

#### Scenario: Sub-árboles distintos
- **WHEN** dos sub-árboles BML son estructuralmente distintos
- **THEN** no se deduplican (no hay colisión de hash)

#### Scenario: Reducción de complejidad
- **WHEN** se ejecuta una fórmula con operaciones repetidas sobre el DAG deduplicado
- **THEN** el tiempo de ejecución crece sub-linealmente respecto al caso sin Hash Consing (objetivo `O(n^k)` con `k < 1`)

### Requirement: Linealización a RPN
El sistema SHALL transformar el DAG deduplicado en un arreglo unidimensional en Notación Polaca Inversa (RPN).

#### Scenario: Equivalencia con el DAG
- **WHEN** se evalúa la RPN linealizada
- **THEN** el resultado coincide con la evaluación del DAG original

#### Scenario: Sin saltos ni recursión
- **WHEN** se inspecciona el flujo de evaluación de la RPN
- **THEN** es una iteración secuencial sobre el arreglo, sin llamadas recursivas

### Requirement: Micro-fragmentación AOT
El sistema SHALL empaquetar el DAG final en fragmentos cuyo tamaño de memoria pre-asignado no supere el umbral de caché objetivo (32 KB para L1 por defecto, configurable a L3).

#### Scenario: Fragmento bajo umbral L1
- **WHEN** se exporta un `.bmlgraph` con umbral por defecto (32 KB)
- **THEN** cada fragmento del binario pesa ≤ 32 KB

#### Scenario: Umbral configurable a L3
- **WHEN** se configura el umbral a L3 (ej. 1 MB)
- **THEN** cada fragmento pesa ≤ 1 MB

### Requirement: Formato binario .bmlgraph
El sistema SHALL exportar el DAG fragmentado en un formato binario `.bmlgraph` con header y fragmentos, calculable y asegurable bajo el umbral de caché objetivo.

#### Scenario: Exportación y reimportación
- **WHEN** se exporta un DAG a `.bmlgraph` y se reimporta
- **THEN** el DAG reconstruido es semánticamente equivalente al original

#### Scenario: Routing preserva semántica
- **WHEN** se ejecutan los fragmentos en orden
- **THEN** el resultado coincide con la ejecución del DAG original sin fragmentar
