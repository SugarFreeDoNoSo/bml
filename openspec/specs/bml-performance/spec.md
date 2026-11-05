# bml-performance Specification

## Purpose
Definir métricas y campañas reproducibles que comparan el mismo trabajo y declaran fases, crecimiento y límites.

## Requirements

### Requirement: Métricas de trabajo no ambiguas
Todo resultado de rendimiento SHALL definir qué cuenta y el intervalo medido:
construcciones BML/s para creación de términos/códigos, aplicaciones de B/s para
evaluación de primitivas, expresiones/s para árboles completos, elementos/s para
elementos matriciales y productos matriciales/s para operaciones completas.
#### Scenario: Convención de FLOP y unidades
- **WHEN** un informe usa FLOP/s o compara unidades
- **THEN** FLOP/s SHALL usarse solo con una convención explícita; `exp2`, `log2` y resta no
SHALL convertirse automáticamente en tres FLOP. Las unidades no SHALL sumarse
ni compararse como si fuesen equivalentes.

#### Scenario: Aplicaciones de B frente a FLOP
- **WHEN** una campaña evalúa un millón de primitivas B
- **THEN** registra aplicaciones de B y tiempo medido
- **AND** no declara tres millones de FLOP sin una convención justificada.

### Requirement: Medición por fases y crecimiento
La campaña SHALL separar síntesis y compilación/preparación, optimización,
serialización cuando aplique, kernel y operación completa. SHALL registrar tiempo
frío y caliente si hay caché, bits y operadores expandidos, profundidad, memoria
real o método de medición declarado, asignaciones y crecimiento de resultados.
El almacenamiento lógico no SHALL confundirse con bytes reservados o residentes.

#### Scenario: Valor precalculado reutilizado
- **WHEN** un kernel reutiliza constantes o elementos preparados
- **THEN** se informa el costo de preparación y el costo por ejecución por separado
- **AND** la medición integral incluye el costo de preparación que corresponda.

### Requirement: Comparación reproducible y puerta de corrección
Cada variante SHALL compararse con una referencia escalar bajo el mismo contrato,
datos dinámicos, presupuestos y política de precisión/errores. SHALL pasar la
validación antes de comparar tiempos.

#### Scenario: Metadatos reproducibles del informe
- **WHEN** se publica una comparación
- **THEN** el informe SHALL incluir CPU y capacidades,
arquitectura, sistema, revisión, compilador/flags, backend, hilos, semilla,
formas, distribución/tamaños de árboles, comandos, calentamiento, repeticiones,
dispersión y política de errores. SHALL registrar resultados de equivalencia y
memoria; no SHALL extrapolar benchmarks sintéticos a inferencia completa.

#### Scenario: Comparar SIMD con escalar
- **WHEN** se presenta una ventaja de AVX2 o NEON
- **THEN** se aportan resultados repetibles de ambos sobre hardware compatible real y datos equivalentes
- **AND** cualquier fallo de corrección impide promover la variante.

#### Scenario: Hardware no disponible
- **WHEN** no se ejecuta una campaña de una arquitectura
- **THEN** se registra pendiente o no ejecutada, sin inventar aceleración.
