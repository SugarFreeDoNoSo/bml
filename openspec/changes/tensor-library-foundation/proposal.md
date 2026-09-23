## Why

El objetivo del usuario es una biblioteca Rust para tensores y GGUF con operaciones
EML/BML. El workspace actual acopla esas responsabilidades con inferencia, HTTP y
distribución. Existen fallos reproducibles de evaluación y de lectura de pesos que
varios tests actuales no detectan. Los benchmarks históricos extrapolan tokens/s
sin ejecutar trabajo equivalente de inferencia.

La separación propuesta por el usuario entre términos cerrados y abiertos ofrece
una base clara para sustitución y cierre. La evaluación y la síntesis numérica
requieren contratos distintos y no son conversiones inversas garantizadas.

## What Changes

- Implementar `ClosedTerm`, `OpenTerm`, `bind`, variables libres y `collapse` en
  un módulo nuevo y compatible del dominio.
- Interpretar el árbol con base EML/BML explícita y devolver errores numéricos.
- Separar álgebra simbólica, almacenamiento tensorial y plan de ejecución.
- Documentar responsabilidades y migración hacia core, GGUF y fachada pública.
- Incorporar diagnóstico reproducible, matriz de tests y gates de aceptación.
- Corregir posteriormente los seis fallos de contratos antes de promover kernels
  o publicar resultados comparativos de rendimiento.

## Capabilities

### New Capabilities

- `symbolic-terms`: términos cerrados/abiertos y evaluación real comprobada.
- `tensor-library-contracts`: límites entre tensor, GGUF, kernels y consumidores.
- `validation-gates`: oráculos, fixtures y medición reproducible.

### Modified Capabilities

Ninguna especificación canónica se modifica en esta primera etapa. Las propuestas
históricas permanecen como contexto; sus supuestos incompatibles se revisarán en
la migración antes de archivarlas como especificaciones vigentes.

## Impact

Implementación inicial: `bml-domain::symbolic`, tests públicos, ejemplo de auditoría
en `bml-tests`, README y documentación. No cambia la semántica de las APIs anteriores.
Migración posterior: domain/parser/compiler/runtime, apps e integración. El cambio
permanece abierto hasta satisfacer los criterios de las tareas pendientes.
