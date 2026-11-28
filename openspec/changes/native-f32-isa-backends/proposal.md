## Why

Las proyecciones F32 nativas de cabeza y FFN necesitan una evaluación de
especialización por ISA. Tener AVX2 en la CPU no demuestra que esas proyecciones
lo utilicen: `crates/kernels/src/lib.rs` compara limbs enteros y su capacidad
SIMD no acredita inferencia F32. La feature histórica `tensor-simd` implementa
operaciones elementales F64, no esta reducción F32 con acumulador F64.

`bml-simd-backends` ya exige selección segura, equivalencia y trabajo vectorizado
declarado. Esta propuesta añade sus criterios específicos para inferencia nativa
sin duplicar ni cerrar esos requisitos. `bml-inference-pipeline` excluye SIMD en
su diseño actual; su tarea 8.6 sobre el DAG simbólico permanece independiente.

## What Changes

- Planificar un control escalar x86-64 y una variante AVX optativa para cuatro
  filas independientes de cabeza/FFN en CLI y ELF nativo.
- Evaluar una variante AVX2 sólo cuando una operación concreta, como packing o
  gather validado, requiera AVX2. Registrar instrucciones y selección efectivas.
- Preservar orden de reducción F64 por fila, semilla `-0.0` de la referencia
  Rust 1.98.1 y cast final F32. Exigir paridad bit a bit en la ruta exacta.
- Separar FMA como experimento optativo con contrato y medición propios.
- Medir el mismo trabajo por ISA, incluyendo preparación/packing, TTFT, decode
  sostenido y memoria; presentar stock llama.cpp como comparación separada.
- Decidir por medición una posible extensión a BMLX simbólico cuando su ruta
  completa esté operativa, conservando su semántica y presupuestos propios.

Este cambio registra trabajo pendiente. No implementa kernels, cambia defaults,
afirma aceleración ni aporta un resultado de BMLX.

## Capabilities

### New Capabilities

Ninguna capacidad canónica nueva.

### Modified Capabilities

- `bml-simd-backends`: especialización optativa de proyecciones F32 nativas,
  identidad de ISA, reducción exacta, controles de inferencia y medición por ISA.

## Impact

La futura implementación podrá afectar `bml-inference`, la selección de kernels
usada por la CLI y el generador/cargador del ELF nativo. Deberá identificar sus
consumidores reales antes de editar código. La propuesta actual sólo modifica
OpenSpec. No modifica el formato BMLX, los límites simbólicos ni los workflows.
