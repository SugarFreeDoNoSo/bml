## 1. Referencia y consumidores

- [ ] 1.1 Identificar los entrypoints F32 efectivos de cabeza/FFN en CLI y ELF nativo, y documentar que comparación de limbs y operaciones elementales F64 no cumplen este requisito.
- [ ] 1.2 Fijar el control escalar Rust 1.98.1: orden por fila, acumulador F64 con semilla `-0.0`, productos/sumas separados y cast final F32; preparar oráculos sensibles a cambios de ese contrato.

## 2. Variantes exactas y selección

- [ ] 2.1 Implementar una variante AVX optativa para cuatro filas independientes, conservando orden de columnas por fila y tratamiento exacto de colas no múltiplos de cuatro.
- [ ] 2.2 Evaluar una variante AVX2 sólo si packing/gather u otra operación identificada requiere instrucciones AVX2; verificar instrucciones y selección efectiva, o documentar con evidencia que la variante sólo requiere AVX sin rotularla AVX2.
- [ ] 2.3 Integrar selección optativa y visible en CLI y ELF, detección de CPU/soporte del sistema operativo, fallback x86-64 convencional y error explícito al forzar una variante no disponible, sin cambiar el default.
- [ ] 2.4 Verificar paridad bit a bit con escalar: signed zero, cancelación, subnormales, cast F32, signos/magnitudes, dimensiones, alineación y colas; incluir controles negativos de semilla, reasociación y FMA cuando cambian el resultado dentro del dominio, y pruebas de fallback; no exigir contraejemplos artificiales fuera del contrato.

## 3. Evidencia y experimentos

- [ ] 3.1 Medir por separado preparación/packing, copias, reutilización, asignaciones, memoria máxima y ejecución, con cotas verificables y fallos limpios.
- [ ] 3.2 Ejecutar benchmark aislado por ISA con mismo GGUF/hash, prompts/IDs, BOS/plantilla, contexto, núcleo/afinidad y presupuesto CPU; alternar orden, repetir y registrar selección/instrucciones efectivas, TTFT, prefill, decode sostenido, dispersión y memoria.
- [ ] 3.3 Contrastar logits, KV e IDs completos con la referencia y presentar stock llama.cpp por separado con revisión/flags/ISA propios, sin atribuir su diferencia exclusivamente a AVX2.
- [ ] 3.4 Evaluar FMA únicamente como experimento optativo separado, con contrato numérico previo, detección de capacidad y equivalencia justificada o reporte de divergencias; no atribuirle automáticamente la paridad de la ruta exacta.

## 4. Decisión para BMLX

- [ ] 4.1 (Deferred: ejecución simbólica completa de 8.6 aún bloqueada.) Cuando la ruta simbólica de 8.6 esté operativa, medir si esta técnica puede extenderse a BMLX preservando operaciones BML, orden, dominio y presupuestos; documentar decisión o bloqueo sin extrapolar resultados de CLI/ELF nativos ni cerrar 8.6.

## 5. Cierre de la implementación futura

- [ ] 5.1 Validar estrictamente OpenSpec y revisar trazabilidad/evidencia de todos los requisitos al finalizar la implementación.
- [ ] 5.2 Ejecutar el gate canónico completo y las pruebas específicas en las ISA disponibles, sin fallos nuevos frente a la base; registrar plataformas no ejecutadas.
- [ ] 5.3 Commit y push de la implementación y evidencia revisables sólo con autorización vigente; ningún cierre o publicación de release implícitos.
