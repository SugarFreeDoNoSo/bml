# Roadmap: contratos y calidad

Base: `57c3bd7` (PR #1). Trabajo iniciado 2026-09-23 y recuperado 2026-09-24.

1. Corregir los seis contratos auditados: reutilización RPN, propiedad del pool de constantes, producto positivo menor que uno, alineación GGUF, orden Q4_0 y rechazo de bloques truncados.
2. Sustituir tests tautológicos y skips que aparentaban éxito. Los tests de modelos externos, strace y perf serán explícitamente optativos; si se solicitan sin su recurso, fallan.
3. Adaptar el proceso de [harness-sdd/uncle-bob-harness](https://github.com/betta-tech/harness-sdd/tree/bb80951622065a32057634d222cf7ed93a2a09cb): escenarios trazables, evidencia TDD, revisión y mutación del código modificado. Sus scripts Python y hooks de Claude no se copian a Rust.
4. Configurar GitHub Actions y hooks Git: tests, doctests, auditoría de contratos, formato, Clippy, cobertura, mutación y dependencias. Conservar los informes también cuando falle un control.
5. Publicar un PR revisable y comprobar sus jobs; no integrar sin revisión.

La siguiente iteración es la biblioteca de tensores descrita en `architecture/TENSOR_LIBRARY.md`: Shape/Tensor/TensorView y kernels CPU de referencia, extracción del lector GGUF, fachada pública y separación de inferencia/tokenizer/runtime. No se marca esa migración como terminada por reparar contratos escalares.

Las mejoras de rendimiento requieren comparar la misma operación con entradas dinámicas y resultados finitos. Completitud simbólica, menor número de nodos y velocidad de ejecución son propiedades distintas.

## Recuperación y evidencia

La limpieza automática del entorno eliminó los archivos locales y logs que aún no estaban publicados. Se reconstruyeron los cambios a partir del registro de la conversación. Los resultados anteriores son históricos; los resultados de cierre deben proceder de una nueva ejecución y quedar en `quality/evidence` y los artifacts de Actions.
