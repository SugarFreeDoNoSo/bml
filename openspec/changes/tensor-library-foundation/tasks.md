## 1. Base verificable

- [x] 1.1 Revisar fuentes y registrar seis discrepancias reproducibles.
- [x] 1.2 Implementar términos cerrados/abiertos, bind, variables libres y collapse.
- [x] 1.3 Implementar interpretación EML/BML explícita y errores de evaluación real.
- [x] 1.4 Añadir 15 tests públicos de contratos y un ejemplo de documentación.
- [x] 1.5 Definir arquitectura destino, alcance MVP y matriz de pruebas.
- [x] 1.6 Añadir diagnóstico ejecutable que falla ante las seis discrepancias previas.

## 2. Corrección de contratos existentes

- [x] 2.1 Reparar reutilización de nodos compartidos en RPN con prueba no adyacente.
- [x] 2.2 Transportar programa y pool juntos en compilación, tests y benchmarks.
- [x] 2.3 Definir y comprobar dominios de reescrituras; cubrir `0.5 * 2`, cero y negativos.
- [x] 2.4 Respetar general.alignment y validar aritmética de rangos GGUF.
- [x] 2.5 Reparar el orden de Q4_0 y rechazar bloques truncados.
- [x] 2.6 Sustituir tests tautológicos, skips exitosos y oráculos compartidos.
- [x] 2.7 Separar el test instrumentado de strace y corregir el bloque del doctest scheduler.

## 3. Biblioteca tensor/GGUF

- [x] 3.1 Crear Shape, Tensor y TensorView con contratos de tamaño y ownership en bml-domain::tensor.
- [x] 3.2 Implementar kernels CPU de referencia y matmul rectangular verificado; EML/BML elementwise con política de dtype explícita.
- [ ] 3.3 Mover reader y codecs a gguf con fixtures F32/F16/Q4_0/Q8_0 y errores tipados.
- [ ] 3.4 Exponer fachada bml y features; mantener adapters de compatibilidad durante la migración.
- [ ] 3.5 Extraer inferencia/tokenizer/KV/sampler y apps de las dependencias del núcleo.

## 4. Evaluación de optimizaciones

- [ ] 4.1 Comparar operaciones iguales con inputs dinámicos, pool completo y oráculo independiente.
- [ ] 4.2 Reportar tiempo, error, memoria y allocs; distinguir bytecode de código máquina.
- [ ] 4.3 Introducir backend experimental solo para dominios validados y con fallback explícito.
- [ ] 4.4 Medir inferencia completa antes de publicar tokens/s comparables. (Deferred: requiere MVP correcto.)
- [ ] 4.5 Introducir complejos/síntesis simbólica con presupuesto y política de ramas. (Deferred: fuera del MVP real.)

## 5. Cierre

- [x] 5.1 `openspec validate tensor-library-foundation` pasa sin errores (OpenSpec 1.13.1, telemetría desactivada).
- [x] 5.2 `cargo test --workspace --locked` pasa, con pruebas instrumentadas separadas explícitamente.
- [x] 5.3 Primera etapa publicada en la rama `docs/tensor-library-contracts` y [PR #1](https://github.com/SugarFreeDoNoSo/bml/pull/1), con validación y limitaciones registradas.
- [ ] 5.4 Archivar el cambio solo al completar o separar formalmente la migración restante.
