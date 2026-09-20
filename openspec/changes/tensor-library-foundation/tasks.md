## 1. Base verificable

- [x] 1.1 Revisar fuentes y registrar seis discrepancias reproducibles.
- [x] 1.2 Implementar términos cerrados/abiertos, bind, variables libres y collapse.
- [x] 1.3 Implementar interpretación EML/BML explícita y errores de evaluación real.
- [x] 1.4 Añadir 15 tests públicos de contratos y un ejemplo de documentación.
- [x] 1.5 Definir arquitectura destino, alcance MVP y matriz de pruebas.
- [x] 1.6 Añadir diagnóstico ejecutable que falla ante las seis discrepancias previas.

## 2. Corrección de contratos existentes

- [ ] 2.1 Reparar reutilización de nodos compartidos en RPN con prueba no adyacente.
- [ ] 2.2 Transportar programa y pool juntos en compilación, tests y benchmarks.
- [ ] 2.3 Definir y comprobar dominios de reescrituras; cubrir `0.5 * 2`, cero y negativos.
- [ ] 2.4 Respetar general.alignment y validar aritmética de rangos GGUF.
- [ ] 2.5 Reparar el orden de Q4_0 y rechazar bloques truncados.
- [ ] 2.6 Sustituir tests tautológicos, skips exitosos y oráculos compartidos.
- [ ] 2.7 Separar el test instrumentado de strace y corregir el bloque del doctest scheduler.

## 3. Biblioteca tensor/GGUF

- [ ] 3.1 Crear Shape, Tensor y TensorView con contratos de tamaño y ownership.
- [ ] 3.2 Implementar kernels CPU de referencia y matmul rectangular verificado.
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
- [ ] 5.2 `cargo test --workspace --locked` pasa, con pruebas instrumentadas separadas explícitamente.
- [ ] 5.3 Commit y push de la primera etapa en rama y PR de revisión; registrar validación y limitaciones.
- [ ] 5.4 Archivar el cambio solo al completar o separar formalmente la migración restante.
