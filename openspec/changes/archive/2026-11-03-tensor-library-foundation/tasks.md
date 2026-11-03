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
- [x] 3.3 Mover reader y codecs a gguf con fixtures F32/F16/Q4_0/Q8_0 y errores tipados. Reconstruido el 2026-10-22; ver quality/evidence/gguf-recovery.json.
- [x] 3.3.1 Integrar vistas GGUF con Tensor y conversión explícita GGML/row-major (#19). Reconstrucción con validación nueva tras pérdida del entorno; ver quality/evidence/gguf-tensor-views-reconstruction.json.
- [x] 3.4 Exponer fachada bml y features; mantener adapters de compatibilidad durante la migración. Implementado en `bml-number-foundation`, con prioridad en el escalar Bml y tensor/GGUF optativos.
- [x] 3.5 Extraer inferencia/tokenizer/KV/sampler y apps de las dependencias del núcleo. Grafo normal/build transitivo verificado con `cargo metadata --all-features --locked`: fachada bml, domain, gguf y kernels no dependen de aplicaciones/compiler/runtime/inference; domain no tiene dependencias. Evidencia: `quality/evidence/numeric-dependency-boundary.json`. El AST/transformer histórico permanece como compatibilidad, sin confundirse con la inferencia causal externa.

## 4. Evaluación de optimizaciones

- [x] 4.1 Comparar operaciones iguales con inputs dinámicos, pool completo y oráculo independiente. `fma_vs_bml` compara la misma recurrencia finita en escalar/RPN compartido/no compartido, con 24 casos y oráculo exacto de paridad. Medición corta y muestras en `quality/evidence/equal-work-rpn-benchmark.json`; VM con carga concurrente, sin afirmaciones de rendimiento dedicado ni tokens/s.
- [x] 4.2 Reportar tiempo, error, memoria y allocs; distinguir bytecode de código máquina. `recurrence_metrics` comparte el workload del benchmark, mide siete muestras por caso con contador System verificado y separa payload/capacidad de programa, reservas y pico adicional de heap. Los 24 casos tienen error exacto cero. `tools/measure_recurrence.py` reproduce `quality/evidence/equal-work-rpn-memory.json`; tiempos instrumentados y memoria solicitada, no RSS ni código máquina. Tamaño de símbolo máquina medido aparte en `bml-hot-loop-code-size.json`.
- [x] 4.3 Introducir backend experimental solo para dominios validados y con fallback explícito. Feature optativa `tensor-simd`: suma/multiplicación f64 contigua con política Reference por defecto o PreferAvx2, selección visible y fallback a kernels domain. Conserva forma/IEEE, sin reducciones, aproximaciones exp/log ni promoción automática. Casos de colas, subnormales, ceros con signo, NaN/inf y errores de forma en `crates/kernels/tests/tensor_simd.rs`; evidencia en `quality/evidence/tensor-simd-validation.json`. ARM solo compilado, no ejecutado.
- [x] 4.4 Medir inferencia completa antes de publicar tokens/s comparables. Campaña Stories15M F32 del 2026-11-02 con hashes de artefactos, pares con tokens/texto idénticos y gate separado de logits; snapshot conservado en `docs/benchmarks/inference-2026-11-02.json`. No son tokens equivalentes sintéticos ni una medición del árbol posterior.
- [x] 4.5 Introducir complejos/síntesis simbólica con presupuesto y política de ramas. `symbolic::complex` implementa la interpretación compleja finita con rama fija, corte firmado y límites de nodos/profundidad; `symbolic::synthesis` enumera árboles hasta 255 nodos con límites de intentos, tolerancia y estados explícitos de agotamiento. C73–C74: siete contratos y nueve controles negativos; original/restaurado y gates aislado/completo aprobados. Evidencia: `quality/evidence/symbolic-extensions-2026-11-03.md`. No se atribuye síntesis exacta ni evaluación de complejos extendidos.

## 5. Cierre

- [x] 5.1 `openspec validate tensor-library-foundation` pasa sin errores (OpenSpec 1.13.1, telemetría desactivada).
- [x] 5.2 `cargo test --workspace --locked` pasa, con pruebas instrumentadas separadas explícitamente.
- [x] 5.3 Primera etapa publicada en la rama `docs/tensor-library-contracts` y [PR #1](https://github.com/SugarFreeDoNoSo/bml/pull/1), con validación y limitaciones registradas.
- [x] 5.4 Completar la migración restante y archivar el cambio. 3.5 y 4.1–4.5 tienen implementación y evidencia enlazada; el gate del árbol completo acredita 736 pruebas Rust, 82 Python y 74/74 contratos. El archivo conserva los límites de cada campaña; los contratos externos y el gate global de mutaciones siguen en sus cambios propios.
