## 1. Semántica e identidades

- [x] 1.1 Fijar gramática y semántica real de BML; separar errores estructurales, dominio y evaluación finita. Semántica y separación documentadas en `docs/architecture/BML_NUMBER.md`; `ExactError` no sustituye `EvalError`.
- [x] 1.2 Documentar y revisar derivaciones base 2 de `add`, `mul`, `div`, `neg` y reglas auxiliares, con dominios intermedios y casos de signo/cero; no usar solo aproximaciones como prueba.
- [x] 1.3 Demostrar el magma libre de términos frente a la interpretación real parcial; mantener EML separado e identificar la base de formatos compartidos. Magma libre e interpretación parcial documentados; BMLS identifica BML y rechaza otras bases.
- [x] 1.4 Diseñar y probar las APIs comprobadas de `Add`, `Sub`, `Mul`, `Div` y `Neg` con `Result`, operandos prestados, límites y errores de dominio no justificado. Implementado en `CertifiedBml` y `Bml`; reconocimiento conservador, pruebas de dominios y cadenas en `bml_recognition`.
- [x] 1.5 Implementar `try_add_assign`/`try_mul_assign` transaccionales; diferir `AddAssign`/`MulAssign` sin panic ni errores ignorados. Métodos transaccionales implementados y probados en `CertifiedBml` y `Bml`.
- [x] 1.6 Definir e implementar operandos mixtos numéricos para enteros y `f64` exacto, con fallos propagados, distinción frente a códigos y ejemplos públicos. `ExactOperand` y operadores mixtos certificados cubren enteros nativos y `f64`, con ejemplos y pruebas; integrado en `Bml` y matrices; `numeric_scalar_is_prepared_once_before_the_kernel` verifica conversión única.

## 2. Conversiones exactas

- [x] 2.1 Implementar parseo decimal entero exacto con la gramática acordada, límite de dígitos, denominador `10^d` y reducción por MCD; cubrir `12.375`, `0.0010`, negativos y cero. `ExactRational::from_decimal_str`, sin dependencias; oráculo euclídeo independiente y casos multipalabra en `bml_exact_input`.
- [x] 2.2 Descomponer `f64::to_bits()` en signo/significando/exponente, incluidos normales y subnormales; normalizar ambos ceros y rechazar NaN/infinidades. `ExactRational::from_f64`, probado con mínimo subnormal de ambos signos, mínimo normal y máximo finito; no afirma síntesis exitosa.
- [x] 2.3 Exponer `from_decimal_str_exact` y `from_f64_exact` con éxito certificado o error de búsqueda/recursos; preservar `from_f64` aproximado y comprobar la diferencia entre ambos `0.1`. APIs devuelven `CertifiedBml` con derivación reproducible; `bml_certified` contrasta los dos objetivos de `0.1`.
- [x] 2.4 Verificar independientemente racionales objetivo, derivaciones exactas y errores; cubrir mínimo subnormal de ambos signos sin confundir parseo correcto con síntesis exitosa. Oráculos entero/EUCLID independientes, pruebas de certificados falsos y mínimo subnormal de ambos signos en `bml_exact_input` y `bml_certified`.

## 3. Síntesis y optimalidad

- [x] 3.1 Implementar estrategias comparables de catálogo, factores, potencias de dos, cercanía, sumas/restas y resultados previos; acotar también factorización y cachés.
- [x] 3.2 Contar operadores, bits y profundidad después de expansión completa; probar que `B(4,2)` tiene cuatro operadores y nueve bits; fijar desempates deterministas.
- [x] 3.3 Registrar candidatos, estrategia, presupuestos y terminación; distinguir mejor encontrado de mínimo probado y reportar tiempo/estabilidad por separado.

## 4. Codificación, almacenamiento y flujos

- [x] 4.1 Probar códigos/enteros de `1`, `2`, `4` y `0`, normalización multipalabra, árboles incompletos, bits sobrantes y límites de profundidad/tamaño.
- [x] 4.2 Añadir/verificar exportaciones fallibles `u64`/`u128` y fronteras 31/32 y 63/64 operadores; conservar almacenamiento multipalabra sin truncamiento ni nuevas dependencias implícitas. Ver `number::stream` y `bml_stream::fixed_width_exports_never_truncate`.
- [x] 4.3 Fijar formato de flujo versionado, orden de bits, cantidad, longitud útil y padding; implementar roundtrip de flujos vacíos, hojas `0` iniciales y múltiples valores; rechazar inconsistencias. Formato BMLS v1 y suite `cargo test -p bml-domain --test bml_stream --locked` (5 pruebas).
- [x] 4.4 Medir crecimiento después de expansiones y preservar operandos ante errores; documentar árbol duplicado frente a un eventual DAG y tamaño de código frente a memoria asignada.
- [x] 4.5 Cubrir igualdad estructural frente a igualdad matemática y metadatos de base cuando un formato comparta BML y EML.

## 5. Exactitud y evaluación

- [x] 5.1 Separar validación/certificado simbólico de evaluación numérica en APIs, errores y documentación. `ExactRational`, `ExactDerivation` y `CertifiedBml` separan objetivo, prueba y árbol; evaluación directa permanece explícita.
- [x] 5.2 Incorporar la regresión `2049=B(12,B(2047,1))`: prueba exacta y overflow esperado del evaluador directo `f64`. `small_exact_result_can_have_overflowing_direct_evaluation` comprueba árbol exacto, objetivo y fallo de evaluación.
- [x] 5.3 Definir e implementar las simplificaciones justificadas o la ruta de precisión ampliada que se adopte, con política y presupuesto explícitos; no aceptar tolerancia como prueba exacta.

## 6. Matrices y operadores Rust

- [x] 6.1 Añadir capacidad explícita de matrices `Bml`, validación de formas/longitudes, límites totales y compatibilidad de fachada sin cambiar tensores IEEE/GGUF.
- [x] 6.2 Implementar `A+B` por elemento, `A*B` matricial, `hadamard` y operaciones con escalar precompilado; operadores con `Result` y sin aritmética de códigos.
- [x] 6.3 Probar matrices rectangulares, reducción no vacía desde primer producto, reducción vacía con cero simbólico, orden izquierdo a derecho, signos/cero y errores localizados sin resultados parciales.
- [x] 6.4 Implementar producto generalizado con combinación, acumulación, inicialización y política de dimensión vacía explícitas; probar acumulación no asociativa.
- [x] 6.5 Definir row-major, acceso comprobado, `Index`/`IndexMut`, presupuestos tras mutación, almacenamiento, reutilización de constantes, buffers y costos de asignación.

## 7. AVX2, NEON y rendimiento

- [x] 7.1 Fijar la referencia escalar y elegir trabajo concreto a vectorizar; documentar construcción, representación o evaluación, instrucciones, `exp2`/`log2`, errores y colas.
- [x] 7.2 Implementar detección segura de capacidades y backends AVX2 y NEON con fallback escalar; documentar invariantes de `unsafe`, backend forzado y plataformas.
- [x] 7.3 Contrastar cada backend con casos válidos/erróneos, colas, signos, cero, dominios y límites; respetar primer error y orden de acumulación. Scalar/AVX2 y Scalar/NEON pasan los contratos estructurales y las suites all-features (8/8 por host) en VMs nativas x86-64 y ARM64 de GitHub Actions. Evidencia nativa: `quality/evidence/native-kernels-41853147652.json`; contraste emulado adicional: `quality/evidence/arm-neon-code-comparison.json`. El kernel estructural no evalúa dominios numéricos ni realiza acumulaciones; el tensor IEEE es una API separada.
- [x] 7.4 Ejecutar campañas reproducibles sobre ARM y x86 reales para cada backend acreditado; registrar hardware ausente o campañas no ejecutadas sin atribuir resultados. Campaña nativa ARM64/NEON y x86-64/AVX2 completada en GitHub Actions sobre `6d15db6`: 8/8 contratos por host, tres cargas de 512 pares, 100 iteraciones y siete repeticiones por backend con afinidad fijada. ZIPs, ELF, fuentes, muestras y CPU verificados en `quality/evidence/native-kernels-41853147652.json`. Son VMs de ISA nativa, sin QEMU; no acreditan PMU/L1 ni rendimiento de inferencia.
- [x] 7.5 Definir construcciones BML/s, aplicaciones de B/s, expresiones/s, elementos/s, productos matriciales/s y FLOP/s con convención explícita, sin equivalencias inventadas.
- [x] 7.6 Medir por separado síntesis/preparación, optimización, kernel y operación completa; registrar CPU, flags, entradas, dispersión, memoria, asignaciones y crecimiento; promover solo variantes correctas medidas. Campañas reproducibles de códigos y fases simbólicas en `quality/evidence/bml-code-simd.json` y `quality/evidence/bml-symbolic-phases.json`; siete repeticiones, allocator instrumentado, oráculos previos, CPU/flags/hashes y límites documentados.

## 8. Validación y documentación

- [x] 8.1 Añadir pruebas de propiedades con generadores acotados y oráculos independientes para roundtrip, códigos inválidos, racionales exactos, dominios y presupuestos.
- [x] 8.2 Comparar escalar, AVX2 y NEON disponibles y costos frente a referencia para igual trabajo; registrar tolerancias solo donde el contrato sea numérico aproximado.
- [x] 8.3 Actualizar documentación pública, escenarios y trazabilidad del proyecto con APIs realmente implementadas, evidencia y limitaciones.

## 9. Cierre

- [x] 9.1 Volver a ejecutar `npx --yes --package @fission-ai/openspec@1.13.0 openspec validate bml-exact-numbers-and-matrices --strict` tras completar la implementación y resolver sus hallazgos. Revalidación del 2026-11-03: requisitos extensos divididos en escenarios conservando sus obligaciones; `npx --yes @fission-ai/openspec validate --all --strict` pasa sin avisos. Esto acredita estructura; la evidencia de ejecución nativa ARM figura en 7.3–7.4.
- [x] 9.2 Ejecutar `cargo test --workspace --locked` y `./scripts/check.sh origin/master` (o base disponible), más suites de features afectadas; separar evidencia ARM y pruebas no ejecutadas. Gate histórico registrado en `target/quality/spec-development-gate.log`: auditoría cero discrepancias, traza 39/39, cero avisos nuevos. Fachada all-features y kernels nativos pasan; nuevo ejemplo de fases ejecutado en release y comprobado con rustfmt/Clippy. La ejecución ARM posterior está documentada en 7.3–7.4.
- [ ] 9.3 Preparar commit y push de la entrega cuando se soliciten; archivar solo al completar los contratos y registrar la evidencia de este cambio. Commit y push solicitados y realizados: `ca8d030` y `1ef5f5d`, con remoto verificado en el segundo. Evidencia: `quality/evidence/publication-spec-closure-2026-11-03.md`. La parte de publicación está cumplida; el archivo requiere una auditoría completa de los contratos y su evidencia, incluidas las campañas nativas posteriores de 7.3–7.4.
