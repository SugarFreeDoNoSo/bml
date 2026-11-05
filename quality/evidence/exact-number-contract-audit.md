# Auditoría de contratos de números exactos y matrices

Fecha: 2026-11-05 UTC. Revisión base auditada: `9aa114a9d74e72e8f09d45e367df5caaa936b375`. El diff de esta tarea solo modifica documentación y evidencia. Las seis specs normativas se conservan sin cambios.

Esta revisión cubre **29 requisitos y 64 escenarios** mediante lectura de implementación, aserciones de las pruebas, documentación y evidencias registradas. No se ejecutó Cargo en este worktree: la integración centraliza builds y pruebas. **S** significa que el comportamiento aplicable está sustentado por las fuentes y pruebas concretas descritas; no significa una ejecución nueva de la prueba por esta auditoría. **C** significa condición no activada por la API o campaña elegida, con justificación explícita. **G** identifica una brecha de evidencia de ejecución específica; se conserva la obligación y no se da por cerrada. No se atribuye cumplimiento por un nombre de test o un checkbox.

El gate del snapshot integrado pasó con cero discrepancias, traza 105/105 y cero diagnósticos nuevos; sus hashes de código se conservan en [wave-one-gate.json](wave-one-gate.json). El cambio se archivó tras la validación; véase [exact-number-archive.json](exact-number-archive.json). La auditoría no cambia los contratos de inferencia, PMU, otras posiciones de TinyLlama ni el gate global de mutaciones de la entrega.

## Fuentes y ejecución acreditada

- Las rutas de código mencionadas sin prefijo viven en `crates/domain/src/number/`, salvo `symbolic::Basis` en `crates/domain/src/symbolic.rs` y `kernels` en `crates/kernels/src/lib.rs`.
- DOC: [BML_NUMBER.md](../../docs/architecture/BML_NUMBER.md), incluidas las derivaciones base 2 y las secciones de matrices, búsqueda, SIMD y métricas.
- Las pruebas siguientes se identifican por paquete Cargo, target y función; no son pruebas sueltas no registradas. Las funciones nombradas se inspeccionaron y sus aserciones se describen por escenario.

| Alias | Paquete / target | Fuente |
|---|---|---|
| ENC | `bml-domain` / `bml_encoding` | [bml_encoding.rs](../../crates/domain/tests/bml_encoding.rs) |
| STREAM | `bml-domain` / `bml_stream` | [bml_stream.rs](../../crates/domain/tests/bml_stream.rs) |
| INPUT | `bml-domain` / `bml_exact_input` | [bml_exact_input.rs](../../crates/domain/tests/bml_exact_input.rs) |
| CERT | `bml-domain` / `bml_certified` | [bml_certified.rs](../../crates/domain/tests/bml_certified.rs) |
| CHECKED | `bml-domain` / `bml_checked` | [bml_checked.rs](../../crates/domain/tests/bml_checked.rs) |
| RECOG | `bml-domain` / `bml_recognition` | [bml_recognition.rs](../../crates/domain/tests/bml_recognition.rs) |
| SEARCH | `bml-domain` / `bml_search` | [bml_search.rs](../../crates/domain/tests/bml_search.rs) |
| SIMPLE | `bml-domain` / `bml_simplified_eval` | [bml_simplified_eval.rs](../../crates/domain/tests/bml_simplified_eval.rs) |
| MATRIX | `bml-domain` / `bml_matrix` | [bml_matrix.rs](../../crates/domain/tests/bml_matrix.rs) |
| MBOUND | `bml-domain` / `bml_matrix_boundaries` | [bml_matrix_boundaries.rs](../../crates/domain/tests/bml_matrix_boundaries.rs) |
| BPROP | `bml-domain` / `bml_bounded_properties` | [bml_bounded_properties.rs](../../crates/domain/tests/bml_bounded_properties.rs) |
| OPS | `bml-domain` / `bml_operations` | [bml_operations.rs](../../crates/domain/tests/bml_operations.rs) |
| CONV | `bml-domain` / `bml_conversion` | [bml_conversion.rs](../../crates/domain/tests/bml_conversion.rs) |
| SYM | `bml-domain` / `symbolic_contracts` | [symbolic_contracts.rs](../../crates/domain/tests/symbolic_contracts.rs) |
| KCODE | `bml-kernels` / `code_comparison` | [code_comparison.rs](../../crates/kernels/tests/code_comparison.rs) |

- E-NATIVE: [native-kernels-41853147652.json](native-kernels-41853147652.json). Campaña nativa ARM64/x86_64 del commit `6d15db6`; su propio verificador registró ZIPs, archivos y ELF. Esta auditoría volvió a comparar los 40 hashes de fuentes de **cada** manifiesto con el árbol base: 80/80 coinciden. Los tests son los ocho de kernels all-features por host, no todas las suites de dominio.
- E-PHASE: [bml-symbolic-phases.json](bml-symbolic-phases.json). Campaña sintética x86 del snapshot histórico `40b3f1b`, con 12 fases y siete muestras por fase; 13/15 hashes de fuentes coinciden. Las dos diferencias son los manifiestos raíz (miembros y dependencias añadidos), no la implementación numérica ni el ejemplo; el perfil release no cambió. Se conserva como medición histórica reproducible, no medición nueva exacta de 9aa114a.
- E-NOAVX: [x86-without-avx2.json](x86-without-avx2.json). Ejecución funcional QEMU x86_64, CPU Nehalem sin AVX2, cuatro contratos KCODE aprobados. Misma revisión de fuentes que este árbol; no rendimiento ni CPU nativa. El integrador añade este archivo junto con la auditoría.
- E-CODE: [bml-code-simd.json](bml-code-simd.json). Primera campaña x86; solo 1/4 hashes coincide actualmente. La ejecución funcional/performance actual de comparación estructural se respalda con E-NATIVE; no se atribuyen sus tiempos históricos al código nuevo.
- [workspace-gate-1aa00b0.json](workspace-gate-1aa00b0.json) registra gate completo exit0, auditoría histórica0, traza105/105 y Clippy sin nuevos diagnósticos. `git diff 1aa00b0 9aa114a -- crates/domain crates/kernels` afecta solo ampliaciones de `bml_recognition.rs` y `bml_stream.rs`: ninguna implementación numérica/kernels cambió. Las pruebas añadidas posteriormente requieren el resultado integrado del gate; no se presupone que se ejecutaran en el gate antiguo.

## Cobertura por requisito y escenario

| Capacidad | Requisitos | Escenarios |
|---|---:|---:|
| bml-code-storage | 6 | 17 |
| bml-exact-synthesis | 7 | 15 |
| bml-rust-operators | 4 | 8 |
| bml-symbolic-matrices | 5 | 12 |
| bml-simd-backends | 4 | 6 |
| bml-performance | 3 | 6 |
| **Total** | **29** | **64** |

### bml-code-storage

Norma: [bml-code-storage/spec.md](../../openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-code-storage/spec.md).

| ID | Requisito | Obligaciones auditadas | Fuente y contraste |
|---|---|---|---|
| bml-code-storage/R01 | Código individual canónico de un árbol BML | Preorden ordenado, hoja entero 0, palabras LE normalizadas y validación de exactamente un árbol; la sintaxis no garantiza dominio real. | `encoding::{from_binary,from_words_le,validate,op_with_limit}`; ENC y BPROP. |
| bml-code-storage/R02 | Igualdad estructural y base identificada | Igualdad/hash estructurales; equivalencia matemática separada y base identificada si un contenedor compartiera interpretaciones. | `number::Bml` deriva Eq/Hash sobre palabras; SEARCH::logical_growth_storage_and_structural_equality_are_separate; STREAM rechaza base distinta de2. |
| bml-code-storage/R03 | Tamaño del árbol y exportación a palabras fijas | Contar 2n+1 bits incluidas hojas; exportar hasta31/63 operadores con error tipado al exceder u64/u128; distinguir memoria lógica, capacidad y memoria medida. | `stream::{operator_count,try_to_u64,try_to_u128}`; `search::TreeCost`; STREAM y SEARCH; E-PHASE/E-NATIVE para memoria medida. |
| bml-code-storage/R04 | Almacenamiento multipalabra con crecimiento acotado | Mantener multipalabra sin dependencia BigUint; calcular crecimiento expandido y tamaños con aritmética comprobada; error sin modificar operandos al exceder presupuesto. | `encoding::op_with_limit` usa checked_add antes de reservar y copia ambos árboles; Cargo de dominio sin dependencias; ENC/STREAM/SEARCH. |
| bml-code-storage/R05 | Flujo versionado que conserva todos los bits útiles | BMLS v1:24 bytes, magic/versión/base/reservados, count/bit_length u64LE, payload MSB-first, padding 0; count/bits/profundidad y tamaño exacto antes de reservar al leer. | `stream::{encode_stream,decode_stream,check_depth}`; STREAM prueba bytes canónicos, truncamiento, profundidad y tamaños extremos; lector hace primera pasada antes de try_reserve_exact(count). |
| bml-code-storage/R06 | Un DAG requiere un formato diferente | Duplicar cada subárbol en formato árbol y contabilizar cada aparición; un futuro DAG exige otro formato, sin promesa actual de compresión. | `encoding::op_with_limit` copia operands en rangos separados; SEARCH::expanded_cost_counts_duplicate_subtrees_and_depth; BMLS no tiene referencias compartidas. |

| ID | Escenario | Contenido comprobado y evidencia específica | Aplicabilidad |
|---|---|---|---|
| bml-code-storage/S01 | Almacenamiento y validez estructural | `from_words_le` quita palabras altas0; `validate` usa slots de preorden; ENC::large_codes_round_trip_across_words_and_normalize y valid_structure_can_fail_real_evaluation separan canonicalidad de dominio. | S |
| bml-code-storage/S02 | Ejemplos de valores y enteros codificados | ENC::prefix_codes_preserve_operand_order_and_evaluate comprueba1/2/4; SEARCH::logical_growth_storage_and_structural_equality_are_separate exige cero1011000/88; tablas DOC y construcción `op` conservan orden. | S |
| bml-code-storage/S03 | Árbol incompleto o símbolos sobrantes | ENC::rejects_invalid_codes comprueba1000/00/1 y from_u64(3); BPROP::exhaustive_small_prefix_codes_match_independent_grammar_and_streams recorre todas cadenas1..13 bits e incluye10, con parser recursivo independiente. | S |
| bml-code-storage/S04 | Sintaxis válida fuera del dominio real | ENC::valid_structure_can_fail_real_evaluation construye segundo argumento cero y exige LogDomain; RECOG rechaza certificarlo; `Basis::apply_real` controla >0. | S |
| bml-code-storage/S05 | Árboles distintos de igual valor | SEARCH::logical_growth_storage_and_structural_equality_are_separate compara uno y B(1, B(1, 1)): árboles distintos con mismo racional certificado; derive Eq/Hash usa palabras, no floats. BMLS identifica base 2 y rechaza3 en STREAM. | S |
| bml-code-storage/S06 | Contabilidad del almacenamiento | STREAM::fixed_width_exports_never_truncate exige2n+1; SEARCH::expanded_cost_counts_duplicate_subtrees_and_depth separa bits/words/capacity; E-PHASE y E-NATIVE miden allocator y declaran que no es RSS. | S |
| bml-code-storage/S07 | Frontera de u64 | STREAM::fixed_width_exports_never_truncate recorre0..64 operadores, acepta31 y rechaza32 para u64; implementación exige bit_len<=64 sin truncar. | S |
| bml-code-storage/S08 | Frontera de u128 | La misma prueba acepta63 y rechaza64 para u128 y reconstruye ambos words desde cada exportación exitosa; implementación exige bit_len<=128. | S |
| bml-code-storage/S09 | El resultado supera el ancho de sus operandos | La transición31→32 en la prueba de anchos usa operandos aún exportables antes de op; conserva multipalabra y el siguiente try_to_u64 falla. ENC además recorre1000 operadores/2001 bits. | S |
| bml-code-storage/S10 | La expansión excede el presupuesto | ENC::construction_enforces_the_requested_bit_limit verifica2vs3 bits; SEARCH::logical_growth_storage_and_structural_equality_are_separate conserva operando al rechazar6 bits para resultado7; CERT verifica síntesis/certificado bajo límites sin aproximación. | S |
| bml-code-storage/S11 | Contrato del contenedor versionado | STREAM::wire_format_preserves_leaves_and_empty_streams exige magic/version/base/reservados y vectores conocidos; stream_header_uses_little_endian_u64_fields_across_byte_boundary verifica ambos campos256u64LE. | S |
| bml-code-storage/S12 | Orden y relleno del payload | STREAM exige payload 0x40 para bits 01000 y0x00 para hojas; malformed_streams_and_limits_are_rejected exige Padding al recibir0x41. `encode_stream` escribe MSB primero y tamaño ceil(bits/8). | S |
| bml-code-storage/S13 | Consumo exacto y límites antes de reservar | STREAM prueba cada truncamiento, bytesextra, count/longitud inconsistentes, versión/base/reservado, presupuesto y profundidad; oversized_stream_headers_fail_before_payload_allocation cubre u64::MAX. Inspección `decode_stream`: verificación de tamaño y primera pasada exacta antes de reservar resultados. | S |
| bml-code-storage/S14 | Hojas iniciales y finales | STREAM::wire_format_preserves_leaves_and_empty_streams comprueba[one, two, one], 5 bits ypayload 0x40 con roundtrip; no pierde hojas de código 0. | S |
| bml-code-storage/S15 | Flujo vacío y ceros consecutivos | La misma prueba exige cabecera24 bytes sin payload para vacío y2 bits/payload 0 para[one, one]. | S |
| bml-code-storage/S16 | Cabecera o relleno inconsistente | STREAM::malformed_streams_and_limits_are_rejected altera count 3→2, trunca cada byte y cambia0x40→0x41, con errores Length/Padding esperados. | S |
| bml-code-storage/S17 | Subárbol repetido | SEARCH::expanded_cost_counts_duplicate_subtrees_and_depth usa t=15(k4): B(t, t) tiene9 operadores/19 bits. `op_with_limit` suma ambos bit_len+1 y copia ambos payloads; BMLS no permite referencias a nodos. | S |

### bml-exact-synthesis

Norma: [bml-exact-synthesis/spec.md](../../openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-exact-synthesis/spec.md).

| ID | Requisito | Obligaciones auditadas | Fuente y contraste |
|---|---|---|---|
| bml-exact-synthesis/R01 | Semántica BML y dominio de las identidades | Representar solo1/B; magma libre ordenado frente a interpretación real parcial; justificar reglas base 2 y dominios intermedios, con EML separado. | `Bml`, `encoding`, `operations`, `symbolic::Basis`; DOC§Justificación algebraica; ENC/OPS/SYM/CERT/RECOG. |
| bml-exact-synthesis/R02 | Conversión decimal exacta | Parsear [+-]?[0-9]+(\.[0-9]+)? como entero/10^d reducido por MCD sin f64; rechazar gramática/límites antes de expansión ilimitada; éxito exacto certificado. | `exact::ExactRational::from_decimal_str`, `certified::Bml::from_decimal_str_exact`; INPUT/CERT. |
| bml-exact-synthesis/R03 | Conversión exacta del valor almacenado en f64 | Descomponer to_bits en signo/significando/exponente incluyendo subnormales; normalizar ceros y rechazar no finitos; conservar API aproximada, sin tolerancia en éxito exacto. | `exact::ExactRational::from_f64`; `certified::Bml::from_f64_exact`; `convert` conserva API aproximada; INPUT/CERT/CONV. |
| bml-exact-synthesis/R04 | Éxito exacto comprobable y fallos acotados | Éxito exacto con derivación reproducible desde racional; distinguir entrada, dominio, recursos y búsqueda sin certificado; nunca aceptar aproximación o prometer síntesis universal. | `ExactError`, `ExactDerivation::{certify,verify,replay}`, `ExactSearchResult`; CERT::proof_replay_rejects_false_claims_forward_references_and_domains y exact_synthesis_limits_do_not_turn_into_approximate_success. |
| bml-exact-synthesis/R05 | Operaciones puras con signos y casos especiales | Operaciones con signo/cero expandidas a1/B y dominios probados; rechazar división por cero o dominio no justificado sin cambiar operandos. | `checked::{checked_add,checked_mul,checked_div,checked_neg}`; CHECKED/CERT/RECOG/BPROP; DOC§Aritmética de valores certificados. |
| bml-exact-synthesis/R06 | Búsqueda y criterio de optimalidad explícitos | Buscar estrategias aplicables con límites; coste expandido por operadores, profundidad y orden lexicográfico; reportar estrategia/presupuesto/terminación; tiempo/estabilidad separados y sin mínimo global no probado. | `search::{ExactSearcher,ExactSearchReport,TreeCost}`; SEARCH incluye estrategias Previous/NearPower/Sum/Difference/Factors/PowerOfTwo/Catalogue y candidate_limit/factor_limit/cache. |
| bml-exact-synthesis/R07 | Exactitud simbólica independiente de evaluación numérica | Separar estructura, dominio, certificado y evaluación; evaluador f64 directo mantiene errores; simplificar solo con prueba y redondeo/presupuesto explícitos. | `encoding::to_f64`, `Basis::apply_real`, `evaluate::to_f64_simplified`, `ExactRational::to_f64_rounded`; CERT/SIMPLE/SYM; DOC§Evaluación simplificada. |

| ID | Escenario | Contenido comprobado y evidencia específica | Aplicabilidad |
|---|---|---|---|
| bml-exact-synthesis/S01 | Justificación de una regla real | DOC§Justificación algebraica deriva E, L, D, adición/negación/resta y regla positiva, con p=2^x, q=p-x>0. `ExactDerivation::replay` verifica racionales y dominios; CERT::positive_rules_reject_each_nonpositive_operand cubre las cuatro posiciones de producto/cociente. | S |
| bml-exact-synthesis/S02 | Árbol válido fuera del dominio real | ENC::valid_structure_can_fail_real_evaluation, OPS::construction_does_not_hide_evaluation_domain_errors y SYM::real_domain_rejects_zero_signed_zero_and_negatives distinguen gramática/dominio; RECOG::unsupported_domains_and_budgets_do_not_gain_certificates no emite certificado inválido. | S |
| bml-exact-synthesis/S03 | El generador no es identidad del magma | `Bml::one` es palabras vacías; `one.op(one)` código 4/100 y no igual al generador 0, verificado en ENC. DOC declara magma libre, no identidad del constructor. | S |
| bml-exact-synthesis/S04 | Interpretación EML y BML separadas | SYM::basis_is_explicit_and_changes_interpretation usa el mismo árbol con Basis::Eml y Basis::Bml; DOC separa exp/ln de exp2/log2. BMLS admite solo base 2: STREAM rechaza base 3; no contenedor dual implícito. | S |
| bml-exact-synthesis/S05 | Fracción decimal reducida | INPUT::decimal_targets_are_reduced_without_float_rounding comprueba12.375→99/8; CERT::exact_conversion_proves_targets_without_float_acceptance exige target y replay exactos; no parse f64 en la ruta exacta. | S |
| bml-exact-synthesis/S06 | Ceros iniciales y gramática | INPUT comprueba0.0010→1/1000, -0.125→-1/8, -0.000→0/1; grammar_and_each_resource_budget_are_enforced rechaza coma, exponente, vacío y límites de dígitos; byte_preflight incluye signo/ceros. | S |
| bml-exact-synthesis/S07 | Decimal y float con el mismo texto visible | INPUT exige decimal0.1=1/10 y f64(0.1)=3602879701896397/36028797018963968 y desigualdad; CHECKED usa operando f64 exacto y CERT distingue éxito certificado de aceptación flotante. | S |
| bml-exact-synthesis/S08 | Subnormales, ceros y valores no finitos | INPUT::ieee_fields_cover_subnormals_normals_and_zero inspecciona words del denominador2^1074 y ambos signos del menor subnormal. Es prueba del racional objetivo; no promesa de síntesis de ese árbol dentro del presupuesto. | S |
| bml-exact-synthesis/S09 | Política de ceros y valores no finitos | INPUT normaliza ambos ceros a racional no negativo 0/1 y rechazaNaN/±inf con NonFinite; CONV conserva API aproximada separada; documentación no promete bits de cero IEEE. | S |
| bml-exact-synthesis/S10 | Presupuesto insuficiente | CERT::exact_synthesis_limits_do_not_turn_into_approximate_success prueba bits, pasos y recursión insuficientes; exact_failure_diagnostics_are_nonempty_and_keep_typed_causes comprueba causas. SEARCH con candidatos 0 devuelve best=None/BudgetExhausted. | S |
| bml-exact-synthesis/S11 | Multiplicación y división con signo | CHECKED::certified_arithmetic_covers_signs_zero_and_division recorre[-2, 2]^2 con resultados enteros y replay; cero divisor exige DivisionByZero, y operandos se comparan al final; `checked` usa magnitudes probadas, no f64. | S |
| bml-exact-synthesis/S12 | Selección y registro de candidatos | SEARCH::strategies_are_auditable_and_selected_cost_is_minimum_among_attempts verifica estrategias/coste mínimo entre intentos; failures_are_reported_without_approximate_success_and_ties_are_deterministic repite ranking y cache; `search` desempata por código lexicográfico después de operadores/profundidad y reporta minimum_proven=false. | S |
| bml-exact-synthesis/S13 | Constante 15 y costo expandido | SEARCH::expanded_cost_counts_duplicate_subtrees_and_depth exige15=B(4, 2), 4 operadores, 9 bits, profundidad3; search 15 usa Catalogue pero no mínimo global probado. | S |
| bml-exact-synthesis/S14 | Una heurística encuentra un candidato | SEARCH::budgets_limit_candidates_factorization_and_cache obtiene best=Some con factor_limit_reached y BudgetExhausted; numeric_observations_are_explicit_and_cannot_change_exact_ranking conserva selección tras medir evaluación y mantiene minimum_proven=false. | S |
| bml-exact-synthesis/S15 | Valor pequeño con intermedio enorme | CERT::small_exact_result_can_have_overflowing_direct_evaluation exige estructura B(12, B(2047, 1)), target2049 y overflow directo; SIMPLE::proven_simplification_avoids_overflow_without_changing_direct_evaluation da2049 tras replay y redondeo, sin cambiar to_f64 directo. | S |

### bml-rust-operators

Norma: [bml-rust-operators/spec.md](../../openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-rust-operators/spec.md).

| ID | Requisito | Obligaciones auditadas | Fuente y contraste |
|---|---|---|---|
| bml-rust-operators/R01 | Primitiva BML distinta de aritmética derivada | Bml::op permanece construcción ordenada sin evaluación; operadores representan aritmética derivada con reglas base 2 y dominios, sin aritmética de códigos; preservar métodos históricos. | `encoding::op`, `operations::Rules`, `checked` y `recognize`; RECOG::structural_methods_remain_distinct_from_checked_operators; DOC§Aritmética de valores certificados. |
| bml-rust-operators/R02 | Operadores fallibles y referencias | Add/Sub/Mul/Div/Neg devuelven Result tipado y admiten préstamos; delegan en APIs comprobadas; documentar uso con?, dominios y presupuestos; error sin salida parcial. | Macros de `checked` para Bml/CertifiedBml y órdenes de operandos; CHECKED/RECOG; DOC§Recuperación conservadora de pruebas. |
| bml-rust-operators/R03 | Asignación fallible transaccional | try_add_assign/try_mul_assign construyen temporal y solo sustituyen tras éxito; no AddAssign/MulAssign con pánico o errores descartados. | `checked` reemplaza después de checked_* exitoso; CHECKED::assignments_commit_only_complete_results y RECOG::raw_tree_assignment_failures_are_transactional. |
| bml-rust-operators/R04 | Operandos mixtos explícitos | Operandos enteros son valores y f64 es racional IEEE exacto, fallos se propagan; variantes simétricas documentadas/probadas; escalar matricial se prepara una vez. | `ExactOperand` y macros reverse_numeric/reverse_tree_numeric; CHECKED/RECOG; MATRIX::numeric_scalar_is_prepared_once_before_the_kernel; DOC§Aritmética/§Matrices. |

| ID | Escenario | Contenido comprobado y evidencia específica | Aplicabilidad |
|---|---|---|---|
| bml-rust-operators/S01 | Compatibilidad y APIs comprobadas | RECOG::structural_methods_remain_distinct_from_checked_operators construye métodos históricos sobre árbol no certificable y contrasta DomainNotJustified del operador; `checked` no cambia `operations` ni su error BuildError. | S |
| bml-rust-operators/S02 | Construcción de primitiva frente a suma | La misma prueba distingue estructuralmente one.op(one) de suma comprobada, aunque ambos tengan racional 2; DOC explica construcción frente a aritmética derivada. | S |
| bml-rust-operators/S03 | Delegación y justificación del dominio | `checked` certifica operandos, compone/reproduce pruebas y propaga ExactError; RECOG::unsupported_domains_and_budgets_do_not_gain_certificates incluye sqrt2 real sin certificado racional. DOC ofrece usos con? y presupuestos. | S |
| bml-rust-operators/S04 | Suma prestada con resultado fallible | RECOG::raw_tree_operators_preserve_order_and_support_chaining usa&a+&b con racional 1, variantes propias/prestadas, cadena y comparación final de operandos; CHECKED realiza el mismo contrato en certificados. | S |
| bml-rust-operators/S05 | División o dominio no admisible | RECOG::raw_tree_assignment_failures_are_transactional exige DivisionByZero para cero certificado y DomainNotJustified para árbol inválido; CHECKED recorre todos signos y cero divisor, sin inferirlo de to_f64. | S |
| bml-rust-operators/S06 | Presupuesto insuficiente al asignar | CHECKED::assignments_commit_only_complete_results conserva destino/rhs ante max_bits10; RECOG conserva raw trees ante presupuesto y dominio inválidos en ambas asignaciones y comprueba éxito posterior. | S |
| bml-rust-operators/S07 | Variantes simétricas y preparación matricial | CHECKED::mixed_operands_are_values_and_preserve_order comprueba4+a, 4-a, a-4, 4*a, 4/a y0.1IEEE; RECOG añade0.5f64 en ambos órdenes; MATRIX usa ExactOperand Counted y exige una conversión. DOC limita/documenta variantes inversas prestadas. | S |
| bml-rust-operators/S08 | Entero como valor frente a código | CHECKED compara1+4→5 mientras from_u64(4) evalúa2; RECOG suma -2+4→2 y propagaNaN; MATRIX::numeric_scalar_is_prepared_once_before_the_kernel añade4a cada valor, con resultados enteros independientes. | S |

### bml-symbolic-matrices

Norma: [bml-symbolic-matrices/spec.md](../../openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-symbolic-matrices/spec.md).

| ID | Requisito | Obligaciones auditadas | Fuente y contraste |
|---|---|---|---|
| bml-symbolic-matrices/R01 | Matrices con elementos Bml y formas comprobadas | Capacidad explícita de Bml preparados, row-major con formas, longitudes, overflow y presupuestos validados; conservar códigos sin conversión implícita IEEE; declarar costes/copias/layout. | `matrix::{BmlMatrix,new,count,prepare}`; MATRIX/MBOUND; DOC§Matrices simbólicas y §Localidad y asignaciones; E-PHASE. |
| bml-symbolic-matrices/R02 | Operaciones matriciales simbólicas y fallibles | A+B elemento a elemento, A*B matricial, hadamard explícito; escalar una vez; kernels de operaciones simbólicas comprobadas, errores localizados/transaccionales. | `matrix::{elementwise,scalar_map,matmul_with_limits}`; matrices inmutables prestadas y buffers locales; MATRIX/MBOUND. |
| bml-symbolic-matrices/R03 | Acceso e invariantes de matrices | Index y acceso comprobado; invariantes de forma/longitud/bits; sustitución fallida transaccional; si hubiera IndexMut revalidar mutaciones. | `BmlMatrix::{get,replace,validate}`, campos privados, Index sin IndexMut; MATRIX/MBOUND. La cláusula IndexMut es condicional y no activa una API nueva. |
| bml-symbolic-matrices/R04 | Orden de reducción y presupuestos de matriz | Reducción no vacía desde primer producto, izquierda a derecha; vacío con cero simbólico; presupuestar elementos, salida e intermedios duplicados; sin reasociación tácita. | `matmul_with_limits`, `charge`, `retain`, `MatrixLimits`; MATRIX::rectangular_product_matches_independent_integer_oracle y empty_products_use_symbolic_zero_and_bound_zero_size_shapes; MBOUND. |
| bml-symbolic-matrices/R05 | Producto matricial generalizado | Producto generalizado con combinación, acumulación, inicio y política vacía explícitos; paréntesis fijos sin presumir identidad/asociatividad deB. | `generalized_product`, `ReductionInit::{First,Value}`; MATRIX::generalized_product_preserves_nonassociative_order_and_empty_policy; MBOUND::first_term_fallback_is_used_only_for_empty_reductions. |

| ID | Escenario | Contenido comprobado y evidencia específica | Aplicabilidad |
|---|---|---|---|
| bml-symbolic-matrices/S01 | Disposición y costos de almacenamiento | DOC§Localidad y asignaciones describe headers contiguos, payloads separados, stride de derecha, sin bloqueo/transposición/reuso global oculto; E-PHASE mide preparación/producto/integral, asignaciones y pico heap, además de tamaño lógico. No afirma localidad PMU medida. | S |
| bml-symbolic-matrices/S02 | Validación y separación de contratos numéricos | `BmlMatrix::new` usa checked_mul y count antes de validar payload; matriz recibe un Vec ya creado por el llamador, no reserva internamente antes de forma. Campos privados; MATRIX formas/overflow; MBOUND presupuestos; fachada mantiene APIs IEEE/GGUF separadas. | S |
| bml-symbolic-matrices/S03 | Matriz simbólica distinta de un tensor IEEE | MATRIX::shapes_access_and_transactional_replacement construye2×3, seis códigos y acceso fila1col0→4; rechaza Vec de longitud1 y usize::MAX×2. `new` conserva Vec<Bml>, sin evaluación implícita. | S |
| bml-symbolic-matrices/S04 | Valores repetidos precompilados | `prepare` valida y certifica cada árbol una vez por matriz/operación; bucles del kernel consumen CertifiedBml, no resintetizan entrada numérica por celda. E-PHASE separa matrix_preparation/operation_prepared_codes/complete; DOC declara que preparación se repite en llamadas independientes. | S |
| bml-symbolic-matrices/S05 | Errores transaccionales de operadores matriciales | MATRIX::failures_are_localized_and_do_not_mutate_inputs exige Scalar{row0, column1, DomainNotJustified} y Budget{row0, column0, inner0}; signed_matmul_and_output_budget_preserve_inputs cubre salida; buffers locales solo se devuelven tras éxito. | S |
| bml-symbolic-matrices/S06 | Operar valores y no sus códigos | MATRIX::elementwise_operations_use_values_not_codes usa códigos4+4(valores2+2), certifica resultado4 y exige código!=8; hadamard y forma errónea conservan operandos. | S |
| bml-symbolic-matrices/S07 | Producto rectangular y crecimiento | MATRIX::rectangular_product_matches_independent_integer_oracle compara2×3·3×4→2×4 con bucles i64 independientes y estructura izquierda; failures_are_localized... y MBOUND prueban presupuestos e índice localizado. | S |
| bml-symbolic-matrices/S08 | Acceso inválido o sustitución excesiva | MATRIX::shapes_access_and_transactional_replacement y MBOUND::replacement_checks_each_budget_and_updates_the_requested_cell prueban índice inválido, tope total/por escalar y matriz idéntica al fallar. Index documenta panic; no IndexMut expuesto. | S |
| bml-symbolic-matrices/S09 | Cambio de orden de reducción | MATRIX::rectangular_product_matches_independent_integer_oracle exige árbol exacto((p0+p1)+p2); generalized_product_preserves_nonassociative_order_and_empty_policy contrasta asociación izquierda/derecha. No ruta alternativa cambia orden de acumulación. | S |
| bml-symbolic-matrices/S10 | Dimensión interna vacía | MATRIX::empty_products_use_symbolic_zero_and_bound_zero_size_shapes exige2×0·0×3→2×3 ytodos códigos88, con error de presupuesto para el primer cero; descarta usar entero 0 como cero matemático. | S |
| bml-symbolic-matrices/S11 | Acumulación no asociativa | MATRIX::generalized_product_preserves_nonassociative_order_and_empty_policy usa combine que conserva izquierdo y accumulate=op, exigeB(B(p0, p1), p2) y desigualdad frente aB(p0, B(p1, p2)); también comprueba Value(seed). | S |
| bml-symbolic-matrices/S12 | Producto generalizado vacío sin identidad | La misma prueba exige EmptyReduction para First{empty: None} con dimensión0; Value(seed) produce seed sin llamar callbacks; MBOUND::first_term_fallback_is_used_only_for_empty_reductions cubre First{empty: Some}. | S |

### bml-simd-backends

Norma: [bml-simd-backends/spec.md](../../openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-simd-backends/spec.md).

| ID | Requisito | Obligaciones auditadas | Fuente y contraste |
|---|---|---|---|
| bml-simd-backends/R01 | Referencia escalar y selección segura de backend | Referencia escalar por kernel y selección segura según arquitectura/capacidades; backend forzado no disponible da error; documentar unsafe/colas, sin SVE/SVE2. | `kernels::{Backend::available,resolve,dispatch}`; KCODE y E-NOAVX; DOC§Kernel SIMD. Auto se resuelve a Scalar; AVX2 x86/x86_64 y NEON AArch64 se detectan antes de target_feature. |
| bml-simd-backends/R02 | Trabajo vectorizado declarado | Declarar el trabajo realmente vectorizado y sus instrucciones/layout/colas/errores; reglas de exp2/log2 solo si hay BML numérico; no atribuir aceleración al ancho. | `bml-kernels` y fachada code_simd declaran comparación estructural; AVX2 cmpeq de4 palabras, NEON de2; sin evaluación/transforms de matrices; KCODE y E-NATIVE. |
| bml-simd-backends/R03 | Equivalencia con la referencia | Preservar estructura, orden, presupuestos, salida y primer error para el mismo trabajo; precisión/dominio/reducciones solo si ruta numérica; matriz idéntica si se ofreciera construcción por backend. | `compare_codes` preflight→resolve→alloc→orden de código; KCODE verifica todas posiciones/colas y errores; no constructores matriciales SIMD ofrecidos. Escenario matricial condicionado no aplicable al kernel escogido, sin tocar su norma. |
| bml-simd-backends/R04 | Evidencia por arquitectura | Validar/medir cada arquitectura nativa antes de afirmar ventajas; compilación cruzada no es ejecución; registrar campañas ausentes e históricas sin resultados inventados. | E-NATIVE: VMs GitHub ISA nativa ARM64/x86_64, 8/8 tests por host y Scalar/Neon o Scalar/Avx2 medidos; E-CODE anterior y emulación son históricos. No prueba PMU/L1/inferencia. |

| ID | Escenario | Contenido comprobado y evidencia específica | Aplicabilidad |
|---|---|---|---|
| bml-simd-backends/S01 | Disponibilidad y contrato del backend | `Backend::available/resolve` usa cfg y detección runtime antes del dispatch; `target_feature` y comentariosSAFETY cubren cada carga sin alineación. KCODE::preflight_preserves_first_error_and_never_modifies_inputs exige BackendUnavailable al forzar uno ausente; E-NATIVE acredita backend ejecutado real. | S |
| bml-simd-backends/S02 | CPU sin AVX2 | E-NOAVX ejecuta los cuatro contratos KCODE en QEMU x86_64 con CPU Nehalem sin AVX2: backend ejercitado [Scalar], Auto=Scalar y backend Avx2 forzado rechazado. Los tres hashes de fuentes coinciden con este árbol; binario, emulador y log se verificaron. Es evidencia funcional emulada de esa condición x86; no ejecución nativa ni medición de rendimiento. | S |
| bml-simd-backends/S03 | Manipular códigos no acredita evaluación | KCODE::structural_order_does_not_evaluate_signs_or_domains incluye cero, negativo y árbol fuera de dominio; comparación se hace por bit_len/palabras yno evalúa exp2/log2 ni construye matrices. DOC y E-NATIVE declaran alcance estructural. | S |
| bml-simd-backends/S04 | Cola y primera falla | KCODE recorre tamaños de 0 a 511 operadores y diferencias desplazadas por 260 profundidades; compara orden contra strings binarios independientes. Preflight comprueba Length, PairLimit, primer BitLimit antes de backend/alloc y no cambia entradas. Árboles malformados se rechazan en constructor Bml; árboles de dominio inválido son datos estructurales válidos. | S |
| bml-simd-backends/S05 | Equivalencia estructural | Condición de este escenario: ofrecer construcción de la misma matriz mediante varios backends. No existe tal API; el requisito general permite elegir comparación de códigos y el diseñoD7 exige declarar ese trabajo. No se añadió constructor artificial ni se modificó el texto normativo. Las matrices usan una referencia simbólica única; compare_codes tiene equivalencia exacta propia acreditada en la fila anterior. | C |
| bml-simd-backends/S06 | Compilación ARM sin ejecución | La compilación cruzada y emulación históricas se identifican como tales. E-NATIVE posterior contiene ELF183/62 yRunnerArchARM64/X64, tests 8/8 y muestras Scalar/Neon oScalar/Avx2. DOC corregido diferencia ambos momentos; no queda una afirmación actual de ARM solo compilado. | S |

### bml-performance

Norma: [bml-performance/spec.md](../../openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-performance/spec.md).

| ID | Requisito | Obligaciones auditadas | Fuente y contraste |
|---|---|---|---|
| bml-performance/R01 | Métricas de trabajo no ambiguas | Definir unidad e intervalo para construcciones, B, expresiones, elementos, matmul; FLOP solo con convención explícita, no tresFLOP automáticos por B ni suma de unidades distintas. | DOC§Unidades de las campañas; `code_compare_bench` cuenta comparaciones y `symbolic_phases` unidades por fase; E-PHASE.units/campaign.phases y E-NATIVE. |
| bml-performance/R02 | Medición por fases y crecimiento | Separar síntesis/preparación, optimización, serialización si aplica, kernel e integral; caché fría/caliente, crecimiento, bits/operadores/profundidad y memoria/asignaciones con método declarado. | E-PHASE12 fases, search_cold/search_warm, stream_roundtrip, simplified_evaluation y matrix_preparation/operation/complete; allocator System instrumentado, pico heap adicional y bytes solicitados, no RSS; E-NATIVE preparación/códigos/comparación. |
| bml-performance/R03 | Comparación reproducible y puerta de corrección | Antes de tiempos validar referencia con mismos datos/contrato/presupuesto; conservar CPU, arquitectura, sistema, revisión, flags, backend, hilos, semilla, formas/tamaños/comandos, warmup, repeticiones, dispersión, errores, memoria. | E-PHASE correctness_gate, input_terms, matrix, phases y metadatos; E-NATIVE manifests+samples siete repeticiones, CPU afinidad, ELF y todos hashes; Auto permanece Scalar, no extrapolación inferencia. |

| ID | Escenario | Contenido comprobado y evidencia específica | Aplicabilidad |
|---|---|---|---|
| bml-performance/S01 | Convención de FLOP y unidades | DOC§Unidades separa aplicacionesB, expresiones, elementos, matmul y FLOP para campañas IEEE con convención declarada; E-PHASE y E-NATIVE reportan unidades concretas, no equivalencia automática3 FLOP por exp2/log2/resta. | S |
| bml-performance/S02 | Aplicaciones de B frente a FLOP | Si se evaluaran un millón de B, se contarían aplicaciones ytiempo. La campaña actual basis_scalar_kernel declara B_applications/work_per_iteration; no usa exactamenteun millón ni se atribuye ese número. La convención está implementada sin inventar una ejecución de ese tamaño. | C |
| bml-performance/S03 | Valor precalculado reutilizado | E-PHASE contiene matrix_preparation, matrix_operation_prepared_codes ymatrix_complete; exact_synthesis/proof_replay/search_cold/search_warm separan costes de números ycache; integral prepara entradas en cada iteración. E-NATIVE separa preparación de lotes ycomparación validada. | S |
| bml-performance/S04 | Metadatos reproducibles del informe | E-PHASE contiene arquitectura/CPU/kernel/revisión/rustc/build, semilla 1, hilos 1, repeticiones7, inputs ytreecosts, matriz2×3·3×4, correctness_gate yallocator. E-NATIVE añadeCPU/flags/afinidad, RUNNER_ARCH, ELF, source/file hashes y7 muestras crudas por backend/carga; el ejemplo define error ydispersion se deriva deesas muestras. | S |
| bml-performance/S05 | Comparar SIMD con escalar | E-NATIVE mide Scalar yNeon en ARM64 yScalar yAvx2 en x86 con datos idénticos/semilla fija, 100 iteraciones y7 repeticiones, tests previos yCPU fijada. Cargas cortas no acreditan ventaja universal; Auto continúa Scalar. E-CODE previo se conserva histórico con hashes diferentes, no promoción actual. | S |
| bml-performance/S06 | Hardware no disponible | La campaña simbólica de fases es solo x86 y así lo declara; no acredita fases matriciales ARM. E-CODE antiguo dejóNEON no ejecutado; E-NATIVE posterior sílo ejecutó para comparación de códigos. Las cifras no se extrapolan a inferencia niPMU/L1. | S |

## Aplicabilidad y hallazgos

1. **Construcción matricial SIMD:** el requisito de trabajo declarado permite manipulación de códigos, construcción o evaluación; el diseño D7 manda elegir explícitamente una. La API ofrecida es comparación estructural. La condición «WHEN se construye la misma matriz BML en escalar, AVX2 y NEON» no obliga a introducir constructores ausentes. Se conserva literalmente la norma y se registra el escenario como condicionado (C); los códigos existentes se comparan exactamente en los backends ofrecidos. No se usa esa fila para afirmar que existe construcción/evaluación matricial SIMD.
2. **Árboles inválidos y errores SIMD:** las entradas de `compare_codes` ya son `Bml` sintácticamente válidos. Los errores estructurales se verifican al construirlos; un árbol inválido en dominio real es un código válido y no debe ser evaluado por la comparación. Los errores propios del lote y presupuestos se verifican antes de ejecutar SIMD y antes de reservar salida.
3. **ARM desactualizado:** el párrafo público que afirmaba que NEON solo compiló se corrige para identificar la campaña inicial y la posterior ejecución nativa E-NATIVE. No se cambian claims numéricos ni Auto.
4. **Búsqueda desactualizada:** el párrafo del constructor exacto decía que faltaba la comparación de estrategias pese a existir `ExactSearcher`; se corrige para distinguir síntesis determinista y búsqueda acotada registrada. No se afirma minimalidad global.
5. **Campañas históricas:** E-PHASE conserva el código numérico auditado pero manifiestos anteriores; E-CODE tiene fuentes previas. La identidad y ámbito quedan explícitos, sin reetiquetar sus mediciones como actuales. No son evidencia de rendimiento de inferencia ni de residencia de caché.
6. **CPU x86 sin AVX2 (SIMD/S02):** E-NOAVX cierra la brecha funcional con ejecución emulada de los cuatro contratos sobre Nehalem. Se verificaron sus tres hashes de fuentes contra este árbol y los hashes del emulador, binario y log local. No se convierte esa comprobación en evidencia de hardware nativo o rendimiento.
7. **Pendiente de integración:** confirmar el gate del snapshot integrado y registrar su revisión/log/hash; validar OpenSpec estrictamente; actualizar 9.3 y archivar solo tras esa comprobación. Esta tarea no ejecuta builds, no marca 9.3 final y no archiva. Las diferencias de documentación normativas permanecen nulas. No se ha identificado una brecha de implementación adicional que justifique añadir APIs o debilitar una spec; la ejecución funcional específica SIMD/S02 está ahora documentada con su alcance emulado.

## Identidad de las fuentes auditadas

Los hashes siguientes fijan specs, implementación numérica, pruebas contrastadas y ejemplos usados. El manifiesto E-NATIVE incluye además las otras fuentes de dominio y el runner. No se incluyen docs editadas por esta tarea en esta lista para evitar confundir el estado previo y su corrección.

| Archivo | SHA-256 |
|---|---|
| `crates/bml/src/lib.rs` | `09cb13cc5583ec66f870f896119d6716f6dab6324e3cf19f580191f6c7a75fb3` |
| `crates/domain/Cargo.toml` | `ab5553ea1a83d4cb7fa5705c43d91c5c4e57495bb6a969acd54b5667bdfecd34` |
| `crates/domain/src/number/certified.rs` | `b3122ee556cd1e7e8316ce6542e994b721e750b9a9e1f1c64e7517aedb42d77c` |
| `crates/domain/src/number/checked.rs` | `11972ce60a80d679a9ab3b1c5f3cc50f5147180727908fea286b46f9732ef561` |
| `crates/domain/src/number/convert.rs` | `d7abdf97b181f0a78b2d641e98c80faac933797690131344f2f6a1a793a6f641` |
| `crates/domain/src/number/encoding.rs` | `e914642dfab9dad767752ff5cfb2bc4e6faa55b37e0d83517918ead86a2139fb` |
| `crates/domain/src/number/evaluate.rs` | `e4602f91da384c89ae2a9eda0223827ba2c24cbaf21124ae366eca73ea49e94d` |
| `crates/domain/src/number/exact.rs` | `da29637bbc49161ca1e684f87bd0e1f971af89e9ed9ff8eda3f2d7af024a73bf` |
| `crates/domain/src/number/matrix.rs` | `f742872b25fb7929e289741248d572e108f7aadbea8497c193bccbba02499d77` |
| `crates/domain/src/number/mod.rs` | `6447938efe5ebf55093c50c6d727210677d202c6a7a58f5dbeb6f6bb6100880b` |
| `crates/domain/src/number/operations.rs` | `0d11bb78e9d3cca51008aa86f064a4a806f6f5422f407b9a0407dfb1d7c57131` |
| `crates/domain/src/number/recognize.rs` | `645905e3daf4bdb551a397019e8de4d8b9198086a43fddaef7d75284690e5e28` |
| `crates/domain/src/number/search.rs` | `cdb193250b2647605f319bc663940ba71d7219dab244214dd51f3b3b7d9cd88f` |
| `crates/domain/src/number/stream.rs` | `230aee44dd7f6fc448d6dda2f1f0771f419f60b400e369b391a62d19e1573057` |
| `crates/domain/src/symbolic.rs` | `da18949682b8fc4abbef3a58b52f7139fdfc562923720eee5cb287f09b41b947` |
| `crates/domain/tests/bml_bounded_properties.rs` | `af160f3e8dd3f93fb7d016ea344ae085172554f37a5141f18b4ddea1ccc4b65e` |
| `crates/domain/tests/bml_certified.rs` | `4b7c0aa26ecb52543ca3f058b0e0201bb84fc00cf4ec96d545f3b52ec93bd27b` |
| `crates/domain/tests/bml_checked.rs` | `173717280975c6df0a7cb5aa407c3791acae71c2042507a3d7a9adf031c9b3fd` |
| `crates/domain/tests/bml_conversion.rs` | `fdcbe1ea4b7453e81da4fa397fa872cf9f1e885183c76d5f679ce3402cc52b06` |
| `crates/domain/tests/bml_encoding.rs` | `c6e1d4d3ea1392208d9b42f1f5d00b3a6ce3be020605b3c8bbcb44999aa47360` |
| `crates/domain/tests/bml_exact_input.rs` | `e1652dd892ba984746d5dbd7d1c63e0a333e03b0d7d0dc54d5cde1529469559d` |
| `crates/domain/tests/bml_matrix.rs` | `0e629e947344071f7903dff3aba1d537cc32be4af5b71b27319e60fa59205af1` |
| `crates/domain/tests/bml_matrix_boundaries.rs` | `f2a0028feb9cdaea03c3d2b531b6e1c4c8c83942306c99b342d00776ce435624` |
| `crates/domain/tests/bml_operations.rs` | `f25de37d4fceb78c32db63a8abf425d848996a7c11d1c6e4cd504353bb50e23d` |
| `crates/domain/tests/bml_recognition.rs` | `c438a121cc956f3af82fdee775f1561fd569c3e845d49b05f57903b1843d853e` |
| `crates/domain/tests/bml_search.rs` | `caa114073446f5ee6fe6c382f4db5929a8bc4b4e893f098981821c5d119a00d2` |
| `crates/domain/tests/bml_simplified_eval.rs` | `2d676b44c25159befd179a204963bdf3861e1e9dda22bbcc717c896ba369b563` |
| `crates/domain/tests/bml_stream.rs` | `455019d6dbac9e2cd03f58a081379ef4252d3038febc5258f27cb158d63fada3` |
| `crates/domain/tests/symbolic_contracts.rs` | `33661bdd0f2c4d02de00100e9816c092b351c6b78c5a1328b686f1df0795801f` |
| `crates/kernels/Cargo.toml` | `ef943b48a6ffbf7755ae2d775e0f36f60e27ceeb08f43f1e1b0aeee11183ea65` |
| `crates/kernels/examples/code_compare_bench.rs` | `a82ba8d2eda2a5d24ae2ab63aa24e2f0e6c9821563859929ed5b2c1802e02f37` |
| `crates/kernels/examples/symbolic_phases.rs` | `b9bb1f03ffcf623168462f86ad969ce8aeab784593c57a439eecb79cbdc53176` |
| `crates/kernels/src/lib.rs` | `82c4511a44cf04c338e823dde904133b1f200f278e12efea66a66617319e23a4` |
| `crates/kernels/tests/code_comparison.rs` | `7f3e33cedc7057ad6305c16686cf7cfd14be4930d5f6790a36d17f5ef677e4ac` |
| `openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-code-storage/spec.md` | `552eaec121ac8155084ec7b31ac0fedaf881539c9bf8936c2de872bdd412f304` |
| `openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-exact-synthesis/spec.md` | `19398242360019e3192ca43ad30a00b647d46610376734c61f6781e4f78aeee2` |
| `openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-performance/spec.md` | `5d17716a8571e1436f94892eed968b8a16f989b5459f332fa42291249ddea2ac` |
| `openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-rust-operators/spec.md` | `304d5c0195f124a9d38ef81d627296feeac6906df079a9967134d9595197d893` |
| `openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-simd-backends/spec.md` | `fb897443036162922754d41692010199634b3eab272bdd69f6009348c11ee30b` |
| `openspec/changes/archive/2026-11-05-bml-exact-numbers-and-matrices/specs/bml-symbolic-matrices/spec.md` | `6ad8a9329b81a369a64f5c3ab2b688f20caf6d74b96e22bc021e3e65228107ee` |
