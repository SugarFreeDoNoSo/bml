## 1. Contract and references
- [x] 1.1 Confirm full 1.0 scope and first target with the user.
- [x] 1.2 Record supported Llama profile and independent reference revision.
- [x] 1.3 Record model identity, fixed prompts and parity tolerances before validation.

## 2. Import and tokenizer
- [x] 2.1 Validate model dimensions, tensor roles, dtypes and finite resources.
- [x] 2.2 Implement SentencePiece BPE, scores/types, byte fallback and BOS/EOS.
- [x] 2.3 Verify tokenizer against pinned llama.cpp, including Unicode/whitespace.

## 3. Model execution
- [x] 3.1 Implement full causal Llama blocks and final logits with independent tests.
- [x] 3.2 Verify KV/prefill/decode equivalence, context bounds and failed-request rollback.
- [x] 3.3 Preserve quantized resources and expose execution/memory plans.
- [x] 3.4 Match full real-model logits and greedy sequences against the reference.

## 4. Resources and AOT
- [x] 4.1 Implement normalized resource schema, checked ranges and SHA-256 integrity.
- [x] 4.2 Generate and link model-specific native executable with embedded resources.
- [x] 4.3 Test malformed/corrupt/truncated resources and unsupported configurations.
- [x] 4.4 Match native standalone output against the validated model/reference.
- [x] 4.5 Run in an isolated environment without model, BML, toolchain or network.

## 5. Release evidence
- [x] 5.1 Verify numeric facade, feature matrix and independent GGUF boundaries.
- [x] 5.2 Record measured prefill/decode latency, RAM and executable size.
- [x] 5.3 Update entrypoint documentation, version and supported-profile limitations.
- [x] 5.4 Review public errors/APIs, resource budgets and legacy compatibility.

## 6. Cierre

Verificación local actualizada el 2026-11-03 en
`quality/evidence/spec-local-validation-2026-11-03.json`: gate canónico completo,
features GGUF independientes y fachada/runtime/kernels all-features aprobados;
cobertura de líneas 86,26 % y funciones 80,15 %. La tarea 6.2 permanece abierta
porque falta la campaña de mutaciones sobre este snapshot.

La campaña histórica de 2.507 candidatos quedó incompleta en
`target/spec-mutations-2026-11-03/mutants.out/outcomes.json`. La campaña
posterior también fue interrumpida (`quality/evidence/global-mutation-status-2026-11-03.json`). Una campaña de 5.172 candidatos sobre `ce61deb` se interrumpió porque compartió binarios con otra compilación y sus resultados no acreditan el gate (`quality/evidence/current-spec-mutation-run.json`). El primer mutante superviviente de `--l1-threshold` motivó una regresión: los modos distribuido y simbólico rechazan el límite insuficiente y ambos replays se detectan (`quality/evidence/cli-fragment-limit-mutation-replay.json`); el snapshot de la campaña no incluye esta regresión. Regresiones y replays
separados registrados en `quality/evidence/spec-mutation-progress-2026-11-03.json`:
37 mutantes matriciales detectados y uno no compilable; 6/6 detectados para
selección de artefactos y límite de fragmentación de la CLI. Las suites actuales
pasan 610 pruebas Rust; estos replays no sustituyen el gate de la campaña completa.

Seguimiento del compilador simbólico:
`quality/evidence/symbolic-boundary-mutation-replays.json` acredita 12 mutantes
de dimensiones/fragmentación y 10 del loader detectados tras añadir los casos
de frontera inclusiva y rechazo de opcodes nativos sin depender del runtime.
Gate canónico actualizado: 613 pruebas Rust, auditoría cero discrepancias,
traza 71/71 y Clippy sin diagnósticos nuevos. Dos mutantes históricos adicionales de detección de contenedores se detectaron con una regresión que cubre prefijos truncados y errores de lectura (`quality/evidence/container-detection-mutation-replays.json`). La campaña completa sigue pendiente.

Seguimiento local posterior (`6a3e6cc`): los 47 supervivientes del informe
incompleto de 1.507 resultados tienen disposición: 38 detectados por contratos
y 9 equivalentes/redundantes bajo los contratos actuales. Los dos timeouts
también tienen detección local. Evidencia:
`quality/evidence/survivor-dispositions-2026-11-03.md`. El gate de esa entrega
aislada pasa 656 pruebas Rust y 63 Python, traza 28/28, cero discrepancias y
cero diagnósticos nuevos. No acredita el resto del árbol pendiente ni aprueba
exclusiones de mutación. La auditoría encontró 962 registros `Unviable` con el
mismo E0505 en una prueba original; se requiere una campaña nueva sobre un
snapshot congelado con baseline aprobado. 6.2 continúa abierta.

El snapshot completo congelado `e997a84452ee5599858c081c41b6bb21b24c44768e0a30c5459fe32484d498a4`
pasa ahora el gate canónico: 726 pruebas Rust, 82 Python, traza 71/71, cero
discrepancias y cero diagnósticos nuevos. Features mínimas/completas aprobadas;
cobertura de líneas 87,16 % y funciones 79,98 %, sobre los pisos vigentes.
La campaña nueva de 3.025 candidatos se interrumpió sobre ese mismo snapshot,
sin exclusiones por equivalencia: 423 resultados terminales (355 detectados,
31 supervivientes, 11 timeouts y 26 no viables), y 2.602 pendientes. No queda
ningún proceso de la campaña activo y el gate global sigue sin aprobar. Estado
verificado en `quality/evidence/global-mutation-status-2026-11-03.json`.
Evidencia: `quality/evidence/global-quality-restart-2026-11-03.json`.
- [x] 6.1 `openspec validate bml-1-0-release` passes.
- [ ] 6.2 Canonical workspace gate, feature checks and applicable coverage/mutation pass. La campaña del snapshot anterior terminó con 1163 candidatos: 769 caught, 246 missed, 25 timeouts y 123 unviable (exit 3). No acredita el código actual ni satisface el gate de mutaciones; evidencia y casos pendientes en `quality/evidence/spec-mutation-campaign-2026-11-02.json`. Se añadieron regresiones de límites del decoder del worker, con gate canónico aprobado (`quality/evidence/worker-decoder-boundaries.json`); replay aislado de los cuatro mutantes históricos del decoder: 4/4 detectados por las pruebas actuales (`quality/evidence/worker-decoder-mutation-replays.json`). La campaña completa sigue pendiente.
- [ ] 6.3 Commit the reviewable release work with reproducible evidence; publish only when authorized. La entrega local revisada se publicó con autorización en `ca8d030` y `1ef5f5d`; el remoto está verificado. Evidencia: `quality/evidence/publication-spec-closure-2026-11-03.md`. Esta publicación parcial no constituye la entrega final BML 1.0: 6.2 continúa sin aprobar y 6.3 permanece abierta hasta el cierre de release.
