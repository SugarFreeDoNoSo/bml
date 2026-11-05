# Auditoría de contratos de los placeholders G1

**Resultado: parcial; no archivar `fix-placeholders` ni cerrar 8.5.9.**

Esta lectura compara el código de la propuesta inicial `3817562` con el snapshot
`9aa114a9d74e72e8f09d45e367df5caaa936b375`. No ejecuta pruebas, campañas ni
modelos nuevos: los resultados numéricos citados pertenecen a los registros
identificados abajo. El gate y las pruebas de la integración final deben
comprobarse por separado. No se modifican specs ni marcas de tareas.

## Cómo se reconstruye el número 14

En `3817562`, `proposal.md` y `design.md` declaran «14 placeholders», pero no
incluyen un inventario numerado de catorce sitios. Las tareas originales agrupan
matmul, RoPE, SwiGLU, lecturas indexadas, tokenización, sampling y generación.
No es posible atribuir una numeración histórica que los documentos no contienen.

H01–H14 son **unidades de auditoría reconstruidas**, agrupadas por API o consumidor
que efectivamente existía en esa revisión. Cada fila identifica su ancla
histórica y su obligación, sin presentar esta agrupación como una lista original
recuperada. La coincidencia del total no acredita cierre. Las superficies
adicionales descubiertas después se auditan aparte y no se excluyen para
mantener artificialmente el número catorce.

Fuentes históricas reproducibles, sin depender de búsquedas de comentarios:

```sh
git show 3817562:openspec/changes/fix-placeholders/proposal.md
git show 3817562:openspec/changes/fix-placeholders/tasks.md
git show 3817562:crates/compiler/src/gguf_compiler.rs
git show 3817562:crates/compiler/src/op_fragments.rs
git show 3817562:crates/compiler/src/rpn.rs
git show 3817562:crates/cli/src/main.rs
git show 3817562:crates/api/src/main.rs
git show 3817562:crates/runtime/src/runtime.rs
```

## Inventario y consumidores actuales

«Completo» en esta tabla significa que la sustitución está implementada y cuenta
con los contratos indicados **en el alcance descrito**. No significa dominio
IEEE universal, paridad con llama.cpp, rendimiento demostrado ni ejecución nueva
en esta auditoría. Los kernels simbólicos siguen siendo fallibles y acotados.

| ID / ancla histórica en `3817562` | Sustitución y consumidor actual | Evidencia numérica o de comportamiento | Estado y límites |
| --- | --- | --- | --- |
| H01. `gguf_compiler::build_transformer_dag`: hidden escalar, primer peso de cada tensor, normalizaciones primitivas, residual literal y cabeza basada en el primer embedding. | `compile_gguf` y `compile_gguf_fast` sustituyen el antiguo builder privado `build_transformer_dag`, ya ausente, y publican un `CompiledArtifact` simbólico mediante `symbolic_artifact`; `symbolic_bounded::dag` usa `BMLTransformer::operation_bounded`. `SymbolicCausalModel` consume las filas externas y los programas persistidos para la capa causal, normalización final y cabeza. | C72 `bml-compiler::compiler_entrypoint_contracts::both_gguf_entrypoints_publish_complete_numerical_causal_plans` ejecuta las entradas reales. C64/C65/C81/C84 contrastan modelo causal, AOT y v4. [Diagnóstico TinyLlama BOS](tinyllama-bos-logit-diagnostic.json): 32.000 logits finitos por ruta, argmax 29958, diferencia máxima 4,879e-6 y media 9,515e-7 respecto al nativo F64/F32-KV. | **Parcial** para el alcance global de compilación simbólica real: hay ejecución completa BOS, pero no tolerancia previa de paridad ni validación simbólica completa TinyLlama en posiciones adicionales. La cabeza independiente y el epsilon se prueban en `inference_contracts`; no se restauran los antiguos resultados escalares. |
| H02. `op_fragments::compile_matmul_fragment`: sobrescribe el último producto sin suma de productos. | API histórica y `compile_matmul_fragment_checked`; la composición de capa usa proyecciones completas. `Runtime`/`HotLoop` consumen el `OperationFragment` con su contexto de constantes. | C85 `bml-runtime::operation_matmul_contracts`: matrices firmadas/rectangulares hasta 4×64, bases no cero, dimensiones vacías, repetición y pila preservada. Oráculo de productos y sumas densos independiente, tolerancia absoluta 2e-7. | **Completo** dentro del perfil declarado. La API configurable verifica operandos e intermedios reales; la histórica conserva precondiciones documentadas. Véase [reparación de operaciones](operation-fragment-repair-2026-11-03.md). |
| H03. `op_fragments::compile_rmsnorm_fragment`: aplica una primitiva por elemento sin media, epsilon ni raíz. | `compile_rmsnorm_fragment[_checked]` en `operation_rms`; `compile_layer_fragments_checked` lo incorpora a atención y MLP. El runtime enlaza el pool propio de cada operación. | C92 `bml-runtime::operation_rms_contracts::complete_rms_fragments_match_independent_normalization_and_preserve_storage`: `[3,4]`, pesos `[2,-0.5]`, epsilon 3,5 → `[1.5,-0.5]`; oráculo independiente de media/raíz, tolerancia 2e-7. También anchos 1/32/288/2048 y epsilon distinto en workers concurrentes. | **Completo** para RMS acotado y su wrapper documentado. No confundir con el helper de factor recíproco descrito en E01. [Evidencia](compact-bounded-2026-11-03.md). |
| H04. RoPE de `build_transformer_dag`: calcula solo un par y descarta `q_new`. | RoPE nativo recorre todas las cabezas/parejas; el compilador simbólico genera el kernel consumido por `SymbolicCausalModel::rotate`. APIs públicas de par y vector conservan coordenadas, frecuencias y outputs explícitos. | `bml-compiler::causal_contracts::rotary_pairs_use_distinct_frequencies_and_absolute_cached_positions`; C93 `bml-runtime::operation_norm_rope_contracts::vector_rope_position_zero_covers_all_1024_pairs_with_constant_bytecode` y prueba de reinicio por cabeza. [Probe real](tinyllama-real-rope-oracle.json): 50.688 parejas Q/K nativas en dos posiciones, cota 32, error máximo 3,169e-12. | **Completo** para las APIs de rotación examinadas. El probe con entradas nativas no demuestra ejecución simbólica completa de esas posiciones; esa obligación permanece en H01. |
| H05. `op_fragments::compile_attention_fragment`: `B(B(q,k),v)` por coordenada, sin dot/escala/softmax. | `operation_attention`: dot firmado, escala, máximo auxiliar, softmax y suma ponderada. Variantes configurable y con scratch; la capa usa el plan completo. `Runtime` y `ValidatedGraph` ejecutan el bytecode. | C97 `bml-runtime::operation_attention_contracts::complete_attention_matches_nonuniform_signed_oracle_with_and_without_scratch`: contextos no uniformes, signos y direcciones que cruzan slots, oráculo denso de probabilidades/valores, tolerancia 2e-6. Se verifican layout, stack y todos los BML. | **Completo** en el contrato de una cabeza. La firma histórica representa un token KV; no acredita GQA ni el transporte distribuido. [Evidencia](operation-layer-2026-11-03.md). |
| H06. `op_fragments::compile_mlp_fragment`: sustituye gate/up por primitivas y elimina down. | `operation_mlp` implementa las matrices gate/up/down completas y SwiGLU estándar. Variantes configurable/rectangular y con scratch; consumidores: composición de capa, runtime y grafo comprobado. | C98 `bml-runtime::operation_mlp_contracts`: oráculo denso con `g/(1+exp(-g))*u`, proyección down independiente, tolerancia 2e-7. `every_matrix_element_influences_the_complete_mlp_and_legacy_is_square` perturba cada matriz; se comparan bits de variantes con/sin scratch. | **Completo** en las dimensiones y cotas declaradas. El wrapper conserva FFN cuadrado. El factor histórico 1,7 fue corregido expresamente a SwiGLU estándar por la spec vigente. |
| H07. `op_fragments::compile_layer_fragments`: encadena operaciones incompletas; carece de RoPE, KV por posición y ambos residuales originales completos. | `operation_layer::compile_layer_fragments_checked` devuelve el plan con layout explícito; `compile_layer_fragments` adapta slots históricos. El runtime secuencial y el scheduler ejecutan las mismas operaciones y pools. | C99 `bml-runtime::operation_layer_contracts::complete_causal_layer_matches_dense_oracle_for_two_positions_and_all_bml_nodes`: ecuaciones independientes, FFN rectangular, nueve tensores con influencia observable, conservación del prefijo KV y memoria ajena. El wrapper se contrasta con bases no cero. | **Completo** en una capa de una cabeza y perfil acotado. El wrapper usa posición cero y escribe `input_slot+18`; requiere el espacio documentado. La construcción 2048/5632 y 17 fragmentos no es ejecución de un modelo grande. |
| H08. `RpnProgram::evaluate_with_ctx`: `VarIndexed` inserta cero al no existir buffer, tanto directo como en bucle. | Las rutas RPN y `Fragment` retornan NaN explícito sin almacenamiento; `HotLoop::execute_fragment_full` y `Runtime` usan el `ResultBuffer` real. | C44 `bml-compiler::contracts::indexed_reads_without_buffer_are_nan_at_every_supported_loop_depth`: profundidades 0/1/2, NaN sin buffer frente al valor 17 almacenado. | **Completo** para la distinción sin/con buffer. El cero del diseño D6 inicial fue corregido expresamente; `RpnOp::Zero` sigue siendo una constante legítima. |
| H09. Prompt sin tokenizer: CLI y API usan `prompt.len()` como único input; tareas 5.1–5.4 todavía sin implementación. | `Vocabulary::from_gguf`/`encode` comparten el tokenizer BPE validado. CLI `run_text_model` y servidor `generate_local` tokenizan antes del forward; BPE usa scores, tipos, BOS y fallback por bytes. | C50 `bml-compiler::tokenizer_contracts`: oráculo de IDs, merges afectados por scores, mayúsculas/espacios, Unicode y metadatos ausentes. [Tokenización real](legacy-bpe-output.json): TinyLlama «Hello» → `[1,15043]` como referencia fijada. | **Completo** para el perfil SentencePiece BPE admitido. La propuesta inicial llamaba BPE a split/lookup; la spec vigente exige el BPE real y rechazos explícitos. No se afirma soporte genérico de cualquier tokenizer. |
| H10. Salida sin detokenización: `generate_tokens` y CLI imprimen prompt/formato fijo; tarea 5.5 ausente. | `Vocabulary::token_piece_bytes`, salida CLI por bytes y ensamblado UTF-8 incremental del servidor. Consumidores locales y distribuidos conservan EOS y piezas parciales. | C51 `bml-cli::token_output::cli_preserves_multibyte_token_pieces_and_stops_at_eos`; C58/C67 HTTP/SSE generan «é» desde dos tokens de bytes y terminan correctamente. Pruebas del helper servidor comprueban secuencias inválidas/incompletas. | **Completo** para el contrato por bytes y política textual declarada. La CLI conserva bytes; el servidor representa secuencias finales inválidas mediante sustitución explícita, no inventa IDs ni palabras. |
| H11. Sampling inexistente: opciones `temp`/número de tokens no controlan una selección de logits. | `sampler::sample` compartido por CLI, JSON/SSE y ruta distribuida; cero usa greedy y cualquier temperatura positiva finita conserva distribución softmax. | Quince tests productivos en `sampler::tests`: ±ln(2), distribución cerrada, temperaturas subnormales, CDF determinista, empates, overflow, uniformes extremos y parámetros inválidos. El control negativo del antiguo umbral `<1e-9` está registrado en [reparación](operation-fragment-repair-2026-11-03.md). | **Completo** para el sampler descrito. La semilla pública continúa ignorada como contrato histórico documentado; no se atribuye reproducibilidad por semilla. |
| H12. CLI `main`: ejecuta un escalar una vez y emite `[BML placeholder: …]`, sin loop autorregresivo. | `run_text_model`, carga causal persistida y forward nativo/simbólico; tokenización, caché, logits, sampling y emisión por bytes. `--bml` tiene una ruta RPN explícita, no un alias de la nativa. | C51/C53/C57, ejecutable CLI y oráculos causales. [TinyLlama histórico](tinyllama-legacy-mapped.json) y [AOT causal](tinyllama-compiled-causal.json): diez tokens reales. [BOS simbólico completo](tinyllama-bos-logit-diagnostic.json): vector completo y argmax. | **Completo** para generación local requerida y perfiles declarados. Los diez tokens no son prueba de paridad llama.cpp; la validación simbólica real adicional sigue en H01. |
| H13. API `generate_tokens`: ignora el resultado numérico y devuelve prompt más texto fijo en JSON. | `LocalModel` selecciona backend nativo, causal persistido o simbólico; `generate_text_local` delega a `generate_local`, que usa logits del modelo y KV por petición. | C58/C67 `bml-server::compiled_server`: proceso HTTP real, JSON exactamente «é» sin eco; pesos reubicados verificados; sustituir programa almacenado y actualizar su hash cambia la respuesta. Pruebas de causalidad aritmética están separadas en C52/C64/C65. | **Completo** para contrato de backend/HTTP probado. El fixture HTTP es pequeño y no acredita precisión de un modelo grande servido por red. [Registros](compiled-causal-server.json), [simbólico](symbolic-aot-server.json). |
| H14. API `completions(stream=true)`: emite literalmente `B`, `M`, `L`. | JSON y SSE comparten `generate_local`; SSE ensambla bytes, emite las piezas calculadas y termina una vez ante EOS, contexto o error. | C58/C67: piezas `[«é», vacío]`, exactamente un terminal y un DONE; contexto agotado/incorrecto, backend reubicado y bytecode modificado. `historical_stream_and_completion_share_causal_decode_without_prompt_echo` comprueba la ruta compartida. | **Completo** para streaming y contenido calculado por el modelo. No se usa el éxito de transporte para afirmar equivalencia numérica grande. |

## Superficies adicionales y migraciones que no deben desaparecer del alcance

| ID | API y consumidor | Evidencia / obligación | Estado |
| --- | --- | --- | --- |
| E01 | `bml_rmsnorm_program` y sus consumidores directos de RPN. | C93 comprueba el contrato de factor recíproco **precalculado**, multiplicación IEEE, signos y escrituras. `bml_rmsnorm_full_program` y el constructor configurable son la alternativa de RMS completo comprobada en C92/C93. | **Completo** con contratos distintos explícitos; el helper de escala no debe describirse como cálculo completo de RMS. |
| E02 | `bml_swiglu_program` y `bml_rope_program`/`bml_rope_vector_program`; `HotLoop` consume los programas con sus bindings. | C86 comprueba sigmoid real y escrituras con ±ln(2); C93 comprueba coordenadas de contexto, ambas componentes, frecuencias por cabeza y 1024 pares. | **Completo** en los perfiles acotados declarados; no se infiere de otra ruta que nunca llame esos helpers. |
| E03 | `Runtime::execute_fragments_parallel` y `execute_with_cores`; consumidores de los `OperationFragment` de H02–H07. En `3817562` se enviaban chunks a hilos sin demostrar las dependencias. | C87 verifica orden RAW/WAR/WAW, índices absolutos/dinámicos, stack compartido y un oráculo secuencial de programas generados. La prueba interna de entrada concurrente tiene timeout. | **Completo** para semántica de dependencias; speedup y residencia L1 no acreditados. |
| E04 | `compile_distributed` y `compile_distributed_bml`, añadidos después del snapshot inicial. | Rechazan modelos válidos con mensaje de migración a `symbolic_artifact::compile_directory_bounded` y `SymbolicPipeline`; C105 conserva ese rechazo explícito. No generan un resultado fingido. | **Retirado explícito**. La sustitución del retorno histórico por el nuevo artefacto es una migración; estas funciones no se cuentan como compiladores funcionales. |
| E05 | `SymbolicPipeline`, workers por rango y CLI `distribute`, `prepare-shards`, `deploy-shards`; sustituyen los embeddings fabricados y el hidden interpretado como logits del camino distribuido posterior. | C100–C104: GQA de tres capas, rangos 1/2/3, sesiones KV, identidad y programas persistidos con `max_nodes=0`; error de confirmación invalida sin afirmar rollback remoto. [Stories15M](distributed-stories-real.json), [workers sin GGUF](source-free-shard-stories.json) y [SSH loopback](ssh-loopback-shards.json) documentan una posición real. | **Parcial**: faltan ejecución entre hosts distintos e inferencia simbólica completa del modelo mayor. El coordinador aún abre el GGUF para embeddings y cabeza; no describirlo como fuente-free. |
| E06 | Cierre G1 y tarea 8.5.9, que abarca APIs consumidoras y toda sustitución pública. | La matriz precedente evita usar ausencia de callers, un grep sin resultados o un gate verde como prueba de corrección. H01 y E05 conservan límites y trabajo verificable pendientes. | **Pendiente**: no cambiar 8.5.9 ni archivar a partir de este informe. |

## Qué falta y cómo comprobarlo

1. **H01, inferencia real adicional.** Conservar el BOS como diagnóstico. Crear
   un nuevo AOT/campaña para `[1,29958]` con contexto 2 y cotas explícitas; los
   probes nativos apoyan RoPE32 y atención20, pero no garantizan que todos los
   intermedios simbólicos de posición 1 entren en las demás cotas. Guardar ambos
   vectores por posición, finitud, identidad y posiciones KV. No inventar una
   tolerancia posterior para convertir el diagnóstico en paridad aprobada.
2. **E05, hosts reales.** Reutilizar artefactos, paquetes y protocolo causal
   comprobados; demostrar cobertura completa sobre dos hosts distintos y
   conservar logits/IDs, hashes de binarios/shards, identidades de hosts y
   fallos de sesión. Dos procesos, contenedores en un host o SSH loopback no
   sustituyen esa campaña. La VM propia futura debe verificarse antes de
   presentar el requisito como satisfecho.
3. **E04, migración.** Mantener visible el rechazo explícito de las antiguas
   funciones y la ruta completa que deben utilizar sus consumidores. No
   restablecer sus antiguos bytecodes incompletos ni atribuirles éxito porque
   no haya callers productivos.
4. **E06, integración final.** Comprobar el inventario sobre el commit integrado,
   ejecutar los targets trazados y el gate canónico, preservar cero discrepancias
   históricas y cero avisos nuevos, y validar OpenSpec estricto. Cada resultado
   debe identificar el snapshot realmente ejecutado. Este informe estático no
   es una campaña nueva de tests ni acredita mutaciones o hardware pendiente.

## Límites de la documentación histórica

- La propuesta inicial incluye `sigmoid(1.7*gate)` y split/lookup llamado BPE;
  la spec y las tareas vigentes corrigieron esos comportamientos. D6 inicial
  también proponía cero sin buffer; el contrato vigente es NaN. Se conservan
  como historia, no como instrucciones para reintroducir defectos.
- La [auditoría del 2026-11-03](fix-placeholders-closure-audit-2026-11-03.md)
  reprodujo matmul y sampler incorrectos en otro snapshot. Sus fallos motivaron
  reparaciones posteriores; no deben describirse como fallos actuales ni
  olvidarse al evaluar si una prueba estructural era suficiente.
- La comparación con llama.cpp se realizó y documenta diferencias: los diez
  tokens históricos terminan en «Step 3: Add» frente a «Step 2: Create»; sobre
  el mismo prefijo el máximo error de logits es 0,2150369823. G8 pide comparar y
  explicar el alcance, no afirmar igualdad inexistente. El diagnóstico BML BOS
  usa otra referencia y otro prefijo; sus errores no reemplazan esa comparación.
- El dominio simbólico real y el tensor nativo IEEE permanecen distintos.
  Ningún resultado de una ruta nativa completa demuestra por sí solo una
  identidad BML, y una declaración de cotas no garantiza evaluabilidad de todos
  los intermedios cerrados.

## Superficies primarias leídas

Además de los archivos históricos indicados, se inspeccionaron las
implementaciones actuales `gguf_compiler`, `symbolic_artifact`,
`symbolic_bounded`, `bml_ops`, `op_fragments`, `operation_rms`,
`operation_attention`, `operation_mlp`, `operation_layer`, `rpn`, `sampler`,
`runtime`, `symbolic_causal`, `symbolic_transport`, CLI y servidor, así como los
tests nombrados en la matriz. Los contratos Cxx se resuelven por package y
target en `quality/traceability.json`; no se usan nombres de archivo de tests
de compiler para los contratos de operaciones que pertenecen a `bml-runtime`.

Esta enumeración y los resultados históricos son evidencia de lectura y
procedencia. El estado final requiere todavía la verificación de integración
y los contratos pendientes señalados; no se afirma «14/14 placeholders cerrados».
