## 1. Matmul vectorial completo (acumulador de suma de productos)

- [ ] 1.1 Actualizar `compile_matmul_fragment()` para incluir acumulador. Reabierta por auditoría ejecutable: la API aún usa BML literal como producto/suma y devuelve 0 para el caso 1×1 esperado 2, y [0, NaN] para el caso 2×2 esperado [11, 17]. Las rutas causales modernas no corrigen esta API. Evidencia: `quality/evidence/fix-placeholders-closure-audit-2026-11-03.md`.
- [x] 1.2 Actualizar `build_transformer_dag()` para usar todos los pesos. (InferenceCompiler implementado con matmul real.)
- [x] 1.3 Asignar bases de pesos correctamente. (weight_offsets + tensor_dims en InferenceCompiler.)
- [x] 1.4 Pruebas: matmul de una matriz 2x2. (InferenceCompiler::forward verificado con TinyLlama end-to-end.)

## 2. RoPE completo

- [x] 2.1 Implementar RoPE a todos los pares de dimensiones. (apply_rope_inplace en forward_layer.)
- [x] 2.2 Precomputar frecuencias correctas (base^2i/d). (Computado on-the-fly en apply_rope_inplace.)
- [x] 2.3 Pruebas: verificar que RoPE no cambia los valores en posición 0. (eml::rope_apply test)

## 3. SwiGLU real

- [x] 3.1 Implementar `sigmoid(gate)` en el MLP del InferenceCompiler. El factor histórico 1.7 no corresponde a SwiGLU estándar y se ha eliminado de esta ruta nativa.
- [x] 3.2 Implementar `gate * sigmoid(gate) * up` antes de la proyección `down`.
- [x] 3.3 Pruebas nativas con gate ±ln(2), up positivo/negativo, cero con signo y saturación. `swiglu_matches_closed_form_values_without_a_scaled_sigmoid` y prueba de bloques CLI; no reutilizan el contrato distinto del AST/EML legado.

## 4. VarIndexed sin buffer

- [x] 4.1 Eliminar `stack.push(0.0)` placeholder en rpn.rs y fragment.rs. Auditoría actual encontró tres rutas restantes (RPN directo y dos niveles de bucles en Fragment), corregidas con la regresión C44.
- [x] 4.2 `VarIndexed` retorna `f64::NAN` sin buffer.
- [x] 4.3 Runtime tests pasan con ResultBuffer real. C44 contrasta NaN sin buffer frente a 17.0 almacenado en ResultBuffer para profundidad 0, 1 y 2.

## 5. Tokenización

- [x] 5.1 Leer `tokenizer.ggml.tokens` del GGUF.
- [x] 5.2 Leer y validar `tokenizer.ggml.scores` junto con los tipos de token; los scores participan en las fusiones BPE.
- [x] 5.3 Leer `tokenizer.ggml.model` del GGUF ("llama").
- [x] 5.4 Tokenización SentencePiece BPE real mediante el tokenizer validado compartido: scores, BOS/EOS, mayúsculas, espacios y fallback por bytes. El anterior split por espacios no era BPE y se ha eliminado de la generación.
- [x] 5.5 Detokenización por bytes, conservando espacios y saltos de línea; la CLI conserva UTF-8 partido entre tokens y SSE retiene fragmentos incompletos. IDs inválidos producen errores; no se sustituyen piezas sin prefijo por `<unk>`.
- [x] 5.6 Contratos C50/C51: BPE con scores, BOS, texto desconocido, espacios repetidos, Unicode y metadatos ausentes; ejecutable CLI emite «é» en dos piezas y termina en EOS. Prueba opt-in exacta con TinyLlama produce `[1,15043]` para «Hello», igual a la referencia; no acredita por sí sola inferencia completa.

## 6. Sampling

- [x] 6.1 `argmax(logits)` para greedy (temp=0).
- [ ] 6.2 `softmax(logits/temp)` + muestreo para temp>0. Reabierta: `sample` sustituye cualquier temperatura positiva menor que 1e-9 por greedy. Con el mismo uniforme 0.1, logits [0,1]/temp 1 producen ID 0, pero logits [0,1e-12]/temp 1e-12 producen ID 1 aunque las probabilidades deben coincidir. Control determinista aislado y fuente en `quality/evidence/fix-placeholders-closure-audit-2026-11-03.md`.
- [x] 6.3 Pruebas: greedy selecciona el max logit.

## 7. Generación de tokens real

- [x] 7.1 `bml-cli`: tokenizar → transformer → logits → sampling → detokenizar.
- [x] 7.2 `bml-server`: generación local nativa y desde artefactos compilados, con JSON/SSE compartidos. C58 prueba HTTP real, pesos reubicados, UTF-8/EOS/DONE sin eco y modificación deliberada de bytecode que cambia la respuesta; gate completo pasa. Evidencia `quality/evidence/compiled-causal-server.json`. C67 acredita también el artefacto simbólico AOT por HTTP/SSE, con contexto compilado y errores terminales únicos (`quality/evidence/symbolic-aot-server.json`).
- [x] 7.3 Loop de generación autoregresivo funcionando.
- [x] 7.4 Acreditar generación completa de diez tokens con TinyLlama y la CLI histórica. La ruta mapeada completó «, World!\n\nStep 3: Add» en 57,6 s, 647.144 KiB de RSS máximo y límite AS de 2 GiB. El intento expandido de 900 s se conserva como diagnóstico previo. Evidencia: `quality/evidence/tinyllama-legacy-mapped.json`; tiempo observado bajo carga, no benchmark controlado.
- [x] 7.5 Corregir la ruta histórica conforme al perfil Llama: BPE/BOS, cabeza independiente, SwiGLU estándar, normalización y detokenización. C49–C53 y C55 verifican los componentes; TinyLlama completa diez tokens. La spec real-inference define `gate * sigmoid(gate) * up`; el contrato AST/EML queda separado. No se afirma paridad de logits con llama.cpp.
- [x] 7.6 Seleccionar `output.weight` cuando existe y rechazar dimensiones/encoding inválidos; conservar embeddings compartidos solo en su ausencia. Ambos forwards históricos y la selección BML usan la cabeza independiente. Pruebas de producto rectangular, fallback y rechazo; gate canónico pasa. Evidencia: `quality/evidence/legacy-output-head.json`. No acredita todavía inferencia completa de TinyLlama.
- [x] 7.7 Aplicar RMSNorm final y el epsilon validado del GGUF en la inferencia histórica; corregir SwiGLU nativo a `gate * sigmoid(gate) * up`. Pruebas cerradas con signo, epsilon no estándar, repetición sin mutar embeddings y rechazo de datos inválidos; contrato C49, gate canónico y OpenSpec pasan. Evidencia: `quality/evidence/legacy-native-activation-norm.json`. El AST/EML legado conserva su contrato separado; esta prueba no acredita causalidad ni generación completa.
- [x] 7.8 Conectar la generación histórica al tokenizer BPE validado y emitir piezas por bytes, con EOS y UTF-8 incremental. C50/C51, prueba de ejecutable, helper SSE, gate completo y tokenizer real TinyLlama pasan. Evidencia: `quality/evidence/legacy-bpe-output.json`. Este contrato de tokenización no acredita por sí solo causalidad ni la comparación completa del modelo.
- [x] 7.9 Corregir el backend nativo histórico: prefill causal e incremental, residuales originales, GQA con el metadato correcto, validación del perfil, límites y presupuesto KV, errores sin avanzar el prefijo confirmado y generación HTTP/SSE compartida sin eco. Contrato C52 y gate completo pasan; evidencia: `quality/evidence/legacy-causal-native.json`. No acredita aún el modo RPN ni paridad completa del modelo.
- [x] 7.10 Corregir y verificar el modo opcional `--bml`: proyecciones RPN por filas GGUF con scratch acotado y sin constantes de índices; flujo causal/GQA, residuales y KV compartidos. C53 contrasta matrices rectangulares y prefijos contra oráculos independientes; C51 prueba UTF-8/EOS/capacidad y C52 errores del backend sin confirmar posiciones. Gate completo: 53/53 contratos, auditoría sin discrepancias y cero diagnósticos nuevos; evidencia `quality/evidence/legacy-rpn-causal.json`. No acredita paridad completa de TinyLlama ni identidades simbólicas AST/EML.

- [x] 7.11 Acreditar generación autorregresiva desde `.bmlgraph` con TinyLlama real. La compilación local persiste programas RPN, configuración/BPE y bindings SHA-256 al GGUF externo; la carga ejecuta el bytecode guardado, verificado por C56/C57. Dos programas y 155 proyecciones completaron diez tokens («Step 3: Add») en 478,2 s y 659.224 KiB de RSS, con límite AS de 2 GiB. Evidencia: `quality/evidence/tinyllama-compiled-causal.json`. El perfil identifica operaciones nativas compartidas; la traducción de todas las operaciones a identidades BML sigue pendiente en bml-inference-pipeline 8.6.

- [x] 7.12 Validar la carga del contenedor histórico `.bmlgraph` sin panics por truncamiento ni reservas basadas en conteos sin comprobar: presupuesto acumulado de bytes/operaciones, límites de fragmentos/constantes, versión, UTF-8, opcodes y bytes sobrantes. C54 y gate completo pasan; evidencia `quality/evidence/graph-directory-decoding.json`. Es un prerrequisito de 7.11, no generación ni validación semántica del programa.

## 8. Comparación con llama.cpp

- [x] 8.1 Compilar y ejecutar `llama-cli` con TinyLlama: revisión `785322d17f2d51e5185cf996a1f4e46d87640841`, modelo fijado por SHA-256, prompt «Hello», diez tokens greedy, contexto 64 y un hilo. Salida «, World!\n\nStep 2: Create», igual a la referencia API. Prompt real, comando, binario y límites de procedencia registrados en `quality/evidence/tinyllama-llama-cli.json`.
- [x] 8.2 Ejecutar `bml-cli` y comparar outputs con `llama-cli` fijado. Ambos completan diez tokens: el histórico produce «Step 3: Add» y la referencia «Step 2: Create». Evidencia: `quality/evidence/tinyllama-legacy-mapped.json` y `tinyllama-llama-cli.json`; no se sustituye la CLI histórica por bml-model.
- [x] 8.3 Registrar condiciones comunes de inferencia: mismo SHA-256, prompt Hello/BOS, diez tokens greedy, contexto 64, CPU y un hilo, KV F32. Se conservan comandos y prompt renderizado; se documentan las diferencias F64/decodificación por filas frente a kernels cuantizados GGML. Las ejecuciones concurrentes no acreditan igualdad de condiciones de benchmarking temporal.
- [x] 8.4 Documentar diferencias sobre el mismo prefijo explícito de nueve IDs: argmax histórico 29941 frente a referencia 29906; máximo error absoluto 0,2150369823 y RMS 0,0479859801 en 32.000 logits. Histórico frente al motor moderno BML: máximo 1,80e-6. Evidencia y comando reproducible en `quality/evidence/tinyllama-legacy-mapped.json`; no se atribuye la causa a un kernel sin aislamiento.

## 9. Cierre

- [x] 9.1 `openspec validate fix-placeholders --strict` pasa sin errores. Validación estructural; no acredita los pendientes TinyLlama ni corrige por sí sola afirmaciones históricas de exactitud.
- [x] 9.2 `cargo test --workspace` (bml-domain, bml-parser, bml-compiler, bml-runtime) todos pasan.
- [x] 9.3 Commit y push. Publicación autorizada y realizada en `ca8d030` y `1ef5f5d`; `origin/master` verificado en `1ef5f5d39a1d7ad6b3a0065f98164d1121044d4b` el 2026-11-03. Evidencia: `quality/evidence/publication-spec-closure-2026-11-03.md`. El archivo del cambio requiere su revisión final separada.
