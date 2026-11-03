# Validación previa a comparar inferencia

La ruta nueva `bml-inference`/`bml-model` tiene su perfil, comandos y evidencia
en [BML 1.0](BML_1_0.md). Los límites del compilador histórico descritos aquí
siguen aplicándose a sus APIs experimentales; no son la implementación nueva
ni heredan sus resultados de validación.

El objetivo es comparar inferencia real de BML y llama.cpp sobre el mismo GGUF.
Las pruebas de lectura, dequantización y proyección son requisitos previos;
sus tiempos no representan tokens por segundo.

## Contrato de ejecución actual

Las rutas `compile_gguf`, `compile_gguf_fast`, `InferenceCompiler::open`,
`compile_distributed` y `compile_distributed_bml` solo admiten la arquitectura
`llama` experimental. Cualquier otra arquitectura, incluida `gemma4`, produce
un error explícito antes de leer el vocabulario o expandir pesos. La inspección
de metadatos y los codecs de `bml-gguf` siguen disponibles para otros modelos.

Aceptar `llama` no certifica equivalencia con llama.cpp: el tokenizer, la
normalización, los residuales, RoPE y la atención de las rutas experimentales
todavía necesitan validación independiente. Gemma4 requiere además su propia
arquitectura; no se puede ejecutar correctamente sustituyendo el nombre del
modelo en la ruta Llama. BML no tiene aquí un backend GPU validado.

GGUF guarda el eje `ne0` contiguo. Un tensor de embeddings `[n_embd, vocab_size]`
contiene un vector contiguo por token. Una proyección `[n_in, n_out]` contiene
una fila contigua de `n_in` pesos por salida. La ruta experimental conserva ese
orden al consultar embeddings, proyectar activaciones y calcular logits con
embeddings compartidos.

## Regresiones pequeñas y reproducibles

El perfil optativo `llama-symbolic-bml-bounded-f64-f32kv-v4` extiende la
aritmética acotada a las nueve familias ejecutadas: proyección aprendida,
productos query/key, suma probability/value, RMSNorm, RoPE, softmax, SwiGLU,
residual y escala. Comprueba cada BML y mantiene los pesos externos. Los dots
usan un cuerpo constante dentro de un bucle: su código no crece con el ancho,
pero los pasos y las lecturas siguen creciendo linealmente. Los nuevos índices
de contexto se guardan en contenedores single-file v3; los contenedores legacy
no los aceptan silenciosamente.

Softmax recibe los scores y una copia de su máximo seleccionado. Esa selección
usa comparaciones del host; todas las restas, exponenciales, sumas y divisiones
permanecen en el programa BML. El validador comprueba el máximo auxiliar. Esta
decisión evita la pérdida de precisión observada con un desplazamiento fijo
cuando todos los scores son negativos. Los literales JSON se leen con
`float_roundtrip`; la comparación con la firma canónica no usa tolerancias.

```sh
cargo run -p bml-cli --release --locked -- compile --model /ruta/stories15M.gguf \
  --output stories-bounded --symbolic --bounded --context-size 3 --symbolic-large-budget
cargo run -p bml-cli --release --locked -- run --model stories-bounded \
  --prompt "" --num-tokens 2 --context-size 3 --temp 0 --symbolic-large-budget
cargo test -p bml-compiler --test full_bounded_contracts --locked
cargo test -p bml-runtime --test indexed_symbolic_execution --locked
```

`--symbolic-large-budget` es una ampliación explícita para diagnóstico:
compilación de hasta 2M nodos/ops por programa y 4M ops totales; ejecución de
hasta 4M pasos y valores por programa. No elimina los límites del contenedor
ni garantiza evaluabilidad de un modelo o prompt. El perfil conserva límites
de operandos distintos para cada familia, que puede rechazar en ejecución.

El diagnóstico con publicación y recarga AOT, presupuesto de construcción cero
al cargar y comparación de todos los logits contra la ruta causal nativa es:

```sh
cargo run --release -p bml-compiler --example bounded_inference_diagnostic --locked -- \
  /ruta/stories15M.gguf /ruta/directorio-nuevo 2
```

La campaña Rust del 3 de noviembre de 2026 compiló Stories15M en 0,612 s,
publicó doce kernels y los recargó con presupuesto de construcción cero.
Completó los inputs `[1, 9038]`, con 32.000 logits finitos por posición y
argmax `9038`/`2501`, iguales a la ruta causal nativa. Los errores absolutos
máximos fueron `1.620e-8` y `2.945e-6`. La ejecución simbólica tardó
aproximadamente 142 y 137 segundos por posición en esta VM; son mediciones
locales del diagnóstico, no una comparación con llama.cpp ni una promesa de
rendimiento. Evidencia: `quality/evidence/full-bounded-2026-11-03.json`.

Un oráculo independiente F32 Llama está disponible sin imports de producción:

```sh
python3 tools/diagnostics/bounded_inference_oracle.py --self-check --output /ruta/autoprueba
python3 tools/diagnostics/bounded_inference_oracle.py /ruta/stories15M.gguf \
  --tokens 1,9038 --output /ruta/oraculo
```

Requiere Python 3.11+ y un compilador C11. Comprueba cada BML en una
implementación separada y compara contra fórmulas nativas independientes;
no sustituye la prueba del artefacto Rust. La selección del máximo de softmax,
la conversión F32 del KV y las constantes de posición están explícitas.

La reducción de código del dot no acredita mejoras de tokens/segundo: cada
peso aún exige muchas operaciones BML. RMS conserva su programa vectorial
lineal y el plan AOT de softmax conserva un programa por longitud de prefijo.
Su escalabilidad, modelos mayores y prompts adicionales requieren campañas
independientes. Las pruebas ordinarias siguen sin necesitar modelos descargados.

El perfil optativo `llama-symbolic-bml-bounded-rms-f64-f32kv-v3` reemplaza
exclusivamente la expansión RMSNorm y comprueba cada operación BML de esa
normalización bajo el contrato de reales finitos. Conserva las identidades de
las demás operaciones y la carga de artefactos v1/v2. Se selecciona con ambos
límites explícitos al compilar:

```sh
cargo run -p bml-cli --locked -- compile --model fixture.gguf --output fixture-symbolic \
  --symbolic --context-size 2 --rms-input-bound 16 --rms-weight-bound 16
cargo test -p bml-domain --test bounded_transformer_contracts --locked
cargo test -p bml-runtime --test checked_symbolic_execution --locked
cargo test -p bml-compiler --test bounded_rms_artifact_contracts --locked
```

Los presupuestos de compilación y ejecución siguen vigentes; el comando de
compilación anterior es para fixtures que caben en los límites por defecto.
La comprobación optativa con un GGUF local usa presupuestos ampliados explícitos:

```sh
cargo run -p bml-compiler --example bounded_rms_diagnostic --locked -- /ruta/stories15M.gguf
```

El diagnóstico Rust del 3 de noviembre de 2026 ejecutó los 288 elementos del
primer RMS de Stories15M, con todos los intermedios reales finitos, igualdad
bit a bit entre DAG y RPN persistido y error absoluto máximo `4.144e-11` frente
al cálculo nativo independiente. El caso cero produce un residuo aproximado de
`-3.912e-10`; no hay equivalencia exacta con IEEE ni tolerancia universal probada.
La ejecución del primer token aún falla en RoPE, cuya expansión no cambia en
v3. Esta evidencia acredita el RMS concreto, no inferencia simbólica completa.

```sh
cargo test -p bml-compiler --test inference_contracts --locked
cargo test -p bml-compiler --lib --locked inference_projection_respects_gguf_contiguous_input_axis
```

Los fixtures F32 usan matrices rectangulares en ambas orientaciones, signos,
fracciones y respuestas calculadas manualmente. El caso Q4_0 distingue tres
bloques de tokens con escalas diferentes. La prueba privada de proyección usa
un desplazamiento no nulo dentro del pool de pesos. Los fixtures de cero capas
aíslan embeddings y logits; no representan un transformer completo.

El caso Q4_0 no uniforme recorre los 16 valores de nibble con escalas +0,5 y
-1, distingue las mitades baja/alta del bloque y verifica embeddings y logits
contra un oráculo aritmético independiente (normas 172/688, producto 336).
El 30 de octubre de 2026 pasaron en Linux cloud las nueve regresiones de
`inference_contracts` y la prueba privada de proyección, con Rust 1.98.1,
`--locked --offline`. Esta comprobación sintética no valida el archivo real,
no usa una referencia ejecutable de llama.cpp y no constituye validación WSL.

El cierre del mismo hito sobre la base
`987e89e6d60e9aa6947538bee01494c51ed2e108` ejecutó:

```sh
CARGO_NET_OFFLINE=true bash scripts/check.sh 987e89e6d60e9aa6947538bee01494c51ed2e108
cargo llvm-cov --workspace --locked --offline --json --output-path /tmp/bml-coverage-20261030.json
python3 tools/quality.py coverage --report /tmp/bml-coverage-20261030.json
python3 tools/mutation_diff.py 987e89e6d60e9aa6947538bee01494c51ed2e108 /tmp/bml-incremental-20261030.diff
CARGO_NET_OFFLINE=true cargo mutants --workspace --list --in-diff /tmp/bml-incremental-20261030.diff
```

Resultado: 312 pruebas Rust aprobadas, cero fallidas, 15 opcionales ignoradas;
cinco pruebas Python aprobadas; formato correcto; cero discrepancias de
contratos; trazabilidad 15/15; Clippy con 123 diagnósticos existentes y cero
nuevos. Cobertura: 59,1222 % de líneas y 67,4444 % de funciones, sobre los
mínimos 52 %/60 %. La enumeración incremental de mutantes no encontró ninguno:
este incremento modifica exclusivamente pruebas y documentación; mutación N/A,
sin atribuirle un porcentaje de detección. No se ejecutó una campaña de mutación
completa ni auditoría RustSec (sin cambios de dependencias). Las comprobaciones
no requirieron descargar modelos. El cotejo racional independiente confirmó
normas 172/688 y producto 336 y que las expectativas de embeddings distinguen
mitades intercambiadas, nibbles intercalados y bloques intercambiados; no se
modificó el decodificador productivo para simular esos fallos.

## TinyLlama en la CLI histórica

La ruta GGUF histórica completó diez tokens greedy con el modelo fijado por
SHA-256 `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556`,
prompt `Hello`, contexto 64 y un hilo. Mantiene 608,2 MiB mapeados y solo
0,4 MiB de normalizaciones expandidas; decodifica una fila por proyección.
Se observaron 57,6 s y 647.144 KiB de RSS máximo bajo carga concurrente,
con límite de espacio de direcciones de 2 GiB. No es un benchmark controlado.

La salida fue `, World!\n\nStep 3: Add`; `llama-cli` fijado produjo
`, World!\n\nStep 2: Create`. Sobre el mismo prefijo de nueve tokens, el
máximo error absoluto entre logits es 0,2150369823 y el argmax difiere
(29941 frente a 29906). El máximo error frente al motor moderno de BML
es 1,80e-6. La comparación documenta la divergencia, sin aislar su causa.
Las proyecciones históricas F64 y los kernels cuantizados GGML no ejecutan
la misma aritmética. [Evidencia, comandos y hashes](../quality/evidence/tinyllama-legacy-mapped.json).
Esto valida generación por GGUF; no acredita la ruta compilada `.bmlgraph`
ni la traducción completa del transformer a identidades BML.

## TinyLlama desde el artefacto compilado

El mismo modelo se compiló como dos programas RPN compartidos por 155
proyecciones, con un manifest de 1.190.154 bytes y dos archivos de programa
de 48 bytes, sin copiar los pesos GGUF. La ejecución de diez tokens desde
`.bmlgraph` completó la misma salida textual que la ruta nativa mapeada en
478,2 s, con 659.224 KiB de RSS máximo y límite AS de 2 GiB.
[Evidencia, comandos y hashes](../quality/evidence/tinyllama-compiled-causal.json).

Son tiempos observados bajo carga concurrente, no un benchmark controlado.
El perfil `llama-causal-f64-f32kv-rpn-v1` ejecuta las proyecciones guardadas,
pero mantiene normalización, RoPE, atención, activación y residuales nativos.
La traducción completa del transformer a identidades BML continúa pendiente.

## Siguiente validación con un archivo real

Modelo seleccionado: `google/gemma-4-E2B-it-qat-q4_0-gguf`, revisión
`675cff42a74c774d6cb76f76d8eacb49b48c9b93`, archivo
`gemma-4-E2B_q4_0-it.gguf`, tamaño esperado **3.349.516.256 bytes**, SHA-256
`fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`.
Estos datos identifican la entrada prevista; las regresiones anteriores no
descargan ni verifican ese archivo.

El siguiente paso requiere un entorno con acceso al archivo existente: comparar
metadatos, tipos, dimensiones y una proyección acotada con una referencia
independiente de llama.cpp. Registrar la revisión de la referencia, los bloques
muestreados, las entradas firmadas, el dtype y las tolerancias. Evitar expandir
todos los pesos a f32 para esta comprobación.

Después deben validarse el tokenizer, un bloque completo, logits del primer
token y prefill/decode con KV. Solo entonces tiene sentido medir inferencia con
el mismo hash de modelo, prompt, contexto, cantidad de tokens, backend, hilos,
semilla y política de muestreo. Separar arranque frío/caliente, TTFT, prefill,
decode y máximos de RAM/VRAM. La memoria compartida de GPU forma parte de la
RAM del sistema y no se suma como VRAM adicional.

## Ejecución portátil

El [runner Docker CPU](../docker/gguf-validation/README.md) permite inspección
y comparación acotada de Q4_0 sin depender de WSL. Recibe el GGUF existente
como montaje de solo lectura y conserva JSON/logs; no habilita inferencia Gemma4.
