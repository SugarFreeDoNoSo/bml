# Validación previa a comparar inferencia

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
