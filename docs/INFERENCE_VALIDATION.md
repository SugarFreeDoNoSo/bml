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
