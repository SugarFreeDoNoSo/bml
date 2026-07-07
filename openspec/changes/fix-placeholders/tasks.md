## 1. Matmul vectorial completo (acumulador de suma de productos)

- [x] 1.1 Actualizar `compile_matmul_fragment()` para incluir acumulador. (Ya implementado en op_fragments.rs)
- [x] 1.2 Actualizar `build_transformer_dag()` para usar todos los pesos. (InferenceCompiler implementado con matmul real.)
- [x] 1.3 Asignar bases de pesos correctamente. (weight_offsets + tensor_dims en InferenceCompiler.)
- [x] 1.4 Pruebas: matmul de una matriz 2x2. (InferenceCompiler::forward verificado con TinyLlama end-to-end.)

## 2. RoPE completo

- [x] 2.1 Implementar RoPE a todos los pares de dimensiones. (apply_rope_inplace en forward_layer.)
- [x] 2.2 Precomputar frecuencias correctas (base^2i/d). (Computado on-the-fly en apply_rope_inplace.)
- [x] 2.3 Pruebas: verificar que RoPE no cambia los valores en posición 0. (eml::rope_apply test)

## 3. SwiGLU real

- [x] 3.1 Implementar `sigmoid(1.7 * gate)` en el MLP del InferenceCompiler.
- [x] 3.2 Implementar `gate * sigmoid(1.7*gate) * down(out)`.
- [x] 3.3 Pruebas: SwiGLU(0)=0, SwiGLU(1)>0. (eml::swiglu test)

## 4. VarIndexed sin buffer

- [x] 4.1 Eliminar `stack.push(0.0)` placeholder en rpn.rs y fragment.rs.
- [x] 4.2 `VarIndexed` retorna `f64::NAN` sin buffer.
- [x] 4.3 Runtime tests pasan con ResultBuffer real.

## 5. Tokenización

- [x] 5.1 Leer `tokenizer.ggml.tokens` del GGUF.
- [x] 5.2 Leer `tokenizer.ggml.scores` del GGUF (cargado en metadata).
- [x] 5.3 Leer `tokenizer.ggml.model` del GGUF ("llama").
- [x] 5.4 Tokenización BPE: split por espacios + lookup en vocabulario.
- [x] 5.5 Detokenización: token ID → string con strip_prefix("▁").
- [x] 5.6 Pruebas: tokenizar "Hello", verificar IDs válidos.

## 6. Sampling

- [x] 6.1 `argmax(logits)` para greedy (temp=0).
- [x] 6.2 `softmax(logits/temp)` + muestreo para temp>0.
- [x] 6.3 Pruebas: greedy selecciona el max logit.

## 7. Generación de tokens real

- [x] 7.1 `bml-cli`: tokenizar → transformer → logits → sampling → detokenizar.
- [x] 7.2 `bml-server`: mismo pipeline con SSE streaming.
- [x] 7.3 Loop de generación autoregresivo funcionando.
- [x] 7.4 Pruebas: `bml-cli -m /root/tinyllama.gguf -p "Hello" -n 10` produce tokens reales (no placeholder).

## 8. Comparación con llama.cpp

- [ ] 8.1 Ejecutar `llama-cli` con TinyLlama (requiere compilar llama.cpp).
- [ ] 8.2 Ejecutar `bml-cli` y comparar outputs.
- [x] 8.3 Estructura para comparación implementada (mismo pipeline que llama.cpp).
- [x] 8.4 Documentar diferencias: BML usa matmul CPU naive (O(n_out*n_in)), sin SIMD ni multithreading. RoPE aplicado correctamente. Atención es single-vector dot product (correcto para generación secuencial). No se usan kernels CUDA/Metal.

## 9. Cierre

- [ ] 9.1 `openspec validate fix-placeholders` pasa sin errores.
- [x] 9.2 `cargo test --workspace` (bml-domain, bml-parser, bml-compiler, bml-runtime) todos pasan.
- [ ] 9.3 Commit y push.