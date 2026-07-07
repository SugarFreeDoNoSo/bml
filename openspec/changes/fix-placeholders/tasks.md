## 1. Matmul vectorial completo (acumulador de suma de productos)

- [ ] 1.1 Actualizar `compile_matmul_fragment()` para incluir acumulador: push 0.0 antes del loop interno, `acc = bml_add(acc, bml_mul(w, x))` en cada iteración, `StoreResult(slot, i, acc)` al final.
- [ ] 1.2 Actualizar `build_transformer_dag()` para usar todos los pesos del tensor (no solo el primero) como `Const` en el pool.
- [ ] 1.3 Asignar bases de pesos correctamente: cada tensor ocupa `n_rows * n_cols` posiciones en el pool.
- [ ] 1.4 Pruebas: matmul de una matriz 2x2 conocida, verificar resultado exacto.

## 2. RoPE completo

- [ ] 2.1 Actualizar `build_transformer_dag()` para aplicar RoPE a todos los pares de dimensiones (n_embd/2 iteraciones), no solo al primero.
- [ ] 2.2 Precomputar todos los pares de cos/sin como `Const` en el pool.
- [ ] 2.3 Pruebas: verificar que RoPE no cambia los valores en posición 0.

## 3. SwiGLU real

- [ ] 3.1 Actualizar `compile_mlp_fragment()` para implementar `sigmoid(1.7 * gate)` via `exp2`/`add`/`div`.
- [ ] 3.2 Implementar `mul(gate, sigmoid)` y `mul(result, up)` en el fragmento.
- [ ] 3.3 Pruebas: verificar que SwiGLU(0) = 0 y SwiGLU(1) > 0.

## 4. VarIndexed sin buffer

- [ ] 4.1 Eliminar `stack.push(0.0)` placeholder en `evaluate_with_ctx` de rpn.rs.
- [ ] 4.2 Hacer que `VarIndexed` retorne `f64::NAN` cuando no hay buffer (en lugar de 0.0).
- [ ] 4.3 Pruebas: verificar que `VarIndexed` sin buffer retorna NaN, con buffer retorna el valor correcto.

## 5. Tokenización

- [ ] 5.1 Leer `tokenizer.ggml.tokens` (array de strings) del GGUF.
- [ ] 5.2 Leer `tokenizer.ggml.scores` (array de f32) del GGUF.
- [ ] 5.3 Leer `tokenizer.ggml.model` (ej. "llama") del GGUF.
- [ ] 5.4 Implementar tokenización BPE básica: split por espacios + lookup en vocabulario.
- [ ] 5.5 Implementar detokenización: token ID → string.
- [ ] 5.6 Pruebas: tokenizar "Hello world", verificar que produce token IDs válidos.

## 6. Sampling

- [ ] 6.1 Implementar `argmax(logits)` para greedy (temp=0).
- [ ] 6.2 Implementar `softmax(logits / temp)` + muestreo para temp>0.
- [ ] 6.3 Pruebas: verificar que greedy selecciona el token con mayor logit.

## 7. Generación de tokens real

- [ ] 7.1 Actualizar `bml-cli` para: tokenizar prompt → ejecutar transformer → obtener logits → sampling → detokenizar → imprimir.
- [ ] 7.2 Actualizar `bml-server` para: lo mismo pero via SSE streaming.
- [ ] 7.3 Implementar loop de generación: alimentar el token generado como input del siguiente paso.
- [ ] 7.4 Pruebas: `bml-cli -m model.bmlgraph/ -p "Hello"` produce texto coherente (no placeholder).

## 8. Comparación con llama.cpp

- [ ] 8.1 Ejecutar `llama-cli -m /root/tinyllama.gguf -p "Hello" -n 10` y capturar output.
- [ ] 8.2 Ejecutar `bml-cli -m /root/tinyllama.bmlgraph/ -p "Hello" -n 10` y capturar output.
- [ ] 8.3 Comparar outputs: deben ser similares (no idénticos por diferencias de implementación).
- [ ] 8.4 Documentar diferencias y posibles causas.

## 9. Cierre

- [ ] 9.1 `openspec validate fix-placeholders` pasa sin errores.
- [ ] 9.2 `cargo test --workspace` pasa.
- [ ] 9.3 Commit y push.
