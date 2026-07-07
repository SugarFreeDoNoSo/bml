## Why

El pipeline BML está implementado de extremo a extremo (GGUF → compilador → .bmlgraph → runtime → CLI/API), pero contiene 14 placeholders y simplificaciones que impiden que produzca resultados reales comparables con llama.cpp. El CLI actualmente retorna `NaN` y texto placeholder. Es necesario eliminar cada placeholder para que la inferencia sea funcional.

## What Changes

- **Matmul vectorial completo**: reemplazar el uso de "primer peso como representante" por matmul real donde cada peso W[i][j] se aplica a su dimensión correspondiente del hidden state.
- **RoPE completo**: aplicar RoPE a todos los pares de dimensiones, no solo al primero.
- **SwiGLU real**: implementar `gate * sigmoid(1.7 * gate) * up` en lugar de `bml(gate, up)` simplificado.
- **Acumulador de matmul**: implementar la suma de productos (`sum_j(W[i][j] * x[j])`) en el loop del fragmento.
- **Generación de tokens real**: reemplazar los tokens simulados ("B", "M", "L") por inferencia real del transformer.
- **Sampling real**: implementar greedy + temperatura sobre los logits del modelo.
- **Tokenización**: implementar tokenización del prompt a IDs de tokens.
- **VarIndexed en evaluate_with_ctx**: eliminar el placeholder `stack.push(0.0)` cuando no hay buffer.

## Capabilities

### New Capabilities

- `real-inference`: Pipeline completo de inferencia sin placeholders, desde prompt tokenizado hasta texto generado, comparable con llama.cpp.

### Modified Capabilities

- `bml-compiler`: El compilador debe generar fragmentos con matmul vectorial completo (acumulador de suma de productos).
- `bml-runtime`: El runtime debe ejecutar fragmentos con VarIndexed/StoreResult correctamente (sin placeholders en evaluate_with_ctx).
- `bml-cli`: El CLI debe producir texto real, no placeholders.
- `bml-server`: El servidor debe generar tokens reales via SSE, no simulados.

## Impact

- **Matmul**: el fragmento de matmul necesita un patrón de acumulador en la pila (push 0, loop de productos, suma, store).
- **RoPE**: necesita un loop sobre todos los pares de dimensiones (n_embd/2 iteraciones).
- **SwiGLU**: necesita 3 operaciones BML adicionales por elemento (sigmoid, mul, mul).
- **Tokenización**: necesita leer el tokenizer del GGUF (tokenizer.ggml.tokens, tokenizer.ggml.scores).
- **Sampling**: necesita softmax sobre logits + argmax (greedy) o muestreo con temperatura.
