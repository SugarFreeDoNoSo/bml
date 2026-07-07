## Context

El pipeline BML está completo de extremo a extremo pero con 14 placeholders. El CLI retorna `NaN` porque el DAG usa un solo peso por tensor en lugar de matmul vectorial. Los tokens generados son simulados ("B", "M", "L"). La tokenización y sampling no existen.

## Goals / Non-Goals

**Goals:**

- G1. Eliminar los 14 placeholders identificados en el código.
- G2. Implementar matmul vectorial completo con acumulador de suma de productos.
- G3. Implementar RoPE completo sobre todos los pares de dimensiones.
- G4. Implementar SwiGLU real (sigmoid + mul).
- G5. Implementar tokenización del prompt (leer tokenizer del GGUF).
- G6. Implementar sampling greedy + temperatura.
- G7. Generar tokens reales en el CLI y servidor.
- G8. Comparar output con llama.cpp para el mismo prompt.

**Non-Goals:**

- NG1. No se implementa KV cache (queda para optimización futura).
- NG2. No se implementa attention completa multi-head (versión simplificada single-head).
- NG3. No se implementa cuantización en runtime (los pesos se dequantizan en compile-time).

## Decisions

- **D1 — Acumulador en pila para matmul.** El fragmento de matmul empuja `0.0` (acumulador inicial) antes del loop interno, y cada iteración hace `acc = bml_add(acc, bml_mul(w, x))`. Al final del loop, `StoreResult(slot, i, acc)`.
- **D2 — RoPE con loop sobre pares.** Un loop de `n_embd/2` iteraciones, cada una aplicando cos/sin precomputados a un par de dimensiones.
- **D3 — SwiGLU como 3 operaciones BML.** `sigmoid(gate) = 1/(1+exp(-gate))` via `exp2`/`add`/`div`, luego `mul(gate, sigmoid)`, luego `mul(result, up)`.
- **D4 — Tokenización desde GGUF.** Leer `tokenizer.ggml.tokens` (array de strings) y `tokenizer.ggml.scores` (array de f32) del GGUF. Implementar tokenización BPE básica.
- **D5 — Sampling greedy.** `argmax(logits)` para temp=0. Para temp>0, `softmax(logits/temp)` + muestreo aleatorio.
- **D6 — VarIndexed sin buffer.** En `evaluate_with_ctx` (sin buffer), `VarIndexed` retorna `0.0` en lugar de placeholder. En `execute_full` (con buffer), lee del buffer.

## Risks / Trade-offs

- **R1 — Matmul lento.** El matmul BML con acumulador será más lento que BLAS. *Mitigación:* el benchmark ya documentó esto; el objetivo es correctitud, no velocidad.
- **R2 — Tokenización BPE compleja.** BPE requiere merges y vocabulario. *Mitigación:* implementar versión básica que funcione para prompts simples.
- **R3 — Memoria de pesos.** Dequantizar todos los pesos a f32 puede usar mucha RAM. *Mitigación:* dequantizar bajo demanda (lazy).
