# Análisis: brecha con llama.cpp y optimizaciones faltantes

## 1. ¿Obtenemos el mismo resultado que llama.cpp?

**NO.** La salida de BML produce tokens sin sentido (`попу`, `коми`, `factory`)
mientras que llama.cpp produce texto coherente. Las razones:

### Problemas de calidad (no de velocidad)

| Problema | llama.cpp | BML | Impacto |
|----------|-----------|-----|---------|
| **Attention** | Softmax sobre todos los tokens del contexto | `tanh(dot * scale).clamp(-10,10)` — single-token, sin softmax real | **Crítico**: la atención no funciona correctamente |
| **KV Cache** | Cachea K y V entre tokens, atiende a toda la secuencia | Recalcula todo desde cero en cada token. Solo atiende al último token | **Crítico**: sin contexto histórico |
| **Posición** | RoPE aplicado correctamente con posición acumulativa | RoPE aplicado pero la posición es solo `input_ids.len()-1` | **Menor**: funciona para tokens secuenciales |
| **Embedding** | Suma o promedio de embeddings del contexto | Suma de embeddings normalizada por sqrt(n) | **Menor**: aproximación razonable |
| **Sampling** | softmax con temperatura, top-k, top-p | `sample()` con argmax/softmax simple | **Menor**: para temp=0 debería dar lo mismo |

**Conclusión de calidad:** BML no produce inferencia correcta porque la atención
está simplificada (sin softmax, sin KV cache). Esto es un problema fundamental
que debe resolverse antes de comparar calidad con llama.cpp.

---

## 2. Optimizaciones que llama.cpp tiene y BML NO

### Optimizaciones de cómputo (impacto en velocidad)

| # | Optimización | llama.cpp | BML actual | Impacto estimado |
|---|-------------|-----------|------------|------------------|
| 1 | **SIMD matmul** (AVX2/AVX-512) | ✅ 4-8 floats por instrucción | ❌ Scalar f64 | **4-8x** |
| 2 | **f32 en lugar de f64** | ✅ f32 (2x throughput, 2x menos RAM) | ❌ f64 en todo | **2x** |
| 3 | **KV Cache** | ✅ Cachea K,V, solo computa nuevo token | ❌ Recalcula todo | **N_tokens x** |
| 4 | **Quantized compute** | ✅ Computa en Q4_0 directo (XNOR+popcount) | ❌ Dequantiza a f32/f64 | **4-8x** |
| 5 | **Flash attention** | ✅ Fused, sin materializar matriz | ❌ Naive dot product | **2x** |
| 6 | **Thread parallelism intra-matmul** | ✅ Paraleliza columnas del matmul | ❌ Threads independientes | **2-4x** |
| 7 | **Weight packing** | ✅ Cache-friendly layout | ❌ Row-major sin packing | **1.5x** |
| 8 | **Operator fusion** | ✅ RMSNorm+matmul fused | ❌ Cada op separada | **1.3x** |
| 9 | **mmap de pesos** | ✅ No carga todo en RAM | ❌ Carga todo en Vec<f32> | **memory** |
| 10 | **Persistent thread pool** | ✅ Reutiliza threads | ❌ Crea threads por request | **latency** |

### Impacto combinado estimado

Si multiplicamos los speedups individuales (asumiendo independencia):

```
SIMD (4x) × f32 (2x) × KV-cache (N_tokens) × quantized (4x) × flash-attn (2x)
= 4 × 2 × N × 4 × 2 = 64N ×
```

Para N=128 tokens: `64 × 128 = 8192x` (teórico máximo).
En práctica: ~100-500x (Amdahl's law, no todo el tiempo es matmul).

**BML está a 0.155x de llama.cpp (6.5x de gap).** Con SIMD+f32+KV-cache
se cerraría la mayor parte del gap.

---

## 3. Las 5 optimizaciones de mayor impacto (priorizadas)

### #1: KV Cache (impacto: enorme)

**Problema:** BML recalcula toda la forward pass para cada token. llama.cpp
cachea K y V de tokens anteriores y solo computa el nuevo token.

```
Sin KV cache (BML actual):
  Token 1: forward(prompt) → K1, V1, Q1
  Token 2: forward(prompt + token1) → K1, V1, Q1, K2, V2, Q2 (recomputa K1,V1,Q1)
  Token 3: forward(prompt + token1 + token2) → recomputa todo
  ...
  Costo: O(n² × n_embd) por token

Con KV cache (llama.cpp):
  Token 1: compute K1, V1, Q1 → cachear K1, V1
  Token 2: compute K2, V2, Q2 → atender a [K1,V1,K2,V2]
  Token 3: compute K3, V3, Q3 → atender a [K1,V1,K2,V2,K3,V3]
  ...
  Costo: O(n × n_embd) por token
```

**Implementación:** agregar `kv_cache: Vec<f64>` a `InferenceCompiler` que
acumula K y V de cada capa entre tokens.

### #2: f32 en lugar de f64 (impacto: 2x)

**Problema:** BML usa `f64` en todo (hidden state, matmul, attention). f32
es 2x más rápido (SIMD puede procesar 8 f32 vs 4 f64 por instrucción AVX2).

**Implementación:** cambiar `Vec<f64>` a `Vec<f32>` en InferenceCompiler.
Es un cambio mecánico pero extenso.

### #3: SIMD para matmul (impacto: 4-8x)

**Problema:** El matmul de BML es scalar: `dot += x[i] * w[i]` un elemento
a la vez. Con AVX2: procesar 4 f64 (o 8 f32) por iteración.

**Implementación:** usar `std::simd` o intrínsecas AVX2 para el inner loop
del matmul. Alternativa: usar `ndarray` con BLAS.

### #4: Attention con softmax real + KV cache (impacto: calidad + velocidad)

**Problema actual:** la atención usa `tanh(dot * scale).clamp(-10,10)`
en lugar de `softmax(Q·K^T / sqrt(d))`. Esto no es atención correcta.

**Implementación:**
```rust
// Con KV cache:
for h in 0..n_heads {
    for prev_pos in 0..=current_pos {
        let dot = q[h] · k_cache[h][prev_pos] * scale;
        scores[prev_pos] = dot;
    }
    let attn = softmax(&scores);  // exp + normalize
    output[h] = Σ attn[i] * v_cache[h][i];
}
```

### #5: mmap de pesos (impacto: memoria + startup)

**Problema:** BML carga 3.9GB de pesos en RAM al arrancar. llama.cpp usa
mmap (page cache del kernel, lazy loading).

**Implementación:** usar `memmap2` (ya es dep del parser) para mapear el
pool de pesos en lugar de `Vec<f32>`.

---

## 4. ¿Podemos ejecutar con N nodos aunque tengamos 4 cores?

**SÍ.** La arquitectura distribuida no depende del número de cores.
Podemos:

### Opción A: Múltiples workers en la misma máquina

```sh
# 4 workers en la misma máquina, cada uno en un puerto distinto
bml-worker --port 9990 &
bml-worker --port 9991 &
bml-worker --port 9992 &
bml-worker --port 9993 &

# Coordinador distribuye a los 4 workers
bml-cli distribute -m model.bmlgraph/ --nodes localhost:9990,localhost:9991,localhost:9992,localhost:9993 -p "Hello" -n 16
```

**Limitación:** con 4 vCPUs, los 4 workers compiten por los mismos cores.
No hay ganancia de throughput, pero sí de **latencia** (pipeline paralelo).

### Opción B: Múltiples máquinas reales

Si tienes N máquinas, cada una con sus propios cores:

```sh
# Máquina 1: worker 0
bml-worker --port 9999 --fragment fragment_0.bmlgraph

# Máquina 2: worker 1
bml-worker --port 9999 --fragment fragment_1.bmlgraph

# Máquina 3 (coordinador)
bml-cli distribute -m model.bmlgraph/ --nodes maquina1:9999,maquina2:9999 -p "Hello" -n 16
```

**Ganancia real:** cada máquina tiene sus propios cores y RAM, sin
contención. El throughput escala linealmente con el número de máquinas
(limitado por la etapa serial del pipeline).

### Opción C: Workers en Docker/contenedores

Se pueden lanzar workers en contenedores con cgroup limits para simular
N máquinas en una sola física:

```sh
docker run --cpus=1 bml-worker --port 9990 &
docker run --cpus=1 bml-worker --port 9991 &
```

### Beneficio real de N nodos (con 4 cores totales)

| Configuración | Throughput | Latencia |
|---|---|---|
| 1 proceso, 4 threads | Alto (sin overhead TCP) | Alta (serial) |
| 4 workers, 1 core c/u | Igual (mismo cómputo total) | Baja (pipeline paralelo) |
| 4 máquinas reales | **4x mayor** | Baja (pipeline paralelo) |

**Conclusión:** con 4 cores en 1 máquina, N nodos no da más throughput.
Pero con N máquinas, sí escala linealmente.

---

## 5. Roadmap priorizado para cerrar la brecha

| Prioridad | Optimización | Impacto estimado | Esfuerzo |
|-----------|-------------|------------------|----------|
| 1 | **KV Cache** | O(n²) → O(n) por token | Medio (~200 líneas) |
| 2 | **Softmax real en attention** | Corrige calidad de output | Bajo (~50 líneas) |
| 3 | **f32 en lugar de f64** | 2x velocidad, 2x menos RAM | Medio (~100 líneas) |
| 4 | **SIMD matmul (AVX2)** | 4-8x en matmul | Alto (~300 líneas) |
| 5 | **mmap de pesos** | Startup instantáneo, menos RAM | Bajo (~50 líneas) |
| 6 | **Thread pool intra-matmul** | 2-4x en matmul | Medio (~200 líneas) |

**Con #1+#2+#3:** BML produciría texto coherente Y sería ~4x más rápido
(0.155x → ~0.6x de llama.cpp).

**Con #1+#2+#3+#4:** ~16x más rápido (0.155x → ~2.5x — comparable o mejor
que llama.cpp en single-thread).
