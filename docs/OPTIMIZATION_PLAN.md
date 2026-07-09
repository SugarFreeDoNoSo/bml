# Plan: optimizaciones para igualar a llama.cpp

## Objetivo

1. **Calidad:** BML debe producir los mismos tokens que llama.cpp para el mismo
   prompt y temperatura.
2. **Velocidad:** cerrar la brecha de 6.5x (0.155x → ~1x de llama.cpp).
3. **Memoria:** usar f16/i8 para valores cacheados (K, V) como hace llama.cpp.

## Tipos de datos por componente

| Componente | Actual (f64) | Objetivo | Razón |
|-----------|-------------|----------|-------|
| Hidden state | f64 (8B) | f32 (4B) | 2x menos RAM, 2x SIMD throughput |
| Pesos (weight pool) | f32 (4B) | f32 (4B) | ya está bien (Q4_0 dequantizado) |
| **K cache** | — | **f16 (2B)** | 4x menos que f64, como llama.cpp |
| **V cache** | — | **f16 (2B)** | 4x menos que f64, como llama.cpp |
| Attention scores | — | f32 (4B) | precisión suficiente para softmax |
| Logits | f64 (8B) | f32 (4B) | 2x menos, sampling no necesita f64 |
| Const pool (BML) | f64 (8B) | f32 (4B) | 2x menos |
| Embedding lookup | f64 (8B) | f32 (4B) | 2x menos |

### Alternativa i8 para K/V cache

llama.cpp usa f16 para K/V. Pero se puede ir más lejos con **i8 (int8)**:
- f16: 2 bytes × (n_layers × seq_len × n_kv_heads × head_dim)
- i8: 1 byte × mismo → 2x menos que f16
- Trade-off: cuantización de K/V con escala por bloque (como Q8_0)
- Precisión: ~1% de pérdida en quality, 2x menos RAM

Para TinyLlama con contexto 2048:
- K cache: 22 capas × 2048 × 4 KV heads × 64 = 11.5M elementos
- f16: 23 MB, f32: 46 MB, f64: 92 MB, i8: 11.5 MB
- Total K+V: f16=46MB, i8=23MB (despreciable vs 3.9GB de pesos)

**Decisión:** usar f16 para K/V (como llama.cpp). i8 es una optimización
futura opcional.

---

## Fases de implementación

### Fase 1: KV Cache + Softmax real (CALIDAD)

**Objetivo:** que BML produzca los mismos tokens que llama.cpp.

**Problema actual:** la atención usa `tanh(clamp)` sin softmax y sin cache.
Cada token recalcula todo el contexto desde cero.

#### 1.1 Estructura KV Cache

```rust
pub struct KVCache {
    // K y V por capa: [n_layers][max_seq_len][n_kv_heads * head_dim]
    k: Vec<f16>,  // f16 como llama.cpp
    v: Vec<f16>,
    // Metadata
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    // Posición actual
    current_pos: u32,
}
```

**Tamaño para TinyLlama (ctx=2048):**
- K: 22 × 2048 × 4 × 64 × 2B = 23 MB
- V: igual = 23 MB
- Total: 46 MB

**Archivos:** `crates/compiler/src/gguf_compiler.rs` (nuevo struct KVCache)

#### 1.2 Forward pass con KV cache

```rust
fn forward_layer_cached(
    &self,
    hidden: &mut Vec<f32>,
    layer: u32,
    pos: u32,
    kv: &mut KVCache,
) {
    // RMSNorm
    self.rmsnorm_f32(hidden, ...);

    // Q, K, V projections (matmul f32)
    let q = self.matmul_f32(hidden, ...);  // [n_heads * head_dim]
    let k = self.matmul_f32(hidden, ...);  // [n_kv_heads * head_dim]
    let v = self.matmul_f32(hidden, ...);  // [n_kv_heads * head_dim]

    // Aplicar RoPE a Q y K
    self.apply_rope_f32(&mut q, pos, head_dim);
    self.apply_rope_f32(&mut k, pos, head_dim);

    // Cachear K y V en f16
    kv.store_k(layer, pos, &k);
    kv.store_v(layer, pos, &v);

    // Attention con softmax sobre todos los tokens cacheados
    let mut output = vec![0.0_f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv_h = h / q_heads_per_kv;

        // Scores: Q[h] · K[prev_pos] para todos los prev_pos <= pos
        let mut scores = vec![0.0_f32; pos + 1];
        for p in 0..=pos {
            let k_cached = kv.load_k(layer, p, kv_h);  // f16 → f32
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[h * head_dim + d] * k_cached[d];
            }
            scores[p] = dot * scale;
        }

        // Softmax real
        let attn = softmax(&scores);

        // Output: Σ attn[p] * V[p]
        for d in 0..head_dim {
            let mut sum = 0.0_f32;
            for p in 0..=pos {
                let v_cached = kv.load_v(layer, p, kv_h);  // f16 → f32
                sum += attn[p] * v_cached[d];
            }
            output[h * head_dim + d] = sum;
        }
    }

    // Output projection + residual
    let o_out = self.matmul_f32(&output, ...);
    for i: hidden[i] += o_out[i];

    // MLP (igual que antes pero en f32)
    ...
}
```

**Archivos a modificar:**
- `crates/compiler/src/gguf_compiler.rs` — nuevo `KVCache`, `forward_layer_cached`, `softmax_f32`
- `crates/cli/src/bml_inference.rs` — usar `forward_layer_cached` en `forward_bml`

**Líneas estimadas:** ~250

#### 1.3 Softmax real

```rust
fn softmax_f32(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
```

#### 1.4 Verificación de calidad

```sh
# Debe producir tokens coherentes (no basura)
bml-cli run -m /root/tinyllama.gguf -p "The capital of France is" -n 16 --temp 0.0
```

**Criterio de éxito:** output coherente (ej: " Paris, the capital...").

**Commit:** `fix(inference): KV cache f16 + softmax real — output coherente`

---

### Fase 2: f32 en todo el pipeline (VELOCIDAD 2x)

**Objetivo:** cambiar todo el cómputo de f64 a f32.

#### 2.1 Cambiar tipos

| Archivo | Cambio |
|---------|--------|
| `gguf_compiler.rs` | `hidden: Vec<f64>` → `Vec<f32>`, `matmul_f64` → `matmul_f32` |
| `bml_inference.rs` | `forward_bml` usa f32 |
| `rpn.rs` | `evaluate_with_ctx` opera en f32 (o deja f64 para BML puro) |
| `buffer.rs` | `ResultBuffer` almacena f32 |
| `hot_loop.rs` | pila de f32 (o mantener f64 para BML puro, convertir en boundary) |

**Decisión:** mantener el hot loop RPN en f64 (para preservar BML puro),
pero el InferenceCompiler (forward pass) en f32. La conversión f32↔f64
solo en los boundaries.

**Líneas estimadas:** ~150 (cambio mecánico)

**Commit:** `perf: f32 en todo el pipeline de inferencia (2x velocidad)`

---

### Fase 3: mmap de pesos (MEMORIA + STARTUP)

**Objetivo:** no cargar 3.9GB en RAM al arrancar.

#### 3.1 mmap del weight pool

```rust
pub struct InferenceCompiler {
    // weight_pool: Vec<f32> → reemplazar por mmap
    weight_mmap: memmap2::Mmap,
    weight_offsets: HashMap<String, u32>,
    ...
}

fn get_weight(&self, offset: usize) -> f32 {
    let bytes = &self.weight_mmap[offset*4..offset*4+4];
    f32::from_le_bytes(bytes.try_into().unwrap())
}
```

**Ventajas:**
- Startup instantáneo (no lee el archivo, solo mmap)
- Page cache del kernel (lazy loading, solo páginas accedidas)
- Múltiples procesos comparten las mismas páginas (copy-on-write)

**Archivos:** `crates/compiler/src/gguf_compiler.rs`

**Líneas estimadas:** ~80

**Commit:** `perf: mmap de pesos — startup instantáneo, menos RAM`

---

### Fase 4: SIMD matmul (VELOCIDAD 4-8x)

**Objetivo:** acelerar el inner loop del matmul con SIMD.

#### 4.1 AVX2 para dot product

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn dot_product_f32_avx2(x: &[f32], w: &[f32]) -> f32 {
    let n = x.len();
    let mut sum = _mm256_setzero_ps();
    for i in (0..n - 7).step_by(8) {
        let xv = _mm256_loadu_ps(&x[i]);
        let wv = _mm256_loadu_ps(&w[i]);
        sum = _mm256_fmadd_ps(xv, wv, sum);
    }
    // Horizontal sum + remainder
    ...
}
```

**Alternativa:** usar `ndarray` con BLAS (OpenBLAS) que ya tiene AVX2.

#### 4.2 Matmul con SIMD

```rust
fn matmul_f32_simd(&self, x: &[f32], w: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; n_out];
    for j in 0..n_out {
        y[j] = dot_product_f32_avx2(x, &w[j*n_in..(j+1)*n_in]);
    }
    y
}
```

**Archivos:** nuevo `crates/compiler/src/simd.rs`

**Líneas estimadas:** ~200

**Commit:** `perf: SIMD AVX2 para matmul (4-8x en inner loop)`

---

### Fase 5: Thread pool intra-matmul (VELOCIDAD 2-4x)

**Objetivo:** paralelizar el matmul entre cores.

#### 5.1 Rayon o threads raw

```rust
fn matmul_parallel(&self, x: &[f32], w: &[f32], n_in: usize, n_out: usize, n_threads: usize) -> Vec<f32> {
    let y: Vec<f32> = (0..n_out).into_par_iter().map(|j| {
        dot_product_f32_simd(x, &w[j*n_in..(j+1)*n_in])
    }).collect();
    y
}
```

**Alternativa sin rayon:** particionar columnas entre threads manuales.

**Archivos:** `crates/compiler/src/gguf_compiler.rs`

**Líneas estimadas:** ~100

**Commit:** `perf: thread pool intra-matmul (2-4x con 4 cores)`

---

### Fase 6: f16 para K/V cache (MEMORIA)

Ya implementado en Fase 1, pero optimizando:
- Usar `half` crate para f16 nativo
- O implementar f16↔f32 conversion inline

---

## Resumen del plan

| Fase | Qué | Impacto calidad | Impacto velocidad | Líneas | Commit |
|------|-----|-----------------|-------------------|--------|--------|
| 1 | KV cache f16 + softmax real | ✅ Fix | O(n²)→O(n) | ~250 | fix: KV cache |
| 2 | f32 en todo | — | 2x | ~150 | perf: f32 |
| 3 | mmap pesos | — | startup 0s | ~80 | perf: mmap |
| 4 | SIMD AVX2 | — | 4-8x matmul | ~200 | perf: SIMD |
| 5 | Thread pool | — | 2-4x matmul | ~100 | perf: threads |
| **Total** | | **Fix calidad** | **~50-100x** | **~780** | |

## Impacto acumulado estimado

| Estado | Ops/seg (1t) | Ratio vs llama.cpp (tg) | Calidad |
|--------|-------------|------------------------|---------|
| Actual | 594M | 0.155x (4 threads) | ❌ Basura |
| +Fase 1 | 594M | 0.155x + O(n²)→O(n) | ✅ Coherente |
| +Fase 2 | ~1,200M | ~0.3x (2x por f32) | ✅ |
| +Fase 3 | 1,200M | ~0.3x (mismo compute) | ✅ |
| +Fase 4 | ~4,800M | ~1.2x (4x por SIMD) | ✅ |
| +Fase 5 | ~9,600M | ~2.4x (2x por threads) | ✅ |

**Con Fases 1-5:** BML sería ~2.4x más rápido que llama.cpp en single-thread
con output de igual calidad.

## Orden de implementación

```
Fase 1 (calidad) → Fase 2 (f32) → Fase 3 (mmap) → Fase 4 (SIMD) → Fase 5 (threads)
```

Cada fase tiene su commit y push. Después de Fase 1, verificamos que el
output coincide con llama.cpp. Después de Fase 5, corremos benchmarks.
