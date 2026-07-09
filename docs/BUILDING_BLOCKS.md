# Building blocks del transformer: desglose, dónde se usan, y relación con BML

## Panorama general

El transformer tiene **6 building blocks** primitivos. Cada uno se implementa
hoy como **función directa en f64** (sin pasar por BML). La pregunta es: cuáles
de estos pueden transformarse a operaciones BML nativas.

---

## Building block 1: MATMUL

### Función matemática

```
y[j] = Σ_i x[i] * W[i][j]    para j = 0..n_out
```

### Dónde se usa

| Uso | Llama | Entrada | Salida | Pesos |
|-----|-------|---------|--------|-------|
| Q projection | `attn_q.weight` | hidden [n_embd] | q [n_embd] | [n_embd, n_embd] |
| K projection | `attn_k.weight` | hidden [n_embd] | k [n_kv*head_dim] | [n_embd, n_kv*head_dim] |
| V projection | `attn_v.weight` | hidden [n_embd] | v [n_kv*head_dim] | [n_embd, n_kv*head_dim] |
| O projection | `attn_output.weight` | attn_out [n_heads*head_dim] | o [n_embd] | [n_heads*head_dim, n_embd] |
| FFN gate | `ffn_gate.weight` | hidden [n_embd] | gate [n_ffn] | [n_embd, n_ffn] |
| FFN up | `ffn_up.weight` | hidden [n_embd] | up [n_ffn] | [n_embd, n_ffn] |
| FFN down | `ffn_down.weight` | swiglu_out [n_ffn] | out [n_embd] | [n_ffn, n_embd] |
| lm_head | `token_embd.weight` | hidden [n_embd] | logits [vocab] | [n_embd, vocab] |
| Embedding | `token_embd.weight` | token_id | emb [n_embd] | [n_embd, vocab] |

### Implementación actual

`crates/compiler/src/gguf_compiler.rs:986` — `matmul_f64()`:
```rust
fn matmul_f64(&self, x: &[f64], weight_name: &str) -> Vec<f64> {
    for j in 0..n_out {
        let mut dot = 0.0;
        for i in 0..n_in {
            dot += x[i] * W[i][j];  // ← FMA (fused multiply-add)
        }
        y[j] = dot;
    }
}
```

### Descomposición BML

El inner loop es: `dot += x[i] * W[i][j]`

Esto se descompone como:
```
*  →  mul(x[i], W[i][j])     = exp2(log2(x[i]) + log2(W[i][j]))
+= →  add(dot, product)       = dot - (-product)
```

Y cada operación se reduce a BML:
```
mul(a, b)  = exp2(add(log2(a), log2(b)))
            = bml(add(bml(1, bml(bml(1, a), 1)), bml(b, 1)), 1)

add(a, b)  = sub(a, neg(b))
            = bml(log2(a), exp2(neg(b)))

log2(a)    = bml(1, bml(bml(1, a), 1))

exp2(a)    = bml(a, 1)

neg(b)     = bml(log2(0), exp2(b))
```

### Tu idea: func(v1, v2) como elemento de operación

El matmul se puede reinterpretar como:
```
y[j] = fold(dot_product, x, W[:, j])
     = Σ_i dot_element(x[i], W[i][j])
```

Donde `dot_element(x_i, w_ij) = x_i * w_ij` es una función de dos escalares.

Si definimos el building block como:
```rust
fn bml_dot(x: f64, w: f64) -> f64 {
    // mul(x, w) en BML
    bml(1.0, bml(bml(1.0, x), 1.0))  // log2(x)
    // ... pero esto requiere x > 0 y w > 0
}
```

**Problema:** BML requiere operandos positivos para `log2`. Los pesos del
transformer pueden ser negativos. La identidad `mul(a,b) = exp2(log2(a)+log2(b))`
solo funciona para `a > 0, b > 0`.

**Solución posible:** separar signo de magnitud:
```
x * w = sign(x) * sign(w) * (|x| * |w|)
       = sign(x*w) * exp2(log2(|x|) + log2(|w|))
```

El signo es 1 bit (que ya tenemos con `One`/`Zero`), y la magnitud usa BML.
El matmul se convierte en:
```
y[j] = Σ_i [sign_i * mul_bml(|x[i]|, |W[i][j]|)]
     = Σ_i [sign_i * exp2(log2(|x[i]|) + log2(|W[i][j]|))]
```

Cada `mul_bml` es un árbol de ~20 nodos BML. Un matmul de n_in×n_out genera
n_in×n_out árboles + un árbol de suma (reducción).

---

## Building block 2: RMSNORM

### Función matemática

```
rms = sqrt(mean(x²) + eps)
y[i] = x[i] / rms * w[i]
```

### Dónde se usa

| Uso | Línea |
|-----|-------|
| Pre-attention norm | `forward_layer:894` |
| Pre-MLP norm | `forward_layer:944` |
| Final norm | `forward:856` |

### Implementación actual

`gguf_compiler.rs:968` — `rmsnorm_inplace()`:
```rust
let mean_sq = sum(x²) / n;
let rms = sqrt(mean_sq + eps);
for i: y[i] = x[i] / rms * w[i];
```

### Descomposición BML

```
x²     = mul(x, x)              → exp2(log2(|x|) + log2(|x|)) = exp2(2*log2(|x|))
mean   = sum(x²) / n            → fold(add, x²) * recip(n)
sqrt   = pow(x, 0.5)            → exp2(0.5 * log2(x))
1/rms  = recip(sqrt(...))       → exp2(-0.5 * log2(mean_sq + eps))
y[i]   = x[i] * (1/rms) * w[i] → mul(x[i], mul(1/rms, w[i]))
```

Todo se reduce a `exp2`, `log2`, `add`, `mul` → BML puro.

---

## Building block 3: RoPE (Rotary Positional Embedding)

### Función matemática

```
x_even' = x_even * cos(θ) - x_odd * sin(θ)
x_odd'  = x_even * sin(θ) + x_odd * cos(θ)

donde θ = pos / base^(2i/head_dim)
```

### Dónde se usa

| Uso | Línea |
|-----|-------|
| Q rotation | `forward_layer:904` |
| K rotation | `forward_layer:905` |

### Implementación actual

`gguf_compiler.rs:1029` — `apply_rope_inplace()`:
```rust
let angle = pos * freq;
let c = angle.cos();
let s = angle.sin();
x_even' = x_even * c - x_odd * s;
x_odd'  = x_even * s + x_odd * c;
```

### Descomposición BML

- `cos(θ)` y `sin(θ)` se **precomputan en compile-time** con `eml::cos/sin`
  y se almacenan como `Const` en el pool. El runtime no los calcula.
- La rotación es:
  ```
  x_even' = mul(x_even, cos) - mul(x_odd, sin)   → sub(mul, mul)
  x_odd'  = mul(x_even, sin) + mul(x_odd, cos)   → add(mul, mul)
  ```
- `mul` y `sub`/`add` ya se redujeron a BML arriba.

**RoPE es 100% BML-transformable** — los cos/sin son constantes y la rotación
es solo mul/add/sub.

---

## Building block 4: ATTENTION (scaled dot-product)

### Función matemática

```
score = (Q · K^T) / sqrt(head_dim)     ← dot product + scale
attn  = softmax(score)                  ← exp + normalize
out   = attn · V                        ← weighted sum
```

### Dónde se usa

`forward_layer:907-934`

### Implementación actual

```rust
dot = Σ_d q[d] * k[d];          // dot product
attn = tanh(dot * scale);        // soft-clip (no softmax real)
out[h,d] = attn * v[d];          // weighted sum
```

### Descomposición BML

- **Dot product**: `Σ q[d]*k[d]` = matmul (building block 1)
- **Scale**: `dot * (1/sqrt(d))` = mul (building block 1)
- **Softmax**: `exp(score_i) / Σ exp(score_j)` = exp + recip + add
  - `exp(x) = bml(x, 1)` ← BML directo
  - `recip(x) = exp2(-log2(x))` ← BML
- **Weighted sum**: `attn * v` = mul (building block 1)

**Attention es 100% BML-transformable.**

---

## Building block 5: SwiGLU

### Función matemática

```
y = gate * sigmoid(1.7 * gate) * up
```

### Dónde se usa

`forward_layer:951-957`

### Implementación actual

```rust
swiglu[i] = g / (1.0 + (-1.7 * g).exp()) * u;   // sigmoid(1.7g) * u
```

### Descomposición BML

```
1.7 * g     = mul(1.7, g)                    → BML (mul)
-1.7 * g    = neg(mul(1.7, g))               → BML (neg)
exp(-1.7g)  = exp2(-1.7g * log2(e))          → BML (exp2 + mul + log2)
1 + exp(...) = add(1, exp(...))              → BML (add)
g / (1+...) = div(g, 1+exp(...))             → BML (div = mul * recip)
* u         = mul(..., u)                    → BML (mul)
```

**SwiGLU es 100% BML-transformable.** Todos los componentes (mul, add, exp,
div, neg) ya se derivaron del paper.

---

## Building block 6: RESIDUAL + EMBEDDING

### Función matemática

```
residual:  y = x + f(x)                    ← add
embedding: y[i] = W[token_id][i]           ← lookup (index)
```

### Dónde se usa

| Uso | Línea |
|-----|-------|
| Attention residual | `forward_layer:940` |
| MLP residual | `forward_layer:962` |
| Embedding | `forward:866` |

### Descomposición BML

- **Residual**: `x + f(x)` = `add(x, f(x))` → BML puro
- **Embedding**: lookup por índice → `VarIndexed { base }` en RPN (ya existe)

---

## Resumen: los 3 operadores primitivos

Todos los building blocks se reducen a **3 operadores primitivos**:

| Primitivo | Operación | Identidad BML | Profundidad del árbol |
|-----------|-----------|---------------|----------------------|
| **mul(a, b)** | a × b | `exp2(log2(a) + log2(b))` | ~20 nodos |
| **add(a, b)** | a + b | `sub(a, neg(b))` | ~15 nodos |
| **exp(x)** | e^x | `bml(x * log2(e), 1)` | ~10 nodos |

Y estos 3 se reducen a **BML puro**:
- `exp2(x) = bml(x, 1)` (2 nodos)
- `log2(x) = bml(1, bml(bml(1, x), 1))` (5 nodos)
- `neg(x) = bml(log2(0), exp2(x))` (10 nodos)
- `sub(a, b) = bml(log2(a), exp2(b))` (8 nodos)
- `add(a, b) = sub(a, neg(b))` (18 nodos)
- `mul(a, b) = exp2(add(log2(a), log2(b)))` (28 nodos)
- `div(a, b) = mul(a, recip(b))` (40 nodos)

---

## El matmul como building block distribuido

### Tu observación: func(v1, v2)

El matmul es la operación más pesada (90% del cómputo del transformer).
Se puede reinterpretar como:

```
matmul(x, W) = map(dot_column, W_columns)
  donde dot_column(w_col) = fold(add_dot, zip(x, w_col))
  donde add_dot(acc, (x_i, w_i)) = acc + x_i * w_i
```

Si definimos:
```rust
fn bml_fma(acc: f64, x: f64, w: f64) -> f64 {
    // acc + x * w  en BML
    bml_add(acc, bml_mul(x, w))
}
```

Entonces cada columna del matmul es un `fold` sobre `bml_fma`. El fold se
implementa con `RpnOp::Loop` en el bytecode RPN.

### Distribución natural

El matmul se parte por columnas:
```
Nodo 0: columnas 0..n_out/4     → dot(x, W[:, 0..n_out/4])
Nodo 1: columnas n_out/4..n_out/2
Nodo 2: columnas n_out/2..3n_out/4
Nodo 3: columnas 3n_out/4..n_out
```

Cada nodo recibe el vector `x` (n_embd valores) y su slice de columnas de W.
El resultado es independiente entre nodos → **paralelo puro**.

El fold de cada columna es un sub-fragmento L1i que cabe en 30 KB.

---

## Tabla de uso actual

| Building block | Implementación actual | Usa BML? | Dónde está |
|----------------|----------------------|----------|------------|
| matmul | `matmul_f64()` en f64 | ❌ No — FMA directo | `gguf_compiler.rs:986` |
| RMSNorm | `rmsnorm_inplace()` en f64 | ❌ No — sqrt/mul directo | `gguf_compiler.rs:968` |
| RoPE | `apply_rope_inplace()` en f64 | ❌ No — cos/sin/mul directo | `gguf_compiler.rs:1029` |
| Attention | dot product en f64 | ❌ No — mul/tanh directo | `gguf_compiler.rs:907` |
| SwiGLU | sigmoid en f64 | ❌ No — exp/mul directo | `gguf_compiler.rs:951` |
| Residual | add en f64 | ❌ No — + directo | `gguf_compiler.rs:940` |
| Embedding | lookup por índice | ✅ `VarIndexed` | `gguf_compiler.rs:814` |
| compile-time cos/sin | `eml::cos/sin` | ✅ Const pool | `eml.rs:77` |
| compile-time RoPE consts | `eml::rope_constants` | ✅ Const pool | `eml.rs:136` |
| Sampling | `sampler::sample` en f64 | ❌ No — argmax/softmax | `sampler.rs` |

### Lo que usa BML hoy

1. **`HashConsRegistry`** construye el DAG BML del transformer (en `compile_gguf`)
   — pero `compile_gguf` es muy lento para 1B pesos y no se usa en producción.
2. **`dispatch_ops`** ejecuta bytecode RPN (que incluye BML ops) — pero
   el bytecode actual es trivial (un nodo `Var(0)` placeholder).
3. **`RealEncoder`** / `BmlWeightPool` codifican pesos como Const — pero
   solo hace deduplicación de valores, no construye árboles BML.

### Lo que NO usa BML hoy

Todo el forward pass del transformer (`InferenceCompiler::forward_layer`)
usa operaciones f64 nativas (FMA, sqrt, cos, sin, exp). El operador BML
no participa en la inferencia real.

---

## Propuesta: matmul como func(v1, v2) BML

### Definición del building block

```rust
/// FMA en BML: acc + x * w
/// Se descompone como:
///   x * w = exp2(log2(|x|) + log2(|w|)) * sign(x) * sign(w)
///   acc + product = sub(acc, neg(product))
fn bml_fma(acc: NodeId, x: NodeId, w: NodeId, sign_x: bool, sign_w: bool) -> NodeId {
    let abs_x = ...; // magnitud de x
    let abs_w = ...; // magnitud de w
    let log_x = log2(abs_x);
    let log_w = log2(abs_w);
    let sum = add(log_x, log_w);
    let product = exp2(sum);
    let signed_product = if sign_x ^ sign_w { neg(product) } else { product };
    add(acc, signed_product)
}
```

### Matmul como fold de bml_fma

```
y[j] = fold(bml_fma, 0, x, W[:, j])
     = bml_fma(bml_fma(bml_fma(0, x[0], W[0][j]), x[1], W[1][j]), x[2], W[2][j]), ...)
```

Esto genera un árbol BML de profundidad O(n_in) por columna.

Con `RpnOp::Loop`, el fold se expresa sin expandir:
```
Loop(n_in, body_len) {
  body:
    VarIndexed(base=x)    // cargar x[i]
    VarIndexed(base=w_j) // cargar W[i][j]
    BML                   // mul (log2 + add + exp2)
    FAdd                  // acc + product
}
```

El cuerpo del loop es ~10 ops RPN = ~10 bytes. Un matmul de n_in=2048
necesita un loop de 2048 iteraciones con 10 ops cada una = 10 bytes de
bytecode + 2048 iteraciones en runtime.

### Distribución

- Cada columna `j` del matmul es independiente → paralelo
- Cada columna es un sub-fragmento: Loop(n_in, 10) = 13 bytes de bytecode
- Un matmul de 2048×2048 = 2048 sub-fragmentos de 13 bytes cada uno
- Total bytecode: 2048 × 13 = 26 KB → cabe en L1i
- Los pesos W se sirven desde L2/L3 (no se copian al sub-fragmento)
