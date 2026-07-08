# Análisis: fragmentación L1i, pipeline paralelo/serial, y tipo nativo BML

## 1. Fragmentación fina (< L1i = 32 KB) con cambio de fragmento en runtime

### Problema actual

La fragmentación distribuida actual parte por **capa del transformer** (168 MB
por fragmento). Eso es correcto para distribución cross-machine (cada nodo
tiene RAM para 168 MB), pero **no aprovecha L1i** dentro de cada nodo.

El hot loop (`dispatch_ops`) cabe en L1i (429 líneas asm ≈ 8 KB), pero los
**pesos** de una capa (168 MB) saturan L2/L3 y generan cache misses en L1d.

### Propuesta: fragmentación multinivel

```
Nivel 0: distribución cross-machine
  ├── Fragmento de capa (168 MB) → un nodo

Nivel 1: dentro de cada nodo, fragmentación L1i
  ├── Sub_fragmento A (30 KB de bytecode + pesos referenciados)
  ├── Sub_fragmento B (30 KB)
  ├── ...
  └── Sub_fragmento N (30 KB)
```

Cada nodo recibe su fragmento de capa (168 MB) y lo **parte en sub-fragmentos
de <32 KB**. El hot loop ejecuta un sub-fragmento, termina, carga el siguiente
en L1i, ejecuta, etc. Esto es el **"cambio de hot loop"** que ya está
documentado en el diseño original:

> "Cuando hay más fragmentos que cores, un core ejecuta varios fragmentos
> secuencialmente. El cambio es cargar el siguiente fragmento (< 32KB) en L1i."

### Cuántos sub-fragmentos por capa?

TinyLlama 1.1B, una capa:
- attn_q: 2048×2048 × 4 bytes = 16 MB
- attn_k: 16 MB (o 8 MB con GQA)
- attn_v: 16 MB
- attn_output: 16 MB
- ffn_gate: 2048×5632 × 4 = 46 MB
- ffn_up: 46 MB
- ffn_down: 46 MB
- norms: ~16 KB
- **Total: ~168 MB**

Sub-fragmentos de 30 KB (bytecode + refs a pesos):
- El bytecode RPN de un sub-fragmento es ~30 KB
- Los pesos se referencian por offset (no se copian al sub-fragmento)
- **~5,600 sub-fragmentos por capa** (168 MB / 30 KB)
- **~123,000 sub-fragmentos totales** (22 capas × 5,600)

Esto es **muchísimo**, pero no es problema:
- Cada sub-fragmento es independiente
- El hot loop los ejecuta secuencialmente sin allocs
- El cambio de fragmento es O(1) (cambiar el slice de ops)

### Beneficio

- **L1i hit rate ~100%**: cada sub-fragmento cabe entero en L1i
- **L1d prefetch**: los pesos referenciados por un sub-fragmento se precargan
  mientras se ejecuta el anterior (software pipelining)
- **Sin thrashing**: el hot loop nunca se contamina con código de otro sub-fragmento

---

## 2. Etapas paralelas y seriales en la distribución

### Sí — hay etapas paralelas y seriales

El transformer tiene una estructura **secuencial por capas** pero
**paralela dentro de cada capa**:

```
Capa 0:  Q | K | V  →  Attention  →  Output  |  Gate | Up  →  SwiGLU  →  Down
         (paralelo)   (serial: depende de Q,K,V)   (paralelo)  (serial: depende de gate,up)
```

### Pipeline del transformer

```
Etapa PARALELA (dentro de una capa):
  ├── Q projection (matmul hidden × Wq)
  ├── K projection (matmul hidden × Wk)     ← independientes
  ├── V projection (matmul hidden × Wv)
  └── FFN gate (matmul hidden × Wgate)
      FFN up (matmul hidden × Wup)

Etapa SERIAL (depende del paralelo anterior):
  ├── RoPE(Q), RoPE(K)                      ← depende de Q, K
  ├── Attention = softmax(Q·K^T / √d) · V   ← depende de Q, K, V
  ├── Output = Attention · Wo               ← depende de Attention
  ├── Residual = hidden + Output
  ├── RMSNorm(Residual)
  └── SwiGLU = gate · σ(1.7·gate) · up      ← depende de gate, up
  ├── Down = SwiGLU · Wdown
  └── Residual = Residual + Down
```

### Distribución entre máquinas

```
           ┌─────────────────────────────────────────────┐
           │  Etapa serial: capa 0 → capa 1 → ... → capa N │
           └─────────────────────────────────────────────┘
                           ↑ cada paso depende del anterior

  Dentro de cada capa (paralelizable entre nodos):
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Nodo A    │  │ Nodo B    │  │ Nodo C    │  │ Nodo D    │
  │ Q proj    │  │ K proj    │  │ V proj    │  │ FFN gate  │
  │ (matmul)  │  │ (matmul)  │  │ (matmul)  │  │ (matmul)  │
  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘
        │             │             │              │
        └─────────────┴─────────────┘              │
                    │ barrier                      │ (puede seguir en paralelo
                    ▼                              │  con la atención)
            ┌──────────────┐                       │
            │  Attention   │  ← serial: necesita Q,K,V
            │  (softmax +  │
            │   V mul)      │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │  Output proj │  ← serial: necesita Attention
            │  + Residual  │
            └──────┬───────┘
                   │
                   │  ┌──────────┐  ┌──────────┐
                   │  │ Nodo A    │  │ Nodo B    │  ← paralelo otra vez
                   │  │ SwiGLU    │  │ Down proj │
                   │  └─────┬────┘  └─────┬────┘
                   │        │             │
                   │        └──────┬──────┘
                   │               │ barrier
                   │               ▼
                   │        ┌──────────────┐
                   └───────►│  Residual    │  ← serial: junta todo
                            └──────┬───────┘
                                   │
                                   ▼
                          Siguiente capa (serial)
```

### Modelo de programación: DAG de fragmentos con dependencias

Cada sub-fragmento declara sus dependencias:
```
fragment {
  id: 42,
  depends_on: [40, 41],   ← barrera: esperar a que 40 y 41 terminen
  produces: "hidden_after_layer_3",
  layer: 3,
  op: "attention",
  weights: [...]
}
```

El coordinador construye un **DAG de fragmentos** y lo schedulea:
1. **Wave 1** (paralela): Q, K, V, gate, up de capa 0 → 5 nodos en paralelo
2. **Wave 2** (serial): Attention (depende de Q,K,V) → 1 nodo
3. **Wave 3** (serial): Output + Residual → 1 nodo
4. **Wave 4** (paralela): gate, up de FFN → 2 nodos
5. **Wave 5** (serial): SwiGLU + Down + Residual → 1 nodo
6. **Wave 6**: siguiente capa...

**Esto es Tensor Parallelism con pipeline scheduling.**

### Speedup teórico

Para TinyLlama con 22 capas y 4 nodos:
- Dentro de cada capa: ~5 ops paralelas (Q,K,V,gate,up) + 3 seriales (attn,output,down)
- Si 5 nodos hacen Q,K,V,gate,up en paralelo: wave 1 = 1 matmul time
- Wave 2 (attention) = 1 matmul time (serial)
- Total por capa: ~4 serial steps en lugar de ~8 sequential
- **Speedup ≈ 2x** por capa (limitado por la parte serial)

Con más nodos (N >> 5): el bottleneck es la etapa serial (Amdahl's law).

---

## 3. Tipo nativo BML: de f64 a un tipo de 1 bit

### Observación clave

La gramática del AST BML es: `S → 1 | Var(id) | Const(id) | BML(S, S)`

Los **nodos del AST** son:
- `One` — constante 1
- `Zero` — constante 0
- `Var(id)` — variable (input)
- `Const(id)` — constante (peso)
- `Bml(left, right)` — operador

**Los nodos no son f64.** Los nodos son **tres valores**: `0`, `1`, o `variable`.
Un nodo es esencialmente **1 bit** (0 o 1) + un tag de tipo (3 variants).

### Pero los valores en la pila SÍ son f64

El hot loop opera sobre una pila de `f64`:
```rust
stack: Vec<f64>  // cada valor es 8 bytes
```

Cada `Bml` saca 2 × f64 de la pila y empuja 1 × f64. Los valores intermedios
son números reales (resultado de `2^x - log2(y)`).

**No podemos reducir la pila a 1 bit** — los valores intermedios son reales.

### Lo que SÍ se puede reducir: el bytecode y los nodos del DAG

#### Bytecode RPN actual

```rust
pub enum RpnOp {
    One,           // tag = 0 (1 byte)
    Zero,          // tag = 6 (1 byte)
    Bml,           // tag = 1 (1 byte)
    Dup,           // tag = 2 (1 byte)
    Var(u32),      // tag = 4 + 4 bytes = 5 bytes
    Const(u32),    // tag = 5 + 4 bytes = 5 bytes
    Loop { count: u32, body_len: u32 }, // 9 bytes
    ...
}
```

Los tags `One`, `Zero`, `Bml`, `Dup`, `FAdd`, `FMul`, `Drop`, `Swap` son
**1 byte** pero solo usan 4 variants (0, 1, 2, 6). Podrían ser **2 bits**.

#### Propuesta: BML nativo como tag de 2 bits

```
Bit 0-1: tipo de nodo
  00 = One (constante 1)
  01 = Zero (constante 0)
  10 = Var (variable, sigue index)
  11 = Bml (operador, no necesita datos extra)
```

Un programa RPN de N ops:
- **Actual**: N × 1 byte (tags) + N × 4 bytes (args de Var/Const) ≈ N × 5 bytes
- **Con 2 bits**: N × 2 bits + N × 4 bytes (args) ≈ N × 4.25 bytes
- **Reducción**: ~15% en bytecode

No es drámatico porque los args de Var/Const dominan.

#### Lo que SÍ es drámatico: los pesos del modelo

Los pesos son `f32` (4 bytes cada uno). TinyLlama 1.1B tiene 1,034,518,528
pesos = **3.9 GB en f32**.

Si los pesos fueran **1 bit** (binarizados): 1,034,518,528 bits = **128 MB**.
Reducción: **30x**.

### Binarización de pesos (Binary Neural Networks)

La literatura de BinaryNet y XNOR-Net muestra que los pesos de un transformer
pueden binarizarse a ±1 (1 bit) con pérdida de accuracy modesta:

- **BinaryNet**: pesos ±1, activaciones binarizadas con sign function
- **XNOR-Net**: pesos ±1, matmul reemplazado por XNOR + popcount
- **BML-Net** (propuesto): pesos como `0` o `1` (BML nativo), matmul reemplazado
  por AND + popcount

Con pesos binarizados:
```
matmul(x, W_bin) = popcount(x AND W_bin)
```

Esto mapea perfectamente al **operador BML**:
- `bml(1, 1) = 2` → peso = 1
- `bml(1, 0) = 2 - (-inf) = inf` → peso = 0 (o log2(0) = -inf)
- `bml(0, 1) = 1 - 0 = 1` → bias

### Propuesta: tipo BML nativo

```rust
/// Tipo nativo BML: 1 bit.
/// 0 o 1. No es f64, no es f32. Es un bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BmlValue {
    Zero,
    One,
}

/// Los pesos del modelo en formato BML binarizado.
/// Cada peso es 1 bit.
pub struct BmlWeights {
    bits: Vec<u64>,  // 64 pesos por u64
}

impl BmlWeights {
    pub fn get(&self, idx: usize) -> BmlValue {
        let word = idx / 64;
        let bit = idx % 64;
        if (self.bits[word] >> bit) & 1 == 1 {
            BmlValue::One
        } else {
            BmlValue::Zero
        }
    }

    pub fn set(&mut self, idx: usize, v: BmlValue) {
        let word = idx / 64;
        let bit = idx % 64;
        match v {
            BmlValue::One => self.bits[word] |= 1 << bit,
            BmlValue::Zero => self.bits[word] &= !(1 << bit),
        }
    }
}
```

### Impacto en distribución

| Métrica | f32 (actual) | BML 1-bit (propuesto) | Reducción |
|---------|-------------|---------------------|-----------|
| Peso por peso | 4 bytes | 1/8 byte (1 bit) | 32x |
| Fragmento de capa | 168 MB | 5.25 MB | 32x |
| Modelo completo | 3.9 GB | 128 MB | 30x |
| Transfer via TCP | ~168 MB/nodo | ~5 MB/nodo | 32x |
| Tiempo de carga | ~5s/nodo | ~0.15s/nodo | 32x |

Con pesos binarizados, **un fragmento de capa entera cabe en L2** (5 MB < 8 MB L2).
Y el modelo completo (128 MB) cabe en **L3** (16 MB) de un solo nodo.

### Trade-offs

1. **Loss of accuracy**: la binarización de pesos reduce la precisión del modelo.
   Literature reports ~5-10% accuracy drop para clasificación, más para generation.

2. **Matmul cambia**: `popcount(x AND W)` en lugar de `dot(x, W)`. Mucho más rápido
   pero requiere activaciones también binarizadas (o cuantizadas a few-bit).

3. **BML operator no se usa directamente**: el operador `bml(x,y) = 2^x - log2(y)`
   opera sobre f64. La binarización reemplaza el matmul, no el operador BML.
   BML sigue siendo el lenguaje de expresión del programa, pero la ejecución
   usa XNOR+popcount para los pesos binarizados.

4. **Training**: requiere quantization-aware training (QAT) para que el modelo
   aprenda pesos binarizados. No se puede post-hoc en un modelo f32 existente
   sin retuning.

### Conclusión

La binarización a 1 bit es **el cambio de mayor impacto** para distribución:
- 30x menos datos por nodo
- Carga instantánea
- Matmul 10x más rápido (XNOR+popcount vs FMA)
- Pero requiere re-entrenar el modelo con QAT

Como BML no entrena modelos (solo hace inferencia), necesitaríamos:
1. Un GGUF ya binarizado (que no existe hoy para TinyLlama)
2. O un paso de binarización post-training con fine-tuning
3. O cambiar el formato de pesos de f32 a 1-bit en el compilador

---

## Resumen de las dos ideas

| Idea | Impacto | Complejidad |
|------|---------|-------------|
| Fragmentación fina L1i (30 KB) | L1i hit 100%, sin thrashing | Media — partir fragmento de capa en sub-fragmentos |
| Pipeline paralelo/serial con DAG | ~2x speedup con 4+ nodos | Alta — scheduler de waves con barreras |
| Tipo BML 1-bit para pesos | 30x menos datos, 10x matmul | Alta — requiere QAT + cambio de formato |
