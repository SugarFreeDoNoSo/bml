# K/V cache: hash consing + SIMD AVX2 + mmap distribuido

## 1. Hash consing para K/V cache

### Observación clave

En un transformer, K y V se computan una vez por token y se cachean.
Pero muchos K/V son **estructuralmente similares**:
- En tokens tempranos, los K/V convergen (attention sinks)
- En capas profundas, muchos heads atienden a los mismos tokens
- Tokens repetidos en el prompt producen K/V idénticos

### Hash consing de K/V

En lugar de almacenar K[layer][pos][head] como un vector independiente,
usamos un `HashConsRegistry` de vectores:

```rust
pub struct KVCache {
    // Registry de vectores K/V únicos
    // Key: hash del contenido del vector (head_dim valores)
    // Value: índice compartido
    k_registry: VectorHashCons,  // deduplica vectores K
    v_registry: VectorHashCons,  // deduplica vectores V
    
    // Tabla de referencias: [layer][pos][kv_head] → registry_id
    k_refs: Vec<Vec<Vec<u32>>>,  // [n_layers][max_seq][n_kv_heads]
    v_refs: Vec<Vec<Vec<u32>>>,
    
    current_pos: u32,
}

pub struct VectorHashCons {
    // Hash del contenido del vector → índice compartido
    table: HashMap<u64, u32>,  // hash(content) → id
    // Storage de vectores únicos (en formato i8 para SIMD)
    storage: Vec<I8Vector>,    // solo vectores únicos
    n_unique: usize,
    n_total: usize,
}
```

### Ejemplo de deduplicación

```
Token 0: K[layer=0][pos=0][head=0] = [0.1, -0.3, 0.5, ...]  → registry_id=0 (nuevo)
Token 1: K[layer=0][pos=1][head=0] = [0.2, -0.1, 0.8, ...]  → registry_id=1 (nuevo)
Token 2: K[layer=0][pos=2][head=0] = [0.1, -0.3, 0.5, ...]  → registry_id=0 (DUPLICADO)
```

El token 2 produce el mismo K que el token 0 → se reutiliza `registry_id=0`.
No se almacena una segunda copia.

### Deduplicación esperada

| Escenario | K/V únicos vs total | Ratio |
|-----------|---------------------|-------|
| Tokens todos distintos | 100% únicos | 1x (sin beneficio) |
| Prompt con tokens repetidos | ~50% únicos | 2x |
| Attention sinks (tokens 0-3 dominantes) | ~10% únicos | 10x |
| Tokens en capas profundas (sparse attention) | ~20% únicos | 5x |

**Estimación conservadora:** 3-5x deduplicación en prompts reales.

---

## 2. Tipo óptimo: i8 para K/V + SIMD AVX2

### Por qué i8 es mejor que f16

| Tipo | Bytes/elemento | AVX2 throughput | Precision |
|------|---------------|-----------------|-----------|
| f64 | 8 | 4 valores/instrucción | Máxima |
| f32 | 4 | 8 valores/instrucción | Buena |
| f16 | 2 | 16 valores/instr (F16C) | Decente |
| **i8** | **1** | **32 valores/instr (VNNI)** | ~1% pérdida |
| i4 | 0.5 | 64 valores/instr (experimental) | ~5% pérdida |

**i8 con AVX2:**
- 32 valores por instrucción (vs 4 de f64 = 8x más rápido)
- Dot product: `_mm256_dpbusd_epi32` (AVX-VNNI) o manual con `_mm256_maddubs_epi16` + `_mm256_madd_epi16`
- Memoria: 8x menos que f64, 4x menos que f32, 2x menos que f16

### Quantización de K/V a i8

```rust
pub struct I8Vector {
    // Cuantización por bloque (como Q8_0 del GGUF):
    // [f16 scale][n × i8 values]
    scale: f16,
    values: Vec<i8>,  // head_dim valores
}

impl I8Vector {
    /// Cuantiza un vector f32 a i8 con escala
    pub fn quantize(data: &[f32]) -> Self {
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = (max_abs / 127.0).max(1e-8);
        let values: Vec<i8> = data.iter()
            .map(|&v| (v / scale).clamp(-127.0, 127.0) as i8)
            .collect();
        I8Vector { scale: half_scale, values }
    }

    /// Dequantiza a f32
    pub fn dequantize(&self) -> Vec<f32> {
        self.values.iter().map(|&q| q as f32 * scale).collect()
    }
}
```

### Dot product Q·K con SIMD AVX2 (i8 × i8 → f32)

```rust
#[cfg(target_arch = "x86_64")]
fn dot_i8_avx2(q: &[i8], k: &[i8], q_scale: f32, k_scale: f32) -> f32 {
    use std::arch::x86_64::*;
    
    let n = q.len();
    let mut sum = _mm256_setzero_si256();
    
    // Procesar 32 i8 por iteración
    for i in (0..n - 31).step_by(32) {
        let qv = _mm256_loadu_si256(q[i..].as_ptr() as *const __m256i);
        let kv = _mm256_loadu_si256(k[i..].as_ptr() as *const __m256i);
        
        // _mm256_maddubs_epi16: multiplica pares (i8 × i8) → i16
        let prod = _mm256_maddubs_epi16(qv, kv);
        // _mm256_madd_epi16: suma pares de i16 → i32
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod, _mm256_set1_epi16(1)));
    }
    
    // Horizontal sum
    let mut result = 0i32;
    let lanes = [0, 1, 2, 3, 4, 5, 6, 7];
    for lane in lanes {
        result += _mm256_extract_epi32(sum, lane);
    }
    
    // Resto
    for i in (n / 32 * 32)..n {
        result += (q[i] as i32) * (k[i] as i32);
    }
    
    result as f32 * q_scale * k_scale
}
```

**Throughput:** 32 dot products por instrucción AVX2 (vs 4 en f64).
**Speedup teórico:** 8x sobre f64 escalar.

---

## 3. Hash consing + i8 = sinergia

### El insight clave

Hash consing e i8 son **sinérgicos**:

1. **Hash consing** deduplica vectores idénticos → menos storage
2. **i8** comprime cada vector → 8x menos bytes por vector
3. **SIMD AVX2** procesa i8 en lotes de 32 → 8x más rápido
4. Combinado: menos memoria + más velocidad

```
Sin hash consing, f64:  180K vectores × 64 × 8B = 92 MB, dot product escalar
Con hash consing, i8:   ~36K vectores únicos × 64 × 1B = 2.3 MB, dot product SIMD 32x
= 40x menos memoria + 8x más rápido por dot product
```

### Estructura combinada

```rust
pub struct KVCachedRegistry {
    // Registry de vectores K/V únicos en formato i8
    k_table: HashMap<u64, u32>,           // hash(i8_vector) → id
    k_storage: Vec<I8Vector>,            // solo vectores K únicos
    
    v_table: HashMap<u64, u32>,
    v_storage: Vec<I8Vector>,            // solo vectores V únicos
    
    // Referencias: [layer][pos][kv_head] → registry_id
    k_refs: Vec<Vec<Vec<u32>>>,
    v_refs: Vec<Vec<Vec<u32>>>,
    
    // Stats
    n_total_k: usize,
    n_unique_k: usize,
    n_total_v: usize,
    n_unique_v: usize,
}

impl KVCachedRegistry {
    /// Cacha K[layer][pos][head] — deduplica si ya existe
    pub fn store_k(&mut self, layer: u32, pos: u32, kv_head: u32, data: &[f32]) {
        let i8vec = I8Vector::quantize(data);
        let hash = i8vec.content_hash();
        
        let id = if let Some(&id) = self.k_table.get(&hash) {
            // DUPLICADO: reutilizar
            id
        } else {
            // NUEVO: almacenar
            let id = self.k_storage.len() as u32;
            self.k_storage.push(i8vec);
            self.k_table.insert(hash, id);
            id
        };
        
        self.k_refs[layer as usize][pos as usize][kv_head as usize] = id;
        self.n_total_k += 1;
    }
    
    /// Carga K[layer][pos][head] como f32
    pub fn load_k(&self, layer: u32, pos: u32, kv_head: u32) -> Vec<f32> {
        let id = self.k_refs[layer as usize][pos as usize][kv_head as usize];
        self.k_storage[id as usize].dequantize()
    }
    
    /// Dot product Q·K con SIMD AVX2 (sin dequantizar)
    pub fn dot_qk(&self, q: &[f32], layer: u32, pos: u32, kv_head: u32) -> f32 {
        let id = self.k_refs[layer as usize][pos as usize][kv_head as usize];
        let k_i8 = &self.k_storage[id as usize];
        let q_i8 = I8Vector::quantize(q);
        dot_i8_avx2(&q_i8.values, &k_i8.values, q_i8.scale, k_i8.scale)
    }
}
```

### Stats en runtime

```rust
pub fn compression_stats(&self) -> (usize, usize, f64) {
    let mem_unique = self.k_storage.len() * 65;  // 64 i8 + 1 f16 scale
    let mem_full = self.n_total_k * 65;
    let ratio = self.n_total_k as f64 / self.k_storage.len() as f64;
    (self.k_storage.len(), self.n_total_k, ratio)  // (unique, total, dedup_ratio)
}
```

---

## 4. mmap distribuido entre cores

### Concepto

El weight pool (3.9GB) se mapea una vez con mmap. Cada core accede a
diferentes regiones del archivo → **diferentes páginas en L1/L2 de cada core**.

```
              mmap (3.9GB, read-only, compartido)
                    │
    ┌───────────────┼───────────────┐
    │               │               │
  Core 0          Core 1          Core 2
  Lee pesos       Lee pesos       Lee pesos
  capas 0-7       capas 8-14      capas 15-21
    │               │               │
  L1 L2            L1 L2            L1 L2
  (páginas         (páginas         (páginas
   distintas)       distintas)       distintas)
```

### Implementación con madvise

```rust
use std::os::unix::io::AsRawFd;

pub struct MmapWeights {
    mmap: memmap2::Mmap,
    // Cada thread hace madvise sobre su rango
}

impl MmapWeights {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { mmap })
    }
    
    /// Prefetch de un rango para el core actual.
    /// Llama a madvise(WILLNEED) para que el kernel precargue las páginas.
    pub fn prefetch_range(&self, offset: usize, len: usize) {
        let ptr = self.mmap.as_ptr() as *const c_void;
        unsafe {
            let addr = (ptr as usize + offset) as *const c_void;
            libc::madvise(addr, len, libc::MADV_WILLNEED);
        }
    }
    
    /// Marca un rango como secuencial (prefetcher hardware lo detecta).
    pub fn sequential(&self, offset: usize, len: usize) {
        let ptr = self.mmap.as_ptr() as *const c_void;
        unsafe {
            let addr = (ptr as usize + offset) as *const c_void;
            libc::madvise(addr, len, libc::MADV_SEQUENTIAL);
        }
    }
    
    /// Lee un peso f32 por offset.
    #[inline]
    pub fn get_f32(&self, offset: usize) -> f32 {
        let bytes = &self.mmap[offset * 4..offset * 4 + 4];
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
    
    /// Lee un slice de pesos como &[f32] (zero-copy desde el mmap).
    pub fn as_f32_slice(&self, offset: usize, len: usize) -> &[f32] {
        let byte_start = offset * 4;
        let byte_end = byte_start + len * 4;
        let bytes = &self.mmap[byte_start..byte_end];
        // Reinterpretar bytes como f32 (little-endian x86)
        unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len)
        }
    }
}
```

### Distribución natural entre cores

No hay cache line bouncing porque el mmap es **read-only**:
- Core 0 lee capas 0-7 → páginas en L2 de core 0
- Core 1 lee capas 8-14 → páginas en L2 de core 1
- Sin coherencia entre cores (páginas read-only no se invalidan)

El OS distribuye automáticamente:
- `mmap(MAP_POPULATE)` pre-llena la page table
- Cada core trae sus páginas a L1/L2 on-demand
- `madvise(MADV_SEQUENTIAL)` para matmul (patrón secuencial)
- `madvise(MADV_WILLNEED)` para prefetch del siguiente fragmento

### Sinergia con sub-fragmentación L1i

```
Sub-fragmento L1i (30 KB bytecode)
  ↓ ejecuta Loop(n_in, body)
  ↓ cada iteración accede a pesos via VarIndexed
  ↓ VarIndexed lee del mmap → page fault la primera vez
  ↓ segunda vez: L1d hit (misma página)
  ↓ prefetcher precarga próxima página
  = pipeline perfecto: L1i para bytecode, L1d para datos
```

---

## 5. Resumen de la arquitectura final

```
┌─────────────────────────────────────────────┐
│              mmap de pesos (3.9GB)           │
│         read-only, compartido entre cores    │
│  Core 0: capas 0-7    Core 1: capas 8-14    │
│  (páginas L2 propias, sin coherencia)       │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│           KV Cache hash-consed (i8)          │
│  k_storage: ~36K vectores únicos × 65B       │
│           = 2.3 MB (vs 92 MB en f64)         │
│  Deduplicación: 3-5x                         │
│  Dot product: SIMD AVX2 (32 i8/instrucción)  │
│  = 8x más rápido que f64 escalar            │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│         Hot loop RPN (L1i, 30 KB)           │
│  dispatch_ops único (429 líneas asm)         │
│  Sub-fragmentos con cambio de hot loop      │
│  Scheduler de waves (paralelo + barreras)   │
└─────────────────────────────────────────────┘
```

### Comparación de tipos

| Componente | Actual | Propuesto | Reducción |
|-----------|--------|-----------|-----------|
| Pesos | f32 Vec (3.9GB en RAM) | f32 mmap (0 GB en RAM, page cache) | ∞ |
| K cache | — (no existe) | i8 hash-consed (~2.3MB) | 40x vs f64 |
| V cache | — (no existe) | i8 hash-consed (~2.3MB) | 40x vs f64 |
| Hidden state | f64 | f32 | 2x |
| Dot product | escalar f64 | SIMD i8 AVX2 (32x/instr) | 8x |
| Hash consing | No aplicado a K/V | Sí, deduplica K/V | 3-5x |

### Impacto acumulado

```
SIMD i8 (8x) × hash-cons dedup (3x) × f32 (2x) × mmap (0 startup)
× KV-cache O(n) × softmax real
= 8 × 3 × 2 = 48x más rápido (teórico)
+ calidad corregida (softmax + KV cache)
```

vs llama.cpp (que usa f16 K/V, sin hash consing):
```
BML i8 hash-consed + SIMD vs llama.cpp f16 + BLAS
= i8 (2x menos memoria que f16) + hash-cons (3-5x dedup)
+ AVX2 32-wide dot product
≈ 2-3x ventaja teórica sobre llama.cpp
```
