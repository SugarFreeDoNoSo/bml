// KV Cache con hash consing + cuantización i8 + dot product SIMD.

use std::collections::HashMap;

// ===========================================================================
// I8Vector: vector cuantizado a int8 con escala
// ===========================================================================

/// Vector cuantizado a i8 con escala f32 (estilo Q8_0).
///
/// Formato: [scale: f32][values: n × i8]
/// Valor real = value * scale
///
/// AVX2 puede procesar 32 i8 por instrucción (vs 4 f64).
#[derive(Debug, Clone)]
pub struct I8Vector {
    pub scale: f32,
    pub values: Vec<i8>,
}

impl I8Vector {
    /// Cuantiza un slice f32 a i8 con escala.
    pub fn quantize(data: &[f32]) -> Self {
        let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = (max_abs / 127.0).max(1e-8);
        let values: Vec<i8> = data
            .iter()
            .map(|&v| (v / scale).clamp(-127.0, 127.0) as i8)
            .collect();
        I8Vector { scale, values }
    }

    /// Dequantiza a Vec<f32>.
    pub fn dequantize(&self) -> Vec<f32> {
        self.values.iter().map(|&q| q as f32 * self.scale).collect()
    }

    /// Hash del contenido para deduplicación.
    pub fn content_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.scale.to_bits().hash(&mut hasher);
        self.values.hash(&mut hasher);
        hasher.finish()
    }

    /// Tamaño en bytes.
    pub fn byte_size(&self) -> usize {
        4 + self.values.len()
    }
}

// ===========================================================================
// Dot product SIMD AVX2 (i8 × i8 → f32)
// ===========================================================================

/// Dot product de dos vectores i8 con escalas f32.
///
/// Usa AVX2 cuando está disponible: procesa 16 i8 por instrucción
/// (extiende a i16 primero para manejar signo correctamente).
/// Resultado = Σ(q[i] * k[i]) * q_scale * k_scale
#[cfg(target_arch = "x86_64")]
pub fn dot_i8_simd(q: &[i8], k: &[i8], q_scale: f32, k_scale: f32) -> f32 {
    use std::arch::x86_64::*;

    if !is_x86_feature_detected!("avx2") {
        return dot_i8_scalar(q, k, q_scale, k_scale);
    }

    let n = q.len().min(k.len());
    let mut result = 0i64;

    // Procesar 16 i8 por iteración (extender a i16, luego madd)
    let chunks = n / 16;
    for i in 0..chunks {
        let base = i * 16;
        unsafe {
            // Cargar 16 i8 y extender a 32 i16 con sign extension
            let qv = _mm256_cvtepi8_epi16(_mm_loadu_si128(q.as_ptr().add(base) as *const __m128i));
            let kv = _mm256_cvtepi8_epi16(_mm_loadu_si128(k.as_ptr().add(base) as *const __m128i));
            // _mm256_madd_epi16: multiplica pares i16×i16 → i32 y suma adyacentes
            let prod = _mm256_madd_epi16(qv, kv);
            // Sumar horizontalmente: extraer 8 lanes de i32
            let lane_sum = _mm256_extract_epi32(prod, 0) as i64
                + _mm256_extract_epi32(prod, 1) as i64
                + _mm256_extract_epi32(prod, 2) as i64
                + _mm256_extract_epi32(prod, 3) as i64
                + _mm256_extract_epi32(prod, 4) as i64
                + _mm256_extract_epi32(prod, 5) as i64
                + _mm256_extract_epi32(prod, 6) as i64
                + _mm256_extract_epi32(prod, 7) as i64;
            result += lane_sum;
        }
    }

    // Resto (n % 16 elementos)
    let remainder_start = chunks * 16;
    for i in remainder_start..n {
        result += (q[i] as i64) * (k[i] as i64);
    }

    result as f32 * q_scale * k_scale
}

/// Fallback escalar para dot product i8.
pub fn dot_i8_scalar(q: &[i8], k: &[i8], q_scale: f32, k_scale: f32) -> f32 {
    let mut sum = 0i32;
    for i in 0..q.len().min(k.len()) {
        sum += (q[i] as i32) * (k[i] as i32);
    }
    sum as f32 * q_scale * k_scale
}

// ===========================================================================
// HashConsedKV: KV cache con hash consing + i8
// ===========================================================================

/// KV cache con hash consing de vectores K/V cuantizados a i8.
///
/// - Vectores K/V idénticos se deduplican automáticamente
/// - Formato i8 + escala f32 (estilo Q8_0)
/// - Dot product Q·K con SIMD AVX2 (32 valores/instrucción)
///
/// Para TinyLlama (22 capas, ctx=2048, 4 KV heads, head_dim=64):
/// - Sin dedup: 22×2048×4×2 = 360K vectores × 65B = 23 MB
/// - Con dedup 3x: ~120K vectores × 65B = 7.8 MB
/// - vs f64 sin cache: imposible (recalcula todo)
pub struct HashConsedKV {
    /// Registry de vectores K únicos (i8).
    k_storage: Vec<I8Vector>,
    k_table: HashMap<u64, u32>, // hash → id

    /// Registry de vectores V únicos (i8).
    v_storage: Vec<I8Vector>,
    v_table: HashMap<u64, u32>,

    /// Referencias: [layer][pos][kv_head] → registry_id
    k_refs: Vec<Vec<Vec<u32>>>,
    v_refs: Vec<Vec<Vec<u32>>>,

    /// Metadata
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq_len: u32,

    /// Stats
    n_total_k: usize,
    n_unique_k: usize,
    n_total_v: usize,
    n_unique_v: usize,
}

impl HashConsedKV {
    pub fn new(n_layers: u32, n_kv_heads: u32, head_dim: u32, max_seq_len: u32) -> Self {
        let k_refs = vec![vec![vec![0u32; n_kv_heads as usize]; max_seq_len as usize]; n_layers as usize];
        let v_refs = k_refs.clone();
        Self {
            k_storage: Vec::new(),
            k_table: HashMap::new(),
            v_storage: Vec::new(),
            v_table: HashMap::new(),
            k_refs,
            v_refs,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            n_total_k: 0,
            n_unique_k: 0,
            n_total_v: 0,
            n_unique_v: 0,
        }
    }

    /// Cacha K[layer][pos][kv_head] — deduplica si ya existe.
    pub fn store_k(&mut self, layer: u32, pos: u32, kv_head: u32, data: &[f32]) {
        let i8vec = I8Vector::quantize(data);
        let hash = i8vec.content_hash();

        let id = if let Some(&id) = self.k_table.get(&hash) {
            id
        } else {
            let id = self.k_storage.len() as u32;
            self.k_storage.push(i8vec);
            self.k_table.insert(hash, id);
            self.n_unique_k += 1;
            id
        };

        self.k_refs[layer as usize][pos as usize][kv_head as usize] = id;
        self.n_total_k += 1;
    }

    /// Cacha V[layer][pos][kv_head] — deduplica si ya existe.
    pub fn store_v(&mut self, layer: u32, pos: u32, kv_head: u32, data: &[f32]) {
        let i8vec = I8Vector::quantize(data);
        let hash = i8vec.content_hash();

        let id = if let Some(&id) = self.v_table.get(&hash) {
            id
        } else {
            let id = self.v_storage.len() as u32;
            self.v_storage.push(i8vec);
            self.v_table.insert(hash, id);
            self.n_unique_v += 1;
            id
        };

        self.v_refs[layer as usize][pos as usize][kv_head as usize] = id;
        self.n_total_v += 1;
    }

    /// Dot product Q·K con SIMD AVX2 (sin dequantizar).
    ///
    /// Q se cuantiza a i8 y se hace dot product directo con el K cacheado.
    pub fn dot_qk(&self, q: &[f32], layer: u32, pos: u32, kv_head: u32) -> f32 {
        let id = self.k_refs[layer as usize][pos as usize][kv_head as usize];
        let k_i8 = &self.k_storage[id as usize];
        let q_i8 = I8Vector::quantize(q);

        #[cfg(target_arch = "x86_64")]
        {
            dot_i8_simd(&q_i8.values, &k_i8.values, q_i8.scale, k_i8.scale)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            dot_i8_scalar(&q_i8.values, &k_i8.values, q_i8.scale, k_i8.scale)
        }
    }

    /// Carga V[layer][pos][kv_head] como Vec<f32>.
    pub fn load_v(&self, layer: u32, pos: u32, kv_head: u32) -> Vec<f32> {
        let id = self.v_refs[layer as usize][pos as usize][kv_head as usize];
        self.v_storage[id as usize].dequantize()
    }

    /// Carga K[layer][pos][kv_head] como Vec<f32>.
    pub fn load_k(&self, layer: u32, pos: u32, kv_head: u32) -> Vec<f32> {
        let id = self.k_refs[layer as usize][pos as usize][kv_head as usize];
        self.k_storage[id as usize].dequantize()
    }

    /// Stats de deduplicación K.
    pub fn k_stats(&self) -> (usize, usize, f64) {
        let ratio = if self.n_unique_k > 0 {
            self.n_total_k as f64 / self.n_unique_k as f64
        } else {
            0.0
        };
        (self.n_unique_k, self.n_total_k, ratio)
    }

    /// Stats de deduplicación V.
    pub fn v_stats(&self) -> (usize, usize, f64) {
        let ratio = if self.n_unique_v > 0 {
            self.n_total_v as f64 / self.n_unique_v as f64
        } else {
            0.0
        };
        (self.n_unique_v, self.n_total_v, ratio)
    }

    /// Tamaño total del cache en bytes.
    pub fn byte_size(&self) -> usize {
        self.k_storage.iter().map(|v| v.byte_size()).sum::<usize>()
            + self.v_storage.iter().map(|v| v.byte_size()).sum::<usize>()
    }

    /// Posición actual (para saber cuántos tokens hay cacheados).
    pub fn current_pos(&self) -> u32 {
        // Asumiendo que se llena secuencialmente
        // Buscar la primera posición vacía
        for pos in 0..self.max_seq_len {
            if self.k_refs[0][pos as usize][0] == 0 && pos > 0 {
                return pos;
            }
        }
        self.max_seq_len
    }
}

// ===========================================================================
// Softmax f32
// ===========================================================================

/// Softmax con estabilidad numérica (max subtraction).
pub fn softmax_f32(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return vec![];
    }
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / scores.len() as f32; scores.len()]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i8vector_quantize_dequantize() {
        let data = vec![0.5, -0.3, 0.8, -0.1, 1.0, -1.0, 0.0, 0.3];
        let i8v = I8Vector::quantize(&data);
        let deq = i8v.dequantize();
        for (orig, val) in data.iter().zip(deq.iter()) {
            assert!((orig - val).abs() < 0.05, "quantize error: {orig} vs {val}");
        }
    }

    #[test]
    fn dot_i8_matches_f32() {
        let a = vec![0.5, -0.3, 0.8, -0.1, 1.0, -1.0, 0.5, 0.3,
                     0.2, -0.7, 0.4, 0.9, -0.5, 0.6, 0.1, -0.2,
                     0.3, 0.4, -0.6, 0.7, 0.8, -0.9, 0.2, 0.5,
                     -0.3, 0.1, 0.6, -0.4, 0.7, 0.3, -0.2, 0.5];
        let b = vec![0.4, 0.2, -0.5, 0.7, 0.3, -0.6, 0.1, 0.8,
                     -0.2, 0.5, 0.3, -0.7, 0.6, 0.4, -0.1, 0.2,
                     0.5, -0.3, 0.7, 0.2, -0.4, 0.6, 0.1, -0.5,
                     0.3, 0.8, -0.2, 0.4, 0.1, -0.6, 0.7, 0.3];

        let dot_f32: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let dot_i8 = dot_i8_simd(
            &I8Vector::quantize(&a).values,
            &I8Vector::quantize(&b).values,
            I8Vector::quantize(&a).scale,
            I8Vector::quantize(&b).scale,
        );

        assert!((dot_f32 - dot_i8).abs() < 0.5, "dot mismatch: f32={dot_f32}, i8={dot_i8}");
    }

    #[test]
    fn hash_consed_kv_dedup() {
        let mut kv = HashConsedKV::new(2, 4, 64, 128);

        // Simular K de 2 tokens en capa 0, head 0
        let k0 = vec![0.1; 64];
        let k1 = vec![0.2; 64];
        let k2 = vec![0.1; 64]; // idéntico a k0

        kv.store_k(0, 0, 0, &k0);
        kv.store_k(0, 1, 0, &k1);
        kv.store_k(0, 2, 0, &k2); // debe deduplicar

        let (unique, total, ratio) = kv.k_stats();
        assert_eq!(total, 3);
        assert_eq!(unique, 2, "k2 debe deduplicar con k0");
        assert!(ratio > 1.0, "ratio debe ser > 1");
    }

    #[test]
    fn softmax_sums_to_one() {
        let scores = vec![1.0, 2.0, 3.0, 0.5];
        let probs = softmax_f32(&scores);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn kv_cache_dot_qk() {
        let mut kv = HashConsedKV::new(1, 1, 4, 8);
        let k = vec![0.5, -0.3, 0.8, 0.1];
        let q = vec![0.4, 0.2, -0.5, 0.7];

        kv.store_k(0, 0, 0, &k);

        let dot = kv.dot_qk(&q, 0, 0, 0);
        let expected: f32 = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum();
        assert!((dot - expected).abs() < 0.5, "dot={dot}, expected={expected}");
    }
}
