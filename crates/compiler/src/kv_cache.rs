//! KV Cache con buffer plano f32 (estilo llama.cpp).
//
// Almacenamiento histórico; no acredita equivalencia con llama.cpp:
// - Buffers planos contiguos por capa (k_l / v_l)
// - Sin hash consing ni deduplicación
// - Almacenamiento directo en f32 (equivalente al tipo de cómputo)
// - Ring buffer implícito vía módulo (pos % max_seq_len)
// - Acceso directo sin dequantizar

// ===========================================================================
// HashConsedKV: KV cache con buffer plano f32 (estilo llama.cpp)
// ===========================================================================

/// KV cache con buffer plano f32, estilo llama.cpp.
///
/// Cada capa tiene un buffer contiguo para K y otro para V:
///   buffer[pos * stride + kv_head * head_dim + d]
/// donde stride = n_kv_heads × head_dim.
///
/// Características:
/// - Almacenamiento directo f32 (sin cuantización i8)
/// - Sin hash consing (cada posición es independiente)
/// - Métodos de almacenamiento heredados: pos % max_seq_len; la inferencia causal rechaza desbordamientos antes de escribir
/// - Acceso O(1) directo a cualquier entrada
pub struct HashConsedKV {
    /// Per-layer flat K buffers: k_buffers[layer][pos * stride + kv_head * head_dim + d]
    k_buffers: Vec<Vec<f32>>,
    /// Per-layer flat V buffers
    v_buffers: Vec<Vec<f32>>,

    n_layers: u32,
    head_dim: u32,
    max_seq_len: u32,
    stride: usize,

    /// Posición actual (tokens procesados).
    current_pos: u32,

    /// Stats: total de entradas almacenadas por capa
    n_total_k: usize,
    n_total_v: usize,
}

/// Default KV payload ceiling for historical inference callers (512 MiB).
pub const DEFAULT_MAX_KV_BYTES: usize = 512 * 1024 * 1024;

impl HashConsedKV {
    /// Crea buffers planos F32 por capa; usar try_new para errores y presupuesto explícitos.
    ///
    /// Args:
    /// - n_layers: número de capas del modelo
    /// - n_kv_heads: número de KV heads (GQA: puede ser < n_heads)
    /// - head_dim: dimensión de cada head
    /// - max_seq_len: tamaño máximo de secuencia (context window)
    pub fn new(n_layers: u32, n_kv_heads: u32, head_dim: u32, max_seq_len: u32) -> Self {
        Self::try_new(n_layers, n_kv_heads, head_dim, max_seq_len, usize::MAX)
            .expect("invalid KV dimensions or allocation failure")
    }

    /// Allocate a cache after checked size arithmetic and an explicit KV payload byte budget.
    pub fn try_new(
        n_layers: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        max_bytes: usize,
    ) -> Result<Self, String> {
        if n_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err("zero KV cache dimension".into());
        }
        let stride = (n_kv_heads as usize)
            .checked_mul(head_dim as usize)
            .ok_or("KV size overflow")?;
        let buffer_size = stride
            .checked_mul(max_seq_len as usize)
            .ok_or("KV size overflow")?;
        let bytes = buffer_size
            .checked_mul(n_layers as usize)
            .and_then(|n| n.checked_mul(8))
            .ok_or("KV size overflow")?;
        if bytes > max_bytes {
            return Err("KV cache exceeds byte budget".into());
        }
        let mut k_buffers = Vec::new();
        let mut v_buffers = Vec::new();
        k_buffers
            .try_reserve_exact(n_layers as usize)
            .map_err(|_| "KV layer allocation failed")?;
        v_buffers
            .try_reserve_exact(n_layers as usize)
            .map_err(|_| "KV layer allocation failed")?;
        for _ in 0..n_layers {
            let allocate = || -> Result<Vec<f32>, String> {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(buffer_size)
                    .map_err(|_| "KV buffer allocation failed")?;
                values.resize(buffer_size, 0.0);
                Ok(values)
            };
            k_buffers.push(allocate()?);
            v_buffers.push(allocate()?);
        }
        Ok(Self {
            k_buffers,
            v_buffers,
            n_layers,
            head_dim,
            max_seq_len,
            stride,
            current_pos: 0,
            n_total_k: 0,
            n_total_v: 0,
        })
    }

    // -----------------------------------------------------------------------
    // Store: escribe K/V en el buffer (ring buffer vía pos % max_seq_len)
    // -----------------------------------------------------------------------

    /// Almacena K[layer][pos][kv_head] en el buffer plano.
    ///
    /// Usa ring buffer: pos_idx = pos % max_seq_len.
    /// Los llamadores causales deben comprobar capacidad antes de escribir.
    pub fn store_k(&mut self, layer: u32, pos: u32, kv_head: u32, data: &[f32]) {
        let pos_idx = (pos % self.max_seq_len) as usize;
        let start = pos_idx * self.stride + kv_head as usize * self.head_dim as usize;
        let len = self.head_dim as usize;
        let dst = &mut self.k_buffers[layer as usize][start..start + len];
        dst.copy_from_slice(&data[..len.min(data.len())]);
        self.n_total_k += 1;
    }

    /// Almacena V[layer][pos][kv_head] en el buffer plano.
    ///
    /// Usa ring buffer: pos_idx = pos % max_seq_len.
    pub fn store_v(&mut self, layer: u32, pos: u32, kv_head: u32, data: &[f32]) {
        let pos_idx = (pos % self.max_seq_len) as usize;
        let start = pos_idx * self.stride + kv_head as usize * self.head_dim as usize;
        let len = self.head_dim as usize;
        let dst = &mut self.v_buffers[layer as usize][start..start + len];
        dst.copy_from_slice(&data[..len.min(data.len())]);
        self.n_total_v += 1;
    }

    // -----------------------------------------------------------------------
    // Load: lee K/V del buffer (acceso directo O(1), sin dequantizar)
    // -----------------------------------------------------------------------

    /// Carga K[layer][pos][kv_head] como Vec<f32> desde el buffer plano.
    ///
    /// Acceso directo, sin dequantizar (a diferencia de la versión i8 anterior).
    /// No implica paridad numérica con una implementación externa.
    pub fn load_k(&self, layer: u32, pos: u32, kv_head: u32) -> Vec<f32> {
        let pos_idx = (pos % self.max_seq_len) as usize;
        let start = pos_idx * self.stride + kv_head as usize * self.head_dim as usize;
        let len = self.head_dim as usize;
        self.k_buffers[layer as usize][start..start + len].to_vec()
    }

    /// Carga V[layer][pos][kv_head] como Vec<f32> desde el buffer plano.
    pub fn load_v(&self, layer: u32, pos: u32, kv_head: u32) -> Vec<f32> {
        let pos_idx = (pos % self.max_seq_len) as usize;
        let start = pos_idx * self.stride + kv_head as usize * self.head_dim as usize;
        let len = self.head_dim as usize;
        self.v_buffers[layer as usize][start..start + len].to_vec()
    }

    // -----------------------------------------------------------------------
    // Dot product Q·K: directo f32 (sin cuantización, como llama.cpp)
    // -----------------------------------------------------------------------

    /// Dot product Q[layer][pos][kv_head] · K[layer][pos][kv_head].
    ///
    /// Producto punto directo en f32 sobre los valores almacenados.
    /// La acumulación usa aritmética F32 nativa.
    pub fn dot_qk(&self, q: &[f32], layer: u32, pos: u32, kv_head: u32) -> f32 {
        let pos_idx = (pos % self.max_seq_len) as usize;
        let start = pos_idx * self.stride + kv_head as usize * self.head_dim as usize;
        let len = self.head_dim as usize;
        let k = &self.k_buffers[layer as usize][start..start + len];
        q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum()
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Stats de K: (total, total, 1.0) — sin deduplicación.
    pub fn k_stats(&self) -> (usize, usize, f64) {
        (self.n_total_k, self.n_total_k, 1.0)
    }

    /// Stats de V: (total, total, 1.0) — sin deduplicación.
    pub fn v_stats(&self) -> (usize, usize, f64) {
        (self.n_total_v, self.n_total_v, 1.0)
    }

    /// Tamaño total del cache en bytes (buffers f32 planos).
    pub fn byte_size(&self) -> usize {
        let per_layer = self.stride * self.max_seq_len as usize * std::mem::size_of::<f32>();
        per_layer * self.n_layers as usize * 2 // K + V
    }

    // -----------------------------------------------------------------------
    // Position tracking (como llama_kv_cache.used / head)
    // -----------------------------------------------------------------------

    /// Posición actual (tokens procesados).
    pub fn current_pos(&self) -> u32 {
        self.current_pos
    }

    /// Physical position capacity; causal callers must reject overflow, not wrap.
    pub fn capacity(&self) -> u32 {
        self.max_seq_len
    }

    /// Check layer, KV-head and head-dimension compatibility before causal writes.
    pub fn matches_layout(&self, layers: u32, heads: u32, dimension: u32) -> bool {
        self.n_layers == layers
            && self.head_dim == dimension
            && (heads as usize).checked_mul(dimension as usize) == Some(self.stride)
    }

    /// Avanza la posición actual en 1 (después de procesar un token).
    pub fn advance(&mut self) {
        self.current_pos += 1;
    }

    /// Avanza la posición en N tokens (después de procesar un batch).
    pub fn advance_by(&mut self, n: u32) {
        self.current_pos += n;
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
    fn store_load_roundtrip() {
        let mut kv = HashConsedKV::new(1, 4, 64, 128);
        let k = vec![0.5f32; 64];
        let v = vec![-0.3f32; 64];

        kv.store_k(0, 0, 0, &k);
        kv.store_v(0, 0, 0, &v);

        let loaded_k = kv.load_k(0, 0, 0);
        let loaded_v = kv.load_v(0, 0, 0);

        for i in 0..64 {
            assert!((loaded_k[i] - k[i]).abs() < 1e-6, "K mismatch at {i}");
            assert!((loaded_v[i] - v[i]).abs() < 1e-6, "V mismatch at {i}");
        }
    }

    #[test]
    fn ring_buffer_wrapping() {
        let mut kv = HashConsedKV::new(1, 1, 4, 8);
        let k0 = vec![0.1, 0.2, 0.3, 0.4];
        let k8 = vec![0.9, 0.8, 0.7, 0.6]; // pos 8 → índice 0 (ring buffer)

        kv.store_k(0, 0, 0, &k0);
        kv.store_k(0, 8, 0, &k8);

        // pos 0 fue sobrescrito por pos 8 en ring buffer
        let loaded = kv.load_k(0, 0, 0);
        assert!((loaded[0] - 0.9).abs() < 1e-6, "Ring buffer wrap failed");
        assert!((loaded[3] - 0.6).abs() < 1e-6, "Ring buffer wrap failed");
    }

    #[test]
    fn dot_qk_exact() {
        let mut kv = HashConsedKV::new(1, 1, 4, 8);
        let k = vec![0.5, -0.3, 0.8, 0.1];
        let q = vec![0.4, 0.2, -0.5, 0.7];

        kv.store_k(0, 0, 0, &k);

        let dot = kv.dot_qk(&q, 0, 0, 0);
        let expected: f32 = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum();
        // Con f32 directo, el error debe ser mínimo (solo redondeo)
        assert!(
            (dot - expected).abs() < 1e-5,
            "dot={dot}, expected={expected}"
        );
    }

    #[test]
    fn multi_layer_multi_head() {
        let mut kv = HashConsedKV::new(2, 4, 64, 128);
        let k = vec![0.1f32; 64];
        let v = vec![0.2f32; 64];

        // Almacenar en capa 1, head 2, pos 5
        kv.store_k(1, 5, 2, &k);
        kv.store_v(1, 5, 2, &v);

        let loaded_k = kv.load_k(1, 5, 2);
        let loaded_v = kv.load_v(1, 5, 2);

        assert_eq!(loaded_k.len(), 64);
        assert_eq!(loaded_v.len(), 64);
        assert!((loaded_k[0] - 0.1).abs() < 1e-6);
        assert!((loaded_v[0] - 0.2).abs() < 1e-6);

        // Verificar que otras posiciones no han sido afectadas
        let empty = kv.load_k(1, 0, 0);
        assert!(
            (empty[0] - 0.0).abs() < 1e-6,
            "Otras posiciones deben estar en 0"
        );
    }

    #[test]
    fn stats_no_dedup() {
        let mut kv = HashConsedKV::new(2, 4, 64, 128);
        let k0 = vec![0.1f32; 64];
        let k1 = vec![0.1f32; 64]; // mismo valor que k0

        kv.store_k(0, 0, 0, &k0);
        kv.store_k(0, 1, 0, &k1);

        let (unique, total, ratio) = kv.k_stats();
        assert_eq!(total, 2);
        // Sin dedup: unique == total (cada store cuenta)
        assert_eq!(unique, total);
        assert!((ratio - 1.0).abs() < 1e-6);
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
    fn softmax_empty() {
        let probs = softmax_f32(&[]);
        assert!(probs.is_empty());
    }
}
