//! Muestreo (sampling) de tokens desde logits.
//!
//! Soporta:
//! - Greedy (argmax, temp=0)
//! - Temperatura con softmax (temp>0)

/// Selecciona el token con mayor logit (greedy / temp=0).
pub fn argmax(logits: &[f64]) -> Option<(u32, f64)> {
    if logits.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_val = logits[0];
    for (i, &val) in logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    Some((best_idx as u32, best_val))
}

/// Softmax con temperatura.
///
/// `temp=0` o `temp < 1e-9` → greedy (delega a argmax).
pub fn sample(logits: &[f64], temperature: f64, _seed: u64) -> Option<u32> {
    if logits.is_empty() {
        return None;
    }
    if temperature < 1e-9 {
        return argmax(logits).map(|(id, _)| id);
    }
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits
        .iter()
        .map(|&l| ((l - max_logit) / temperature).exp())
        .sum();
    let pick: f64 = rand_val(); // rango [0, 1)
    let mut accum = 0.0;
    for (i, &logit) in logits.iter().enumerate() {
        accum += ((logit - max_logit) / temperature).exp() / sum;
        if pick <= accum {
            return Some(i as u32);
        }
    }
    Some((logits.len() - 1) as u32)
}

/// Genera un valor aleatorio en [0.0, 1.0) usando xorshift64.
fn rand_val() -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0);
    let mut state = STATE.load(Ordering::Relaxed);
    if state == 0 {
        state = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
    }
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    STATE.store(state, Ordering::Relaxed);
    state as f64 / u64::MAX as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_basic() {
        let logits = [0.1, 0.5, 0.3];
        let (id, val) = argmax(&logits).unwrap();
        assert_eq!(id, 1);
        assert!((val - 0.5).abs() < 1e-9);
    }

    #[test]
    fn argmax_empty() {
        assert!(argmax(&[]).is_none());
    }

    #[test]
    fn sample_temp_zero_is_greedy() {
        let logits = [0.1, 0.9, 0.3, 0.2];
        let id = sample(&logits, 0.0, 0).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn sample_temp_returns_valid_id() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        for _ in 0..20 {
            let id = sample(&logits, 1.0, 42).unwrap();
            assert!(id < 4);
        }
    }

    #[test]
    fn sample_empty_returns_none() {
        assert!(sample(&[], 1.0, 0).is_none());
    }
}