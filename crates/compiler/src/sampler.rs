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
/// Solo `temperature=0` usa greedy. Toda temperatura positiva y finita usa
/// softmax, incluidas las temperaturas subnormales. Los logits deben ser
/// finitos; entradas vacías, logits no finitos y temperaturas negativas o no
/// finitas devuelven `None`.
///
/// `_seed` se conserva por compatibilidad y no controla el generador compartido.
pub fn sample(logits: &[f64], temperature: f64, _seed: u64) -> Option<u32> {
    sample_with_draw(logits, temperature, rand_val)
}

fn sample_with_draw(logits: &[f64], temperature: f64, draw: impl FnOnce() -> f64) -> Option<u32> {
    if temperature == 0.0 {
        if logits.iter().any(|logit| !logit.is_finite()) {
            return None;
        }
        return argmax(logits).map(|(id, _)| id);
    }
    TemperatureSoftmax::new(logits, temperature)?.select(draw())
}

struct TemperatureSoftmax<'a> {
    logits: &'a [f64],
    temperature: f64,
    max_logit: f64,
    sum: f64,
}

impl<'a> TemperatureSoftmax<'a> {
    fn new(logits: &'a [f64], temperature: f64) -> Option<Self> {
        if !temperature.is_finite()
            || temperature <= 0.0
            || logits.iter().any(|logit| !logit.is_finite())
        {
            return None;
        }
        let max_logit = logits.iter().copied().reduce(f64::max)?;
        // Each weight is in [0, 1], and a maximum has weight 1. The sum is
        // positive and finite even when every smaller weight underflows.
        let sum = logits
            .iter()
            .map(|&logit| temperature_weight(logit, max_logit, temperature))
            .sum();
        Some(Self {
            logits,
            temperature,
            max_logit,
            sum,
        })
    }

    fn probability(&self, logit: f64) -> f64 {
        temperature_weight(logit, self.max_logit, self.temperature) / self.sum
    }

    fn select(&self, uniform: f64) -> Option<u32> {
        if !(0.0..1.0).contains(&uniform) {
            return None;
        }
        let mut cumulative = 0.0;
        let mut last_positive = None;
        for (i, &logit) in self.logits.iter().enumerate() {
            let probability = self.probability(logit);
            if probability > 0.0 {
                last_positive = Some(i as u32);
                cumulative += probability;
                // Half-open CDF intervals exclude zero-mass tokens, including
                // at uniform=0 and exactly at a preceding interval's end.
                if uniform < cumulative {
                    return last_positive;
                }
            }
        }
        // Rounding can leave the CDF slightly below 1. Never fall back to a
        // trailing token whose probability underflowed to zero.
        last_positive
    }
}

fn temperature_weight(logit: f64, max_logit: f64, temperature: f64) -> f64 {
    let difference = logit - max_logit;
    let exponent = if difference.is_finite() {
        difference / temperature
    } else {
        // Finite logits can have an unrepresentable difference across zero.
        // Divide first only in this case; opposite signs prevent inf - inf.
        logit / temperature - max_logit / temperature
    };
    exponent.exp()
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
    unit_interval(state)
}

fn unit_interval(bits: u64) -> f64 {
    // The upper 53 bits are exactly representable as f64. Converting all 64
    // bits first can round u64::MAX to 2^64 and incorrectly produce 1.0.
    (bits >> 11) as f64 / (1_u64 << 53) as f64
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

    #[test]
    fn positive_subnormal_temperature_preserves_softmax_selection() {
        let reference = TemperatureSoftmax::new(&[0.0, 1.0], 1.0).unwrap();
        for temperature in [1.0, 1e-12, f64::MIN_POSITIVE, f64::from_bits(1), f64::MAX] {
            let logits = [0.0, temperature];
            let distribution = TemperatureSoftmax::new(&logits, temperature).unwrap();
            assert_probabilities(&distribution, &[0.2689414213699951, 0.7310585786300049]);
            assert_eq!(
                sample_with_draw(&logits, temperature, || 0.1),
                Some(0),
                "temperature={temperature}"
            );
            for step in 0..100 {
                let uniform = f64::from(step) / 100.0;
                assert_eq!(
                    sample_with_draw(&logits, temperature, || uniform),
                    reference.select(uniform),
                    "temperature={temperature}, uniform={uniform}"
                );
            }
        }
    }

    #[test]
    fn zero_uniform_skips_zero_probability() {
        assert_eq!(sample_with_draw(&[-f64::MAX, 0.0], 1.0, || 0.0), Some(1));
    }

    fn assert_probabilities(distribution: &TemperatureSoftmax<'_>, expected: &[f64]) {
        assert_eq!(distribution.logits.len(), expected.len());
        let mut sum = 0.0;
        for (&logit, &expected) in distribution.logits.iter().zip(expected) {
            let actual = distribution.probability(logit);
            assert!(actual.is_finite() && (0.0..=1.0).contains(&actual));
            assert!(
                (actual - expected).abs() <= 4.0 * f64::EPSILON,
                "expected {expected}, got {actual}"
            );
            sum += actual;
        }
        assert!((sum - 1.0).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn temperature_probabilities_match_closed_form_oracles() {
        // exp(-ln 2), exp(0), exp(ln 2) have weights 1:2:4.
        let ln2 = std::f64::consts::LN_2;
        for temperature in [0.8, 1.0, 2.0] {
            let logits = [-ln2 * temperature, 0.0, ln2 * temperature];
            let distribution = TemperatureSoftmax::new(&logits, temperature).unwrap();
            assert_probabilities(&distribution, &[1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0]);
            // Permutation changes token positions without changing their mass.
            let permuted = [logits[2], logits[0], logits[1]];
            let distribution = TemperatureSoftmax::new(&permuted, temperature).unwrap();
            assert_probabilities(&distribution, &[4.0 / 7.0, 1.0 / 7.0, 2.0 / 7.0]);
        }
    }

    #[test]
    fn finite_extreme_logits_avoid_difference_and_quotient_overflow() {
        // The exact normalized logits are [-1, 1], although max - min
        // cannot be represented before division by the temperature.
        let logits = [-f64::MAX, f64::MAX];
        let distribution = TemperatureSoftmax::new(&logits, f64::MAX).unwrap();
        assert_probabilities(&distribution, &[0.11920292202211755, 0.8807970779778823]);
        assert_eq!(distribution.select(0.1), Some(0));
        assert_eq!(distribution.select(0.2), Some(1));

        // Dividing before centering would instead produce inf - inf for
        // equal large logits at a small temperature.
        for logits in [[f64::MAX, f64::MAX], [-f64::MAX, -f64::MAX]] {
            let distribution = TemperatureSoftmax::new(&logits, f64::from_bits(1)).unwrap();
            assert_probabilities(&distribution, &[0.5, 0.5]);
        }
        let distribution = TemperatureSoftmax::new(&logits, f64::from_bits(1)).unwrap();
        assert_probabilities(&distribution, &[0.0, 1.0]);
        assert_eq!(distribution.select(0.0), Some(1));
    }

    #[test]
    fn tiny_temperature_keeps_tied_maxima_in_the_distribution() {
        for temperature in [1e-12, f64::MIN_POSITIVE, f64::from_bits(1)] {
            let logits = [-1.0, 1.0, 1.0];
            let distribution = TemperatureSoftmax::new(&logits, temperature).unwrap();
            assert_probabilities(&distribution, &[0.0, 0.5, 0.5]);
            assert_eq!(sample_with_draw(&logits, temperature, || 0.25), Some(1));
            assert_eq!(sample_with_draw(&logits, temperature, || 0.75), Some(2));
        }
    }

    #[test]
    fn cdf_intervals_exclude_zero_mass_and_the_upper_endpoint() {
        let logits = [-f64::MAX, 0.0, 0.0, -f64::MAX];
        let distribution = TemperatureSoftmax::new(&logits, 1.0).unwrap();
        assert_probabilities(&distribution, &[0.0, 0.5, 0.5, 0.0]);
        for (uniform, expected) in [(0.0, 1), (-0.0, 1), (0.5, 2), (unit_interval(u64::MAX), 2)] {
            assert_eq!(distribution.select(uniform), Some(expected));
        }
        for uniform in [
            -f64::EPSILON,
            1.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert_eq!(distribution.select(uniform), None);
        }
    }

    #[test]
    fn rounded_cdf_falls_back_only_to_positive_mass() {
        // Ten additions of 0.1 leave the CDF below 1 in f64.
        let mut logits = [0.0; 11];
        logits[10] = -f64::MAX;
        let distribution = TemperatureSoftmax::new(&logits, 1.0).unwrap();
        let cumulative: f64 = distribution
            .logits
            .iter()
            .map(|&logit| distribution.probability(logit))
            .sum();
        let uniform = unit_interval(u64::MAX);
        assert!(cumulative <= uniform);
        assert_eq!(distribution.select(uniform), Some(9));
    }

    #[test]
    fn invalid_sampling_parameters_are_rejected_without_drawing() {
        for temperature in [
            -1.0,
            -f64::from_bits(1),
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
        ] {
            assert_eq!(
                sample_with_draw(&[0.0, 1.0], temperature, || panic!("unexpected draw")),
                None
            );
        }
        for temperature in [0.0, 1.0, f64::from_bits(1)] {
            for invalid_logit in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                for logits in [[invalid_logit, 1.0], [1.0, invalid_logit]] {
                    assert_eq!(
                        sample_with_draw(&logits, temperature, || panic!("unexpected draw")),
                        None
                    );
                }
            }
            assert_eq!(
                sample_with_draw(&[], temperature, || panic!("unexpected draw")),
                None
            );
        }
    }

    #[test]
    fn zero_temperature_preserves_first_argmax_without_drawing() {
        for temperature in [0.0, -0.0] {
            assert_eq!(
                sample_with_draw(&[-1.0, 1.0, 1.0], temperature, || panic!("unexpected draw")),
                Some(1)
            );
        }
    }

    #[test]
    fn random_bits_map_to_a_half_open_unit_interval() {
        assert_eq!(unit_interval(0), 0.0);
        assert_eq!(unit_interval(1 << 63), 0.5);
        assert_eq!(unit_interval(u64::MAX), 1.0 - f64::EPSILON / 2.0);
        for bits in [0, 1, 2047, 2048, 1 << 63, u64::MAX - 1, u64::MAX] {
            assert!((0.0..1.0).contains(&unit_interval(bits)));
        }
    }
}
