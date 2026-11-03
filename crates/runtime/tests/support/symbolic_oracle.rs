//! Independent fixed-size oracle for symbolic_fixture::fixture(0.25).
//! Uses no production projection, normalization, rotary or activation helper.
pub fn logits(ids: &[u32], with_mlp: bool) -> Vec<f64> {
    fn norm(x: [f64; 4]) -> [f64; 4] {
        let root = (x.iter().map(|v| v * v).sum::<f64>() / 4.0 + 0.5).sqrt();
        x.map(|v| 0.5 * v / root)
    }
    fn rotate(x: [f64; 2], position: usize) -> [f64; 2] {
        let (s, c) = (position as f64).sin_cos();
        [x[0] * c - x[1] * s, x[0] * s + x[1] * c]
    }
    let mut keys = Vec::<[f32; 2]>::new();
    let mut values = Vec::<[f32; 2]>::new();
    let mut hidden = [0.0; 4];
    for (position, &id) in ids.iter().enumerate() {
        hidden = [0.0; 4];
        if id == 69 {
            hidden[0] = 0.25;
        }
        if id == 70 {
            hidden[1] = 0.25;
        }
        let n = norm(hidden);
        let q = n.map(|v| v * 0.25);
        keys.push(rotate([q[0], q[1]], position).map(|v| v as f32));
        values.push([q[0] as f32, q[1] as f32]);
        for head in 0..2 {
            let query = rotate([q[head * 2], q[head * 2 + 1]], position);
            let scores: Vec<_> = keys
                .iter()
                .map(|k| (query[0] * f64::from(k[0]) + query[1] * f64::from(k[1])) / 2.0_f64.sqrt())
                .collect();
            let exps: Vec<_> = scores.iter().map(|s| s.exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..2 {
                let attention: f64 = values
                    .iter()
                    .zip(&exps)
                    .map(|(v, e)| e / sum * f64::from(v[d]))
                    .sum();
                hidden[head * 2 + d] += 0.25 * attention;
            }
        }
        if with_mlp {
            let n = norm(hidden);
            for (h, v) in hidden.iter_mut().zip(n) {
                let g = 0.25 * v;
                let up = -0.25 * v;
                *h += 0.25 * g / (1.0 + (-g).exp()) * up;
            }
        }
    }
    let mut logits = vec![0.0; 260];
    for (out, v) in logits.iter_mut().zip(norm(hidden)) {
        *out = 0.5 * v;
    }
    logits
}
