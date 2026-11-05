//! Native-only observation of the public projection callback. No BML evaluation.
use bml_compiler::{
    gguf_compiler::{swiglu_scalar, InferenceCompiler},
    symbolic_bounded::BoundedProfile,
    symbolic_kernels::Operation,
    HashConsedKV,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Default)]
struct Range {
    count: usize,
    minimum: f64,
    maximum: f64,
}
impl Range {
    fn add(&mut self, values: &[f64]) -> Result<(), String> {
        for &value in values {
            if !value.is_finite() {
                return Err(format!("nonfinite observed value {value}"));
            }
            if self.count == 0 {
                self.minimum = value;
                self.maximum = value;
            } else {
                self.minimum = self.minimum.min(value);
                self.maximum = self.maximum.max(value);
            }
            self.count += 1;
        }
        Ok(())
    }
    fn report(&self) -> Value {
        json!({"values":self.count,"minimum":(self.count>0).then_some(self.minimum),
            "maximum":(self.count>0).then_some(self.maximum),
            "maximum_absolute":(self.count>0).then_some(self.minimum.abs().max(self.maximum.abs()))})
    }
}
#[derive(Default)]
struct Role {
    calls: usize,
    input: Range,
    external: Range,
    output: Range,
}
impl Role {
    fn report(&self) -> Value {
        json!({"calls":self.calls,"inputs":self.input.report(),
            "external":self.external.report(),"native_or_derived_outputs":self.output.report()})
    }
}
#[derive(Default)]
struct AttentionSample {
    query: Vec<f64>,
    key_f32: Vec<f64>,
    value_f32: Vec<f64>,
}

/// Holds only this text's previous native Q/K/V observations and role ranges.
pub struct Probe {
    profile: BoundedProfile,
    epsilon: f64,
    hidden: Vec<f64>,
    samples: Vec<BTreeMap<u32, AttentionSample>>,
    roles: BTreeMap<&'static str, Role>,
    projections: Vec<Value>,
    order: Vec<String>,
    gate: Option<Vec<f64>>,
    down: Option<Vec<f64>>,
    rows: usize,
    kv_checks: usize,
}
impl Probe {
    pub fn new(model: &InferenceCompiler, profile: BoundedProfile) -> Self {
        Self {
            profile,
            epsilon: model.rms_norm_epsilon(),
            hidden: Vec::new(),
            samples: Vec::new(),
            roles: BTreeMap::new(),
            projections: Vec::new(),
            order: Vec::new(),
            gate: None,
            down: None,
            rows: 0,
            kv_checks: 0,
        }
    }
    fn observe(
        &mut self,
        operation: Operation,
        width: usize,
        input: &[f64],
        external: &[f64],
        output: &[f64],
        location: &str,
    ) -> Result<(), String> {
        let role = self.roles.entry(operation.name()).or_default();
        role.calls += 1;
        role.input.add(input)?;
        role.external.add(external)?;
        role.output.add(output)?;
        self.profile
            .validate_values(operation, width, self.epsilon, input, external)
            .map_err(|error| format!("{location} {}: {error}", operation.name()))
    }
    fn norm(
        &mut self,
        model: &InferenceCompiler,
        name: &str,
        actual: &[f64],
    ) -> Result<(), String> {
        let start = *model
            .weight_offsets()
            .get(name)
            .ok_or("missing norm weights")? as usize;
        let weights: Vec<_> = model
            .weight_pool()
            .get(start..start + self.hidden.len())
            .ok_or("truncated norm weights")?
            .iter()
            .map(|&value| f64::from(value))
            .collect();
        let denominator = (self.hidden.iter().map(|value| value * value).sum::<f64>()
            / self.hidden.len() as f64
            + self.epsilon)
            .sqrt();
        let normalized: Vec<_> = self
            .hidden
            .iter()
            .zip(&weights)
            .map(|(&value, &weight)| value / denominator * weight)
            .collect();
        self.observe(
            Operation::Norm,
            self.hidden.len(),
            &self.hidden.clone(),
            &weights,
            &normalized,
            name,
        )?;
        same_bits(
            actual,
            &normalized,
            "mirrored norm input differs from native projection",
        )
    }
    fn rotate(
        &mut self,
        model: &InferenceCompiler,
        position: u32,
        input: &[f64],
        name: &str,
    ) -> Result<Vec<f64>, String> {
        let dimension = model.head_dim() as usize;
        if dimension == 0 || !dimension.is_multiple_of(2) || !input.len().is_multiple_of(dimension)
        {
            return Err("invalid observed RoPE dimensions".into());
        }
        let mut rotated = Vec::with_capacity(input.len());
        for (index, pair) in input.as_chunks::<2>().0.iter().enumerate() {
            let within = (index % (dimension / 2)) * 2;
            // Match the native rotation's operation order for the independent
            // K/F32 bookkeeping check. No symbolic arithmetic is executed.
            let frequency = 1.0
                / model
                    .rope_frequency_base()
                    .powf(within as f64 / dimension as f64);
            let angle = f64::from(position) * frequency;
            let (cosine, sine) = (angle.cos(), angle.sin());
            let output = [
                pair[0] * cosine - pair[1] * sine,
                pair[0] * sine + pair[1] * cosine,
            ];
            self.observe(Operation::Rope, 2, pair, &[cosine, sine], &output, name)?;
            rotated.extend(output);
        }
        Ok(rotated)
    }
    fn projection(
        &mut self,
        model: &InferenceCompiler,
        position: u32,
        input: &[f64],
        name: &str,
    ) -> Result<Vec<f64>, String> {
        if name.ends_with(".attn_q.weight") {
            self.norm(
                model,
                &name.replace(".attn_q.weight", ".attn_norm.weight"),
                input,
            )?;
        } else if name.ends_with(".ffn_gate.weight") {
            self.norm(
                model,
                &name.replace(".ffn_gate.weight", ".ffn_norm.weight"),
                input,
            )?;
        } else if name == model.output_weight_name() {
            self.norm(model, "output_norm.weight", input)?;
        }
        if name.ends_with(".ffn_down.weight") {
            same_bits(
                input,
                self.down.as_deref().ok_or("missing observed SwiGLU")?,
                "native SwiGLU input differs",
            )?;
        }
        let operation = if name.ends_with(".ffn_down.weight") {
            Operation::DownDot
        } else {
            Operation::Dot
        };
        let mut weight_range = Range::default();
        let mut row = 0;
        let output = model.project_rows(input, name, |values, weights| {
            let decoded: Vec<_> = weights.iter().map(|&value| f64::from(value)).collect();
            self.observe(
                operation,
                values.len(),
                values,
                &decoded,
                &[],
                &format!("{name} row={row}"),
            )?;
            weight_range.add(&decoded)?;
            row += 1;
            // Exactly the public native projection probe's fold order.
            Ok(values
                .iter()
                .zip(weights)
                .fold(0.0, |sum, (&value, &weight)| {
                    sum + value * f64::from(weight)
                }))
        })?;
        let mut inputs = Range::default();
        inputs.add(input)?;
        let mut outputs = Range::default();
        outputs.add(&output)?;
        self.roles
            .get_mut(operation.name())
            .ok_or("empty projection")?
            .output
            .add(&output)?;
        self.rows += row;
        self.order.push(name.to_owned());
        self.projections.push(
            json!({"tensor":name,"role":operation.name(),"rows_checked":row,
            "inputs":inputs.report(),"weights":weight_range.report(),"outputs":outputs.report()}),
        );
        if let Some(rest) = name.strip_prefix("blk.") {
            let (layer, _) = rest.split_once('.').ok_or("invalid projection name")?;
            let layer: u32 = layer.parse().map_err(|_| "invalid projection layer")?;
            if name.ends_with(".attn_q.weight") {
                let query = self.rotate(model, position, &output, name)?;
                self.samples[position as usize]
                    .entry(layer)
                    .or_default()
                    .query = query;
            } else if name.ends_with(".attn_k.weight") {
                let rotated = self.rotate(model, position, &output, name)?;
                self.samples[position as usize]
                    .entry(layer)
                    .or_default()
                    .key_f32 = round_f32(&rotated)?;
            } else if name.ends_with(".attn_v.weight") {
                self.samples[position as usize]
                    .entry(layer)
                    .or_default()
                    .value_f32 = round_f32(&output)?;
            }
        }
        if name.ends_with(".ffn_gate.weight") {
            self.gate = Some(output.clone());
        } else if name.ends_with(".ffn_up.weight") {
            let gate = self.gate.take().ok_or("missing gate projection")?;
            if gate.len() != output.len() {
                return Err("observed gate/up shapes differ".into());
            }
            let mut activated = Vec::with_capacity(gate.len());
            for (&gate, &up) in gate.iter().zip(&output) {
                let value = swiglu_scalar(gate, up);
                self.observe(Operation::Swiglu, 1, &[gate, up], &[], &[value], name)?;
                activated.push(value);
            }
            self.down = Some(activated);
        }
        if name.ends_with(".attn_output.weight") || name.ends_with(".ffn_down.weight") {
            if self.hidden.len() != output.len() {
                return Err("observed residual shapes differ".into());
            }
            for (index, &right) in output.iter().enumerate() {
                let left = self.hidden[index];
                self.observe(
                    Operation::Add,
                    1,
                    &[left, right],
                    &[],
                    &[left + right],
                    name,
                )?;
                self.hidden[index] = left + right;
            }
        }
        Ok(output)
    }
    fn attention(
        &mut self,
        model: &InferenceCompiler,
        position: u32,
        cache: &HashConsedKV,
    ) -> Result<(), String> {
        let dimension = model.head_dim() as usize;
        let heads = model.config().n_heads as usize;
        let kv_heads = model.n_kv_heads() as usize;
        for layer in 0..model.config().n_layers {
            for prior in 0..=position {
                let sample = self.samples[prior as usize]
                    .get(&layer)
                    .ok_or("missing observed layer")?;
                if sample.query.len() != heads * dimension
                    || sample.key_f32.len() != kv_heads * dimension
                    || sample.value_f32.len() != kv_heads * dimension
                {
                    return Err("incomplete observed Q/K/V dimensions".into());
                }
                for head in 0..kv_heads {
                    for (actual, expected) in [
                        (cache.load_k(layer, prior, head as u32), &sample.key_f32),
                        (cache.load_v(layer, prior, head as u32), &sample.value_f32),
                    ] {
                        if actual
                            .iter()
                            .map(|v| v.to_bits())
                            .ne(expected[head * dimension..(head + 1) * dimension]
                                .iter()
                                .map(|v| (*v as f32).to_bits()))
                        {
                            return Err(format!("native KV bookkeeping differs layer={layer} position={prior} head={head}"));
                        }
                        self.kv_checks += 1;
                    }
                }
            }
            for head in 0..heads {
                let query = self.samples[position as usize][&layer].query
                    [head * dimension..(head + 1) * dimension]
                    .to_vec();
                let kv_head = head / (heads / kv_heads);
                let mut scores = Vec::new();
                for prior in 0..=position {
                    let key = self.samples[prior as usize][&layer].key_f32
                        [kv_head * dimension..(kv_head + 1) * dimension]
                        .to_vec();
                    let dot = query
                        .iter()
                        .zip(&key)
                        .fold(0.0, |sum, (&q, &k)| sum + q * k);
                    self.observe(
                        Operation::AttentionDot,
                        dimension,
                        &query,
                        &key,
                        &[dot],
                        "native-derived attention",
                    )?;
                    let scale = 1.0 / (dimension as f64).sqrt();
                    let score = dot * scale;
                    self.observe(
                        Operation::Scale,
                        1,
                        &[dot, scale],
                        &[],
                        &[score],
                        "native-derived score",
                    )?;
                    scores.push(score);
                }
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut context = scores.clone();
                context.push(maximum);
                let exps: Vec<_> = scores.iter().map(|score| (score - maximum).exp()).collect();
                let denominator: f64 = exps.iter().sum();
                let probabilities: Vec<_> = exps.iter().map(|value| value / denominator).collect();
                self.observe(
                    Operation::Softmax,
                    scores.len(),
                    &context,
                    &[],
                    &probabilities,
                    "native-derived softmax",
                )?;
                for channel in 0..dimension {
                    let values: Vec<_> = (0..=position)
                        .map(|prior| {
                            self.samples[prior as usize][&layer].value_f32
                                [kv_head * dimension + channel]
                        })
                        .collect();
                    let output = probabilities
                        .iter()
                        .zip(&values)
                        .fold(0.0, |sum, (&p, &v)| sum + p * v);
                    self.observe(
                        Operation::WeightedSum,
                        values.len(),
                        &probabilities,
                        &values,
                        &[output],
                        "native-derived weighted sum",
                    )?;
                }
            }
        }
        Ok(())
    }
    pub fn position(
        &mut self,
        model: &InferenceCompiler,
        id: u32,
        cache: &mut HashConsedKV,
    ) -> Result<Vec<f64>, String> {
        let position = cache.current_pos();
        if position as usize != self.samples.len() {
            return Err("nonsequential native probe".into());
        }
        self.hidden = model.get_embedding(id)?;
        self.samples.push(BTreeMap::new());
        self.roles.clear();
        self.projections.clear();
        self.order.clear();
        self.rows = 0;
        self.kv_checks = 0;
        self.gate = None;
        self.down = None;
        let logits = model.forward_cached_with_projection(&[id], cache, &mut |input, name| {
            self.projection(model, position, input, name)
        })?;
        let mut expected_order = Vec::new();
        let mut expected_rows = 0;
        for layer in 0..model.config().n_layers {
            for role in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                expected_order.push(format!("blk.{layer}.{role}.weight"));
            }
        }
        expected_order.push(model.output_weight_name().into());
        for name in &expected_order {
            expected_rows += model
                .tensor_dims()
                .get(name)
                .ok_or("missing projection dimensions")?
                .1;
        }
        if self.order != expected_order || self.rows != expected_rows {
            return Err("incomplete native projection coverage".into());
        }
        self.attention(model, position, cache)?;
        if self.roles.get("rms_norm").map(|role| role.calls)
            != Some(model.config().n_layers as usize * 2 + 1)
        {
            return Err("incomplete RMS coverage".into());
        }
        let layers = model.config().n_layers as usize;
        let heads = model.config().n_heads as usize;
        let width = model.config().n_embd as usize;
        let kv_width = model.n_kv_heads() as usize * model.head_dim() as usize;
        let down_rows: usize = (0..model.config().n_layers)
            .map(|layer| model.tensor_dims()[&format!("blk.{layer}.ffn_down.weight")].1)
            .sum();
        let activations: usize = (0..model.config().n_layers)
            .map(|layer| model.tensor_dims()[&format!("blk.{layer}.ffn_up.weight")].1)
            .sum();
        for (name, expected) in [
            ("dot", expected_rows - down_rows),
            ("down_dot", down_rows),
            ("rope", layers * (width + kv_width) / 2),
            ("residual_add", layers * width * 2),
            ("swiglu", activations),
            ("attention_dot", layers * heads * (position as usize + 1)),
            ("scale", layers * heads * (position as usize + 1)),
            ("softmax", layers * heads),
            ("weighted_sum", layers * heads * model.head_dim() as usize),
        ] {
            if self.roles.get(name).map(|role| role.calls) != Some(expected) {
                return Err(format!("incomplete {name} coverage; expected {expected}"));
            }
        }
        if self.kv_checks != layers * model.n_kv_heads() as usize * (position as usize + 1) * 2 {
            return Err("incomplete native KV verification".into());
        }
        if self.roles.len() != 10 {
            return Err("incomplete native-derived family coverage".into());
        }
        if logits.len() != model.vocab().len() || logits.iter().any(|value| !value.is_finite()) {
            return Err("invalid native preflight logits".into());
        }
        Ok(logits)
    }
    pub fn report(&self) -> Value {
        json!({"projection_rows_checked":self.rows,"projection_callbacks":self.order.len(),
            "projections":self.projections,"kv_head_rows_verified_in_bits":self.kv_checks,
            "roles":self.roles.iter().map(|(name,role)|(*name,role.report())).collect::<BTreeMap<_,_>>(),
            "attention_scope":"Symbolic-family operands derived from native Q/F64 and K,V/F32; distinct from native internal Q/dot/probabilities/F32. No BML execution."})
    }
}
fn round_f32(values: &[f64]) -> Result<Vec<f64>, String> {
    let rounded: Vec<_> = values
        .iter()
        .map(|&value| f64::from(value as f32))
        .collect();
    if rounded.iter().any(|value| !value.is_finite()) {
        return Err("nonfinite rounded KV value".into());
    }
    Ok(rounded)
}
fn same_bits(actual: &[f64], expected: &[f64], error: &str) -> Result<(), String> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .map(|v| v.to_bits())
            .ne(expected.iter().map(|v| v.to_bits()))
    {
        return Err(error.into());
    }
    Ok(())
}
