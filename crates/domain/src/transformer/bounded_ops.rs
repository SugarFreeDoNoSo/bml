//! Opt-in bounded arithmetic for transformer operation templates.
//!
//! Every activation calculation produces BML nodes. Bounds and literal
//! construction do not certify floating-point accuracy or evaluability: the
//! evaluator must check every BML intermediate. Historical builders are intact.

use super::{BMLTransformer, BoundedBuildError};
use crate::NodeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Product,
    Add,
    DotStep,
    Rope,
    Swiglu,
    Softmax,
}

/// Validated operand declarations for a distinct bounded arithmetic template.
///
/// Constructors specify the input layout. The literal prefix contains only
/// configuration constants, never activations or learned model weights.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundedOpConfig {
    kind: Kind,
    width: usize,
    left: f64,
    right: f64,
}

fn bound(value: f64) -> Result<(), BoundedBuildError> {
    if !value.is_finite() || !(1.0 / 256.0..=256.0).contains(&value) {
        Err(BoundedBuildError::Parameter("operand_bound"))
    } else {
        Ok(())
    }
}

impl BoundedOpConfig {
    /// Signed product of two inputs with inclusive absolute operand bounds.
    /// Their bound product must be at most 480, leaving exponential headroom.
    pub fn product(left: f64, right: f64) -> Result<Self, BoundedBuildError> {
        bound(left)?;
        bound(right)?;
        if left * right > 480.0 {
            return Err(BoundedBuildError::Parameter("operand_bound_product"));
        }
        Ok(Self {
            kind: Kind::Product,
            width: 1,
            left,
            right,
        })
    }

    /// Sum of two signed inputs. The fixed shift is twice the left bound.
    pub fn add(left: f64, right: f64) -> Result<Self, BoundedBuildError> {
        bound(left)?;
        bound(right)?;
        if 3.0 * left + right > 960.0 {
            return Err(BoundedBuildError::Parameter("addition_range"));
        }
        Ok(Self {
            kind: Kind::Add,
            width: 1,
            left,
            right,
        })
    }

    /// Dot recurrence inputs `[accumulator, left, right]` with fixed sum shift 256.
    ///
    /// The accumulator must be finite and strictly below 256. There is no
    /// claimed universal lower bound or guaranteed summation range: every BML
    /// operation must still be checked during execution. The loop's initial
    /// accumulator is supplied by its caller; this template owns no hidden state.
    pub fn dot_step(left: f64, right: f64) -> Result<Self, BoundedBuildError> {
        let mut config = Self::product(left, right)?;
        config.kind = Kind::DotStep;
        Ok(config)
    }

    /// Rotate two inputs using two external coefficients `[cosine, sine]`.
    /// Each coefficient has inclusive absolute bound one.
    pub fn rope(input_bound: f64) -> Result<Self, BoundedBuildError> {
        bound(input_bound)?;
        if input_bound > 240.0 {
            return Err(BoundedBuildError::Parameter("rotation_range"));
        }
        Ok(Self {
            kind: Kind::Rope,
            width: 2,
            left: input_bound,
            right: 1.0,
        })
    }

    /// SwiGLU inputs `[gate, up]`, with a gate bound no larger than 16.
    /// The sigmoid uses a fixed exponential scaling, keeping its denominator
    /// above `exp(-gate_bound)` without evaluating the unstable `exp(-gate)`.
    pub fn swiglu(gate_bound: f64, up_bound: f64) -> Result<Self, BoundedBuildError> {
        let mut config = Self::product(gate_bound, up_bound)?;
        if gate_bound > 16.0 {
            return Err(BoundedBuildError::Parameter("sigmoid_range"));
        }
        config.kind = Kind::Swiglu;
        Ok(config)
    }

    /// Softmax inputs are `width` scores followed by their selected maximum.
    ///
    /// Selecting/copying that maximum is a caller responsibility, checked by
    /// [`Self::validate_values`]. Every subtraction, exponential, sum and
    /// division remains in the BML DAG. Supplying a fixed upper bound instead
    /// can lose substantial relative accuracy when all scores are negative.
    pub fn softmax(width: usize, input_bound: f64) -> Result<Self, BoundedBuildError> {
        bound(input_bound)?;
        if input_bound > 16.0 {
            return Err(BoundedBuildError::Parameter("softmax_range"));
        }
        if width == 0 || width > 65536 {
            return Err(BoundedBuildError::Parameter("softmax_width"));
        }
        Ok(Self {
            kind: Kind::Softmax,
            width,
            left: input_bound,
            right: 1.0,
        })
    }

    /// Number of runtime inputs, including the explicit softmax maximum.
    pub fn inputs(&self) -> usize {
        match self.kind {
            Kind::DotStep => 3,
            Kind::Softmax => self.width + 1,
            _ => 2,
        }
    }
    /// Number of external model/coefficient values following the literal prefix.
    pub fn external(&self) -> usize {
        if self.kind == Kind::Rope {
            2
        } else {
            0
        }
    }
    /// Number of scalar results produced by the template.
    pub fn outputs(&self) -> usize {
        match self.kind {
            Kind::Rope => 2,
            Kind::Softmax => self.width,
            _ => 1,
        }
    }

    /// Deterministic owned constants, appended exactly once per construction.
    pub fn fixed_literals(&self) -> Vec<f64> {
        let mut values = vec![64.0, 2.0];
        let product = |values: &mut Vec<f64>, left: f64, right: f64| {
            values.extend([
                2.0 * left,
                1.0 / left,
                2.0 * right,
                1.0 / right,
                left * right,
                2.0 * left * right,
            ]);
        };
        match self.kind {
            Kind::Product => product(&mut values, self.left, self.right),
            Kind::Add => values.push(2.0 * self.left),
            Kind::DotStep => {
                product(&mut values, self.left, self.right);
                values.push(256.0);
            }
            Kind::Rope => {
                product(&mut values, self.left, 1.0);
                values.push(2.0 * self.left);
            }
            Kind::Swiglu => {
                values.extend([
                    -self.left,
                    2.0 * self.left,
                    (-self.left).exp(),
                    std::f64::consts::LOG2_E,
                ]);
                product(&mut values, 2.0 * self.left, 2.0);
                product(&mut values, self.left, 1.0);
                product(&mut values, self.left, self.right);
            }
            Kind::Softmax => {
                values.extend([2.0 * self.left, std::f64::consts::LOG2_E]);
                product(&mut values, 2.0 * self.left, 2.0);
            }
        }
        values
    }

    /// Reject mismatched shapes, nonfinite operands and violated declarations.
    /// This check does not replace finite-real checking of every BML operation.
    pub fn validate_values(
        &self,
        inputs: &[f64],
        external: &[f64],
    ) -> Result<(), BoundedBuildError> {
        if inputs.len() != self.inputs() || external.len() != self.external() {
            return Err(BoundedBuildError::Shape);
        }
        for (index, &value) in inputs.iter().enumerate() {
            let valid = if self.kind == Kind::DotStep && index == 0 {
                value < 256.0
            } else {
                let limit = match self.kind {
                    Kind::Product | Kind::Add | Kind::Swiglu if index == 1 => self.right,
                    Kind::DotStep if index == 2 => self.right,
                    _ => self.left,
                };
                value.abs() <= limit
            };
            if !value.is_finite() || !valid {
                return Err(BoundedBuildError::Input(index));
            }
        }
        for (index, &value) in external.iter().enumerate() {
            if !value.is_finite() || value.abs() > 1.0 {
                return Err(BoundedBuildError::Weight(index));
            }
        }
        if self.kind == Kind::Softmax {
            let maximum = inputs[..self.width]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if inputs[self.width] != maximum {
                return Err(BoundedBuildError::Input(self.width));
            }
        }
        Ok(())
    }
}

impl BMLTransformer {
    fn product_with_literals(
        &mut self,
        x: NodeId,
        y: NodeId,
        common: &[NodeId],
        p: &[NodeId],
    ) -> NodeId {
        let unit_x = self.bounded_normalize(x, p[0], p[1], common[1], common[0]);
        let unit_y = self.bounded_normalize(y, p[2], p[3], common[1], common[0]);
        self.bounded_product_from_normalized(unit_x, unit_y, p[4], p[5], common[1], common[0])
    }

    fn bounded_softmax_stage_literals(
        &mut self,
        inputs: &[NodeId],
        arity: usize,
        config: &BoundedOpConfig,
    ) -> Result<[NodeId; 10], BoundedBuildError> {
        if config.kind != Kind::Softmax {
            return Err(BoundedBuildError::Parameter("softmax_stage"));
        }
        if inputs.len() != arity {
            return Err(BoundedBuildError::Shape);
        }
        for &node in inputs {
            if node as usize >= self.soa.len() {
                return Err(BoundedBuildError::InvalidNode(node));
            }
        }
        if self
            .soa
            .len()
            .checked_add(16384)
            .is_none_or(|size| size > u32::MAX as usize)
            || self.soa.const_pool.len() > u32::MAX as usize - 10
        {
            return Err(BoundedBuildError::SizeOverflow);
        }
        let mut literals = [0; 10];
        for (node, value) in literals.iter_mut().zip(config.fixed_literals()) {
            *node = self.constant(value);
        }
        Ok(literals)
    }

    fn bounded_softmax_exp_with_literals(
        &mut self,
        score: NodeId,
        maximum: NodeId,
        c: &[NodeId; 10],
    ) -> NodeId {
        let negative_maximum = self.bounded_neg(maximum, c[2]);
        let shifted = self.bounded_add(score, negative_maximum, c[2]);
        let exponent = self.product_with_literals(shifted, c[3], &c[..2], &c[4..10]);
        self.exp2(exponent)
    }

    /// Build the bounded softmax exponential from `[score, maximum]`.
    ///
    /// Initialize the running sum with this result for the first score. The
    /// maximum is copied from the complete input vector and must be validated
    /// by [`BoundedOpConfig::validate_values`] before any stage executes. This
    /// is exactly the exponential formula of [`Self::operation_bounded`].
    ///
    /// Every stage adds the same ten fixed literals, has width-independent node
    /// count, and checks configuration kind, shape, nodes and capacity before
    /// mutation. Finite-real evaluation checks remain required for every BML.
    pub fn softmax_bounded_exp(
        &mut self,
        inputs: &[NodeId],
        config: &BoundedOpConfig,
    ) -> Result<NodeId, BoundedBuildError> {
        let c = self.bounded_softmax_stage_literals(inputs, 2, config)?;
        Ok(self.bounded_softmax_exp_with_literals(inputs[0], inputs[1], &c))
    }

    /// Build one ordered bounded softmax reduction from `[total, score, maximum]`.
    ///
    /// Start with [`Self::softmax_bounded_exp`] of the first score, then apply
    /// this stage to scores `1..width` in order. Starting at zero, regrouping
    /// sums or replacing the selected maximum changes the represented formula.
    /// Configuration, graph and execution requirements match the first stage.
    pub fn softmax_bounded_reduce(
        &mut self,
        inputs: &[NodeId],
        config: &BoundedOpConfig,
    ) -> Result<NodeId, BoundedBuildError> {
        let c = self.bounded_softmax_stage_literals(inputs, 3, config)?;
        let value = self.bounded_softmax_exp_with_literals(inputs[1], inputs[2], &c);
        Ok(self.bounded_positive_add(inputs[0], value, c[1]))
    }

    /// Build the shared bounded softmax reciprocal from `[total]`.
    ///
    /// The positive total must come from the ordered exponential reduction.
    /// This stage preserves the original log2, bounded negation and exp2
    /// sequence. Its output is shared by [`Self::softmax_bounded_output`].
    /// Configuration, graph and execution requirements match the first stage.
    pub fn softmax_bounded_inverse(
        &mut self,
        inputs: &[NodeId],
        config: &BoundedOpConfig,
    ) -> Result<NodeId, BoundedBuildError> {
        let c = self.bounded_softmax_stage_literals(inputs, 1, config)?;
        let log_total = self.log2(inputs[0]);
        let negative_log = self.bounded_neg(log_total, c[0]);
        Ok(self.exp2(negative_log))
    }

    /// Build one bounded softmax probability from `[score, maximum, inverse]`.
    ///
    /// `inverse` comes from [`Self::softmax_bounded_inverse`]. Recomputing the
    /// exponential uses exactly the original operands and BML sequence, so no
    /// width-sized cache of exponentials is required. Configuration, graph and
    /// execution requirements match the first stage.
    pub fn softmax_bounded_output(
        &mut self,
        inputs: &[NodeId],
        config: &BoundedOpConfig,
    ) -> Result<NodeId, BoundedBuildError> {
        let c = self.bounded_softmax_stage_literals(inputs, 3, config)?;
        let value = self.bounded_softmax_exp_with_literals(inputs[0], inputs[1], &c);
        Ok(self.bounded_positive_mul(value, inputs[2], c[0]))
    }

    /// Build the configured operation from existing input/external node IDs.
    ///
    /// The caller supplies the declared runtime inputs and enforces
    /// [`BoundedOpConfig::validate_values`]. All activation arithmetic is BML;
    /// only fixed configuration literals use native arithmetic. Shapes and IDs
    /// are checked before modifying the builder. Checked evaluation remains
    /// necessary even for accepted operand declarations.
    pub fn operation_bounded(
        &mut self,
        inputs: &[NodeId],
        external: &[NodeId],
        config: &BoundedOpConfig,
    ) -> Result<Vec<NodeId>, BoundedBuildError> {
        if inputs.len() != config.inputs() || external.len() != config.external() {
            return Err(BoundedBuildError::Shape);
        }
        for &node in inputs.iter().chain(external) {
            if node as usize >= self.soa.len() {
                return Err(BoundedBuildError::InvalidNode(node));
            }
        }
        let literals = config.fixed_literals();
        if config
            .inputs()
            .checked_mul(4096)
            .and_then(|n| n.checked_add(16384))
            .and_then(|n| n.checked_add(self.soa.len()))
            .is_none_or(|n| n > u32::MAX as usize)
            || self
                .soa
                .const_pool
                .len()
                .checked_add(literals.len())
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(BoundedBuildError::SizeOverflow);
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(config.outputs())
            .map_err(|_| BoundedBuildError::Allocation)?;
        let c: Vec<_> = literals
            .into_iter()
            .map(|value| self.constant(value))
            .collect();
        match config.kind {
            Kind::Product => {
                output.push(self.product_with_literals(inputs[0], inputs[1], &c[..2], &c[2..8]))
            }
            Kind::Add => output.push(self.bounded_add(inputs[0], inputs[1], c[2])),
            Kind::DotStep => {
                let product = self.product_with_literals(inputs[1], inputs[2], &c[..2], &c[2..8]);
                output.push(self.bounded_add(inputs[0], product, c[8]));
            }
            Kind::Rope => {
                let xc = self.product_with_literals(inputs[0], external[0], &c[..2], &c[2..8]);
                let ys = self.product_with_literals(inputs[1], external[1], &c[..2], &c[2..8]);
                let minus_ys = self.bounded_neg(ys, c[8]);
                output.push(self.bounded_add(xc, minus_ys, c[8]));
                let xs = self.product_with_literals(inputs[0], external[1], &c[..2], &c[2..8]);
                let yc = self.product_with_literals(inputs[1], external[0], &c[..2], &c[2..8]);
                output.push(self.bounded_add(xs, yc, c[8]));
            }
            Kind::Swiglu => {
                let shifted = self.bounded_add(inputs[0], c[2], c[3]);
                let exponent = self.product_with_literals(shifted, c[5], &c[..2], &c[6..12]);
                let numerator = self.exp2(exponent);
                let denominator = self.bounded_positive_add(c[4], numerator, c[1]);
                let log_denominator = self.log2(denominator);
                let minus_log = self.bounded_neg(log_denominator, c[0]);
                let inverse = self.exp2(minus_log);
                let sigmoid = self.bounded_positive_mul(numerator, inverse, c[0]);
                let silu = self.product_with_literals(inputs[0], sigmoid, &c[..2], &c[12..18]);
                output.push(self.product_with_literals(silu, inputs[1], &c[..2], &c[18..24]));
            }
            Kind::Softmax => {
                let negative_maximum = self.bounded_neg(inputs[config.width], c[2]);
                let mut exponentials = Vec::with_capacity(config.width);
                for &input in &inputs[..config.width] {
                    let shifted = self.bounded_add(input, negative_maximum, c[2]);
                    let exponent = self.product_with_literals(shifted, c[3], &c[..2], &c[4..10]);
                    exponentials.push(self.exp2(exponent));
                }
                let mut total = exponentials[0];
                for &value in &exponentials[1..] {
                    total = self.bounded_positive_add(total, value, c[1]);
                }
                let log_total = self.log2(total);
                let negative_log = self.bounded_neg(log_total, c[0]);
                let inverse = self.exp2(negative_log);
                for value in exponentials {
                    output.push(self.bounded_positive_mul(value, inverse, c[0]));
                }
            }
        }
        Ok(output)
    }
}
