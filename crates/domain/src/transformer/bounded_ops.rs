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
