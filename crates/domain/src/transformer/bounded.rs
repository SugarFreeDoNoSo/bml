//! Explicitly bounded, experimental finite-real RMS expansion.
//!
//! This is a separate arithmetic construction: legacy expansions and their
//! rounding remain unchanged. Bounds describe the operands supplied at runtime,
//! not facts inferred from arbitrary graph nodes. Even accepted parameters do
//! not certify every floating-point intermediate or an error tolerance.

use super::BMLTransformer;
use crate::NodeId;

/// A malformed bounded operation or a violated runtime operand declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundedBuildError {
    /// A parameter is outside the supported experimental range.
    Parameter(&'static str),
    /// Input and weight lengths differ from the configured nonempty width.
    Shape,
    /// An operand does not refer to an existing node in this builder.
    InvalidNode(NodeId),
    /// The graph or constant indices would exceed their representable range.
    SizeOverflow,
    /// The output allocation could not be reserved.
    Allocation,
    /// An activation is nonfinite or exceeds the declared absolute bound.
    Input(usize),
    /// A weight is nonfinite or exceeds the declared absolute bound.
    Weight(usize),
}

impl std::fmt::Display for BoundedBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parameter(name) => write!(f, "unsupported bounded RMS parameter: {name}"),
            Self::Shape => f.write_str("incompatible bounded RMS shapes"),
            Self::InvalidNode(id) => write!(f, "unknown bounded RMS operand node {id}"),
            Self::SizeOverflow => f.write_str("bounded RMS graph size overflow"),
            Self::Allocation => f.write_str("cannot reserve bounded RMS outputs"),
            Self::Input(index) => write!(f, "bounded RMS input {index} exceeds its declaration"),
            Self::Weight(index) => write!(f, "bounded RMS weight {index} exceeds its declaration"),
        }
    }
}

impl std::error::Error for BoundedBuildError {}

/// Validated construction parameters for a distinct bounded RMS arithmetic.
///
/// Let `M` and `W` be the activation and weight absolute bounds, and
/// `K = ceil(sqrt(width)) + 1`. This experimental construction accepts
/// `M,W` in `[2^-8,256]`, `K <= 256`, `K*W <= 480`, and scaled epsilon
/// `epsilon/M²` in `[2^-32,1]`. The product limit leaves a margin below the
/// exponential overflow at 1024: the largest fixed rescaling exponent is
/// `2*K*W <= 960`. These are construction limits, not a universal numerical
/// accuracy guarantee. An evaluator must still check all BML intermediates.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundedRmsConfig {
    width: usize,
    epsilon: f64,
    input_bound: f64,
    weight_bound: f64,
    normalized_bound: f64,
    log_variance_bound: f64,
}

impl BoundedRmsConfig {
    /// Validate parameters without constructing any graph or evaluating inputs.
    pub fn new(
        width: usize,
        epsilon: f64,
        input_bound: f64,
        weight_bound: f64,
    ) -> Result<Self, BoundedBuildError> {
        if width == 0 || width > 255 * 255 {
            return Err(BoundedBuildError::Parameter("width"));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(BoundedBuildError::Parameter("epsilon"));
        }
        if !input_bound.is_finite() || !(1.0 / 256.0..=256.0).contains(&input_bound) {
            return Err(BoundedBuildError::Parameter("input_bound"));
        }
        if !weight_bound.is_finite() || !(1.0 / 256.0..=256.0).contains(&weight_bound) {
            return Err(BoundedBuildError::Parameter("weight_bound"));
        }
        let normalized_bound = (width as f64).sqrt().ceil() + 1.0;
        if normalized_bound * weight_bound > 480.0 {
            return Err(BoundedBuildError::Parameter("normalized_weight_product"));
        }
        let scaled_epsilon = epsilon / (input_bound * input_bound);
        if !(2.0_f64.powi(-32)..=1.0).contains(&scaled_epsilon) {
            return Err(BoundedBuildError::Parameter("scaled_epsilon"));
        }
        let log_variance_bound = scaled_epsilon
            .log2()
            .abs()
            .max((1.0 + scaled_epsilon).log2().abs())
            + 1.0;
        Ok(Self {
            width,
            epsilon,
            input_bound,
            weight_bound,
            normalized_bound,
            log_variance_bound,
        })
    }

    /// Number of activation and learned-weight elements.
    pub fn width(&self) -> usize {
        self.width
    }
    /// Unscaled epsilon in the represented RMS formula.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }
    /// Inclusive absolute activation bound.
    pub fn input_bound(&self) -> f64 {
        self.input_bound
    }
    /// Inclusive absolute learned-weight bound.
    pub fn weight_bound(&self) -> f64 {
        self.weight_bound
    }

    /// Canonical literal prefix appended once by the bounded RMS builder.
    ///
    /// This contains only configuration-derived constants, never learned
    /// weights or activation values. External weight references must be
    /// placed after this prefix (or remapped by the caller) when sharing its
    /// constant namespace. No sample execution is needed to recover it.
    pub fn fixed_literals(&self) -> Vec<f64> {
        let m = self.input_bound;
        let w = self.weight_bound;
        let k = self.normalized_bound;
        let l = self.log_variance_bound;
        vec![
            64.0,
            2.0,
            2.0 * m,
            1.0 / m,
            self.width as f64 * (self.epsilon / (m * m)),
            1.0 / self.width as f64,
            0.5,
            l,
            2.0 * l,
            1.0 / l,
            k,
            2.0 * k,
            1.0 / k,
            2.0 * w,
            1.0 / w,
            k * w,
            2.0 * k * w,
        ]
    }

    /// Check the declared operand ranges before executing the compiled graph.
    ///
    /// Successful validation does not replace per-node finite-real checks.
    pub fn validate_values(&self, input: &[f64], weights: &[f64]) -> Result<(), BoundedBuildError> {
        if input.len() != self.width || weights.len() != self.width {
            return Err(BoundedBuildError::Shape);
        }
        for (index, value) in input.iter().enumerate() {
            if !value.is_finite() || value.abs() > self.input_bound {
                return Err(BoundedBuildError::Input(index));
            }
        }
        for (index, value) in weights.iter().enumerate() {
            if !value.is_finite() || value.abs() > self.weight_bound {
                return Err(BoundedBuildError::Weight(index));
            }
        }
        Ok(())
    }
}

impl BMLTransformer {
    // With D(a,b)=a-b for positive a, N_C(x)=D(D(C,x),C)=-x
    // and A_C(x,y)=D(C,D(D(C,x),y))=x+y. Both require x<C.
    pub(super) fn bounded_neg(&mut self, x: NodeId, c: NodeId) -> NodeId {
        let difference = self.sub(c, x);
        self.sub(difference, c)
    }

    pub(super) fn bounded_add(&mut self, x: NodeId, y: NodeId, c: NodeId) -> NodeId {
        let difference = self.sub(c, x);
        let remainder = self.sub(difference, y);
        self.sub(c, remainder)
    }

    pub(super) fn bounded_positive_add(
        &mut self,
        positive_x: NodeId,
        y: NodeId,
        c: NodeId,
    ) -> NodeId {
        let negative_y = self.bounded_neg(y, c);
        self.sub(positive_x, negative_y)
    }

    pub(super) fn bounded_positive_mul(&mut self, x: NodeId, y: NodeId, c64: NodeId) -> NodeId {
        let log_x = self.log2(x);
        let log_y = self.log2(y);
        let exponent = self.bounded_add(log_x, log_y, c64);
        self.exp2(exponent)
    }

    pub(super) fn bounded_normalize(
        &mut self,
        x: NodeId,
        twice_bound: NodeId,
        inverse_bound: NodeId,
        two: NodeId,
        c64: NodeId,
    ) -> NodeId {
        let shifted = self.bounded_add(x, twice_bound, twice_bound);
        let scaled = self.bounded_positive_mul(shifted, inverse_bound, c64);
        self.sub(scaled, two)
    }

    // Inputs already normalized into [-1,1]; the old signed-product identity
    // then has small positive components. Shift by two before rescaling so
    // opposite-sign endpoints cannot supply log2(0).
    pub(super) fn bounded_product_from_normalized(
        &mut self,
        x: NodeId,
        y: NodeId,
        product_bound: NodeId,
        twice_product_bound: NodeId,
        two: NodeId,
        c64: NodeId,
    ) -> NodeId {
        let product = self.mul_signed(x, y);
        let shifted = self.bounded_add(product, two, two);
        let scaled = self.bounded_positive_mul(shifted, product_bound, c64);
        self.sub(scaled, twice_product_bound)
    }

    fn bounded_rms_stage_literals(
        &mut self,
        input: &[NodeId],
        arity: usize,
        config: &BoundedRmsConfig,
    ) -> Result<[NodeId; 17], BoundedBuildError> {
        if input.len() != arity {
            return Err(BoundedBuildError::Shape);
        }
        for &node in input {
            if node as usize >= self.soa.len() {
                return Err(BoundedBuildError::InvalidNode(node));
            }
        }
        // Every stage fits in 16384 nodes, including its 17 owned constants.
        // constant() adds one pool entry and one node; all other builder
        // operations only grow nodes, so const_pool.len() <= soa.len().
        // This node-index margin therefore also protects the ConstId range.
        if self
            .soa
            .len()
            .checked_add(16384)
            .is_none_or(|size| size > u32::MAX as usize)
        {
            return Err(BoundedBuildError::SizeOverflow);
        }
        let mut literals = [0; 17];
        for (node, value) in literals.iter_mut().zip(config.fixed_literals()) {
            *node = self.constant(value);
        }
        Ok(literals)
    }

    /// Build one bounded RMS reduction step from `[total, x]`.
    ///
    /// The result is `total + square(normalize(x))`, using exactly the
    /// arithmetic of [`Self::rms_norm_bounded`]. Initialize `total` with
    /// `config.fixed_literals()[4]`, then apply this stage once per activation
    /// in input order. The stage adds the same 17 fixed literals and a constant
    /// number of nodes, independent of the configured width.
    ///
    /// Input shape, node IDs and index capacity are checked before mutation.
    /// Original inputs must satisfy [`BoundedRmsConfig::validate_values`]; every
    /// BML intermediate still requires finite-real evaluation checks.
    pub fn rms_norm_bounded_reduce(
        &mut self,
        input: &[NodeId],
        config: &BoundedRmsConfig,
    ) -> Result<NodeId, BoundedBuildError> {
        let [c64, two, twice_m, inv_m, ..] = self.bounded_rms_stage_literals(input, 2, config)?;
        let normalized = self.bounded_normalize(input[1], twice_m, inv_m, two, c64);
        let square = self.mul_signed(normalized, normalized);
        Ok(self.bounded_positive_add(input[0], square, two))
    }

    /// Build the shared bounded RMS preparation from `[total]`.
    ///
    /// `total` must be the result of the configured number of ordered
    /// [`Self::rms_norm_bounded_reduce`] steps. The roots are
    /// `[inverse_root, shift]`, consumed by [`Self::rms_norm_bounded_output`].
    /// This stage adds the same 17 fixed literals and a constant number of
    /// nodes, independent of width. Shape, node IDs and capacity errors leave
    /// the builder unchanged; finite-real checks remain required at execution.
    pub fn rms_norm_bounded_prepare(
        &mut self,
        input: &[NodeId],
        config: &BoundedRmsConfig,
    ) -> Result<[NodeId; 2], BoundedBuildError> {
        let [c64, two, _, _, _, inv_width, half, log_bound, twice_log_bound, inv_log_bound, k, ..] =
            self.bounded_rms_stage_literals(input, 1, config)?;
        let one = self.one();
        let variance = self.bounded_positive_mul(input[0], inv_width, c64);
        let log_variance = self.log2(variance);
        let normalized_log =
            self.bounded_normalize(log_variance, twice_log_bound, inv_log_bound, two, c64);
        let normalized_half = self.bounded_normalize(half, two, one, two, c64);
        let half_log = self.bounded_product_from_normalized(
            normalized_log,
            normalized_half,
            log_bound,
            twice_log_bound,
            two,
            c64,
        );
        let root = self.exp2(half_log);
        let log_root = self.log2(root);
        let minus_log_root = self.bounded_neg(log_root, c64);
        let inverse_root = self.exp2(minus_log_root);
        let shift = self.bounded_positive_mul(k, root, c64);
        Ok([inverse_root, shift])
    }

    /// Build one bounded RMS output from `[x, weight, inverse_root, shift]`.
    ///
    /// `x` is the original activation, not the normalized reduction operand;
    /// the last two operands come from [`Self::rms_norm_bounded_prepare`]. The
    /// result uses exactly the weighted output formula of
    /// [`Self::rms_norm_bounded`]. Weight nodes may reference external constants.
    /// Only the same 17 configuration literals are appended to the owned pool.
    ///
    /// Shape, node IDs and capacity are checked before mutation. Graph size is
    /// independent of width; runtime range and finite-real checks still apply.
    pub fn rms_norm_bounded_output(
        &mut self,
        input: &[NodeId],
        config: &BoundedRmsConfig,
    ) -> Result<NodeId, BoundedBuildError> {
        let [c64, two, twice_m, inv_m, _, _, _, _, _, _, k, twice_k, inv_k, twice_w, inv_w, product_bound, twice_product_bound] =
            self.bounded_rms_stage_literals(input, 4, config)?;
        let x = self.bounded_normalize(input[0], twice_m, inv_m, two, c64);
        let numerator = self.bounded_add(x, input[3], two);
        let ratio = self.bounded_positive_mul(numerator, input[2], c64);
        let normalized_x = self.sub(ratio, k);
        let unit_x = self.bounded_normalize(normalized_x, twice_k, inv_k, two, c64);
        let unit_weight = self.bounded_normalize(input[1], twice_w, inv_w, two, c64);
        Ok(self.bounded_product_from_normalized(
            unit_x,
            unit_weight,
            product_bound,
            twice_product_bound,
            two,
            c64,
        ))
    }

    /// Build RMSNorm with explicitly bounded arithmetic, separate from v1/v2.
    ///
    /// The mathematical result is `x[i]*weight[i]/sqrt(mean(x*x)+epsilon)`.
    /// Runtime inputs must satisfy [`BoundedRmsConfig::validate_values`].
    /// Every activation operation is a BML node; native arithmetic is limited
    /// to fixed configuration literals. This construction is approximate and
    /// may lose accuracy near zero; it does not promise native IEEE rounding,
    /// signed-zero behavior, or universal evaluability. Runtime execution must
    /// reject nonfinite, overflow, underflow and invalid-log intermediates.
    ///
    /// Invalid shapes, node IDs and index capacities are rejected before
    /// appending any nodes or constants. On success the new owned constants
    /// are exactly [`BoundedRmsConfig::fixed_literals`] in that order.
    pub fn rms_norm_bounded(
        &mut self,
        input: &[NodeId],
        weights: &[NodeId],
        config: &BoundedRmsConfig,
    ) -> Result<Vec<NodeId>, BoundedBuildError> {
        if input.len() != config.width || weights.len() != config.width {
            return Err(BoundedBuildError::Shape);
        }
        for &node in input.iter().chain(weights) {
            if node as usize >= self.soa.len() {
                return Err(BoundedBuildError::InvalidNode(node));
            }
        }
        // A conservative preflight bound also covers all fixed helper nodes.
        let maximum_nodes = config
            .width
            .checked_mul(4096)
            .and_then(|n| n.checked_add(16384));
        if maximum_nodes
            .and_then(|n| n.checked_add(self.soa.len()))
            .is_none_or(|n| n > u32::MAX as usize)
            || self.soa.const_pool.len() > u32::MAX as usize - 17
        {
            return Err(BoundedBuildError::SizeOverflow);
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(config.width)
            .map_err(|_| BoundedBuildError::Allocation)?;
        let literals: Vec<_> = config
            .fixed_literals()
            .into_iter()
            .map(|v| self.constant(v))
            .collect();
        let [c64, two, twice_m, inv_m, epsilon_sum, inv_width, half, log_bound, twice_log_bound, inv_log_bound, k, twice_k, inv_k, twice_w, inv_w, product_bound, twice_product_bound] =
            literals.as_slice()
        else {
            unreachable!()
        };
        let one = self.one();
        let normalized: Vec<_> = input
            .iter()
            .map(|&x| self.bounded_normalize(x, *twice_m, *inv_m, *two, *c64))
            .collect();
        let mut total = *epsilon_sum;
        for &x in &normalized {
            let square = self.mul_signed(x, x);
            total = self.bounded_positive_add(total, square, *two);
        }
        let variance = self.bounded_positive_mul(total, *inv_width, *c64);
        let log_variance = self.log2(variance);
        let normalized_log =
            self.bounded_normalize(log_variance, *twice_log_bound, *inv_log_bound, *two, *c64);
        let normalized_half = self.bounded_normalize(*half, *two, one, *two, *c64);
        let half_log = self.bounded_product_from_normalized(
            normalized_log,
            normalized_half,
            *log_bound,
            *twice_log_bound,
            *two,
            *c64,
        );
        let root = self.exp2(half_log);
        let log_root = self.log2(root);
        let minus_log_root = self.bounded_neg(log_root, *c64);
        let inverse_root = self.exp2(minus_log_root);
        let shift = self.bounded_positive_mul(*k, root, *c64);
        for (&x, &weight) in normalized.iter().zip(weights) {
            let numerator = self.bounded_add(x, shift, *two);
            let ratio = self.bounded_positive_mul(numerator, inverse_root, *c64);
            let normalized_x = self.sub(ratio, *k);
            let unit_x = self.bounded_normalize(normalized_x, *twice_k, *inv_k, *two, *c64);
            let unit_weight = self.bounded_normalize(weight, *twice_w, *inv_w, *two, *c64);
            output.push(self.bounded_product_from_normalized(
                unit_x,
                unit_weight,
                *product_bound,
                *twice_product_bound,
                *two,
                *c64,
            ));
        }
        Ok(output)
    }
}
