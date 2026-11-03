//! Tensor-operation DAG construction in the legacy BML basis.
//!
//! These are mathematical expansions, not native tensor kernels. Valid shapes
//! do not imply numerically evaluable graphs: intermediate f64 overflow,
//! underflow and cancellation can occur even when the native result is finite.
use super::BMLTransformer;
use crate::NodeId;

/// A structural error detected before appending an operation to the graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OperationBuildError {
    /// Operand lengths do not match the requested operation.
    Shape,
    /// A product of dimensions cannot be represented by usize.
    SizeOverflow,
    /// The output allocation could not be reserved.
    Allocation,
    /// An operand is not a node in this builder.
    InvalidNode(NodeId),
    /// RMS normalization requires a finite, strictly positive epsilon.
    InvalidEpsilon,
}

impl std::fmt::Display for OperationBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape => f.write_str("incompatible BML operation shapes"),
            Self::SizeOverflow => f.write_str("BML operation dimensions overflow"),
            Self::Allocation => f.write_str("cannot reserve BML operation outputs"),
            Self::InvalidNode(id) => write!(f, "unknown BML operand node {id}"),
            Self::InvalidEpsilon => f.write_str("RMS epsilon must be finite and positive"),
        }
    }
}
impl std::error::Error for OperationBuildError {}

impl BMLTransformer {
    fn check_operands(&self, operands: &[NodeId]) -> Result<(), OperationBuildError> {
        for &id in operands {
            if id as usize >= self.soa.len() {
                return Err(OperationBuildError::InvalidNode(id));
            }
        }
        Ok(())
    }

    fn sum_nodes(&mut self, values: &[NodeId]) -> NodeId {
        let Some((&first, rest)) = values.split_first() else {
            return self.constant(0.0);
        };
        rest.iter().fold(first, |sum, &value| self.add(sum, value))
    }

    fn reciprocal_real(&mut self, positive: NodeId) -> NodeId {
        let log = self.log2(positive);
        let negative = self.neg_real(log);
        self.exp2(negative)
    }

    fn natural_exp(&mut self, value: NodeId, log2_e: NodeId) -> NodeId {
        let exponent = self.scale_signed(value, log2_e);
        self.exp2(exponent)
    }

    /// Translate row-major `(rows, inner) * (inner, columns)` to BML nodes.
    ///
    /// Products support signed operands. Accumulation follows increasing inner
    /// index; empty contractions produce zero. This does not promise native
    /// IEEE rounding, numerical range or evaluation cost.
    pub fn matmul(
        &mut self,
        left: &[NodeId],
        right: &[NodeId],
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<Vec<NodeId>, OperationBuildError> {
        let left_len = rows
            .checked_mul(inner)
            .ok_or(OperationBuildError::SizeOverflow)?;
        let right_len = inner
            .checked_mul(columns)
            .ok_or(OperationBuildError::SizeOverflow)?;
        let output_len = rows
            .checked_mul(columns)
            .ok_or(OperationBuildError::SizeOverflow)?;
        if left.len() != left_len || right.len() != right_len {
            return Err(OperationBuildError::Shape);
        }
        self.check_operands(left)?;
        self.check_operands(right)?;
        if output_len == 0 {
            return Ok(Vec::new());
        }
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| OperationBuildError::Allocation)?;
        for row in 0..rows {
            for column in 0..columns {
                let mut terms = Vec::with_capacity(inner);
                for k in 0..inner {
                    terms.push(self.mul_signed(left[row * inner + k], right[k * columns + column]));
                }
                output.push(self.sum_nodes(&terms));
            }
        }
        Ok(output)
    }

    /// Translate `x[i] * weight[i] / sqrt(mean(x*x) + epsilon)` to BML.
    ///
    /// Inputs must be nonempty. Signed activations and weights are permitted
    /// algebraically; numerical evaluation remains partial. The square root
    /// is expanded as `exp2(0.5 * log2(variance))`, including variance below one.
    pub fn rms_norm(
        &mut self,
        input: &[NodeId],
        weights: &[NodeId],
        epsilon: f64,
    ) -> Result<Vec<NodeId>, OperationBuildError> {
        if input.is_empty() || input.len() != weights.len() {
            return Err(OperationBuildError::Shape);
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(OperationBuildError::InvalidEpsilon);
        }
        self.check_operands(input)?;
        self.check_operands(weights)?;
        let squares: Vec<_> = input.iter().map(|&x| self.mul_signed(x, x)).collect();
        let sum = self.sum_nodes(&squares);
        let inverse_width = self.constant(1.0 / input.len() as f64);
        let mean = self.scale_signed(sum, inverse_width);
        let eps = self.constant(epsilon);
        let variance = self.add(mean, eps);
        let log_variance = self.log2(variance);
        let half = self.constant(0.5);
        let half_log = self.scale_signed(log_variance, half);
        let root = self.exp2(half_log);
        let inverse_root = self.reciprocal_real(root);
        let mut output = Vec::with_capacity(input.len());
        for (&x, &weight) in input.iter().zip(weights) {
            let normalized = self.scale_signed(x, inverse_root);
            output.push(self.mul_signed(normalized, weight));
        }
        Ok(output)
    }

    /// Translate adjacent-pair RoPE using supplied cosine/sine operand nodes.
    ///
    /// There must be one cosine and sine per pair. Trigonometric constants may
    /// be precomputed for the position; this operation does not synthesize sin
    /// or cos in the real BML basis. Every pair is rotated, in input order.
    pub fn rope(
        &mut self,
        input: &[NodeId],
        cosines: &[NodeId],
        sines: &[NodeId],
    ) -> Result<Vec<NodeId>, OperationBuildError> {
        if !input.len().is_multiple_of(2)
            || cosines.len() != input.len() / 2
            || sines.len() != cosines.len()
        {
            return Err(OperationBuildError::Shape);
        }
        self.check_operands(input)?;
        self.check_operands(cosines)?;
        self.check_operands(sines)?;
        let mut output = Vec::with_capacity(input.len());
        for ((pair, &cos), &sin) in input.as_chunks::<2>().0.iter().zip(cosines).zip(sines) {
            let x_cos = self.mul_signed(pair[0], cos);
            let y_sin = self.mul_signed(pair[1], sin);
            let minus_y_sin = self.neg_real(y_sin);
            output.push(self.add(x_cos, minus_y_sin));
            let x_sin = self.mul_signed(pair[0], sin);
            let y_cos = self.mul_signed(pair[1], cos);
            output.push(self.add(x_sin, y_cos));
        }
        Ok(output)
    }

    /// Translate natural-exponential softmax using exp2, sums and division.
    ///
    /// Unlike native stabilized softmax, this graph has no max/comparison node.
    /// Callers can supply already shifted logits; the represented real function
    /// is invariant to a common shift, but its finite evaluation range is not.
    pub fn softmax(&mut self, input: &[NodeId]) -> Result<Vec<NodeId>, OperationBuildError> {
        if input.is_empty() {
            return Err(OperationBuildError::Shape);
        }
        self.check_operands(input)?;
        let log2_e = self.constant(std::f64::consts::LOG2_E);
        let exps: Vec<_> = input.iter().map(|&x| self.natural_exp(x, log2_e)).collect();
        let sum = self.sum_nodes(&exps);
        let inverse = self.reciprocal_real(sum);
        Ok(exps.into_iter().map(|x| self.mul(x, inverse)).collect())
    }

    /// Translate standard Llama SwiGLU: `gate * sigmoid(gate) * up`.
    ///
    /// This uses natural exp, not a 1.7-scaled sigmoid approximation. Both
    /// branches may be signed; input lengths must agree.
    pub fn swiglu(
        &mut self,
        gate: &[NodeId],
        up: &[NodeId],
    ) -> Result<Vec<NodeId>, OperationBuildError> {
        if gate.len() != up.len() {
            return Err(OperationBuildError::Shape);
        }
        self.check_operands(gate)?;
        self.check_operands(up)?;
        let log2_e = self.constant(std::f64::consts::LOG2_E);
        let one = self.one();
        let mut output = Vec::with_capacity(gate.len());
        for (&g, &u) in gate.iter().zip(up) {
            let negative = self.neg_real(g);
            let exp = self.natural_exp(negative, log2_e);
            let denominator = self.add(one, exp);
            let sigmoid = self.reciprocal_real(denominator);
            let silu = self.scale_signed(g, sigmoid);
            output.push(self.mul_signed(silu, u));
        }
        Ok(output)
    }
}
