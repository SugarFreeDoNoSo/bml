//! Building blocks BML: operaciones vectoriales expresadas como RPN con Loop.

use crate::rpn::{RpnOp, RpnProgram};

// ===========================================================================
// bml_dot: dot product como fold de FMA con Loop
// ===========================================================================

/// Programa RPN para un dot product: y = Σ x[i] * w[i] sobre n_in elementos.
///
/// Usa `VarIndexed` para indexar x y w por el contador del loop.
/// El resultado queda en el tope de la pila al terminar.
///
/// Cuerpo del loop (5 ops):
/// ```text
/// Dup                  // duplicar contador (necesitamos 2: x y w)
/// VarIndexed(x_base)   // pop contador, push x[counter]
/// Swap                 // poner contador en tope
/// VarIndexed(w_base)   // pop contador, push w[counter]
/// FMul                 // x[i] * w[i]
/// FAdd                 // acc + product
/// ```
pub fn bml_dot_program(x_base: u32, w_base: u32, n_in: u32) -> RpnProgram {
    let mut program = RpnProgram::new();

    // Acumulador inicial = 0
    program.push(RpnOp::Zero);

    // Cuerpo del loop
    let body = vec![
        RpnOp::Dup,
        RpnOp::VarIndexed { base: x_base },
        RpnOp::Swap,
        RpnOp::VarIndexed { base: w_base },
        RpnOp::FMul,
        RpnOp::FAdd,
    ];

    let body_len = body.len() as u32;
    program.push(RpnOp::Loop {
        count: n_in,
        body_len,
    });
    for op in body {
        program.push(op);
    }

    program
}

// ===========================================================================
// bml_matmul: matmul como map de dot products
// ===========================================================================

/// Programa RPN para un matmul completo: y = x · W.
///
/// Para cada columna j (0..n_out), calcula el dot product de x con W[:, j].
/// El resultado de cada columna se almacena via StoreResult.
///
/// Cada columna es independiente → paralelizable entre nodos.
pub fn bml_matmul_program(
    x_base: u32,
    w_base: u32,
    n_in: u32,
    n_out: u32,
    output_slot: u32,
) -> RpnProgram {
    let mut program = RpnProgram::new();

    for j in 0..n_out {
        // Acumulador = 0
        program.push(RpnOp::Zero);

        // w_col_base = w_base + j * n_in (offset de la columna j)
        let w_col_base = w_base + j * n_in;

        // Cuerpo del loop (mismo que bml_dot)
        let body = vec![
            RpnOp::Dup,
            RpnOp::VarIndexed { base: x_base },
            RpnOp::Swap,
            RpnOp::VarIndexed { base: w_col_base },
            RpnOp::FMul,
            RpnOp::FAdd,
        ];

        let body_len = body.len() as u32;
        program.push(RpnOp::Loop {
            count: n_in,
            body_len,
        });
        for op in body {
            program.push(op);
        }

        // Almacenar resultado: y[j] = tope de pila
        program.push(RpnOp::Const(j)); // offset = j
        program.push(RpnOp::StoreResult { slot: output_slot });
    }

    program
}

// ===========================================================================
// bml_rmsnorm: y[i] = x[i] * (1/rms) * w[i] como fold
// ===========================================================================

/// Apply a precomputed inverse RMS with native IEEE multiplication.
///
/// `y[i] = (x[i] * reciprocal_rms) * weight[i]`. `x_base` and `w_base`
/// are absolute `ResultBuffer` element bases; outputs go to `output_slot[i]`.
/// `recip_rms_const` retains its original meaning as an external constant index
/// for a precomputed `1/rms`; it is not an epsilon. This historical RPN helper,
/// like the neighboring dot-product helper, keeps its explicit native IEEE
/// arithmetic contract (including its nonfinite and signed-zero behavior).
/// It does not compute mean squares, epsilon or the root, and does not claim
/// a BML identity. [`bml_rmsnorm_full_program`] provides complete BML RMSNorm;
/// [`bml_rmsnorm_scale_program_checked`] provides explicit bounded BML scaling.
/// Outputs must not overwrite inputs or weights needed by a later iteration.
pub fn bml_rmsnorm_program(
    x_base: u32,
    w_base: u32,
    n_embd: u32,
    recip_rms_const: u32,
    output_slot: u32,
) -> RpnProgram {
    let body = vec![
        RpnOp::Dup,
        RpnOp::VarIndexed { base: x_base },
        RpnOp::Const(recip_rms_const),
        RpnOp::FMul,
        RpnOp::Pick { depth: 1 },
        RpnOp::VarIndexed { base: w_base },
        RpnOp::FMul,
        RpnOp::Swap,
        RpnOp::StoreResult { slot: output_slot },
    ];
    let mut program = RpnProgram::new();
    program.ops.push(RpnOp::Loop {
        count: n_embd,
        body_len: body.len() as u32,
    });
    program.ops.extend(body);
    program
}

/// Absolute operand declarations for applying a precomputed inverse RMS.
#[derive(Clone, Copy, Debug)]
pub struct RmsScaleBounds {
    /// Bound on each input element.
    pub input: f64,
    /// Positive upper bound on the precomputed inverse RMS.
    pub reciprocal: f64,
    /// Bound on each learned weight.
    pub weight: f64,
}

impl Default for RmsScaleBounds {
    fn default() -> Self {
        Self {
            input: 16.0,
            reciprocal: 1.0,
            weight: 16.0,
        }
    }
}

/// Explicit bounded application of a precomputed inverse RMS.
///
/// Bind `program().const_pool` as the execution context's constants after
/// checking the same inputs/weights with [`Self::validate_values`]. Buffer
/// capacity and disjoint output storage remain caller responsibilities.
#[derive(Clone, Debug)]
pub struct CheckedRmsScaleProgram {
    program: RpnProgram,
    bounds: RmsScaleBounds,
    width: usize,
    dag: bml_domain::NodeSoA,
    root: bml_domain::NodeId,
}

impl CheckedRmsScaleProgram {
    /// Bytecode and the complete owned constant context.
    pub fn program(&self) -> &RpnProgram {
        &self.program
    }

    /// Check shapes, declared ranges and every BML intermediate before writes.
    pub fn validate_values(&self, input: &[f64], weight: &[f64]) -> Result<(), String> {
        if input.len() != self.width || weight.len() != self.width {
            return Err("RMS scale operand shape mismatch".into());
        }
        for (index, (&x, &w)) in input.iter().zip(weight).enumerate() {
            if !x.is_finite()
                || x.abs() > self.bounds.input
                || !w.is_finite()
                || w.abs() > self.bounds.weight
            {
                return Err(format!("RMS scale operand bound at index {index}"));
            }
            validate_vector_dag(&self.dag, &[self.root], &[x, w])
                .map_err(|e| format!("RMS scale index {index}: {e}"))?;
        }
        Ok(())
    }
}

/// Compile a configurable application of a positive precomputed inverse RMS.
///
/// Both successive signed products must satisfy `BoundedOpConfig::product`;
/// accepted operand bounds alone do not guarantee finite intermediate results.
/// The reciprocal is stored as a fixed constant; no reduction is performed.
pub fn bml_rmsnorm_scale_program_checked(
    x_base: u32,
    w_base: u32,
    n_embd: u32,
    reciprocal_rms: f64,
    output_slot: u32,
    bounds: RmsScaleBounds,
) -> Result<CheckedRmsScaleProgram, String> {
    use bml_domain::{transformer::BoundedOpConfig, BMLTransformer};
    check_vector_read(x_base, n_embd)?;
    check_vector_read(w_base, n_embd)?;
    if !reciprocal_rms.is_finite() || reciprocal_rms <= 0.0 || reciprocal_rms > bounds.reciprocal {
        return Err("invalid precomputed inverse RMS".into());
    }
    let scale =
        BoundedOpConfig::product(bounds.input, bounds.reciprocal).map_err(|e| e.to_string())?;
    let learned = BoundedOpConfig::product(bounds.input * bounds.reciprocal, bounds.weight)
        .map_err(|e| e.to_string())?;
    let mut builder = BMLTransformer::new();
    let x = builder.var(0);
    let w = builder.var(1);
    let reciprocal = builder.constant(reciprocal_rms);
    let scaled = builder
        .operation_bounded(&[x, reciprocal], &[], &scale)
        .map_err(|e| e.to_string())?[0];
    let root = builder
        .operation_bounded(&[scaled, w], &[], &learned)
        .map_err(|e| e.to_string())?[0];
    let dag = builder.into_soa();
    let scalar = crate::linearize(&dag, root);
    let program = vector_output_program(&scalar, [x_base, w_base], n_embd, output_slot);
    Ok(CheckedRmsScaleProgram {
        program,
        bounds,
        width: n_embd as usize,
        dag,
        root,
    })
}

fn check_vector_read(base: u32, count: u32) -> Result<(), String> {
    if count > 0 {
        base.checked_add(count - 1)
            .ok_or("buffer address overflow")?;
    }
    Ok(())
}

/// Complete RMSNorm program with shared compiler implementation and preflight.
#[derive(Clone, Debug)]
pub struct CheckedRmsProgram {
    program: RpnProgram,
    compiled: crate::op_fragments::CheckedRmsFragment,
}

impl CheckedRmsProgram {
    /// Bytecode and complete owned constant context.
    pub fn program(&self) -> &RpnProgram {
        &self.program
    }

    /// Check actual inputs, learned weights and every BML intermediate.
    pub fn validate_values(&self, input: &[f64], weights: &[f64]) -> Result<(), String> {
        self.compiled.validate_values(input, weights)
    }
}

/// Compile complete RMSNorm, including mean squares, epsilon, root and weights.
///
/// This explicit API differs from [`bml_rmsnorm_program`], which applies an
/// already computed inverse RMS. It delegates to the shared compact RMS
/// compiler; no separate formula or native normalization is introduced.
/// Bind `program().const_pool` after preflighting the same values. Input and
/// weight bases are absolute buffer addresses; the output is a slot number.
pub fn bml_rmsnorm_full_program(
    input_base: u32,
    weight_base: u32,
    width: u32,
    epsilon: f64,
    output_slot: u32,
    bounds: crate::symbolic_kernels::RmsBounds,
) -> Result<CheckedRmsProgram, String> {
    let compiled = crate::op_fragments::compile_rmsnorm_fragment_checked(
        "rmsnorm",
        input_base,
        output_slot,
        weight_base,
        width,
        epsilon,
        bounds,
    )?;
    let program = RpnProgram {
        ops: compiled.operation().fragment.ops.clone(),
        const_pool: compiled.constants().to_vec(),
    };
    Ok(CheckedRmsProgram { program, compiled })
}

fn validate_vector_dag(
    dag: &bml_domain::NodeSoA,
    roots: &[u32],
    inputs: &[f64],
) -> Result<(), String> {
    use bml_domain::{symbolic::Basis, NodeKind};
    let mut values = vec![0.0; dag.len()];
    for (index, kind) in dag.kinds.iter().enumerate() {
        values[index] = match *kind {
            NodeKind::One => 1.0,
            NodeKind::Zero => 0.0,
            NodeKind::Var(id) => inputs[id as usize],
            NodeKind::Const(id) => dag.const_pool[id as usize],
            NodeKind::Bml => Basis::Bml
                .apply_real(
                    values[dag.lefts[index] as usize],
                    values[dag.rights[index] as usize],
                )
                .map_err(|e| format!("BML node {index}: {e}"))?,
        };
    }
    if roots.iter().any(|&root| !values[root as usize].is_finite()) {
        return Err("nonfinite vector output".into());
    }
    Ok(())
}

fn indexed_scalar_body(
    scalar: &RpnProgram,
    mut variable: impl FnMut(u32, u32) -> Vec<RpnOp>,
) -> Vec<RpnOp> {
    let mut body = Vec::with_capacity(scalar.ops.len() + 8);
    let mut depth = 0;
    for &op in &scalar.ops {
        match op {
            RpnOp::Var(id) => {
                body.extend(variable(id, depth));
                depth += 1;
            }
            RpnOp::One | RpnOp::Zero | RpnOp::Const(_) | RpnOp::Pick { .. } | RpnOp::Dup => {
                body.push(op);
                depth += 1;
            }
            RpnOp::Bml | RpnOp::Drop => {
                body.push(op);
                depth -= 1;
            }
            RpnOp::Swap => body.push(op),
            _ => unreachable!("scalar BML DAG only"),
        }
    }
    assert_eq!(depth, 1);
    body
}

fn vector_output_program(
    scalar: &RpnProgram,
    bases: [u32; 2],
    count: u32,
    slot: u32,
) -> RpnProgram {
    let mut body = indexed_scalar_body(scalar, |id, depth| {
        vec![
            RpnOp::Pick { depth },
            RpnOp::VarIndexed {
                base: bases[id as usize],
            },
        ]
    });
    body.extend([RpnOp::Swap, RpnOp::StoreResult { slot }]);
    let mut program = RpnProgram::new();
    program.const_pool = scalar.const_pool.clone();
    program.ops.push(RpnOp::Loop {
        count,
        body_len: body.len() as u32,
    });
    program.ops.extend(body);
    program
}

// ===========================================================================
// bml_swiglu: y = gate * sigmoid(gate) * up
// ===========================================================================

/// Standard SwiGLU, `y[i] = gate[i] * sigmoid(gate[i]) * up[i]`.
///
/// `gate_base` and `up_base` are absolute indices into `ResultBuffer`, whereas
/// `output_slot` identifies a slot. Each output is written at that slot's
/// element index; no reduction or temporary value remains on the stack.
/// Outputs must not overwrite gate/up values needed by a later iteration.
/// The caller supplies `log2(e)` at `const_log2e` and `1.0` at `const_one` in
/// its external constant context. The historical `const_1_7` argument is
/// obsolete and deliberately unused: standard SwiGLU has no such scale.
///
/// The loop reuses the signed bounded SwiGLU identity with gate/up declarations
/// of absolute value at most 16. Bytecode storage is independent of vector
/// width. Every activation calculation and fixed literal construction uses
/// BML; no additional external constants are required. As with the other BML
/// builders, this is a partial numerical contract: callers must check input
/// bounds and finite results, and it does not promise native IEEE range or
/// signed-zero behavior. No activation is clamped or evaluated natively.
pub fn bml_swiglu_program(
    gate_base: u32,
    up_base: u32,
    n_embd: u32,
    _const_1_7: u32,
    const_log2e: u32,
    const_one: u32,
    output_slot: u32,
) -> RpnProgram {
    let scalar = swiglu_scalar_with_external_constants(const_log2e, const_one);

    // The loop's element index stays below the scalar evaluation stack. Each
    // variable loads through a copy of that index; local Pick depths remain
    // valid because all other stack effects match the scalar program exactly.
    let mut body = Vec::with_capacity(scalar.ops.len() + 4);
    let mut depth = 0;
    for op in scalar.ops {
        match op {
            RpnOp::Var(id) => {
                let base = match id {
                    0 => gate_base,
                    1 => up_base,
                    _ => unreachable!("SwiGLU has two scalar inputs"),
                };
                body.push(RpnOp::Pick { depth });
                body.push(RpnOp::VarIndexed { base });
                depth += 1;
            }
            RpnOp::Const(_) => {
                body.push(op);
                depth += 1;
            }
            RpnOp::Bml => {
                body.push(op);
                depth -= 1;
            }
            RpnOp::Pick { .. } | RpnOp::Dup => {
                body.push(op);
                depth += 1;
            }
            RpnOp::Drop => {
                body.push(op);
                depth -= 1;
            }
            RpnOp::Swap => body.push(op),
            _ => unreachable!("SwiGLU lowering uses only BML and stack operations"),
        }
    }
    assert_eq!(depth, 1, "scalar lowering leaves exactly one result");
    body.push(RpnOp::Swap);
    body.push(RpnOp::StoreResult { slot: output_slot });
    let mut program = RpnProgram::new();
    program.push(RpnOp::Loop {
        count: n_embd,
        body_len: body.len() as u32,
    });
    program.ops.extend(body);
    program
}

// Rebind a bounded operation's configuration literals to existing BML terms.
// This retains the domain implementation's arithmetic while preserving this
// legacy API's external constant namespace (only log2(e) and one).
fn append_swiglu_bounded_operation(
    builder: &mut bml_domain::BMLTransformer,
    inputs: [bml_domain::NodeId; 2],
    config: bml_domain::transformer::BoundedOpConfig,
    literals: &[(f64, bml_domain::NodeId)],
) -> bml_domain::NodeId {
    use bml_domain::{BMLTransformer, NodeKind};
    let mut template = BMLTransformer::new();
    let x = template.var(0);
    let y = template.var(1);
    let root = template
        .operation_bounded(&[x, y], &[], &config)
        .expect("fixed valid bounded operation")[0];
    let soa = template.into_soa();
    let literal = |value: f64| {
        literals
            .iter()
            .find(|(candidate, _)| candidate.to_bits() == value.to_bits())
            .expect("every configured literal has an explicit BML construction")
            .1
    };
    let mut mapped = Vec::with_capacity(soa.len());
    for (index, kind) in soa.kinds.iter().enumerate() {
        let node = match kind {
            NodeKind::One => literal(1.0),
            NodeKind::Zero => literal(0.0),
            NodeKind::Var(id) => inputs[*id as usize],
            NodeKind::Const(id) => literal(soa.const_pool[*id as usize]),
            NodeKind::Bml => builder.bml(
                mapped[soa.lefts[index] as usize],
                mapped[soa.rights[index] as usize],
            ),
        };
        mapped.push(node);
    }
    mapped[root as usize]
}

fn dyadic_literals(
    builder: &mut bml_domain::BMLTransformer,
    one: bml_domain::NodeId,
) -> Vec<(f64, bml_domain::NodeId)> {
    // Exact dyadic terms for the fixed bound 16. Every intermediate is finite.
    let two = builder.bml(one, one);
    let four = builder.bml(two, one);
    let zero = builder.bml(one, four);
    let three = builder.bml(two, two);
    let eight = builder.bml(three, one);
    let five = builder.bml(three, eight);
    let six = builder.bml(three, four);
    let seven = builder.bml(three, two);
    let sixteen = builder.bml(four, one);
    let thirty_two = builder.bml(five, one);
    let sixty_four = builder.bml(six, one);
    let one_twenty_eight = builder.bml(seven, one);
    let two_fifty_six = builder.bml(eight, one);
    let nine = builder.bml(four, one_twenty_eight);
    let five_twelve = builder.bml(nine, one);
    let minus_one = builder.bml(one, eight);
    let half = builder.bml(minus_one, one);
    let minus_four = builder.bml(one, sixty_four);
    let one_sixteenth = builder.bml(minus_four, one);
    let minus_five = builder.bml(one, one_twenty_eight);
    let one_thirty_second = builder.bml(minus_five, one);
    let two_to_32 = builder.bml(thirty_two, one);
    let minus_sixteen = builder.bml(four, two_to_32);
    vec![
        (0.0, zero),
        (1.0, one),
        (2.0, two),
        (4.0, four),
        (16.0, sixteen),
        (32.0, thirty_two),
        (64.0, sixty_four),
        (128.0, one_twenty_eight),
        (256.0, two_fifty_six),
        (512.0, five_twelve),
        (0.5, half),
        (1.0 / 16.0, one_sixteenth),
        (1.0 / 32.0, one_thirty_second),
        (-16.0, minus_sixteen),
    ]
}

fn swiglu_scalar_with_external_constants(const_log2e: u32, const_one: u32) -> RpnProgram {
    use bml_domain::{transformer::BoundedOpConfig, BMLTransformer};
    let mut builder = BMLTransformer::new();
    let one = builder.external_constant(const_one);
    let log2e = builder.external_constant(const_log2e);
    let gate = builder.var(0);
    let up = builder.var(1);

    let mut literals = dyadic_literals(&mut builder, one);
    let minus_sixteen = literals
        .iter()
        .find(|(value, _)| *value == -16.0)
        .unwrap()
        .1;
    literals.push((std::f64::consts::LOG2_E, log2e));
    let exponent = append_swiglu_bounded_operation(
        &mut builder,
        [minus_sixteen, log2e],
        BoundedOpConfig::product(16.0, 2.0).expect("fixed product bounds"),
        &literals,
    );
    let minimum_exp = builder.bml(exponent, one);
    literals.push(((-16.0_f64).exp(), minimum_exp));
    let root = append_swiglu_bounded_operation(
        &mut builder,
        [gate, up],
        BoundedOpConfig::swiglu(16.0, 16.0).expect("fixed SwiGLU bounds"),
        &literals,
    );
    crate::linearize(&builder.into_soa(), root)
}

// ===========================================================================
// bml_rope: rotación de un par (even, odd) con cos/sin precomputados
// ===========================================================================

/// Rotate one adjacent coordinate pair and write both components to a slot.
///
/// This legacy API retains its original coordinate namespace: `x_even_base`
/// and `x_odd_base` are **EvalContext input IDs**, not buffer offsets. `cos_const`
/// and `sin_const` refer to external constants. The result is written to
/// `output_slot[0..2]` and leaves the existing stack unchanged.
/// The signed finite-real BML profile declares `|x|,|y| <= 16` and
/// `|cosine|,|sine| <= 1`; no logarithm of zero or infinite negation is emitted.
/// Use [`bml_rope_vector_program`] for all pairs, configurable bounds, explicit
/// frequencies/position and absolute input addresses in the result buffer.
pub fn bml_rope_program(
    x_even_base: u32,
    x_odd_base: u32,
    cos_const: u32,
    sin_const: u32,
    output_slot: u32,
) -> RpnProgram {
    use bml_domain::{BMLTransformer, NodeKind};
    let (dag, roots) = rope_pair_dag(16.0).expect("fixed RoPE bounds");
    let mut builder = BMLTransformer::new();
    let values = [
        builder.var(x_even_base),
        builder.var(x_odd_base),
        builder.external_constant(cos_const),
        builder.external_constant(sin_const),
    ];
    let one = builder.one();
    let literals = dyadic_literals(&mut builder, one);
    let mut mapped = Vec::with_capacity(dag.len());
    for (index, kind) in dag.kinds.iter().enumerate() {
        let literal = |value: f64| {
            literals
                .iter()
                .find(|(v, _)| v.to_bits() == value.to_bits())
                .expect("fixed RoPE literal")
                .1
        };
        mapped.push(match *kind {
            NodeKind::One => one,
            NodeKind::Zero => literal(0.0),
            NodeKind::Var(id) => values[id as usize],
            NodeKind::Const(id) => literal(dag.const_pool[id as usize]),
            NodeKind::Bml => builder.bml(
                mapped[dag.lefts[index] as usize],
                mapped[dag.rights[index] as usize],
            ),
        });
    }
    let converted = builder.into_soa();
    let mut program = RpnProgram::new();
    for (component, root) in roots.into_iter().enumerate() {
        program
            .ops
            .extend(crate::linearize(&converted, mapped[root as usize]).ops);
        program.ops.push(if component == 0 {
            RpnOp::Zero
        } else {
            RpnOp::One
        });
        program.ops.push(RpnOp::StoreResult { slot: output_slot });
    }
    program
}

/// Layout, position and numerical declaration for adjacent-pair RoPE.
#[derive(Clone, Copy, Debug)]
pub struct RopeVectorConfig {
    /// Number of independently rotated heads.
    pub heads: u32,
    /// Even coordinates per head; frequencies restart at each head.
    pub head_dim: u32,
    /// Absolute token position used to precompute coefficients.
    pub position: u32,
    /// Positive finite frequency base (typically 10,000).
    pub frequency_base: f64,
    /// Inclusive absolute input bound accepted by `BoundedOpConfig::rope`.
    pub input_bound: f64,
}

/// Complete vector RoPE with owned coefficients and finite-real preflight.
///
/// Each head is contiguous and its pairs are `(0,1), (2,3), ...`. The program
/// reads absolute `ResultBuffer` input addresses and writes all components to
/// the output slot. Bind its complete owned constants after checking the same
/// input values. Output storage must not overwrite inputs needed by later pairs.
#[derive(Clone, Debug)]
pub struct CheckedRopeProgram {
    program: RpnProgram,
    config: RopeVectorConfig,
    coefficients: Vec<[f64; 2]>,
    dag: bml_domain::NodeSoA,
    roots: [u32; 2],
}

impl CheckedRopeProgram {
    /// Bytecode and complete constant pool (including precomputed coefficients).
    pub fn program(&self) -> &RpnProgram {
        &self.program
    }
    /// Layout and numerical configuration used at compilation.
    pub fn config(&self) -> RopeVectorConfig {
        self.config
    }
    /// Preflight all components, coefficients and BML intermediates.
    pub fn validate_values(&self, input: &[f64]) -> Result<(), String> {
        if input.len() != 2 * self.coefficients.len() {
            return Err("RoPE input shape mismatch".into());
        }
        for (pair, (values, coefficients)) in input
            .as_chunks::<2>()
            .0
            .iter()
            .zip(&self.coefficients)
            .enumerate()
        {
            if values
                .iter()
                .any(|v| !v.is_finite() || v.abs() > self.config.input_bound)
            {
                return Err(format!("RoPE input bound at pair {pair}"));
            }
            validate_vector_dag(
                &self.dag,
                &self.roots,
                &[values[0], values[1], coefficients[0], coefficients[1]],
            )
            .map_err(|e| format!("RoPE pair {pair}: {e}"))?;
        }
        Ok(())
    }
}

fn rope_pair_dag(bound: f64) -> Result<(bml_domain::NodeSoA, [u32; 2]), String> {
    use bml_domain::{transformer::BoundedOpConfig, BMLTransformer};
    let config = BoundedOpConfig::rope(bound).map_err(|e| e.to_string())?;
    let mut builder = BMLTransformer::new();
    let inputs = [builder.var(0), builder.var(1)];
    let coefficients = [builder.var(2), builder.var(3)];
    let output = builder
        .operation_bounded(&inputs, &coefficients, &config)
        .map_err(|e| e.to_string())?;
    Ok((builder.into_soa(), [output[0], output[1]]))
}

/// Compile all adjacent RoPE pairs with O(1) bytecode and O(width) coefficients.
///
/// Coefficients are precomputed from `angle = position * base^(-2*pair/head_dim)`;
/// all arithmetic on model activations uses BML. Width is explicitly capped at
/// 65,536 coordinates, dimensions must be positive/even, and all read addresses
/// must fit u32. Accepted bounds still require actual-value finite-real preflight.
pub fn bml_rope_vector_program(
    input_base: u32,
    output_slot: u32,
    config: RopeVectorConfig,
) -> Result<CheckedRopeProgram, String> {
    let width = config
        .heads
        .checked_mul(config.head_dim)
        .ok_or("RoPE dimensions overflow")?;
    if config.heads == 0
        || config.head_dim == 0
        || !config.head_dim.is_multiple_of(2)
        || width > 65536
    {
        return Err(
            "RoPE needs positive even head dimensions and at most 65536 coordinates".into(),
        );
    }
    if !config.frequency_base.is_finite() || config.frequency_base <= 0.0 {
        return Err("invalid RoPE frequency base".into());
    }
    check_vector_read(input_base, width)?;
    let (dag, roots) = rope_pair_dag(config.input_bound)?;
    let mut coefficients = Vec::with_capacity(width as usize / 2);
    for _ in 0..config.heads {
        for pair in 0..config.head_dim / 2 {
            let angle = if config.position == 0 {
                0.0
            } else {
                f64::from(config.position)
                    * config
                        .frequency_base
                        .powf(-2.0 * f64::from(pair) / f64::from(config.head_dim))
            };
            let (sin, cos) = angle.sin_cos();
            if !sin.is_finite() || !cos.is_finite() {
                return Err("nonfinite RoPE coefficient".into());
            }
            coefficients.push([cos, sin]);
        }
    }
    let two_id = dag.const_pool.len() as u32;
    let coefficient_base = two_id + 1;
    let mut program = RpnProgram::new();
    program.const_pool = dag.const_pool.clone();
    program.const_pool.push(2.0);
    program
        .const_pool
        .extend(coefficients.iter().flatten().copied());
    let mut body = Vec::new();
    for (component, &root) in roots.iter().enumerate() {
        let scalar = crate::linearize(&dag, root);
        body.extend(indexed_scalar_body(&scalar, |id, depth| {
            let mut load = vec![RpnOp::Pick { depth }, RpnOp::Const(two_id), RpnOp::FMul];
            if !id.is_multiple_of(2) {
                load.extend([RpnOp::One, RpnOp::FAdd]);
            }
            load.push(if id < 2 {
                RpnOp::VarIndexed { base: input_base }
            } else {
                RpnOp::ContextConstIndexed {
                    base: coefficient_base,
                }
            });
            load
        }));
        body.extend([RpnOp::Pick { depth: 1 }, RpnOp::Const(two_id), RpnOp::FMul]);
        if component == 1 {
            body.extend([RpnOp::One, RpnOp::FAdd]);
        }
        body.push(RpnOp::StoreResult { slot: output_slot });
    }
    body.push(RpnOp::Drop);
    program.ops.push(RpnOp::Loop {
        count: width / 2,
        body_len: body.len() as u32,
    });
    program.ops.extend(body);
    Ok(CheckedRopeProgram {
        program,
        config,
        coefficients,
        dag,
        roots,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bml_dot_program_structure() {
        let prog = bml_dot_program(0, 100, 4);
        assert!(prog.len() > 6);
        assert!(prog
            .ops
            .iter()
            .any(|op| matches!(op, RpnOp::Loop { count: 4, .. })));
    }

    #[test]
    fn bml_matmul_program_structure() {
        let prog = bml_matmul_program(0, 100, 4, 3, 0);
        let n_stores = prog
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::StoreResult { .. }))
            .count();
        assert_eq!(n_stores, 3);
        let n_loops = prog
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::Loop { .. }))
            .count();
        assert_eq!(n_loops, 3);
    }

    #[test]
    fn bml_matmul_bytecode_size() {
        let prog = bml_matmul_program(0, 100, 2048, 2048, 0);
        let total_bytes: usize = prog
            .ops
            .iter()
            .map(|op| match op {
                RpnOp::One
                | RpnOp::Zero
                | RpnOp::Bml
                | RpnOp::Dup
                | RpnOp::FAdd
                | RpnOp::FMul
                | RpnOp::Drop
                | RpnOp::Swap => 1,
                RpnOp::Var(_)
                | RpnOp::Const(_)
                | RpnOp::VarIndexed { .. }
                | RpnOp::ContextVarIndexed { .. }
                | RpnOp::ContextConstIndexed { .. }
                | RpnOp::StoreResult { .. }
                | RpnOp::Pick { .. } => 5,
                RpnOp::Loop { .. } => 9,
            })
            .sum();
        println!(
            "Matmul 2048x2048: {} ops, {} bytes ({:.1} KB)",
            prog.len(),
            total_bytes,
            total_bytes as f64 / 1024.0
        );
        assert!(total_bytes < 80_000);
    }

    #[test]
    fn bml_rmsnorm_program_structure() {
        let prog = bml_rmsnorm_program(0, 100, 4, 200, 0);
        assert!(prog
            .ops
            .iter()
            .any(|op| matches!(op, RpnOp::Loop { count: 4, .. })));
    }

    #[test]
    fn bml_rope_uses_bml() {
        let prog = bml_rope_program(0, 1, 100, 101, 0);
        let n_bml = prog
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::Bml))
            .count();
        assert!(n_bml > 0, "RoPE debe usar Bml para negacion BML pura");
        let n_fmul = prog
            .ops
            .iter()
            .filter(|op| matches!(op, RpnOp::FMul))
            .count();
        assert_eq!(n_fmul, 0);
        assert_eq!(
            prog.ops
                .iter()
                .filter(|op| matches!(op, RpnOp::StoreResult { .. }))
                .count(),
            2
        );
    }
}
