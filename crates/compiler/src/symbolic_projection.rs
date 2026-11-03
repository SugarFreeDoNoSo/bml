//! Bounded BML-only row templates with external Const weight references.
//!
//! This is separate from the native IEEE projection profile. Inputs address
//! Var indices and a decoded weight row addresses Const indices. The serialized
//! program contains no weights. Signed BML expansion remains numerically
//! partial, even when native multiplication and addition would be finite.
use crate::{linearize, RpnProgram};
use bml_domain::BMLTransformer;

/// A row template and its declared external execution dimensions.
pub struct SymbolicProjection {
    /// BML arithmetic and stack instructions, with an empty owned constant pool.
    pub program: RpnProgram,
    /// Number of input values and external constants required on each execution.
    pub columns: usize,
    /// Number of nodes allocated before linearization.
    pub nodes: usize,
}

/// Build a nonempty signed dot product using BMLTransformer under a node budget.
///
/// An exact structural size preflight runs before allocating the row DAG. The
/// budget includes input and constant-reference nodes. This unrolled template
/// has linear storage/work in the row width, not the native loop's constant
/// bytecode size. Runtime opcode, stack and step limits must also be checked.
pub fn compile_row(columns: usize, max_nodes: usize) -> Result<SymbolicProjection, String> {
    if columns == 0 {
        return Err("symbolic projection requires a nonempty row".into());
    }
    // Measure the actual expansion rules rather than maintaining a magic size
    // that can become stale when the BML identities change.
    let mut probe = BMLTransformer::new();
    let x = probe.var(0);
    let y = probe.var(1);
    let start = probe.soa().len();
    probe.mul_signed(x, y);
    let product_nodes = probe.soa().len() - start;
    let start = probe.soa().len();
    probe.add(x, y);
    let add_nodes = probe.soa().len() - start;
    let nodes = columns
        .checked_mul(product_nodes + 2)
        .and_then(|n| {
            (columns - 1)
                .checked_mul(add_nodes)
                .and_then(|s| n.checked_add(s))
        })
        .ok_or("symbolic projection size overflow")?;
    if nodes > max_nodes || nodes > u32::MAX as usize {
        return Err("symbolic projection exceeds its node budget".into());
    }
    let mut builder = BMLTransformer::with_capacity(nodes);
    let mut sum = None;
    for column in 0..columns {
        let input = builder.var(column as u32);
        let weight = builder.external_constant(column as u32);
        let term = builder.mul_signed(input, weight);
        sum = Some(match sum {
            None => term,
            Some(previous) => builder.add(previous, term),
        });
    }
    debug_assert_eq!(builder.soa().len(), nodes);
    let program = linearize(
        &builder.into_soa(),
        sum.expect("nonempty row checked above"),
    );
    debug_assert!(program.const_pool.is_empty());
    Ok(SymbolicProjection {
        program,
        columns,
        nodes,
    })
}
