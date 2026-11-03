//! Independent arithmetic oracles and per-node real-domain checks for the
//! signed legacy AST expansions. This does not certify arbitrary f64 inputs.
use bml_domain::{BMLTransformer, NodeId, NodeKind};

fn finite_real_nodes(
    t: &BMLTransformer,
    root: NodeId,
    inputs: &[f64],
) -> Result<f64, &'static str> {
    let mut values = Vec::<f64>::new();
    for id in 0..t.soa().len() as NodeId {
        let n = t.soa().get(id);
        let value = match n.kind {
            NodeKind::One => 1.0,
            NodeKind::Var(index) => inputs[index as usize],
            NodeKind::Bml => {
                let left = values[n.left.unwrap() as usize];
                let right = values[n.right.unwrap() as usize];
                if right <= 0.0 {
                    return Err("nonpositive logarithm argument");
                }
                let power = left.exp2();
                if power == 0.0 || !power.is_finite() {
                    return Err("exponential range");
                }
                power - right.log2()
            }
            _ => panic!("expansion must use only One, Var and Bml"),
        };
        if !value.is_finite() {
            return Err("nonfinite intermediate");
        }
        values.push(value);
    }
    let nodes: Vec<_> = (0..t.soa().len() as NodeId)
        .map(|id| t.soa().get(id))
        .collect();
    let actual = bml_domain::ast::evaluate(
        &nodes,
        root,
        &bml_domain::ast::EvalContext::new(inputs, &[]),
    );
    assert_eq!(actual, values[root as usize]);
    Ok(actual)
}

#[test]
fn signed_products_match_fractional_oracles_with_finite_real_intermediates() {
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let y = t.var(1);
    let root = t.mul_signed(x, y);
    // Integer numerator products are exact and independent of the expansion.
    for a in -2..=2 {
        for b in -2..=2 {
            let expected = f64::from(a * b) / 4.0;
            let actual =
                finite_real_nodes(&t, root, &[f64::from(a) / 2.0, f64::from(b) / 2.0]).unwrap();
            assert!((actual - expected).abs() < 1e-9, "{a}/2 * {b}/2: {actual}");
        }
    }
}

#[test]
fn negation_and_positive_scaling_do_not_use_infinite_intermediates() {
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let root = t.neg_real(x);
    for numerator in -8..=8 {
        let x = f64::from(numerator) / 4.0;
        let actual = finite_real_nodes(&t, root, &[x]).unwrap();
        assert!((actual + x).abs() < 1e-10);
    }
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let y = t.var(1);
    let root = t.scale_signed(x, y);
    for numerator in -4..=4 {
        for denominator in 1..=4 {
            let x = f64::from(numerator) / 4.0;
            let y = f64::from(denominator) / 4.0;
            let actual = finite_real_nodes(&t, root, &[x, y]).unwrap();
            assert!((actual - x * y).abs() < 1e-9);
        }
    }
}

#[test]
fn signed_expansions_do_not_claim_native_range_or_invalid_scaling_domains() {
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let y = t.var(1);
    let root = t.mul_signed(x, y);
    assert!(finite_real_nodes(&t, root, &[1024.0, 1.0]).is_err());
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let y = t.var(1);
    let root = t.scale_signed(x, y);
    assert!(finite_real_nodes(&t, root, &[1.0, 0.0]).is_err());
    assert!(finite_real_nodes(&t, root, &[1.0, -1.0]).is_err());
}

#[test]
fn legacy_negation_keeps_its_ieee_range_contract() {
    let mut t = BMLTransformer::new();
    let x = t.var(0);
    let root = t.neg(x);
    let nodes: Vec<_> = (0..t.soa().len() as NodeId)
        .map(|id| t.soa().get(id))
        .collect();
    for x in [-100.0, -10.0, 0.0, 10.0, 100.0] {
        let actual =
            bml_domain::ast::evaluate(&nodes, root, &bml_domain::ast::EvalContext::new(&[x], &[]));
        assert_eq!(actual, -x);
    }
}
