use bml_domain::symbolic::{
    complex::{Complex64, EvaluationBudget, LogBranch},
    synthesis::{synthesize, Interpretation, StopReason, SynthesisBudget, SynthesisError},
    Basis, ClosedTerm,
};

fn budget(nodes: usize, candidates: usize) -> SynthesisBudget {
    SynthesisBudget {
        max_nodes: nodes,
        max_candidates: candidates,
    }
}

#[test]
fn synthesis_finds_small_pure_trees_in_both_bases_and_explicit_complex_branch() {
    let pair = ClosedTerm::op(ClosedTerm::One, ClosedTerm::One);
    for (basis, target) in [(Basis::Bml, 2.0), (Basis::Eml, std::f64::consts::E)] {
        let result = synthesize(
            Complex64::real(target),
            Interpretation::Real(basis),
            1e-14,
            budget(3, 2),
        )
        .unwrap();
        assert_eq!(result.stop, StopReason::ToleranceMet);
        assert_eq!(result.attempted, 2);
        let candidate = result.best.unwrap();
        assert_eq!(candidate.term, pair);
        assert_eq!(candidate.nodes, 3);
        assert!(candidate.error <= 1e-14);
    }
    let branch = LogBranch { turns: 1 };
    let target = Complex64 {
        re: 2.0,
        im: -std::f64::consts::TAU / std::f64::consts::LN_2,
    };
    let result = synthesize(
        target,
        Interpretation::Complex {
            basis: Basis::Bml,
            branch,
        },
        1e-14,
        budget(3, 2),
    )
    .unwrap();
    assert_eq!(result.stop, StopReason::ToleranceMet);
    let candidate = result.best.unwrap();
    assert_eq!(candidate.term, pair);
    assert_eq!(candidate.value, target);
    assert_eq!(
        candidate
            .term
            .evaluate_complex(
                Basis::Bml,
                branch,
                EvaluationBudget {
                    max_nodes: 3,
                    max_depth: 2
                }
            )
            .unwrap(),
        target
    );
}

#[test]
fn synthesis_budget_exhaustion_returns_a_candidate_without_claiming_success() {
    let real = Interpretation::Real(Basis::Bml);
    // Establish counted grammar progress before amplifying the node ceiling.
    // The wide case below retains its original 255-node/candidate-budget test.
    let progress = synthesize(Complex64::real(3.0), real, 0.0, budget(3, 2)).unwrap();
    assert_eq!(progress.stop, StopReason::NodeBudget);
    assert_eq!(progress.attempted, 2);
    let best = progress.best.unwrap();
    assert_eq!(best.value, Complex64::real(2.0));
    assert_eq!(best.nodes, 3);
    let bounded = synthesize(Complex64::real(3.0), real, 0.0, budget(255, 2)).unwrap();
    assert_eq!(bounded.stop, StopReason::CandidateBudget);
    assert_eq!(bounded.attempted, 2);
    assert_eq!(bounded.best.unwrap().nodes, 3);
    let result = synthesize(Complex64::real(3.0), real, 0.0, budget(6, 100)).unwrap();
    assert_eq!(result.stop, StopReason::NodeBudget);
    assert_eq!(result.attempted, 4); // Catalan(0) + Catalan(1) + Catalan(2).
    let candidate = result.best.unwrap();
    assert_eq!(candidate.value, Complex64::real(2.0));
    assert_eq!(candidate.nodes, 3); // Equal error at size five does not replace it.
    assert_eq!(candidate.error, 1.0);
    let result = synthesize(Complex64::real(3.0), real, 0.0, budget(7, 1)).unwrap();
    assert_eq!(result.stop, StopReason::CandidateBudget);
    assert_eq!(result.attempted, 1);
    assert_eq!(result.best.unwrap().term, ClosedTerm::One);
    let result = synthesize(Complex64::real(3.0), real, 0.0, budget(7, 0)).unwrap();
    assert_eq!(result.stop, StopReason::CandidateBudget);
    assert_eq!(result.attempted, 0);
    assert!(result.best.is_none());
    let result = synthesize(Complex64::real(1.0), real, 0.0, budget(0, 10)).unwrap();
    assert_eq!(result.stop, StopReason::NodeBudget);
    assert_eq!(result.attempted, 0);
    assert!(result.best.is_none());
}

#[test]
fn synthesis_enumerates_ordered_shapes_and_counts_invalid_numerical_candidates() {
    let policy = Interpretation::Real(Basis::Bml);
    // The third prefix is Op(One, Op(One,One)) = 1, not the mirrored tree = 4.
    let prefix = synthesize(Complex64::real(4.0), policy, 0.0, budget(5, 3)).unwrap();
    assert_eq!(prefix.stop, StopReason::CandidateBudget);
    assert_eq!(prefix.attempted, 3);
    assert_eq!(prefix.best.unwrap().value, Complex64::real(2.0));
    // Through nine nodes: 1 + 1 + 2 + 5 + 14 = 23 trees, including log(0).
    let result = synthesize(Complex64::real(100.0), policy, 0.0, budget(9, 100)).unwrap();
    assert_eq!(result.stop, StopReason::NodeBudget);
    assert_eq!(result.attempted, 23);
    let candidate = result.best.unwrap();
    assert_eq!(candidate.value, Complex64::real(16.0));
    assert_eq!(candidate.error, 84.0);
    assert_eq!(candidate.nodes, 7);
    let result = synthesize(Complex64::real(3.0), policy, 0.0, budget(7, 9)).unwrap();
    assert_eq!(result.stop, StopReason::ToleranceMet);
    let pair = ClosedTerm::op(ClosedTerm::One, ClosedTerm::One);
    assert_eq!(
        result.best.unwrap().term,
        ClosedTerm::op(pair.clone(), pair)
    );
    let result = synthesize(Complex64::real(1.125), policy, 0.125, budget(1, 1)).unwrap();
    assert_eq!(result.stop, StopReason::ToleranceMet); // Inclusive tolerance edge.
}

#[test]
fn synthesis_exhausts_exact_small_grammar_sizes_and_keeps_the_first_tie() {
    let policy = Interpretation::Real(Basis::Bml);
    // Full binary trees have Catalan counts 1, 1, 2, 5, 14 at sizes 1..=9.
    // Even budgets add no shape. No such small BML tree can reach f64::MAX.
    for (nodes, count) in [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 2),
        (5, 4),
        (6, 4),
        (7, 9),
        (8, 9),
        (9, 23),
        (10, 23),
    ] {
        let result =
            synthesize(Complex64::real(f64::MAX), policy, 0.0, budget(nodes, 100)).unwrap();
        assert_eq!(result.stop, StopReason::NodeBudget, "node budget {nodes}");
        assert_eq!(result.attempted, count, "node budget {nodes}");
        assert_eq!(result.best.is_none(), count == 0);
    }
    // One=1 and Op(One,One)=2 are equally close; all nine trees still count.
    let tie = synthesize(Complex64::real(1.5), policy, 0.0, budget(8, 9)).unwrap();
    assert_eq!(tie.stop, StopReason::NodeBudget);
    assert_eq!(tie.attempted, 9);
    let first = tie.best.unwrap();
    assert_eq!(first.term, ClosedTerm::One);
    assert_eq!(first.nodes, 1);
    assert_eq!(first.error, 0.5);
}

#[test]
fn synthesis_preserves_one_first_dfs_ranks_at_candidate_budget_edges() {
    let one = ClosedTerm::One;
    let pair = ClosedTerm::op(one.clone(), one.clone());
    let four = ClosedTerm::op(pair.clone(), one.clone());
    let zero = ClosedTerm::op(one.clone(), four.clone());
    let three = ClosedTerm::op(pair.clone(), pair);
    let sixteen = ClosedTerm::op(four.clone(), one);
    // BML is exp2(left)-log2(right). These exact values first occur at
    // positions 4, 6, 7 and 9 in the nine ordered trees through seven nodes.
    for (target, rank, nodes, term) in [
        (4.0, 4, 5, four),
        (0.0, 6, 7, zero),
        (3.0, 7, 7, three),
        (16.0, 9, 7, sixteen),
    ] {
        let before = synthesize(
            Complex64::real(target),
            Interpretation::Real(Basis::Bml),
            0.0,
            budget(7, rank - 1),
        )
        .unwrap();
        assert_eq!(before.stop, StopReason::CandidateBudget);
        assert_eq!(before.attempted, rank - 1);
        assert!(before.best.unwrap().error > 0.0);
        let found = synthesize(
            Complex64::real(target),
            Interpretation::Real(Basis::Bml),
            0.0,
            budget(7, rank),
        )
        .unwrap();
        assert_eq!(found.stop, StopReason::ToleranceMet);
        assert_eq!(found.attempted, rank);
        let candidate = found.best.unwrap();
        assert_eq!(candidate.term, term);
        assert_eq!(candidate.nodes, nodes);
        assert_eq!(candidate.value, Complex64::real(target));
        assert_eq!(candidate.error, 0.0);
    }
}

#[test]
fn synthesis_rejects_invalid_requests_before_enumeration() {
    let policy = Interpretation::Real(Basis::Bml);
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert_eq!(
            synthesize(Complex64::real(value), policy, 0.0, budget(1, 1)),
            Err(SynthesisError::Target)
        );
    }
    assert_eq!(
        synthesize(Complex64 { re: 1.0, im: 1.0 }, policy, 0.0, budget(1, 1)),
        Err(SynthesisError::Target)
    );
    for tolerance in [-1.0, f64::INFINITY, f64::NAN] {
        assert_eq!(
            synthesize(Complex64::real(1.0), policy, tolerance, budget(1, 1)),
            Err(SynthesisError::Tolerance)
        );
    }
    assert_eq!(
        synthesize(Complex64::real(1.0), policy, 0.0, budget(256, 1)),
        Err(SynthesisError::NodeLimit)
    );
    assert_eq!(
        synthesize(Complex64::real(1.0), policy, 0.0, budget(usize::MAX, 1)),
        Err(SynthesisError::NodeLimit)
    );
}
