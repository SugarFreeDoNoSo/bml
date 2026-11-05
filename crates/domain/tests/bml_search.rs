use bml_domain::number::{
    Bml, CandidateOutcome, ExactRational, ExactSearchOptions, ExactSearcher, ExactStrategy,
    SearchTermination,
};
fn target(n: i64) -> ExactRational {
    ExactRational::from_i64(n)
}

#[test]
fn expanded_cost_counts_duplicate_subtrees_and_depth() {
    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let four = two.op(&one).unwrap();
    let fifteen = four.op(&two).unwrap();
    let cost = fifteen.tree_cost(9).unwrap();
    assert_eq!(
        (cost.operators, cost.bits, cost.depth, cost.words),
        (4, 9, 3, 1)
    );
    let duplicate = fifteen.op(&fifteen).unwrap();
    let cost = duplicate.tree_cost(19).unwrap();
    assert_eq!((cost.operators, cost.bits, cost.depth), (9, 19, 4));
    assert!(duplicate.tree_cost(18).is_err());
    let leaf = one.tree_cost(1).unwrap();
    assert_eq!(
        (leaf.operators, leaf.bits, leaf.depth, leaf.words),
        (0, 1, 0, 0)
    );
}

#[test]
fn strategies_are_auditable_and_selected_cost_is_minimum_among_attempts() {
    let options = ExactSearchOptions::default();
    let mut search = ExactSearcher::new(options);
    for n in [0, 1, 2, 3, 4, 12, 15, 16, -12] {
        let result = search.search(&target(n));
        let best = result.best.as_ref().unwrap();
        assert_eq!(best.target(), &target(n));
        assert!(!result.report.minimum_proven);
        best.derivation()
            .verify(best.term(), best.target(), options.synthesis)
            .unwrap();
        let cost = best.term().tree_cost(options.synthesis.max_bits).unwrap();
        for candidate in &result.report.candidates {
            if let CandidateOutcome::Certified(other) = candidate.outcome {
                assert!((cost.operators, cost.depth) <= (other.operators, other.depth));
            }
        }
        assert!(matches!(
            result.report.candidates[result.report.selected.unwrap()].outcome,
            CandidateOutcome::Certified(_)
        ));
    }
    let result = search.search(&target(12));
    for strategy in [
        ExactStrategy::Previous,
        ExactStrategy::NearPower,
        ExactStrategy::Sum,
        ExactStrategy::Difference,
        ExactStrategy::Factors,
    ] {
        assert!(
            result
                .report
                .candidates
                .iter()
                .any(|c| c.strategy == strategy),
            "{strategy:?}"
        );
    }
    assert_eq!(result.report.factor_trials, 2);
    let result = search.search(&target(16));
    assert!(result
        .report
        .candidates
        .iter()
        .any(|c| c.strategy == ExactStrategy::PowerOfTwo));
    let result = search.search(&target(15));
    assert!(result
        .report
        .candidates
        .iter()
        .any(|c| c.strategy == ExactStrategy::Catalogue));
    assert_eq!(result.best.unwrap().term().operator_count(), 4);
}

#[test]
fn budgets_limit_candidates_factorization_and_cache() {
    let base = ExactSearchOptions::default();
    let mut none = ExactSearcher::new(ExactSearchOptions {
        max_candidates: 0,
        ..base
    });
    let result = none.search(&target(101));
    assert!(result.best.is_none());
    assert!(result.report.candidates.is_empty());
    assert_eq!(
        result.report.termination,
        SearchTermination::BudgetExhausted
    );
    let mut bounded = ExactSearcher::new(ExactSearchOptions {
        max_factor_trials: 3,
        max_cache_entries: 1,
        ..base
    });
    let result = bounded.search(&target(101));
    assert_eq!(result.report.factor_trials, 3);
    assert!(result.report.factor_limit_reached);
    assert_eq!(
        result.report.termination,
        SearchTermination::BudgetExhausted
    );
    assert!(result.best.is_some());
    assert_eq!(bounded.cached_results(), 1);
    bounded.search(&target(2));
    assert_eq!(bounded.cached_results(), 1);
    let result = bounded.search(&target(101));
    assert!(!result
        .report
        .candidates
        .iter()
        .any(|c| c.strategy == ExactStrategy::Previous));
    bounded.clear_cache();
    assert_eq!(bounded.cached_results(), 0);
    let mut tiny = ExactSearcher::new(ExactSearchOptions {
        max_cache_bits: 1,
        ..base
    });
    tiny.search(&target(15));
    assert_eq!(tiny.cached_results(), 0);
    tiny.search(&target(1));
    assert_eq!(tiny.cached_results(), 1);
    let mut tiny = ExactSearcher::new(ExactSearchOptions {
        max_cache_steps: 0,
        ..base
    });
    tiny.search(&target(1));
    assert_eq!(tiny.cached_results(), 0);
    let mut limited = ExactSearcher::new(ExactSearchOptions {
        max_candidates: 1,
        ..base
    });
    let result = limited.search(&target(12));
    assert_eq!(result.report.candidates.len(), 1);
    assert!(result.report.candidate_limit_reached);
}

fn assert_independent_cache_eviction(options: ExactSearchOptions) {
    let mut searcher = ExactSearcher::new(options);
    let one = searcher.search(&target(1)).best.unwrap();
    assert_eq!(one.term().bit_len(), 1);
    assert_eq!(one.derivation().steps.len(), 1);
    assert_eq!(searcher.cached_results(), 1);

    let two = searcher.search(&target(2)).best.unwrap();
    assert_eq!(two.term().bit_len(), 3);
    assert_eq!(two.derivation().steps.len(), 2);
    // Each entry fits on its own. The previous entry must be evicted when
    // either cumulative budget is exceeded, even if the other has ample room.
    assert_eq!(searcher.cached_results(), 1, "{options:?}");

    let repeated = searcher.search(&target(1));
    assert_eq!(repeated.report.cache_lookups, 1);
    assert!(!repeated
        .report
        .candidates
        .iter()
        .any(|candidate| candidate.strategy == ExactStrategy::Previous));
    assert_eq!(searcher.cached_results(), 1);
}

#[test]
fn cache_bit_budget_evicts_independently_of_the_step_budget() {
    assert_independent_cache_eviction(ExactSearchOptions {
        max_cache_bits: 3,
        ..ExactSearchOptions::default()
    });
}

#[test]
fn cache_step_budget_evicts_independently_of_the_bit_budget() {
    assert_independent_cache_eviction(ExactSearchOptions {
        max_cache_steps: 2,
        ..ExactSearchOptions::default()
    });
}

#[test]
fn repeated_search_does_not_duplicate_the_same_certified_cache_entry() {
    let mut searcher = ExactSearcher::new(ExactSearchOptions::default());
    let cold = searcher.search(&target(2)).best.unwrap();
    for _ in 0..3 {
        let warm = searcher.search(&target(2));
        assert_eq!(warm.best.as_ref().unwrap(), &cold);
        assert_eq!(warm.report.cache_lookups, 1);
        assert_eq!(warm.report.candidates[0].strategy, ExactStrategy::Previous);
        assert_eq!(searcher.cached_results(), 1);
    }
}

#[test]
fn cumulative_cache_budgets_accept_exact_totals() {
    for options in [
        ExactSearchOptions {
            max_cache_bits: 4,
            ..ExactSearchOptions::default()
        },
        ExactSearchOptions {
            max_cache_steps: 3,
            ..ExactSearchOptions::default()
        },
    ] {
        let mut searcher = ExactSearcher::new(options);
        let one = searcher.search(&target(1)).best.unwrap();
        let two = searcher.search(&target(2)).best.unwrap();
        assert_eq!(one.term().bit_len() + two.term().bit_len(), 4);
        assert_eq!(
            one.derivation().steps.len() + two.derivation().steps.len(),
            3
        );
        assert_eq!(searcher.cached_results(), 2, "{options:?}");
        for (n, lookups) in [(1, 1), (2, 2)] {
            let warm = searcher.search(&target(n));
            assert_eq!(warm.report.cache_lookups, lookups);
            assert_eq!(warm.report.candidates[0].strategy, ExactStrategy::Previous);
            assert_eq!(searcher.cached_results(), 2);
        }
    }
}

#[test]
fn failures_are_reported_without_approximate_success_and_ties_are_deterministic() {
    let mut options = ExactSearchOptions::default();
    options.synthesis.max_bits = 0;
    let result = ExactSearcher::new(options).search(&target(15));
    assert!(result.best.is_none());
    assert!(result
        .report
        .candidates
        .iter()
        .all(|c| matches!(c.outcome, CandidateOutcome::Rejected(_))));
    let options = ExactSearchOptions::default();
    let mut search = ExactSearcher::new(options);
    let cold = search.search(&target(15));
    let warm = search.search(&target(15));
    assert_eq!(
        cold.best.as_ref().unwrap().term(),
        warm.best.as_ref().unwrap().term()
    );
    assert_eq!(warm.report.candidates[0].strategy, ExactStrategy::Previous);
    let again = ExactSearcher::new(options).search(&target(15));
    assert_eq!(cold.report.candidates, again.report.candidates);
    assert_eq!(cold.report.selected, again.report.selected);
    let rational = ExactRational::from_decimal_str("-0.125", options.synthesis.rational).unwrap();
    let result = search.search(&rational);
    assert_eq!(result.best.unwrap().target(), &rational);
}

#[test]
fn numeric_observations_are_explicit_and_cannot_change_exact_ranking() {
    let mut result = ExactSearcher::new(ExactSearchOptions::default()).search(&target(2049));
    assert!(result.report.direct_evaluation.is_none());
    let before = result.best.clone();
    let selected = result.report.selected;
    let _measurement = result.measure_direct_evaluation().unwrap();
    assert_eq!(result.best, before);
    assert_eq!(result.report.selected, selected);
    assert!(result.report.direct_evaluation.is_some());
    assert!(!result.report.minimum_proven);
    let mut empty = ExactSearcher::new(ExactSearchOptions {
        max_candidates: 0,
        ..ExactSearchOptions::default()
    })
    .search(&target(1));
    assert!(empty.measure_direct_evaluation().is_none());
}

#[test]
fn logical_growth_storage_and_structural_equality_are_separate() {
    let one = Bml::one();
    let two = one.op(&one).unwrap();
    let another_one = one.op(&two).unwrap();
    assert_ne!(one, another_one);
    let options = ExactSearchOptions::default();
    assert_eq!(
        one.certify_exact(options.synthesis).unwrap().target(),
        another_one
            .certify_exact(options.synthesis)
            .unwrap()
            .target()
    );
    let four = two.op(&one).unwrap();
    let zero = one.op(&four).unwrap();
    assert_eq!(zero.to_binary(), "1011000");
    assert_eq!(zero.to_u64(), Some(88));
    let original = two.clone();
    let cost = two.tree_cost(3).unwrap();
    assert_eq!(cost.bits, 3);
    assert!(cost.word_capacity >= cost.words);
    assert_eq!(cost.words, 1);
    assert!(two.op_with_limit(&two, 6).is_err());
    assert_eq!(two, original);
    let expanded = two.op_with_limit(&two, 7).unwrap();
    assert_eq!(expanded.tree_cost(7).unwrap().operators, 3);
}
