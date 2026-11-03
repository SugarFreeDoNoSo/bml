//! Deterministic bounded comparison of certified synthesis candidates.
use super::{Bml, CertifiedBml, ExactError, ExactRational, ExactSynthesisOptions};
use crate::symbolic::EvalError;
use std::{
    cmp::Ordering,
    time::{Duration, Instant},
};

/// Expanded tree cost, independent of numeric evaluation and proof DAG sharing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeCost {
    /// Count of all B applications, including repeated subtrees.
    pub operators: usize,
    /// Useful tree bits, including the one-bit leaf.
    pub bits: usize,
    /// Maximum B nesting; a leaf has depth zero.
    pub depth: usize,
    /// Used u64 words in the integer encoding.
    pub words: usize,
    /// Allocated word capacity; not total memory of a certificate.
    pub word_capacity: usize,
}
impl Bml {
    /// Count the serialized tree within an explicit bit budget.
    pub fn tree_cost(&self, max_bits: usize) -> Result<TreeCost, ExactError> {
        if self.bit_len() > max_bits {
            return Err(ExactError::Limit);
        }
        let mut stack: Vec<usize> = Vec::new();
        for bit in 0..self.bit_len() {
            let depth = if self.bit(bit) {
                let a = stack.pop().ok_or(ExactError::InvalidProof)?;
                let b = stack.pop().ok_or(ExactError::InvalidProof)?;
                a.max(b).checked_add(1).ok_or(ExactError::Limit)?
            } else {
                0
            };
            stack.try_reserve(1).map_err(|_| ExactError::Limit)?;
            stack.push(depth);
        }
        Ok(TreeCost {
            operators: self.operator_count(),
            bits: self.bit_len(),
            depth: stack[0],
            words: self.words.len(),
            word_capacity: self.words.capacity(),
        })
    }
}

/// Candidate family, not an assertion that it found a globally optimal code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExactStrategy {
    /// Reuse a previously certified target from this explicitly bounded searcher.
    Previous,
    /// Small fixed catalogue of independently constructed primitive trees.
    Catalogue,
    /// Direct power-of-two decomposition.
    PowerOfTwo,
    /// Upper-power difference recursion, also used for general rational targets.
    NearPower,
    /// Positive integer factor pair and explicit sign adjustment.
    Factors,
    /// Lower power plus the remaining integer magnitude.
    Sum,
    /// Next upper power minus the remaining integer magnitude.
    Difference,
}
/// Outcome of one attempted candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateOutcome {
    /// Target and proof checked; includes expanded cost.
    Certified(TreeCost),
    /// No unverified or approximate candidate is accepted.
    Rejected(ExactError),
}
/// One auditable attempt in deterministic exploration order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateRecord {
    /// Family used to construct the candidate.
    pub strategy: ExactStrategy,
    /// Catalogue code, factor divisor, or nearby power used by this candidate.
    pub parameter: Option<u64>,
    /// Exact success or typed rejection.
    pub outcome: CandidateOutcome,
}
/// Reason exploration stopped; neither variant proves global minimality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchTermination {
    /// All applicable configured strategies completed their bounded exploration.
    StrategiesCompleted,
    /// A candidate or factor-trial limit prevented further exploration.
    BudgetExhausted,
}
/// Search and persistent reuse limits. No global cache is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactSearchOptions {
    /// Per-construction, proof and rational budgets.
    pub synthesis: ExactSynthesisOptions,
    /// Maximum attempted candidates per search, including rejected candidates.
    pub max_candidates: usize,
    /// Maximum trial divisors tested per search; no unbounded factorization.
    pub max_factor_trials: usize,
    /// Maximum retained prior results; zero disables reuse storage.
    pub max_cache_entries: usize,
    /// Maximum total expanded term bits in the retained cache.
    pub max_cache_bits: usize,
    /// Maximum total derivation steps in the retained cache.
    pub max_cache_steps: usize,
}
impl Default for ExactSearchOptions {
    fn default() -> Self {
        Self {
            synthesis: ExactSynthesisOptions::default(),
            max_candidates: 32,
            max_factor_trials: 256,
            max_cache_entries: 16,
            max_cache_bits: 4_194_304,
            max_cache_steps: 16_384,
        }
    }
}
/// Search evidence; timing never participates in the ordering of candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactSearchReport {
    /// Exact budgets used for this search.
    pub options: ExactSearchOptions,
    /// Attempt records, including failures, in deterministic order.
    pub candidates: Vec<CandidateRecord>,
    /// Index of the selected candidate, if any.
    pub selected: Option<usize>,
    /// Number of divisibility tests actually performed.
    pub factor_trials: usize,
    /// Number of cache entries compared with the requested target.
    pub cache_lookups: usize,
    /// Whether an applicable candidate was blocked by the candidate budget.
    pub candidate_limit_reached: bool,
    /// Whether integer factor exploration stopped before reaching sqrt(n).
    pub factor_limit_reached: bool,
    /// Exploration termination, never a minimum proof.
    pub termination: SearchTermination,
    /// Total measured synthesis/search time, separate from tree cost.
    pub elapsed: Duration,
    /// Always false: this heuristic does not provide an exhaustive minimum proof.
    pub minimum_proven: bool,
    /// Optional direct numeric observation, measured only when explicitly requested.
    /// A finite result does not establish numerical accuracy or stability in general.
    pub direct_evaluation: Option<DirectEvaluationMeasurement>,
}
/// Direct f64 evaluation evidence, separate from exact proof and search ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectEvaluationMeasurement {
    /// Elapsed time of the direct evaluator only.
    pub elapsed: Duration,
    /// Observed finite result or domain/range failure.
    pub result: Result<f64, EvalError>,
}
/// Best certified result among explored candidates, or no success within budget.
/// Direct float evaluation/stability must be measured separately on `best.term()`.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactSearchResult {
    /// None means all attempted candidates failed, or no candidate budget existed.
    pub best: Option<CertifiedBml>,
    /// Reproducible strategy/cost/budget evidence plus separately measured time.
    pub report: ExactSearchReport,
}
impl ExactSearchResult {
    /// Measure the selected tree numerically, without changing its certificate,
    /// search ranking, or exact target. None means no candidate was certified.
    pub fn measure_direct_evaluation(&mut self) -> Option<&DirectEvaluationMeasurement> {
        let best = self.best.as_ref()?;
        let start = Instant::now();
        let result = best.term().to_f64();
        self.report.direct_evaluation = Some(DirectEvaluationMeasurement {
            elapsed: start.elapsed(),
            result,
        });
        self.report.direct_evaluation.as_ref()
    }
}
/// Explicit bounded cache owner for repeated exact synthesis requests.
pub struct ExactSearcher {
    options: ExactSearchOptions,
    cache: Vec<CertifiedBml>,
}
impl ExactSearcher {
    /// Create an empty searcher; no hidden persistent or process-wide state.
    pub fn new(options: ExactSearchOptions) -> Self {
        Self {
            options,
            cache: vec![],
        }
    }
    /// Number of currently retained prior results.
    pub fn cached_results(&self) -> usize {
        self.cache.len()
    }
    /// Drop all retained results without changing search budgets.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    /// Compare applicable strategies. Every accepted candidate has the requested
    /// exact target; failed candidates remain visible and never become approximations.
    pub fn search(&mut self, target: &ExactRational) -> ExactSearchResult {
        let start = Instant::now();
        let options = self.options;
        let mut run = SearchRun {
            target,
            options,
            best: None,
            best_cost: None,
            report: ExactSearchReport {
                options,
                candidates: vec![],
                selected: None,
                factor_trials: 0,
                cache_lookups: 0,
                candidate_limit_reached: false,
                factor_limit_reached: false,
                termination: SearchTermination::StrategiesCompleted,
                elapsed: Duration::ZERO,
                minimum_proven: false,
                direct_evaluation: None,
            },
        };
        for previous in &self.cache {
            run.report.cache_lookups += 1;
            if previous.target() == target {
                run.attempt(ExactStrategy::Previous, None, || {
                    previous
                        .derivation()
                        .verify(previous.term(), target, options.synthesis)
                });
                break;
            }
        }
        // Catalogue entries are fixed valid syntax; each is certified and its
        // exact value compared, rather than inferred from a float evaluation.
        let catalogue = [
            (0u64, 1i64),
            (4, 2),
            (24, 4),
            (88, 0),
            (100, 3),
            (112, 16),
            (452, 15),
        ];
        for (code, value) in catalogue {
            if *target == ExactRational::from_i64(value) {
                run.attempt(ExactStrategy::Catalogue, Some(code), || {
                    Bml::from_u64(code)
                        .map_err(|_| ExactError::InvalidProof)?
                        .certify_exact(options.synthesis)
                });
            }
        }
        let natural = target.denominator_words_le() == [1];
        let magnitude = if natural {
            match target.numerator_words_le() {
                [] => Some(0),
                [n] => Some(*n),
                _ => None,
            }
        } else {
            None
        };
        let power = natural
            && target
                .numerator_words_le()
                .iter()
                .map(|w| w.count_ones() as usize)
                .sum::<usize>()
                == 1;
        run.attempt(
            if power {
                ExactStrategy::PowerOfTwo
            } else {
                ExactStrategy::NearPower
            },
            None,
            || Bml::from_rational_exact(target, options.synthesis),
        );
        if let Some(n) = magnitude.filter(|n| *n > 1) {
            let negative = target.is_negative();
            let build = |value: u64| -> Result<CertifiedBml, ExactError> {
                let target = ExactRational::from_decimal_str(
                    &value.to_string(),
                    options.synthesis.rational,
                )?;
                Bml::from_rational_exact(&target, options.synthesis)
            };
            let sign = |value: CertifiedBml| {
                if negative {
                    value.checked_neg(options.synthesis)
                } else {
                    Ok(value)
                }
            };
            let lower = 1u64 << (63 - n.leading_zeros());
            if lower < n {
                run.attempt(ExactStrategy::Sum, Some(lower), || {
                    sign(build(lower)?.checked_add(&build(n - lower)?, options.synthesis)?)
                });
                if let Some(upper) = lower.checked_mul(2) {
                    run.attempt(ExactStrategy::Difference, Some(upper), || {
                        sign(build(upper)?.checked_sub(&build(upper - n)?, options.synthesis)?)
                    });
                }
            }
            let mut divisor = 2u64;
            while divisor <= n / divisor {
                if run.report.factor_trials >= options.max_factor_trials {
                    run.report.factor_limit_reached = true;
                    break;
                }
                if run.report.candidates.len() >= options.max_candidates {
                    run.report.candidate_limit_reached = true;
                    break;
                }
                run.report.factor_trials += 1;
                if n.is_multiple_of(divisor) {
                    run.attempt(ExactStrategy::Factors, Some(divisor), || {
                        sign(build(divisor)?.checked_mul(&build(n / divisor)?, options.synthesis)?)
                    });
                }
                divisor += 1;
            }
        }
        if run.report.candidate_limit_reached || run.report.factor_limit_reached {
            run.report.termination = SearchTermination::BudgetExhausted;
        }
        run.report.elapsed = start.elapsed();
        let mut result = ExactSearchResult {
            best: run.best,
            report: run.report,
        };
        if let Some(best) = &result.best {
            self.remember(best);
        }
        result.report.elapsed = start.elapsed();
        result
    }
    fn remember(&mut self, value: &CertifiedBml) {
        let options = self.options;
        if options.max_cache_entries == 0
            || value.term().bit_len() > options.max_cache_bits
            || value.derivation().steps.len() > options.max_cache_steps
        {
            return;
        }
        if self
            .cache
            .iter()
            .any(|entry| entry.target() == value.target() && entry.term() == value.term())
        {
            return;
        }
        while !self.cache.is_empty()
            && (self.cache.len() >= options.max_cache_entries
                || self
                    .cache
                    .iter()
                    .try_fold(value.term().bit_len(), |n, x| {
                        n.checked_add(x.term().bit_len())
                    })
                    .is_none_or(|n| n > options.max_cache_bits)
                || self
                    .cache
                    .iter()
                    .try_fold(value.derivation().steps.len(), |n, x| {
                        n.checked_add(x.derivation().steps.len())
                    })
                    .is_none_or(|n| n > options.max_cache_steps))
        {
            self.cache.remove(0);
        }
        self.cache.push(value.clone());
    }
}
struct SearchRun<'a> {
    target: &'a ExactRational,
    options: ExactSearchOptions,
    best: Option<CertifiedBml>,
    best_cost: Option<TreeCost>,
    report: ExactSearchReport,
}
impl SearchRun<'_> {
    fn attempt(
        &mut self,
        strategy: ExactStrategy,
        parameter: Option<u64>,
        build: impl FnOnce() -> Result<CertifiedBml, ExactError>,
    ) {
        if self.report.candidates.len() >= self.options.max_candidates {
            self.report.candidate_limit_reached = true;
            return;
        }
        let outcome = match build().and_then(|value| {
            if value.target() != self.target {
                return Err(ExactError::InvalidProof);
            }
            let cost = value.term().tree_cost(self.options.synthesis.max_bits)?;
            Ok((value, cost))
        }) {
            Ok((value, cost)) => {
                let ordering = match (&self.best, self.best_cost) {
                    (Some(best), Some(previous)) => cost
                        .operators
                        .cmp(&previous.operators)
                        .then_with(|| cost.depth.cmp(&previous.depth))
                        .then_with(|| {
                            value
                                .term()
                                .to_words_le()
                                .iter()
                                .rev()
                                .cmp(best.term().to_words_le().iter().rev())
                        }),
                    _ => Ordering::Less,
                };
                if ordering == Ordering::Less {
                    self.best = Some(value);
                    self.best_cost = Some(cost);
                    self.report.selected = Some(self.report.candidates.len());
                }
                CandidateOutcome::Certified(cost)
            }
            Err(error) => CandidateOutcome::Rejected(error),
        };
        self.report.candidates.push(CandidateRecord {
            strategy,
            parameter,
            outcome,
        });
    }
}
