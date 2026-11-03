//! Conservative structural recognition of proven expansion rules.
//! This is not a general decision procedure for real-domain membership.
use super::{Bml, CertifiedBml, ExactDerivation, ExactError, ExactStep, ExactSynthesisOptions};
use std::collections::BTreeMap;

#[derive(Clone)]
enum Pattern {
    One,
    Var(usize),
    Op(Box<Pattern>, Box<Pattern>),
}
fn op(a: Pattern, b: Pattern) -> Pattern {
    Pattern::Op(Box::new(a), Box::new(b))
}
fn exp(a: Pattern) -> Pattern {
    op(a, Pattern::One)
}
fn log(a: Pattern) -> Pattern {
    op(Pattern::One, exp(op(Pattern::One, a)))
}
fn difference(a: Pattern, b: Pattern) -> Pattern {
    op(log(a), exp(b))
}
fn addition(x: Pattern, y: Pattern) -> Pattern {
    let p = exp(x.clone());
    let q = op(x, p.clone());
    difference(p, difference(q, y))
}
fn subtraction(x: Pattern, y: Pattern) -> Pattern {
    let p = exp(x.clone());
    let q = op(x, p.clone());
    difference(p, addition(q, y))
}
fn negation(x: Pattern) -> Pattern {
    let p = exp(x.clone());
    let q = op(x, p.clone());
    difference(q, p)
}

#[derive(Clone, Copy)]
enum Rule {
    One,
    Zero,
    Exp,
    Near,
    Add,
    Sub,
    Neg,
    Mul,
    Div,
}
fn patterns() -> Vec<(Pattern, Rule)> {
    let (a, b) = (Pattern::Var(0), Pattern::Var(1));
    vec![
        (Pattern::One, Rule::One),
        (op(Pattern::One, exp(exp(Pattern::One))), Rule::Zero),
        // Recognize complete arithmetic macros before their primitive outer node.
        (exp(subtraction(log(a.clone()), log(b.clone()))), Rule::Div),
        (exp(addition(log(a.clone()), log(b.clone()))), Rule::Mul),
        (subtraction(a.clone(), b.clone()), Rule::Sub),
        (addition(a.clone(), b.clone()), Rule::Add),
        (negation(a.clone()), Rule::Neg),
        (op(a.clone(), exp(b)), Rule::Near),
        (exp(a), Rule::Exp),
    ]
}

// Equal subtrees have equal IDs. Interning only aids recognition; the returned
// Bml remains the fully expanded integer tree, never a shared-node format.
struct Recognizer {
    nodes: Vec<Option<(usize, usize)>>,
    patterns: Vec<(Pattern, Rule)>,
    derived: Vec<Option<usize>>,
    steps: Vec<ExactStep>,
    options: ExactSynthesisOptions,
    comparisons: usize,
}
impl Recognizer {
    fn matches(
        &mut self,
        pattern: &Pattern,
        root: usize,
    ) -> Result<Option<[Option<usize>; 2]>, ExactError> {
        let mut variables = [None, None];
        let mut pending = vec![(pattern, root)];
        while let Some((pattern, node)) = pending.pop() {
            self.comparisons = self.comparisons.checked_add(1).ok_or(ExactError::Limit)?;
            // Fixed templates have <256 nodes; also bound failed match attempts.
            if self.comparisons > self.options.max_steps.saturating_mul(4096) {
                return Err(ExactError::Limit);
            }
            match pattern {
                Pattern::One => {
                    if node != 0 {
                        return Ok(None);
                    }
                }
                Pattern::Var(slot) => match variables[*slot] {
                    Some(previous) if previous != node => return Ok(None),
                    _ => variables[*slot] = Some(node),
                },
                Pattern::Op(left, right) => {
                    let Some((a, b)) = self.nodes[node] else {
                        return Ok(None);
                    };
                    pending.push((right, b));
                    pending.push((left, a));
                }
            }
        }
        Ok(Some(variables))
    }
    fn derive(&mut self, node: usize, depth: usize) -> Result<usize, ExactError> {
        if depth > self.options.max_recursion {
            return Err(ExactError::Limit);
        }
        if let Some(step) = self.derived[node] {
            return Ok(step);
        }
        for i in 0..self.patterns.len() {
            let (pattern, rule) = self.patterns[i].clone();
            if let Some(variables) = self.matches(&pattern, node)? {
                let a = match variables[0] {
                    Some(n) => Some(self.derive(n, depth + 1)?),
                    None => None,
                };
                let b = match variables[1] {
                    Some(n) => Some(self.derive(n, depth + 1)?),
                    None => None,
                };
                let a = || a.ok_or(ExactError::InvalidProof);
                let b = || b.ok_or(ExactError::InvalidProof);
                let step = match rule {
                    Rule::One => ExactStep::One,
                    Rule::Zero => ExactStep::Zero,
                    Rule::Exp => ExactStep::Exp2(a()?),
                    Rule::Near => ExactStep::PowerDifference(a()?, b()?),
                    Rule::Add => ExactStep::Add(a()?, b()?),
                    Rule::Sub => ExactStep::Sub(a()?, b()?),
                    Rule::Neg => ExactStep::Neg(a()?),
                    Rule::Mul => ExactStep::PositiveMul(a()?, b()?),
                    Rule::Div => ExactStep::PositiveDiv(a()?, b()?),
                };
                if self.steps.len() >= self.options.max_steps {
                    return Err(ExactError::Limit);
                }
                self.steps.try_reserve(1).map_err(|_| ExactError::Limit)?;
                let index = self.steps.len();
                self.steps.push(step);
                self.derived[node] = Some(index);
                return Ok(index);
            }
        }
        Err(ExactError::DomainNotJustified)
    }
}

impl Bml {
    /// Recover a certificate for recognized exact expansion rules.
    ///
    /// Recognition is conservative: a valid real tree can return
    /// `DomainNotJustified`. No sign, domain or equality is inferred from floats.
    /// Expanded input bits, distinct nodes, pattern work and recursion are bounded.
    pub fn certify_exact(
        &self,
        options: ExactSynthesisOptions,
    ) -> Result<CertifiedBml, ExactError> {
        if self.bit_len() > options.max_bits || options.max_steps == 0 {
            return Err(ExactError::Limit);
        }
        let mut nodes = vec![None];
        let mut intern = BTreeMap::new();
        let mut stack = Vec::new();
        for index in 0..self.bit_len() {
            let node = if self.bit(index) {
                let a = stack.pop().ok_or(ExactError::InvalidProof)?;
                let b = stack.pop().ok_or(ExactError::InvalidProof)?;
                if let Some(id) = intern.get(&(a, b)) {
                    *id
                } else {
                    if nodes.len() >= options.max_steps {
                        return Err(ExactError::Limit);
                    }
                    nodes.try_reserve(1).map_err(|_| ExactError::Limit)?;
                    let id = nodes.len();
                    nodes.push(Some((a, b)));
                    intern.insert((a, b), id);
                    id
                }
            } else {
                0
            };
            if stack.len() >= options.max_steps {
                return Err(ExactError::Limit);
            }
            stack.try_reserve(1).map_err(|_| ExactError::Limit)?;
            stack.push(node);
        }
        let root = stack.pop().ok_or(ExactError::InvalidProof)?;
        if !stack.is_empty() {
            return Err(ExactError::InvalidProof);
        }
        let mut recognizer = Recognizer {
            derived: vec![None; nodes.len()],
            nodes,
            patterns: patterns(),
            steps: vec![],
            options,
            comparisons: 0,
        };
        let root = recognizer.derive(root, 0)?;
        let proof = ExactDerivation {
            steps: recognizer.steps,
            root,
        };
        let result = proof.certify(options).map_err(|error| match error {
            ExactError::Rational(super::ExactInputError::NonIntegralExponent) => {
                ExactError::DomainNotJustified
            }
            other => other,
        })?;
        if result.term() != self {
            return Err(ExactError::InvalidProof);
        }
        Ok(result)
    }
}
