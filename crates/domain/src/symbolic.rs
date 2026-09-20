//! Álgebra libre de términos, independiente de su interpretación numérica.
//!
//! `ClosedTerm = One | Op(ClosedTerm, ClosedTerm)`.
//! `OpenTerm` añade variables e inyección de términos cerrados. `bind` es
//! sustitución parcial; `collapse` comprueba que no quedan variables libres.
//! El árbol no lleva una base implícita: el llamador elige [`Basis`] al evaluar.
//!
//! Este AST con `Box` está destinado a fórmulas pequeñas. La compilación de
//! grafos grandes necesita una arena con IDs y recorridos iterativos. Estos
//! términos no son el almacenamiento de los elementos de un tensor.
//!
//! ```
//! use bml_domain::symbolic::{Basis, ClosedTerm, OpenTerm};
//! use std::collections::HashMap;
//!
//! let formula = OpenTerm::op(OpenTerm::Var("x".into()), ClosedTerm::One.into());
//! let env = HashMap::from([("x".to_owned(), ClosedTerm::One)]);
//! let closed = formula.bind(&env).collapse()?;
//! assert_eq!(closed.evaluate_real(Basis::Bml)?, 2.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// Término cerrado: contiene únicamente el generador y aplicaciones binarias.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClosedTerm {
    /// Generador distinguido.
    One,
    /// Aplicación ordenada; no implica conmutatividad ni asociatividad.
    Op(Box<Self>, Box<Self>),
}

impl ClosedTerm {
    /// Construye una aplicación sin evaluar ni simplificar sus operandos.
    pub fn op(left: Self, right: Self) -> Self {
        Self::Op(Box::new(left), Box::new(right))
    }

    /// Evalúa por referencia en reales finitos, bajo una base explícita.
    ///
    /// Un término sintácticamente cerrado puede tener una evaluación inválida.
    /// Se rechaza cualquier argumento logarítmico no positivo e intermedio
    /// no finito. Las derivaciones que necesitan `log(0)` o complejos requieren
    /// otro evaluador; no se sustituyen errores por `NaN` silenciosamente.
    pub fn evaluate_real(&self, basis: Basis) -> Result<f64, EvalError> {
        match self {
            Self::One => Ok(1.0),
            Self::Op(left, right) => {
                basis.apply_real(left.evaluate_real(basis)?, right.evaluate_real(basis)?)
            }
        }
    }
}

/// Término abierto: permite variables libres y subárboles cerrados.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenTerm {
    /// Inyección estructural de un término cerrado, sin conversión numérica.
    Closed(ClosedTerm),
    /// Variable identificada por su nombre.
    Var(String),
    /// Aplicación ordenada.
    Op(Box<Self>, Box<Self>),
}

impl OpenTerm {
    /// Construye una aplicación simbólica.
    pub fn op(left: Self, right: Self) -> Self {
        Self::Op(Box::new(left), Box::new(right))
    }

    /// Sustituye las variables presentes en `env`; preserva las desconocidas.
    ///
    /// No evalúa y no modifica el término original. Como los valores del
    /// entorno son cerrados, la sustitución no introduce nuevas variables.
    pub fn bind(&self, env: &HashMap<String, ClosedTerm>) -> Self {
        match self {
            Self::Closed(term) => Self::Closed(term.clone()),
            Self::Var(name) => env
                .get(name)
                .cloned()
                .map(Self::Closed)
                .unwrap_or_else(|| Self::Var(name.clone())),
            Self::Op(left, right) => Self::op(left.bind(env), right.bind(env)),
        }
    }

    /// Nombres libres, únicos y ordenados; útil antes de completar un entorno.
    pub fn free_variables(&self) -> BTreeSet<&str> {
        let mut names = BTreeSet::new();
        let mut pending = vec![self];
        while let Some(term) = pending.pop() {
            match term {
                Self::Closed(_) => {}
                Self::Var(name) => {
                    names.insert(name.as_str());
                }
                Self::Op(left, right) => {
                    pending.push(right);
                    pending.push(left);
                }
            }
        }
        names
    }

    /// Convierte a término cerrado o identifica la primera variable libre.
    ///
    /// El cierre es estructural: no prueba que la evaluación numérica exista.
    pub fn collapse(self) -> Result<ClosedTerm, UnboundVariable> {
        match self {
            Self::Closed(term) => Ok(term),
            Self::Var(name) => Err(UnboundVariable { name }),
            Self::Op(left, right) => Ok(ClosedTerm::op(left.collapse()?, right.collapse()?)),
        }
    }
}

impl From<ClosedTerm> for OpenTerm {
    fn from(term: ClosedTerm) -> Self {
        Self::Closed(term)
    }
}

/// Variable que impide cerrar una expresión.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnboundVariable {
    /// Nombre de la primera variable libre, recorriendo de izquierda a derecha.
    pub name: String,
}

impl fmt::Display for UnboundVariable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unbound variable: {}", self.name)
    }
}

impl std::error::Error for UnboundVariable {}

/// Interpretación de todos los nodos `Op` de una expresión.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// `exp(x) - ln(y)`, base natural del artículo.
    Eml,
    /// `exp2(x) - log2(y)`, variante binaria del proyecto.
    Bml,
}

impl Basis {
    /// Aplica el operador a valores reales finitos con `y > 0`.
    ///
    /// La comprobación no certifica precisión relativa ni ausencia de
    /// cancelación; cada transformación debe tener su propio presupuesto de error.
    pub fn apply_real(self, x: f64, y: f64) -> Result<f64, EvalError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(EvalError::NonFiniteInput);
        }
        if y <= 0.0 {
            return Err(EvalError::LogDomain { value: y });
        }
        let (exp, log) = match self {
            Self::Eml => (x.exp(), y.ln()),
            Self::Bml => (x.exp2(), y.log2()),
        };
        if !exp.is_finite() {
            return Err(EvalError::Overflow);
        }
        if exp == 0.0 {
            return Err(EvalError::Underflow);
        }
        let result = exp - log;
        if result.is_finite() {
            Ok(result)
        } else {
            Err(EvalError::Overflow)
        }
    }
}

/// Fallo de evaluación bajo el contrato de reales finitos.
#[derive(Clone, Debug, PartialEq)]
pub enum EvalError {
    /// Alguno de los operandos es infinito o `NaN`.
    NonFiniteInput,
    /// El argumento del logaritmo no es estrictamente positivo.
    LogDomain {
        /// Argumento rechazado.
        value: f64,
    },
    /// Un resultado intermedio o final excede el rango finito.
    Overflow,
    /// La exponencial de un operando finito se redondea a cero.
    Underflow,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput => f.write_str("non-finite operand"),
            Self::LogDomain { value } => write!(f, "logarithm requires y > 0, got {value}"),
            Self::Overflow => f.write_str("floating-point overflow"),
            Self::Underflow => f.write_str("exponential underflow to zero"),
        }
    }
}

impl std::error::Error for EvalError {}
