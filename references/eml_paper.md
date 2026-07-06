# All elementary functions from a single operator

Andrzej Odrzywołek
Institute of Theoretical Physics, Jagiellonian University, 30-348 Krakow, Poland
E-mail: andrzej.odrzywolek@uj.edu.pl

License: CC BY 4.0
arXiv:2603.21852v2 [cs.SC] 04 Apr 2026

## Abstract

A single two-input gate suffices for all of Boolean logic in digital hardware. No comparable primitive has been known for continuous mathematics: computing elementary functions such as sin, cos, √, and log has always required multiple distinct operations. Here we show that a single binary operator,

eml(x, y) = exp(x) − ln(y),

together with the constant 1, generates the standard repertoire of a scientific calculator. This includes constants such as e, π, and i; arithmetic operations including +, −, ×, /, and exponentiation as well as the usual transcendental and algebraic functions. For example, e^x = eml(x, 1), ln x = eml(1, eml(eml(1, x), 1)), and likewise for all other operations. That such an operator exists was not anticipated; I found it by systematic exhaustive search and established constructively that it suffices for the concrete scientific-calculator basis. In EML (Exp-Minus-Log) form, every such expression becomes a binary tree of identical nodes, yielding a grammar as simple as S → 1 | eml(S, S). This uniform structure also enables gradient-based symbolic regression: using EML trees as trainable circuits with standard optimizers (Adam), I demonstrate the feasibility of exact recovery of closed-form elementary functions from numerical data at shallow tree depths up to 4. The same architecture can fit arbitrary data, but when the generating law is elementary, it may recover the exact formula.

## Summary paragraph

Elementary functions such as exponentiation, logarithms and trigonometric functions are the standard vocabulary of STEM education. Each comes with its own rules and a dedicated button on a scientific calculator; our derivations rely on many of them simultaneously, even though we know they are heavily redundant and can be expressed through one another, e.g. sin x = cos(x − π/2), √x = x^(1/2), etc. They are the workhorse of quantitative science, appearing in basic and empirical laws and inside the engines of numerical methods like differential equation solvers, integration quadratures and Fourier analysis [38]. In digital electronics, a remarkable fact underlies universality: a single two-input gate, NAND (the Sheffer stroke), suffices to build any Boolean circuit [43]. Continuous mathematics has lacked such a primitive: calculators must expose many distinct buttons. Classical reductions, from logarithm tables [30, 3] and the slide rule through Euler's formula [7] to the exp-log representation [25] with algebraic adjunctions [40], reduced them to a few, but no further. Despite this, it remains unclear whether this apparent diversity is intrinsic, or whether a smaller generative basis exists. Here we show that the operator eml(x, y) = exp(x) − ln(y), together with the constant 1, does exactly that: it reconstructs arithmetic, all standard elementary transcendental functions, and constants including integers, fractions, radicals, e, π and i. In simpler terms, a two-button calculator (1, eml) suffices for everything a full scientific calculator can do. Existence of the EML operator reveals that elementary functions are members of a much simpler class than previously recognized. Every EML expression is a binary tree of identical nodes, yielding an exceptionally simple grammar: S → 1 | eml(S, S), a context-free language that is isomorphic to well-studied combinatorial objects like full binary trees and Catalan structures. Elementary formulas become circuits [48] composed of identical elements, much like digital hardware built from NAND gates. This uniform representation provides a complete and regular search space for continuous symbolic regression [47, 8]: parameterized EML trees can be optimized by standard gradient methods, and when the generating law is elementary, trained weights can snap to exact closed-form expressions. In effect, a single trainable architecture gains the potential to discover [26] any elementary formula from data. The EML operator may be the tip of an iceberg. Preliminary searches have already returned related operators with even stronger properties, including a ternary variant that requires no distinguished constant.

## Significance statement

Everyone learns many mathematical operations in school: fractions, roots, logarithms, and trigonometric functions (+, −, ×, /, √, sin, cos, log, …), each with its own rules and a dedicated button on a scientific calculator. Higher mathematics reveals that many of these are redundant: for example, trigonometric ones reduce to the complex exponential. How far can this reduction go? We show that it goes all the way: a single operation, eml(x, y), replaces every one of them. A calculator with just two buttons, EML and the digit 1, can compute everything a full scientific calculator does. This is not a mere mathematical trick. Because one repeatable element suffices, mathematical expressions become uniform circuits, much like electronics built from identical transistors, opening new ways to encoding, evaluating, and discovering formulas across scientific computing.

## 1 Introduction

Single, reusable primitives play a disproportionately large aesthetic and practical role in mathematics, engineering, and even biology. Widely known classical examples include the NAND gate (and its dual, Peirce Arrow, logical NOR) for Boolean 0/1 logic [43, 27], the operational amplifier [2] for positive and negative feedback processes, and, more recently, the rectified linear unit (ReLU) "ramp" activation function [29] in deep learning [16]. We also mention Wolfram's single axiom [50], K, S combinators from combinatory logic [41, 11], Interaction Combinators [21], and fuzzy versions of the Sheffer stroke [1]. Other well-known examples are one-instruction set computers (OISC), e.g. SUBLEQ [28], Conway's FRACTRAN [4] and the Rule 110 cellular automaton [50, 5].

Whether the existence of a single sufficient operator or element is conceptually crucial is disputed [24]. Nevertheless, its value for understanding, pedagogy, and public communication is substantial. Indeed, classical first-order logic does not need to single out NAND at all: it works perfectly well with a whole redundant family of connectives (XOR, AND, NOT, …) that we introduce for convenience in applications. But the realization that all of them are definable from a single primitive is a striking and conceptually deep structural fact. A similar kind of realization accompanies the recognition of DNA and RNA [10] as a nearly universal information carrier in evolutionary biology.

Sheffer-type elements are rare, and mining them typically requires time, compute, effort, and a bit of luck. They also appear in seemingly distant areas, for example in discrete geometry. The recently discovered solution to the "einstein" problem had a major impact on the tiling community [44]. The present author already has one such element in his portfolio, namely an igloo construction brick derived from the (2,3,5) spherical Möbius triangle [33].

Elementary functions, for many students epitomized by the dreaded _sine_ and _cosine_, play a central role in quantitative reasoning. They can be combined in countably many ways with integers, rational numbers, and mathematical constants such as π and e, using the four basic arithmetic operations (+, −, ×, /) and exponentiation. Elementary functions are at the core of STEM education and at the foundation of modern technological civilization. This includes not only simple formulas or empirical models, but also most numerical algorithms, for example ordinary differential equation solver RKF45 [14], integration rules (Gaussian, Tanh-Sinh, …) [45, 38], and the Fast Fourier Transform [6], whose twiddle factors are elementary functions. Finite expressions in elementary functions are supported in virtually all modern programming languages. While it is generally known that most standard functions and arithmetic operations are heavily redundant (e.g. tan x = sin x / cos x, √x = x^(1/2), arsinh x = ln(x + √(1 + x^2)), x/y = x × y^(−1) etc.), especially in the complex domain (e.g. cosh x = cos(ix), sinh ix = i sin x), they have never been regarded as candidates for a single primitive operator. Historically, two major milestones in understanding elementary arithmetic operations from this point of view are (i) the discovery of logarithms and the subsequent creation of tables [30, 3] followed by slide rule, and (ii) Euler's formula e^(iπ) + 1 = 0 [7]. Logarithms were introduced to reduce multiplication to addition. The exp-log pair allows them to be expressed in terms of one another:

x × y = e^(ln x + ln y),    x + y = ln(e^x × e^y).    (1)

Euler's formula

e^(iφ) = cos φ + i sin φ    (2)

shows that, once the imaginary unit i = √(−1) is introduced, trigonometric functions can be viewed as another face of 'exp' and 'ln' [25]. This perspective leads naturally to what scientists call exp–log functions, namely finite expressions built from variables, named constants, arithmetic operations, together with exp and log, in the spirit of differential and computer algebra [42, 39]. In the classical differential-algebraic setting, one often works with a broader notion of elementary function, defined relative to a chosen field of constants and allowing algebraic adjunctions [40], i.e., adjoining roots of polynomial equations (cf. `Root` in Wolfram Language [18]). That level of generality is not needed here. The present paper takes the ordinary scientific-calculator point of view: start from a concrete list of familiar constants, functions and operations, and ask how far they can be reduced without losing practical functionality. The precise starting list is given later in Table 1.

Many reductions inside that list are, of course, classical and well known; what seems not to have been investigated systematically is the endpoint of the process, namely reduction to a single binary operator paired with a single distinguished constant. In particular, the so-called "broken calculator" problem [15] is a popular formulation of computations with a reduced set of available keys or operations. I used this approach to construct a sequence of increasingly "primitive" yet fully functional calculators with 4, 3, 2, and finally a single operator. This operator is the EML (Exp–Minus–Log),

eml(x, y) = exp(x) − ln(y).    (3)

Using the EML, a surprisingly simple binary operator, we can express any standard real elementary function in terms of repeated applications of (3) to the input variables x, y, z, … and a single distinguished constant, 1. This constant is needed to neutralize the logarithmic term in (3) via ln(1) = 0. Computations must be done in the complex domain, e.g: generating constants like i and π requires evaluating ln(−1), so eml(x, y) internally operates over ℂ using the principal branch.

If we do not need external input variables, e.g. for use in an actual pocket calculator, two buttons are sufficient to retain full functionality: one binary operator, EML, and one terminal symbol, 1. No further reduction of operator count is possible, because at least one binary operator and at least one terminal symbol are required. The existence of such a binary operator, which is itself an elementary function, is somewhat unexpected.

In fact, the EML Sheffer operator (3) is as simple as it appears, and in principle the article could end here; its consequences would eventually surface on their own. Nevertheless, it seems worthwhile to explain the methods used in the search for it (Sect. 2), to present the intermediate and final search results (Sect. 3), and to outline potential applications (Sect. 4) that are already visible on the horizon. The article concludes with a short summary, possible follow-up directions, and open questions.

## 2 Methods

I employed systematic 'ablation' testing to identify minimal operator sets for a fixed scientific-calculator starting list. The list, given in Table 1, contains 36 primitives: named constants and input variables, standard unary functions, and binary operations. I iteratively removed one element from this list (constant, function or binary operation) and verified whether the remaining set could still reconstruct original primitives. The Mathematica/Wolfram core language [18] instruction set (Table 2, 2nd row) served as a reference for an already-optimized and thoroughly tested for almost 40 years minimalist system. Research started out of curiosity whether this system could be reduced further.

**Table 1:** Starting list of constants/variables, functions (unary) and binary operators used to initiate the reduction procedure. This table also fixes the concrete scientific-calculator notion of "elementary function" used throughout the paper: finite expressions built from these named symbols. The exp(x) = e^x, ln x is natural logarithm, inv(x) = 1/x denotes reciprocal, minus(x) = −x is a sign flip. Square is denoted by sqr(x) = x^2 and σ(x) = 1/(1 + e^(−x)) is the logistic sigmoid. Trigonometric and hyperbolic functions with their inverses have usual meaning. In addition to four basic arithmetic binary operations we use arbitrary base logarithm log_x y, exponentiation pow(x, y) = x^y, arithmetic mean avg(x, y) = (x + y)/2 and hypotenuse hypot(x, y) = √(x^2 + y^2).

| Type | Elements | Count |
| --- | --- | --- |
| Constants | π, e, i, −1, 1, 2, x, y | 8 |
| Functions | exp, ln, inv, half, minus, √, sqr, σ, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh, arsinh, arcosh, artanh | 20 |
| Operations | +, −, ×, /, log, pow, avg, hypot | 8 |
| Total |  | 36 |

An operator set was deemed "complete" if it could reconstruct all primitives from Table 1, on the real axis where appropriate. This includes trigonometric functions and their inverses (sin, cos, tan, arcsin, …), hyperbolic functions and their inverses (sinh, cosh, tanh, arsinh, …), algebraic operations (√, reciprocal, …), and fundamental constants (π, e, i, −1, 0, 1, 2, …).

The challenge of executing such a test is illustrated by the following simple exercise from rational functions generation, attributed to [32]. Given only the three operations: suc(x) = x + 1 (successor), pre(x) = x − 1 (predecessor), and inv(x) = 1/x (reciprocal/inverse), let us compute negation, i.e., −x. The non-obvious solution is:

suc(inv(pre(inv(suc(inv(x))))) = 1 / (1 / (1/x + 1) − 1) + 1 = −x.

This exemplifies the formulas expected to be encountered in our search: nested expressions with depth 5-9 that defy intuitive construction. There is no regular method to find them automatically, except for (human-assisted) brute-force search, nowadays also enhanced using AI [31].

Direct symbolic verification proved computationally intractable. Typical Kolmogorov-style complexity K [20] is usually K ≲ 7. In practice it was searched up to K = 9. Here K denotes the length of a Reverse Polish Notation (RPN) calculator program, equivalent to a formula. To speed up the search I developed a hybrid numeric bootstrapping verification approach, working as follows. We test whether a target operation, e.g. hypotenuse √(x^2 + y^2), can be expressed using a given operator set. Instead of symbolic manipulation on x, y, I substituted for x and y algebraically independent (not in exp-log class) transcendental constants, such as e.g., x = γ ≃ 0.577216 (Euler-Mascheroni constant), y = A ≃ 1.28243 (Glaisher-Kinkelin constant). Then, √(γ^2 + A^2) was evaluated numerically, and compared to all expressions generated from 1 and (3). This method follows from the simple observation that any generally valid formula must also be valid at a single point. Under the Schanuel conjecture [46], coincidental equality between such expressions is vanishingly unlikely, making this a reliable sieve for formula equivalence candidates. In practice, I compute the double-precision numerical value of the operator sought, and then run constant recognition software ("inverse symbolic calculator") on this result, which returns a candidate formula in the form of RPN calculator code [35] or as an expression tree, depending on the variant of the method used. The search procedure is heuristic and serves only to exclude evidently false formulas and discover candidate witnesses; independent verification is separate and is given in Supplementary Information (Part II), which provides symbolic checks, numerical cross-validation, and a constructive completeness proof sketch for the Table 1 class.

In detail, the search procedure worked as follows. Rather than searching for all elementary operations (Table 1) directly in pure EML form (which remains computationally infeasible even numerically), I employed iterative bootstrapping:

1. Start with two lists:
   - set of constants, functions and operators to be verified for robustness, e.g. S0 = {1, eml}
   - set of constants, functions and operators to be computed, e.g., C0 = {π, e, i, −1, … exp, ln, …, +, −, ×, …} cf. Table 1. The list C0 also defines what I mean by 'computing _all_ elementary functions'.
2. Search for a expression computing some element from the list Ci using only primitives from the list Si.
3. If one is found, move it from list Ci to list Si+1. In the EML example, the first result found is the formula for e = eml(1, 1). After the first step, the lists become S1 = S0 ∪ {e} = {1, eml, e}, C1 = C0 \ {e} = {π, i, −1, … exp, ln, …, +, −, ×, …}.
4. Repeat steps 2-3 until all primitives from initial list C0 are reconstructed, i.e. until Ci becomes empty: Ci = ∅.

The verification procedure, VerifyBaseSet, was implemented using my own Mathematica SymbolicRegression package [35]. A version that is three orders of magnitude faster was recently translated by GPT Codex 5.3 into Rust [35], allowing for re-check of EML in seconds, rather than hours.

The ablation process, i.e. running above search with some of the elements from Table 1 removed, yielded progressively smaller calculator configurations (named Calc 3, 2, 1, 0 in Table 2), each requiring different primitives, constants in particular. Some configurations could generate required constants from arbitrary input on their own (via operations like x − x = 0, e^0 = 1, 0 − 1 = −1, … in Calc 2), while others required specific constants (e.g., Calc 1 requires e or π, Wolfram requires i, cf. footnote in Table 2). Then the search stalled, and it became evident that the continuous Sheffer operator, if it exists, is not among the familiar named functions. For the final reduction, I began, guided by intuition gathered over many experiments, to enumerate elementary binary functions as candidate single operators, paired with similarly generated constants. Each candidate pair { constant, operator } was tested using the VerifyBaseSet procedure. This, after numerous failures, and a few discarded false-positives, revealed that the eml(x, y), given by (3), paired with the constant 1, forms a complete basis for reconstructing the elementary functions. Later I realized that EML is not unique, discovering its close cousins: EDL (requiring constant e) and a third variant using −∞ as a terminal symbol. Details are the subject of the next section.

## 3 Results

**Figure 1:** Bootstrapping "phylogenetic" tree of the elementary functions found from EML, (3), as LUCA (Last Universal Common Ancestor) and 1. Spiral unwinds according to subsequent found primitives, and arrows show from which elements it was composed of. Those using EML and 1 directly are marked using thick arrows. For full adjacency matrix, and entire discovery chain, see Supplementary Information, Fig. S1 and Table S2.

**Table 2:** Progressive reduction from 36-button scientific calculator from Table 1 to the EML Sheffer operator. Calc 3, 2 and 0 accept any real constant or variable as input, generating all required values internally. The count-down column tallies all distinct primitives including one input variable; for systems marked "none" under Constants, any real value serves as the starting input. Wolfram Mathematica primitives are given for reference. Last lines are placeholders for speculated undiscovered primitives. The penultimate row is for a binary operator, hypothesized to generate, unlike EML, constants from arbitrary input. The last row is for a unary operator, which retains all good properties of well-tested activation functions in neural nets, while allowing for exact evaluation of elementary functions when combined with standard "matrix" arithmetic.

| | Constants | Unary | Binary | Count-down |
| --- | --- | --- | --- | --- |
| Base-36 | cf. Table 1 | cf. Table 1 | cf. Table 1 | 36 |
| Wolfram* | π, e, i | ln | +, ×, ∧ | 7 |
| Calc 3 | none | exp, ln, −x, 1/x | + | 6 |
| Calc 2 | none | exp, ln | − | 4 |
| Calc 1 | e or π | none | x^y, log_x y | 4 |
| Calc 0 | none | exp | log_x y | 3 |
| EML | 1 | none | eml(x, y) | 3 |
| ? | none | none | ? | 2 |
| ? | none | ? | +, −, × | 5 |

* The imaginary unit alone is sufficient: −1 = i × i, 1 = −1 × (−1), 0 = −1 + 1. π = −1 × i × ln(−1), e = (−1)^((iπ)^(−1)).

Table 2 presents the complete reduction sequence, in historical order. Each configuration represents a complete 'scientific calculator' capable of computing every expression built from the primitives listed in Table 1. Calc 3 (Table 2, 3rd row) was first to dethrone Wolfram Language primitives set (Table 2, 2nd row) by retaining negation and reciprocal alongside exp, ln, and addition. This system of 6 primitives is able to generate constants on its own via x + (−x) = 0, etc. Calc 2 achieved what Calc 3 does using only exp, ln and subtraction, while still preserving ability to generate constants on its own. The non-commutative operator (subtraction) proved crucial, as it provides both expression-tree growth and inversion capabilities.

Calc 1 represents a fundamentally different, top-down approach, using binary exponentiation (a rank-3 hyper-operation) and its inverse (binary logarithm) as base operators instead of lower-rank operations such as addition and subtraction. This configuration works with e or π as the terminal constant. Despite extensive searches, no other constant was found for which Calc 1 works.

Calc 0 absorbs the constant e into the exp function itself, reducing the system to 3 primitives. This configuration strongly suggested that a single binary operator might exist at all, motivating further search.

The final reduction to EML was achieved by recognizing a pattern: all minimal configurations involve pairs of inverse functions (including self-inverses) and non-commutative operations. Testing combinations of inverse functions at the input with asymmetric operations at the output yielded the first continuous sufficient operator. A month later I realized that it has at least two additional cousins: EDL and -EML:

eml(x, y)  = exp(x) − ln(y)   with constant 1,    (4a)
edl(x, y)  = exp(x) / ln(y)   with constant e,    (4b)
−eml(y, x) = ln(x) − exp(y)   with constant −∞.   (4c)

The first successful discovery search run (cf. Fig. 1) can be replicated with this 3-line Mathematica code:

```
Import["SymbolicRegression.m"]
EML[x_, y_] := Exp[x] - Log[y]
VerifyBaseSet[{1}, {}, {EML}]
```

where the package can also be imported directly from the repository [35]. Depending on computer speed, usually in less than an hour, the above procedure re-generates all 36 elementary operations from Table 1. For example, the natural logarithm becomes:

ln(z) = eml(1, eml[eml(1, z), 1]),    (5)

and so on. The resulting EML expressions range from depth 1 (exponential, e^x = eml(x, 1)) to depth 8 (multiplication), with most basic math functions requiring larger depths (Table 4). A much faster and more thorough test (multiple real transcendentals, both positive and negative, arbitrary-precision check) is provided by a Rust re-implementation of VerifyBaseSet [35], rust_verify.

The exhaustive search served only to discover candidate identities. Their verification is deferred to Supplementary Information, Part II, which contains symbolic simplification of the full discovery chain (Fig. 1) in Wolfram Mathematica [18], independent numerical cross-checks across four implementations (C, NumPy, PyTorch, and mpmath), and a constructive completeness proof sketch.

## 4 Usage and applications

The uniform tree structure of EML expressions suggests several directions for implementation and application.

### 4.1 EML compiler

The output of the VerifyBaseSet procedure provides the data (see Fig. 1) required to reconstruct any primitive or composite elementary expression in terms of EML Sheffer, (4a). I provide a prototype EML compiler, coded in Python, that converts formulas into pure EML form. An EML expression can be evaluated symbolically in Mathematica, or numerically in any IEEE754-compliant language. Pure EML form could also be executed on hardware (or an emulated machine) that has only a single instruction: the EML itself. In particular, the EML code can be executed on a single instruction stack machine, closely resembling a single-button RPN calculator. Pure-EML form could possibly be implemented efficiently in FPGA or analog circuits.

The simplest input-output example is provided by the ln x function. After compilation to EML form, we expect to obtain (5), although the simplest possible form is not expected in general. The equivalent RPN code for ln is a sequence of K = 7 instructions

{1, 1, x, eml, 1, eml, eml},

or, denoting eml → E, as the RPN string 11xE1EE. For the expression tree, see Fig. 2 (on top).

A few comments on the implementation are required. EML-compiled expressions work on the real axis, both positive and negative, except for a few isolated points, especially at zero and domain endpoints. Internal computations, for trigonometric functions in particular, must be done in the complex domain. Because the simplest form of the natural logarithm, (5), obtained from EML, (3), is equivalent to

ln z = e − log(e^e / z),

using the standard choice of principal branch for complex logarithm, we obtain a jump of 2πi for the negative real axis, due to the 1/z term. This prevents use of e.g. some standard formulas for i, relying on ln(−1) = iπ, for which we get the wrong sign. A solution working for all real x ≠ 0 is to redefine the branch for EML itself in such a way that ln z (and everything derived from it, cf. Fig. 1) follows standard implementation of principal branch. Another option, used in EML compiler, is to manually correct i sign.

EML-compiled formulas work flawlessly in symbolic Mathematica and IEEE754 floating-point, e.g. `<math.h>` in C. This is because some formulas internally might rely on the following properties of extended reals:

ln 0 = −∞,    e^(−∞) = 0.

These are properly implemented in Mathematica using symbolic processing, and in floating point using inf and signed zeros. But EML expressions in general do not work 'out of the box' in, e.g., pure Python/Julia or numerical Mathematica. In the first case, this is because special floats are trapped and raise errors. However, EML works in NumPy [17] and PyTorch [36]. In Mathematica, we have an automatically extensible range of floats leading to Overflow[]. Interestingly, the Lean 4 proof assistant [12] takes a different approach. Because Lean requires all functions to be total, it assigns the complex logarithm at zero a default "junk value" (Complex.log 0 = 0), causing the straightforward formalization of the EML chain to fail. As a bottom line I stress that all the above difficulties (edge cases) are not much different from those usually encountered in every kind of floating-point or symbolic computation. EML compiler is available for testing under EML_toolkit/EmL_compiler subdir of [35]; see also Supplementary Information, Sect. 4. Software and reproducibility.

### 4.2 Elementary functions as binary trees and analog circuits

Noteworthy, in EML notation, any elementary function expression tree is binary. The context-free grammar is trivial: S → 1 | eml(S, S). For functions, input variables are added as additional terminal symbols (e.g. x in the univariate case). This has many practical advantages over standard methods.

**Figure 2:** Examples of binary EML trees equivalent to few important simplest formulas.

Some simple examples of tree/circuit representations are shown in Fig. 2. The examples shown are natural logarithm, identity, negation minus(x) = −x, reciprocal inv(x) = 1/x, and multiplication. Ability to compute the identity function using an EML tree of depth 4 allows some input variables to be moved down the tree (see next Subsection). Other elementary functions, e.g. trigonometric ones, have trees too large to be shown in print, cf. Table 4.

**Table 3:** EML Sheffer as a new kind of basic building block for symbolic and analog computations. Dot marks output, and, as (3) is non-commutative, arrow determine chirality and operation order. First counterclockwise input after dot is exp x, then ln y, and dot EML "perform" subtraction. In Fig. 2 we keep this convention to avoid confusion, but clockwise variant is valid as well. I'm following habits of OpAmp symbol usage here.

| NAND gate | Op-Amp | Transistor | EML Sheffer |
| --- | --- | --- | --- |
|  | + − |  |  |

Circuits using EML operator as a new element (Table 3) might be useful for analog computing [48]. One of the old problems in this field is construction of predefined multivariate elementary functions [49]. Using EML compiler (Sect. 4.1), we can convert any expression (Fig. 2) to such a circuit, with the topology of a binary tree. EML tree provides a uniform treatment of generic elementary functions.

**Table 4:** Complexity of various functions in EML tree representation. EML Compiler column gives RPN code length K for expressions generated from EML compiler. The value of K of EML formula can be computed using e.g. Mathematica LeafCount. For the identity function x, the compiler returns x directly (leaf count 1); the shortest non-trivial EML expression have leaf count 9. Last column show results of direct exhaustive search for shortest expressions. Numbers in parentheses show length of formulas which do not use the extended reals (±inf in floating-point). If search timed out, reached lower limit for K is given.

| Constant | EML Compiler | Direct search |
| --- | --- | --- |
| 1 | 1 | 1 (1) |
| 0 | 7 | 7 (7) |
| −1 | 17 | 15 (17) |
| 2 | 27 | 19 (19) |
| −2 | 43 | 27 (27) |
| 1/2 | 91 | 29 (35) |
| −1/2 | 107 | 31 (37) |
| 2/3 | 143 | 39 (39) |
| −2/3 | 159 | 45 (47) |
| √2 | 165 | >>47 |
| i | 131 | >>55 |
| e | 3 | 3 |
| π | 193 | >>53 |

| Function | EML Compiler | Direct search |
| --- | --- | --- |
| x | 1 | 9 |
| e^x | 3 | 3 |
| ln x | 7 | 7 |
| −x | 57 | 15 |
| 1/x | 65 | 15 |
| x − 1 | 43 | 11 |
| x + 1 | 27 | 19 |
| x/2 | 131 | 27 |
| 2x | 131 | 19 |
| √x | 139 | 43 ≥?>35 |
| x^2 | 75 | 17 |

| Operator | EML Compiler | Direct search |
| --- | --- | --- |
| x − y | 83 | 11 (11) |
| x + y | 27 | 19 (19) |
| x × y | 41 | 17 (17) |
| x/y | 105 | 17 (17) |
| x^y | 49 | 25 |
| log_x y | 117 | 29 |
| (x + y)/2 | 287 | >>27 |
| √(x^2 + y^2) | 175 | >>27 |

### 4.3 Symbolic Regression by continuous optimization

Modern symbolic regression (SR) methods [47, 8, 9] aim to discover closed-form expressions from data, but they typically search over heterogeneous grammars involving many distinct operators. Knowledge of a single operator, (3), which is sufficient to compute any elementary function, allows us to create a multiparameter "master" formula. Such a master formula, easily constructed in EML form due to binary expression tree, for some combinations of its parameters, is equivalent to elementary functions. By construction, such a general tree includes all possible formulas up to the given leaf count (tree depth). These trees are big by the standards of a typical mathematical analysis course, but small compared to what is in use in modern AI. For example, a full binary tree of depth n has a total of 2^n − 1 branches and 2^n leaves. For the largest transformers with trillions (10^12) of parameters, the equivalent tree would have a depth of 40. The large values in Table 4 reflect the unoptimized prototype EML compiler (Subsect. 4.1); direct exhaustive search yields substantially shorter expressions, as the rightmost column demonstrates.

The master formula can be constructed as follows. For simplicity we present the univariate case; the construction extends to an arbitrary number of input variables. Both inputs to eml(x, y) can be: either 1, input variable x, or the result from the previous eml, which we denote f. Let us represent such a general input by a linear combination

α_i + β_i x + γ_i f.    (6)

Now, setting specific values, we can recover all three cases

- 1, for α_i = 1, β_i = 0, γ_i = 0,
- x, for α_i = 0, β_i = 1, γ_i = 0,
- f, for α_i = 0, β_i = 0, γ_i = 1.

At lowest tree level (leaves), there are only 1 and x, so there are no parameters γ_i.

As an example, I present full level-2 master formula

F(x) = eml[
    α_1 + β_1 x + γ_1 eml(α_3 + β_3 x, α_4 + β_4 x),
    α_2 + β_2 x + γ_2 eml(α_5 + β_5 x, α_6 + β_6 x)
].

If we set α_1 = 0, β_1 = 1, γ_1 = 0 and α_2 = 1, β_2 = γ_2 = 0, we recover exp(x). Using α_1 = α_2 = 1 and all β_i = γ_i = 0 we get the constant e. Setting α_1 = β_1 = 0, γ_1 = 1, α_2 = 1, β_2 = γ_2 = 0, α_3 = 0, β_3 = 1, γ_3 = 0 and α_4 = 1, β_4 = γ_4 = 0, we obtain double exponential exp(e^x). The total number of parameters for the level-2 master formula is 14. In general, the level-n master formula has 5 × 2^n − 6 parameters, see Supplementary Information, Part III.

While one could, in principle, create such a master formula using more usual elementary functions (+, -, /, **, sin, cos, …, etc.), it would be ridiculously complicated and would lack any regular structure. Moreover, in practical SR one typically works with a reduced subset of operations, running the risk that the chosen set is incomplete, i.e., unable to express all elementary functions. The EML master formula is complete by design.

Number of parameters in (6) can be reduced to two using some switch function. Or, alternatively, in a more modern approach, one can treat parameters α_i, β_i, γ_i as logits, and pass them through softmax function [19] to convert them into probabilities normalized to α_i + β_i + γ_i = 1. In two examples below, I'll use both methods.

The simplest proof of concept is provided by fitting some numerical data obtained from example elementary function, the ln x, using the complete level 3 binary tree with the above structure. Using simplex reparameterization, the number of free parameters reduces from 34 to 20. I managed to find all weights using plain NMinimize, a 'black-box' optimizer from Mathematica [18]. The mean squared fitting error is at the level of numerical precision, and the resulting formula, after rounding weights to the nearest vertex of the simplex (i.e. snapping each to 0 or 1), is exactly ln x. Not only in the provided data range, but also beyond that. Generalization/extrapolation is nearly perfect in the above example. See Wolfram Mathematica notebook Log_fit.nb from EML toolkit [35] and Supplementary Information (Sect III.B) for details.

In practice, such a naive one-liner approach does not scale. Therefore, I performed several experiments using more recent techniques of Machine Learning. Simple functions of two variables, taken from composed EML, such as ln(e − ln(e^x − ln y)), were used as targets. Python code was created, which used DTYPE = torch.complex128 data type. Main issues encountered during training of EML net were related to range overflow due to multiply composed exponentials, as well as floating-point errors (NaN) caused by the specific implementation of complex arithmetic. They can be eliminated by clamping arguments and values for exp(x), and careful (i.e. without impairing torch automatic differentiation) inspection of both real and imaginary parts. Parameters from (6) were treated as logits. Optimization was a multi-stage process: first, usual training using a stochastic gradient optimizer (Adam), then a hardening phase pushing weights in (6) towards 0 or 1. Finally, weights were clamped to exact symbolic values. Systematic experiments (over 1000 runs with varied seeds and initialization strategies) show that blind recovery from random initialization succeeds in 100% of runs at depth 2, approximately 25% at depths 3–4, and below 1% at depth 5. At depth 6, no blind recovery was observed in 448 attempts. When successful, the snapped weights yield mean squared errors at the level of machine epsilon squared (~10^(−32)), consistent with exact symbolic recovery. Noteworthy, when the weights of the correct EML tree are perturbed by Gaussian noise, the optimization converges back to the exact values in 100% of runs, even for trees of depth 5 and 6. This demonstrates that the EML tree representation is valid and that the correct basins of attraction exist. Finding them from random initialization becomes harder with depth. If one manages to improve convergence of EML trees beyond proof of concept, possibly using another binary operator similar to (4) but with better properties (non-exponential asymptotics, no domain issues), then we could achieve symbolic regression of data using gradient-based methods [37]. See Supplementary Information for systematic training experiments, code and details.

## 5 Conclusions and open questions

The operator EML, (3), provides a single sufficient primitive from which real elementary functions can be constructed and evaluated. Consequently, a wide class of computations built from such functions can also be cast in EML form. It is not unique; several close variants are likely to exhibit similar properties, including EDL, (4b), and the swapped-argument form −eml(y, x), (4c). More operators of this kind exist. Perhaps an entire continuous family of them awaits discovery, with properties more convenient than (3). For example, the requirement for one of the constants: −∞, 1, e, … to be always present among terminal symbols makes its use less elegant and more complicated (cf. Subsect. 4.3) in comparison with, e.g., standard neural nets or the NAND gate. The latter is able to generate 0s and 1s out of 'anything'. The EML operator does not have this useful property. Whether an EML-type binary Sheffer working without pairing with a distinguished constant exists is an open question. Proving such impossibility for any given candidate is non-trivial: one might expect f(x, x) being constant to suffice, but consider B(x, y) = x − y/2, for which B(x, x) = x/2 yet B(B(x, x), x) = 0. Such traps illustrate why systematic search is essential in this work. A ternary operator, T(x, y, z) = e^x / ln x × ln z / e^y, for which T(x, x, x) = 1 is next candidate for further analysis [34].

Whether a univariate Sheffer exists, serving simultaneously as a neural activation function [13] and as a generator of all elementary functions, remains open (see SI, Sect. 5).

One might complain that the EML representation of elementary functions requires complex arithmetic for real math, at least internally. Just as quantum computing uses complex amplitudes to compute real probabilities, EML uses complex intermediates to compute real elementary functions. This seems inevitable. We must somehow compute the imaginary unit i, π, and all trigonometric/hyperbolic functions via Euler's formula, (2). For that, we use ln x for x < 0. A continuous Sheffer working purely in the real domain seems impossible. My search for alternatives, e.g., using pairs of trigonometric/hyperbolic functions and their inverses instead of exp/ln, found nothing. Quite surprisingly, the requirement to use complex numbers internally causes only minor problems in practice of using (3) in Computer Algebra Systems or numerical simulations.

Since standard activation functions are themselves elementary, any conventional neural network is a special case of an EML tree architecture. Current networks can learn symbolic algebra [22] and digit-level arithmetic [23], but their internal mechanisms remain opaque [26], and efficient exact evaluation of elementary functions as continuous real-valued operations is still beyond their reach. EML representations go further: as demonstrated in Subsect. 4.3, trained weights can snap to exact binary values, recovering closed-form elementary subexpressions alongside approximations. When this succeeds, the discovered circuits are legible as elementary function expressions — a form of interpretability unavailable to conventional architectures.

### AI use disclosure

The core idea, the discovery of the EML Sheffer operator, the verification methodology, and results are entirely the author's own work. Large language models (including recent Claude, Grok, Gemini and ChatGPT) were used mainly for language editing and coding assistance.

### Data availability

Code, scripts, and supplementary reproducibility materials used to generate the figures, tables, and numerical results are available in the SymbolicRegressionPackage repository [35]. An archival snapshot of the exact submission version is deposited in Zenodo DOI: 10.5281/zenodo.19183008 (https://doi.org/10.5281/zenodo.19183008). The archival package contains the source code, scripts, and README files needed to rerun the reported results.

### Acknowledgments

Computational resources were partially provided by Google Cloud Research Credits.

### Supplementary Information

Extensive three-part Supporting Information is supplied as a separate SI Appendix PDF.

## References

- [1] M. Baczyński, P. Berruezo, P. Helbin, S. Massanet, W. Niemyska, and D. Ruiz-Aguilera (2022). On the sheffer stroke operation in fuzzy logic. Fuzzy Sets and Systems 431, pp. 110–128. Note: Logic and Related Topics. External Links: ISSN 0165-0114, Document, Link. Cited by: §1.
- [2] H. S. Black (1934). Stabilized feedback amplifiers. The Bell System Technical Journal 13 (1), pp. 1–18. External Links: Document. Cited by: §1.
- [3] H. Briggs (1624). Arithmetica logarithmica. London. Cited by: Summary paragraph, §1.
- [4] J. H. Conway (1987). FRACTRAN: a simple universal programming language for arithmetic. In Open Problems in Communication and Computation, T. M. Cover and B. Gopinath (Eds.), pp. 4–26. External Links: ISBN 978-1-4612-4808-8, Document, Link. Cited by: §1.
- [5] M. Cook (2004). Universality in elementary cellular automata. Complex Systems 15 (1), pp. 1–40. External Links: Document. Cited by: §1.
- [6] J. W. Cooley and J. W. Tukey (1965). An algorithm for the machine calculation of complex Fourier series. Mathematics of Computation 19 (90), pp. 297–301. External Links: Document. Cited by: §1.
- [7] R. Cotes (1722). Harmonia mensurarum, sive analysis & synthesis per rationum & angulorum mensuras promotae: accedunt alia opuscula mathematica. Cambridge. Note: Posthumous; edited by Robert Smith. Cited by: Summary paragraph, §1.
- [8] M. Cranmer, A. Sanchez-Gonzalez, P. Battaglia, R. Xu, K. Cranmer, D. Spergel, and S. Ho (2020). Discovering symbolic models from deep learning with inductive biases. Advances in Neural Information Processing Systems 33, pp. 17429–17442. Cited by: Summary paragraph, §4.3.
- [9] M. Cranmer (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. External Links: 2305.01582. Cited by: §4.3.
- [10] F. H. C. Crick and J. D. Watson (1953). Genetical implications of the structure of deoxyribonucleic acid. Nature 171 (4361), pp. 964–967. External Links: Document. Cited by: §1.
- [11] H.B. Curry and R. Feys (1958). Combinatory logic. Combinatory Logic, North-Holland Publishing Company. External Links: LCCN lc59001593, Link. Cited by: §1.
- [12] L. de Moura and S. Ullrich (2021). The lean 4 theorem prover and programming language. In Automated Deduction – CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings, Berlin, Heidelberg, pp. 625–635. External Links: Document, Link. Cited by: §4.1.
- [13] S. R. Dubey, S. K. Singh, and B. B. Chaudhuri (2022). Activation functions in deep learning: a comprehensive survey and benchmark. Neurocomputing 503, pp. 92–108. External Links: Document. Cited by: §5.
- [14] E. Fehlberg (1970). Klassische runge-kutta-formeln vierter und niedrigerer ordnung mit schrittweiten-kontrolle und ihre anwendung auf wärmeleitungsprobleme. Computing 6 (1), pp. 61–71. External Links: ISSN 1436-5057, Document, Link. Cited by: §1.
- [15] D. Finkel (2017). Broken calculator warm up. Note: https://mathforlove.com/lesson/broken-calculator-warmup/ Accessed 2025-11-18. Cited by: §1.
- [16] I. Goodfellow, Y. Bengio, and A. Courville (2016). Deep learning. MIT Press, Cambridge, MA. External Links: ISBN 9780262035613, Link. Cited by: §1.
- [17] C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk, M. Brett, A. Haldane, J. F. del Río, M. Wiebe, P. Peterson, P. Gérard-Marchant, K. Sheppard, T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, and T. E. Oliphant (2020-09). Array programming with NumPy. Nature 585 (7825), pp. 357–362. External Links: Document. Cited by: §4.1.
- [18] W. R. Inc. (2025). Mathematica, Version 14.3. Note: Champaign, IL, 2025. External Links: Link. Cited by: §1, §2, §3, §4.3.
- [19] E. Jang, S. Gu, and B. Poole (2017). Categorical reparameterization with Gumbel-Softmax. In Proceedings of the 5th International Conference on Learning Representations (ICLR), External Links: Link. Cited by: §4.3.
- [20] A. N. Kolmogorov (1965). Three approaches to the quantitative definition of information. Problems of Information Transmission 1 (1), pp. 1–7. Cited by: §2.
- [21] Y. Lafont (1997-08). Interaction combinators. Inf. Comput. 137 (1), pp. 69–101. External Links: ISSN 0890-5401, Link, Document. Cited by: §1.
- [22] G. Lample and F. Charton (2020). Deep learning for symbolic mathematics. In Proceedings of the 8th International Conference on Learning Representations (ICLR), External Links: Link. Cited by: §5.
- [23] N. Lee, K. Sreenivasan, J. D. Lee, K. Lee, and D. Papailiopoulos (2024). Teaching arithmetic to small transformers. In The Twelfth International Conference on Learning Representations (ICLR), External Links: Link. Cited by: §5.
- [24] B. Linsky (2011). The evolution of principia mathematica: bertrand russell's manuscripts and notes for the second edition. Cambridge University Press. External Links: ISBN 9781139497336, Link. Cited by: §1.
- [25] J. Liouville (1835). Mémoire sur l'intégration d'une classe de fonctions transcendantes. Journal für die reine und angewandte Mathematik (Crelle's Journal) 13, pp. 93–118. External Links: Document. Cited by: Summary paragraph, §1.
- [26] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljačić, T. Y. Hou, and M. Tegmark (2025). KAN: Kolmogorov–Arnold networks. In International Conference on Learning Representations (ICLR), External Links: Link. Cited by: Summary paragraph, §5.
- [27] J. Łukasiewicz (1963). Elements of mathematical logic. Pergamon Press, Oxford. Note: English translation of Elementy logiki matematycznej (1929). Cited by: §1.
- [28] O. Mazonka and A. Kolodin (2011). A simple multi-processor computer based on subleq. External Links: 1106.2593, Link. Cited by: §1.
- [29] V. Nair and G. E. Hinton (2010). Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML 2010), pp. 807–814. Cited by: §1.
- [30] J. Napier (1614). Mirifici logarithmorum canonis descriptio. Andrew Hart, Edinburgh. Cited by: Summary paragraph, §1.
- [31] B. Naskrecki and K. Ono (2025-10). Mathematical discovery in the age of artificial intelligence. Nature Physics 21 (10), pp. 1504–1506. External Links: Document, Link. Cited by: §2.
- [32] D. J. Newman (1982). A problem seminar. Sources in the History of Mathematics and Physical Sciences, Springer-Verlag, New York. External Links: ISBN 0-387-90765-3. Cited by: §2.
- [33] A. Odrzywolek (2014). How to build the perfect igloo. Eureka 63, pp. 1001110–1001111. Cited by: §1.
- [34] A. Odrzywolek (2026). A ternary Sheffer operator for elementary functions?. Acta Physica Polonica B. Note: in preparation. Cited by: §5.
- [35] A. Odrzywołek (2026). SymbolicRegressionPackage: basic building blocks for symbolic regression. Note: https://github.com/VA00/SymbolicRegressionPackage Includes EML toolkit. Cited by: §2, §2, §3, §3, §4.1, §4.3, §5.
- [36] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala (2019). PyTorch: an imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pp. 8024–8035. Cited by: §4.1.
- [37] B. K. Petersen, M. Landajuela, T. N. Mundhenk, C. P. Santiago, S. Kim, and J. T. Kim (2021). Deep symbolic regression: recovering mathematical expressions from data via risk-seeking policy gradients. In International Conference on Learning Representations (ICLR), External Links: Link. Cited by: §4.3.
- [38] W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery (2007). Numerical recipes: the art of scientific computing. 3 edition, Cambridge University Press, Cambridge. External Links: ISBN 978-0-521-88068-8. Cited by: Summary paragraph, §1.
- [39] D. Richardson, B. Salvy, J. Shackell, and J. van der Hoeven (1996). Asymptotic expansions of exp-log functions. Technical report Technical Report RR-2859, INRIA. Note: inria-00073832. External Links: Link. Cited by: §1.
- [40] J. F. Ritt (1948). Integration in finite terms: Liouville's theory of elementary methods. Columbia University Press, New York. Cited by: Summary paragraph, §1.
- [41] M. Schönfinkel (1924). Über die bausteine der mathematischen logik. Mathematische Annalen 92 (3), pp. 305–316. External Links: ISSN 1432-1807, Document, Link. Cited by: §1.
- [42] J. Shackell (1990). Growth estimates for exp—log functions. Journal of Symbolic Computation 10 (6), pp. 611–632. External Links: ISSN 0747-7171, Document, Link. Cited by: §1.
- [43] H. M. Sheffer (1913). A set of five independent postulates for Boolean algebras, with application to logical constants. Transactions of the American Mathematical Society 14 (4), pp. 481–488. External Links: Document. Cited by: Summary paragraph, §1.
- [44] D. Smith, J. S. Myers, C. S. Kaplan, and C. Goodman-Strauss (2024). An aperiodic monotile. Combinatorial Theory 4 (1). External Links: Document, Link. Cited by: §1.
- [45] H. Takahasi and M. Mori (1974). Double exponential formulas for numerical integration. Publications of the Research Institute for Mathematical Sciences 9 (3), pp. 721–741. External Links: Document. Cited by: §1.
- [46] G. Terzo (2008). Some consequences of schanuel's conjecture in exponential rings. Communications in Algebra 36 (3), pp. 1171–1189. External Links: Document, Link, https://doi.org/10.1080/00927870701410694. Cited by: §2.
- [47] S. Udrescu and M. Tegmark (2020). AI Feynman: a physics-inspired method for symbolic regression. Science Advances 6 (16), pp. eaay2631. External Links: Document. Cited by: Summary paragraph, §4.3.
- [48] B. Ulmann (2022). Analog computing. De Gruyter Oldenbourg, Berlin, Boston. External Links: Link, Document, ISBN 9783110787740. Cited by: Summary paragraph, §4.2.
- [49] B. Ulmann (2024). Beyond zeros and ones – analog computing in the twenty-first century. International Journal of Parallel, Emergent and Distributed Systems 39 (2), pp. 139–151. External Links: Document, Link, https://doi.org/10.1080/17445760.2023.2296672. Cited by: §4.2.
- [50] S. Wolfram (2002). A new kind of science. Wolfram Media. External Links: ISBN 1-57955-008-8, Link. Cited by: §1.
