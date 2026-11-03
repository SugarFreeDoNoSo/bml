# Evaluación compleja y síntesis simbólica acotada

Las extensiones de `bml_domain::symbolic` operan sobre `ClosedTerm = One | Op`.
El contrato real de `evaluate_real` sigue siendo explícito y parcial. Los pesos,
el escalar exacto `Bml` y el runtime causal tienen contratos propios.

## Complejos finitos

`symbolic::complex` expone `Complex64`, `LogBranch` y `EvaluationBudget`.
`Basis::apply_complex(x, y, branch)` interpreta `exp(x) - ln(y)` en EML o
`2^x - log2(y)` en BML. El logaritmo usa
`atan2(y.im, y.re) + 2*pi*branch.turns`; para BML se divide también la parte
imaginaria por `ln(2)`. En el corte negativo, `+0.0i` selecciona `+pi` y
`-0.0i` selecciona `-pi`. La rama es fija en todo el árbol, no continuación
analítica a lo largo de una trayectoria.

`ClosedTerm::evaluate_complex(basis, branch, budget)` visita el árbol de manera
iterativa, hijo izquierdo primero. `max_nodes` cuenta ocurrencias y `max_depth`
incluye la raíz. Un presupuesto cero impide incluso evaluar `One`. Los límites
no gobiernan la construcción, clonación ni destrucción del AST prestado.

Se rechazan operandos no finitos, `log(0)`, overflow y magnitud exponencial
redondeada a cero. Se permite redondeo ordinario, subnormales no nulos, underflow
de una componente y cancelación. El logaritmo del módulo evita desbordar `hypot`
o elevar directamente al cuadrado componentes extremas. No hay certificación
del error global ni semántica de infinitos complejos.

## Síntesis por enumeración

`symbolic::synthesis::synthesize(target, interpretation, tolerance, budget)` busca
árboles en orden creciente de nodos y después por su código prefijo, `One` antes
de `Op`. `Interpretation` fija la base y, en modo complejo, la rama. El modo real
exige un target sin parte imaginaria.

Los dos presupuestos independientes son `max_nodes` por árbol y `max_candidates`
intentados, incluidas evaluaciones inválidas. El máximo admitido es **255 nodos**,
para acotar profundidad y memoria de esta API de fórmulas pequeñas. La memoria
de búsqueda no acumula todos los árboles: conserva el mejor y el recorrido DFS
actual. Un target o una tolerancia no finitos, tolerancia negativa o límite de
nodos superior se rechazan antes de enumerar.

`SearchResult` devuelve el número intentado, el mejor `Candidate` finito si lo
hay y `StopReason::{ToleranceMet, CandidateBudget, NodeBudget}`. El candidato
incluye árbol, valor, error absoluto euclídeo y tamaño. Los empates retienen el
primer árbol, por tanto el menor tamaño. Agotar un límite puede devolver un
candidato fuera de tolerancia; el llamador debe inspeccionar el estado.

La coincidencia numérica, incluso con error f64 cero, no es una demostración
de igualdad simbólica ni síntesis exacta. Agotar el presupuesto tampoco prueba
que el target sea irrepresentable. El módulo contiene un ejemplo compilado por
rustdoc; las pruebas públicas son `symbolic_complex_contracts` y
`symbolic_synthesis_contracts`.
