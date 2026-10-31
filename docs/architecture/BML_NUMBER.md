# El escalar BML

`bml_domain::number::Bml` representa un término cerrado puro. La fachada `bml`
ofrece este escalar por defecto. Los tensores IEEE y GGUF
conservan sus contratos propios.

## Representación

La gramática es `T = One | Op(T, T)`, con recorrido prefijo: `0` representa
`One` (valor 1), y `1` representa `Op` seguido de sus operandos izquierdo y
derecho. La interpretación es `B(x,y) = 2^x - log2(y)`:

| Bits | Término | Valor |
|---|---|---:|
| `0` | `One` | 1 |
| `100` | `Op(One, One)` | 2 |
| `11000` | `Op(Op(One, One), One)` | 4 |

Los bits forman un entero sin signo, con el primer símbolo como bit más
significativo. El almacenamiento usa palabras `u64` de menor peso primero y
no guarda una longitud adicional. El entero cero representa el término `0`; su
forma canónica en palabras es vacía. La igualdad es estructural: dos expresiones
con el mismo valor no son necesariamente iguales.

`Bml::from_binary`, `from_u64` y `from_words_le` validan un árbol completo;
las palabras altas nulas se normalizan. `to_binary`, `to_u64`, `to_words_le` y
`bit_len` exponen su representación. `from_u64` interpreta una codificación,
no un número matemático: `from_u64(4)` produce `100`, cuyo valor es 2.
`one`, `op` y `op_with_limit` construyen términos sin evaluarlos.

## Reglas de operaciones

Los métodos `Bml` construyen un árbol y devuelven `Result<Bml, BuildError>`.
`Rules { max_bits }` permite configurar el límite de cada resultado intermedio;
el valor predeterminado es 1.048.576 bits. La construcción no comprueba el
valor de los operandos. Los dominios siguientes son condiciones matemáticas de
las identidades; `to_f64` puede fallar o perder precisión por intermedios.

Sean `E(x) = B(x,1)`, `L(x) = B(1,E(B(1,x)))`,
`D(a,b) = B(L(a),E(b))` para `a > 0`, `p = E(x)` y `q = B(x,p) = p - x > 0`.
Las reglas implementadas son:

| Método | Expansión matemática | Dominio |
|---|---|---|
| `exp2(x)` | `E(x)` | real |
| `log2(x)` | `L(x)` | `x > 0` |
| `add(x,y)` | `D(p, D(q,y))` | reales |
| `neg(x)` | `D(q,p)` | real |
| `sub(x,y)` | `D(p, add(q,y))` | reales |
| `mul(x,y)` | `E(add(L(x),L(y)))` | `x,y > 0` |
| `div(x,y)` | `E(sub(L(x),L(y)))` | `x,y > 0` |
| `recip(x)` | `E(neg(L(x)))` | `x > 0` |
| `pow(x,y)` | `E(scale(L(x),y))` | `x,y > 0` |
| `sqrt(x)` | `E(scale(L(x), E(neg(1))))` | `x > 0` |

Aquí `scale(z,y) = D(mul(E(z),y), mul(B(z,E(z)),y))` para `y > 0`.
Las expresiones se expanden hasta usar únicamente `One` y `Op`. La referencia
local [artículo EML](../../references/eml_paper.md) describe el operador en base
natural y sus construcciones; estas reglas son una adaptación a base 2, no una
transcripción directa de sus tablas. Seno, coseno, exponencial natural y
evaluación compleja quedan para una fase posterior, con sus ramas y dominios.

## Conversión y errores

`to_f64()` evalúa en reales finitos con errores de dominio y rango, incluido
overflow y underflow exponencial a cero. El cierre estructural no garantiza
evaluación válida. `from_f64` y `from_f64_with_options` sintetizan bits puros:
buscan primero un racional de denominador pequeño dentro de la tolerancia y,
si no encuentran uno, construyen el racional diádico exacto del `f64`. La
evaluación final debe cumplir `abs(actual - solicitado) <= absolute_tolerance +
relative_tolerance * abs(solicitado)`. El cero negativo se normaliza a cero.

`SynthesisOptions` configura `max_bits`, `max_denominator` (1024 por defecto),
`absolute_tolerance` (0) y `relative_tolerance` (1e-12). Se devuelven errores
tipados `NonFiniteInput`, `InvalidOptions`, `Build`, `Evaluation` o `Precision`.
La búsqueda es acotada: incluso una entrada finita puede fallar por tamaño,
dominio, rango o precisión. No se usa un pool de constantes ni se promete una
conversión universal o estabilidad numérica general.
