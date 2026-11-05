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

## Objetivos racionales exactos

`ExactRational` representa el objetivo matemático reducido, separado del árbol
`Bml`. `ExactRational::from_decimal_str(texto, límites)` acepta solamente
`[+-]?[0-9]+(\.[0-9]+)?`: construye enteros multipalabra y reduce por MCD, sin
pasar por floats. Así, `"12.375"` da `99/8`, `"0.0010"` da `1/1000` y
`"-0.000"` da cero no negativo. Los dígitos iniciales y finales cuentan para
el presupuesto, y el denominador no reducido también debe caber.

`ExactRational::from_f64(valor, límites)` extrae signo, significando y exponente
de los bits IEEE. `0.1` produce `3602879701896397/36028797018963968`, distinto
del objetivo decimal `1/10`. El mínimo subnormal produce `1/2^1074`, con el
signo correspondiente; ambos ceros se normalizan. NaN e infinitos se rechazan.

`ExactInputLimits` limita dígitos (1024), bits por componente (4096) y pasos de
resta del MCD binario (16384). La división exacta posterior recorre como máximo
el número de bits permitido; no hay una búsqueda sin límite. Los errores son
`ExactInputError::{Syntax, NonFinite, Limit}` para parseo; la aritmética
racional añade `DivisionByZero` y `NonIntegralExponent`. Los componentes se consultan como
palabras `u64` de menor peso primero, sin setters que puedan invalidarlos.
Estos parsers no sintetizan un `Bml` ni certifican un árbol existente;
una entrada exacta válida no implica éxito de la síntesis bajo presupuesto.

## Exportaciones y flujos

`try_to_u64` y `try_to_u128` exportan el código entero o devuelven
`StreamError::WordWidth`. Caben 31 y 63 operadores, respectivamente; el árbol
expandido tiene `2n+1` bits. `operator_count` cuenta todas las apariciones de
subárboles, no nodos únicos de un DAG. El tamaño lógico excluye la capacidad
del `Vec`, su cabecera y cualquier otra asignación.

`Bml::encode_stream` y `decode_stream` usan BMLS v1: cabecera de 24 bytes con
`BMLS`, versión `u8=1`, base `u8=2` (BML), dos bytes nulos, cantidad `u64` LE y
longitud útil `u64` LE. Sigue el payload de códigos concatenados, MSB primero
dentro de cada octeto; los bits bajos de relleno son cero. `[1,2,1]` ocupa cinco
bits útiles `01000`, payload `0x40`. Un flujo vacío conserva su cabecera con
cantidad y longitud cero. No admite EML ni referencias compartidas.

`StreamLimits` limita bits totales, cantidad de términos y profundidad (hoja
de profundidad cero). El lector valida estructura, campos, tamaño exacto y
relleno antes de reservar el vector de resultados. Esto no valida los dominios
reales de los árboles. Las pruebas públicas son `bml_stream` y
`bml_exact_input`; los tests de reducción usan un oráculo euclídeo `u128`
independiente del MCD binario multipalabra de producción.

## Síntesis y certificados exactos

`Bml::from_decimal_str_exact` y `Bml::from_f64_exact` devuelven `CertifiedBml`:
un árbol expandido (`term()`), su racional objetivo (`target()`) y una derivación
(`derivation()`). `into_term()` conserva solamente el árbol. Las variantes
`_with_options` aceptan `ExactSynthesisOptions`. La API aproximada `from_f64`
mantiene su contrato anterior.

`ExactDerivation` es un programa de reglas con referencias exclusivamente a
pasos anteriores. `certify` reconstruye el término y calcula su racional;
`verify` además comprueba un árbol y un objetivo declarados. Referencias futuras,
raíces ausentes, estructuras o valores distintos se rechazan. El valor de cada
paso se calcula con aritmética entera multipalabra exacta. Un signo no se deduce
de una aproximación. Las reglas positivas de producto y cociente rechazan
operandos negativos o cero; las extensiones comprobadas generales son trabajo
separado. `ExactError` distingue error racional, construcción, presupuesto,
prueba inválida y dominio no justificado. Ninguna regla usa `to_f64`.

El constructor actual usa un catálogo de cero/uno, potencias de dos y cercanía
a potencias, con memoización de enteros. Para `n > 1` que no sea potencia de
dos, toma `k=ceil(log2(n))`, `r=2^k-n` y construye `B(k,B(r,1))`. Como
`0 < r < n` y `k < n`, la descomposición disminuye; las cotas de pasos y
recursión la limitan además explícitamente. Un racional positivo `n/d` usa la
regla de cociente y un negativo usa negación. No se promete tamaño mínimo.
La comparación acotada de estrategias se ofrece por separado mediante
`ExactSearcher`, con candidatos, presupuestos y terminación registrados;
la sección de búsqueda describe su contrato.

Los presupuestos predeterminados son 4096 pasos, 128 niveles de descomposición,
1.048.576 bits por árbol y 16.777.216 bits de árboles retenidos durante la
reproducción, además de los límites racionales. Cada fallo conserva los
operandos. `2049` se construye como `B(12,B(2047,1))`: su certificado usa
`4096-2047`, sin materializar el número intermedio `2^2047`. El árbol sigue
siendo puro y su evaluador directo informa overflow. Certificar no cambia
silenciosamente la política numérica de `to_f64`.

### Justificación algebraica de las reglas reproducidas

Los árboles finitos forman el magma libre del constructor binario ordenado
sobre `1`: construir no identifica árboles distintos ni impone leyes
algebraicas. La interpretación real es parcial porque `log2(y)` exige `y>0`;
la interpretación EML usa otra base y no hereda estas reglas.

`E(x)=B(x,1)=2^x` es positivo para todo real finito matemático. Para `x>0`,
`B(1,x)=2-log2(x)`, `E(B(1,x))=4/x>0` y
`L(x)=B(1,E(B(1,x)))=2-log2(4/x)=log2(x)`. Por tanto
`D(a,b)=B(L(a),E(b))=a-b` requiere únicamente `a>0`; el segundo argumento
logarítmico es siempre positivo. No se exige `b>0`.

Para `p=2^x` y `q=p-x`, ambos son positivos. Si `x<=0`, es inmediato;
si `0<x<=1`, `2^x>1>=x`; si `x>=1`, la función `2^x-x` vale uno en uno
y tiene derivada `ln(2)*2^x-1>0`. Así `D(p,D(q,y))=p-(q-y)=x+y`,
`D(q,p)=-x` y `D(p,add(q,y))=x-y`, con todos los primeros argumentos de
`D` positivos. Para `x,y>0`, `E(add(L(x),L(y)))=xy` y
`E(sub(L(x),L(y)))=x/y`. Cero usa `B(1,B(B(1,1),1))=2-log2(4)=0`.
La regla de cercanía usa `B(x,E(y))=2^x-y`; el certificado solo admite
exponentes `x` enteros para calcular el racional `2^x` dentro del presupuesto.
Estas identidades no afirman que la evaluación flotante de cada intermedio
sea representable ni que la igualdad estructural decida igualdad matemática.

## Aritmética de valores certificados

`CertifiedBml` ofrece `checked_add`, `checked_sub`, `checked_mul`, `checked_div`
y `checked_neg` con `ExactSynthesisOptions`. Los operadores `+`, `-`, `*`, `/`
y negación unaria devuelven `Result<CertifiedBml, ExactError>` y aceptan
operandos propios o prestados. Combinan las derivaciones, reubican sus referencias
y reproducen la prueba completa bajo el presupuesto. No reemplazan los métodos
estructurales históricos de `Bml`. Los operadores comprobados de `Bml` recuperan primero una derivación
mediante `certify_exact`; conservan el árbol como resultado y no cambian los
métodos estructurales históricos.

La multiplicación/división con signo usa las pruebas racionales de los operandos:
si `x<0`, `neg(x)` prueba `|x|>0`, y análogamente para `y`. Se aplica la regla
positiva al par de magnitudes y se niega el resultado exactamente cuando los
signos difieren. Un producto con cero demostrado produce el cero canónico.
`0/y=0` se permite solamente después de comprobar `y!=0`. La división por cero
produce `ExactError::Rational(ExactInputError::DivisionByZero)` antes de cualquier
simplificación. Estas reglas justifican los dominios de las expansiones
positivas sin usar aproximaciones ni asumir que cualquier árbol sea real.

`try_add_assign` y `try_mul_assign` construyen y verifican en un temporal; solo
reemplazan el destino tras el éxito. No se implementan `AddAssign`/`MulAssign`:
el retorno `()` no puede propagar los errores de construcción o recursos.

Los operandos mixtos admiten todos los enteros nativos firmados/sin signo y
`f64`, mediante `ExactOperand`. Los enteros son valores matemáticos; `f64`
usa su racional IEEE exacto y propaga errores de síntesis. Se admiten
`&certificado + numero` y `numero + &certificado`, también para resta,
producto y división, preservando el orden. Las variantes con número a la
izquierda requieren el certificado prestado. Para aplicar límites propios,
convertir con `ExactOperand::into_certified(opciones)` y llamar al método
`checked_*`. No hay conversión implícita desde códigos enteros.

### Recuperación conservadora de pruebas para árboles

`Bml::certify_exact(opciones)` reconoce las expansiones completas de cero, uno,
exponenciación entera, cercanía a potencias, suma, resta, negación y las reglas
positivas de producto/cociente. Reproduce la derivación y exige igualdad exacta
con el árbol original antes de devolver un certificado. La deduplicación
estructural es solo un índice temporal del analizador: el almacenamiento público
sigue siendo el árbol expandido sin metadatos de prueba.

Un árbol sintácticamente válido puede ser real y aun así devolver
`DomainNotJustified`, por ejemplo una exponenciación con exponente racional no
entero. Este reconocedor no decide los reales arbitrarios ni usa floats como
oráculo. Limita bits de entrada, nodos distintos y pila mediante `max_bits` y
`max_steps`, recursión mediante `max_recursion` y comparaciones de patrones a
`4096 * max_steps`. La reproducción aplica además los presupuestos racionales,
de pasos y de materialización. Rechazar por presupuesto no demuestra invalidez
matemática.

`(&a + &b)?`, los otros operadores y `Bml::checked_*` aplican este análisis
antes de construir el resultado. Los operadores aceptan también valores propios
y números mixtos. `try_add_assign`/`try_mul_assign` son transaccionales también
para `Bml`. En cambio, `a.add(&b)`, `a.mul(&b)` y los demás métodos históricos
siguen construyendo términos bajo sus precondiciones documentadas: no se les
ha añadido una comprobación implícita ni un cambio en su tipo de error.

## Matrices simbólicas

`BmlMatrix` está disponible en `bml_domain::number` y en la fachada `bml`, sin
activar `tensor`. `new(filas, columnas, Vec<Bml>, límites)` conserva los códigos
y valida el producto de dimensiones, longitud, bits por elemento y bits totales.
No valida los dominios reales hasta una operación comprobada. Los elementos
son row-major: `as_slice()` y `get(fila,columna)` ofrecen acceso prestado;
`Index<(usize,usize)>` provoca panic para un índice inválido. No hay `IndexMut`.
`replace` valida la sustitución completa antes de modificar la celda, preservando
forma, longitud y presupuestos. Los tensores IEEE/GGUF no cambian.

`&a + &b` es suma por elemento; `&a * &b` es producto matricial. Ambos devuelven
`Result` y admiten también matrices propias. `hadamard` es producto por elemento.
Los operadores usan los límites de la matriz izquierda para la operación;
cada matriz conserva además sus invariantes propios. Los métodos
`add_with_limits`, `hadamard_with_limits` y `matmul_with_limits` permiten límites
explícitos. `add_scalar` y `scale` aceptan un `ExactOperand`, convertido una sola
vez antes del recorrido, incluidos escalares ya certificados.

La preparación de cada operación recupera las pruebas de los elementos una vez
por matriz de entrada. El kernel trabaja con estos valores certificados; no
resintetiza números por celda. Los productos no vacíos inician con el primer
producto y suman de izquierda a derecha, sin un cero inicial artificial. Una
dimensión interna vacía produce el cero canónico `1011000` en cada salida. Un
resultado sin celdas retorna directamente, incluso si una dimensión vacía está
acompañada por otra dimensión muy grande.

`generalized_product` recibe combinación, acumulación y `ReductionInit`.
`First { empty: None }` exige una reducción no vacía;
`First { empty: Some(valor) }` solo usa el valor para el caso vacío;
`Value(valor)` inicia todas las reducciones con ese valor. Siempre reduce
`acumular(acumular(p0,p1),p2)`; no presupone asociatividad ni identidad de B.
Los callbacks deciden la política matemática de su operación y deben construir
sus resultados con presupuestos propios; el kernel vuelve a comprobar el tamaño
de todos los resultados e intermedios recibidos. Un callback puede ser puramente
estructural si esa es la operación generalizada elegida.

`MatrixLimits` limita elementos por matriz, bits por matriz, bits acumulados de
los resultados de combinación/acumulación y los presupuestos escalares. Los
intermedios de reducción cuentan aunque se descarten después. El presupuesto
escalar limita las pruebas y construcciones internas de cada operación. La
comprobación de tamaño de salida se hace por celda, antes de retenerla. Los
errores escalares y presupuestos durante el kernel llevan fila, columna e índice
interno cuando procede. Fallos de forma/preparación se distinguen; ningún error
entrega una salida parcial ni modifica entradas.

### Localidad y asignaciones de la referencia

`Vec<Bml>` hace contiguas las cabeceras, no los payloads de palabras de cada
árbol. La referencia recorre salidas por filas y columnas, y la dimensión interna
de izquierda a derecha; el acceso a la matriz derecha tiene stride de columnas.
No hay aún bloqueo de caché ni transposición temporal, porque cambiar el orden de
acumulación alteraría estructuras y presupuestos. Se reservan vectores de
preparación y salida antes del kernel; las operaciones escalares siguen asignando
árboles y pruebas. Cada celda de salida posee su árbol expandido. Las pruebas
pueden reutilizar referencias internamente pero no comprimen ese árbol.

La preparación se repite en operaciones independientes: no hay caché global ni
buffer persistente oculto. El escalar se comparte por referencia durante el
recorrido, aunque cada expansión duplica las apariciones requeridas. `total_bits`
reporta tamaño lógico y excluye capacidad, cabeceras y pruebas temporales; no es
una medición de RAM. Las campañas de asignaciones/memoria/rendimiento quedan en
la sección de evaluación de optimizaciones de la spec.

## Comparación acotada de estrategias

`ExactSearcher::new(ExactSearchOptions)` compara construcciones certificadas sin
alterar el constructor exacto básico. `search(&objetivo)` devuelve el mejor
candidato explorado y un informe; `best=None` significa que no hubo un éxito
certificado dentro del presupuesto. Los errores de candidatos quedan en el
informe, sin convertirse en resultados aproximados.

Las familias son: catálogo fijo de árboles pequeños; potencias de dos y
recursión de cercanía; suma con la potencia inferior; resta desde la potencia
superior; pares de factores; reutilización del mismo objetivo ya certificado.
Las estrategias de suma/resta/factores se aplican a magnitudes enteras que caben
en `u64`, con signo explícito; la ruta básica y el caché admiten los racionales
multipalabra. Esto no promete explorar todas las descomposiciones posibles.
La factorización prueba divisores en orden ascendente con un máximo explícito.

Cada intento registra familia, parámetro (código de catálogo, potencia o divisor),
costo certificado o error. `TreeCost` cuenta operadores, bits, profundidad,
palabras usadas y capacidad de palabras del árbol expandido. `B(4,2)` cuesta
cuatro operadores y nueve bits, y duplicar un subárbol cuenta ambas apariciones.
La selección minimiza operadores, después profundidad y después código binario
lexicográfico. Un empate completo conserva el primer candidato explorado.
Tiempo y evaluabilidad no influyen en esta elección.

Los valores predeterminados permiten 32 candidatos y 256 pruebas de divisor.
Cada candidato sigue sujeto a `ExactSynthesisOptions`; el número de intentos
limita también el trabajo repetido de síntesis/verificación. El caché pertenece
al objeto y retiene como máximo 16 resultados, 4.194.304 bits de árboles y
16.384 pasos de prueba. Los componentes racionales de cada entrada conservan
sus límites propios. La expulsión es FIFO; cero entradas desactiva el caché.
`cached_results` y `clear_cache` hacen explícito su ciclo de vida. No se deduce
el costo de memoria total solamente a partir de los bits del árbol.

El informe registra límites usados, intentos, consultas al caché, pruebas de
divisor, motivo de parada e índice seleccionado. `minimum_proven` es siempre
falso: completar las estrategias configuradas no prueba optimalidad global.
`elapsed` mide toda la búsqueda, incluida la retención en caché. La observación
numérica inicia como no medida. `measure_direct_evaluation()` registra por
separado tiempo y resultado/error de `to_f64`, sin cambiar el certificado ni
la selección. Un resultado finito no demuestra precisión o estabilidad general.

## Evaluación simplificada explícita

`CertifiedBml::to_f64_simplified(opciones)` reproduce y verifica la derivación,
obtiene su racional exacto y lo redondea una vez a binary64. La variante de
`Bml` recupera primero una prueba mediante el reconocedor conservador. Ambas
usan únicamente las identidades documentadas y devuelven `SimplifiedEvalError`
si la prueba, el presupuesto o la conversión final fallan. No intentan decidir
expresiones reales arbitrarias. `Bml::to_f64()` conserva su evaluación directa.

`ExactRational::to_f64_rounded(límites)` aplica round-to-nearest, ties-to-even.
Calcula el exponente mediante longitudes y comparaciones enteras, escala el
numerador o denominador, y hace división entera con resto. El resto se compara
con `denominador-resto` para decidir si supera la mitad; en un empate solo se
incrementa un significando impar. No divide floats ni usa tolerancias. El
resultado incluye subnormales; un racional no nulo que redondearía a cero
produce `Underflow`, y un resultado infinito produce `Overflow`. Cero matemático
produce `+0.0`. Los límites de bits se comprueban también sobre los enteros
desplazados; el trabajo de división está acotado por esas longitudes. Los
límites de dígitos/MCD pertenecen al parseo y a las operaciones racionales,
no a esta división final.

Por ejemplo, la prueba de `2049=B(12,B(2047,1))` reduce exactamente a `4096-2047`
y la ruta simplificada devuelve `2049.0`; la directa sigue informando overflow.
El objetivo decimal `1/10` se redondea a un `f64`, pero permanece exactamente
`1/10` en el certificado. El redondeo no cambia su valor matemático demostrado.
Las pruebas incluyen oráculos de división hardware con operandos enteros
representables exactamente, parseo decimal estándar, patrones IEEE generados,
empates y transiciones normal/subnormal/overflow/underflow.

## Kernel SIMD de comparación de códigos

`bml-kernels` implementa comparación estructural de códigos BML multipalabra.
La fachada lo expone como `bml::code_simd` con la feature `code-simd`. El crate
`bml-domain` conserva cero dependencias y `deny(unsafe_code)`. Este kernel no
construye matrices, no evalúa B, no calcula `exp2`/`log2` y no convierte códigos
en valores aritméticos. Los árboles se conservan sin cambios; no hay reducción
numérica que reasociar ni tolerancia numérica que configurar.

`compare_codes` recibe dos lotes de `Bml`, un backend y `CompareLimits`.
Comprueba longitud, máximo de pares, primer exceso de bits acumulados y
capacidad del backend, en ese orden, antes de reservar la salida. Cada resultado
es el orden del entero codificado sin signo. Para igual longitud se comparan
palabras desde la más significativa. Un árbol fuera del dominio real sigue
siendo comparable estructuralmente; esto no lo certifica como real. La sintaxis
ya fue validada al construir `Bml`. No se devuelven resultados parciales.

La referencia escalar compara palabras en orden descendente. AVX2 prueba
igualdad de cuatro palabras con `loadu`, `cmpeq_epi64` y `movemask_epi8`; NEON
AArch64 usa dos palabras con `vld1q_u64`, `vceqq_u64` y extracción de máscaras.
Si un bloque difiere, se localiza la primera palabra desigual en orden escalar
descendente; las colas usan la referencia. Las cargas no requieren alineación
y nunca atraviesan el final de un slice. Cada función específica está anotada
con `target_feature` y se llama solo después de detectar la capacidad en runtime.
La interfaz pública es segura; los invariantes de cada bloque `unsafe` están
comentados en `crates/kernels/src/lib.rs`. ARM de 32 bits y SVE/SVE2 no se anuncian
como soportados; forzar un backend no disponible devuelve un error explícito.
Referencias oficiales: [AVX2](https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_cmpeq_epi64.html),
[NEON](https://doc.rust-lang.org/core/arch/aarch64/fn.vceqq_u64.html) y
[detección x86](https://doc.rust-lang.org/std/macro.is_x86_feature_detected.html).

`Auto` permanece en escalar. La campaña inicial en x86-64 encontró ventaja de
AVX2 para códigos largos y regresión para códigos cortos; no se promueve una
regla automática universal. En esa campaña inicial NEON solo compiló para
`aarch64-unknown-linux-gnu`; las muestras, CPU, flags, revisión y hashes
históricos se conservan en `quality/evidence/bml-code-simd.json`.
La campaña posterior de GitHub Actions `41853147652` ejecutó Scalar/NEON en
una VM ARM64 nativa y Scalar/AVX2 en una VM x86-64 nativa: ocho contratos por
host y tres cargas con siete repeticiones por backend, tras comprobar la
corrección. `quality/evidence/native-kernels-41853147652.json` conserva fuentes,
ELF, CPU, muestras y hashes verificados. Acredita comparación estructural de
códigos; no mide evaluación BML numérica, inferencia ni comportamiento PMU/L1.

Reproducción: `cargo test -p bml-kernels --locked`,
`cargo run -p bml-kernels --example code_compare_bench --release --locked -- 100`
y, con el target instalado, `cargo check -p bml-kernels --all-targets --target
aarch64-unknown-linux-gnu --locked`. El ejemplo separa preparación de lotes y
comparación, usa entradas preparadas dinámicamente, siete repeticiones y warmup
excluido. Un allocator instrumentado cuenta llamadas, bytes solicitados y pico
adicional de heap vivo; no mide RSS ni elimina el costo de la instrumentación.
La fase medida incluye validación y asignación del vector de resultados: no se
presenta como latencia de intrínsecos aislados. La unidad es comparaciones de
códigos por segundo; no son aplicaciones de B/s, FLOP/s ni tokens/s.

### Unidades de las campañas de rendimiento

Cada campaña debe declarar entrada y trabajo contados antes de medir. Las
unidades no son intercambiables:

- **Construcciones BML/s:** llamadas de construcción completadas por segundo,
  con número de operadores/bits creados y fallos registrados por separado. Una
  macro aritmética es una llamada, no una sola aplicación de B.
- **Aplicaciones de B/s:** nodos B efectivamente evaluados por segundo en un
  evaluador directo. Las hojas no cuentan; las apariciones duplicadas sí. Una
  simplificación que evita nodos no puede atribuirse su ejecución.
- **Expresiones/s:** árboles completos procesados por segundo, indicando si se
  construyeron, certificaron o evaluaron, y su distribución de tamaños.
- **Elementos/s:** elementos de entrada/salida de una operación por segundo,
  declarando cuál se cuenta. Para comparación se usa una pareja de códigos como
  unidad de trabajo, no un elemento numérico.
- **Productos matriciales/s:** llamadas completas a matmul por segundo, junto
  con dimensiones, orden, política de dimensión vacía y bits antes/después. No
  se confunde con productos escalares dentro de una reducción.
- **FLOP/s:** solo para una campaña de aritmética flotante con convención
  explícita. Por ejemplo, matmul convencional puede reportar `2*m*n*k` como
  conteo nominal (FMA=dos FLOP), distinguiéndolo del conteo exacto de la reducción
  sin suma inicial: `m*n*(2*k-1)` para `k>0`. Una aplicación de B con exp/log no
  se declara automáticamente tres FLOP; sintetizar/certificar códigos enteros
  tampoco genera una equivalencia FLOP/s.

Las mediciones deben separar preparación/síntesis, búsqueda/optimización,
kernel y operación completa, sin sumar tasas de fases distintas. Las tasas de
éxito y error, dispersión, memoria, asignaciones y crecimiento de árboles forman
parte de la evidencia. Una mejora en alguna de estas unidades no implica mejora
en tokens/s de un modelo.

La campaña `quality/evidence/bml-symbolic-phases.json` se reproduce con
`cargo run -p bml-kernels --example symbolic_phases --release --locked -- 10`.
Registra siete muestras por fase, entradas, semilla, CPU, compilador, hashes del
código, asignaciones y heap solicitado. Separa parseo exacto, síntesis, búsqueda
con caché fría/caliente, replay de pruebas, serialización, evaluación directa y
simplificada, aplicación escalar de B y preparación/producto matricial completo.
El producto sobre códigos preparados todavía incluye reconocimiento de pruebas;
no mide un kernel aritmético aislado. La síntesis incluye parseo decimal.

Son cargas sintéticas pequeñas en una única VM x86. De las nueve expresiones,
dos fallan en evaluación directa; su tasa cuenta intentos, no éxitos. Todas pasan
la ruta simplificada frente al parseo `f64` independiente. La matriz `2×3` por
`3×2` pasa un oráculo entero y crece de 24 bits de entrada a 3332 de salida.
La búsqueda con caché caliente resultó más lenta en esta muestra; no se promueve
como optimización universal. Las muestras crudas y su dispersión permiten
revisar estas observaciones sin convertirlas en garantías de rendimiento.
