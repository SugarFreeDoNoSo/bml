## Context

`Bml` ya almacena el entero de un árbol prefijo en palabras `u64`, sin longitud
individual persistida. `from_u64` interpreta un código, no el valor numérico del
entero. `from_f64` busca racionales con tolerancia y valida mediante `to_f64`;
esto es un contrato diferente de la conversión simbólica exacta solicitada.
Los kernels tensoriales actuales trabajan con IEEE y los modelos preservan sus
pesos como buffers o bloques cuantizados.

`Bml` contiene un `Vec<u64>` y su igualdad es estructural. Los métodos aritméticos
actuales construyen términos y devuelven `BuildError`; `mul`/`div` tienen dominio
positivo. La nueva ruta comprobada deberá justificar dominios y signos sin
confundir esa construcción histórica con una validación semántica universal.

## Goals / Non-Goals

- G1: especificar valor objetivo exacto, dominio, prueba y presupuesto de síntesis.
- G2: establecer tamaño real del árbol y del almacenamiento, incluidos flujos.
- G3: conservar `Bml` en matrices y definir operaciones y errores por elemento.
- G4: ofrecer operadores Rust fallibles y una referencia escalar verificable.
- G5: contrastar AVX2 y NEON sobre operaciones concretas en su hardware objetivo.
- G6: definir unidades OPS, costo real y evidencia reproducible de rendimiento.
- NG1: prometer síntesis universal, decisión de equivalencia arbitraria, mínimo
  global, evaluación `f64` siempre posible, compresión o aceleración garantizada.
- NG2: sustituir los tensores IEEE/GGUF existentes, añadir complejos o redefinir
  la evidencia de inferencia/AOT ya obtenida.

## Decisions

### D1. Semántica y pruebas de reglas

El conjunto de términos finitos `T ::= 1 | B(T,T)` forma el magma libre sobre
el generador `1` mediante el constructor ordenado. No se presupone identidad,
asociatividad ni conmutatividad. Los límites de la implementación pueden impedir
materializar un árbol del conjunto abstracto.

La interpretación `B(x,y)=2^x-log2(y)` está definida para `x` real e `y>0`;
es una operación parcial sobre los reales, no un magma real total. Separar
validez sintáctica, dominio matemático y rango del evaluador concreto. Documentar
derivaciones base 2 de `add`, `sub`, `mul`, `div`, `neg` y reglas auxiliares,
incluidas condiciones de todos los intermedios. EML usa `exp(x)-ln(y)` y requiere
su propia interpretación y pruebas. Un contenedor compartido identifica la base.

### D2. Objetivos exactos y compatibilidad

La gramática decimal inicial será `[+-]?[0-9]+(\.[0-9]+)?`, sin espacios, coma ni
notación exponencial. El punto es el separador de la API; `12,375` en notación
matemática se introduce como `"12.375"`. Contar todos los dígitos fraccionarios
antes de reducir por MCD, con límites de entrada previos a calcular `10^d`.
Rechazar otras formas explícitamente permite ampliar la gramática más adelante.

Para `f64`, extraer `s`, fracción `f` y exponente almacenado `e` de `to_bits()`:
normales `m=2^52+f`, `q=e-1023-52`; subnormales `m=f`, `q=-1074`.
El objetivo es `(-1)^s m 2^q`, reducido como racional. Rechazar NaN/infinidades;
normalizar `+0.0`, `-0.0` y ceros decimales al mismo cero matemático. La exactitud
se refiere al valor, no a preservar el signo IEEE del cero ni el payload de NaN.
Conservar el comportamiento de `from_f64` existente y documentar las tres APIs.

### D3. Exactitud sin dependencia del evaluador flotante

Cada éxito exacto tendrá una derivación verificable desde el racional objetivo
y reglas probadas; no basta una comparación aproximada ni un roundtrip `f64`.
No se exige decidir igualdad de árboles arbitrarios. Un presupuesto agotado o
una exactitud no justificada produce error explícito, nunca éxito aproximado.
`to_f64` puede fallar aun cuando la construcción sea exacta; un evaluador
simplificado o de mayor precisión será una opción explícita con límites propios.

### D4. Estrategias y objetivo de síntesis

Comparar candidatos de catálogo, factorización, potencias de dos y expresiones
cercanas, suma/resta y resultados previamente compilados. Presupuestar también
factorización y consultas/materialización de resultados reutilizados.
El criterio principal será el número de operadores `B` tras expansión completa;
menor profundidad y orden lexicográfico del código desempatan de forma estable.
Tiempo de evaluación y estabilidad se reportan aparte: no se deducen del tamaño.
Registrar candidatos explorados, límites, estrategia y motivo de terminación;
un resultado heurístico se llama mejor encontrado bajo ese presupuesto.

### D5. Almacenamiento y flujo

Mantener palabras `u64` de menor peso primero y errores de recursos; no introducir
`BigUint` ni una dependencia nueva en `bml-domain` por esta especificación.
Las exportaciones a una palabra son comprobadas: 31 operadores para `u64`, 63
para `u128`. El tamaño matemático `2n+1` excluye metadatos y capacidad asignada.
El formato de flujo será distinto del entero individual: versión, cantidad,
longitud útil en bits y payload prefijo empaquetado, con orden de bits y padding
canónico fijados antes de implementarlo. Un DAG compartido requerirá otro formato.

### D6. Matrices BML y operaciones de Rust

Introducir matrices explícitas de `Bml` sin conversión implícita a floats. `A+B`
es elemento a elemento, `A*B` es producto matricial y `hadamard` es explícito.
Los operadores aritméticos devuelven `Result`, también para referencias; las
asignaciones usan métodos `try_add_assign`/`try_mul_assign` transaccionales.
Se difieren `AddAssign`/`MulAssign`: su retorno `()` no propaga errores. La
conversión mixta es numérica, nunca `from_u64` como código. Definir la política de
acceso y presupuestos para que `IndexMut` no permita eludir invariantes.

Fijar reducciones de izquierda a derecha, primer producto como acumulador de
reducciones no vacías y cero simbólico para reducción vacía del producto estándar.
El producto generalizado define combinación, acumulación, inicio y caso vacío
por separado; no se presume identidad de `B`. Preservar operandos y no devolver
resultados parciales en caso de fallo. Cuando el dominio no pueda justificarse,
retornar error de dominio no justificado, sin alegar que `f64` es un decisor exacto.

### D7. AVX2 y NEON como experimentos verificables

Elegir explícitamente manipulación de códigos, evaluación numérica o procesamiento
de elementos independientes; una implementación no acredita las otras dos.
Las rutas vectoriales preservan orden, resultados, errores y límites de la
referencia. Una transformación que cambie asociación/árbol necesita prueba y
política explícitas antes de compararse. Mantener fallback escalar y verificar
capacidades de CPU/objetivo antes de ejecutar instrucciones específicas. Medir
en ARM y x86 reales, para cada backend acreditado, tiempo, memoria,
asignaciones y crecimiento de códigos con igual trabajo y entradas dinámicas.
Sin hardware o campaña, registrar pendiente/no ejecutado; un fallback escalar
permite corrección, pero no demuestra ventaja SIMD. SVE/SVE2 queda fuera de la
implementación inicial; su existencia no implica soporte ni ancho fijo.

### D8. Métricas y promoción

Separar síntesis, preparación, kernel y operación completa. Definir unidades de
trabajo antes de medir; una primitiva `B` no equivale automáticamente a tres FLOP.
Cada variante se compara con una línea base escalar de igual semántica y entradas,
tras pasar la puerta de corrección. Registrar CPU, compilador/flags, revisión,
distribución de árboles, precisión, errores, memoria, asignaciones y dispersión.
La variante más rápida solo se promueve si supera la validación; ausencia de
ventaja o de hardware se documenta sin extrapolaciones a inferencia.

## Risks / Trade-offs

- R1: árboles y racionales intermedios grandes. Limitar dígitos, bits, profundidad,
  nodos, candidatos, factorización y memoria antes de asignar o expandir.
- R2: identidades exactas con intermedios desbordados. Separar certificado
  simbólico de evaluación; incluir `2049=B(12,B(2047,1))` como regresión.
- R3: optimización por bits perjudica profundidad/estabilidad. Reportar todas
  las métricas y mantener los objetivos explícitos.
- R4: cachés o DAG ocultan costos. Contar el árbol realmente serializado y
  distinguir tiempo frío/caliente, memoria compartida y materialización.
- R5: las reglas de multiplicación/división actuales tienen restricciones de
  positividad. Probar extensiones con signo y cero antes de ofrecerlas como
  operaciones generales; rechazar división por cero y dominios no justificados.

## Trazabilidad de las 13 observaciones

| Observación | Spec | Tareas |
|---|---|---|
| 1. Magma y dominio | `bml-exact-synthesis` | 1.1–1.3 |
| 2. OPS y operadores Rust | `bml-rust-operators` | 1.4–1.6 |
| 3. AVX2, NEON y CPU | `bml-simd-backends` | 7.1–7.4 |
| 4. Trabajo vectorizable | `bml-simd-backends` | 7.1–7.4 |
| 5. Matrices BML precompiladas | `bml-symbolic-matrices` | 6.1–6.5 |
| 6. Operaciones matriciales | `bml-symbolic-matrices` | 6.2–6.5 |
| 7. Codificación entera | `bml-code-storage` | 4.1, 4.3, 4.5 |
| 8. Capacidad | `bml-code-storage` | 4.2–4.4 |
| 9. Conversiones numéricas | `bml-exact-synthesis` | 2.1–2.4 |
| 10. Estrategias de síntesis | `bml-exact-synthesis` | 3.1–3.3 |
| 11. Tamaño, exactitud y costo | `bml-exact-synthesis`, `bml-performance` | 3.2–3.3, 5.1–5.3 |
| 12. Métricas OPS | `bml-performance` | 7.5–7.6 |
| 13. Validación y estado | las seis specs | 8.1–9.3 |
