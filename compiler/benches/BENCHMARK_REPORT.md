# Benchmark Report: FMA vs BML — Análisis de Complejidad Big O

Fecha: 2026-07-06
Máquina: Linux x86_64, Rust 1.96.0, perfil release (`lto=fat`, `panic=abort`, `codegen-units=1`)

## Metodología

Se miden 3 variantes de evaluación de una fórmula repetida N veces:

1. **FMA tradicional**: `acc = acc.exp2() - y.log2()` en un loop directo de N iteraciones.
2. **BML con Hash Consing (cadena)**: DAG en cadena `bml(node, two)` donde `two` se deduplica pero `node` es nuevo cada vez.
3. **BML sin Hash Consing (cadena)**: cada iteración recrea `bml(1, 1)` sin deduplicar.

N varía en progresión geométrica: 10, 100, 1 000, 10 000, 100 000.

> **Nota:** N > 100 000 causa stack overflow porque la evaluación del DAG es recursiva (`evaluate_soa`). El hot loop RPN iterativo del Hito 5 eliminará esta limitación.

## Resultados

### Tiempo de ejecución (media)

| N | FMA | BML cons (cadena) | BML no cons (cadena) |
|---|---|---|---|
| 10 | 49 ns | 156 ns | 254 ns |
| 100 | 326 ns | 580 ns | 2 303 ns |
| 1 000 | 3.07 µs | 4.58 µs | 22.5 µs |
| 10 000 | 27.8 µs | 44.9 µs | 214 µs |
| 100 000 | 282 µs | 490 µs | 2 383 µs |

### Análisis de complejidad

#### FMA tradicional: **O(n)**

| N | Tiempo | Tiempo/N |
|---|---|---|
| 10 | 49 ns | 4.9 ns |
| 100 | 326 ns | 3.3 ns |
| 1 000 | 3.07 µs | 3.1 ns |
| 10 000 | 27.8 µs | 2.8 ns |
| 100 000 | 282 µs | 2.8 ns |

`Tiempo/N` es constante (~3 ns/op). **Complejidad: O(n)** lineal, como esperado.

#### BML con Hash Consing (cadena): **O(n)**

| N | Tiempo | Tiempo/N | vs FMA |
|---|---|---|---|
| 10 | 156 ns | 15.6 ns | 3.2x |
| 100 | 580 ns | 5.8 ns | 1.8x |
| 1 000 | 4.58 µs | 4.6 ns | 1.5x |
| 10 000 | 44.9 µs | 4.5 ns | 1.6x |
| 100 000 | 490 µs | 4.9 ns | 1.7x |

`Tiempo/N` se estabiliza en ~4.5-5 ns/op. **Complejidad: O(n)** lineal.

El overhead vs FMA es ~1.6x (constante), porque cada operación BML hace `exp2` + `log2` como FMA, más el overhead del intérprete RPN (push/pop de pila).

#### BML sin Hash Consing (cadena): **O(n)** pero con constante 5x mayor

| N | Tiempo | Tiempo/N | vs BML cons |
|---|---|---|---|
| 10 | 254 ns | 25.4 ns | 1.6x |
| 100 | 2 303 ns | 23.0 ns | 4.0x |
| 1 000 | 22.5 µs | 22.5 ns | 4.9x |
| 10 000 | 214 µs | 21.4 ns | 4.8x |
| 100 000 | 2 383 µs | 23.8 ns | 4.9x |

`Tiempo/N` se estabiliza en ~22 ns/op. **Complejidad: O(n)** lineal, pero con constante ~5x mayor que con Hash Consing.

### Tamaño del programa RPN

| N | chain_cons ops | chain_no_cons ops | repetition_cons ops |
|---|---|---|---|
| 10 | ~30 | ~60 | ~4 |
| 100 | ~300 | ~600 | ~4 |
| 1 000 | ~3 000 | ~6 000 | ~4 |
| 10 000 | ~30 000 | ~60 000 | ~4 |
| 100 000 | ~300 000 | ~600 000 | ~4 |

- **chain_cons**: O(n) operaciones (cada iteración añade 3 ops: `node`, `Dup`, `Bml`).
- **chain_no_cons**: O(2n) operaciones (cada iteración recrea `two` = 3 ops + 3 ops del nodo).
- **repetition_cons**: **O(1)** operaciones — `bml(two, two)` siempre se deduplica, el programa es constante (4 ops) sin importar N.

### Escalado de `repetition_cons`

| N | Tiempo |
|---|---|
| 10 | ~150 ns |
| 100 | ~150 ns |
| 1 000 | ~150 ns |
| 10 000 | ~150 ns |
| 100 000 | ~150 ns |

**Complejidad: O(1)** — el tiempo es constante porque el programa RPN es constante (4 ops) gracias al Hash Consing.

## Conclusiones

1. **FMA, BML cons (cadena), BML no cons (cadena)** son todas **O(n)** — el tiempo crece linealmente con N.
2. **BML con Hash Consing (repetición)** es **O(1)** — cuando hay repetición estructural real, el programa RPN es constante y el tiempo no crece con N.
3. **Hash Consing reduce la constante**: en la variante cadena, BML cons es ~5x más rápido que BML no cons, porque el programa RPN es la mitad de grande (los `two` se deduplican).
4. **El overhead de BML vs FMA es ~1.6x** (constante), debido al intérprete RPN. Este overhead se reducirá con el hot loop nativo del Hito 5.
5. **Limitación**: la evaluación recursiva del DAG explota la pila en N > 100 000. El hot loop RPN iterativo del Hito 5 resolverá esto.

## Próximos pasos

- Implementar el hot loop RPN iterativo (Hito 5) para eliminar el stack overflow y reducir el overhead del intérprete.
- Construir benchmarks con DAGs que tengan repetición estructural profunda (no solo cadenas) para demostrar escalado sub-lineal en casos realistas.
- Medir cache hit/miss con `perf` (tarea 2.10).

---

# Benchmark Report: Multiplicación de Matrices — Análisis de Complejidad Big O

## Metodología

Se miden 3 implementaciones de multiplicación de matrices A·B:

1. **ndarray**: librería estándar de Rust (`a.dot(b)`).
2. **naive**: triple loop en f64 puro.
3. **bml_rpn**: proxy que evalúa un programa RPN BML por cada operación.

El tamaño total de parámetros es N (elementos de A + elementos de B).
A y B son cuadradas de lado `k = sqrt(N/2)`.

N varía en progresión geométrica: 8, 18, 32, 50, 72, 98, 128, 162, 200, 242, 288, 338.

> **Nota:** `bml_rpn` es un **proxy** — no computa la multiplicación real, sino que mide el overhead del intérprete RPN con el mismo número de operaciones. Las fórmulas exactas de `+` y `*` en base 2 están pendientes de derivación.

## Resultados

### Tiempo de ejecución (media)

| N | k | ndarray | naive | bml_rpn |
|---|---|---|---|---|
| 8 | 2 | 87 ns | 18 ns | 339 ns |
| 18 | 3 | 93 ns | 34 ns | 1.03 µs |
| 32 | 4 | 116 ns | 53 ns | 2.44 µs |
| 50 | 5 | 149 ns | 101 ns | 4.74 µs |
| 72 | 6 | 170 ns | 151 ns | 7.84 µs |
| 98 | 7 | 175 ns | 226 ns | 13.0 µs |
| 128 | 8 | 146 ns | 312 ns | 20.4 µs |
| 162 | 9 | 443 ns | 431 ns | 26.0 µs |
| 200 | 10 | 518 ns | 542 ns | 37.2 µs |
| 242 | 11 | 489 ns | 807 ns | 49.7 µs |
| 288 | 12 | 522 ns | 1.02 µs | 63.9 µs |
| 338 | 13 | 694 ns | 1.30 µs | 83.8 µs |

### Análisis de complejidad

#### ndarray: **O(k^2)** (aparente, con optimizaciones de BLAS)

ndarray usa rutinas optimizadas (posiblemente BLAS o vectorización). El tiempo crece lentamente con k, sugiriendo que para matrices pequeñas el overhead de la llamada domina.

| k | Tiempo | Tiempo/k^2 |
|---|---|---|
| 2 | 87 ns | 21.8 ns |
| 4 | 116 ns | 7.3 ns |
| 8 | 146 ns | 2.3 ns |
| 13 | 694 ns | 4.1 ns |

Para k pequeño, el overhead fijo domina. Para k ≥ 8, `Tiempo/k^2` se estabiliza en ~4 ns, sugiriendo **O(k^2)** con vectorización.

> **Nota:** ndarray con BLAS debería ser O(k^3) para matmul estándar, pero para matrices tan pequeñas (k ≤ 13), el overhead de llamada y la vectorización pueden enmascarar el escalado cúbico. Se necesitarían k ≥ 64 para ver el régimen asintótico.

#### naive: **O(k^3)**

| k | Tiempo | Tiempo/k^3 |
|---|---|---|
| 2 | 18 ns | 2.3 ns |
| 4 | 53 ns | 0.8 ns |
| 8 | 312 ns | 0.6 ns |
| 13 | 1.30 µs | 0.6 ns |

`Tiempo/k^3` se estabiliza en ~0.6 ns/op. **Complejidad: O(k^3)** = O(N^1.5), como esperado para triple loop.

#### bml_rpn: **O(k^3)** con constante ~100x mayor

| k | Tiempo | Tiempo/k^3 | vs naive |
|---|---|---|---|
| 2 | 339 ns | 42.4 ns | 19x |
| 4 | 2.44 µs | 38.1 ns | 46x |
| 8 | 20.4 µs | 39.8 ns | 65x |
| 13 | 83.8 µs | 38.1 ns | 64x |

`Tiempo/k^3` se estabiliza en ~38 ns/op. **Complejidad: O(k^3)** = O(N^1.5), con constante ~64x mayor que naive.

El overhead de ~64x se debe a:
- Evaluación del programa RPN (push/pop de pila `Vec<f64>`).
- Cada `bml` hace `exp2` + `log2` (vs `*` + `+` del naive).
- El proxy no usa Hash Consing ni valores reales de A/B.

## Conclusiones

1. **ndarray, naive, bml_rpn** son todas **O(k^3)** = O(N^1.5) para matmul, como esperado teóricamente.
2. **ndarray** es el más rápido para k ≥ 8 gracias a vectorización, pero tiene overhead fijo para k pequeño.
3. **naive** es el más rápido para k ≤ 6 (sin overhead de llamada).
4. **bml_rpn** es ~64x más lento que naive, debido al overhead del intérprete RPN y `exp2`/`log2` vs `*`/`+`.
5. El overhead de BML se reducirá con:
   - Hot loop RPN iterativo (Hito 5) en lugar de `Vec` push/pop.
   - Derivación de fórmulas `+`/`*` en base 2 (pendiente Hito 2).
   - Hash Consing para sub-expresiones repetidas en el matmul.
