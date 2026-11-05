# Revisión pendiente de dos supervivientes de exactitud

**Estado: pendiente de revisión; raw `MissedMutant` preservados; sin modificar
política.** Este documento presenta pruebas de invariantes para revisión, no
aprueba exclusiones ni reclasifica resultados de la campaña. No modifica el
gate, sus denominadores, los baselines ni los resultados originales.
El alcance se limita a los dos IDs descritos y a la limpieza RMS separada;
no clasifica otros supervivientes.

La regla vigente en [docs/QUALITY.md](../../docs/QUALITY.md), línea 57, dice:

> Cada mutante viable debe ser detectado.

> No hay exclusiones por equivalencia preaprobadas.

El archivo de política leído tiene SHA-256
`2e38d8c3128fab8f90d3cfeb2661ee60f2ae299a6f66b5bd4128fa3bb9db1a16`.

El porcentaje sigue siendo `caught / (caught + missed + timeout)`. Por tanto,
estas pruebas estáticas no convierten una campaña con supervivientes en una
campaña aprobada. Esta revisión no ejecuta Cargo, builds ni replays nuevos.

## Procedencia y resultados originales

Los dos casos pertenecen al shard `2/64` de la campaña GitHub Actions
`41950170549`, con cargo-mutants `27.1.0`. Ambos conservan el resultado original
`MissedMutant`.

| Campo | Valor |
| --- | --- |
| Revisión congelada | `9a6bc9680d5915576ca92528f60956efe9796cef` |
| Árbol Git | `42ec84ca07eeb4d826cfe40725b9dac6909a8ffe` |
| Base completa del desarrollo | `40b3f1be2d3f97028b741750ed61ab72f501a555` |
| SHA-256 de `crates/domain/src/number/exact.rs` | `da29637bbc49161ca1e684f87bd0e1f971af89e9ed9ff8eda3f2d7af024a73bf` |

Los hashes siguientes corresponden a bytes de los archivos descargados, no a
una serialización nueva de su contenido JSON. Las rutas son relativas al root
del repositorio y permanecen bajo `target/`:

| Archivo original | SHA-256 |
| --- | --- |
| `target/quality/mutation-shard-41950170549-2/target/quality/shard-result-2.json` | `4cf4e4f1d6db858d5a1c6054482e3a9cdde5b0236262cf6f937550784c213bfe` |
| `target/quality/mutation-shard-41950170549-2/target/final-mutation-shard-2/mutants.out/outcomes.json` | `cc8df9c899b8072c9bc059b17da520dfd71f85691ad5378238fb77de2a689686` |
| `target/quality/mutation-shard-41950170549-2/target/final-mutation-shard-2/mutants.out/missed.txt` | `a5ae462b952b71b05450bb38b7f58f65da25bb31786d64fae5406d110bcfe622` |

Los diffs citados abajo están en el subdirectorio `diff/` de ese mismo
`mutants.out`. El archivo fuente congelado se recupera sin ejecutar pruebas:

```sh
git show 9a6bc9680d5915576ca92528f60956efe9796cef:crates/domain/src/number/exact.rs
```

## Caso 1: producto con un operando cero

ID completo:

```text
crates/domain/src/number/exact.rs:470:21: replace || with && in multiply
```

- Diff: `crates__domain__src__number__exact.rs_line_470_col_21.diff`.
- SHA-256 del diff: `6e2d2b0ef2e4c4acafea2a590943439b69f28c192a2aa9e06f4dd1c931ab99b6`.
- Fuente: [exact.rs](../../crates/domain/src/number/exact.rs), líneas 57–74,
  170–194, 371–445 y 469–497 en la revisión congelada.

La guarda original retorna el vector vacío si cualquiera de los operandos es
vacío. El mutante retorna allí solamente si ambos son vacíos.

La afirmación revisada es **equivalencia de valores y errores numéricos bajo
asignaciones exitosas**, para los operandos canónicos admitidos por las APIs
públicas actuales. No es equivalencia total de comportamiento ni de recursos.

Condiciones e invariantes:

1. Los componentes de `ExactRational` son privados y canónicos. Un numerador
   vacío representa cero y el denominador es positivo. Los getters exponen
   slices inmutables, no una forma de introducir limbs no canónicos.
2. Los callers de `multiply` en `checked_add`, `checked_mul` y `checked_div`
   comprueban primero ambos racionales contra `limits.max_bits`. `checked_sub`
   delega en `checked_add`; la división rechaza previamente el divisor cero.
3. Si exactamente un slice es vacío, el otro tiene longitud de bits `b` con
   `1 <= b <= max_bits`. La cota usada por el mutante es entonces `b - 1`:
   no hay underflow y no supera ese límite.
4. El vector temporal del mutante queda enteramente a cero. Con el primer
   slice vacío no hay iteraciones exteriores; con el segundo vacío, los
   cuerpos interiores no iteran y el carry escrito es cero. `normalize`
   elimina todos esos limbs y la comprobación final recibe cero bits.
5. Con ambos slices vacíos, ambas guardas retornan cero. Con ambos no vacíos,
   ambos programas ejecutan la misma multiplicación. Los callers reciben los
   mismos limbs y realizan la misma reducción matemática y comprobaciones
   posteriores.

La diferencia de recursos es real: el mutante puede reservar y recorrer un
vector de `a.len() + b.len()` limbs donde el original retorna inmediatamente.
Aunque su longitud final sea cero, su capacidad puede permanecer asignada.
Si esa reserva falla, puede retornar `ExactInputError::Limit` donde la guarda
original no necesita reservar. No se afirma igualdad de asignaciones, memoria
retenida, trabajo ni tiempos. No se propone usar OOM, direcciones de memoria o
capacidad interna como regresiones de un contrato público inexistente.

La guarda de cero sigue siendo una optimización útil. Este análisis no propone
eliminarla ni cambiar su sintaxis para ocultar la mutación.

## Caso 2: ensamblado de campos IEEE con OR o XOR

ID completo:

```text
crates/domain/src/number/exact.rs:623:46: replace | with ^ in ExactRational::to_f64_rounded
```

- Diff: `crates__domain__src__number__exact.rs_line_623_col_46.diff`.
- SHA-256 del diff: `982412c3a7b4dc50d6624fdd49836b4d887345bc9a55dc14e9716cc1678de69c`.
- Fuente: [exact.rs](../../crates/domain/src/number/exact.rs), líneas 555–633
  en la revisión congelada.

La expresión afectada es únicamente el ensamblado de la magnitud de un
resultado normal:

```text
((exponent + 1023) << 52) | (significand - 2^52)
```

Prueba de campos disjuntos:

1. El cálculo entero del exponente, incluida su corrección por comparación,
   selecciona el intervalo binario del racional positivo. En la rama normal,
   el significando antes de redondear está en `[2^52, 2^53)`.
2. El redondeo al más próximo, con empates a par, puede producir `2^53`.
   Ese único carry se renormaliza desplazando el significando y aumentando
   el exponente. El overflow se rechaza antes del ensamblado.
3. Por tanto, al llegar a la expresión, `2^52 <= significand < 2^53` y
   `-1022 <= exponent <= 1023`. La fracción `significand - 2^52` ocupa
   únicamente los bits 0–51; el exponente sesgado desplazado ocupa solamente
   los bits 52–62. Su intersección es cero.
4. Para esos operandos, `E | F == E ^ F`. El signo se añade después en el
   bit 63 mediante otra operación que este mutante no cambia.

El cero retorna antes. Los resultados subnormales pasan por la otra rama,
que utiliza directamente el significando; también el redondeo subnormal
capaz de alcanzar el menor normal se trata allí. Esos casos no ejecutan la
expresión mutada. Los errores de límites, overflow y underflow tampoco
dependen de elegir OR o XOR en ese ensamblado.

Ninguna entrada válida que alcance esa expresión rompe la separación de
campos. Las pruebas públicas de redondeo y fronteras IEEE siguen siendo
pertinentes, pero no pueden distinguir esta sustitución bajo esos invariantes.
No se cambia el empaquetado ni se propone un cambio sintáctico para eliminar
el superviviente del listado.

## Limpieza RMS separada: una guarda redundante

Este apartado no declara equivalente ni aprobado el mutante RMS:

```text
crates/domain/src/transformer/bounded.rs:260:62: replace - with / in BMLTransformer::bounded_rms_stage_literals
```

| Evidencia | SHA-256 |
| --- | --- |
| Fuente RMS de la campaña `9a6bc968` | `328a439c25cf59fda16e0f720267baa053bf070ba9f833b896ce8ca5f560d80b` |
| Diff `crates__domain__src__transformer__bounded.rs_line_260_col_62.diff` | `b85a9db39157f926aa333802f5dc73a57e52787e6cd6a9c114ee140b76c4d815` |
| Fuente RMS con la limpieza preparada sobre `22c443649d9d5ba52c929dc707553cf0f7e23fa8` | `6d57cd1c4d00e9e275328667ad01c9b0624f74dee738cc24231dfb42a16c0727` |

En [BMLTransformer](../../crates/domain/src/transformer.rs), `soa` es privado,
su acceso público es de solo lectura y no existe una admisión de `NodeSoA`
arbitrario. `constant()` añade una entrada al pool y un nodo; los demás
constructores añaden nodos sin ampliar el pool. Así se conserva
`const_pool.len() <= soa.len()`.

La guarda preservada comprueba, mediante suma comprobada, que
`soa.len() + 16384 <= u32::MAX`. Si pasa, el pool también tiene ese margen;
como `16384 > 17`, no puede incumplir el antiguo límite independiente de las
17 constantes de la etapa. El término `const_pool.len() > u32::MAX - 17`
era, por tanto, redundante en los estados alcanzables del constructor.

Sustituir la resta por división no es equivalente globalmente: introduce un
rechazo prematuro a partir de 252.645.136 constantes, mientras la guarda de
nodos todavía podría pasar. No se propone reservar cientos de millones de
nodos para probar ese estado. La limpieza retira únicamente el término
redundante original y documenta el invariante; conserva el preflight de nodos,
las comprobaciones de forma e IDs y la política de evaluación numérica.

El `MissedMutant` histórico de RMS permanece en los archivos originales.
Una limpieza de producción con otra fuente y hash no lo convierte
retroactivamente en `CaughtMutant`.

## Decisión pendiente

Se solicita revisar las condiciones de las dos pruebas de invariantes y su
alcance respecto de la política vigente. Esta nota no autoriza exclusiones,
no elimina supervivientes del denominador y no certifica el resultado global
de la campaña. Cualquier decisión posterior debe ser explícita y conservar
los resultados originales y su procedencia.

### Condiciones propuestas para una autorización futura

Sin aplicar todavía ninguna excepción, una política revisada podría admitir
únicamente entradas identificadas por mutante, fuente, diff y prueba de
invariantes, con revisión independiente. Cada revisión tendría que delimitar
también las diferencias de recursos y las garantías públicas que siguen
siendo exigibles. Una modificación de la fuente o de esas condiciones
invalidaría la autorización hasta una nueva revisión.

Los informes originales conservarían `MissedMutant` y el porcentaje bruto;
una equivalencia aprobada se mostraría por separado y nunca se contaría como
`CaughtMutant`. Todo superviviente sin revisión válida, timeout, informe
incompleto o error de infraestructura seguiría haciendo fallar el gate.
Esta propuesta requiere autorización expresa porque cambia la regla vigente;
el código del gate actual permanece intacto.
