# Foundation 4.5: complejos y síntesis acotada

Base de implementación: `330be43a335e7366b68a10229896f874659acb0b`.

## Contratos implementados

* `bml_domain::symbolic::complex`: operador EML/BML complejo finito, rama
  logarítmica fija explícita, corte negativo con cero imaginario firmado,
  evaluación iterativa y presupuestos de ocurrencias/profundidad. Logaritmo del
  módulo escalado para componentes extremas; errores tipados de dominio/rango.
* `bml_domain::symbolic::synthesis`: enumeración determinista por tamaño y orden
  prefijo, hasta 255 nodos por árbol, presupuesto independiente de candidatos,
  tolerancia absoluta y estado de terminación. Los candidatos numéricamente
  inválidos consumen intentos. Agotamiento y éxito por tolerancia son distintos.

No se añaden dependencias a domain. La evaluación real existente conserva su
contrato; la nueva síntesis aproximada no certifica igualdad simbólica ni sustituye
la síntesis exacta del escalar `Bml`. Políticas y API: `docs/SYMBOLIC_EXTENSIONS.md`.

C73 y C74 registran siete pruebas públicas. Las expectativas incluyen identidades
analíticas complejas, ambos lados del corte, ramas no principales, límites
exactos de nodos/profundidad, logaritmos de componentes máximas/subnormales,
árboles pequeños conocidos y los 23 árboles hasta nueve nodos (suma de números
de Catalan), incluyendo candidatos con `log(0)`. Rustdoc compila el ejemplo.

## Verificación de la entrega aislada

Snapshot: `target/delivery-snapshots/symbolic-extensions`. Ejecutor:
`target/quality/verify_symbolic_extensions.py`. El primer build usa un target
nuevo; cada variante se reconstruye sobre la misma ruta y se restaura después.

Original y restaurado: **7/7 contratos**. Nueve controles negativos fallan por
aserciones (exit 101): omitir rama; perder signo del corte; usar `hypot` directo
en magnitudes extremas; omitir límites de nodos/profundidad; permitir un intento
extra; omitir el techo de 255 nodos; excluir el borde de tolerancia; invertir
operandos al reconstruir el árbol. Fuentes mutadas y logs están preservados.

`./scripts/check.sh HEAD` termina con exit **0**: **666 pruebas Rust, 63 Python,
trazabilidad 31/31, cero discrepancias y cero diagnósticos nuevos** (67 actuales,
baseline 192). Hashes originales/restaurados y de todos los logs verificados.

Registro: `target/quality/symbolic-extensions-verification.json`, SHA-256
`d3f7900d0c10e97f513c0642a7a6d05556580ca63183dd8b0380e37491a19741`.
Log del gate: `target/quality/symbolic-extensions-delivery-gate.log`, SHA-256
`ab92118b9db97d4b3b3a4d67c0f258a6dd65e3248b3b9aa5e7af50ad048596d9`.

## Verificación del árbol completo

`./scripts/check.sh origin/master` termina con exit **0**: **736 pruebas Rust,
82 Python, trazabilidad 74/74, cero discrepancias y cero diagnósticos nuevos**
(63 actuales, baseline 192). Se verificaron los hashes de fuentes antes/después.
Log `target/quality/symbolic-extensions-worktree-gate.log`, SHA-256
`37d24abca2dde6f536582a57a3ca610774a0eab5ba0cd1d137d198680571b438`.
Registro JSON homónimo, SHA-256
`bf329741326c1092e2883dc6df114f12ab2302662271b9da60d2e4e007d2018c`.

OpenSpec estricto: **9 aprobados, 0 fallidos**, solo validación estructural.
El primer intento detectó dos textos de requisito demasiado largos; se movieron
detalles a escenarios sin alterar código, contratos o pruebas. Ambos logs se
conservan; el aprobado es `target/quality/symbolic-extensions-openspec-v2.log`,
SHA-256 `41509253a9e7f8de902e8da4d67e858bd27224c1e84792c9746df72ce39443e0`.
El snapshot de pruebas conserva la redacción anterior; la entrega usa la
redacción estructuralmente aprobada.

## Mutaciones y obligaciones independientes

Está lanzada una campaña completa **de los dos módulos nuevos**, con baseline,
`--locked`, lista de candidatos y restauración de hashes en
`target/quality/symbolic-extensions-mutation-execution.json`. Su finalización y
resultado se registrarán al recibir la notificación; no se declara aprobada aquí.
Los nueve controles negativos son evidencia focalizada, no aprobación global.

La campaña global de 3025 candidatos usa un snapshot anterior y no contiene
esta implementación. Los requisitos de rango/escalabilidad de pipeline 8.6,
hardware ARM/PMU/cloud y autorización de publicación conservan sus estados.
