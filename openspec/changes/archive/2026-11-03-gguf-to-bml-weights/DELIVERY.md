# Entrega y cierre: GGUF → BMLGGUF

## Estado verificable

- Implementación publicada en nueve commits; contrato, documentación, evidencia
  y herramientas publicados en `1ef5f5d`. La biblioteca está en `858d8fc`.
  Su ascendencia en `origin/master` se verificó el 2026-11-03.
- Gate del snapshot entregado hasta `bb1f9ff`: **630 pruebas Rust y 63 Python**,
  cero discrepancias históricas, trazabilidad 28/28 y cero diagnósticos nuevos.
  Evidencia: `quality/evidence/delivery-server-fallible.json`. El servidor maneja
  los resultados falibles de tokenizer e inferencia. Los cambios posteriores
  de batching/remoto tienen su propio alcance y evidencia.
- Controles anteriores del árbol completo: gate 703 pruebas; GGUF sin/con
  features 50/57; cobertura workspace 87,02 % líneas / 79,79 % funciones.
  Esos conteos no describen el snapshot seleccionado de 630 pruebas.
- Mutación completa del crate: 443 candidatos, 391 detectados, 52 no compilables,
  cero supervivientes y timeouts. Evidencia: `quality/evidence/model-weights-mutation-v9.json`.
- Validación estricta global de OpenSpec de la entrega: 9 elementos aprobados. Esto valida
  estructura, no satisface los requisitos abiertos de otras specs.
- Auditoría de cierre: los ocho requisitos están acreditados. La biblioteca
  coincide con el commit publicado; los hashes de gate, perfiles GGUF,
  cobertura, mutación y campañas por fase se comprobaron de nuevo sin repetirlas.
  Evidencia: `quality/evidence/model-weights-archive-2026-11-03.md`.

## Commits integrados

| Commit | Alcance | Verificación aislada |
|---|---|---|
| `d03d4a2` | Números exactos y matrices | 158 pruebas domain |
| `999f6df` | Q6_K y filas GGUF | 50/57 pruebas sin/con features |
| `858d8fc` | Biblioteca BMLGGUF | 50/76 pruebas sin/con features |
| `bec670b` | Metadatos y tokenizer compartidos | 54 pruebas inference |
| `dffba3c` | Compiler causal y carga preparada | 145 pruebas compiler; build dedicado |
| `fa2dceb` | Proyecciones simbólicas preparadas | 48 pruebas runtime; build dedicado |
| `1c984ee` | Runtime causal checked-F32 | 400 pruebas domain/compiler/runtime; build nuevo |
| `29ca9c4` | CLI convert/inspect/run | 16 pruebas CLI; build nuevo |
| `bb1f9ff` | Compatibilidad del servidor | Gate integrado: 630 Rust + 63 Python |

Cada paso conserva su selección y hashes de logs en `quality/evidence/delivery-*.json`.
El gate integrado se ejecutó sobre el árbol
`7a298d15b1bf6345a9af74f0e7a14a233f42be27`; `bb1f9ff` solo añade a ese árbol
su JSON de evidencia. Se usó un build nuevo con símbolos de depuración
desactivados por capacidad de disco, conservando las aserciones de prueba.

## Material histórico de revisión

Base Git: `40b3f1be2d3f97028b741750ed61ab72f501a555`.

En `target/quality/model-weights-delivery/`:

- `primary.patch`: crate, nuevas interfaces, pruebas, medición, contrato y evidencia.
- `integration-review.patch`: integración en manifiestos, compiler, runtime,
  inference, CLI y documentación compartida. Incluye trabajo previo interdependiente;
  requiere revisión por fragmentos antes de preparar un commit aislado.
- `manifest.json`: 69 archivos con SHA-256 y pertenencia a cada parche.
- `openspec-validation.log`: validación estricta de los cambios y specs disponibles.

Ambos parches pasaron `git apply --reverse --check` contra el árbol exportado.
Eso comprueba correspondencia con el snapshot, **no** compilación autónoma sobre
HEAD. El índice Git se mantuvo intacto durante esa exportación. Estos parches
anteceden a los commits anteriores y no deben aplicarse sobre la entrega actual.
La publicación posterior está acreditada en
`quality/evidence/publication-spec-closure-2026-11-03.md`.

Los hashes exportados anteceden a este documento y al informe de preparación;
los cambios posteriores de entrega/checklist no alteran las fuentes verificadas.

## Cierre de 5.3

Los prerrequisitos siguientes ya están integrados:

1. Números exactos y certificados de `crates/domain/src/number/`, incluidos
   `certified.rs`, `exact.rs`, `stream.rs` y sus exportaciones.
2. Soporte GGUF/Q6_K y validación de metadatos en `crates/gguf`.
3. Perfil causal y kernels simbólicos del compiler/runtime, tokenizer compartido
   y validadores de `bml-inference`.
4. Registro workspace, dependencias y lockfile coherentes con esas bases.

La preparación, integración, publicación y revisión de contratos están acreditadas:

1. Contrato, documentación, evidencias y herramientas de medición revisados en
   el índice seleccionado. OpenSpec estricto pasa en su snapshot. La herramienta
   se compiló y ejecutó desde cero: conversión/carga/ejecución con exit 0/0/0,
   fuente eliminada antes de cargar y tokens `[1,0]`. Esta repetición usa build
   debug para comprobar reproducibilidad funcional, no rendimiento release.
   Evidencia: `quality/evidence/delivery-documentation.json`.
2. Conservar el alcance de cada control: v9 certifica mutación del crate de pesos;
   el gate de 630 pruebas certifica el snapshot integrado. No son una campaña
   global de mutación ni una nueva medición de cobertura del snapshot.
3. La publicación fue autorizada y realizada: los commits de código y `1ef5f5d`
   son antecesores de la rama remota verificada. Los parches históricos no son
   entregas pendientes.
4. El archivo formal conserva los anexos normativos en
   `openspec/specs/bml-model-weights/{api.md,format-v1.md}` y el historial de la
   propuesta en `openspec/changes/archive/2026-11-03-gguf-to-bml-weights/`.
   La evidencia de cierre registra el resultado de la validación y del archivo.

## Alcance respecto de otras specs

El cierre de pesos no acredita mutación global, ejecución ARM/NEON, contadores
PMU/L1, estrés cloud ni escalabilidad de inferencia real. Los estados vigentes
de esas tareas pertenecen a sus propios cambios y evidencias.

El requisito de inferencia de este cambio exige un fixture autónomo con oráculo
independiente. Su escenario de campaña real permite registrar fallos acotados
y fases no ejecutadas. La nueva inferencia simbólica v4 no es un prerrequisito
de archivo de `gguf-to-bml-weights`.
