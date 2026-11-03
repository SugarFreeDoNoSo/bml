# Entrega local: GGUF → BMLGGUF

## Estado verificable

- Implementación integrada en nueve commits locales; contrato, documentación,
  evidencia y herramientas seleccionados para el commit documental de entrega.
- Gate del snapshot entregado hasta `bb1f9ff`: **630 pruebas Rust y 63 Python**,
  cero discrepancias históricas, trazabilidad 28/28 y cero diagnósticos nuevos.
  Evidencia: `quality/evidence/delivery-server-fallible.json`. El servidor maneja
  los resultados falibles de tokenizer e inferencia; su reescritura pendiente
  de batching/remoto permanece en el árbol de trabajo.
- Controles anteriores del árbol completo: gate 703 pruebas; GGUF sin/con
  features 50/57; cobertura workspace 87,02 % líneas / 79,79 % funciones.
  Esos conteos no describen el snapshot seleccionado de 630 pruebas.
- Mutación completa del crate: 443 candidatos, 391 detectados, 52 no compilables,
  cero supervivientes y timeouts. Evidencia: `quality/evidence/model-weights-mutation-v9.json`.
- Validación estricta global de OpenSpec: 9 elementos aprobados. Esto valida
  estructura, no satisface los requisitos abiertos de otras specs.

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
No se ha publicado ni archivado el cambio.

Los hashes exportados anteceden a este documento y al informe de preparación;
los cambios posteriores de entrega/checklist no alteran las fuentes verificadas.

## Condiciones restantes de 5.3

Los prerrequisitos siguientes ya están integrados:

1. Números exactos y certificados de `crates/domain/src/number/`, incluidos
   `certified.rs`, `exact.rs`, `stream.rs` y sus exportaciones.
2. Soporte GGUF/Q6_K y validación de metadatos en `crates/gguf`.
3. Perfil causal y kernels simbólicos del compiler/runtime, tokenizer compartido
   y validadores de `bml-inference`.
4. Registro workspace, dependencias y lockfile coherentes con esas bases.

La preparación e integración local del código está acreditada. Antes de archivar:

1. Contrato, documentación, evidencias y herramientas de medición revisados en
   el índice seleccionado. OpenSpec estricto pasa en su snapshot. La herramienta
   se compiló y ejecutó desde cero: conversión/carga/ejecución con exit 0/0/0,
   fuente eliminada antes de cargar y tokens `[1,0]`. Esta repetición usa build
   debug para comprobar reproducibilidad funcional, no rendimiento release.
   Evidencia: `quality/evidence/delivery-documentation.json`.
2. Conservar el alcance de cada control: v9 certifica mutación del crate de pesos;
   el gate de 630 pruebas certifica el snapshot integrado. No son una campaña
   global de mutación ni una nueva medición de cobertura del snapshot.
3. Publicar solo cuando exista autorización expresa. No hay push realizado.
4. Archivar únicamente tras completar las condiciones de cierre; 5.3 permanece
   abierta mientras publicación/archivo estén pendientes. Este documento no
   convierte los requisitos de otras specs en bloqueos de implementación de pesos.

## Requisitos restantes de las otras specs

| Cambio | Pendientes reales |
|---|---|
| `bml-1-0-release` | 6.2: mutación completa aplicable al workspace; 6.3: commit/publicación. El éxito del crate de pesos no certifica el workspace. |
| `bml-exact-numbers-and-matrices` | 7.3–7.4: ejecución y campaña ARM/NEON real; 9.3: entrega/archivo. Compilación cruzada no sustituye ejecución ARM. |
| `bml-implementation-plan` | 5.3: comportamiento L1 con evidencia adecuada; 5.10: estrés cloud; 5.11: hito completo. |
| `bml-inference-pipeline` | 8.6: ruta heredada, escalabilidad y validación numérica real; 9.2: Loom del work-stealing real; 9.6: contadores L1i; 10.3: commit/push. |
| `fix-placeholders` | 9.3: commit/push. |
| `tensor-library-foundation` | 4.5: complejos/síntesis y ramas, fuera del MVP; 5.4: completar o separar formalmente la migración antes de archivar. |

La campaña histórica `target/spec-mutations-2026-11-03/mutants.out/outcomes.json`
sigue con `end_time=null`: 1410 candidatos declarados, 379 caught, 47 missed,
2 timeout y 982 unviable registrados. No es un resultado final acreditado.

Los requisitos ARM/PMU/cloud necesitan acceso al entorno correspondiente. La
mutación global, la verificación de work-stealing, el trabajo simbólico restante
y la revisión de commits siguen siendo tareas de ingeniería pendientes; no se
atribuyen automáticamente a falta de hardware.
