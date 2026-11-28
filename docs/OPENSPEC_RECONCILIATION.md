# Conciliación de tareas OpenSpec — 2026-11-28

Base revisada: `e84e07eac3f39d7a941829687f99247c0a4bfda0`, confirmada como
`master` mediante lectura remota. Esta revisión cambia documentación y estados;
no integra los candidatos nativos, ejecuta inferencia, campañas o PMU, ni publica
una entrega. Las evidencias históricas conservan su revisión y alcance originales.

## Regla de estado y recuento

El [generador](../tools/build_dashboard.py) cuenta cinco planes: los cuatro
históricos y `native-f32-isa-backends`. Excluye `archive/`. Una casilla `[x]`
cuenta como completa incluso si su nota dice «parcial». Una vacía se clasifica
como diferida sólo cuando su misma línea contiene `Deferred`; las demás son
pendientes activas. Las notas en líneas separadas no aparecen en el tablero.
No se modifica ese algoritmo ni se añaden casillas para contar evidencias.

| Plan | Antes: completas / activas / diferidas | Después: completas / activas / diferidas |
| --- | --- | --- |
| bml-implementation-plan | 49 / 2 / 1 | 46 / 5 / 1 |
| bml-inference-pipeline | 22 / 3 / 0 | 22 / 3 / 0 |
| fix-placeholders | 51 / 2 / 0 | 51 / 2 / 0 |
| bml-1-0-release | 66 / 2 / 0 | 65 / 4 / 0 |
| native-f32-isa-backends | 0 / 13 / 1 | 0 / 13 / 1 |
| **Total (212 → 213)** | **188 / 22 / 2** | **184 / 27 / 2** |

Las 24 abiertas anteriores incluían dos diferidas. La conciliación v1 dejó
184 completas, 26 activas y dos diferidas de 212, con cuatro reaperturas.
La actualización posterior pedida por Matías añade únicamente
`bml-1-0-release/4.6`: ahora son 213 tareas, con 29 abiertas, de las cuales
27 activas y dos diferidas. El ZIP v1 entregado conserva su contenido y recuento.
Son conteos de los checklists, no una recertificación de todas las pruebas
históricas en el HEAD actual. El aumento del denominador responde al nuevo hito
autorizado; no se archiva trabajo pendiente ni se reduce la aceptación.

Los estados descriptivos de esta revisión significan:

- **Publicada:** implementación/evidencia presente en la base; no presupone que
  haya satisfecho todas las cláusulas de una tarea agregada.
- **Verificada local:** resultados de un candidato identificado, aún sin integrar;
  no convierten en `[x]` una implementación ausente de master.
- **Parcial:** sólo una parte del criterio tiene evidencia; conserva `[ ]`.
- **Bloqueada:** falta un input, host, permiso o resultado concreto; sigue activa.
- **Diferida:** postergación ya explícita; se conservan exactamente las dos existentes.
- **Decisión local cerrada sin promover:** un experimento se descarta dentro de
  su alcance; no instala el backend ni sustituye la evidencia exigida para cerrar
  su tarea documental en el repositorio.

## Cuatro correcciones de casillas

| ID estable | Cambio | Motivo y evidencia |
| --- | --- | --- |
| bml-implementation-plan/2.8 | `[x]` → `[ ]`, parcial | Exige crecimiento sublineal. Su nota reconoce una cadena lineal; el [benchmark de trabajo equivalente](benchmarks/REPORT.md) declara que no demuestra ventajas asintóticas. Una mejora constante no prueba `O(n^k), k<1`. Falta definir n y verificar una familia de repetición estructural bajo trabajo comparable. |
| bml-implementation-plan/2.10 | `[x]` → `[ ]`, bloqueada | Falta medición hardware L1/L2 de la carga requerida. [perf_cache.rs](../tests/stress/perf_cache.rs) usa eventos genéricos; no acredita L1/L2 ni el supuesto fallback software. Los dos eventos de la tarea son de L1: faltan también L2 y denominadores para hits. |
| bml-implementation-plan/2.13 | `[x]` → `[ ]`, parcial | El cierre incluye adjuntar `perf`, todavía pendiente en 2.10. [runtime-cache-probe-functional.json](../quality/evidence/runtime-cache-probe-functional.json) sólo acredita funcionalidad. Loom de append-only/cola no instrumenta mmap ni la prueba shm. |
| bml-1-0-release/6.2.45 | `[x]` → `[ ]`, parcial | Las reparaciones `cce9347` y el ajuste posterior están publicadas; el replay aislado del worker verifica un caso, no el conjunto de timeouts de los 58 shards históricos. Falta su correspondencia y verificación causal. Se conserva «eliminar causas observadas»; sustituirlo por «aplicar reparaciones» reduciría la aceptación. No duplica la campaña global de 6.2. |

No se crea otra tarea PMU: 2.13 consume la evidencia de 2.10. El acceso a un host
autorizado con PMU puede compartirse con 5.3/9.6, conservando cargas y métricas
distintas. El [protocolo L1i](QUALITY.md) existente exige siete muestras de
`Runtime::execute_sub_fragments`, instrucciones y eventos L1i contados íntegramente
y el umbral previo; una medición L1i no sustituye automáticamente L1/L2 del estrés.

Se mantienen 2.11/2.12 con el alcance funcional y temporal de su
[prueba shm](../tests/stress/distributed_shm.rs): el intervalo incluye creación
de hilos, lectura, deserialización, evaluación y escritura, no latencia aislada
de transporte. No acredita dos hosts ni prueba Loom del mmap. Se mantienen
pipeline/9.2 por su entrega Loom posterior y pipeline/9.4 por su auditoría binaria;
sus contratos acotados no exigen que la medición PMU ya esté completa.

## Obligaciones históricas que continúan abiertas

Cada fila identifica una tarea existente; no es un nuevo checklist.

| ID estable | Estado | Evidencia utilizable y dependencia exacta |
| --- | --- | --- |
| bml-implementation-plan/5.3 | Parcial / bloqueada | Tamaño de símbolos y aislamiento publicados; falta comportamiento L1 medido. [Driver funcional](../quality/evidence/runtime-cache-probe-functional.json) y 5.13/5.14 preparan la carga, no sustituyen PMU. |
| bml-implementation-plan/5.10 | Diferida | Estrés cloud con host autorizado y bare metal PMU según el criterio conservado de 5.10; no se ejecuta ni autoriza en esta revisión. |
| bml-implementation-plan/5.11 | Parcial | Cierre agregado de runtime/graph/RPC/estrés; depende de 5.3/5.10, sin duplicar sus mediciones. |
| bml-inference-pipeline/8.6 | Parcial, componentes publicados | [Perfil acotado histórico](../quality/evidence/full-bounded-2026-11-03.json) y [programas compactos](../quality/evidence/compact-bounded-2026-11-03.md). Falta validación simbólica actual de modelos mayores y prompts adicionales con dominio/presupuestos conservados. ELF nativo, híbrido y cuatro filas no son el recorrido completo. |
| bml-inference-pipeline/8.6.7 | Parcial / bloqueada | [Receta original](REAL_PROMPT_DIAGNOSTICS.md) y [verificador](../quality/evidence/current-real-prompts-verifier/README.md): GGUF exacto, 13 archivos AOT y binario medido original; Hello `[1,15043]`, Hi `[1,6324]`, ocho vectores completos de 32.000 f64 y ambos summaries. Hello BOS/posición 0 no basta. Un sustituto de TinyLlama o recompilación posterior no es replay del binario original. |
| bml-inference-pipeline/9.6 | Bloqueada | PMU del cambio de fragmentos; compartir preparación/acceso con 5.3 y preservar su criterio L1i. Tamaño, Loom y shm no demuestran residencia. |
| fix-placeholders/8.5.8 | Parcial, transporte publicado | C100–C105 y [SSH loopback](../quality/evidence/ssh-loopback-shards.json) acreditan rangos, sesiones y ejecuciones locales acotadas. Faltan dos hosts distintos e inferencia completa del modelo mayor. Los nuevos pilotos híbridos locales tampoco los sustituyen. |
| fix-placeholders/8.5.9 | Pendiente | Auditoría final H01–H14/E01–E06 sobre la revisión integrada. El [inventario](../quality/evidence/placeholder-contract-audit.md) es preparación, no cierre 14/14; H01/E05 siguen parciales. |
| bml-1-0-release/6.2 | Parcial | Gate canónico, features, cobertura y mutaciones aplicables sobre la revisión entregada. La [preparación seca Release](RELEASE_MUTATION_DRY_RUN.md) registra `execution=not-run` y `campaign_evidence=false`. Falta ejecución completa y evidencia, no sólo preparación. |
| bml-1-0-release/6.3 | Pendiente | Publicaciones parciales históricas conservadas. La entrega final depende de 6.2 y su revisión/autorización; no equivale a publicar una release en esta revisión. |

Se conserva 8.6.5 como diagnóstico histórico completado. El punto H01 del informe
original solicitaba `[1,29958]`, ya realizado; se añadió una actualización a ese
informe para no pedir repetirlo. Eso no cierra el distinto replay Hello/Hi 8.6.7.
No se reabren archivos archivados ni se atribuyen fallos históricos al código actual.

## Release: alcance y causalidad

El intento `45479617529` corresponde a `10d2b2f`: 58/64 shards, 4.652/5.132
resultados, 3.903 caught, 329 missed, 26 timeout y 394 unviable. Faltan los
originales 29/33/36/39/40/43. Estos conteos no son supervivientes en HEAD.
Se retira la atribución «OOM 16 GiB causado por cuatro funciones»: un `exit 143`
no acredita esa causa ni el consumo de memoria. Las reparaciones estructurales
y sus CI conservan su evidencia específica en 6.2.43–6.2.46; no se reclasifican
resultados históricos ni se declara eliminación universal de timeouts. La
casilla 6.2.45 se reabre: las reparaciones enumeradas y un replay acotado no
demuestran por sí solos la eliminación de las causas históricas exigida. Se
conserva ese resultado como aceptación, sin sustituirlo por una actividad de
implementación ni duplicar el gate global de 6.2.

La base Release documentada es `deec3aa8d2516ae19e335a5af77e6c38f8a854d8`;
la del workflow histórico es `40b3f1be2d3f97028b741750ed61ab72f501a555`.
No son intercambiables. Los inventarios de candidatos dependen de base y head;
ningún listado previo es una campaña ejecutada ni se extrapola a otro head.
Los presupuestos
documentales 30/180 y operativos 360/600 siguen distinguidos en la receta seca;
su suficiencia requiere evidencia de ejecución. No se cambia ningún presupuesto.
C87 identifica dependencias del planificador, no el gate global de Release 1.0.

## Candidatos nativos: Windows, Linux y decisión AVX2

Las 14 casillas de `native-f32-isa-backends` permanecen abiertas en esta
conciliación sobre master. Las 1.1/1.2/2.1 marcadas en candidatos privados no se
copian como implementación publicada. Linux v3 y Windows v4/packing son revisiones
distintas: un resultado del primero no valida automáticamente el segundo.

Estos identificadores estables describen informes comunicados el 2026-11-28,
corroborados con el historial de resultados, sin publicar identificadores de
conversación, rutas privadas ni ubicaciones de archivos privados. Los recibos
originales del worker Windows no pudieron inspeccionarse íntegramente en esta
revisión; se conserva ese límite y no se inventan hashes de binarios o fuentes.

| ID de evidencia | Resultado comunicado y alcance |
| --- | --- |
| WIN-V4 | Windows nativo, Ryzen 7 PRO 5750G, Stories15M F32, un hilo: AVX v4 104,5–104,6 frente a escalar 89,0–89,1 tokens/s; 97 pruebas nativas v3/v4 comunicadas y logits/IDs iguales. `auto` permanece escalar. |
| WIN-GATHER | AVX2 gather ejecutado y descartado localmente: 39,6–39,7 frente a AVX 104,5–104,7 tokens/s, con paridad bit a bit comunicada. Se cierra esa decisión experimental sin promover el diseño; no se descarta AVX2 en general ni se llama AVX2 al diseño rápido que sólo usa AVX. |
| WIN-PACK | Packing optativo con presupuesto explícito de 36.864.000 bytes: 100 pruebas comunicadas, logits/IDs iguales, 150–151 frente a 95–97 tokens/s en el control comunicado. Es payload adicional, no una medición de RSS máximo. |
| WIN-CROSS | Incluyendo arranque/preparación: 1 token, 3–4 ms más lento; 4 tokens, unos 7 ms más rápido; 16 tokens, 52–54 ms más rápido. No se midieron 2/3 tokens. No es una ventaja universal ni un resultado Linux/ELF. |

| ID de tarea | Estado reconciliado y condición para cerrar |
| --- | --- |
| native-f32-isa-backends/1.1 | Verificada local; integrar el informe de consumidores y su trazabilidad con la entrega correspondiente. |
| native-f32-isa-backends/1.2 | Verificada local; integrar la referencia/oráculos efectivos y conservar sus controles sensibles. |
| native-f32-isa-backends/2.1 | Verificada local; integrar la variante exacta, distinguiendo v3, v4 y packing. |
| native-f32-isa-backends/2.2 | Decisión local cerrada sin promover gather; falta incorporar evidencia revisable de identidad, instrucciones y selección para cerrar la tarea del repositorio. No se implementa otra variante para inflar el conteo. |
| native-f32-isa-backends/2.3 | Parcial; validar selección visible/fallback/errores en CLI y ELF Linux del candidato exacto y declarar plataformas no ejecutadas. |
| native-f32-isa-backends/2.4 | Parcial; conservar signed zero, cancelación, subnormales, cast, colas y negativos. Falta cerrar la matriz comparable del candidato integrado; no usar sólo tokens iguales. |
| native-f32-isa-backends/3.1 | Parcial WIN-PACK; completar preparación, copias, asignaciones, pico de memoria, reutilización e invalidación y fallos de presupuesto con procedencia. |
| native-f32-isa-backends/3.2 | Parcial WIN-V4/WIN-PACK/WIN-CROSS; completar las métricas y condiciones fijadas, TTFT/prefill/decode, dispersión/memoria e ISA efectiva del candidato exacto. |
| native-f32-isa-backends/3.3 | Parcial; logits/IDs Windows no acreditan aquí KV completo ni una nueva comparación stock llama.cpp. Mantener esa referencia separada. |
| native-f32-isa-backends/3.4 | Pendiente; política y experimento FMA separado, sin heredar equivalencia ni inventar un contraejemplo fuera del dominio. |
| native-f32-isa-backends/4.1 | Diferida hasta bml-inference-pipeline/8.6; el rendimiento nativo no acredita BMLX. |
| native-f32-isa-backends/5.1 | Pendiente del cierre de implementación: estructura y trazabilidad integral, no sólo validar esta conciliación. |
| native-f32-isa-backends/5.2 | Bloqueada para cierre: gate canónico base/candidato y plataformas aplicables. La suite Windows de 100 pruebas y el control focalizado Linux no son ese gate. |
| native-f32-isa-backends/5.3 | Pendiente de integración/publicación revisable; no hay push de implementación ni de documentación en este paso. |

## Hoja de ruta: parte uno y fragmentos ELF autónomos

**La parte uno permanece abierta.** Su único criterio final de cierre es
completar y verificar [bml-1-0-release/4.6](../openspec/changes/bml-1-0-release/tasks.md),
añadida después de los hitos de recursos/AOT 4.1–4.5. No se añade una segunda
casilla para «parte uno» ni se convierten los prerrequisitos en su cierre.
El [contrato de artefactos autónomos](../openspec/changes/bml-1-0-release/specs/standalone-artifact/spec.md)
separa los nuevos ELF nativos por subcapa del transporte simbólico de 8.5.8.

Los siguientes resultados locales verificados fueron comunicados en la revisión
del 2026-11-28; se registran con su alcance, sin repetirlos en esta actualización
documental ni atribuirles hashes de ejecutables no inspeccionados aquí:

| Evidencia estable | Resultado y límite |
| --- | --- |
| P1-MONO-ELF | El ELF nativo monolítico de Stories15M F32 genera tokens sin consumir un GGUF externo durante la ejecución. Acredita el prerrequisito autónomo monolítico; no genera ELF independientes por subcapa. |
| P1-HYBRID-1HOST | Un prototipo híbrido distinto fragmenta el cálculo con dos procesos workers en un único host; 128.000 logits coinciden bit a bit frente a un proceso. No son fragmentos ELF compilados independientes ni evidencia de dos hosts. |
| P1-PACKED-AVX-LINUX | La validación final del candidato con AVX empaquetado en Linux/ELF mediante Docker sigue pendiente; el equipo Ryzen estaba desconectado al recibirse esta actualización. Los resultados Windows no sustituyen esa validación. |

Stories15M F32 conserva como identidad pública de referencia 98.357.920 bytes,
SHA-256 `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`.
Para 4.6 se deberá fijar además la revisión del GGUF real usado, del compilador
y de cada ELF, junto con los prompts, BOS/plantilla, contexto, IDs y criterio
de logits previo. No se afirma que ya exista el nuevo conjunto de artefactos.

La aceptación de 4.6 comprende ELF por cada fragmento de subcapa seleccionado,
recursos propios sin GGUF externo en la ruta de ejecución, intercambio de
activaciones/resultados y agregación determinista. Primero exige generación
completa local y paridad acordada de IDs/logits prefill/decode frente al control
monolítico; después, la misma verificación en hosts distintos. Hasta acreditar
ambas etapas permanecen abiertas 4.6 y la parte uno. No se declara Lambda ni
se cierra 8.5.8 o 8.6 por esos resultados nativos.

La validación CLI/ELF local y el benchmark Docker/AVX pueden continuar
independientemente de 4.6. Pasarlos no cierra la parte uno. Los dos prerrequisitos
locales anteriores están verificados; si el alcance de la primera parte incluye
AVX empaquetado en Linux, ese alcance tampoco está cerrado mientras falte su
validación. Este cambio planifica la tarea final; no implementa fragmentos,
contacta hosts, ejecuta benchmarks ni autoriza despliegues.

## Validación de esta conciliación y su actualización

El gate aplicable es documental: OpenSpec estricto, revisión del diff, generación
estática del tablero antes/después y comprobación de conteos, estados e IDs sin
duplicados. La base falla OpenSpec estricto por la longitud del requisito
«Hot loop RPN confinado a 32 KB de instrucciones». Se reubican sus mismas
cláusulas de procedencia, bytecode y alcance en un escenario del mismo requisito;
se conservan 32.768 bytes, siete repeticiones, el umbral y el rechazo de contadores
incompletos, sin crear otra obligación ni relajar su aceptación.
No se modifica el generador, runtime, tests de inferencia, workflows
ni criterios numéricos. No se ha ejecutado el gate canónico Rust/Python de
`scripts/check.sh`; no se presenta como aprobado. Los resultados de esta revisión
se conservan aparte con hashes y el parche revisable.
