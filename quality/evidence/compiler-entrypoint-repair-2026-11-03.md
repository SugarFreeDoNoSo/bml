# Entradas públicas de compilación GGUF

Base de la entrega: `97ddb16842138605837f2f6f96dba619b399d2ec`.

## Implementación y contrato

`compile_gguf` y `compile_gguf_fast` ahora construyen el plan AOT causal completo
de `symbolic_artifact`. Se retiraron `build_transformer_dag` y la identidad escalar
que antes devolvía la entrada rápida. `compile_gguf_with_limits` permite contexto
y presupuestos explícitos. El umbral de fragmentación procede del hardware.

El resultado de esas APIs experimentales cambia a `CompiledArtifact`: encapsula
programas, manifiesto y hashes, ofrece `manifest()` y publica con `publish()`.
La compilación y la publicación usan el mismo código que `compile_directory`.
El formato escalar histórico conserva sus lectores/escritores. Migración de
llamadores y ejemplo: `docs/SYMBOLIC_COMPILATION.md`.

C72 registra dos contratos públicos en `compiler_entrypoint_contracts`:

* Ambas entradas publican todos los kernels BML de una capa GQA con RoPE y MLP
  firmado no nulo. Prefill y decode incremental concuerdan con el oráculo
  independiente, tolerancia absoluta `2e-7`; la carga tiene `max_nodes=0`.
* Contexto y umbrales se respetan; presupuestos insuficientes fallan; la
  publicación rechaza destinos no vacíos sin sobrescribir el manifiesto.

El oráculo existente se extrajo a `crates/runtime/tests/support/symbolic_oracle.rs`.
Su cuerpo es idéntico al de la base después de normalizar el nombre de la función:
SHA-256 `ca8120b8194b55b8352c7a87f548d2ba04c42260c8bc0ddd110e98e18762ca87`.
Comparación reproducida en `target/quality/compiler-entrypoint-oracle-provenance.json`.
No usa funciones numéricas del compilador/runtime.

## Controles negativos

Snapshot aislado: `target/delivery-snapshots/compiler-entrypoint-repair`.
Ejecutor: `target/quality/verify_compiler_entrypoints.py`. Build inicialmente nuevo;
las variantes se reconstruyen en la misma ruta. Las seis alteraciones independientes
fallan en aserciones de los contratos, con exit 101:

1. Reducir implícitamente el contexto del modelo a uno.
2. Ignorar el umbral del hardware y usar 32768.
3. Reducir a uno el contexto de la entrada rápida.
4. Ignorar el contexto explícito solicitado.
5. Sustituir los presupuestos explícitos por defaults.
6. Omitir la publicación del manifiesto.

Original y restaurado: 2/2 contratos aprobados. El primer intento del ejecutor
se detuvo por un selector ambiguo antes de aplicar una mutación; se conserva el
registro inicial. La versión v2 acota el selector a la llamada correspondiente.
Registros y hashes por ejecución: `target/quality/compiler-entrypoint-verification-v2.json`.
Las fuentes originales de esos replays están conservadas, con sus hashes
verificados, en `target/quality/compiler-entrypoint-replay-originals/`.

El primer gate completo encontró `/tmp` (tmpfs de 2 GiB) lleno: dos contratos de
bytes fallaron con `ENOSPC`. Se conservan los logs de esos intentos y se repite
con `TMPDIR=$REPO/target/tool-install-tmp`, sin cambiar código ni aserciones.
El siguiente intento aprobó las pruebas pero encontró tres avisos `err_expect`
al incorporar `Debug` al nuevo tipo de resultado. Se cambiaron esas tres
aserciones a `expect_err`; no se modificó el baseline de Clippy.

## Gate del árbol completo

`./scripts/check.sh origin/master` termina con exit **0**: **728 pruebas Rust,
82 Python, trazabilidad 72/72, cero discrepancias y cero diagnósticos nuevos**
(63 actuales, baseline 192). Formatting aprobado. Las identidades de fuentes
se verificaron antes/después. Log
`target/quality/compiler-entrypoint-worktree-gate-v3.log`, SHA-256
`5fcfbc515892a796aa942e94988440abe453edaba49f2142bc760f58013d912e`.

GGUF sin features y con todas las features, ambos con `--locked`: **50 y 57
pruebas aprobadas**. OpenSpec estricto: **9 aprobados, 0 fallidos**, validación
estructural. La campaña global y las pruebas del snapshot aislado tienen
registros separados.

## Gate de la entrega seleccionada

`./scripts/check.sh HEAD` en el snapshot aislado termina con exit **0**:
**658 pruebas Rust, 63 Python, trazabilidad 29/29, cero discrepancias y cero
diagnósticos nuevos** (67 actuales, baseline 192). Formatting aprobado. Los diez
archivos de implementación, contratos, documentación de API y trazabilidad del
índice coinciden por SHA-256 con el inventario probado; los otros dos archivos
seleccionados registran la evidencia y los requisitos del entorno.

Registro aprobado: `target/quality/compiler-entrypoint-delivery-final.json`.
Log: `target/quality/compiler-entrypoint-v4-delivery-gate.log`, SHA-256
`71b7b7b98b6b53a5d1673ea69f1d5a230ce6d6e2fe145b99bab7c71c9e898d76`.
El archivo `compiler-entrypoint-verification-final.json` conserva el intento
anterior fallido por Clippy y no es este resultado aprobado.

## Diagnóstico de modelos reales

Probe y lockfile conservados en `target/quality/symbolic-model-probe/`. Se usa
contexto uno, `--locked`, sin fallback a aritmética nativa. La comparación nativa
es una ejecución separada. No se publican mediciones de rendimiento.

| Modelo | SHA-256 | Resultado observado |
| --- | --- | --- |
| Stories15M F32 | 61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a | Default rechaza el presupuesto por programa. Con límites explícitos ampliados compila diez kernels; el primer BOS falla en RMSNorm por rango/dominio. La ruta nativa devuelve 32000 logits finitos. |
| TinyLlama Q4_0 | da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556 | Default rechaza el presupuesto de nodos. La llamada con límites ampliados no termina dentro de 180 s bajo carga concurrente; no hay resultado de inferencia acreditado. |

Los límites ampliados fueron: 2.000.000 nodos; 4.000.000 opcodes totales y por
programa; 4096 programas; fragmentos de 32768 bytes. La carga usa cero nodos de
construcción y límites de 4.000.000 opcodes, pasos y valores. No se modificaron
los defaults ni los techos del formato.

Registros: `target/quality/compiler-entrypoint-model-diagnostics.json` y
`target/quality/compiler-entrypoint-tinyllama-diagnostic-v2.json`. El primer intento
de TinyLlama agotó el plazo esperando el lock de Cargo; se conservó y se repitió
invocando directamente el binario ya construido. Solo esta segunda ejecución
acredita el rechazo del presupuesto original y el timeout posterior.

8.6 permanece abierta: estos resultados identifican trabajo de rango numérico,
proyecciones desenrolladas y planes de prefijos. La carencia ARM/PMU se documenta
aparte en `spec-environment-blockers-2026-11-03.json`.

La campaña global iniciada sobre `global-quality-restart-6a3e6cc` conserva su
snapshot y no incluye esta entrega posterior. Estos controles negativos no son
una campaña global ni sustituyen su gate.

## Identidades de los registros

| Archivo bajo target/quality/ | SHA-256 |
| --- | --- |
| compiler-entrypoint-worktree-gate-v3.json | 644dc64653f57a2fc52125eb20ffb1a56951c7505a96e12a5804eae5445cbda3 |
| compiler-entrypoint-delivery-final.json | 8256e1f05dc04d5388cdecdd20c122789444a3d5148f77e19154a12f65f46a1c |
| compiler-entrypoint-verification-v2.json | d07f91aa308f0302cb67a28c1956b5f6ee8ffa045477cfb4733a14af5943a549 |
| compiler-entrypoint-model-diagnostics.json | 2c5f2e0fec0f938e9c6b57191878052ae62388569ee4b9c7fef5b425ac021f35 |
| compiler-entrypoint-tinyllama-diagnostic-v2.json | 17a970569591e4eae2ef5ff424b5af3c8a4259273b08b2d9f5631ebabfd753c2 |
| compiler-entrypoint-gguf-features.json | 1b140ae77c56d4f267bd7b5392f86ce3e64610636e43d7f95ffe46053f6ec060 |
| compiler-entrypoint-oracle-provenance.json | 62c34e5325e0e9d49577d480eb1fd5fba6888e54b963b4bda90dfe2c9135b54e |
| compiler-entrypoint-openspec.log | 41509253a9e7f8de902e8da4d67e858bd27224c1e84792c9746df72ce39443e0 |
