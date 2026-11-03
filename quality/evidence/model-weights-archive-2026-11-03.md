# Archivo de GGUF → BMLGGUF — 2026-11-03

## Dictamen y alcance

Los ocho requisitos de `gguf-to-bml-weights` están acreditados por las fuentes,
pruebas y evidencias publicadas. La auditoría de cierre volvió a leer los
contratos y verificó hashes y ascendencia Git; no repitió builds, mediciones ni
campañas de mutación. Los informes históricos permanecen intactos.

La biblioteca publicada en `858d8fc` coincide byte a byte con los 17 archivos
actuales de `crates/model-weights`. La documentación y evidencia de entrega se
publicaron en `1ef5f5d`. Ambos commits son antecesores de la referencia local
`origin/master` comprobada durante esta auditoría: `8a3479084db1ada2cd9ebf8110520a103a5ba014`.
La comprobación remota de publicación consta en
`quality/evidence/publication-spec-closure-2026-11-03.md`; esta auditoría no hizo
un nuevo push ni presenta la lectura de una referencia local como consulta remota.

La inferencia autónoma exigida es la del fixture admitido. El escenario normativo
«Campaña real sin éxito completo» permite registrar fallos acotados. El archivo
no acredita inferencia real v4, compresión o aceleración general, mutación global,
ejecución ARM/NEON, contadores PMU/L1 ni estrés cloud.

## Requisito → evidencia

| Requisito canónico | Implementación y pruebas | Evidencia registrada |
|---|---|---|
| Conversión explícita y autónoma | `convert::from_gguf`; `conversion::independent_f32_f16_and_quantized_blocks_convert_and_reload_without_source`; CLI `converts_and_inspects_without_source_with_explicit_budgets` | `model-weights-conversion.json`, `model-weights-cli.json` |
| Valor exacto y política de entrada | Comparación racional de certificados; `signed_subnormal_ieee_targets_are_stored_exactly_without_float_evaluation`, `half_and_quantized_nonfinite_weights_reject_at_decoded_index` y fallos localizados/transaccionales | `model-weights-contract-review.json` |
| Formato acotado e integridad | `WeightReader` valida header256/index80, rangos, presupuestos y hashes antes de materializar; truncaciones, trailer, índices rehasheados, metadata y verificación parcial explícita | `model-weights-reader.json`, `bmlgguf-v1-format-review.json` |
| Orden tensorial | `distinct_rectangular_values_keep_flat_order_across_row_crossing_blocks`; `independent_fixture_loads_blocks_in_ggml_order_and_verifies_all` | `model-weights-contract-review.json`, `model-weights-reader.json` |
| Presupuestos y publicación transaccional | `WeightWriter` exige `CertifiedBml`, reproduce derivaciones y realiza readback antes de publicar sin reemplazo; pruebas de límites inclusivos, caché, replay, carrera y limpieza tras fallo | `model-weights-writer.json`, `model-weights-budgets.json`, `model-weights-mutation-v9.json` |
| Carga e inferencia preparadas | `prepared_blocks_preserve_trees_and_project_against_integer_oracle`, `structurally_valid_unjustified_weight_loads_but_execution_has_no_float_fallback`; `converted_autonomous_model_tokenizer_logits_and_generation_match_independent_oracle` elimina el GGUF antes de comprobar tokenizer/logits/tokens | `model-weights-projection.json`, `model-weights-causal.json`, `model-weights-run.json` |
| Evidencia por fase | Fixture con procesos separados de conversión/carga/ejecución, fuente retirada y tokens `[1,0]`; dos intentos reales con diagnósticos localizados y fases no ejecutadas declaradas | `model-weights-fixture-phases.json`, `model-weights-real-campaign.json`, repetición funcional `delivery-documentation.json` |
| API independiente y errores tipados | Núcleo sin compiler/runtime/inference; `gguf-import` opcional; `Error::source` preserva causas; reader parcial y `verify_all` diferenciados; contratos de downcast y ubicación | `model-weights-foundation.json`, `model-weights-reader.json`, `delivery-model-weights-library.json` |

Los nombres de informes de esta tabla se resuelven dentro de `quality/evidence/`.
Las pruebas de biblioteca pertenecen a `crates/model-weights/tests/`; las de
proyección/causal, a `crates/runtime/tests/`; las CLI, a `crates/cli/tests/`.

## Controles y límites de la evidencia

- Campaña v9 de **bml-model-weights**: 443 candidatos, 391 detectados,
  52 inviables, 0 supervivientes y 0 timeouts; terminó con exit 0.
  No es una campaña global del workspace.
- Snapshot previo de v9: gate 703 pruebas Rust, perfiles GGUF mínimo/all-features
  50/57; cobertura workspace 18.614/21.390 líneas (87,02 %) y
  1.721/2.157 funciones (79,79 %). Los hashes de los informes coinciden.
- Snapshot seleccionado de entrega hasta `bb1f9ff`: gate 630 pruebas Rust y
  63 Python, auditoría histórica sin discrepancias, trazabilidad 28/28 y cero
  diagnósticos nuevos. Sus controles se describen en
  `delivery-server-fallible.json`; no se confunden con el snapshot v9.
- Fixture por fases: 12 tensores, 2.188 pesos; fuente 16.000 bytes y salida
  13.029 bytes. Conversión/carga/ejecución terminaron con exit 0; se eliminó
  la fuente antes de cargar y se generaron `[1,0]`. Muestras únicas; RSS del
  proceso de ejecución incluye la carga previa. No son medidas de rendimiento
  estadístico ni evidencia de compresión de modelos reales.
- Stories15M F32: fallo de derivación en `token_embd.weight[32203]`.
  Stories15M Q4_0: fallo del presupuesto de replay en `output.weight[75234]`.
  Fuentes intactas, ningún destino publicado ni temporales restantes;
  carga/inferencia no ejecutadas. Ambos logs coinciden con sus hashes.

## Hashes comprobados de evidencia ejecutada

| Archivo | SHA-256 |
|---|---|
| `target/quality/model-weights-v9-gate.log` | `3a1f9a6a0a5feccd8a82d0f9576216ea78fdb0c466f36607e0bb67a05353d3a1` |
| `target/quality/model-weights-v9-gguf-minimal.log` | `9be66ba4e75fa2ba0f7b83989ccd983123466ade10f0336271423a8556c6b8c4` |
| `target/quality/model-weights-v9-gguf-all.log` | `c7bc1b26ec84fb7f1d56ea282067d11bbd18b9c663494c6145ca57cb7ce63539` |
| `target/model-weights-mutations-v9/mutants.out/outcomes.json` | `47e1c7417767c94b2c805025fb3b4466cef561705d2a6ba151243f27a1bdb09c` |
| `target/quality/model-weights-v9-workspace-coverage.json` | `e1b79a3731466bb07434e8450730423a8cc05dc8e29a65c4befdbe3d3820f1a5` |
| `target/quality/prepared-fixture-phases/report.json` | `13999faa1046246b63fdf78d24ca4a1e001a462ac3af119ced630860222223f7` |
| `target/quality/real-weight-conversion/stories15M.log` | `9bd2257093d9bdf8407884960068adfe35b3ef698c83241695b96e19d5da1921` |
| `target/quality/real-weight-conversion/stories15M-q4.log` | `58f23b5cab7f4615432e83dde0e76566095e6fa6b01118f78a3e3f8991874b3b` |

## Hashes de fuentes y pruebas publicadas

Los siguientes archivos coinciden con `858d8fc`; el SHA-256 corresponde al
contenido revisado, no a un nuevo resultado de ejecución.

| Archivo | SHA-256 |
|---|---|
| `crates/model-weights/Cargo.toml` | `96bf42acaa09b315ee768ad35815ad4978a3063b85e3800c62e5474ae58e1ee8` |
| `crates/model-weights/src/convert.rs` | `2b0f299f2b0330eb4e6cd9bc189492c3db9dbeee48fe06eaa11763cd36ced000` |
| `crates/model-weights/src/error.rs` | `87c73bbab402ab9518397c232c5756bf02ee8aa58c8022bf6e41784ccdbf576f` |
| `crates/model-weights/src/gguf_preflight.rs` | `7928cef4ae71f2ebde64e640720f08e31f92394ece2bfe0c35bc4068e0a99451` |
| `crates/model-weights/src/lib.rs` | `ba90fc75febfa1a3098055a10df36bfb15a5805e2da399bd43a94bbf6a321a9b` |
| `crates/model-weights/src/limits.rs` | `80f67c8fb557fe64369ef3745ca2a7b25fee347dbefa092ff4751d126702742f` |
| `crates/model-weights/src/memory.rs` | `a428325c47146bc665b05d841074c88f38845f0b49a9c5bfe2ef35fb8889bd2e` |
| `crates/model-weights/src/metadata.rs` | `115b3c338c9987974742d91adaca17e982f826fbff849874c6d22effc35b564b` |
| `crates/model-weights/src/metadata_encode.rs` | `97d0ce77f3adadf775a92e0436f334bb2054b6b261dfd5f5152df1d25f3d6f79` |
| `crates/model-weights/src/reader.rs` | `610a2f12c9052e3265e8f2a1e0dd198c6984741d70c2b7d3b8b4fb8bfb750756` |
| `crates/model-weights/src/types.rs` | `6405d7c7492e43f0eefb9e239863a01583423dd85449881ed35f5201dcf130e3` |
| `crates/model-weights/src/writer.rs` | `c225b2bd80fbc7e503dfbabfd7c0cb26ba69ad3dca718986541186387ca77168` |
| `crates/model-weights/tests/contracts.rs` | `aadfc3f1fe250f8f560b3586976c3dd715b2df639c5df7489d5f65bfe26c8681` |
| `crates/model-weights/tests/conversion.rs` | `961d1312418dd0ed0a7f73ff1f5376b4499e5831a140b777e88e9d77e7252845` |
| `crates/model-weights/tests/reader.rs` | `770010e4c9fcfc68ccf9150c710a106242e9ad1a23731c51754906ac7a4bfb6e` |
| `crates/model-weights/tests/writer.rs` | `a8c8a20c0fcfa7ac6e5eef8fdb9b5bddce29a55ee959e48f44e1ba880f1d4b8c` |
| `crates/model-weights/tests/writer_collision.rs` | `1065acf4413c274a5f1b35e707b8a46af5e0a954ec3f0e61d388c84bee8562c5` |

## Archivo y validación documental

Los anexos normativos se conservan en
`openspec/specs/bml-model-weights/api.md` y `format-v1.md`; la spec canónica
referencia esas rutas estables. La propuesta, diseño, tareas y entrega se
conservan en `openspec/changes/archive/2026-11-03-gguf-to-bml-weights/`.
Las referencias contenidas en informes históricos mantienen sus rutas originales.

Comandos de cierre, con `OPENSPEC_TELEMETRY=0` y
`TMPDIR=/home/dev/Work/bml/target/tool-install-tmp`:

```sh
npx --yes --package @fission-ai/openspec@1.13.0 openspec validate gguf-to-bml-weights --strict
npx --yes --package @fission-ai/openspec@1.13.0 openspec archive gguf-to-bml-weights --yes
npx --yes --package @fission-ai/openspec@1.13.0 openspec validate --all --strict
```

Resultados del cierre:

- Validación estricta previa: exit 0, `Change 'gguf-to-bml-weights' is valid`.
- Archivo: exit 0, tareas completas; capacidad `bml-model-weights` creada con
  ocho requisitos añadidos, sin modificaciones ni eliminaciones de requisitos.
  Destino: `openspec/changes/archive/2026-11-03-gguf-to-bml-weights/`.
- Validación global posterior: exit 0, **11 elementos aprobados, 0 fallidos**.
- Comprobación documental: 22 tareas cerradas, ocho requisitos canónicos,
  propósito definido, anexos idénticos entre archivo y ubicación canónica y
  todos los enlaces Markdown relativos internos resueltos.
- Los enlaces vigentes de `docs/BML_1_0.md` y del diseño de
  `bml-inference-pipeline` se actualizaron durante la integración. Las rutas
  históricas de los informes anteriores se conservaron.

SHA-256 de los anexos normativos finales, iguales en ambas ubicaciones:

| Anexo | SHA-256 |
|---|---|
| `api.md` | `7f6fb2ee4f736a3a549a0b9994a0ef9b49765c9d868a1816b80de3729658947f` |
| `format-v1.md` | `57037d41f2ae372600be79670c6564446c7e2f1eec05d72a63016c3f1cedb038` |

El commit y publicación de este registro de archivo corresponden a la
integración documental posterior; no se presentan como parte del push histórico.
