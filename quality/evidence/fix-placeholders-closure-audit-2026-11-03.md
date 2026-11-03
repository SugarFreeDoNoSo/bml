# Auditoría de cierre de `fix-placeholders` — 2026-11-03

**Resultado: no archivar.** La revisión requisito por requisito encontró dos
obligaciones marcadas como completas que no se cumplen en las APIs nombradas.
Las rutas causales modernas y sus gates aprobados no sustituyen esa evidencia.
Se reabren únicamente 1.1 y 6.2; no se cambia la implementación ni se debilita la
spec para coincidir con el código existente.

## Defectos reproducidos

1. **Matmul de `op_fragments`, tarea 1.1.**
   `crates/compiler/src/op_fragments.rs:68` publica `compile_matmul_fragment`
   como `y = W*x`, pero las instrucciones de producto y acumulación en 88–89
   son aplicaciones literales de `BML`, no multiplicación y suma. El `Zero;
   Dup` del comienzo del cuerpo exterior duplica el acumulador, no el contador
   de fila descrito por el comentario. Ejecutar el fragmento público con el
   `HotLoop` y `ResultBuffer` actuales produce los resultados incorrectos de
   abajo, sin panic. Las pruebas locales `matmul_fragment_structure` y
   `fragment_size_under_32kb` comprueban estructura/tamaño, no esos resultados.
   La tarea 1.4 conserva su marca porque existen otros contratos de matrices;
   su referencia a TinyLlama no demuestra la corrección de esta API particular.
2. **Temperaturas positivas pequeñas, tarea 6.2.**
   `crates/compiler/src/sampler.rs:30` convierte toda temperatura `<1e-9` en
   greedy. Para logits `[0,1e-12]` y temperatura `1e-12`, las probabilidades
   requeridas son `[0.2689414213699951,0.7310585786300049]`, las mismas que para
   `[0,1]` y temperatura `1`. La producción devuelve siempre ID 1 por ese
   branch; el defecto se deduce del flujo, no de una inferencia estadística.
   Un control aislado sustituye solamente la fuente aleatoria por los mismos
   cien uniformes `seed/100`: hay 27 diferencias entre las escalas; con
   `pick=0.1` la escala normal devuelve 0 y la pequeña devuelve 1. `_seed` se
   ignora en producción, por lo que no se afirma que iguales semillas reales
   deban producir muestras idénticas. El control cambia solo la fuente de
   azar en una copia temporal; no modifica el sampler productivo.

La ejecución del probe terminó con **exit 0**. Ese código registra el fallo de
las operaciones; su salida 0 no significa que sus resultados cumplan el contrato:

```text
matmul width=1 rows=1 inputs=[1.0] weights=[2.0] expected=[2.0] actual=[0.0] panic=false
matmul width=2 rows=2 inputs=[1.0, 2.0] weights=[3.0, 4.0, 5.0, 6.0] expected=[11.0, 17.0] actual=[0.0, NaN] panic=false
controlled_equal_uniforms mismatches=27 seeds=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26] pick_0_1_normal=Some(0) pick_0_1_scaled=Some(1)
positive_temperature logits=[0.0, 1e-12] temperature=0.000000000001 expected_probabilities=[0.2689414213699951, 0.7310585786300049] zero_count=0 one_count=100
```

## Mapa de requisitos y evidencia actual

| Requisito / escenarios | Implementación y prueba primaria | Resultado para este cierre |
| --- | --- | --- |
| Matmul completo: 2x2 y acumulador | `InferenceCompiler::project_rows` recorre cada fila y `project` acumula todos los productos; `inference_contracts::tied_embedding_projection_matches_independent_rectangular_dot_products`, `rpn_causal::rpn_signed_rectangular_projection_uses_contiguous_rows_and_checked_ranges`, C60/C63/C80 | Las rutas nativa, RPN y simbólica tienen evidencia propia. **La API explícita de 1.1 falla**, como muestra el probe; no puede darse por sustituida sin cumplir su contrato. |
| Traducción de todos los pesos, 1.2–1.3 | `build_transformer_dag`/`compile_gguf_fast` publican el plan completo; C72 `compiler_entrypoint_contracts`; offsets y dimensiones validados en `gguf_compiler` | Acreditado en su alcance, con presupuestos y límites numéricos explícitos. No se atribuye a los helpers de `op_fragments`. |
| RoPE completo: posición 0 y todos los pares | `gguf_compiler.rs::apply_rope_inplace` anida todas las cabezas y `head_dim/2` pares; `causal_contracts::rotary_pairs_use_distinct_frequencies_and_absolute_cached_positions`, tests `eml::rope_apply_identity_at_zero` y C60 | La implementación cubre todos los pares; la prueba causal verifica frecuencias distintas y posiciones absolutas. El registro TinyLlama ejecuta un modelo de 2048 dimensiones. No se exige aquí cerrar el rango de toda otra API simbólica. |
| SwiGLU: cero, uno y ±ln(2) | `swiglu_scalar`; C49 `swiglu_matches_closed_form_values_without_a_scaled_sigmoid`; causal oracle con MLP no nulo C52/C64/C81 | La ruta nativa requerida usa `gate*sigmoid(gate)*up`, sin 1.7. AST/EML conserva su contrato separado, como dicen las tareas. |
| Normalización final: pesos/epsilon y metadatos inválidos | C49 `final_rmsnorm_uses_metadata_epsilon_and_weights_before_both_output_paths` y `invalid_final_norm_or_epsilon_is_rejected_before_execution`; `compute_logits` y validaciones de carga | Acreditado en ambas rutas nativas del contrato. |
| `VarIndexed` sin buffer, tareas 4.1–4.3 | C44 `indexed_reads_without_buffer_are_nan_at_every_supported_loop_depth`, RPN/fragment y runtime con buffer real | Acreditado: NaN sin buffer frente a 17 almacenado, a profundidades 0/1/2. El diseño D6 que aún dice cero está obsoleto. |
| Tokenización y bytes/UTF-8 incremental | C50 `tokenizer_contracts`, C51 `token_output`, C58/C67 HTTP; `legacy-bpe-output.json` y tokenizer TinyLlama fijado | BPE/scoring/BOS/fallback por bytes, errores de metadata y emisión UTF-8 están cubiertos. La evidencia de tokenizer no se presenta como paridad completa del modelo. |
| Sampling greedy y temperatura | `sampler::argmax_basic`, `sample_temp_zero_is_greedy`, código softmax para temperatura 0.8; `sample_temp_returns_valid_id` solo prueba rango del ID | Greedy y el caso ordinario funcionan. **La obligación general temp>0 falla** por el umbral, y la prueba de ID válido no lo detecta. |
| Generación causal: prefill/decode, GQA, límites y reintento | C52 `causal_contracts`, C53 RPN, C64 simbólico; preservación de prefijo y posición confirmada comprobada ante errores de proyección/aritmética | Acreditado para las rutas y perfiles explícitos probados; no es una promesa de evaluabilidad universal de las identidades BML. |
| CLI y servidor generan texto del modelo | C51/C57 ejecutables CLI; C58/C67 servidor HTTP/SSE real con UTF-8/EOS/DONE y bytecode deliberadamente modificado; `tinyllama-legacy-mapped.json`, `tinyllama-compiled-causal.json` | La CLI `run` y el servidor activo ejecutan el modelo, sin eco ni tokens inventados. Los diez tokens TinyLlama se registraron con fuentes/binarios identificados. |
| Artefacto causal persistido: relocation, hashes y preflight | C54 decoding, C56 `causal_artifact_contracts`; mutación del programa guardado cambia el resultado, contextos y budgets se validan, cargas relocadas verifican SHA-256 | Acreditado. El perfil causal nativo declara las operaciones compartidas; esa selección explícita no constituye un fallback oculto. |
| Perfil simbólico AOT, servidor y vectores compartidos | C65/C66/C67 AOT y HTTP, C69/C70/C71 vectorización y compatibilidad v1/v2; C81/C84 añaden v4 sin reemplazar contratos anteriores | Programas persistidos sin recompilación, planes/perfiles/contextos validados y salidas incompletas rechazadas. La traducción completa exigida por `gguf-compiler` conserva su propio seguimiento. |
| Comparación con llama.cpp, tareas 8.1–8.4 / G8 | `tinyllama-llama-cli.json` y `tinyllama-legacy-mapped.json`: mismo modelo SHA, prompt/IDs, greedy, contexto y KV F32; comparación del mismo prefijo de nueve IDs | Comparación realizada y divergencia documentada: texto «Step 3: Add» frente a «Step 2: Create»; error máximo de logits 0.2150369823. Este cambio no exige igualdad de logits ni atribuye una causa no aislada. |
| Cierre 9.1–9.3 | Gate reciente `full-bounded-2026-11-03.json`: 778 Rust, 82 Python, 84/84 contratos, cero discrepancias y cero Clippy nuevo; publicación registrada | Los comandos y publicación están acreditados, pero sus pruebas no cubrían los dos defectos descubiertos. Un gate verde no basta para archivar un requisito contradicho. |

## Límites del alcance y documentación que debe reconciliarse

- El proposal conserva `sigmoid(1.7*gate)`, mientras la spec y tarea 3.1
  corrigen expresamente a SwiGLU estándar. Design D6 dice `VarIndexed=0` sin
  buffer; tareas 4.1–4.3 y C44 requieren NaN. Son decisiones históricas
  superadas que deben quedar identificadas al cerrar, no nuevas obligaciones
  de restaurar comportamiento incorrecto.
- `op_fragments` también conserva helpers públicos declaradamente simplificados
  para RMS/atención/MLP, incluido `_down_base` sin uso (`op_fragments.rs:234`).
  Esto impide una afirmación global de «eliminar todos los placeholders» de G1.
  Las tareas 3.x están expresamente acotadas al `InferenceCompiler` nativo y
  no se reabren por confundir esos contratos. La corrección de esas APIs debe
  mantener el objetivo original; esta auditoría no las reclasifica como
  correctas ni reduce el alcance para permitir un archivo prematuro.
- La CLI pública `distribute` sigue accesible desde `Command::Distribute` y
  construye embeddings `tid*0.001/(i+1)` (`main.rs:893`), tratando después el
  hidden como logits (`main.rs:934`). Se registra como placeholder real para
  la revisión de G1/distribución. El escenario CLI de esta spec nombra `run`,
  así que no se reabre 7.1 ni se importa automáticamente aquí todo el alcance
  de la spec distribuida.
- Ningún requisito de este cambio exige batches fusionados, speedup de
  throughput ni pesos físicamente repartidos. La documentación actual de
  servidor/worker declara batches secuenciales y workers que conservan su
  fuente completa. No son bloqueos nuevos de este archivo por sí mismos.
- Los perfiles nativos compartidos y los perfiles BML son explícitos. No se
  usa el éxito de una ruta nativa para acreditar una identidad simbólica ni
  los dos tokens Stories15M de v4 para acreditar TinyLlama simbólico completo.

## Reproducción y procedencia

La auditoría se hizo sobre HEAD `55be54c473a0acf8e703d2dc82b71d0f6273a3ee`. Las únicas modificaciones de
repo son este informe y las reaperturas documentales. Probe aislado:
`target/quality/fix-placeholders-audit-probe/`.

1. Crear allí un crate independiente con `[workspace]` y dependencias path
   a `../../../crates/domain`, `../../../crates/compiler` y
   `../../../crates/runtime`; copiar `Cargo.lock` raíz y resolver la entrada
   local mediante `cargo metadata --offline`.
2. Copiar `crates/compiler/src/sampler.rs` a `src/sampler_control.rs` y cambiar
   **solo** `let pick: f64 = rand_val();` por
   `let pick: f64 = (_seed % 100) as f64 / 100.0;`.
3. Usar este `src/main.rs` y ejecutar con toolchain 1.98.1:

```rust
mod sampler_control;
use bml_compiler::{op_fragments::compile_matmul_fragment,sampler};
use bml_domain::EvalContext;
use bml_runtime::{buffer::ResultBuffer,HotLoop};
use std::panic::{catch_unwind,AssertUnwindSafe};
fn main() {
    for (width,rows,inputs,weights,reference) in [
        (1u32,1u32,vec![1.0],vec![2.0],vec![2.0]),
        (2u32,2u32,vec![1.0,2.0],vec![3.0,4.0,5.0,6.0],vec![11.0,17.0]),
    ] {
        let size=weights.len().max(inputs.len()).max(rows as usize);
        let fragment=compile_matmul_fragment("audit",0,2,size as u32,rows,width,0);
        let mut buffer=ResultBuffer::new(3,size);
        buffer.slot_mut(0)[..inputs.len()].copy_from_slice(&inputs);
        buffer.slot_mut(1)[..weights.len()].copy_from_slice(&weights);
        buffer.slot_mut(2).fill(f64::NAN);
        let constants=[f64::from(width)];
        let context=EvalContext::new(&[],&constants);
        let mut hot=HotLoop::with_capacity(128);
        let result=catch_unwind(AssertUnwindSafe(||hot.execute_fragment_full(&fragment.fragment,&context,&mut buffer)));
        println!("matmul width={width} rows={rows} inputs={inputs:?} weights={weights:?} expected={reference:?} actual={:?} panic={}", &buffer.slot(2)[..rows as usize],result.is_err());
    }
    let logits=[0.0,1e-12];let temperature=1e-12;
    let output:Vec<_>=(0..100).map(|seed|sampler::sample(&logits,temperature,seed).unwrap()).collect();
    let control:Vec<_>=(0..100).filter(|&seed| {
        sampler_control::sample(&[0.0,1.0],1.0,seed)
            !=sampler_control::sample(&[0.0,1e-12],1e-12,seed)
    }).collect();
    println!("controlled_equal_uniforms mismatches={} seeds={control:?} pick_0_1_normal={:?} pick_0_1_scaled={:?}",control.len(),sampler_control::sample(&[0.0,1.0],1.0,10),sampler_control::sample(&[0.0,1e-12],1e-12,10));
    let zero_count=output.iter().filter(|&&id|id==0).count();
    println!("positive_temperature logits={logits:?} temperature={temperature} expected_probabilities={:?} zero_count={zero_count} one_count={}",[1.0/(1.0+1.0f64.exp()),1.0/(1.0+(-1.0f64).exp())],100-zero_count);
}
```

```sh
cargo run --manifest-path target/quality/fix-placeholders-audit-probe/Cargo.toml --locked --offline
```

Se usó el target compartido `target/global-restart-build`, perfiles dev/test
sin debug e incremental desactivado; no se lanzó ningún gate, build amplio,
modelo descargado ni campaña de mutaciones. El probe no establece resultados
sobre código futuro corregido.

SHA-256 de los archivos observados:

| Archivo | SHA-256 |
| --- | --- |
| `crates/compiler/src/op_fragments.rs` | `54d76c94a6e8dff32f1d5c9b88841289296b93cc47e72b7cf06949d3dec59376` |
| `crates/compiler/src/sampler.rs` | `9ee06569e0a47bb4ae372a80e388577e0e58e335739ab4b3d907201716d3afc2` |
| `crates/cli/src/main.rs` | `47da84e7b43c2d25a828f88c6b5890030ff476431ecc7e346d755391cab39275` |
| `target/quality/fix-placeholders-audit-probe/src/main.rs` | `75a787227f496a15e9d5f9fd4811bd8df1e4042fdf189d533dc8f5a91c308c09` |
| `target/quality/fix-placeholders-audit-probe/src/sampler_control.rs` | `549f3413c83fc36337fc5b715094e9acc47b2a27e56833deb9a8579c6407fe09` |
| `target/quality/fix-placeholders-audit-probe/Cargo.toml` | `bc0afa6951e3633216368f9d834c22e5f465344c29a2612ce44c2550eb7cf699` |
| `target/quality/fix-placeholders-audit-probe/Cargo.lock` | `1d050fb9e113033facf366aa2d28fec66a94a4564428793fec170fc921b12a3b` |
| `target/quality/fix-placeholders-audit-probe/stdout.log` | `9d491cbce64cc5c71e01b13feba82333acf5055ea09f41a70a632dc76a5f9740` |
| `target/quality/fix-placeholders-audit-probe/stderr.log` | `f7b3c44baebb788e8da0c2c9b337d189d6f4aafb46b6fdd5c4ceea4102585a47` |
