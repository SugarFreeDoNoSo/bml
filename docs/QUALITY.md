# Contratos y controles de calidad

Se adaptan escenarios trazables, evidencia TDD, revisión y mutación de [harness-sdd/uncle-bob-harness](https://github.com/betta-tech/harness-sdd/tree/bb80951622065a32057634d222cf7ed93a2a09cb). Clippy, cobertura, RustSec y hooks Git son controles adicionales implementados para Rust.

## Ejecución reproducible

```sh
./scripts/check.sh origin/master
./scripts/install-hooks.sh
cargo llvm-cov --workspace --locked --json --output-path target/quality/coverage.json
python3 tools/quality.py coverage --report target/quality/coverage.json
python3 tools/mutation_diff.py origin/master target/quality/changes.diff
cargo mutants --workspace --cargo-arg=--locked --test-workspace true --in-diff target/quality/changes.diff
python3 tools/quality.py mutations --report mutants.out/outcomes.json
```

Rust está fijado en `rust-toolchain.toml`; las versiones de las herramientas están fijadas en los workflows. El hook pre-commit comprueba el contenido staged, pre-push ejecuta los controles ordinarios y ambos conservan el código de error. Instalación solo local al repositorio; `--no-verify` permite omitir hooks. Las reglas de la rama deben exigir los checks de Actions para impedir merges cuando fallen; este PR no modifica esas reglas.

### Tamaño estático del hot loop

`python3 tools/measure_hot_loop.py --output quality/evidence/bml-hot-loop-code-size.json`
compila y ejecuta un probe enlazado con el perfil release del workspace, fat LTO
y stripping desactivado para conservar símbolos. Usa `llvm-nm` y `llvm-objdump`
del toolchain fijado; registra compilador, flags adicionales, hashes de fuentes
y binario, tamaño de los símbolos `dispatch_ops` y disassembly en
`target/quality/hot-loop-probe.asm`. Falla si no encuentra símbolos medibles.

La medición x86_64 registrada es de **1411 bytes** para el símbolo conservado.
El informe también inspecciona las tablas de símbolos normales y dinámicos del
probe completo: no contiene símbolos de RPC ni imports de sockets reconocidos.
`tools/test_measure_hot_loop.py` comprueba el detector con imports versionados y
rutas Rust de red. Este resultado solo cubre el probe independiente; la auditoría
del worker descrita más abajo aporta evidencia separada sobre su dispatcher.
No cuenta copias inline, funciones llamadas ni datos de tablas de salto; no es
el tamaño de un `.rlib`, ni una prueba de aislamiento de red o residencia L1i.
Los contadores PMU y la ejecución bajo carga siguen siendo verificaciones
separadas. El probe comprueba `B(1,1)=2`; las suites de contratos verifican las
otras operaciones, no esta medición de tamaño.

| Control | Criterio |
|---|---|
| Tests y doctests | La suite ordinaria pasa y los optativos figuran como ignored |
| Auditoría | Cero discrepancias en los seis contratos históricos |
| Trazabilidad | Veintinueve escenarios con tests registrados por package y target de Cargo; la ejecución se comprueba aparte |
| Formato | Rustfmt en archivos Rust nuevos o modificados |
| Clippy | Ningún diagnóstico nuevo por archivo, lint y mensaje |
| Cobertura | Líneas y funciones sobre los mínimos versionados |
| Mutación | Baseline correcto, campaña completa, cero supervivientes y cero timeouts |
| Dependencias | Auditoría RustSec al cambiar manifests/lockfile y semanalmente |

Actions conserva informes incluso cuando falla un control. Los jobs usan permisos de lectura, acciones fijadas por SHA y ejecución de PR sin secretos ni `pull_request_target`.

La base de Clippy registra 192 diagnósticos preexistentes de `57c3bd7` con Rust 1.98.1. `--cap-lints warn` permite inventariar también lints denegados heredados: cualquier diagnóstico nuevo o error de compilación sigue haciendo fallar el script. Esta deuda debe reducirse por cambios revisados; no regenerar la base automáticamente.

La cobertura estable mide líneas y funciones del workspace ordinario. No incluye ramas ni doctests: estos últimos se ejecutan por separado; ramas requieren otra campaña de instrumentación. Los mínimos iniciales tienen margen cercano a un punto porcentual respecto de la medición que los originó. Cambiar el presupuesto requiere revisión.

La mutación compara el árbol base formateado con el código actual para excluir cambios exclusivamente de formato. Cada mutante viable debe ser detectado. El porcentaje es `caught / (caught + missed + timeout)`; los no compilables se reportan aparte. Una campaña vacía o sin mutantes viables es N/A, nunca 100 %. Una campaña incompleta falla. La opción manual `full_mutation` mide todo el workspace. No hay exclusiones por equivalencia preaprobadas.

El CI ordinario lista primero sus candidatos y conserva revisión, base, hashes
del diff y de la selección, comando y salida original en `target/quality/`.
Una selección vacía verificada produce N/A sin ejecutar ni simular un baseline.
Un listado fallido, una respuesta ambigua o resultados incompletos hacen fallar
el job. Las selecciones no vacías siguen pasando por el gate original; N/A no
acredita la campaña completa del desarrollo.

La campaña completa del desarrollo se puede lanzar manualmente con
`gh workflow run mutation-campaign.yml --ref master -f shards=64`.
El workflow congela su propio SHA y el diff completo contra
`40b3f1be2d3f97028b741750ed61ab72f501a555`; no usa el diff del último push.
Comprueba ambas variantes de features GGUF y lista cada shard por separado.
Usa 64 shards por defecto (32 como alternativa) y hasta ocho runners estándar
simultáneos. Cada baseline prueba los paquetes mutados de su shard según
cargo-mutants 27.1.0; cada mutante se prueba contra todo el workspace.

El hash del manifiesto llega desde el job de preparación y se comprueba al
descargarlo. Se conservan candidatos, resultados originales, logs, regresiones
generadas y resultados sellados, incluso ante supervivientes o timeouts.
El agregado exige todos los shards del mismo intento y aplica el gate original.
Los artefactos duran tres días: descargarlos antes de su caducidad y repetir el
workflow completo cuando sea necesario; reintentar jobs individuales no mezcla
evidencia de intentos anteriores. La configuración y el inicio de la campaña
no acreditan su finalización ni un resultado aprobado.

El recolector reconoce las dos rutas de persistencia `SourceParallel` de
proptest 1.11: el árbol `proptest-regressions/*.txt` y el sufijo
`.proptest-regressions` junto a un test sin ancestro `lib.rs`/`main.rs`.
Ambas deben corresponder exactamente a una fuente Rust versionada, ser archivos
regulares acotados y contener semillas válidas. Durante la preparación no se
admiten archivos nuevos; al sellar se conservan sus hashes y sus contenidos en
el artefacto. Cualquier otro archivo fuente o entrada nueva sigue rechazándose.

Para el apagado del runner del shard20 del run `41950170549`,
`mutation-recovery.yml` permite ejecutar únicamente su partición original con
el checkout `9a6bc968` y el plan de preparación fijado por ID y SHA256.
La API debe acreditar la interrupción terminal por shutdown143 y la ausencia de
artefacto original. Se vuelve a comprobar la restauración de las fuentes.
Si el recolector congelado rechaza únicamente un sufijo proptest legítimo,
el controlador revisado puede emitir un sello propio sobre el checkout aún
vivo, con comprobación completa de fuentes, candidatos, baseline y resultados;
la procedencia distingue ambos selladores. Esto no revalida artefactos antiguos
que perdieron sus entradas ni altera sus resultados brutos. El agregado
original sigue exigiendo su intento completo. Una colección externa debe
verificar contra API la procedencia y el artefacto de cada recuperación,
seleccionar exactamente un sello por índice y aplicar después la misma unión
completa y el gate de mutaciones original. Preparar o iniciar la recuperación
no acredita que haya terminado ni que el gate esté aprobado.

La misma recuperación admite únicamente el fallo de recolección revisado del
shard21/job `126992338544`. Su ZIP `13386138847` se fija por SHA256 y se
inspecciona en memoria con límites de tamaño, rutas, entradas y expansión;
se contrastan candidatos, baseline, argv y el error final del recolector.
Esos resultados siguen sin atestación de fuentes/semillas: sólo justifican
un nuevo intento, y se conservan bajo el controlador sin copiarse al checkout
fresh. La partición21/64 debe ejecutar de nuevo baseline y todos sus mutantes,
con la fuente9a intacta y el mismo sellado posterior. La autorización antigua
del shard20 sigue siendo compatible con el verificador. Sus fallos de runner
se registran aparte y no aportan resultados sellados.

`mutation-recovery.yml` admite también el job original27 (`126992339813`),
con artefacto `13389245750` y ZIP fijado por SHA. La inspección comprueba los
81 candidatos y el fallo final de colección; sólo autoriza una repetición fresca.
Ese raw antiguo sigue sin atestación y no entra en la unión. Se conservan los
defaults21, la procedencia20/21 y el sellado posterior con fuente viva y semillas.

`mutation-subshards.yml` prepara una recuperación separada del shard20 tras
los dos apagados observados, cuya causa sigue sin determinarse. El helper
original de la fuente9a lista independientemente los 256 shards del mismo
universo y diff. Sólo ejecuta `20/256`, `84/256`, `148/256` y `212/256`:
21/20/20/20 candidatos, con unión exacta de los 81 del padre20/64. Mantiene
`--test-workspace true`, baseline habilitado, tiempos360/600 y entorno original.
La preparación ejecuta además Build/Test reales del workspace sin mutar.

Cada parte conserva su sello v1 y el baseline de sus propios paquetes.
El envelope v2 guarda por separado esos cuatro baselines y el preflight real
del workspace; no fabrica un baseline padre de nueve paquetes. El merger
valida las hojas con las reglas v1 existentes y conserva todos los resultados
originales. Exige hashes transmitidos desde prepare, procedencia API y bytes
ZIP contrastados, fuente comprobada al finalizar y los logs/estados/semillas.
Los timestamps API sin fracción se interpretan dentro de su segundo declarado;
los tiempos internos y los timestamps con fracción conservan comparaciones exactas.

Las partes tienen deadline externo de20min y job de25min para reservar margen
al upload. Cada una publica al terminar. Telemetría, stdout con `tee` y copias
parciales son sólo diagnóstico; una parte ausente, interrumpida, inconsistente
o sin fuente atestada impide el envelope completo. No se infiere cero de una
métrica no disponible ni se aceptan prefijos parciales. Preparar o iniciar este
workflow no aprueba la campaña completa ni recupera el raw antiguo20/27.

## Instrumentación optativa

Los casos TinyLlama `Hello`/`Hi`, su preflight nativo por roles y el diagnóstico
BML completo tienen un protocolo separado en
[REAL_PROMPT_DIAGNOSTICS.md](REAL_PROMPT_DIAGNOSTICS.md). Requieren selección
exacta del test, inputs fijados, captura de build y directorios nuevos por modo.

```sh
cargo test -p bml-parser --test zero_copy_strace -- --ignored --exact strace_shows_mmap_not_read
cargo test -p bml-tests --test perf_cache -- --ignored --exact perf_stat_cache_events_on_program
BML_TEST_GGUF=/ruta/modelo.gguf cargo test -p bml-gguf open_real_tinyllama -- --ignored
```

Strace traza el proceso del parser; perf mide un proceso que ejecuta el evaluador. Si se solicitan sin permisos o herramientas, fallan. No ejecutar `--ignored` indiscriminadamente, pues incluye las entradas auxiliares de esos hijos. Los tests `thread_concurrency` usan hilos del sistema, sin demostrar exploración exhaustiva de interleavings mediante Loom.

### Sonda de subfragmentos con control externo de PMU

```sh
cargo build --release -p bml-runtime --example subfragment_cache_probe --locked
target/release/examples/subfragment_cache_probe --verify
target/release/examples/subfragment_cache_probe --warmup 10000 --iterations 1000000
```

La sonda llama a `Runtime::execute_sub_fragments` con cuatro arrays de opcodes
prestados, bucles anidados y un buffer compartido entre subfragmentos. Las cuatro
entradas `[x,y]` son `[0,1]`, `[1,2]`, `[2,3]`, `[3,4]`; el resultado independiente
`2^x + 4*y + 8` da los literales `[13,18,24,32]`. `--verify` comprueba cada caso,
el contenido del buffer y un checksum de 12 llamadas sin PMU ni interacción.
Los contadores `--warmup` y `--iterations` deben ser positivos y como máximo
`2^48`, para que el checksum entero sea representable exactamente en `f64`.

El setup reserva arrays, pila y buffers. El workload reutiliza esa memoria,
mantiene sus entradas/resultados opacos mediante `black_box` y no asigna memoria.
Después del calentamiento y sus comprobaciones, `READY` identifica PID, ruta,
iteraciones y `totalfragmenttransitions = 4*iterations - 1`: cuenta las parejas
adyacentes de arrays ejecutados, incluidas las que cruzan llamadas; excluye el
calentamiento. El operador conecta y habilita sus contadores al PID publicado y
envía `go` seguido de salto de línea por stdin. La sonda ejecuta el workload,
comprueba checksum/buffer, emite `FINISHED` y espera `stop` seguido de salto de
línea. El operador deshabilita los contadores antes de enviar `stop`; el detalle
`RESULT` y la salida del proceso ocurren después. La sonda no invoca `perf`.

Este protocolo deja dentro del intervalo externo el overhead mínimo de
sincronización: lectura de `go`, comprobaciones finales, escritura/vaciado de
`FINISHED` y la espera hasta que el operador deshabilite los contadores. No mide
ni resta ese overhead. Los arrays son bytecode para el mismo dispatcher; esta
sonda ejercita esa ruta y no establece residencia L1i ni una medición exclusiva
del dispatcher. Los archivos de contadores por sí solos tampoco lo establecen.

Para la campaña, seguir el [protocolo L1i acordado](../openspec/changes/bml-implementation-plan/specs/bml-runtime/spec.md):
fijar la afinidad al mismo core antes del calentamiento y conservarla en las
siete repeticiones, con cantidades idénticas de calentamiento y trabajo.
Registrar `instructions` y `L1-icache-load-misses` sobre el mismo PID e intervalo,
ambos sin calificador o ambos con `:u`, sin multiplexado parcial. Conservar los
JSON originales, CPU, afinidad, comandos y hashes de fuentes y ejecutable;
auditar el tamaño y alcance del código máquina del binario que se mide, sin
reutilizar el tamaño de otro worker. El límite es menor de 32 KiB y como máximo
un miss por mil instrucciones en cada repetición. Tras comprobar procedencia
y significado de los eventos en esa CPU, validar el sobre JSON con
`python3 tools/validate_l1i_campaign.py campaña.json`; el formato se documenta
en la cabecera del script. La validación de datos no sustituye esas auditorías.

## Contratos y límites

- El escalar `Bml` codificado en bits, sus reglas y síntesis acotada están
  implementados en `bml-domain::number`; requieren validación independiente de
  gramática, dominios, límites y precisión. Véanse [contratos](architecture/BML_NUMBER.md)
  y [tareas](../openspec/changes/bml-number-foundation/README.md).
- La síntesis predeterminada limita el resultado a 1.048.576 bits y busca
  denominadores hasta 1024, con tolerancia absoluta 0 y relativa 1e-12. Incluso
  valores finitos pueden fallar por recursos, evaluación o precisión.
- Las rutas experimentales de ejecución rechazan arquitecturas distintas de `llama` antes de expandir pesos. Las regresiones de embeddings, proyecciones y logits verifican el orden GGUF con `ne0` contiguo. No certifican inferencia Gemma4 ni equivalencia con llama.cpp; véase [validación de inferencia](INFERENCE_VALIDATION.md).
- `bml-compiler::tokenizer::Vocabulary` usa el tokenizer SentencePiece BPE validado de `bml-inference`; `encode` y `decode` devuelven `Result`. La generación usa `token_piece_bytes`, preservando espacios y UTF-8 partido entre tokens. La CLI escribe bytes; SSE espera secuencias completas y sustituye bytes inválidos o incompletos al terminar. Los fixtures numéricos de cero capas sin metadatos BPE no pueden codificar texto. Esto no acredita la causalidad del motor histórico.
- `InferenceCompiler::forward` y `forward_cached` devuelven `Result` y ejecutan prefijos causales con residuales sin normalizar, GQA según `attention.head_count_kv` y KV F32. Validan IDs, layout y capacidad antes de escribir; los errores numéricos no avanzan la posición confirmada. Las filas aún no confirmadas se sobrescriben al reintentar. CLI y servidor usan `HashConsedKV::try_new` con un presupuesto de payload KV de 512 MiB. El almacenamiento heredado directo sigue usando índices circulares, pero la inferencia no permite envolver posiciones. El modo `--bml` comparte ese flujo causal y ejecuta las proyecciones con productos escalares RPN `FMul`/`FAdd`, usando dos filas de scratch; no implementa las identidades simbólicas AST/EML. C53 verifica matrices rectangulares con offset, límites y prefijos GQA contra un oráculo cerrado; C51 cubre UTF-8, EOS y capacidad del CLI. Los backends de proyección se validan por dimensión y finitud antes de consumir sus resultados. Evidencia: `quality/evidence/legacy-rpn-causal.json`.
- CLI y servidor históricos usan `InferenceCompiler::open_mapped`: mantienen matrices cuantizadas en el mapping GGUF y decodifican una fila para cada producto escalar, conservando acumulación F64 y KV F32. `open` conserva la variante expandida para comparación. `get_embedding` devuelve `Result`; en modo mapeado `weight_pool` y `weight_offsets` solo contienen normalizaciones. El archivo mapeado debe permanecer sin modificaciones. C12 cubre lectura por filas, C55 pesos Q4/Q6 con oráculos y C53 atención causal nativa/RPN con ambos almacenamientos.
- El artefacto causal nuevo usa `model.bmlgraph.json` y programas RPN serializados con SHA-256, configuración, BPE y bindings al GGUF externo completo. `bml-cli compile` local lo produce y `run` ejecuta los programas cargados; `--weights` permite reubicar el mismo GGUF, verificando su hash. Los programas pasan `ValidatedGraph` antes de ejecutar. C56 prueba GQA, ejecución del bytecode persistido y rechazos; C57 prueba los comandos con UTF-8/EOS/contexto. El perfil `llama-causal-f64-f32kv-rpn-v1` mantiene normalización, RoPE, atención y residuales nativos compartidos: no acredita aún su traducción a identidades BML. Los directorios escalares antiguos ya no anuncian generación de texto.
- `bml-server --model modelo.bmlgraph/ [--weights modelo.gguf]` carga el mismo artefacto causal y ejecuta sus programas tanto en JSON como en SSE; las rutas nativa y compilada comparten admisión, caché por request y ensamblado UTF-8. C58 inicia el ejecutable en un puerto efímero, realiza HTTP real, verifica texto/EOS/DONE sin eco, error de contexto y una modificación de bytecode que cambia la respuesta. El servidor rechaza contenedores escalares antiguos al arrancar, en vez de aceptar el modelo y responder 501.
- El contenedor histórico por directorio `.bmlgraph` se carga con `GraphLoadLimits`: por defecto 64 MiB codificados, 65.536 fragmentos, 1.000.000 operaciones y 8.000.000 constantes. Los límites son acumulativos; se rechazan truncamientos, bytes sobrantes, versión desconocida, UTF-8 inválido y opcodes desconocidos. C54 prueba fixtures binarios independientes. Estos límites no son un presupuesto RSS ni validan el stack o los pasos de ejecución (eso corresponde a `ValidatedGraph`); el formato aún no acredita generación autorregresiva desde el artefacto.
- RPN conserva cada nodo alcanzable una vez, con bytecode y pila lineales en los nodos alcanzables. Inicializar el mapa de posiciones y copiar el pool cuesta además O(nodos totales + constantes). El mayor uso de pila obliga a medir de nuevo el rendimiento.
- `.bmlgraph` v2 guarda el pool de constantes; el lector sigue admitiendo v1. Consumidores antiguos necesitan actualización para leer v2. Una API `*_with_ctx` usa el pool del contexto explícito.
- En `BMLTransformer`, `add` admite signos en el dominio numérico probado; `sub` exige minuendo positivo; `mul`, `div`, `recip` operandos positivos; `pow` x > 1 e y > 0. Las expansiones no garantizan estabilidad f64 general. `neg` conserva su convención IEEE con infinitos intermedios. La nueva `neg_real` usa la diferencia de dos términos positivos, sin introducir `log2(0)` y con un rango numérico más estrecho; `scale_signed` y `mul_signed` expanden productos con signos mediante componentes positivos. C59 comprueba signos, cero y fracciones contra un oráculo independiente, valida cada intermedio real/finito y mantiene visibles fallos de rango/dominio. No acredita estabilidad general ni traducción del transformer completo. Estas restricciones no corresponden a los kernels nativos de `tensor`.
- `BMLTransformer::{matmul,rms_norm,rope,softmax,swiglu}` construyen los cinco sub-DAGs con `One`, `Var`, `Const` y `Bml` (C60). Los pesos y los coeficientes fijos pueden referenciarse mediante `constant`, cuyo pool debe acompañar al evaluador. C61 linealiza y serializa cada salida, rechaza bytecode aritmético nativo y ejecuta los fragmentos con `ValidatedGraph`. La prueba de DAG inspecciona cada intermedio; el hot loop conserva su semántica IEEE. Softmax no incorpora un máximo estabilizador; RoPE recibe cosenos/senos precalculados y gira pares adyacentes. SwiGLU es `gate * sigmoid(gate) * up`. Estas pruebas acotadas no acreditan el compilador GGUF completo, pesos zero-copy en esa futura ruta ni estabilidad general.
- `ValidatedGraph::new_with_external_constants` exige un pool almacenado vacío y declara el número exacto de constantes prestadas (C62). `execute_with_constants` comprueba modo, entradas, constantes y buffer antes de escribir; el contexto cuenta en `max_values`. Los grafos de pool propio conservan su API y no admiten sustituciones implícitas. `symbolic_projection::compile_row` genera referencias `Const` externas bajo un presupuesto exacto de nodos; C63 serializa el programa vacío de pesos y lo ejecuta con filas de un GGUF F32 mapeado. Decodificar/convertir una fila a f64 puede asignar memoria: la afirmación es ausencia de pesos dentro del programa, no ausencia de toda copia durante ejecución. Esta plantilla BML está desenrollada y no cambia el perfil causal nativo ni demuestra todavía el compilador simbólico completo.
- `runtime::symbolic_causal::SymbolicCausalModel` integra esas operaciones en una ruta causal BML completa, con activaciones F64 y KV F32 (C64). Matrices, normalización, rotaciones, scores, escala, softmax, combinación de valores, SwiGLU y residuales usan programas cuya lista permitida excluye aritmética nativa. El host conserva lookup/decodificación, constantes posicionales y almacenamiento KV. Un fixture GQA con MLP no nula contrasta cuatro prefijos, decode incremental y reintentos con un oráculo independiente; tolerancia absoluta 2e-7. Los presupuestos limitan nodos, programas y opcodes almacenados. El constructor directo sobre GGUF compila kernels bajo demanda y el hot loop conserva semántica IEEE: se comprueban salidas finitas, no cada intermedio real. Un caso numéricamente válido en modo nativo falla explícitamente en BML. C64 por sí solo no acredita persistencia, modelos grandes ni equivalencia numérica general.
- C65/C66 verifican `symbolic.bmlgraph.json`, el plan AOT completo para un contexto declarado, los hashes del GGUF/programas y los literales canónicos. `open_artifact` valida todo el bytecode antes de inferencia y prohíbe compilación bajo demanda. Los archivos no contienen pesos aprendidos; los grafos tienen pools vacíos y reciben constantes externas. La prueba altera un programa residual guardado y comprueba los logits resultantes, además de cargar con presupuesto de construcción cero. El CLI admite `bml-cli compile -m model.gguf -o model.symbolic --symbolic --context-size 8` y `bml-cli run -m model.symbolic -p "Hello" -n 2 --temp 0`; `--weights` reubica el mismo GGUF. La generación limita su capacidad al contexto compilado. Persistir el plan no amplía el rango numérico BML ni acredita rendimiento/capacidad con modelos reales.
- C67 extiende la ruta HTTP local al artefacto simbólico: `bml-server --model model.symbolic --weights relocated.gguf` carga y valida el mismo plan AOT. JSON y SSE usan el límite compilado, además del contexto del modelo; un prompt demasiado largo devuelve error y SSE cierra con un único evento terminal y `[DONE]`. Los fragmentos de bytes UTF-8 no se emiten incompletos. Un programa guardado modificado cambia el resultado HTTP; los directorios con marcadores nativo y simbólico simultáneos se rechazan al arrancar. Se conservan los tests del perfil nativo y de admisión/cancelación.
- C69–C71 comparten los sub-DAGs de RMSNorm, softmax y RoPE entre salidas. El perfil `llama-symbolic-bml-f64-f32kv-v2` persiste un programa por operación vectorial, con índices de salida canónicos en el prefijo de constantes y pesos externos después. `linearize_outputs` conserva cada nodo alcanzable, escribe en slot 0 y deja un resultado administrativo; la aritmética sigue siendo BML. El runtime inicializa el buffer a NaN y exige todas las salidas finitas: detecta salidas omitidas, no certifica cada escritura adicional. Los artefactos escalares v1 siguen cargándose con su layout original y se contrastan bit a bit con v2, sin recompilar al cargar. El tamaño de bytecode de RMSNorm/softmax crece linealmente con el ancho; para ancho 32 se midieron 50.233/14.047 instrucciones frente a 982.688/448.864 escalares. El caso RMSNorm128 necesita aumentar explícitamente `max_output_ops`; los defaults permanecen intactos. Evidencia `quality/evidence/symbolic-shared-vector-programs.json`. El rango numérico, las proyecciones desenrolladas y el plan completo de prefijos siguen limitando modelos grandes; estos conteos no son una medición de velocidad.
- GGUF valida rangos exactos para escalares, Q4_0, Q8_0 y Q6_K. Otros cuantizadores solo permiten inspección de metadatos; `tensor_data` devuelve None. Se rechazan arrays anidados. El archivo mapeado debe permanecer sin modificaciones ni truncamiento mientras se usa.
- C114 expone `KernelSignature::buffer_literal_count()` para calcular, sin asignación, el prefijo fijo más los offsets de salida desde una firma base. Admite un conteo `u32::MAX` y rechaza exceso u overflow antes de materializar offsets. Los pesos externos quedan fuera del conteo; el caller aplica sus presupuestos y la fábrica conserva las comprobaciones de layout y reserva. La prueba usa firmas públicas de un literal y un contador de asignaciones calibrado, sin reservar el vector teórico de 32 GiB.

- `tensor` ofrece buffers contiguos f32/f64, vistas prestadas y kernels CPU. EML/BML elementwise valida en f64 y comprueba la conversión de salida; broadcasting y strides arbitrarios siguen pendientes.
- Reader/codecs y vistas GGUF reconstruidos en `bml-gguf`, con puente tensorial optativo; fuzzing, validación completa de bytecode y concurrencia instrumentada siguen en el roadmap. Microbenchmarks no certifican inferencia ni tokens/s reales.

## Recuperación GGUF (2026-10-22)

El bloque B14/#18 se reconstruye desde `df146c2`; los resultados locales perdidos
no son evidencia. Por defecto `bml-gguf` solo depende de `memmap2`; `bml-parser` delega sin
cambiar su `io::Result` y compiler usa el codec compartido. Los fixtures verifican
F32/F16/Q4_0/Q8_0 y las 65.536 codificaciones binary16. Se corrige la identificación
GGML: 29 es IQ1_M, 30 BF16; ninguno tiene decoder f32 en este incremento. Los
valores auxiliares del enum para metadatos no son IDs serializables GGML.

Los errores tipados conservan la categoría y contexto del fallo. El decoder no
acepta colas/truncamientos ni devuelve salidas parciales. Los arrays anidados
siguen rechazados explícitamente. Esto no certifica todos los archivos GGUF ni
inferencia, rendimiento, fuzzing o una campaña completa de mutación.

## Reconstrucción de vistas GGUF (#19)

La pérdida del entorno local exige reconstruir y validar de nuevo este incremento.
`view.rs` y el consumidor externo se recuperan de texto completo del transcript;
`tensor_views.rs` se escribe de nuevo a partir de contratos y fragmentos, sin
atribuirle identidad byte a byte con el archivo perdido.

El feature `tensor` añade `bml-domain` temporalmente hasta extraer `bml-core`.
Las vistas están ligadas al reader; la conversión invierte ejes GGML sin reordenar
valores. Los tests cubren matrices rectangulares, rangos 1/3/4, F16/Q4_0/Q8_0,
ownership, errores tipados y una restricción de lifetime mediante doctest.
`bml-tests` activa el feature como dependencia de desarrollo y registra un
consumidor real, de modo que los comandos existentes del workspace lo incluyen.

Esta reconstrucción pasó `scripts/check.sh`, las suites aisladas sin features y
con `--all-features`, cobertura y mutación del delta #19 con tests del workspace. No se modifican
baselines ni workflows. El estado comprobado está en
`quality/evidence/gguf-tensor-views-reconstruction.json`; no usar cifras antiguas.

### Publicación append-only y Loom

`Runtime::execute_append_only` publica en `AppendOnlyResults`, que reserva slots
una vez, conserva los bits IEEE y nunca recicla índices. Al llenarse devuelve el
resultado no almacenado; los valores anteriores permanecen inmutables. El método
histórico `Runtime::execute` continúa usando un ring y puede sobrescribir al rotar.

```sh
cargo test -p bml-runtime --lib --locked append_only
cargo test -p bml-runtime --lib --features loom --locked append_only::loom_tests
```

Con `cfg(test)` y la feature `loom`, el mismo archivo/algoritmo sustituye solo los
atómicos de std por los de Loom. Las pruebas exploran dos productores/lector
y una carrera de capacidad uno; el estrés nativo cubre 4.000 resultados de cuatro
runtimes. No se atribuye esa exploración a ArrayQueue/Deque de Crossbeam, canales
ni mmap: sus atómicos no están instrumentados por estas pruebas. Las pruebas
históricas con `std::thread` tampoco pasan a ser pruebas Loom por añadir la feature.

`PendingWork` usa ahora `bml-queue`, una adaptación del algoritmo ArrayQueue de
Crossbeam 0.3.13 con procedencia y licencias en `crates/queue/`. Producción y el
modelo compilan el mismo archivo, sustituyendo atómicos, accesos a UnsafeCell y
backoff. La API pública sigue siendo nativa incluso con todas las features; el
runtime conserva `deny(unsafe_code)`. Ejecutar:

```sh
cargo test -p bml-queue --all-features --locked
just test-loom
```

Seis modelos cubren saturación, transferencia única, publicación, reutilización
concurrente y destrucción. Un modelo consume antes de terminar el productor para
no obtener sincronización accidental mediante `join`; detecta en copias aisladas
la relajación del Release de publicación y del Acquire de consumo. Son escenarios
pequeños, sin una prueba general de liveness ni cobertura de reclamación/resize
del deque de work-stealing. Evidencia: `quality/evidence/pending-work-loom.json`.

El framing TCP legacy del worker rechaza payloads declarados superiores a 64 MiB
antes de reservar memoria o leer el cuerpo. Para shards mayores debe configurarse
`bml-worker --max-message-bytes N`; `N` es un presupuesto positivo representable
por el campo u32 del protocolo. Los consumidores de la API pueden usar
`recv_msg_with_limit` con su propio presupuesto. Una recepción rechazada invalida
la sesión: el cuerpo no consumido no debe interpretarse como otra cabecera.
Las pruebas `frame_budget_rejects_header_before_body_and_accepts_exact_limit` y
`configured_frame_limit_closes_oversized_session_before_reading_body` verifican
el límite inclusivo y el rechazo usando sockets reales sin enviar el cuerpo.

Dentro de ese framing, `BatchRequest` legado exige una dimensión u32 que coincida
con el shard y exactamente esa cantidad de valores f64 little-endian. Rechaza
truncamientos, dimensiones discordantes y bytes sobrantes con `BatchResult` vacío
y cierre de sesión. `legacy_batch_tcp_validates_dimensions_and_exact_payload_length`
comprueba por TCP la carga del shard, su ACK y los mensajes válidos e inválidos;
un shard sin capas preserva el vector byte por byte. Esta prueba valida el protocolo
legado, no inferencia causal completa ni generación de tokens.

La carga legacy de `--fragment` restaura la configuración desde el header hermano.
El lector limita el header a 1 MiB y el shard a 512 MiB; comprueba versión, UTF-8,
truncamientos, conteos y rangos antes de reservar por conteos declarados. El worker
valida las formas que ejecuta: RMSNorm `[n_embd, 1]` y Q `[n_embd, n_embd]`, con
dimensión positiva y rangos de pesos válidos. Un shard malformado se rechaza sin
sustituir el cargado ni envenenar el mutex. El ejecutor recorre sólo capas con
operaciones presentes, en orden, sin iterar rangos declarados de capas vacías.
El límite de archivo no equivale al pico de RSS: bytes y pesos decodificados
coexisten. C68 y las regresiones TCP/file del
worker acreditan estos límites, no un transformer completo.

`bml_runtime::graph::ValidatedGraph` realiza preflight sobre un `.bmlgraph` ya
parseado: conserva constantes y pila entre fragmentos, verifica referencias y
underflow, limita instrucciones almacenadas, pasos expandidos y valores vivos,
y exige un único resultado final. Los bucles deben quedar dentro del fragmento
y no superar dos niveles; incluso un cuerpo vacío consume presupuesto por cada
índice de iteración. Los cuerpos con cero iteraciones también se inspeccionan
para detectar referencias y estructura inválidas. La pila se reserva de forma
falible antes de ejecutar. Un cambio de dimensiones del contexto/buffer se
rechaza antes de escribir resultados. La operación `ExecuteGraph` utiliza este preflight antes de ejecutar los
fragmentos recibidos; la carga legacy `ExecuteFragment` conserva su respuesta de
confirmación y no se presenta como inferencia.

La RPC de grafos usa los tipos 10 (`ExecuteGraph`) y 11 (`GraphResult`). El body
v1 contiene cinco u32 little-endian: versión, número de inputs, número de slots,
tamaño de slot y longitud del `.bmlgraph`; siguen los inputs f64, el buffer f64
plano y el grafo íntegro. El máximo de body es 16 MiB incluso si el límite legacy
es mayor. El decoder acota fragmentos/opcodes a 100 000 y valores a 1 000 000;
el preflight limita además pasos expandidos a 1 000 000. La respuesta comienza
con versión y estado u32. Un rechazo devuelve estado 1 sin datos parciales; un
éxito devuelve estado 0, resultado f64, número de valores u32 y buffer f64.

`NodeHandle::execute_graph` conserva los inputs/buffer del caller y devuelve un
resultado nuevo. El worker no retiene estado mutable entre esas peticiones. Se
pueden encadenar etapas en distintos workers transfiriendo el buffer devuelto;
la prueba `graph_rpc_two_worker_stages_match_complete_local_graph` compara dos
workers TCP reales con el mismo DAG local, incluida la reutilización de sesiones.
No hay particionado automático del compilador ni retry de peticiones ambiguas.
Los fallos de transporte/respuesta malformada cierran el socket del cliente; un
rechazo de validación completo permite reutilizar la sesión. Las sesiones KV
causales usan el protocolo separado de `bml-inference::remote`.

`NodeHandle::execute_graph_append_only` publica el resultado RPC en
`AppendOnlyResults`: primero el escalar y después todos los valores del buffer,
en un rango nuevo que nunca se recicla. La reserva del rango completo es atómica;
si no cabe, no consume posiciones y devuelve `GraphPublishError::StorageFull`
con el `GraphResult` íntegro. Ese resultado puede publicarse después mediante
`GraphResult::publish` sin volver a ejecutar la RPC. Los errores RPC se distinguen
de esta saturación local; no se reintentan peticiones remotas ambiguas.

El descriptor solo se devuelve después de publicar todos los campos. Lectores
que inspeccionen `reserved_count` durante la escritura pueden observar celdas aún
no publicadas (`get` devuelve None); no se promete una instantánea atómica de
varios valores. Loom instrumenta los atomics reales de reserva y publicación:
comprueba rangos disjuntos, saturación sin reservas parciales y estabilidad de
los resultados. Esta evidencia no cubre los atomics no instrumentados de las
colas Crossbeam. La prueba de dos workers publica ambas etapas, comprueba que
los resultados anteriores no cambian y conserva una respuesta cuando se llena
el almacén.

`python3 tools/audit_hot_loop_binary.py --output
quality/evidence/worker-hot-loop-isolation.json` inspecciona el dispatcher dentro
del **worker real** x86-64 release, con símbolos retenidos y RPC/socket enlazados.
Resuelve las llamadas indirectas mediante relocaciones GOT (sin confiar en las
etiquetas que muestra objdump), comprueba sus desplazamientos RIP y decodifica
las tablas de salto desde segmentos ELF. Cada destino debe ser una instrucción
del mismo dispatcher; las llamadas deben estar en la lista revisada de recursión,
exp2/log2 y crecimiento de pila/pánico por límites. Cualquier destino nuevo o no
resuelto hace fallar la auditoría. Los tests inyectan llamadas de red, syscalls y
saltos externos para verificar que no se acepten silenciosamente.

La auditoría se repitió tras integrar los artefactos causales y el servidor HTTP/SSE;
el worker reconstruido mantiene **1590 bytes**, 13 llamadas resueltas y dos saltos
por tabla internos (hashes actuales en la evidencia). Los 1411 bytes del probe anterior
son otro binario/snapshot. La auditoría no suma los callees de allocator/pánico,
no analiza hooks externos ni interposición de librerías, y no acredita caché L1
ni contadores PMU. El preflight reserva la pila máxima y valida referencias para
la ruta ordinaria de ejecución. Una arquitectura o compilador cuyo patrón no
reconozca el auditor requiere revisar la evidencia, no desactivar la comprobación.


## NEON functional checks under AArch64 emulation

`quality/evidence/arm-neon-code-comparison.json` records eight all-feature
kernel tests on QEMU AArch64 and eight on native x86. The four structural-code
comparison tests explicitly exercised Scalar/NEON in the emulator and
Scalar/AVX2 on x86. They cover word positions and unsigned ordering, vector
tails, non-evaluable symbolic terms, limits, first-error ordering and unchanged
inputs. AArch64 tests assert NEON availability so it cannot be silently skipped.
The other four tests cover the separate native IEEE tensor API; on AArch64 it
uses its Reference fallback, not NEON arithmetic.

For functional reproduction from an x86-64 host, install the pinned toolchain's
musl target and provide `qemu-aarch64` on PATH. Use a separate Cargo target
directory and the pinned toolchain's self-contained linker:

```sh
rustup target add aarch64-unknown-linux-musl
BML_CROSS_SYSROOT="$(rustc --print sysroot)"
CARGO_TARGET_DIR="$PWD/target/arm-correctness-build" \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER="$BML_CROSS_SYSROOT/lib/rustlib/x86_64-unknown-linux-gnu/bin/rust-lld" \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUNNER="qemu-aarch64 -cpu max" \
cargo test -p bml-kernels --all-features --target aarch64-unknown-linux-musl \
  --locked -- --nocapture
```

The recorded run used a copy of the pinned toolchain under `target/arm-rustup`
and the Debian QEMU binary under `target/arm-qemu/runtime/usr/bin`; hashes and
preserved test executables are recorded in the evidence. QEMU results establish
functional behavior under emulation. They provide no real-ARM throughput,
hardware counters or SIMD speedup evidence; task 7.4 remains open.


## Native kernel campaigns on GitHub-hosted runners

`.github/workflows/native-kernels.yml` runs the same structural-code experiment
on `ubuntu-24.04` (x86-64/AVX2) and `ubuntu-24.04-arm` (AArch64/NEON). GitHub
lists both as standard runners available free for public repositories:
[GitHub-hosted runners reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).
The workflow runs on relevant master pushes and can also be dispatched manually.

`tools/native_kernel_campaign.py` rejects a host/runner/ELF architecture mismatch
and Cargo cross-runner overrides. It executes the eight all-feature kernel
contracts, builds the pinned-toolchain native benchmark, and pins the benchmark
to one allowed CPU. Each of three fixed workloads uses 512 pairs, 100 iterations
and seven repetitions, with warm-up excluded. Scalar and the required SIMD
backend must both produce complete samples; an availability flag alone is not
evidence of execution. Timings and allocation counts describe structural code
comparison, not BML numerical evaluation or inference throughput.

Each job uploads the native executable, full samples, test/build logs, CPU
metadata, source hashes and an output manifest. Source hashes must remain
unchanged throughout the campaign. Job success and downloaded artifact checks
are required before crediting task 7.4; creating this workflow is not a completed
ARM measurement. Hosted-runner wall times are workload-specific measurements,
not hardware-counter or instruction-cache-residency evidence.


The first completed native run is
[41853147652](https://github.com/SugarFreeDoNoSo/bml/actions/runs/41853147652)
on revision `6d15db6`. Both hosted native VMs passed eight all-feature tests;
ARM64 exercised Scalar/NEON and x86-64 exercised Scalar/AVX2. Downloaded ZIP
checksums, every reported file hash, ELF machine types and all captured source
hashes were independently verified. Full CPU metadata and all measured samples
are retained in `quality/evidence/native-kernels-41853147652.json`. These results
complete the native comparison-kernel campaign; they do not establish PMU/L1
residency or numerical/inference speedups.
