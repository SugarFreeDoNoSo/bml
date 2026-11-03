# BML 1.0: candidato de entrega

El candidato incorpora la biblioteca numérica, un motor Llama CPU en
`bml-inference` y la generación de ejecutables nativos con recursos incluidos.
La compilación candidata 1.0.0 pasa la comparación con llama.cpp tanto mediante el
CLI como mediante el ejecutable autónomo. La entrega 1.0 sigue abierta hasta
cerrar los controles de calidad; los manifiestos preparan la versión 1.0.0,
todavía sin declarar la entrega completa ni publicada.
El estado verificable está en la [evidencia compacta](../quality/evidence/bml-1.0-candidate.json)
y las [tareas de entrega](../openspec/changes/bml-1-0-release/tasks.md).

## Perfil y contratos

El perfil contrastado con llama.cpp es **Stories15M GGUF F32, CPU, Linux amd64
GNU**. El modelo tiene SHA-256
`61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`.
La referencia está fijada al commit de llama.cpp
`785322d17f2d51e5185cf996a1f4e46d87640841`.

El motor implementa Llama denso con RMSNorm, RoPE estándar, atención causal con
KV, MLP SwiGLU y proyección de logits. El tokenizer es SentencePiece BPE del
perfil `tokenizer.ggml.model=llama`: usa scores, tipos, fallback por bytes y
metadatos BOS/EOS. La generación selecciona el máximo logit (*greedy*), resuelve
empates por el menor ID y termina al producir EOS o alcanzar `--predict`.
No hay sampling por temperatura, top-k o top-p.

Los recursos admiten F32, F16, Q4_0, Q8_0 y Q6_K; los pesos cuantizados conservan su
codificación y se decodifican por filas. Esa capacidad no extiende la evidencia
de paridad de Stories15M F32 a modelos cuantizados ni a otros modelos. Las
activaciones y el KV son F32, con acumulaciones escalares F64. Las arquitecturas,
variantes de RoPE y roles tensoriales ajenos al perfil aceptado producen errores.

Q6_K tiene [pruebas contra 2.048 valores de GGML y de exportación/recarga](../quality/evidence/q6-k-codec.json).
La [prueba diagnóstica de TinyLlama Q4_0 con salida Q6_K](../quality/evidence/tinyllama-quantized-diagnostic.json)
carga y genera, pero diverge de la referencia en el octavo token de «Hello».
Esta observación no acredita paridad cuantizada ni rendimiento comparable.

La fachada `bml` conserva el escalar `Bml` y las capacidades tensor/GGUF optativas.
Los contratos son distintos: `Bml` y los términos simbólicos comprueban dominio
y rango; los kernels tensoriales nativos usan aritmética IEEE. La inferencia no
convierte cada peso en un término simbólico ni acredita una aceleración debida
al operador BML. Véanse [el escalar](architecture/BML_NUMBER.md),
[los tensores](architecture/TENSOR_LIBRARY.md) y [los controles](QUALITY.md).

## Conversión GGUF → pesos BML

La conversión implementada está definida en la
[especificación de pesos BML](../openspec/specs/bml-model-weights/spec.md):

`modelo.gguf → síntesis exacta acotada de pesos → modelo.bmlgguf → carga de matrices BML → inferencia simbólica`.

El contenedor autónomo `.bmlgguf` guarda pesos como términos BML,
formas y metadatos tipados (incluidos tokenizer/configuración presentes en la fuente).
La biblioteca y los comandos `bml-cli convert-weights` e `inspect-weights` están
implementados. La API de inferencia simbólica admite el perfil explícito de
evaluación comprobada descrito abajo. Las
[tareas y criterios de cierre](../openspec/changes/archive/2026-11-03-gguf-to-bml-weights/tasks.md)
incluyen validación exacta por peso, presupuestos, carga sin el GGUF original y
mediciones de conversión e inferencia.

```sh
cargo run -p bml-cli --locked -- convert-weights -m modelo.gguf -o modelo.bmlgguf \
  --max-synthesis-steps-per-weight 4096 --max-synthesis-steps-total 1000000 \
  --max-block-terms 1024 --max-output-bytes 68719476736 \
  --max-resident-bytes 268435456 --max-cache-bytes 1048576
cargo run -p bml-cli --locked -- inspect-weights -m modelo.bmlgguf
cargo run -p bml-cli --locked -- inspect-weights -m modelo.bmlgguf --verify-all
```

`convert-weights --help` e `inspect-weights --help` muestran todos los límites y
sus valores por defecto. Las opciones de lectura incluyen tamaño de archivo,
metadatos, índices, cantidad de tensores/bloques/términos, bits y profundidad.
Los comandos imprimen los límites solicitados; la conversión informa hashes,
conteos, tiempos por fase y pico de memoria contabilizada (no RSS).
El CLI termina con código 1 ante error operativo/de contrato y 2 ante argumentos
inválidos. El destino debe ser nuevo; un error no publica un archivo parcial.

Los dtypes admitidos son **F32, F16, Q4_0, Q8_0 y Q6_K**. La conversión conserva
el valor IEEE exacto o el F32 del codec cuantizado, normaliza ambos ceros y rechaza
NaN/inf. No implica que una arquitectura esté admitida por inferencia. La evidencia
actual usa fixtures pequeños; no acredita conversión completa de un modelo real.
Síntesis y replay tienen contadores independientes sujetos al límite global
declarado; cachear un peso ahorra síntesis pero no su replay. Los límites internos
del certificador también se aplican y pueden rechazar un peso antes del límite
externo. Las opciones `--max-synthesis-total-bits` y `--max-synthesis-recursion`
solo configuran la fase de síntesis.

La inspección por defecto valida cabecera/catálogo e indica `payload: not-verified`.
`--verify-all` recorre hashes y gramática de todos los bloques; solo entonces
imprime `payload: verified-all`. Esto acredita integridad, no equivalencia
matemática con una fuente GGUF arbitraria. Se puede inspeccionar sin la fuente.

### Proyecciones simbólicas desde pesos preparados

`bml_runtime::prepared_projection::PreparedProjection::load` toma un
`WeightReader`, un `TensorId` y `MatrixLimits`. Carga un tensor de rango 2:
GGML `[columnas, filas]` pasa a matriz row-major `[filas, columnas]` sin cambiar
el orden plano. Valida cantidad y bits agregados antes de reservar la matriz;
los hashes de cada bloque se comprueban antes de entregar sus términos.

`apply(&activations, limits)` calcula `weights × activations`, donde las
activaciones BML tienen forma `[columnas, batch]`. El resultado conserva términos
BML; no hay evaluación de pesos a floats ni resíntesis de los pesos cargados.
La ejecución usa reconocimiento/certificación de expresiones y operaciones
matriciales exactas acotadas: puede fallar aunque el contenedor sea estructuralmente
válido. `ProjectionError` conserva la causa de almacenamiento o el error matricial
localizado, sin entregar salida parcial ni sustituirla por aritmética IEEE.

La proyección posee los árboles y puede reutilizarse sin el archivo abierto.
El fixture independiente verifica una matriz `[2,3]` con resultado exacto `[5,7]`,
incluso cargando con cero pasos permitidos de síntesis. Esta API de proyección
mantiene resultados como árboles BML.

### Inferencia causal desde BMLGGUF: frontera numérica explícita

`SymbolicCausalModel::open_bmlgguf(path, read_limits, max_evaluated_weights,
WeightEvaluation::CheckedExactF32, symbolic_limits)` carga el tokenizer y la
configuración del contenedor y valida el mismo perfil Llama que la entrada GGUF.
El llamador debe seleccionar expresamente la política numérica: cada árbol se
reconoce/certifica, se evalúa a F64 y se convierte a F32 **solo si el racional
exacto de ese F32 coincide con el objetivo certificado del árbol**. Se rechazan
errores de reconocimiento, dominio, rango o pérdida de exactitud con tensor/índice.
No hay síntesis de pesos ni fallback. `max_evaluated_weights` limita la expansión
del modelo; esta API materializa los pesos y no promete inferencia streaming.

El perfil reportado es `bmlgguf-checked-exact-f32-symbolic-causal-v1`. Las
activaciones se ejecutan mediante los kernels RPN BML del motor causal existente,
con KV F32; sus logits son resultados numéricos, no racionales exactos. La carga
de pesos exactos no acredita exactitud matemática de todas las activaciones.

El fixture independiente de una capa con GQA, RoPE y MLP no nulo se convierte,
se borra su GGUF y se abre desde BMLGGUF. La prueba contrasta los IDs del tokenizer,
logits completos y dos pasos de generación greedy con un oráculo escalar externo
a los kernels de producción (tolerancia absoluta `2e-7`). Se conserva el espacio
inicial explícito al decodificar el token SentencePiece `▁`.
La evidencia está en `quality/evidence/model-weights-causal.json`; todavía no
acredita un modelo real.

El CLI exige seleccionar esta frontera numérica y su límite de materialización:

```sh
cargo run -p bml-cli --locked -- run -m modelo.bmlgguf -p "AB" -n 2 --temp 0 \
  --weight-evaluation checked-exact-f32 --max-evaluated-weights 10000
```

El número `10000` es un ejemplo para fixtures pequeños, no un presupuesto medido
para modelos reales. La identificación del contenedor usa su magic, incluso si
su extensión es distinta. Sin las dos opciones explícitas, un BMLGGUF se rechaza;
no se cambia automáticamente de perfil. Las opciones son incompatibles con
`--bml` y `--weights`, que seleccionan otras rutas. El comando imprime la política,
el presupuesto solicitado y el perfil causal antes de generar.
La carga conserva los límites predeterminados de `ReadLimits`; un fallo de
materialización o integridad termina antes de generar y no modifica el modelo.
La prueba CLI convierte un fixture, borra el GGUF y genera desde el contenedor.

### Campaña de conversión con modelos reales (2026-11-03)

Se ejecutó el CLI **release**, toolchain `1.98.1`, sobre dos archivos locales
identificados por SHA-256. Cada fila es un intento, no una media estadística:

| Fuente | Bytes de entrada | Tiempo de intento | RSS máximo (KiB) | Resultado localizado |
|---|---:|---:|---:|---|
| Stories15M F32 | 98.357.920 | 5,912 s | 112.492 | `token_embd.weight[32203]`: límite de derivación exacta |
| Stories15M Q4_0 | 14.469.344 | 2,482 s | 30.372 | `output.weight[75234]`: replay requiere 1.000.006 pasos, límite 1.000.000 |

Hashes de fuente:

- F32: `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`.
- Q4_0: `da8239d4b6175eac7db338d794a6e35790e36647c76d1d4ef1d2d0112a220dc1`.

Comando por fuente, con un destino nuevo:

```sh
target/release/bml-cli convert-weights -m FUENTE.gguf -o DESTINO.bmlgguf \
  --max-synthesis-steps-per-weight 4096 --max-synthesis-steps-total 1000000 \
  --max-weight-bits 1048576 --max-block-terms 1024 --max-block-bytes 8388608 \
  --max-output-bytes 1073741824 --max-resident-bytes 268435456 \
  --max-cache-bytes 1048576
```

Se midieron tiempo de pared con `time.monotonic` y RSS/CPU por proceso hijo con
`os.wait4` (Linux, `ru_maxrss` en KiB). La compilación y el hash de fuente quedan
fuera del tiempo; el hash previo puede calentar la caché de archivos. RSS incluye
páginas mapeadas y no equivale al presupuesto de memoria propia contabilizada.

**Ambos intentos fallaron sin publicar destino**, sin temporales restantes y con
hash de fuente intacto. No hay tamaño final de modelo ni mediciones de carga o
inferencia: esas fases no se ejecutaron. El error F32 no identifica qué límite
interno de derivación se agotó, por lo que no se atribuye a un campo concreto.
Estos resultados documentan el alcance bajo esos presupuestos; no prueban
imposibilidad con otros límites ni permiten afirmar compresión o rendimiento.
Comandos, hash del binario, diagnósticos completos y hashes de logs están en
[`model-weights-real-campaign.json`](../quality/evidence/model-weights-real-campaign.json).

### Medición por fases de un fixture exitoso

La campaña reproducible usa el fixture independiente Llama de una capa, 12
tensores y 2.188 pesos. Su GGUF tiene 16.000 bytes (SHA-256
`60b0498ae693893d3122c64a618a48c573d7d27e9edecedec36747044e2c7145`);
el BMLGGUF publicado ocupa 13.029 bytes (SHA-256
`73fb23a1c4f16dd285c7c5bdc16bd58c964cca4a415b6520bd79032fd836e85f`).
La fuente se elimina antes de cargar y ejecutar.

```sh
cargo build -p bml-cli --release --locked
cargo build -p bml-runtime --example measure_prepared_fixture --release --locked
python3 tools/measure_prepared_fixture.py --directory target/quality/fixture-phases-NUEVO
```

El directorio debe ser nuevo; conserva modelo, logs y `report.json`. La compilación
no forma parte de las mediciones. Resultado observado (una muestra por fase):

| Proceso | Tiempo de pared total | Temporizador interno | RSS máximo del proceso |
|---|---:|---:|---:|
| Conversión | 10,815 ms | lectura/decoding 0,453 ms; síntesis 0,234 ms; escritura/verificación 6,305 ms | 18.892 KiB |
| Carga sola | 27,823 ms | carga 26,515 ms | 18.892 KiB |
| Ejecución | 34,869 ms | carga 15,510 ms; tokenización 0,002 ms; ejecución 17,767 ms | 18.892 KiB |

La ejecución excluye carga y tokenización de su temporizador; incluye creación
de KV, compilación perezosa de kernels, prefill y dos muestras greedy (IDs `[1,0]`)
con el forward intermedio necesario. El perfil es
`bmlgguf-checked-exact-f32-symbolic-causal-v1`, con 10.000 pesos como cap explícito.
El convertidor contabilizó un pico propio de 14.367.008 bytes y 1.160 bytes de caché,
distintos del RSS del proceso. Se reutilizaron 2.184 pesos y se emitieron 13 pasos
de síntesis.

RSS se obtiene con `os.wait4` para toda la vida de cada hijo, incluido arranque;
en ejecución incluye también la carga previa. No es un pico aislado del kernel.
La caché del sistema no se controla y los temporizadores wall/CPU/internos tienen
alcances distintos. Los pesos simples y repetidos del fixture no acreditan
compresión ni rendimiento de modelos reales; tampoco convierten en éxitos los
fallos reales de la sección anterior. Evidencia y hashes de logs:
[`model-weights-fixture-phases.json`](../quality/evidence/model-weights-fixture-phases.json).

La revisión de contratos numéricos añade comparación racional certificada por
cada uno de los 330 pesos del fixture de cinco dtypes, normalización estructural
de ambos ceros y conversión exacta de subnormales mínimos/máximos con ambos signos
en F32/F16. También comprueba rechazo de NaN/inf en F16 y codecs cuantizados.
La exactitud de almacenamiento no exige que esos términos sean evaluables
directamente como floats. Orden rectangular, corrupción/truncamientos, límites
y limpieza transaccional están mapeados a pruebas ejecutadas en
[`model-weights-contract-review.json`](../quality/evidence/model-weights-contract-review.json).

## Uso de `bml-model`

Desde la raíz del repositorio, con el toolchain de `rust-toolchain.toml`:

```sh
cargo build -p bml-inference --bin bml-model --locked

target/debug/bml-model inspect --model target/bml-1.0/stories15M.gguf

target/debug/bml-model run \
  --model target/bml-1.0/stories15M.gguf \
  --prompt "Once upon a time" --predict 4 --context 128
```

`inspect` devuelve configuración, plan de operaciones y estimación de memoria
numérica para el contexto declarado por el modelo. `run` devuelve JSON con IDs
de entrada y salida, `eos_id`, bytes generados, texto y tiempos de prefill/decode.
Los tiempos de esa ejecución no constituyen una comparación de rendimiento.
El campo `bytes` conserva la salida exacta; `text` presenta UTF-8 con sustitución
si la generación termina en una secuencia de bytes incompleta.

`--tokens 1,9038,2501,263,931` sustituye a `--prompt` para ejecutar IDs explícitos.
El contexto debe admitir todos los tokens de entrada y la generación solicitada.
Por defecto se solicitan 16 tokens y se usa el menor valor entre 512 y el contexto
del modelo. `--predict 0` permite consultar el resultado del prefill.
`--logits` añade los vectores completos: uno tras el prefill y otro por cada token
generado, incluido el último. Su uso es diagnóstico y puede producir JSON grande.

Para guardar recursos normalizados y volver a usarlos:

```sh
target/debug/bml-model export \
  --model target/bml-1.0/stories15M.gguf \
  --output target/bml-1.0/stories15M.bmlres

target/debug/bml-model inspect --model target/bml-1.0/stories15M.bmlres
target/debug/bml-model run \
  --model target/bml-1.0/stories15M.bmlres \
  --prompt "Hello" --predict 4 --context 128
```

`export` y `compile` requieren una entrada GGUF y una ruta de salida que todavía
no exista. El contenedor normalizado incluye configuración, tokenizer, plan
tipado, pesos, límites y metadatos de procedencia e integridad.

## Compilación y ejecución autónoma

```sh
target/debug/bml-model compile \
  --model target/bml-1.0/stories15M.gguf \
  --output target/bml-1.0/stories15M-bml-final \
  > target/bml-1.0/aot-final-build.json

target/bml-1.0/stories15M-bml-final \
  --prompt "Once upon a time" --predict 4 --context 128
```

La compilación genera una función nativa específica del modelo: inicio del token,
secuencia explícita de capas y proyección final. Rust la enlaza con los kernels
y se incorporan los recursos al ELF. El plan y el SHA-256 del recurso completo
quedan vinculados al código generado; el cargador comprueba su correspondencia,
los rangos y la integridad antes de ejecutar.

**Durante la compilación** hacen falta esta distribución de fuentes, el toolchain
Rust fijado, enlazador y dependencias Cargo en caché. La compilación generada usa
Cargo offline; la primera preparación de esa caché puede requerir red. Los
archivos intermedios se guardan bajo `target/bml-aot`.

**Durante la ejecución** basta el ejecutable y un sistema Linux amd64 GNU
compatible con su cargador dinámico, glibc y bibliotecas del sistema enlazadas.
El programa usa `/proc/self/exe`, por lo que necesita procfs accesible. No necesita
el GGUF original, los archivos `.bmlres`, las fuentes BML, Cargo, Rust, llama.cpp
ni conexión de red. Es un ejecutable con recursos propios; no es un ELF estático
independiente de la ABI del sistema.

El artefacto 1.0.0 pasó la ejecución aislada registrada en
`target/bml-1.0/standalone-final/report.json`: una raíz mínima contenía
únicamente el ejecutable y las bibliotecas declaradas, con procfs y namespaces
de usuario, montaje, PID y red. La traza no registró llamadas de red, procesos
hijos, modificaciones del sistema de archivos ni accesos exitosos inesperados.
Para `Hello`, cuatro tokens y contexto 128, coincidieron los IDs, bytes y texto
con la referencia; los 160 000 logits cumplieron la tolerancia declarada.

Para reproducirlo con el artefacto generado y de procedencia conocida:

```sh
python3 tools/validate_standalone.py \
  --artifact target/bml-1.0/stories15M-bml-final \
  --reference-json target/bml-1.0/parity-final/case-01-reference.json \
  --output target/bml-1.0/standalone-repeat \
  --trusted-generated-artifact \
  --prompt Hello --predict 4 --context 128
```

El verificador requiere `readelf`, `ldd`, `strace`, `unshare`, `chroot` y permisos
para esos namespaces. Conserva la traza y compara la salida completa. Una falta
de permisos o herramientas hace fallar la prueba. La evidencia corresponde al
hash final registrado; debe repetirse si cambia el ejecutable.

## Límites de memoria

Cada sesión aplica por defecto un presupuesto de **512 MiB** para los payloads
numéricos de KV y memoria temporal. La API
`Model::session_with_memory_limit(context, maximum_bytes)` permite fijarlo
explícitamente. La comprobación precede a la reserva de la sesión. El presupuesto
no representa una cota del RSS del proceso: no incluye pesos mapeados, tokenizer,
metadatos, overhead del asignador ni la acumulación de diagnósticos JSON.

El CLI y el artefacto aplican además un límite independiente de **64 MiB** para
los logits retenidos por `--logits`. Antes de generar se comprueba, con aritmética
sin desbordamiento, `(predict + 1) × vocabulario × sizeof(f32)`; una petición que
supere la cota se rechaza, aunque pudiera terminar antes por EOS. Esta cota limita
el payload numérico retenido, no el tamaño del JSON serializado.
Un fallo de reserva se comunica como error, pero la sobreasignación de memoria
del sistema operativo impide prometer disponibilidad física a partir de una
reserva satisfactoria.

## Reproducir la comparación

El modelo se suministra localmente; las pruebas ordinarias no lo descargan.
El comparador necesita el checkout limpio de llama.cpp fijado arriba en
`target/bml-1.0/llama.cpp`, sus bibliotecas estáticas ya compiladas y un compilador
C++17. La configuración registrada usa Release, CPU, un hilo, KV F32, sin flash
attention ni repacking; `GGML_NATIVE`, OpenMP, BLAS y backends GPU están apagados.

Para preparar esa referencia en un checkout nuevo, con Git, CMake y un compilador
C++17 instalados (la descarga requiere red):

```sh
git clone https://github.com/ggml-org/llama.cpp.git target/bml-1.0/llama.cpp
git -C target/bml-1.0/llama.cpp checkout 785322d17f2d51e5185cf996a1f4e46d87640841
cmake -S target/bml-1.0/llama.cpp -B target/bml-1.0/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CPU=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BLAS=OFF \
  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF
cmake --build target/bml-1.0/llama.cpp/build --target llama --parallel 2
```

La identidad del modelo se comprueba por el SHA-256 declarado, no por el nombre
del archivo. La consulta de procedencia conservada identifica
`ggml-org/tiny-llamas`, revisión `99dd1a73db5a37100bd4ae633f4cfce6560e1567`,
archivo `stories15M.gguf`. El usuario suministra ese modelo localmente.

```sh
python3 tools/validate_inference.py \
  --model target/bml-1.0/stories15M.gguf \
  --expected-model-sha256 61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a \
  --bml target/debug/bml-model \
  --build-reference \
  --artifact target/bml-1.0/stories15M-bml-final \
  --output target/bml-1.0/parity-final
```

`--build-reference` compila el pequeño oráculo contra esas bibliotecas locales y
registra compilador, revisión y hashes. No descarga ni compila llama.cpp. Sin
`--model` ni `BML_TEST_GGUF`, el script informa `not_run` y termina con código 77.

El corpus fija cuatro prompts con cuatro tokens generados y cinco casos de
tokenización con espacios, controles, Unicode y texto parecido a tokens
especiales, todos con contexto 128. Se comparan IDs, bytes, texto y todos los
logits, con tolerancia fija:

```text
abs(bml - referencia) <= 1e-4 + 1e-4 * abs(referencia)
```

El reporte completo conserva identidad del modelo, referencia, binarios y
fuentes; el comparador rechaza entradas incompletas, no finitas o modificadas
durante la prueba. La [evidencia del candidato](../quality/evidence/bml-1.0-candidate.json)
registra nueve casos aprobados por motor para 1.0.0, 800 000 logits comparados por
motor y un error absoluto máximo de `1.28905884e-5`, sin cambios de hashes durante
la ejecución. Los IDs, bytes y texto coinciden exactamente. Los logits completos
quedan en el reporte local, fuera del repositorio.
La paridad numérica no establece superioridad en velocidad, memoria o energía
frente a llama.cpp. Eso requiere mediciones equivalentes y por separado.

El artefacto final `stories15M-bml-final` ocupa 98 932 768 bytes y tiene SHA-256
`74bef3b5a510b58fa0fbadc676133c65f427f5b4791754801bbd6b80d3261579`.
Su reporte de compilación es `target/bml-1.0/aot-final-build.json`; el corpus
completo está en `target/bml-1.0/parity-final/report.json`.

## Medición del artefacto 1.0.0

Estas mediciones corresponden a `stories15M-bml-final`, con el hash declarado
arriba. Se ejecutaron el 2 de noviembre de 2026, después de comprobar que no
quedaban compilaciones ni otras campañas de validación activas en ese equipo. El reporte completo es
`target/bml-1.0/benchmark-final/report.json`, SHA-256
`3514d023d22041bb51d0ac403ffc5bac79dc7459f71e5028b751e02574e617cb`.
La medición histórica v3 se conserva en `target/bml-1.0/benchmark-v3/report.json`
y en el registro de evidencia del commit `53ba3fe`.

Se midieron `Once upon a time` y `Hello`, con contexto 128 y 16 tokens solicitados,
greedy, un hilo por motor y sin `--logits`, sobre Intel Xeon (Cascadelake). Cada
prompt y motor tuvo un calentamiento y tres repeticiones, siempre en procesos
nuevos y con orden alternado. Tokens de entrada, tokens generados, texto y bytes
coincidieron en todas las ejecuciones. La caché del sistema operativo no se vació
ni se controló; estos resultados no representan arranques con caché fría.

| Prompt | Motor | Prefill medio (ms) | Decode medio (ms) | Decode medio (tokens/s) | Proceso completo medio (ms) | RSS máximo medio (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Once upon a time | BML 1.0.0 | 107.35 | 344.01 | 46.51 | 1926.91 | 105.33 |
| Once upon a time | llama.cpp | 12.09 | 143.77 | 111.30 | 225.67 | 112.26 |
| Hello | BML 1.0.0 | 44.34 | 352.24 | 45.43 | 1952.50 | 105.36 |
| Hello | llama.cpp | 11.04 | 148.79 | 107.70 | 232.73 | 111.61 |

Los tiempos internos excluyen arranque, carga y comprobaciones de integridad;
el tiempo del proceso completo los incluye. El RSS es el máximo de cada proceso,
obtenido con `wait4(pid)` porque GNU `time` no estaba instalado; no es la suma
acumulada de los hijos. Un MiB equivale a 1024 KiB. El tiempo de pared usa
`perf_counter`, con observación de salida a intervalos de hasta 1 ms más el
retardo del planificador. Mínimos, máximos, desviación y hashes están en la
evidencia compacta y las muestras completas en el reporte local.

Estas compilaciones tienen ajustes de CPU diferentes: BML usa `target-cpu=x86-64`,
y las bibliotecas llama.cpp habilitan SSE4.2, AVX, AVX2, F16C, FMA y BMI2, aunque
`GGML_NATIVE=OFF`. La tabla describe estos binarios en este equipo; no aísla el
efecto del algoritmo, la precisión o las instrucciones vectoriales. En esta
medición llama.cpp produjo más tokens por segundo y BML tuvo menor RSS máximo.
No se extrapola ese resultado a otros modelos, contextos o equipos.

El ejecutable final ocupa 98 932 768 bytes (94.35 MiB), incluidos los recursos; el
GGUF ocupa 98 357 920 bytes. El registro de compilación identifica Rust 1.98.1.
Para repetir la medición con una nueva carpeta de resultados:

```sh
python3 tools/benchmark_inference.py \
  --model target/bml-1.0/stories15M.gguf \
  --expected-model-sha256 61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a \
  --artifact target/bml-1.0/stories15M-bml-final \
  --artifact-build-report target/bml-1.0/aot-final-build.json \
  --output target/bml-1.0/benchmark-repeat
```

El reporte de compilación debe identificar el hash del artefacto, el del GGUF,
el target, `rustc_version` y `elapsed_ms`. El script rechaza salidas divergentes,
tiempos inválidos y archivos modificados durante la medición.

## Reproducir los controles de entrega

La base del delta 1.0 es `deec3aa8d2516ae19e335a5af77e6c38f8a854d8`.
Se usa ese commit fijo porque `origin/master` ya incluye el candidato y dejaría
de representar la base anterior a estos cambios. Los comandos requieren el
toolchain de `rust-toolchain.toml`, las herramientas fijadas en los workflows y
un entorno que permita los sockets locales usados por las pruebas ordinarias.
Estas pruebas no necesitan descargar un modelo.

```sh
BML_RELEASE_BASE=deec3aa8d2516ae19e335a5af77e6c38f8a854d8
./scripts/check.sh "$BML_RELEASE_BASE"
python3 tools/mutation_diff.py "$BML_RELEASE_BASE" target/quality/release-changes.diff
cargo mutants --workspace --cargo-arg=--locked --test-workspace true \
  --jobs 2 --timeout 30 --build-timeout 180 \
  --in-diff target/quality/release-changes.diff \
  --output target/bml-1.0/mutation-repeat
python3 tools/quality.py mutations \
  --report target/bml-1.0/mutation-repeat/mutants.out/outcomes.json
cargo llvm-cov --workspace --locked --json --output-path target/quality/coverage.json
python3 tools/quality.py coverage --report target/quality/coverage.json
cargo audit --deny warnings
```

La campaña de mutación debe terminar y cubrir su inventario completo; un reporte
parcial no acredita el control. Los criterios, las variantes de features y los
límites de cobertura se describen en [los controles de calidad](QUALITY.md).
La entrega sigue abierta mientras falte esa evidencia final.

## Alcance de la entrega

`bml-model` y el artefacto descrito aquí son las entradas del nuevo motor.
Los CLI antiguos, el compilador experimental, el servidor HTTP y los ejemplos
históricos siguen teniendo sus propios límites; no heredan esta validación.
GPU, otras arquitecturas, tokenizers diferentes y otras plataformas tampoco
quedan acreditados por el corpus actual.

El cierre requiere completar los controles de calidad y cerrar las tareas
OpenSpec, manteniendo la evidencia vinculada al artefacto entregado.
Hasta entonces, este documento describe un **candidato**, no una release 1.0
publicada o terminada.
# Sesiones de ejecución por etapas

`Model::layer_session(range, context, maximum_bytes)` crea una sesión causal
para un rango fijo y no vacío de capas. El coordinador obtiene la entrada con
`embed_token`, pasa el hidden state por `LayerSession::forward(position, hidden)`
y aplica `final_logits` al terminar todas las capas. La posición debe coincidir
con el siguiente token esperado; errores de forma, posición o aritmética no
avanzan la sesión. Las sesiones son independientes y tienen reset explícito.

La implementación actual conserva el modelo y la reserva KV completos bajo el
presupuesto indicado. El commit ocurre por etapa, sin transacción distribuida.
Las pruebas de `model_contracts` comparan logits y recuperación tras fallos;
la reducción de memoria y el rendimiento todavía no están acreditados.

`bml_inference::remote::{LayerClient, serve_layer_connection, RemoteLimits}`
permite ejecutar estas etapas por TCP con framing acotado y sesiones dedicadas
por conexión. Las pruebas con dos workers locales comparan logits completos.
La apertura exige `Model::fingerprint()` y el worker verifica la identidad
numérica antes de reservar KV. SHA-256 cubre configuración, nombres, dimensiones,
encodings y bytes de tensores en orden canónico; excluye el tokenizer y el layout
del archivo. Se calcula una vez por modelo. Detecta errores de despliegue; no
autentica al servidor. No hay reconexión automática ni commit transaccional entre workers. La integración con
el servidor HTTP sigue pendiente. El proceso `bml-layer-worker` permite arrancar
una etapa sobre un GGUF local:

```sh
cargo run -p bml-inference --bin bml-layer-worker --release --locked -- \
  --model model.gguf --layers 0:1 --listen 127.0.0.1:9998 --max-sessions 4
```

El extremo final del rango es exclusivo. El worker imprime una línea JSON de
readiness con dirección efectiva, rango e identidad del modelo; `--listen
127.0.0.1:0` selecciona un puerto libre. Contexto, memoria por sesión, frames y
timeout son configurables (`--help`); al saturarse cierra la nueva conexión sin
reservar KV. `--once` atiende una conexión y termina para pruebas/controladores.
La prueba de proceso usa un GGUF sintético y compara tres tokens con la ruta
local; no descarga modelos ni acredita rendimiento de despliegue.


`RemotePipeline::connect(model, addresses, context, limits)` coordina una sesión
completa sobre endpoints ordenados. Verifica identidad, ancho y cobertura exacta
de capas; `prefill`, `decode`, `position`, `reset` y `close` gestionan el ciclo de
vida. IDs inválidos y desbordamiento de contexto se rechazan antes de enviar
trabajo. Un fallo de etapa/proyección invalida todo el pipeline, pues otros
workers pueden haber confirmado el token. Se debe abrir una sesión nueva; no
hay rollback distribuido. El coordinador conserva el modelo local para embedding
y proyección final, por lo que esto aún no acredita partición de memoria.

El servidor admite la ruta remota con:

```sh
cargo run -p bml-server --release --locked -- -m model.gguf --port 8080 \
  --layer-workers 127.0.0.1:9998,127.0.0.1:9999
```

Los workers deben cubrir todas las capas en el orden indicado y usar el mismo
modelo. El scheduler acota admisión; cada petición abre sesiones remotas propias.
`/v1/completions` devuelve texto o SSE según `stream`, conserva caracteres UTF-8
que atraviesan límites entre tokens y termina el stream con `[DONE]`. La prueba
del handler compara ambas respuestas con generación local sobre un GGUF pequeño.
Un fallo remoto devuelve 502 en modo completo o un evento de error en un stream
ya abierto. El modo remoto vuelve a validar el modelo del coordinador por
solicitud y ejecuta los requests del batch secuencialmente; no acredita fusión
numérica de batches, paralelismo entre requests ni mejora de throughput.

Para réplicas completas, repetir `--layer-workers` una vez por pipeline. Cada
ocurrencia contiene la lista ordenada de etapas de una réplica; el scheduler
reparte batches completos round-robin entre esas listas. `/health` expone
`dispatched_batches` además de solicitudes pendientes. La prueba con dos réplicas
verifica destinos distintos y respuestas JSON/SSE iguales a la ruta local.
Las solicitudes de cada batch siguen ejecutándose secuencialmente; no se afirma
paralelismo de réplicas ni vectorización conjunta por añadir este reparto.
