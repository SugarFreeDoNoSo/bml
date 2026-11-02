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

Los recursos admiten F32, F16, Q4_0 y Q8_0; los pesos cuantizados conservan su
codificación y se decodifican por filas. Esa capacidad no extiende la evidencia
de paridad de Stories15M F32 a modelos cuantizados ni a otros modelos. Las
activaciones y el KV son F32, con acumulaciones escalares F64. Las arquitecturas,
variantes de RoPE y roles tensoriales ajenos al perfil aceptado producen errores.

La fachada `bml` conserva el escalar `Bml` y las capacidades tensor/GGUF optativas.
Los contratos son distintos: `Bml` y los términos simbólicos comprueban dominio
y rango; los kernels tensoriales nativos usan aritmética IEEE. La inferencia no
convierte cada peso en un término simbólico ni acredita una aceleración debida
al operador BML. Véanse [el escalar](architecture/BML_NUMBER.md),
[los tensores](architecture/TENSOR_LIBRARY.md) y [los controles](QUALITY.md).

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
  --output target/bml-1.0/stories15M-bml-final

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

## Medición del candidato v3

Estas mediciones corresponden al artefacto histórico v3. El ejecutable final
1.0.0 tiene otra identidad y su paridad y aislamiento se verificaron por separado;
la tabla no constituye una medición de rendimiento del ejecutable final.

Se midieron `Once upon a time` y `Hello`, con contexto 128 y 16 tokens solicitados,
greedy, un hilo por motor y sin `--logits`, sobre Intel Xeon (Cascadelake). Cada
prompt y motor tuvo un calentamiento y tres repeticiones, siempre en procesos
nuevos y con orden alternado. Tokens de entrada, tokens generados, texto y bytes
coincidieron en todas las ejecuciones. La caché del sistema operativo no se vació
ni se controló; estos resultados no representan arranques con caché fría.

| Prompt | Motor | Prefill medio (ms) | Decode medio (ms) | Decode medio (tokens/s) | Proceso completo medio (ms) | RSS máximo medio (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Once upon a time | BML v3 | 109.64 | 351.67 | 45.50 | 1975.20 | 105.37 |
| Once upon a time | llama.cpp | 12.64 | 148.97 | 107.51 | 232.19 | 113.65 |
| Hello | BML v3 | 43.85 | 369.57 | 43.37 | 1935.90 | 105.28 |
| Hello | llama.cpp | 11.81 | 153.65 | 104.18 | 236.57 | 113.53 |

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

El ejecutable v3 ocupa 98 932 768 bytes (94.35 MiB), incluidos los recursos; el
GGUF ocupa 98 357 920 bytes. Su compilación registrada duró 63.21 s con Rust 1.98.1.
Para repetir la medición con una nueva carpeta de resultados:

```sh
python3 tools/benchmark_inference.py \
  --model target/bml-1.0/stories15M.gguf \
  --expected-model-sha256 61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a \
  --artifact target/bml-1.0/stories15M-bml-v3 \
  --artifact-build-report target/bml-1.0/aot-v3-build.json \
  --output target/bml-1.0/benchmark-repeat
```

El reporte de compilación debe identificar el hash del artefacto, el del GGUF,
el target, `rustc_version` y `elapsed_ms`. El script rechaza salidas divergentes,
tiempos inválidos y archivos modificados durante la medición.

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
