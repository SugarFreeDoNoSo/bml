# TinyLlama: dos textos predeclarados y dos posiciones

Este protocolo añade `Hello → [1,15043]` y `Hi → [1,6324]`, un texto por proceso.
El encoder Rust del GGUF debe confirmar esos IDs exactos y el contexto 2 antes
de cualquier forward. Un texto distinto, un tokenizado distinto o un outdir
existente abortan el caso. No hay truncado, continuación greedy ni sustitución
de textos al observar una violación de dominio.

El driver reside en el target `real_model_two_position_diagnostic`. El test
histórico fijo `[1,29958]` conserva su nombre y comportamiento. Los dos nombres
exactos nuevos son:

- `real_model_native_prompt_profile_preflight`: sólo nativo y observación de
  operandos; no ejecuta bytecode BML.
- `real_model_two_position_prompt_symbolic_logit_diagnostic`: nativo y BML,
  cachés independientes nuevas de capacidad 2 y cuatro vectores por texto.

## Construcción y procedencia

Root debe publicar el driver revisado y capturar la construcción actual en un
checkout limpio y target exclusivo. Antes de construir, conservar argv, env
whitelist, toolchain, manifiestos/lock y hashes de las fuentes. Retener stdout
JSONCargo/stderr/status originales, exigir compiler-artifact no fresco para el
driver y los paquetes numéricos, y copiar el ejecutable anunciado a una ruta
nueva inmutable con hashes antes/después. No reutilizar el target de replays.

```sh
cargo test --release -p bml-runtime --test real_model_two_position_diagnostic \
  --locked --no-run --message-format=json
```

`BML_TEST_SOURCE_REVISION` es el SHA completo del código capturado. El driver
lo conserva como etiqueta declarada; la captura externa de build/fuentes debe
autenticarla. El propio driver registra además el hash de su ejecutable.

El GGUF queda fijado a SHA256
`da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556`.
El AOT existente tiene exactamente 13 archivos, manifest
`0b6e9bc33693e9581676fad805fec2cf2173410c1ba112ddce99f490709bbd97`,
versión 4, contexto 2 y el perfil completo con RMS160/4, projection64/4,
down256/1.5, attention20, value16, RoPE32, gate18, up16, residual160,
scale256 y score16. Los trece hashes están fijados en el driver. El loader
autentica GGUF/programas; cargar y validar bytecode no lo ejecuta.

## Orden de ejecución por root

Los comandos siguientes son el protocolo para una ejecución posterior. Este
documento no acredita que hayan corrido. `MODEL`, `AOT` y `BINARY` son rutas
absolutas de los inputs fijados y la copia inmutable del binario; `REVISION` es
su SHA de código autenticado. Todos los outdirs deben ser nuevos.

```sh
BML_TEST_PROMPT=Hello BML_TEST_SOURCE_REVISION="$REVISION" \
  BML_TEST_GGUF="$MODEL" BML_TEST_SYMBOLIC_ARTIFACT="$AOT" \
  BML_TEST_LOGITS_DIR="$PREFLIGHT_HELLO" "$BINARY" \
  real_model_native_prompt_profile_preflight --ignored --exact --nocapture
BML_TEST_PROMPT=Hi BML_TEST_SOURCE_REVISION="$REVISION" \
  BML_TEST_GGUF="$MODEL" BML_TEST_SYMBOLIC_ARTIFACT="$AOT" \
  BML_TEST_LOGITS_DIR="$PREFLIGHT_HI" "$BINARY" \
  real_model_native_prompt_profile_preflight --ignored --exact --nocapture
```

Conservar y comprobar ambos `native-preflight-summary.json`, sus protocolos y
reportes por posición. Su status es `native_only_preflight_pass` y declara
`bml_executed=false`; **nunca califica una ejecución completa**. Ante una
violación, conservar el reporte fallido y no lanzar ninguno de los casos BML.
No subir cotas ni elegir otro texto bajo este protocolo.

Sólo después de ambos preflights completos, usar dos outdirs distintos:

```sh
BML_TEST_PROMPT=Hello BML_TEST_SOURCE_REVISION="$REVISION" \
  BML_TEST_GGUF="$MODEL" BML_TEST_SYMBOLIC_ARTIFACT="$AOT" \
  BML_TEST_LOGITS_DIR="$FULL_HELLO" "$BINARY" \
  real_model_two_position_prompt_symbolic_logit_diagnostic --ignored --exact --nocapture
BML_TEST_PROMPT=Hi BML_TEST_SOURCE_REVISION="$REVISION" \
  BML_TEST_GGUF="$MODEL" BML_TEST_SYMBOLIC_ARTIFACT="$AOT" \
  BML_TEST_LOGITS_DIR="$FULL_HI" "$BINARY" \
  real_model_two_position_prompt_symbolic_logit_diagnostic --ignored --exact --nocapture
```

El protocolo se escribe antes de los forwards. Cada vector se guarda al estar
disponible; el informe de posición completa exige ambos backends finitos y
KV1/2. `summary.json` sólo aparece tras ambas posiciones y las diez familias
BML realmente llamadas. `max_nodes=0` prohíbe construir kernels durante la
ejecución. Los guards BML siguen activos; un preflight nativo no garantiza que
los intermedios simbólicos permanezcan dentro del dominio.

## Qué comprueba el preflight

El callback público recorre todas las filas con el mismo fold F64 de la
proyección nativa, valida input/pesos por rol y mantiene rangos por tensor.
La reconstrucción de RMS y de ambos residuales por capa se contrasta en bits
con los inputs normalizados del callback; gate/up y la entrada real de down
tienen sus comprobaciones propias. RoPE observa todos los pares Q/K y los
parámetros reales de posición/base. Las K/V reconstruidas y redondeadas a F32
se comparan en bits con todas las filas KV nativas del prefijo.

La reconstrucción de dot/scale/softmax/weighted_sum usa Q/F64 y K,V/F32 nativos
como **entradas derivadas para los roles simbólicos**. La atención interna del
forward nativo usa Q/dot/probabilidades F32: no se sustituye esa atención ni se
afirma identidad entre trayectorias. Se validan todas las dimensiones y
contadores esperados; los resultados derivados no son una ejecución BML.

## Disco y memoria

Cada vector tiene 32.000 f64LE, 256.000 bytes. Los dos procesos completos guardan
ocho vectores: 2.048.000 bytes. Los preflights guardan otros cuatro vectores
nativos: 1.024.000 bytes. Reservar al menos 16 MiB para vectores, protocolos,
rangos/reportes y logs, más la copia del binario y el target de compilación
exclusivo. El AOT/GGUF existentes no se copian ni alteran.

Cada caché KV de capacidad 2 ocupa 90.112 bytes; el completo necesita dos y el
preflight una. Q/K/V observadas para dos posiciones añaden aproximadamente
1 MiB de elementos, además de Vec/JSON/allocator. Se decodifica una fila de
pesos cada vez; el GGUF permanece mapeado, sin expansión completa de matrices.
El RSS real incluye páginas del modelo, kernels, intérprete y allocator.

Root debe medir RSS/PSS y memoria disponible/cgroup antes de decidir concurrencia
en 4 GiB. El mmap compartido no demuestra por sí solo ahorro físico. Si falta
memoria, mantener los mismos dos casos y ejecutarlos secuencialmente. La lectura
de todas las filas y los guards nativos consumen trabajo real; “native-only”
significa ausencia de ejecución BML, no coste cero ni garantía de latencia.

Máximo/media/RMS/argmax son observaciones sin umbral global preestablecido. No
deducir paridad, rendimiento, generalidad para otros prompts ni cierre de todos
los compromisos de 8.6 a partir de estos dos casos. Salidas parciales se retienen
como diagnóstico y no cuentan como casos completos.
