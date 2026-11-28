# Fragmentos ELF nativos: staging y validación local

Esta receta prepara ejecutables Linux AMD64 independientes para rangos de filas
de la cabeza F32. Cada worker incorpora sólo sus filas. El coordinador conserva
el modelo normalizado completo, incluida una copia de la cabeza: este primer
incremento no reparte toda la memoria del modelo ni todas sus subcapas.

La compilación recibe un GGUF existente; la ejecución de los ELF no debe recibir
ese archivo. Las imágenes no contienen modelos. El control monolítico y el
coordinador fragmentado usan el mismo contrato escalar nativo, sin FMA ni AVX
empaquetado. Los resultados no acreditan BMLX simbólico, dos hosts o Lambda.
La tarea `bml-1-0-release/4.6` y la parte uno siguen abiertas hasta sus criterios
completos. El benchmark Docker/AVX conserva su seguimiento independiente.

## Estado de esta receta

El executor tenía Docker 28.4.0 y ninguna imagen local. **No se ejecutó aquí un
build Docker ni se descargaron imágenes o modelos para esta receta.** Los
digests de las bases Rust 1.98.1 y Python 3.12.12 proceden del Dockerfile GGUF ya
versionado. La verificación de scripts o de ELF fuera de Docker no sustituye la
ejecución de esta receta. Registrar los IDs de las imágenes y resultados cuando
se ejecute en un entorno autorizado.

## Preparar imágenes y carpetas

Desde la raíz del checkout exacto revisado, construir sólo si están disponibles
las bases y el caché requeridos o hay autorización de acceso normal para su
obtención. No eludir accesos denegados. Estos comandos no descargan el GGUF.

```sh
docker build --platform=linux/amd64 --target compiler \
  -f docker/elf-fragments/Dockerfile \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --build-arg SOURCE_DIRTY=true -t bml-elf-fragment-compiler:local .
docker build --platform=linux/amd64 --target runtime \
  -f docker/elf-fragments/Dockerfile \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --build-arg SOURCE_DIRTY=true -t bml-elf-fragment-runtime:local .
docker image inspect bml-elf-fragment-compiler:local bml-elf-fragment-runtime:local \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}'
```

`SOURCE_DIRTY=true` describe un parche local sin commit; ajustar sólo con evidencia
del árbol. Conservar además el SHA-256 del parche y el `Cargo.lock`. El contexto
usa el archivo `Dockerfile.dockerignore` propio: incluye fuentes permitidas y
excluye GGUF, targets, Git y `.env`. No se monta ningún secreto en ejecución.

Escoger carpetas nuevas y un modelo ya autorizado; sustituir las rutas siguientes:

```sh
model_dir=/ruta/a/modelos
model_name=stories15M-f32.gguf
run_root=/ruta/a/una/carpeta/nueva
mkdir -p "$run_root/results" "$run_root/work" "$run_root/inputs"
cat > "$run_root/inputs/cases.json" <<'JSON'
[
  {"name":"hello","prompt":"Hello","predict":4,"context":32},
  {"name":"hi","prompt":"Hi","predict":4,"context":32}
]
JSON
```

Para Stories15M F32 se espera el archivo de 98.357.920 bytes con SHA-256
`61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`.
Fijar también su revisión de origen en la evidencia. Un fixture sintético requiere
su propio hash, límites de contexto y etiqueta; no cuenta como prueba del modelo
entrenado. Los casos pueden usar `tokens: [1, ...]` en lugar de `prompt`, nunca
ambos; conservar los IDs efectivos que emite el verificador.

## Compilar con el GGUF de sólo lectura

La imagen de compilación conserva fuente y caché Cargo. Se copia ese caché a un
directorio de trabajo nuevo para permitir locks sin escribir en la imagen ni
depender del Cargo del host. El target y los resultados son exclusivos de esta
corrida. El sub-build AOT usa el Rust fijado y modo offline; no habilitar red para
resolver fallos en esta fase.

```sh
docker run --rm --network=none --read-only \
  --memory=4g --memory-swap=4g --cpus=2 --pids-limit=128 \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  --mount "type=bind,src=$model_dir,dst=/models,readonly" \
  --mount "type=bind,src=$run_root/results,dst=/results" \
  --mount "type=bind,src=$run_root/work,dst=/work" \
  --env CARGO_HOME=/work/cargo --entrypoint /bin/sh \
  bml-elf-fragment-compiler:local -c \
  'test ! -e /work/cargo && cp -R /usr/local/cargo /work/cargo && exec /src/target/x86_64-unknown-linux-gnu/release/bml-model "$@"' \
  -- compile-fragments --model "/models/$model_name" \
  --output /results/bundle --target /work/target --workers 2
```

`results/bundle` debe no existir. Conservar fallos y staging si el build no
termina; no etiquetar ese intento como validado. Los límites anteriores son
contención del experimento, no promesa de suficiencia para cualquier modelo.

## Obtener la referencia canónica

Sólo esta fase de referencia vuelve a montar el GGUF. Registrar el hash del
binario `bml-model` de la imagen y los del modelo y `build-report.json`:

```sh
bml_sha=$(docker run --rm --network=none --read-only \
  --entrypoint sha256sum bml-elf-fragment-runtime:local /opt/bml/bml-model | cut -d ' ' -f 1)
model_sha=$(sha256sum "$model_dir/$model_name" | cut -d ' ' -f 1)
build_sha=$(sha256sum "$run_root/results/bundle/build-report.json" | cut -d ' ' -f 1)
docker run --rm --network=none --read-only \
  --memory=4g --memory-swap=4g --cpus=2 --pids-limit=128 \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=256m \
  --mount "type=bind,src=$model_dir,dst=/models,readonly" \
  --mount "type=bind,src=$run_root/inputs,dst=/inputs,readonly" \
  --mount "type=bind,src=$run_root/results,dst=/results" \
  bml-elf-fragment-runtime:local prepare-reference \
  --bml-model /opt/bml/bml-model --bml-model-sha256 "$bml_sha" \
  --model "/models/$model_name" --model-sha256 "$model_sha" \
  --cases /inputs/cases.json --output /results/reference.json
ref_sha=$(sha256sum "$run_root/results/reference.json" | cut -d ' ' -f 1)
```

## Ejecutar sin el GGUF ni las fuentes

La última fase monta solamente `bundle/run`, su informe de build y la referencia.
No montar `bundle/build`, fuentes, cachés, toolchain ni el directorio de modelos.
El filesystem de la imagen runtime contiene Python y bibliotecas del sistema
para el verificador, pero ningún compilador ni GGUF. El verificador comprueba
identidad/ELF AMD64 antes de ejecutar; no invoca `ldd`.

```sh
mkdir -p "$run_root/runtime-results"
docker run --rm --network=none --read-only \
  --memory=4g --memory-swap=4g --cpus=2 --pids-limit=128 \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --mount "type=bind,src=$run_root/results/bundle/run,dst=/bundle/run,readonly" \
  --mount "type=bind,src=$run_root/results/bundle/build-report.json,dst=/bundle/build-report.json,readonly" \
  --mount "type=bind,src=$run_root/results/reference.json,dst=/inputs/reference.json,readonly" \
  --mount "type=bind,src=$run_root/runtime-results,dst=/results" \
  bml-elf-fragment-runtime:local validate \
  --bundle /bundle --build-report-sha256 "$build_sha" \
  --reference-json /inputs/reference.json --reference-sha256 "$ref_sha" \
  --output /results/validation --trusted-generated-artifacts
```

La salida de validación debe conservar comandos, hashes, IDs, todos los logits
comparados y fallos de controles negativos. Un timeout, error, falta de recurso
o diferencia numérica es fallo; no se relajan tolerancias. Una ejecución local
en Docker sigue siendo un único host. Las mediciones de rendimiento, el ensayo
entre hosts y cualquier despliegue requieren sus protocolos y autorización
separados; este paquete no ejecuta AWS ni cierra la parte uno.
