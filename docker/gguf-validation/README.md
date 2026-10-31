# Validación GGUF portátil (CPU)

Este contenedor inspecciona un tensor **rank-2 Q4_0** usando `bml-gguf`,
compara su desquantización con la función original `dequantize_row_q4_0` de
llama.cpp y calcula una proyección acotada con activaciones sintéticas firmadas.
No ejecuta tokenizer, atención, capas ni generación. No valida inferencia de
TinyLlama/Gemma4 ni mide TPS. La proyección pertenece al probe de validación;
no certifica el matmul del compilador experimental. Una arquitectura como
`gemma4` puede inspeccionarse, pero se registra explícitamente como no soportada
para inferencia BML. Otros tipos de tensor producen un error, no un falso PASS.

## Construir

Desde la raíz del repositorio, con Docker/BuildKit y red para imágenes y crates:

```sh
docker build -f docker/gguf-validation/Dockerfile \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --build-arg SOURCE_DIRTY="$(test -z "$(git status --porcelain)" && echo false || echo true)" \
  -t bml-gguf-validation:local .
python3 docker/gguf-validation/smoke.py --image bml-gguf-validation:local
```

Bases y frontend Docker fijados por digest, Rust 1.98.1, Python 3.12.12,
Cargo.lock congelado. La referencia está vendorizada con licencia MIT,
commit y hashes en `reference/provenance.json`; no requiere bajar llama.cpp
ni modelos al ejecutar. El shim solo proporciona estructura Q4_0, conversión
IEEE binary16 y E/S; la función upstream se conserva literalmente.

En cloud con proxy CA, agregar `--secret id=proxy_ca,src="$CODEX_PROXY_CERT"`.
La CA se monta solo durante el build; no se copia a la imagen. Si el estado de
Buildx apunta a un directorio de solo lectura, se puede indicar
`BUILDX_CONFIG=/tmp/bml-buildx` sin modificar credenciales de Docker. La imagen
Rust ya fija el toolchain: no se copia el rust-toolchain.toml del workspace para
evitar que rustup instale componentes de desarrollo innecesarios durante build.
No desactivar TLS. No montar credenciales en ejecución.

## Usar un modelo existente

Linux/macOS con Docker en modo contenedores Linux (ajustar rutas):

```sh
mkdir -p results
docker run --rm --network=none --read-only \
  --memory=1g --memory-swap=1g --cpus=2 --pids-limit=64 \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=/ruta/modelos,dst=/models,readonly" \
  --mount "type=bind,src=$(pwd)/results,dst=/results" \
  bml-gguf-validation:local \
  --model /models/modelo.gguf --tensor token_embd.weight \
  --rows 2 --blocks 1 --timeout 60
```

PowerShell con Docker Desktop en modo Linux:

```powershell
New-Item -ItemType Directory -Force .\results | Out-Null
$results = (Resolve-Path .\results).Path
docker run --rm --network=none --read-only --memory=1g --memory-swap=1g --cpus=2 --pids-limit=64 --cap-drop=ALL --security-opt=no-new-privileges --mount "type=bind,src=C:\ruta\modelos,dst=/models,readonly" --mount "type=bind,src=$results,dst=/results" bml-gguf-validation:local --model /models/modelo.gguf --tensor token_embd.weight --rows 2 --blocks 1 --timeout 60
```

No requiere reconectar WSL. Docker debe tener acceso a la carpeta del modelo.
El ejemplo no descarga el GGUF y requiere que el tensor elegido sea Q4_0.
El archivo debe permanecer inmutable mientras está mapeado; readonly en el
contenedor no evita escrituras de otro proceso del host. Docker limita memoria
residente; el mmap reserva espacio virtual del tamaño del archivo sin expandir
todos los pesos. Metadatos y páginas visitadas sí consumen RAM. El límite de 1 GiB
es una configuración de contención, no una promesa de suficiencia para cualquier
archivo. OOM o interrupción externa pueden impedir escribir el resultado final.

`--row-start` elige la primera fila. Máximo 8 filas y 128 bloques de 32 elementos
por fila (4096 valores); por defecto 2 filas y 1 bloque. Cada subprocess tiene
un timeout 1..600 segundos. El muestreo toma un prefijo contiguo de cada fila;
no valida el resto del tensor. `--hash-model` opta explícitamente por leer todo
el archivo en streaming para calcular SHA256; sin esa opción solo se hashean
los bloques muestreados y **no** se verifica la identidad integral del modelo.

Resultados persistentes: `result.json`, `run.log` y `row-N.q4_0` con los bytes
muestreados. JSON incluye dimensiones, arquitectura, versión GGUF, tamaños,
hashes de muestras, referencia, build, comandos, tolerancias y proyecciones.
Activación: `x[i]=((i%17)-8)/8`; dot de referencia calculado con `math.fsum`
sobre valores de la función C upstream. Tolerancias `atol=1e-6`, `rtol=1e-6`.
Código de salida 0 significa que pasó la muestra; 2 significa error/fallo.
Usar una carpeta de resultados nueva por corrida. No hay GPU, inferencia ni
resultados WSL en este entregable. ARM/multiarch requiere build y prueba propios.

## Exportar e importar sin registry

```sh
docker save bml-gguf-validation:local | gzip > bml-gguf-validation-amd64.tar.gz
sha256sum bml-gguf-validation-amd64.tar.gz
# En la máquina destino:
gzip -dc bml-gguf-validation-amd64.tar.gz | docker load
```

Verificar el SHA256 del archivo recibido antes de importarlo. La exportación
contiene una imagen Linux/amd64 si esa fue la plataforma construida; no es una
imagen multiarch ni una VM. La identidad de imagen se comprueba con
`docker image inspect bml-gguf-validation:local --format '{{.Id}}'`.
