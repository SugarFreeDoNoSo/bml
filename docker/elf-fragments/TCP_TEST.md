# Diagnóstico TCP local de fragmentos ELF nativos

Esta segunda etapa conserva la entrada por stdin del prototipo anterior.
El transporte adicional es **`tcp-test` optativo y sólo loopback literal**.
No tiene TLS ni autenticación criptográfica de peers; la igualdad del descriptor
no identifica criptográficamente al proceso que atiende un puerto. No se debe
exponer a una red ni presentar como producción, dos hosts, Lambda o cierre de
`bml-1-0-release/4.6` y de la parte uno.

Los ejecutables deben reconstruirse desde la revisión TCP; los ELF anteriores
no implementan estos argumentos. Esta receta no construye imágenes, descarga
modelos ni usa AWS. Conserva el GGUF real fijado y la referencia canónica de
`validate_fragment_elf.py prepare-reference`, o etiqueta el fixture sintético
si esa fue la entrada utilizada. La paridad de un fixture no acredita Stories15M.

## Roles y entrada

El verificador crea tres directorios nuevos bajo su salida:

- `coordinator-only/`: únicamente el ELF coordinador con su modelo normalizado
  incorporado. Este incremento aún conserva una copia completa de la cabeza.
- `worker-0-only/`: únicamente `fragment_0`, con sus filas incorporadas.
- `worker-1-only/`: únicamente `fragment_1`, con sus filas incorporadas.

Cada proceso worker se inicia en su directorio con:

```text
fragment_N --tcp-test-listen 127.0.0.1:0
```

Anuncia una única línea JSON acotada con formato
`bml-native-fragment-tcp-listener-v1`, dirección y descriptor. Atiende una
conexión y termina. El verificador valida la identidad y dirección, sin sondeos
de conexión ni reintentos que consuman esa conexión. Envía las direcciones en
orden de filas al coordinador:

```text
coordinator --mode tcp-test --fragment-endpoints ADDRESS_0,ADDRESS_1 \
  --prompt Hello --predict 4 --context 32 --logits
```

Los roles son procesos independientes de un único host. Directorios separados
no son aislamiento del sistema de archivos; la medición debe declarar cualquier
comprobación adicional de accesos. Ningún comando del runtime recibe el GGUF,
fuentes ni un compilador. No se cambia la ruta stdin por defecto.

## Verificación reproducible

Usar un `build-report.json` de los nuevos ELF y referencias ya conservadas de al
menos dos casos con el mismo SHA-256 del GGUF. Los hashes siguientes deben
provenir de la entrega confiable; no ejecutar un artefacto desconocido sólo
porque coincide con un manifiesto creado por ese mismo origen desconocido.

```sh
timeout 120s python3 tools/validate_fragment_tcp.py \
  --bundle /ruta/al/bundle-tcp \
  --build-report-sha256 HASH_DEL_BUILD_REPORT \
  --reference-json /ruta/a/reference.json \
  --reference-sha256 HASH_DE_REFERENCE \
  --output /ruta/nueva/resultado-tcp \
  --timeout 110 --trusted-generated-artifacts
```

El plazo interno es global, no se renueva por caso; su máximo permitido es
120 segundos e incluye una reserva de hasta cinco segundos para cleanup.
La receta exterior conserva un límite de 120 segundos. Sólo se terminan procesos creados por el verificador;
no se señalan procesos encontrados por nombre o puertos. No hay reintentos.

La prueba verifica SHA-256, cabeceras ELF AMD64 y cobertura de filas antes de
ejecutar. Compara todos los logits de prefill/decode bit a bit con el coordinador
monolítico y compara IDs/logits visibles con la referencia CLI canónica.
`tcp-test` permanece en las salidas originales y en el informe; sólo una copia
en memoria se adapta al nombre de modo esperado por el comparador v1 reutilizado.
Los dos workers deben terminar con código cero en cada caso positivo.

Controles negativos: workers intercambiados, endpoint duplicado, modo/argumento
desconocido, dirección no loopback, puerto reservado sin listener y desconexión
de un socket de prueba propio. El caso `--tokens 4294967295` exige rechazar
antes de conectar: tras el exit se comprueban los dos listeners propios mediante
`accept` no bloqueante, sin esperar un plazo fijo. Ninguna de estas pruebas
contacta hosts externos ni usa un puerto ajeno conocido.

## Docker y evidencia pendiente

El Dockerfile incorpora ambos verificadores; su `ENTRYPOINT` predeterminado
sigue siendo el verificador v1. La receta TCP siguiente usa **un contenedor**
sin red externa: sus tres procesos comparten el loopback de ese contenedor.
No se ejecutó un build, pull ni ensayo Docker de esta revisión. La imagen local
debe contener la fuente TCP revisada; registrar su ID y arquitectura AMD64 antes
de usarla. `--pull=never` impide obtener una imagen ausente automáticamente.

Partir de un bundle TCP y referencia canónica ya verificados. Preparar un
directorio nuevo para salida, con permisos del UID actual. Sustituir el ID y
los hashes por los de la entrega confiable; no montar el GGUF ni `bundle/build`:

```sh
tcp_runtime_image=sha256:ID_DE_IMAGEN_LOCAL_VERIFICADA
run_root=/ruta/a/resultados_conservados
tcp_output=/ruta/a/salida_tcp_nueva
build_sha=HASH_DEL_BUILD_REPORT_VERIFICADO
ref_sha=HASH_DE_REFERENCE_VERIFICADO
mkdir "$tcp_output"
docker run --rm --pull=never --platform=linux/amd64 --network=none --read-only \
  --memory=4g --memory-swap=4g --cpus=2 --pids-limit=128 \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  --mount "type=bind,src=$run_root/bundle/run,dst=/bundle/run,readonly" \
  --mount "type=bind,src=$run_root/bundle/build-report.json,dst=/bundle/build-report.json,readonly" \
  --mount "type=bind,src=$run_root/reference.json,dst=/inputs/reference.json,readonly" \
  --mount "type=bind,src=$tcp_output,dst=/results" \
  --entrypoint timeout "$tcp_runtime_image" 120s \
  python /opt/bml/validate_fragment_tcp.py \
  --bundle /bundle --build-report-sha256 "$build_sha" \
  --reference-json /inputs/reference.json --reference-sha256 "$ref_sha" \
  --output /results/validation --timeout 110 --trusted-generated-artifacts
```

El límite exterior corre dentro del contenedor, no sólo sobre el cliente Docker.
Los límites de memoria/CPU contienen el ensayo; no acreditan suficiencia para
cualquier modelo. El verificador copia sólo cada ELF a su directorio de rol,
valida hashes y lanza estos entrypoints; no hay un segundo ejecutor paralelo:

```text
/results/validation/worker-0-only/fragment_0 --tcp-test-listen 127.0.0.1:0
/results/validation/worker-1-only/fragment_1 --tcp-test-listen 127.0.0.1:0
/results/validation/coordinator-only/coordinator --mode tcp-test \
  --fragment-endpoints ADDRESS_0,ADDRESS_1 --prompt Hello --predict 4 --context 32 --logits
```

El verificador entrega los puertos anunciados al coordinador, que se conecta una
sola vez a cada worker. Los comandos anteriores describen los roles, no son
pasos adicionales al `docker run`.
La imagen runtime contiene Python, bibliotecas del sistema y la CLI de referencia;
no contiene GGUF, fuentes ni compilador. Los mounts son únicamente ejecutables,
manifiesto y referencia en lectura, más la salida escribible.

Tres contenedores distintos con `--network=none` tendrían loopbacks distintos
y no podrían comunicarse así. No se publica ningún puerto ni se ofrece una
receta entre contenedores o hosts. La ejecución OCI permanece pendiente;
este único contenedor tampoco acredita aislamiento entre roles o dos hosts.

El `report.json` conserva identidades, argumentos, endpoints, PIDs de procesos
propios, estados, todos los logits en archivos separados y el primer desacuerdo
numérico si ocurre. Un fallo, timeout o cleanup no confirmado conserva estado
fallido. La exactitud local y la autonomía sin GGUF deben seguir distinguiéndose
de la futura validación entre hosts y del contrato simbólico BMLX.
