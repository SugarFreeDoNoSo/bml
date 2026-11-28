# Fragmentos ELF: contrato de transporte privado y preparación OCI

Esta etapa añade un puente diagnóstico Python `ssl` con TLS mutuo. No cambia
la matemática ni el protocolo de frames de los ELF. El coordinador nativo sigue
usando `--mode tcp-test` con endpoints **loopback**; un proxy Python conecta cada
endpoint remoto mediante TLS a una IPv4 privada explícita. El servidor Python
autentica al driver antes de arrancar el worker ELF como hijo por stdin/stdout.
El salto loopback confía en los procesos de la misma task; no es autenticación
dentro del ELF ni prueba de aislamiento del filesystem del peer.

El runtime necesita Python y su biblioteca SSL: **el transporte privado todavía
no es un ELF autónomo**. No requiere GGUF, compilador, paquetes Python externos
ni el ejecutable `openssl` en ejecución. Los certificados se emiten fuera del
runtime. La ruta simbólica BMLX, la tarea `bml-1-0-release/4.6` y la parte uno
conservan sus criterios; esta preparación no los cierra.

## Estado y topologías

El intento local de crear una namespace de red se detuvo porque
`/proc/self/uid_map` era de sólo lectura. No se reintentó ni se cambiaron permisos.
No se ejecutaron builds, pulls ni contenedores OCI de esta etapa. Dos interfaces
privadas del executor siguen perteneciendo al mismo kernel: una validación con
ellas acredita procesos y transporte local, no hosts distintos.

| Plan | Composición prevista | Endpoints que recibe el ELF coordinador |
| --- | --- | --- |
| A: una task, tres roles | Driver y dos servidores worker en la misma task; puertos TLS configurables 44761 y 44762 si comparten IP. | Dos proxies loopback, en orden de filas. Cada proxy habla TLS con su worker. |
| B: dos tasks | Task A: driver y worker0 local. Task B: servidor TLS y worker1 remoto en 44761 configurable. | Primero worker0 nativo loopback, verificado por hash; segundo proxy loopback hacia worker1 por TLS. |

En B, worker0 conserva la entrada v2 `--tcp-test-listen 127.0.0.1:0` y no necesita
TLS en ese salto local. El worker remoto usa stdin/stdout tras la autenticación
TLS. En A, cada servidor TLS usa su propio worker stdio. El número de roles de
inferencia no cuenta los procesos Python auxiliares y el supervisor.

El verificador local puede componer ambos planes en una sola máquina; debe
declarar `distinct_hosts_verified: false`. Dos tasks tampoco prueban por sí
solas dos hosts físicos: la validación futura necesita evidencia de ubicación
y paridad de la corrida completa. No hay despliegue ni llamada AWS en esta receta.

## Identidad y configuración de los roles

La entrada del helper es explícita y no cambia el `ENTRYPOINT` v1 de la imagen:

```text
python /opt/bml/fragment_private_transport.py worker --private-ipv4 --config /run/config/worker.json
python /opt/bml/fragment_private_transport.py client --private-ipv4 --config /run/config/client.json
```

Cada configuración JSON tiene `version: 1`, `run_id` de 32 caracteres hexadecimales
minúsculos y el `descriptor` v1 exacto de las filas, tomado del `build-report.json`
con hash fijado. No reconstruir ni redondear campos del descriptor. Los campos
desconocidos se rechazan. El objeto `tls` contiene:

| Campo | Valor exigido por el contrato de la corrida |
| --- | --- |
| `ca_file` | Archivo con la CA exclusiva de esta corrida. La clave de la CA nunca llega al runtime. |
| `cert_file`, `key_file` | Certificado y clave propios del rol; archivos privados legibles por su UID. |
| `peer_cert_sha256` | SHA-256 del certificado DER exacto del peer autorizado, además de validar su cadena. No es el hash del PEM. |
| `server_name` | Identidad SAN DNS esperada del peer: `worker-N.RUNID.bml.invalid` en el cliente y `driver.RUNID.bml.invalid` en el worker. |

Los nombres SAN son identidades TLS; **no se resuelven por DNS**. La conexión
usa la IP literal fijada, permitiendo emitir certificados antes de asignar las
IPs de las tasks. Worker y driver usan certificados distintos, con usos de
servidor y cliente apropiados. Se exige TLS 1.3, ALPN `bml-native-head-v1` y
autenticación mutua; el certificado debe tener únicamente el SAN esperado. No hay
HMAC propio ni sustitución de CA/pin por mera pertenencia a una subred.

Campos adicionales del servidor `worker`:

- `elf_file`, `elf_sha256`: únicamente el ELF del fragmento de ese rol y su hash
  fijado; verificar también el build y la cabecera AMD64 antes de provisionarlo.
- `bind_ip`: una IPv4 RFC1918 literal concreta, nunca `0.0.0.0` ni loopback.
- `port`: puerto configurado; por defecto 44761. El puerto cero sólo pertenece
  al modo de pruebas explícito `--test-ephemeral-port`, no al plan de tasks.
- `allowed_peer_ips`: lista no vacía de IPs exactas del driver en el modo normal.
- `bootstrap_peer_from_certificate`: `false` normalmente. Si es `true`, la
  lista anterior debe estar vacía y aplica el arranque acotado descrito abajo.
- `startup_seconds`: entero entre 1 y 120, por defecto 120.

Campos adicionales del proxy `client`:

- `peer_ip` y `port`: IP RFC1918 literal y puerto del worker, conocidos por el
  orquestador antes de iniciar el proxy; el puerto por defecto es 44761.
- `source_ip`: opcional o `null`; si se fija, debe ser RFC1918. La IP efectiva de
  la conexión también se valida, sin aceptar una ruta pública por omisión.
- `listen_ip: "127.0.0.1"`, `listen_port`: endpoint local para el coordinador;
  puerto cero permite que el sistema asigne uno, anunciado antes de aceptar.
- `startup_seconds`: entero entre 1 y 120, por defecto 120.

RFC1918 significa exactamente `10.0.0.0/8`, `172.16.0.0/12` o `192.168.0.0/16`.
No usar `is_private` como sustituto de esos rangos. El orquestador debe verificar
que las IPs pertenecen a los roles de esta corrida; una IP privada por sí sola
no identifica al worker ni demuestra conectividad autorizada.

## Arranque y fallos: contrato para el orquestador

1. Fijar revisión, digests de imagen y ELF, fuente GGUF de compilación, descriptor
   y referencia canónica. Preparar credenciales nuevas de la corrida fuera de la
   imagen y del contexto Docker; distribuir sólo lo necesario para cada rol.
2. Iniciar los servidores con bind privado explícito. Capturar una única línea
   JSON de disponibilidad por servidor; no abrir conexiones de sondeo que
   consuman su única conexión de sesión.
3. Con IP de driver conocida, usar `allowed_peer_ips`. Si se desconoce al inicio,
   elegir explícitamente bootstrap: aceptar sólo un origen RFC1918, autenticar
   CA, nombre esperado y pin DER del driver, y **después** fijar esa IP para esa
   sesión. No aprender identidad desde el primer paquete ni desde una IP sola.
4. El orquestador conoce las IPs de worker y las pone explícitamente en cada
   configuración client. Arrancar los proxies y leer sus endpoints loopback;
   en B arrancar además worker0 local y verificar su descriptor/hash.
5. Invocar el coordinador con los endpoints en el orden exacto de filas y un
   caso fijado. Conservar IDs y todos los logits de prefill/decode para comparar
   con la ejecución monolítica y la referencia CLI del mismo modelo/contexto.
6. Exigir fin correcto del coordinador, proxies y workers. Ante error, terminar
   sólo los procesos/tareas propios de la corrida y guardar el primer fallo.
   No reintentar ni cambiar transporte o credenciales silenciosamente.

Disponibilidad del servidor y del proxy, respectivamente:

```text
{"format":"bml-private-fragment-worker-v1","address":"IP:PORT","run_id":"RUN_ID"}
{"format":"bml-private-fragment-client-v1","address":"127.0.0.1:PORT","run_id":"RUN_ID"}
```

Estos anuncios no acreditan una sesión autenticada terminada. Los errores y el
evento de bootstrap van por stderr, sin material de claves/certificados. Un
certificado identifica al peer TLS autorizado; no acredita por sí solo el hash
del ejecutable ni la integridad del filesystem remoto. El descriptor `Ready`
del protocolo numérico sigue siendo obligatorio.

Cada proceso atiende una sola conexión. El límite no es un token criptográfico
consumido persistentemente: relanzar un proceso con la misma configuración
recrearía su listener. El orquestador impide reutilización accidental de una
corrida; un nuevo intento necesita identidad y credenciales nuevas explícitas.
El verificador local reutiliza deliberadamente las credenciales de una corrida
entre sus casos/planes secuenciales, con procesos nuevos de una sola conexión.
No mantiene un registro persistente que impida reutilizar ese manifiesto después:
esa prohibición pertenece al contrato del orquestador, no a una garantía TLS.

| Presupuesto | Límite |
| --- | --- |
| Espera de arranque | Hasta 120 s por proceso, no renovados por intento. |
| Handshake TLS y fase/frame | 20 s absolutos; el goteo no renueva el plazo. |
| Sesión | 120 s, con deadline compartido por sus operaciones. |
| Cuerpo de frame | 256 KiB. |
| Solicitudes de sesión | 256 como máximo. |
| Supervisor exterior de cada rol | 240 s, incluyendo arranque y cierre; debe terminar también sus hijos. |
| Supervisor exterior de la corrida | 360 s como máximo para todos sus pasos y cleanup, sin renovar por rol. |

Los topes del supervisor son requisitos de preparación; registrar cuáles impone
realmente el ejecutor utilizado. Reservar los últimos cinco segundos para matar
procesos propios que no terminen con TERM: umbrales nominales 235+5 s por rol y
355+5 s por corrida. Confirmar el cierre de hijos y contenedor; no basta enviar
una señal. Fallar por timeout o presupuesto nunca equivale
a una validación parcial aprobada. No elevar estas cotas para completar un modelo.

## Ensayo local reproducible del verificador

La CLI implementada ejecuta **exactamente dos casos** de la referencia, cada uno
con un control monolítico y ambos planes A/B: cuatro comparaciones completas.
Requiere dos IPv4 RFC1918 distintas ya asignadas al mismo entorno autorizado;
no crea namespaces, no configura interfaces ni accede a otro computador. El
orquestador local fija `startup_seconds: 15` y puertos efímeros explícitos para
evitar conflictos de los roles de prueba. No acredita la reserva OCI 44761/44762.

Preparar las credenciales fuera del checkout y del contexto Docker, en un
directorio absoluto nuevo cuyo padre ya existe. Si ya hay credenciales de esta
corrida, usar ese manifiesto sin regenerarlas. El emisor necesita OpenSSL 3 ya
instalado, trabaja offline con un presupuesto total de 30 s y emite certificados
de un día; no es una dependencia del runtime:

```sh
credentials_dir=/ruta/privada/nueva/credenciales_de_corrida
python3 tools/fragment_run_credentials.py --output "$credentials_dir"
```

La carpeta contiene `issuer/` privado y un `manifest.json` público de identidades
con formato `bml-fragment-run-credentials-v1`. Cada rol dispone únicamente de
`runtime/driver/`, `runtime/worker-0/` o `runtime/worker-1/`, con `ca.crt`,
`cert.crt` y `key.key`. El verificador abre el manifiesto y esos archivos de rol;
no abre ni copia `issuer/`. No montar el árbol completo de credenciales en una
task. Las claves deben pertenecer al UID efectivo con modo 0400 o 0600, sin
symlinks; las claves cifradas que exijan contraseña se rechazan.

Conservar los hashes confiables del bundle v2 reconstruido y de la referencia,
así como la identidad real o sintética del GGUF usado para producirlos. La carpeta
de salida no debe existir y su padre debe existir. Sustituir las dos direcciones
por direcciones locales autorizadas; no adivinarlas ni elegir hosts remotos:

```sh
worker_ip=IP_RFC1918_LOCAL_EXISTENTE_PARA_WORKER
driver_ip=OTRA_IP_RFC1918_LOCAL_EXISTENTE_PARA_DRIVER
timeout --kill-after=5s 355s python3 tools/validate_fragment_private.py \
  --bundle /ruta/a/bundle_verificado \
  --build-report-sha256 HASH_DEL_BUILD_REPORT_VERIFICADO \
  --reference-json /ruta/a/reference.json \
  --reference-sha256 HASH_DE_REFERENCE_VERIFICADO \
  --credentials-dir "$credentials_dir" \
  --worker-ip "$worker_ip" --driver-ip "$driver_ip" \
  --output /ruta/privada/nueva/salida_de_validacion \
  --timeout 340 --trusted-generated-artifacts --test-ephemeral-ports
```

El plazo interno máximo permitido es 360 s (por defecto 240), con hasta diez
segundos reservados a cleanup; la receta usa 340 para dejar margen al supervisor
exterior nominal de 360. Cada invocación del coordinador tiene además un límite
de hasta 120 s. El emisor se prepara antes de esa ventana de ejecución.

`report.json` registra fases, comandos, procesos, cleanup e identidades públicas;
los archivos por caso conservan los logits. El staging local incluye copias de
claves de rol en directorios privados: **no adjuntar ni publicar la carpeta de
salida completa**. Para compartir resultados seleccionar sólo el informe y
evidencia explícitamente saneada, nunca `tls/`, configs privados o credenciales.
El programa declara explícitamente que no verifica hosts distintos, namespaces,
aislamiento de filesystem, TLS nativo del ELF, modelo real ni BMLX simbólico.
El entrypoint local no provisiona tasks ni sustituye la futura prueba OCI.

## Preparación OCI y credenciales

El Dockerfile conserva las bases fijadas de Rust 1.98.1 y Python 3.12.12 de v2,
añade sólo los helpers Python y mantiene la entrada v1. Antes de un ensayo OCI,
verificar el ID de la nueva imagen AMD64 y `ssl.HAS_TLSv1_3`/versión SSL dentro
de ella; la versión SSL del executor no acredita la de la imagen. Comprobar
también `timeout` y acceso Linux a `/proc/self/fd`: el helper usa descriptores
verificados para cargar certificados/claves y ejecutar el ELF. No se añade
`openssl` CLI ni una dependencia Python. La preparación no ejecuta esa comprobación.

Los secretos viven fuera del checkout/contexto: nunca `COPY`, `ARG`, `ENV`,
argumentos con PEM o logs. Los parámetros llevan rutas, no contenido secreto.
Montar archivos de CA/certificado/clave por rol en lectura, con permisos mínimos
para el UID del proceso; no aplicarles el `chmod a+rX` de las fuentes de la imagen.
No montar la clave de la CA, el GGUF, fuentes, compilador ni cachés en los roles.
La afirmación de v1 «sin secretos montados» sólo describe el flujo anterior sin TLS.

El runtime no necesita TaskRole ni llamadas IAM. La provisión de secretos y
el execution role pertenecen al orquestador/plataforma y se preparan aparte.
No suponer que configurar una referencia a un secreto produce automáticamente
los archivos requeridos: el mecanismo de inyección debe materializarlos fuera
del runtime, con ownership adecuado y sin incluirlos en imágenes o salida.

Ejemplo de entrada OCI de **un servidor remoto**, sobre una red privada local
ya preparada. Es una receta pendiente, no un comando ejecutado ni un despliegue.
`private_network`, IP, configuración, certificados y ELF deben corresponder al
mismo rol verificado; no crear una red ni obtener imágenes como parte de este paso:

```sh
private_runtime_image=sha256:ID_DE_IMAGEN_LOCAL_VERIFICADA
private_network=RED_PRIVADA_LOCAL_PREPARADA
worker_ip=IP_RFC1918_DEL_ROL
role_root=/ruta/privada/al/rol_worker1
docker run --rm --pull=never --platform=linux/amd64 --read-only \
  --network "$private_network" --ip "$worker_ip" \
  --memory=4g --memory-swap=4g --cpus=2 --pids-limit=128 \
  --cap-drop=ALL --security-opt=no-new-privileges \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  --mount "type=bind,src=$role_root/fragment_1,dst=/opt/role/fragment_1,readonly" \
  --mount "type=bind,src=$role_root/worker.json,dst=/run/config/worker.json,readonly" \
  --mount "type=bind,src=$role_root/tls/ca.crt,dst=/run/tls/ca.crt,readonly" \
  --mount "type=bind,src=$role_root/tls/cert.crt,dst=/run/tls/cert.crt,readonly" \
  --mount "type=bind,src=$role_root/tls/key.key,dst=/run/tls/key.key,readonly" \
  --entrypoint timeout "$private_runtime_image" --kill-after=5s 235s \
  python /opt/bml/fragment_private_transport.py worker \
  --private-ipv4 --config /run/config/worker.json
```

La task driver requiere su propia configuración/credenciales, coordinador y,
en B, sólo el ELF local worker0. Arranca el helper `client` indicado arriba y
entrega su endpoint al coordinador; nunca le pasa la IP privada al `tcp-test`
nativo, que sigue limitado a loopback. La composición completa se prepara en
`tools/validate_fragment_private.py`; el ejemplo de servidor no la sustituye.
Las rutas de `worker.json` deben apuntar a los destinos **dentro** del contenedor
(`/opt/role/fragment_1` y `/run/tls/...`); no copiar sin adaptar una configuración
del ensayo local que contiene rutas absolutas del executor.

No usar `--network=none` entre roles que deben comunicarse por una IP privada.
No publicar puertos (`-p`), usar red de host ni compartir claves entre roles.
Contenedores distintos necesitan una namespace/red adecuada, cuya propiedad y
restricciones se verifican antes de la prueba. El ensayo OCI y su aislamiento
siguen pendientes; no se elude la restricción de namespace del executor.

## Contrato de red para la preparación externa

El padre puede preparar una red privada y un SG dedicado fuera de este cambio:
sin IPs públicas ni ingress de Internet; ingress de los puertos worker elegidos
sólo desde ese mismo SG, y egress mínimo para el peer/puerto previsto. Cualquier
salida necesaria para inyectar secretos, obtener la imagen o entregar logs debe
identificarse y restringirse en la fase de plataforma, separada del cálculo.
No afirmar que el SG, subredes, endpoints, roles o secretos ya fueron creados.
No hay SDK AWS, políticas aplicadas ni escrituras cloud en estos archivos.

El informe de cada corrida debe registrar plan real, roles y ubicación disponible,
hashes de ELF/imagen/configuración sin secretos, IDs de token, logits comparados,
límites efectivos, resultado del handshake y del cleanup, y primer fallo.
No guardar PEM, claves o credenciales en el JSON ni inferir rendimiento o cierre
distribuido de una prueba local. La entrega de fuentes puede continuar con OCI
pendiente, conservando esa limitación explícita.
