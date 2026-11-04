# Integración simbólica distribuida

Esta entrega conecta los programas BML bounded persistidos a sesiones TCP,
procesos `bml-worker` y `bml-cli`. El manifiesto fija los hashes del GGUF y cada
programa. La identidad del handshake es SHA-256 de los bytes exactos de ese
manifiesto, después de validar la fuente, las firmas, las constantes, los hashes
y el plan completo. El cliente y cada worker cargan con `max_nodes=0`: no se
reconstruyen DAGs ni se sustituyen por kernels nativos.

El worker configura previamente artefacto, fuente opcional, rango, contexto,
presupuesto KV y límite de frame. Una conexión posee su propio modelo y caché
F32; `prepare` deja la posición sin confirmar, `commit` la avanza, `abort`
permite reemplazar la fila pendiente y `reset` crea una caché vacía. El frame
versionado usa f64 little-endian y comprueba longitudes y finitud antes de
reservar o evaluar. El protocolo no cifra ni autentica al par: la identidad
solo detecta artefactos distintos.

El coordinador verifica rangos contiguos exactos, identidad, ancho y contexto;
lee embeddings, tokenizer, normalización final y cabeza reales. Ejecuta todos
los rangos antes de confirmar. Si cualquier operación o confirmación remota
falla, invalida todo el pipeline y conserva su posición local: alguna sesión
remota podría haber confirmado y no hay reintento automático ni rollback
remoto. La CLI distribuye rangos contiguos entre los nodos disponibles, ignora
nodos sobrantes, valida temperatura y contexto, genera de forma autoregresiva y
emite bytes directamente, conservando EOS y fragmentos UTF-8 incompletos.

## Evidencia focalizada

- `symbolic_transport_contracts`: dos conexiones TCP/rangos, identidades
  incorrectas y modificadas, frame sobredimensionado, sesiones independientes,
  aborto/reintento/reset y dos confirmaciones perdidas, incluida una después de
  que el primer worker confirme. En ambos fallos el coordinador se invalida.
- `symbolic_worker_contracts`: procesos worker reales con un modelo GQA de tres
  capas, rangos de uno/dos/tres workers, nodo adicional sin asignar, rechazo de
  huecos, duplicaciones y sesiones no iniciales, y terminación real del último worker. Los logits
  coinciden bit a bit con el forward simbólico local en tres tokens.
- `symbolic_distributed_cli`: compila con `--distributed`, reubica el GGUF y
  genera dos tokens con uno, dos y tres workers; un cuarto nodo no recibe
  conexión. Los logits de referencia simbólica y nativa difieren menos de
  `2e-7` en este fixture y los tokens greedy coinciden. EOS produce salida
  vacía y un byte UTF-8 parcial `0xC3` se escribe sin sustitución. El test se
  ejecutó tanto con servidores del mismo proceso como con procesos worker reales
  mediante `BML_WORKER_BIN`.
- `indexed_buffer_artifact_contracts::distributed_ranges_execute_replaced_valid_persisted_bytecode`:
  al sustituir un programa válido y actualizar su hash, el resultado distribuido
  cambia. Así se comprueba que se ejecuta el bytecode almacenado.

Para reproducir la variante de procesos reales, compilar el worker con el
`rust-toolchain.toml` fijado y ejecutar:

```sh
cargo build -p bml-worker --locked
BML_WORKER_BIN="${CARGO_TARGET_DIR:-target}/debug/bml-worker" cargo test -p bml-cli --test symbolic_distributed_cli --locked
cargo test -p bml-worker --test symbolic_worker_contracts --locked
cargo test -p bml-runtime --test symbolic_transport_contracts --locked
cargo test -p bml-compiler --test indexed_buffer_artifact_contracts --locked distributed_ranges_execute
```

La validación es de un fixture local, no de TinyLlama ni de velocidad o L1. Cada
worker aún requiere acceso al modelo entero, pero reserva KV solo para su rango;
no existe transferencia automática de pesos por shard. Los constructores
públicos históricos `compile_distributed` y `compile_distributed_bml` están
retirados y devuelven un error explícito de migración; la CLI rechaza sus
artefactos históricos como entrada para generación. La tarea 8.5.8 sigue abierta
por el alcance restante del pipeline.

## Modelo real: Stories15M

Una prueba optativa exacta con el GGUF local Stories15M y el artefacto bounded
v4 existente ejecutó el token BOS `[1]` mediante dos procesos worker release,
asignados a capas `[0,3)` y `[3,6)`. El coordinador comparó los 32.000 logits
con el forward nativo del mismo GGUF: todos fueron finitos, ambos argmax fueron
`9038`, el error absoluto máximo fue `1.6198153041102614e-8` y el error RMS
`2.2718347465181055e-9`. Se verificó la posición confirmada y la carga con
`max_nodes=0`. El tiempo observado de 149,4 segundos corresponde a una ejecución
bajo carga y no es un benchmark controlado.

`distributed-stories-real.json` conserva el comando exacto del test ignorado,
el hash del GGUF, manifiesto, binarios, fuente y log. Los 284 hashes de fuente
fueron idénticos antes y después de la ejecución final. La ejecución anterior,
previa al control explícito de posición del cliente y a las pruebas reforzadas,
se conserva en `distributed-stories-real-initial.json` con sus propios hashes. La repetición previa al ajuste del sender se conserva en `distributed-stories-real-before-net-audit.json`.
Esto acredita un token de Stories15M, no un modelo mayor ni prompts adicionales. La referencia nativa usa el mismo GGUF; no se afirma paridad con llama.cpp.


## Auditoría del dispatcher

La auditoría release actual se guarda en `distributed-symbolic-hot-loop.json`:
2533 bytes, 19 llamadas y dos tablas de salto, con un helper local `trunc` de
74 bytes. No acredita comportamiento L1 ni aislamiento transitivo de hooks. El
primer intento detectó que el enlazador había integrado el sender de red; se
conservó como símbolo separado mediante `#[inline(never)]`. El segundo reveló
dos símbolos con el mismo nombre desambiguado: el auditor ahora acepta solo
direcciones exactas presentes en la tabla de símbolos y una prueba negativa
comprueba que una dirección intermedia sigue siendo rechazada. No se relajó la
lista de destinos permitidos.

## KV por rango

Cada sesión asigna solo las capas que ejecuta y conserva índices globales. Las
pruebas de tres capas verifican el presupuesto exacto, ejecución causal bit a bit,
abort, commit y reset. La repetición con Stories15M y dos procesos release
produjo 32.000 logits finitos, argmax 9038 igual al nativo y error máximo
1.6198153041102614e-8; `layer-local-kv-stories.json` conserva la instantánea y
el comando. Después de medir solo cambió la expresión de una prueba de rango
invertido para satisfacer Clippy. No es una medición de rendimiento.
