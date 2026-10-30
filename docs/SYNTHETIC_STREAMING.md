# Streaming F32 hasta el último tamaño del plan de 3 GB

30 de octubre de 2026. Gate sintético AOT, no inferencia de un modelo. A fija en compilación; dos x/B distintos inyectados por stdin en runtime. Datos F32, acumulación y salida F64. NumPy 2.3.5 float64 por bloques, un hilo, referencia independiente.

## Máximos reales y parada

- Matvec: **20480×20480**, último tamaño de la secuencia 10·2^k admitido por el límite de 3 GB. GGUF completo **1.677.721.888 bytes**, no solo el tensor. Dos entradas pasaron y dos ejecuciones adicionales en imagen limpia coincidieron con sus hashes, sin BML ni GGUF montados.
- Matmul: **2560×2560**, dos entradas pasaron. Para 5120, la estimación cúbica calibrada con el peor kernel 2560 y margen ×2 fue **186,980 s >60 s**: no se asignó B ni se ejecutó. Los tamaños superiores también quedaron en preflight, no son timeouts ni medidas. No se ejecutaron matmul 5120/10240/20480.
- Siguiente matriz de la secuencia: 40960, **6.710.886.400 bytes** solo para F32 A; excede 3 GB y no se genera.

## Presupuestos verificados

No se cambió el cgroup global de 32 GiB. Cada trabajo completo (generación, compilador, referencia y kernel) se ejecutó en un contenedor con **memory.max=7 GiB**, sin swap y con dos CPU; 1 GiB adicional se reserva conservadoramente para controlador/cliente Docker dentro del presupuesto de 8 GiB. El controlador solo maneja JSON y vectores pequeños; nunca carga matrices completas. Sus comandos no emplean servidores ni destinos alternativos.
Antes de cada tamaño se registraron MemAvailable, memory.current/memory.max e inactive_file recuperable del cgroup host; se exige holgura conservadora >=8 GiB. Dentro del trabajo se verifica el límite efectivo, disco y estimación por operación. No se cuenta el pico global del host como memoria exclusiva de la prueba.
Kernel: timeout de pared y CPU **60 s por entrada**; compilación: 120 s de pared, proceso con 60 s CPU/6 GiB de direcciones; controlador: 420 s por trabajo con parada del contenedor nombrado ante timeout. Cualquier fallo del job se propaga como exit distinto de cero. Las ejecuciones limpias tienen 7 GiB y una CPU y son seriales, después del job, no simultáneas. No hubo OOM ni timeout en los tamaños realmente aceptados.

## Streaming y formato autónomo

Generación PCG64 por 64 filas: no hay listas Python de n² elementos. mmap del GGUF, lector Python independiente de cabeceras/layout y probe bml-gguf de metadatos/muestras acotadas. El probe no decodifica todo el tensor. La referencia convierte solo un bloque de A a F64; para matmul B puede mantenerse en una representación F64 compacta bajo preflight. Hashes SHA256 se calculan por streaming.
El primer intento 20480 se rechazó por preflight de disco antes de escribir datos. Se resolvió el diseño, no se relajó la reserva: se calcula la referencia, se cierra el mmap, se desplaza el payload hacia atrás por bloques de 4 MiB y se reemplaza la cabecera por el launcher ELF en el **mismo archivo**. Ese archivo deja de ser GGUF. La salida contiene el recurso F32 normalizado, alineado y footer versionado; no incluye un GGUF oculto que se vuelva a parsear en runtime. Los casos previos 2560/5120/10240 usaron anexado desde un GGUF separado; el kernel y contrato son iguales. La transformación in-place se verificó con tests de solapamiento multi-bloque y el caso completo 20480.
El footer BMLF32V1 registra dimensión, offset, longitud y checksum FNV-1a64. El launcher comprueba magic, forma, rangos, alineación y checksum, luego mapea su propio ejecutable mediante /proc/self/exe. FNV64 detecta corrupción accidental; no es autenticación criptográfica. SHA256 integral del GGUF y ejecutable queda en los manifiestos. Tests pequeños verifican corrupción de payload/footer y truncamiento, además de entradas distintas y productos esperados.
Autonomía probada: sin BML, toolchain ni GGUF en ejecución limpia; requiere **Linux/amd64, loader/glibc compatibles y procfs**. No es un binario universal ni completamente estático. Pesos/tokenizer de un Llama/Gemma real no están implementados por este gate. El footer es un prototipo para un tensor F32, no el formato completo de recursos de transformer de R3.

## Resultados medidos

Dos entradas por operación; intervalos son las dos observaciones, no confianza estadística. Kernel excluye chequeo de recurso y E/S. Integridad se mide aparte. Proceso incluye arranque, mmap/checksum y E/S binaria; no incluye la validación posterior Python. Error máximo **0** en todos estos fixtures de valores binarios exactos; atol=rtol=1e-9. No se usa esto como garantía general para cuantizadores/modelos ni se convierte a TPS.

| n | Operación | Kernel s | Integridad s | Proceso s | Pico cgroup trabajo bytes |
|---:|---|---:|---:|---:|---:|
| 2560 | matvec | 0.023661–0.024439 | 0.032229–0.041969 | 0.058886–0.067767 | 401432576 |
| 2560 | matmul | 11.392735–11.686279 | 0.032244–0.032906 | 11.450788–11.743634 | 401432576 |
| 5120 | matvec | 0.091871–0.092010 | 0.124143–0.132880 | 0.218925–0.227495 | 250044416 |
| 10240 | matvec | 0.367971–0.369698 | 0.517814–0.528607 | 0.890078–0.901853 | 899952640 |
| 20480 | matvec | 1.478090–1.493149 | 2.037609–2.157110 | 3.523659–3.656353 | 1775857664 |

| n | GGUF completo bytes | Generación s | Probe BML s | Compilación s | Embedding s | Ejecutable bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 2560 | 26214688 | 0.018793 | 0.001224 | 0.118154 | 0.044280 | 26234920 |
| 5120 | 104857888 | 0.073050 | 0.001316 | 0.119650 | 0.156800 | 104878120 |
| 10240 | 419430688 | 0.259185 | 0.001323 | 0.127872 | 0.634716 | 419450920 |
| 20480 | 1677721888 | 3.696793 | 0.001890 | 0.124427 | 1.044041 | 1677742120 |

En 20480 el checksum completo al arrancar cuesta unos 2 s frente a ~1,5 s del kernel: no se oculta esa diferencia como aceleración. La integridad/embedding son costes separados; el hash previo FNV de construcción y las referencias tienen sus propias fases, no se suman a la columna compilación.

## Reproducir sin descargas automáticas

Necesita Docker local, Rust 1.98.1, CPython 3.12 y NumPy 2.3.5 existentes para preparar el contexto público; no se copian credenciales ni modelos. Bases Docker por digest. El job utiliza Python 3.12.12/NumPy 2.3.5 y GCC del builder bookworm. La imagen limpia es la del gate previo sin BML. En cloud preservar proxy/CA del daemon y usar BUILDX_CONFIG escribible; no cambiar credenciales ni desactivar TLS.

```sh
cargo build -p bml-gguf --example stream_fixture_probe --release --locked --offline
python3 tools/prepare_stream_context.py --output /ruta/nueva/contexto
docker build --network=none -t bml-stream-job:local /ruta/nueva/contexto
python3 tools/run_synthetic_stream.py --results /ruta/nueva/resultados --work /ruta/nueva/trabajo
```

El controlador ejecuta una matriz/operación cada vez y elimina GGUF, intermediarios, entradas/salidas densas y binario después de las comprobaciones limpias. Solo quedan JSON/logs/hashes pequeños; no se empaquetan ni se transfieren gigabytes. Los directorios de trabajo terminaron con unos pocos KiB de estructura, sin pesos duplicados. El primer informe de preflight fallido se conserva, diferenciado de la corrida exitosa tras la mejora.
La corrida inicial del controlador devolvía cero incluso al encontrar un job fallido: su JSON y container_exit sí registraron correctamente ese fallo de preflight. Se corrigió la propagación (ahora devuelve 2) y se añadió regresión; no se reinterpreta el fallo inicial como una prueba pasada.

## Complejidad y estado de desarrollo

Para este algoritmo clásico con A fija y x/B dinámico: matvec Θ(n²), matmul Θ(n³), almacenamiento denso Θ(n²). No son cotas universales para todas las especializaciones posibles. No se precomputa un producto de dos operandos constantes. El tope de 3 GB no autoriza matmul de cualquier tamaño: el presupuesto de tiempo sigue siendo independiente.
F16/Q4_0/Q8_0 siguen pendientes; no se inventa soporte en dimensiones no múltiplo de bloque. R1 (Llama/tokenizer/logits/KV), R2 tensorial general y recursos completos R3/R4 permanecen pendientes. La release Docker 21b90c1 y su diagnóstico de API bloqueada no se modificaron. No hubo publicación, transferencia de artefactos nuevos ni cambios de permisos/cgroups globales.

## Checks de cierre

`CARGO_NET_OFFLINE=true bash scripts/check.sh d43be743c309227e90cecc36690b21ee2c8d0b23`:
312 Rust aprobadas, 18 Python aprobadas, 15 Rust opcionales ignoradas; cero
discrepancias, trazabilidad 15/15, formato correcto y cero diagnósticos Clippy
nuevos (123 existentes). Clippy estricto de ejemplos bml-gguf pasó. Las cinco
regresiones nuevas cubren autonomía/matvec/matmul con dos entradas, corrupción,
truncamiento, desplazamiento in-place con solapamiento multi-bloque, límites de
memoria/disco y propagación del fallo del job. No se reiniciaron las corridas
numéricas ya completadas para estos checks. No se ejecutó campaña completa de
mutación del C/Python ni se usa cobertura Rust como cobertura del kernel emitido.
