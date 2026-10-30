# Gate sintético F32: GGUF, matvec/matmul y AOT

Este documento conserva la corrida inicial 10–80. La ampliación real a tamaños
medianos y sus presupuestos separados están en [escalado](SYNTHETIC_AOT_SCALING.md).

30 de octubre de 2026. Implementación experimental, no inferencia Llama/Gemma.
La secuencia vigente es n=10·2^k. Límite de matriz **y archivo GGUF completo**:
3 GB decimales. No se descargan modelos ni se añade este material a la imagen
Docker de validación del commit 21b90c1.

## Formato, contrato y referencia

`tools/synthetic_aot.py` genera GGUF v3 LE estándar con una matriz F32 de
GGML dims `[ne0=n,ne1=n]`, datos contiguos por fila de salida, offset relativo 0
y alineación de 32 bytes. Metadata de strings identifica fixture/semilla; no
pretende codificar un grafo estándar GGUF ni soportar inferencia arbitraria.
El contrato está en el manifiesto: matvec `y[r]=sum A[r,k]*x[k]`; matmul
`C[r,c]=sum A[r,k]*B[k,c]`, B/C row-major. Solo A es recurso embebido; las
activaciones x/B se inyectan por stdin en cada ejecución.

Semilla 20261030; pesos `Random(seed).randint(-128,128)/16`, activaciones de una
segunda secuencia `Random(seed+1).randint(-64,64)/8`. Son números binarios exactos;
el formato de pesos es F32 y las activaciones/acumulación del kernel son F64.
No atribuir a estos resultados rendimiento de GEMM F32 puro. Se guarda el hash
GGUF, fuente C emitida, binario, inputs, outputs, referencia y comandos.

Un lector Python de subconjunto GGUF (`tools/gguf_fixture_reader.py`), separado
del escritor y de BML, comprueba encabezados/dimensiones/rangos y valores.
`bml-gguf` importa independientemente y exporta pesos F32 normalizados mediante
`synthetic_export`; se exige igualdad exacta del roundtrip. La referencia de
operaciones es Python float64 con `math.fsum`, no el código C generado.
No es una prueba con el lector de llama.cpp; este gate no requiere ese motor.

## Codegen y prueba aislada

Se emite C con `static const float A[]` y loops directos; cc compila con
`-std=c11 -O2 -fno-fast-math`. No invoca llama.cpp ni BML al ejecutarse.
El GGUF y el export intermedio se eliminan antes del runtime; un contenedor
limpio recibe solo el binario readonly, stdin y librerías del sistema, sin red,
BML, toolchain ni modelos. Se vuelve a guardar el fixture reproducible después
de ejecutar para su entrega. No se prueba autonomía de un transformer.

Imagen limpia derivada de Python 3.12.12 slim-bookworm, digest
`sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c`,
sin instalar BML/cc/rustc. Su Python no se usa para ejecutar el kernel.
El binario usa libc/libm declaradas por ldd: **no** se afirma enlace totalmente
estático, ni portabilidad fuera del target Linux/amd64 probado.

Antes de asignar matrices se exige al menos 1 GiB de holgura considerando
MemAvailable y memory.max-current. n real se limita a 128; en la secuencia
vigente se midieron 10/20/40/80. Máximo tensor medido 25.600 bytes. Compilador y
subprocesos locales: 512 MiB de espacio virtual, 30 s CPU y timeout 60 s.
Docker CLI se exceptúa del tope de espacio virtual por la reserva del runtime
Go; la ejecución real dentro del contenedor sí tiene 128 MiB, una CPU, sin swap,
capabilities ni escritura en la imagen. Se registran picos RSS del proceso;
los picos/corrientes del cgroup cloud incluyen otros procesos y no son memoria
atribuible exclusivamente al kernel. Un primer intento falló por el límite de
espacio virtual del cliente Go; se corrigió separando esos dos presupuestos.

## Resultados medidos

Una muestra por tamaño/operación, sin pretensión estadística. Tiempos del proceso
host incluyendo inicio, parseo stdin y JSON stdout; no latencias puras del kernel.

| n | Generar ms | Importar ms | Codegen ms | Compilar ms | Binario bytes | Matvec ms | Matmul ms | RSS aislado KiB mv/mm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.316 | 2.231 | 0.147 | 60.552 | 16464 | 1.830 | 2.558 | 7076 / 6820 |
| 20 | 0.170 | 1.936 | 0.219 | 57.579 | 16464 | 2.159 | 2.205 | 6820 / 6820 |
| 40 | 0.539 | 3.373 | 0.568 | 58.338 | 20560 | 2.062 | 2.814 | 7076 / 6820 |
| 80 | 2.523 | 4.198 | 2.064 | 69.129 | 41040 | 3.579 | 6.240 | 6820 / 7060 |

Tolerancias predefinidas atol=1e-9 y rtol=1e-9. Las ocho comprobaciones aisladas
pasaron y coincidieron con las ejecuciones host. También se rechazan activaciones
faltantes, NaN o sobrantes. El resultado de cero error aplica a estos fixtures;
no es una garantía para todos los modelos, valores ni cuantizadores.

## Escalado analítico, no ejecutado por encima de 80

F32: A=4n² bytes; matvec ~2n² FLOPs, workspace/tráfico mínimo 4n²+8n bytes;
matmul ~2n³ FLOPs y A+B+C=12n² bytes. Conteos escalares exactos restan n y n²,
respectivamente, si multiplicación y suma cuentan por separado. Tráfico ideal
matmul 12n²; un loop ingenuo sin reutilización puede mover ~8n³+4n² bytes.
Estos presupuestos F32 no son el workspace F64 del kernel emitido.

| n | A F32 bytes | Matvec FLOPs ~ | Matmul FLOPs ~ | A+B+C F32 bytes |
|---:|---:|---:|---:|---:|
| 10 | 400 | 200 | 2000 | 1200 |
| 20 | 1600 | 800 | 16000 | 4800 |
| 40 | 6400 | 3200 | 128000 | 19200 |
| 80 | 25600 | 12800 | 1024000 | 76800 |
| 160 | 102400 | 51200 | 8192000 | 307200 |
| 320 | 409600 | 204800 | 65536000 | 1228800 |
| 640 | 1638400 | 819200 | 524288000 | 4915200 |
| 1280 | 6553600 | 3276800 | 4194304000 | 19660800 |
| 2560 | 26214400 | 13107200 | 33554432000 | 78643200 |
| 5120 | 104857600 | 52428800 | 268435456000 | 314572800 |
| 10240 | 419430400 | 209715200 | 2147483648000 | 1258291200 |
| 20480 | 1677721600 | 838860800 | 17179869184000 | 5033164800 |

El archivo se presupuesta con 64 KiB adicionales conservadores para cabeceras
(mayores que las de este fixture). El siguiente tamaño 40960 requiere
6.710.886.400 bytes solo para A: se excluye del plan de 3 GB.
No se materializaron tamaños por encima de 80. La simulación JSON incluye
`max(FLOPs/compute,traffic/bandwidth)` con supuestos ilustrativos de 10 GFLOP/s y
20 GB/s, **no calibrados** a este host: son cotas ideales, no predicciones ni
benchmarks. No ejecutar matmul 20480 sin un presupuesto independiente de tiempo,
RAM total y tráfico. El límite de archivo no autoriza su ejecución.

## Reproducir

```sh
cargo build -p bml-gguf --example synthetic_export --release --locked --offline
# Preparar una imagen limpia local con el digest indicado, sin BML ni toolchain.
python3 tools/synthetic_aot.py --sizes 10 20 40 80 --seed 20261030 \
  --clean-image bml-aot-clean:local --output /ruta/nueva/resultados
```

Sin `--clean-image` se comprueba solo host y el campo de aislamiento queda sin
verificar. La imagen no se descarga automáticamente. Los resultados locales de
esta corrida están en
`/workspace/.bml-env/deliverables/synthetic-aot-static-final-20261030`;
son rutas de este ejecutor, no adjuntos remotos. `summary.json` separa mediciones
de estimaciones; cada subdirectorio conserva fixture GGUF, kernel C, ejecutable
y measurement.json con entradas y resultados.

F16/Q4_0/Q8_0 son etapas futuras. El exportador acepta solo F32. Q4_0/Q8_0
requieren filas completas de 32 elementos: para n=10 rechazar o diseñar padding
explícito con dimensiones lógicas/físicas y tolerancias antes de implementarlos.

## Matriz fija y complejidad

A queda fija durante la compilación. x/B se inyecta al ejecutar: no se
precomputa el resultado de una entrada constante. Para los loops clásicos
implementados, matvec es Θ(n²), matmul Θ(n³) y almacenamiento denso Θ(n²).
No son cotas inferiores universales para especializaciones de matrices fijas.
Si ambos operandos fueran constantes, se podría precomputar el producto;
materializar una salida densa n×n sigue costando Θ(n²), mientras que devolver
solo una referencia es otro contrato. Ese caso no representa esta prueba.

Generación, importación BML, emisión C y compilación se miden por separado.
Tiempos de ejecución son por una entrada dinámica y contienen inicio/E/S; no
se infiere aceleración a partir de una sola muestra ni se compara con TPS.

## Checks de cierre

`CARGO_NET_OFFLINE=true bash scripts/check.sh 069b5d691423b976c0c4b11a8b13bb6c8b621702`:
312 pruebas Rust y 10 Python aprobadas, 15 Rust opcionales ignoradas, cero
discrepancias de contratos, trazabilidad 15/15, formato correcto y cero nuevos
diagnósticos Clippy (123 existentes). `cargo clippy -p bml-gguf --examples
--locked --offline -- -D warnings` pasó. Cobertura workspace offline: 59,1222 %
líneas / 67,4444 % funciones, sobre mínimos 52/60. La instrumentación no mide
el C generado ni sustituye sus pruebas de ejecución. Enumeración cargo-mutants
incremental: sin mutantes aplicables en este alcance de ejemplos; no se ejecutó
una campaña Python/C ni se atribuye 100 % de detección. Se verificó además rechazo
del exportador a fixture F16 10×10 y del CLI a tamaño real 40960 antes de crear
archivos. No se ejecutaron tamaños grandes, inferencia, GPU ni tests de modelos.
