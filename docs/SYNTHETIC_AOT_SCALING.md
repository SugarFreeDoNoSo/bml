# Escalado medio del gate AOT sintético

30 de octubre de 2026. Extensión del gate F32 de 10–80; no inferencia ni benchmark de un transformer.
La matriz A queda fija en compilación. x/B se inyecta en runtime, en dos variantes distintas por operación.
Pesos F32, activaciones/acumulación F64; semilla 20261030, NumPy PCG64; referencia NumPy 2.3.5 float64 BLAS con un hilo, independiente del C clásico generado.

## Construcción y autonomía

`tools/synthetic_scale.py` importa GGUF v3 real mediante bml-gguf y verifica su payload/forma con el lector Python independiente. Los pesos se enlazan mediante assembler `.incbin` a una sección ELF `.rodata` alineada: no se emiten literales C por peso. La complejidad del fuente C es constante respecto al número de pesos, pero el tamaño del binario sigue siendo Θ(n²).
El GGUF y weights.bin se retiran antes de ejecutar. Cada prueba aislada monta únicamente el binario, sin red, sin BML, sin toolchain ni GGUF. La imagen limpia conserva libc y loader Linux; la salida es ELF Linux/amd64 dinámico, no un ejecutable universal o totalmente estático. El fixture se restaura después para entrega.

## Medido

Dos entradas por operación; se muestran intervalos entre ambas, no intervalos estadísticos. Tiempo kernel por CLOCK_MONOTONIC alrededor de los loops, excluye parseo, asignación y serialización; proceso host incluye inicio/E/S. No presentar la diferencia de arranque como aceleración BML.

| n | Operación | Kernel ms | Proceso host ms | Pico RSS aislado KiB | Binario bytes |
|---:|---|---:|---:|---:|---:|
| 160 | matvec | 0.079–0.080 | 2.832–3.320 | 6948–7204 | 119056 |
| 160 | matmul | 1.870–2.234 | 20.905–22.747 | 6692–6820 | 119056 |
| 320 | matvec | 0.324–0.521 | 4.972–6.409 | 6820–6820 | 426256 |
| 320 | matmul | 11.945–14.146 | 64.971–68.935 | 6820–7076 | 426256 |
| 640 | matvec | 1.480–1.850 | 4.030–5.296 | 6692–6820 | 1655056 |
| 640 | matmul | 100.799–103.399 | 323.143–325.946 | 9544–9616 | 1655056 |
| 1280 | matvec | 6.542–7.415 | 12.818–13.022 | 7924–7976 | 6570256 |
| 1280 | matmul | 914.087–982.674 | 1780.540–1890.032 | 33536–33564 | 6570256 |
| 2560 | matvec | 25.822–46.464 | 34.294–57.205 | 27072–27340 | 26231056 |

Los 18 casos positivos aislados (9 pares operación/tamaño × 2 entradas) pasaron. Todos los errores máximos fueron cero con estas entradas binarias exactas; eso no implica igualdad general para cualquier dato. Los hashes de salida difieren al cambiar la entrada runtime, sin recompilar. Umbrales atol=rtol=1e-9.

## Límites y parada

Compilación: 512 MiB de espacio virtual, 30 s CPU y 30 s de pared por comando. Ejecución host: 10 s de pared; contenedor: 128 MiB, sin swap, una CPU, 15 s de pared. Driver: preflight de al menos 1 GiB de holgura y cota conservadora de 512 MiB para su workspace. No se retiraron límites para terminar la escala.
Matmul: máximo realmente ejecutado 1280. En 2560, A F32 + B/C F64 + 16 MiB de reserva requieren 147.849.216 bytes > 134.217.728 bytes; se marca no ejecutado antes de crear esas entradas. No hubo timeout en los tamaños aceptados.
Matvec: máximo realmente ejecutado 2560. El siguiente 5120 requiere 104.857.600 bytes de pesos > 64 MiB y la cota conservadora de temporales del driver supera 512 MiB; no se materializa. El límite de 3 GB por GGUF/matriz sigue siendo un techo de formato/entrega, no permiso para asignar esa RAM ni garantía de poder ejecutar matmul.
RSS por proceso aislado está en la tabla; snapshots de memory.current/memory.peak y MemAvailable por etapa están en los JSON. El cgroup es de todo el entorno, su pico incluye otros procesos y no se atribuye al kernel. El RSS del ejecutable tampoco es una medida del compilador ni del driver.

## Integridad y casos negativos

En n=160 se alteraron bytes del payload y se transpuso A sin cambiar dimensiones; bml-gguf puede leer esos GGUF válidos, pero la comparación con la identidad del payload esperada por la semilla detecta ambos. GGUF no proporciona por sí solo un checksum que pruebe la intención del layout. No se afirma que el lector rechace cualquier contenido numéricamente modificado.
Dimensiones inválidas y archivo ausente fueron rechazados por el importador; truncamiento por el lector independiente. Los tests de presupuestos verifican paradas distintas para matvec y matmul. La integridad del recurso incrustado al arrancar aún no implementa el footer/hash de R3: permanece pendiente.

## Reproducir

NumPy es dependencia del driver de referencia, no del ejecutable; no se instala ni se descarga automáticamente. Los tests de contrato que no usan NumPy siguen funcionando sin esa dependencia. Usar un directorio nuevo para cada corrida.

```sh
cargo build -p bml-gguf --example synthetic_export --release --locked --offline
python3 tools/synthetic_scale.py --clean-image bml-aot-clean:local --output /ruta/nueva/escala
```

Preparar la imagen limpia conforme a `SYNTHETIC_AOT_VALIDATION.md`; no usar como runtime limpio la imagen del probe que contiene BML. Sin --clean-image la independencia del entorno queda sin verificar.

## Diagnóstico de publicación

Solo lectura local, sin nueva solicitud HTTP ni mutación: política restringida aplicada; github.com aparece permitido y api.github.com no aparece en las reglas locales ni en la lista efectiva reportada. GH_TOKEN está presente, GITHUB_TOKEN no; no se leyó ni mostró contenido, ni se asumió que sea un token crudo o tenga las credenciales del conector del coordinador.
El error retenido es `Post https://api.github.com/graphql: Forbidden`; no hay headers retenidos para atribuir definitivamente el rechazo a GitHub o al proxy. La configuración visible explica un posible bloqueo de destino y requiere resolver la ruta autorizada antes de publicar; no se cambiaron permisos, destinos o credenciales.

Fuentes/mediciones nuevas se conservan separadas de la imagen/release autorizada 21b90c1. No se incorporaron pesos descargados ni se publicó ningún artefacto nuevo.

## Checks de cierre

Suite agregada offline contra base 71ea055: 312 pruebas Rust aprobadas, 13 Python
aprobadas, 15 Rust opcionales ignoradas; cero discrepancias de contratos,
trazabilidad 15/15, formato correcto y Clippy sin diagnósticos nuevos (123
existentes). Clippy estricto de ejemplos bml-gguf pasó. Cobertura workspace
offline 59,1333 % líneas / 67,4444 % funciones, por encima de 52/60; no mide
el C generado. No se ejecutó campaña completa de mutación Python/C.
