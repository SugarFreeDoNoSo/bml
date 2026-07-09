# Hardware y entorno de benchmark

Fecha: 2026-07-09

## Hardware

| Componente | Valor |
|---|---|
| CPU | Intel Xeon Processor (Cascadelake) |
| Arquitectura | x86_64 |
| vCPUs totales | 4 (2 cores físicos, hyperthreading) |
| L1d | 128 KiB (4 instancias, 32 KB/core) |
| L1i | 128 KiB (4 instancias, 32 KB/core) |
| L2 | 8 MiB (2 instancias) |
| L3 | 16 MiB (1 instancia) |
| RAM | 7.8 GiB |

## Software

| Componente | Versión |
|---|---|
| OS | Debian GNU/Linux 13 (trixie) |
| Kernel | 6.12.94+deb13-amd64 |
| Rust | 1.96.0 |
| llama.cpp | build f36e5c3, CPU backend |

## Modelo

| Componente | Valor |
|---|---|
| Archivo | `/root/tinyllama.gguf` |
| Tipo | TinyLlama-1.1B |
| Cuantización | Q4_0 |
| Tamaño | 606 MB |
| Parámetros | 1.1B |

## Optimizaciones BML aplicadas

1. Hot loop refactorizado: dispatch_ops único (429 líneas asm)
2. Sub-fragmentación L1i: sub-fragmentos de <30 KB
3. Pesos BML nativos: BmlWeightPool (8x compresión vs f32)
4. Scheduler de waves: DAG con dependencias
5. Building blocks BML: bml_matmul, bml_rmsnorm, bml_rope, bml_swiglu
6. VectorFragment: distribución de columnas de matmul via TCP
7. Worker daemon: ejecuta VectorFragments remotamente
