# Hardware y entorno de benchmark

Fecha: 2026-07-07

## Hardware

| Componente | Valor |
|---|---|
| CPU | Intel Xeon Processor (Cascadelake) |
| Arquitectura | x86_64 |
| Sockets | 1 |
| Cores por socket | 2 |
| Threads por core | 2 |
| vCPUs totales | 4 |
| L1d | 128 KiB (4 instancias) |
| L1i | 128 KiB (4 instancias) |
| L2 | 8 MiB (2 instancias) |
| L3 | 16 MiB (1 instancia) |
| RAM | 7.8 GiB |

## Software

| Componente | Versión |
|---|---|
| OS | Debian GNU/Linux 13 (trixie) |
| Kernel | 6.12.94+deb13-amd64 |
| Rust | 1.96.0 (30a34c682 2026-05-25) |
| cargo | 1.96.0 |
| rust-analyzer | 1.96.0 (ac68faa 2026-05-25) |
| perf | 6.12 |
| strace | disponible |
| cmake | disponible |
| gcc/g++ | disponible |

## llama.cpp

| Componente | Valor |
|---|---|
| Repositorio | `/root/llama.cpp` |
| Build commit | f36e5c3 |
| Backend | CPU |
| Binarios | `./build/bin/llama-bench` |

## Modelo de prueba

| Componente | Valor |
|---|---|
| Archivo | `/root/tinyllama.gguf` |
| Tipo | TinyLlama-1.1B |
| Cuantización | Q4_0 |
| Tamaño | 635,998,512 bytes (~606 MB) |
| Parámetros | 1,100,060,672 (~1.1B) |
