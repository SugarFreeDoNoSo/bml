# bml-bench — BML Benchmark Report

## Hot loop (raw, single-thread)

| Métrica | Valor |
|---|---|
| Ops/seg | 622383774 ± 24847319 |
| Tiempo/op | 1.607 ns |
| Programa | 100K ops × 1000 iters/muestra |
| Repeticiones | 5 |

## Modelo: TinyLlama-1.1B

| Campo | Valor |
|---|---|
| Parámetros | 1.1B |
| FLOPs/token | 2.20e9 |
| BML ops/token | 1.10e9 |
| Tamaño Q4 | 0.6 GB |

## Prompt processing (pp=512 tokens)

| Métrica | Valor |
|---|---|
| tokens/seg | 0.573371 ± 0.007414 |
| ns (muestra) | 158578208 ± 2023804 |
| samples_ts | [0.5704528408558891, 0.5664717144253805, 0.5680893367883474, 0.5747204450127533, 0.5871190609414362] |

## Generation (tg=128 tokens)

| Métrica | Valor |
|---|---|
| tokens/seg | 0.554241 ± 0.023714 |
| ns (muestra) | 164332888 ± 7210320 |
| samples_ts | [0.5766802907500692, 0.5167928717064125, 0.5733425280274553, 0.5358762631184948, 0.5685132258604182] |

## Multicore scaling

| Threads | Ops/seg | Tokens/seg (extrapolado) | Speedup |
|---|---|---|---|
| 1 | 543830353 ± 48374514 | 0.494391 | 1.00x |
| 2 | 1145415489 ± 53216098 | 1.041287 | 2.11x |
| 4 | 2172277495 ± 92364961 | 1.974798 | 3.99x |

