# bml-bench — BML Benchmark Report

## Hot loop (raw, single-thread)

| Métrica | Valor |
|---|---|
| Ops/seg | 607604245 ± 20443154 |
| Tiempo/op | 1.646 ns |
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
| tokens/seg | 0.439557 ± 0.100269 |
| ns (muestra) | 218678310 ± 53399928 |
| samples_ts | [0.5876641329762639, 0.5053303289127498, 0.40450277408245167, 0.2918820153933613, 0.40840690186034917] |

## Generation (tg=128 tokens)

| Métrica | Valor |
|---|---|
| tokens/seg | 0.505955 ± 0.062827 |
| ns (muestra) | 182649115 ± 23987354 |
| samples_ts | [0.4132225093923989, 0.5767230066103034, 0.5546937362179997, 0.4513664176744747, 0.5337684767681269] |

## Multicore scaling

| Threads | Ops/seg | Tokens/seg (extrapolado) | Speedup |
|---|---|---|---|
| 1 | 627788421 ± 16241945 | 0.570717 | 1.00x |
| 2 | 1187577217 ± 71640411 | 1.079616 | 1.89x |
| 4 | 2175162378 ± 103236747 | 1.977420 | 3.46x |

