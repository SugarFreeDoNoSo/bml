# bml-bench — BML Benchmark Report

## Hot loop (raw, single-thread)

| Métrica | Valor |
|---|---|
| Ops/seg | 593941345 ± 40903199 |
| Tiempo/op | 1.684 ns |
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
| tokens/seg | 0.580913 ± 0.016048 |
| ns (muestra) | 156612208 ± 4297355 |
| samples_ts | [0.600308224535977, 0.5997051705452359, 0.5641132115803239, 0.56534785658005, 0.5750900223780065] |

## Generation (tg=128 tokens)

| Métrica | Valor |
|---|---|
| tokens/seg | 0.564246 ± 0.016487 |
| ns (muestra) | 161255920 ± 4784525 |
| samples_ts | [0.5777984243130736, 0.5747858525665862, 0.539181237535926, 0.5794909203209796, 0.5499723305620661] |

## Multicore scaling

| Threads | Ops/seg | Tokens/seg (extrapolado) | Speedup |
|---|---|---|---|
| 1 | 583907230 ± 20716089 | 0.530825 | 1.00x |
| 2 | 989002001 ± 171666704 | 0.899093 | 1.69x |
| 4 | 1956775766 ± 98593334 | 1.778887 | 3.35x |

