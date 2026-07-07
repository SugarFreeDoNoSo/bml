# bml-bench — BML Benchmark Report

## Hot loop (raw)

| Métrica | Valor |
|---|---|
| Ops/seg | 511731499 ± 36231278 |
| Tiempo/op | 1.954 ns |
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
| tokens/seg | 0.480253 ± 0.011733 |
| ns (muestra) | 189406935 ± 4613015 |
| samples_ts | [0.4807183552825641, 0.46835309870367425, 0.4884608390194335, 0.4972589028138747, 0.4664726900937343] |

## Generation (tg=128 tokens)

| Métrica | Valor |
|---|---|
| tokens/seg | 0.461278 ± 0.074157 |
| ns (muestra) | 203920819 ± 42612867 |
| samples_ts | [0.47508496433754765, 0.3149348661661737, 0.5082096548472885, 0.5074551036125332, 0.500703712530665] |

