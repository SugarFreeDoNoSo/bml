BML FINAL BENCHMARK REPORT
================================================================================

Hot loop performance (local machine):
  Ops/second: 604016231
  Time per op: 1.656 ns
  Sample size: 100000 ops x 1000 iterations
  Elapsed: 165.558465ms

================================================================================
MODEL ANALYSIS (extrapolated from hot loop ops/sec)
================================================================================

Model: TinyLlama-1.1B (1.1B params, 0.6GB Q4)
  FLOPs/token: 2.20e9
  BML ops/token: 1.10e9
  Tokens/second (1 machine): 0.5491

  Provider                            vCPUs     $/hr  machines@20tpS     $/Mtok
  --------------------------------------------------------------------------------
  Hetzner CCX13 (4 vCPU)                  4    0.064            37.0      32.38
  Hetzner CCX33 (16 vCPU)                16    0.193            10.0      24.41
  Hetzner CCX63 (32 vCPU)                32    0.386             5.0      24.41
  Vultr HP-4 (4 vCPU)                     4    0.179            37.0      90.55
  Vultr HP-16 (16 vCPU)                  16    0.714            10.0      90.30
  GCP N2D-4 (4 vCPU)                      4    0.097            37.0      49.07
  GCP N2D-16 (16 vCPU)                   16    0.388            10.0      49.07
  AWS c7i-4xlarge (16 vCPU)              16    0.720            10.0      91.06

Model: Llama-7B (7B params, 3.5GB Q4)
  FLOPs/token: 1.40e10
  BML ops/token: 7.00e9
  Tokens/second (1 machine): 0.0863

  Provider                            vCPUs     $/hr  machines@20tpS     $/Mtok
  --------------------------------------------------------------------------------
  Hetzner CCX13 (4 vCPU)                  4    0.064           232.0     206.03
  Hetzner CCX33 (16 vCPU)                16    0.193            58.0     155.33
  Hetzner CCX63 (32 vCPU)                32    0.386            29.0     155.33
  Vultr HP-4 (4 vCPU)                     4    0.179           232.0     576.24
  Vultr HP-16 (16 vCPU)                  16    0.714            58.0     574.63
  GCP N2D-4 (4 vCPU)                      4    0.097           232.0     312.26
  GCP N2D-16 (16 vCPU)                   16    0.388            58.0     312.26
  AWS c7i-4xlarge (16 vCPU)              16    0.720            58.0     579.45

Model: Llama-13B (13B params, 6.5GB Q4)
  FLOPs/token: 2.60e10
  BML ops/token: 1.30e10
  Tokens/second (1 machine): 0.0465

  Provider                            vCPUs     $/hr  machines@20tpS     $/Mtok
  --------------------------------------------------------------------------------
  Hetzner CCX13 (4 vCPU)                  4    0.064           431.0     382.62
  Hetzner CCX33 (16 vCPU)                16    0.193           108.0     288.46
  Hetzner CCX63 (32 vCPU)                32    0.386            54.0     288.46
  Vultr HP-4 (4 vCPU)                     4    0.179           431.0    1070.15
  Vultr HP-16 (16 vCPU)                  16    0.714           108.0    1067.16
  GCP N2D-4 (4 vCPU)                      4    0.097           431.0     579.91
  GCP N2D-16 (16 vCPU)                   16    0.388           108.0     579.91
  AWS c7i-4xlarge (16 vCPU)              16    0.720           108.0    1076.13

Model: Llama-70B (70B params, 35.0GB Q4)
  FLOPs/token: 1.40e11
  BML ops/token: 7.00e10
  Tokens/second (1 machine): 0.0086

  Provider                            vCPUs     $/hr  machines@20tpS     $/Mtok
  --------------------------------------------------------------------------------
  Hetzner CCX13 (4 vCPU)                  4    0.064          2318.0    2060.28
  Hetzner CCX33 (16 vCPU)                16    0.193           580.0    1553.26
  Hetzner CCX63 (32 vCPU)                32    0.386           290.0    1553.26
  Vultr HP-4 (4 vCPU)                     4    0.179          2318.0    5762.35
  Vultr HP-16 (16 vCPU)                  16    0.714           580.0    5746.26
  GCP N2D-4 (4 vCPU)                      4    0.097          2318.0    3122.62
  GCP N2D-16 (16 vCPU)                   16    0.388           580.0    3122.62
  AWS c7i-4xlarge (16 vCPU)              16    0.720           580.0    5794.55

================================================================================
COMPARISON WITH LLAMA.CPP (TinyLlama-1.1B Q4_0, 4 vCPU)
================================================================================

llama.cpp prompt processing: 148.34 tokens/sec
llama.cpp text generation:   30.43 tokens/sec
BML (extrapolated):          0.549106 tokens/sec

Ratio BML/llama.cpp (generation): 0.0180x

NOTE: BML is currently much slower because:
  1. Each bml op does exp2+log2 (~5ns) vs FMA (~3ns)
  2. The RPN interpreter has overhead (push/pop, match dispatch)
  3. No SIMD, no BLAS, no flash attention
  4. The hot loop uses Vec<f64> instead of fixed buffer

Projected performance with optimizations:
  - Hot loop native (no Vec): ~2x faster
  - SIMD (4x f64 per op): ~4x faster
  - exp2/log2 bit-twiddling: ~2x faster
  - Combined: ~16x faster = 8.7857 tokens/sec

