# Reproducción del límite de rango en RMSNorm simbólico

Ejecutor y lockfile: `target/quality/symbolic-range-probe/`. Construye el DAG
real de `BMLTransformer::rms_norm`, evalúa todos sus nodos topológicamente y
registra la primera operación no finita, además de una referencia aritmética
independiente. No modifica ni recompila los artefactos AOT anteriores.

| Caso | Primer fallo | Resultado de referencia |
| --- | --- | --- |
| Stories15M, primer BOS, 288 componentes | Nodo 110395: `BML(2526102804.114869, 1)` desborda. Primer output inválido: índice 167, input 0.10322366654872894, peso 0.5973223447799683. | Output 2.608945943403005; todos los resultados nativos finitos. |
| Un elemento, input 1, peso 4, epsilon 1e-5 | Nodo 662: `BML(4294637102.9411016, 1)` desborda. | 3.999980000149999. |
| Control: input 0.01, peso 1, epsilon 1e-5 | Ningún nodo no finito. | Inverso de la raíz 95.34625892455924; salida finita. |

El caso mínimo reproduce el problema sin depender de hardware o modelo externo.
Las expansiones de productos firmados forman sumas positivas que vuelven a
exponenciar valores grandes; un resultado nativo finito no garantiza la
evaluabilidad de esa expansión. Esta evidencia mantiene **8.6 abierta**.

Comando aprobado como **diagnóstico**, no como validación de inferencia:

```sh
cargo run --manifest-path target/quality/symbolic-range-probe/Cargo.toml --locked -- target/bml-1.0/stories15M.gguf
```

Registro: `target/quality/symbolic-norm-diagnostic-v2.json`, SHA-256
`635da84a201e96553763435e83243b8458fc7d05a31f3d32afa28f86f2163608`.
Log: `target/quality/symbolic-norm-diagnostic-v2.log`, SHA-256
`fb22f9d5e0bd1dd601ee50dfe54a71653fc89880166d6c9e12d226475ad87ff2`.
El registro conserva hashes del probe, lockfile y ambas fuentes de construcción.
Las fuentes del diagnóstico de modelos anterior se restauraron y verificaron
contra `compiler-entrypoint-tinyllama-diagnostic-v2.json`.
