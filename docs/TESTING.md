# Estrategia de pruebas para la biblioteca

Cada prueba debe comprobar un contrato observable con un resultado independiente.
El número de tests y un proceso sin panic no acreditan corrección numérica.

## Capas y criterios

| Capa | Casos mínimos | Oráculo / criterio |
|---|---|---|
| Álgebra simbólica | bind vacío/parcial/total, variable repetida, entornos disjuntos y colisiones, collapse | Leyes estructurales; conjunto exacto de variables libres |
| Primitiva EML/BML | valores conocidos, cero, negativos, subnormales, NaN/inf, overflow/underflow | Política numérica explícita; alta precisión en zonas finitas |
| Reescrituras | suma/producto con signos, `0.5 * 2`, cancelación, extremos | Operación original independiente; dominio de cada intermedio |
| Tensores | shape, rank cero, dimensiones cero, overflow, reshape, orden de índices | Datos manuales y shape esperado; errores tipados |
| Kernels | dot, matmul rectangular, elementwise; contigüidad | Implementación escalar independiente; luego BLAS/ggml como diferencial |
| GGUF | F32/F16/Q4_0/Q8_0, metadata, alignments 32/64, varios tensores | Fixtures pequeñas con valores completos y rangos exactos |
| GGUF inválido | truncamiento, overflow de tamaños, contadores absurdos, UTF-8 inválido, dtype/endian no soportado | Error tipado; sin panic, lectura fuera de rango o reserva ilimitada |
| Compilación | nodos compartidos no adyacentes, pool de constantes, validación de IDs | AST frente a plan ejecutado sobre los mismos inputs y pool |
| Runtime | efectos de pila, loops, buffers, fragmentación, capacidad insuficiente | Resultados y límites explícitos; rechazar programa inválido |
| Concurrencia | estado mutable realmente compartido, cola y cancelación | Loom con `loom::model` y primitivas instrumentadas |
| Inferencia opcional | tokenizer, tensores intermedios, logits, secuencia determinista | Mismo GGUF/hash, tokenizer, contexto y parámetros que la referencia |

La suite nueva `crates/domain/tests/symbolic_contracts.rs` implementa la primera
fila y el contrato de errores de evaluación real. La auditoría ejecutable
`crates/tests/examples/contract_audit.rs` expone fallos anteriores y sale con
error mientras persistan; no es una suite aprobada de la biblioteca futura.

## Comparación numérica

Para resultados esperados finitos, comprobar primero que ambos valores son
finitos y después `abs(actual - expected) <= atol + rtol * abs(expected)`.
`atol` y `rtol` se justifican por dtype, operación y tamaño de reducción. Un
único umbral absoluto global no sirve para escalas muy diferentes.

Para dot/matmul, considerar además una cota proporcional a la suma de los valores
absolutos de los productos: la cancelación puede producir un resultado pequeño
con error relativo grande. Documentar acumulación y uso de FMA. Reservar igualdad
de bits para formatos, enteros y determinismo prometido; no exigirla entre kernels
con distinto orden de reducción. NaN/inf/signed-zero se prueban en casos separados.

Las propiedades generan entradas dentro del dominio declarado y también casos
fuera de él, que deben producir el error esperado. Conservar semillas y casos
reducidos de proptest en git: la exclusión actual de `proptest-regressions/` debe
retirarse al migrar. No descartar silenciosamente los casos difíciles.

## Fixtures y oráculos

- Incluir GGUF pequeños reproducibles, sin modelos descargados en cada PR.
- Una fixture F32 rectangular debe detectar orientación y orden de dimensiones.
- Q4_0 debe probar todos los nibbles y dos escalas diferentes. Los primeros
  16 valores vienen de los nibbles bajos y los siguientes 16 de los altos.
- Los resultados esperados se escriben independientemente del decoder bajo test.
  Los fixtures diferenciales externos registran versión/commit, SHA-256 y origen.
- Separar el generador de fixtures de la lógica de lectura; un round-trip puede
  ocultar el mismo defecto en encoder y decoder.
- Un test de modelo grande usa `#[ignore = "requires BML_TEST_GGUF"]` y falla
  claramente si se invoca sin fixture. Evitar `return` exitoso por falta de archivo.

Fuentes de formato: [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
y [dequantización GGML](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c).

## Organización y ejecución

Los tests unitarios privados permanecen junto al módulo; los contratos públicos
en `crates/<owner>/tests/`; las pruebas entre crates en `crates/tests`; rendimiento
en `benches/`; fuzzing en `fuzz/` cuando exista el reader comprobado de bytes.
Los `[[test]]` actuales de `crates/tests/Cargo.toml` ya registran los archivos de
`tests/` raíz: no están ausentes de Cargo. Se pueden mover por claridad después.

Comandos disponibles en esta rama:

```sh
cargo test -p bml-domain --test symbolic_contracts --locked
cargo test -p bml-domain --doc --locked
cargo run -p bml-tests --example contract_audit --locked
cargo test --workspace --locked
```

El diagnóstico tiene salida no cero intencional ante discrepancias; no se debe
convertir en un check verde permitiendo resultados no finitos. Al corregir cada
caso, añadir un test de regresión en el crate propietario y mantener el diagnóstico
hasta completar la migración.

Propuesta de gates de CI, por introducir gradualmente:

| Gate | Frecuencia | Condición |
|---|---|---|
| Formato, check, unitarios, contratos y GGUF sintético | Cada PR; Linux y macOS para portabilidad | Deterministas y sin requerir `/root/tinyllama.gguf` o ptrace |
| Compiler/runtime experimental | Cada PR que los modifica | Estado visible separado del MVP; no esconder fallos |
| Fuzz parser/IR; Miri donde aplique | Programado | Corpus versionado, sin panic/UB en entradas arbitrarias |
| Loom | Cambios de concurrencia | Código compartido instrumentado; no solo `std::thread` |
| Modelo real y comparación diferencial | Ejecución optativa con fixture fijada | Calidad y resultados antes de rendimiento |
| Benchmarks/PMU/strace | Runner controlado | Capacidades verificadas; entorno registrado |

El test actual con strace traza `cat`, no el acceso del parser. Su fallo por ptrace
en un contenedor no demuestra un error de GGUF. El reemplazo debe trazar un proceso
del parser y separar ausencia de capacidad de fallo funcional. No hacer depender
todos los tests ordinarios de esa herramienta.

## Metodología de rendimiento

1. Validar igualdad semántica y presupuesto de error antes de cronometrar.
2. Usar entradas dinámicas y observar la salida (`black_box`); conservar pools,
   dimensiones y bytes realmente recorridos. Evitar un grafo constante que se
   pliega en compilación o una cadena que se satura en infinito.
3. Medir separadamente lectura, dequantización, compilación, preparación y ejecución.
4. Comparar la misma operación con los mismos dtype, shapes, hilos y entradas.
5. Registrar hardware, compilador, perfil, flags, tamaño, calentamiento, repeticiones,
   dispersión, memoria y asignaciones. Medir tiempo total y tiempo por elemento.
6. Llamar `ops/s` al throughput del operador. Publicar `tokens/s` solo cuando se
   ejecuta y valida el mismo trabajo de inferencia. Las proyecciones van aparte.
7. Distinguir bytes del bytecode (datos), código máquina del intérprete (L1i) y
   datos de trabajo (L1d/L2/L3). Contar líneas de assembly no mide bytes de código.
8. Probar “cero allocs” con un contador de asignaciones en la región de ejecución,
   después de preparar buffers; comprobar solo la capacidad de una pila no basta.

No hay un test universal “BML debe ser más rápido”. Una optimización se acepta
para una carga y un hardware declarados, con resultados correctos y una mejora
repetible. Mantener disponible la ruta CPU nativa como referencia.
