## Context

La referencia F32 de `crates/inference/src/model.rs` convierte operandos F32 a
F64, reduce por fila con `sum::<f64>()` y convierte el resultado a F32. El contrato
de esta propuesta fija Rust 1.98.1, su semilla `-0.0` y el orden de la reducción.
Se deberá comprobar el entrypoint efectivo de cada consumidor: otras rutas del
repositorio usan activaciones F64 y un fold explícito distinto. No se sustituirá
silenciosamente una referencia por otra.

La CLI nativa y un ELF autónomo son consumidores de inferencia nativa. Un
artefacto BMLX simbólico tiene programas, controles de dominio y presupuestos
distintos. La extensión a ese backend necesita una decisión posterior a la
operatividad y medición de 8.6, no una extrapolación del ELF.

## Goals / Non-Goals

- G1: especialización optativa y comprobable de cabeza/FFN F32 en CLI y ELF.
- G2: comparar escalar, AVX y sólo cuando corresponda AVX2, con fallback seguro.
- G3: conservar bits, orden, precisión, límites y errores de la referencia exacta.
- G4: medir preparación y ejecución con procedencia e ISA efectivas verificadas.
- NG1: implementar SIMD, modificar defaults o completar 8.6 en esta propuesta.
- NG2: atribuir inferencia F32 al comparador de limbs o a kernels elementales F64.
- NG3: prometer velocidad, residencia en caché o rendimiento de BMLX sin evidencia.

## Decisions

### D1 — Separar ISA disponible, solicitada y ejecutada

Se expondrán control escalar y selección optativa de variante, conservando el
default vigente. El despacho comprobará capacidades de CPU y soporte del sistema
operativo para estado vectorial; el fallback deberá funcionar en x86-64
convencional. Una variante forzada no disponible fallará explícitamente. Se
registrarán selección solicitada/efectiva y las instrucciones relevantes del
binario enlazado y del camino ejecutado. La CPU Ryzen con AVX2 no basta como prueba.

AVX es la primera variante candidata. AVX2 tendrá una tarea condicionada a
identificar y validar una operación que realmente lo necesite, incluyendo el
beneficio y coste de packing/gather si se utiliza. Si todo el trabajo requiere
sólo AVX, se conservará esa etiqueta y se documentará que AVX2 no fue necesario;
no se fabricará una variante distinta únicamente para cambiar el nombre.

### D2 — Cuatro salidas independientes, orden escalar dentro de cada fila

Cada lane mantendrá una fila y recorrerá columnas en el orden de la referencia,
con ambos operandos F32 convertidos a F64 antes del producto, acumulador F64
inicializado a `-0.0`, multiplicación y suma separadas, y cast
final a F32. No habrá reducción horizontal que reasocie productos ni FMA implícito
en la ruta exacta. Las colas de filas y columnas no múltiplos de cuatro, offsets
sin alineación especial y tamaños mínimos conservarán el mismo contrato.

Los oráculos incluirán ceros con signo, productos y sumas con cancelación sensible,
subnormales, magnitudes y signos distintos, redondeo del cast final y errores de
dimensión/dominio aplicables. Se compararán bits F32 de salida, no sólo una
tolerancia. Los tests deberán distinguir un acumulador `+0.0`, una reducción
reasociada y FMA de la referencia cuando estos produzcan un resultado distinto.

### D3 — Packing y presupuestos visibles

Una disposición por cuatro filas podrá necesitar packing o gather. Se medirá
construcción, copia, memoria adicional, asignaciones, reutilización e invalidación
de pesos preparados. Se mantendrán cotas de entrada, memoria, trabajo y fallos
limpios; la preparación no quedará oculta en otro backend o fuera del resultado
completo. También se reportará ejecución con pesos ya preparados, claramente
separada del coste total.

### D4 — Experimento y comparación justos

La matriz controlada fijará GGUF y hash, revisión, toolchain/flags, prompts e IDs,
BOS/plantilla, contexto, KV inicial, greedy, cantidad de tokens nuevos, mismo
núcleo/afinidad y presupuesto de CPU. Se alternará el orden con repeticiones y
calentamiento explícitamente contabilizado. Se reportarán TTFT/prefill, latencia
por token, decode sostenido excluyendo BOS/prompt/EOS, dispersión y memoria.

La corrección completa de logits, KV e IDs precederá a cualquier interpretación
de rendimiento; si se conserva evidencia fuera de la ventana, se declararán las
copias que sigan dentro. El aislamiento por ISA usará la misma implementación y
flags salvo la variante probada. Stock llama.cpp, con su revisión/flags e ISA
propios, se publicará en una tabla separada: su diferencia no es una medición
aislada del efecto AVX/AVX2 en BML.

### D5 — FMA y BMLX requieren decisiones separadas

FMA será un experimento explícito, separado de la ruta exacta, con detección de
capacidad, política numérica previa y evidencia de equivalencia o divergencias.
El producto de dos F32 finitos promovidos a F64 tiene como máximo 48 bits de
significand y cabe también por rango en F64: no tiene el redondeo intermedio que
habitualmente distingue FMA de multiplicación/suma separadas. Se admitirá una
equivalencia justificada para este dominio y orden, sin inventar contraejemplos
fuera del contrato; los ceros con signo y el dominio efectivo seguirán cubiertos.
No heredará una afirmación de paridad sin esa justificación y verificación.
Una posible extensión a BMLX se
decidirá sólo cuando pueda medirse su recorrido simbólico operativo: deberá
preservar cada operación BML, orden, dominio, cotas locales/globales y primer
error. Si no es viable, permanecerá diferida con el bloqueo concreto.

## Risks / Trade-offs

- El packing puede consumir más tiempo o memoria que el ahorro del kernel.
- Vectorizar filas preserva el orden pero no garantiza una ventaja en todas las
  formas, colas, tamaños de caché o distribuciones de pesos.
- La selección del compilador puede emitir instrucciones distintas de las
  esperadas; los flags y el nombre del backend no sustituyen su verificación.
- La reasociación o una semilla incorrecta pueden alterar signed zero y bits;
  FMA requiere su evaluación separada según el dominio efectivo de operandos.
- Un ELF nativo correcto y rápido no demuestra que la ruta BMLX esté operativa.
