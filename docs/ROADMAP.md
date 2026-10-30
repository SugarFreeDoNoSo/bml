# Roadmap BML: GGUF → ejecutable autónomo

Actualizado: 30 de octubre de 2026. Primer target propuesto: **Linux/amd64 CPU**.

## Objetivo y definición de terminado

BML es el compilador AOT usado durante la construcción: recibe un GGUF y genera
un ejecutable para un target concreto. Ese ejecutable incorpora pesos, tokenizer,
metadatos y rutinas necesarios y funciona sin BML instalado, sin compilador,
sin Docker y sin el GGUF original en ejecución. Las dependencias del sistema,
ABI y CPU deben declararse y verificarse por target; no se promete un binario
universal ni compresión de pesos. Los datos del modelo permanecen dentro del
artefacto, sujetos a su licencia.

La salida final debe ejecutar el plan y kernels de BML. Un wrapper que invoque
llama.cpp no cumple este objetivo. llama.cpp se usa como referencia independiente
con revisión fijada, no como motor oculto del ejecutable final.

## Estado real inspeccionado

Base técnica inspeccionada: `21b90c15982f6eef4bda91a5159d2f8aaf8db2a9`.

- Implementados: lector mmap y codecs F32/F16/Q4_0/Q8_0 en `bml-gguf`;
  Shape/Tensor/TensorView y kernels CPU de referencia en `bml-domain::tensor`;
  DAG/NodeSoA, IR escalar RPN, constantes y serialización BmlGraph; runtime de
  programas/fragmentos y CLI que genera directorios `.bmlgraph`.
- Validación parcial: layouts, rangos y muestras cuantizadas; probe Docker con
  comparación Q4_0 contra función C original de llama.cpp y dot sintético.
  No certifica todo GGUF, matmul del compilador ni inferencia completa.
- Pendiente: Llama numéricamente correcto, tokenizer exacto, paridad de KV,
  IR tensorial de transformer, codegen/enlace AOT y ejecutable con recursos
  embebidos. El tokenizer actual es simplificado; cargar pesos globalmente
  como F32 en InferenceCompiler no es el diseño final de recursos cuantizados.
- La arquitectura `llama` se acepta experimentalmente; eso no demuestra paridad.
  Gemma4 se rechaza para inferencia. GPU y otros targets no están validados.

## Etapas y criterios de aceptación

### R0 — Importación y validación GGUF acotada

**Estado: parcial.** Consolidar lector, codecs y adaptación de ejes GGML al
layout tensorial sin expandir todos los pesos a F32.

Aceptar cuando un perfil Llama explícito compruebe arquitectura, tokenizer,
formas, tipos, offsets, tamaños, alineación y límites; rechace variantes o datos
incompletos sin defaults silenciosos; y registre hashes del modelo y muestras.
Las pruebas deben cubrir matrices rectangulares, signos, escalas, nibbles,
truncamiento y cuantizadores no soportados contra referencia independiente.
El probe Docker es evidencia parcial de esta etapa, no su cierre completo.

### R0.S — GGUF sintético y autonomía de un kernel

**Estado: gate F32 pequeño ejecutado; escalado grande pendiente.** Usar matrices
aleatorias reproducibles por semilla, fijas en compilación, y tamaño n=10·2^k: 10,20,40,80… El límite
es 3.000.000.000 bytes por matriz y también por GGUF completo. En F32, la última
dimensión de esa secuencia admitida es 20480 (tensor 1.677.721.600 bytes más
cabeceras); 40960 excede. Este plan reemplaza el extremo anterior, no exige
materializar matrices gigantes ni consumir toda la RAM disponible.

Aceptar el gate pequeño cuando un GGUF v3 estándar (cabeceras, alineación,
metadatos y dimensiones) pase roundtrip en bml-gguf y un lector Python separado;
el contrato A·x/A·B se registre fuera de GGUF, que no define un grafo estándar;
y pesos F32 embebidos en un kernel nativo reproduzcan una referencia independiente
con entradas dinámicas. Ejecutar con solo el binario montado en un entorno sin
BML/GGUF y registrar dependencias del sistema, semilla, layout, hashes y errores.

Ejecutado: n=10,20,40,80; matvec/matmul y ocho ejecuciones aisladas con error
máximo cero en fixtures de valores binarios exactos. Backend experimental de
emisión C y enlace libc; no implementa aún el IR/codegen de un transformer.
Esto demuestra autonomía del kernel sintético, no cierra R1–R4 ni el contenedor
versionado completo de recursos de R3. Fuentes y evidencia en
[prueba sintética](SYNTHETIC_AOT_VALIDATION.md).

El vector x o segunda matriz B se inyecta en runtime. Separar tiempos de
generación, importación, codegen y compilación de tiempo por entrada, tamaño del
binario y picos de memoria. Para los loops clásicos implementados: matvec Θ(n²),
matmul Θ(n³), almacenamiento denso Θ(n²); no son cotas universales para toda
especialización de una matriz fija. Si ambos operandos son constantes, se puede
precomputar el producto, pero materializar su salida sigue siendo Θ(n²); devolver
solo una referencia es otro contrato y no demuestra acelerar entrada dinámica.

Antes de ampliar, comprobar memoria disponible y cgroup, pico RSS, memoria de
workspace y presupuesto de tiempo para cada operación. Este gate limita n real
a 128 (la secuencia actual se detiene en 80), compilación a 512 MiB de espacio
virtual/30 s CPU y ejecución aislada a 128 MiB. Separar mediciones reales de
estimaciones de bytes, FLOPs y tráfico; tiempos de proceso con E/S no son tiempos
puros del kernel ni TPS. Matmul n=20480 no está autorizado por el mero límite de
archivo: A+B+C F32 ocuparían 5.033.164.800 bytes antes de overhead.

F16/Q4_0/Q8_0 vendrán después con tolerancias y kernels propios. El exportador
actual rechaza tipos distintos de F32. Para Q4_0/Q8_0, ne0=10 no forma bloques
completos de 32: rechazar o especificar padding y dimensiones lógicas/físicas
antes de implementarlo; no inventar soporte por cambiar la etiqueta dtype.

### R1 — Llama mínimo correcto

**Estado: pendiente; ruta experimental existente.** Elegir un perfil de modelo
pequeño concreto y reproducir su semántica antes de optimizarlo.

Aceptar cuando tokenizer/detokenizer, BOS/EOS y tokens coincidan con la
referencia; RMSNorm, RoPE, GQA/atención causal, residuales y SwiGLU pasen
comparaciones intermedias; logits de prefill y varios pasos decode cumplan
presupuestos numéricos fijados previamente; KV con/sin caché sea coherente; y
una batería fija de prompts produzca la secuencia greedy esperada. Fijar modelo,
SHA256, revisión/flags de referencia, contexto, hilos y política de muestreo.
NaN, diferencias de forma o divergencias no resueltas impiden avanzar.

### R2 — IR tensorial, plan estático y kernels/codegen

**Estado: pendiente; DAG/RPN escalar y kernels tensoriales parciales reutilizables.**
Representar ops tensoriales, recursos cuantizados, activaciones y estado KV por
separado. Planificar vida útil de buffers, scratch y memoria según contexto.
Mantener pesos comprimidos y decodificar solo los bloques necesarios.

Aceptar cuando el plan preserve formas, dtype, strides, ejes y precisión; sus
kernels BML reproduzcan los resultados de R1; y un emisor genere código nativo
por target, inicialmente Rust con llamadas directas a kernels y enlace del
runtime necesario. Medir límites de RAM de construcción/ejecución. No convertir
cada peso en nodos escalares ni usar tamaño de grafo como prueba de rendimiento.

### R3 — Recursos embebidos y enlace autónomo

**Estado: pendiente.** Integrar pesos cuantizados, tokenizer, configuración y
plan en un contenedor normalizado dentro del ejecutable. Para el primer ELF,
evaluar payload anexado, footer versionado y mmap readonly del propio archivo;
validar offsets u64, longitudes, alineación, hashes y límites antes de usarlo.

Aceptar cuando un ejecutable sintético autónomo funcione después de retirar
los archivos de entrada, detecte payloads dañados y no abra un GGUF ni cargue
BML desde el sistema. Después, aplicar el mismo contrato al perfil Llama de R1.
El manifiesto debe incluir target/ABI/CPU, versiones de herramientas, commits,
hashes de origen/segmentos/tokenizer, licencia, límites de contexto, scratch y
KV. El hash final del ejecutable se registra externamente para evitar circularidad.
Revisar permisos de redistribución antes de publicar recursos con modelo.

### R4 — Prueba offline en entorno limpio

**Estado: pendiente.** Llevar solo el ejecutable a un entorno limpio del target,
sin red, BML instalado, toolchain, llama.cpp ni GGUF original.

Aceptar cuando inspección de enlaces y trazado de archivos demuestren las
dependencias declaradas, no haya búsquedas de recursos externos, y se repitan
los tokens/logits/generación de R1 contra resultados de referencia capturados
por separado. Registrar comandos, hashes, límites y resultados. El contenedor
es un banco de pruebas, no un requisito del usuario final. Probar también errores
de recursos, contexto excesivo y corrupción. Esta etapa cierra la autonomía.

### R5 — Rendimiento y optimizaciones

**Estado: posterior a corrección y autonomía.** Medir TTFT, prefill, decode,
memoria residente, tamaño del artefacto y coste de compilación, distinguiendo
arranque frío/caliente. Comparar igual modelo, entradas, tokens, contexto, hilos,
hardware y política de muestreo. Optimizar SIMD, cuantización y scheduling con
regresiones de paridad; proyecciones sintéticas no representan TPS de inferencia.

### R6 — GPU y expansión de targets

**Estado: posterior.** Evaluar GPU, ARM y otros OS/ABI solo con backend, packaging,
licencias y pruebas de paridad/ejecución propios. No declarar soporte multiarch
por disponer de un Dockerfile. Gemma4 exige su arquitectura y validación propias.

## Próximo incremento verificable

El gate R0.S ya demuestra un kernel F32 autónomo pequeño. Continuar con el
contenedor versionado y rechazo de recursos dañados de R3 (A0 en el
[diseño AOT](architecture/AOT_STANDALONE.md)): formato mínimo de recursos con
validación de rangos/hashes, codegen/enlace de una proyección Q4_0 y ejecución
del binario aislado sin BML/GGUF. No requiere descargar un modelo y no cierra R1.
Después completar R0/R1 con un Llama pequeño existente autorizado, desarrollar
R2/R3 para ese perfil y exigir R4 antes de anunciar inferencia autónoma.

## Evidencia y entregables separados

La base histórica de contratos/calidad se integró en PR #2. Mantener tests,
doctests, auditoría, trazabilidad, formato, Clippy, cobertura y mutación de código
productivo modificado. Ignorados, no ejecutados y bloqueados no cuentan como
aprobados; evidencia histórica no sustituye ejecución del commit evaluado.

La imagen Docker de validación y sus fuentes exportadas pertenecen al commit
`21b90c15982f6eef4bda91a5159d2f8aaf8db2a9`: muestras rank-2 Q4_0 sin pesos de
modelo incluidos. Este roadmap se conserva en un commit documental separado;
no modifica esos artefactos ni amplía su alcance a inferencia completa. Los
futuros binarios con modelo requieren autorización y licencia específicas.
