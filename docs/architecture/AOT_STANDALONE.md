# GGUF → ejecutable autónomo por plataforma

Estado: diseño propuesto el 30 de octubre de 2026, basado en inspección del
checkout `21b90c15982f6eef4bda91a5159d2f8aaf8db2a9`. No implementado.

## Contrato del producto

BML se instala en la máquina de compilación. El compilador recibe un GGUF y
produce un ejecutable para un target concreto. En ejecución no requiere BML
instalado, compilador Rust, Python, Docker ni el GGUF original. Pesos, tokenizer,
metadatos y rutinas necesarios viajan dentro del artefacto. Los datos del modelo
no desaparecen ni se presupone una reducción de tamaño. Las dependencias del
sistema/ABI deben declararse y verificarse; autonomía no significa binario
universal. Primer target propuesto: Linux x86_64 CPU de referencia, sin AVX
obligatorio; evaluar enlace musl estático antes de prometer ausencia de libc
externa. Windows, macOS, ARM, GPU y Gemma4 quedan fuera del primer hito.

## Inventario observado y límites

| Componente existente | Evidencia en código | Reutilización/límite |
|---|---|---|
| Lector y codecs | `crates/gguf/src/{gguf,codecs,view,mmap}.rs` | mmap readonly y decode explícito F32/F16/Q4_0/Q8_0; conservar bloques sin expansión global |
| Tensor CPU | `crates/domain/src/tensor/` | Shape/Tensor/TensorView, strides y kernels de referencia; el orden row-major requiere adaptación explícita de dimensiones GGML |
| IR escalar | `compiler/src/dag.rs`, `rpn.rs`, `fragment.rs` | NodeSoA/DAG, RpnProgram, pool de constantes y BmlGraph serializado; no equivale a IR tensorial de transformer ni objeto nativo |
| Serialización | `gguf_compiler.rs::serialize_to_dir`, `load_from_dir` | Directorio `.bmlgraph` con configuración/pool/fragmentos; no ejecutable autocontenido |
| Runtime | `runtime/src/{runtime,hot_loop,scheduler}.rs` | Ejecuta programas/grafos y fragmentos; no se encontró backend de emisión/enlace nativo en compiler |
| Inferencia | `gguf_compiler.rs::InferenceCompiler` | Carga pesos a un pool F32; no usar esta carga global como diseño de recursos cuantizados |
| Tokenizer | `compiler/src/tokenizer.rs::Vocabulary` | Vocabulario disponible, pero tokenización simplificada con fallback BOS; no certifica paridad con SentencePiece/BPE del modelo |
| CLI | `cli/src/main.rs` | Compila a `.bmlgraph` y ejecuta con BML; no cumple todavía el contrato autónomo |

La ruta experimental admite `llama` y rechaza otras arquitecturas, incluida
Gemma4. Aceptar el nombre Llama no certifica sus capas, logits o generación.
El probe Docker ya construido verifica muestras Q4_0 con una rutina upstream
llama.cpp y un dot independiente: no valida el matmul del compilador, un
transformer ni un backend AOT. La imagen no contiene pesos.

## Pipeline propuesto

1. **Importación validada**: comprobar arquitectura/configuración, tokenizer,
   nombres, formas, tipos, offsets, tamaños y límites con aritmética checked.
   Congelar un perfil Llama pequeño admitido; rechazar variantes/ops/dtypes
   desconocidos antes de generar salida. No inferir valores obligatorios por
   defaults silenciosos. Usar lectura acotada y streaming.
2. **IR tensorial tipada**: distinguir `TensorResource` (ID, dims GGML, strides,
   dtype, cuantizador, offset/longitud/alineación) de activaciones y estado KV.
   Nodos explícitos: embedding, RMSNorm, Q/K/V, RoPE, atención causal/GQA,
   softmax, proyección, residual, SwiGLU y logits. Registrar ejes, precisión y
   epsilons en cada op. No transformar cada peso en nodos escalares BML.
   Reutilizar el IR escalar solo cuando su dominio numérico sea válido.
3. **Plan de memoria/ejecución**: orden estático de ops, vida útil de buffers,
   scratch acotado y presupuesto KV según contexto y dtype. Mantener Q4_0
   comprimido y decodificar por bloque/fila para GEMV CPU; F32/F16 para recursos
   requeridos como normas. Comprobar productos rectangulares, signos, offset,
   extremos de bloques y escalas con referencias independientes. Optimizaciones
   SIMD después de la paridad del kernel de referencia.
4. **Codegen inicial**: emitir Rust tipado con llamadas directas a kernels CPU
   seleccionados y un plan de recursos constante. Rust/rustc actúa como backend
   AOT; no hace falta un backend LLVM propio para demostrar el contrato. Enlazar
   estáticamente las partes necesarias del runtime en el ejecutable, sin buscar
   una instalación BML al arrancar. Evitar el scheduler distribuido en este hito.
5. **Recursos embebidos**: crear un contenedor binario normalizado de tensores
   cuantizados, tokenizer y metadatos; no limitarse a esconder un GGUF y volver a
   analizarlo en runtime. Para el primer ELF, anexar ese payload al ejecutable
   enlazado con un footer versionado de offsets/longitudes u64 y hashes. El
   launcher mapea su propio archivo readonly, valida rangos/hashes y expone vistas
   de segmentos. Probar explícitamente que el loader ELF admite el formato y que
   el proceso no depende de un archivo auxiliar. No usar `include_bytes!` de todos
   los pesos como única estrategia: puede aumentar RAM del compilador/enlazador.
   Cambiar el packaging para firma de otros formatos de ejecutable será un hito
   propio. Los recursos no incluyen credenciales ni rutas privadas de build.
6. **Interfaz runtime**: entradas de texto y/o token IDs, límites de contexto y
   generación, BOS/EOS, modo greedy determinista y salida JSON de tokens/logits
   de diagnóstico. Tokenizer/detokenizer exactos para el perfil elegido, sin
   sustituir desconocidos por BOS. No usar red ni GGUF externo al ejecutar.

## Manifiesto e identidad

Incorporar versión de formato/ABI, target triple/OS/CPU features, endianness,
precisión y kernels; commit BML, rustc/linker y flags; hash del GGUF de origen,
configuración/tokenizer, tabla de tensores y hashes de segmentos; revisión de
referencia y perfil de validación; contexto máximo, límites de trabajo, scratch
y KV estimados. Conservar licencias/atribuciones del código y del modelo, origen
y permisos de redistribución. Copiar tokenizer/pesos a un ejecutable no elimina
sus condiciones de licencia. No publicar artefactos con modelo sin revisión de
licencia y autorización específica. El hash final del ejecutable y el resultado
de reproducibilidad van en un manifiesto externo, evitando autorreferencia.

## Roadmap y criterios verificables

| Etapa | Entregable | Criterio para avanzar |
|---|---|---|
| A0: autonomía sintética | Emisor de un ELF CPU con dos filas Q4_0 sintéticas y recursos normalizados embebidos | Ejecutar tras retirar GGUF/input temporal y sin BML en PATH; proyección y hashes correctos; errores de footer/segmentos truncados; recursos y dependencias inspeccionados |
| A1: núcleo Llama | IR/plan y kernels de un bloque con parámetros y activaciones sintéticos | Comparar tensores intermedios contra referencia independiente; cubrir RMSNorm, RoPE, causalidad/GQA, residuales, SwiGLU y KV |
| A2: perfil real pequeño | Importador/tokenizer exactos y ejecutable privado de un GGUF Llama pequeño ya disponible | Tokens iguales; logits prefill y varios pasos decode comparables a llama.cpp fijado; cache/no-cache coherentes; ejecución limpia sin GGUF ni instalación BML |
| A3: rendimiento | Medidas comparables de prefill/decode, TTFT, memoria residente y tamaño de artefacto | Solo después de A2; mismas entradas, tokens, hilos, contexto y hardware; sin convertir pruebas de proyección a TPS |

Próximo incremento implementable **sin descargar modelos**: A0, primero un formato
mínimo de recursos con tests de límites/hashes, después emisor y enlace del
fixture, y finalmente prueba de ejecución en un contenedor limpio que solo
reciba el ejecutable. Medir memoria de compilación y ejecución y verificar con
`readelf`/`ldd` y trazado de archivos las dependencias reales. No basta con pasar
un launcher que abra otro GGUF o requiera la CLI BML.

Para A2, fijar revisión/flags de llama.cpp, SHA256 exacto del GGUF, prompts/IDs,
política BOS/EOS, hilos, contexto y semilla antes de ejecutar. Registrar logits
completos de vocabulario en casos pequeños, error absoluto/relativo y orden de
candidatos; proponer inicialmente atol/rtol 1e-4 para F32 y documentar un
presupuesto específico por kernel cuantizado antes de aceptar Q4_0. Son umbrales
propuestos, no resultados observados ni garantías de paridad. Fallar ante NaN,
formas o longitudes distintas. Registrar casos de empate y exigir igualdad de
secuencia greedy sobre un conjunto fijado; si diverge, investigar desde el
primer tensor/logit discrepante sin relajar umbrales automáticamente.

## Estado de entrega

La release pública de la imagen de validación fue autorizada, pero su consulta
previa a GitHub falló con `Post "https://api.github.com/graphql": Forbidden`.
No se intentó crear release/tag ni subir assets. El paquete Docker del commit
21b90c1 y sus hashes están preservados localmente; este diseño no modifica esa
imagen, sus fuentes exportadas, ni añade pesos a la release pendiente.
