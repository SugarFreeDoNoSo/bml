## Context

`Bml` dispone de síntesis exacta certificada y serialización `BMLS` versionada.
El reader GGUF decodifica tensores por bloques/filas y los artefactos simbólicos
actuales guardan programas con referencias a pesos GGUF externos. Falta un
formato persistente que contenga los pesos mismos como árboles BML.

## Goals / Non-Goals

- G1: convertir un GGUF admitido a un único archivo autónomo `.bmlgguf`.
- G2: preservar valores exactos bajo una política declarada y fallar con tensor
  e índice cuando no sea posible certificar un peso dentro del presupuesto.
- G3: cargar bloques de términos preparados sin volver a sintetizar sus pesos.
- G4: conectar esa representación con la inferencia simbólica y medirla.
- NG1: atribuir al formato compatibilidad binaria con GGUF o llama.cpp.
- NG2: prometer síntesis universal, compresión o aceleración sin evidencia.
- NG3: ampliar implícitamente las arquitecturas o dtypes de inferencia admitidos.

## Decisions

### D1. Nombre y frontera del contenedor

Se adopta provisionalmente `.bmlgguf`. Tiene magic y versión propios: cambiar
la extensión de un GGUF no convierte sus pesos. `.bmllm` es una alternativa de
nombre pendiente de una decisión futura, no un alias que el loader deba adivinar.
La especificación binaria v1 está fijada en [format-v1.md](format-v1.md): cabecera
de 256 bytes, metadatos tipados, índice tensorial variable, índice de bloques de
80 bytes y payloads BMLS. Incluye offsets, hashes, presupuestos y orden de validación.

### D2. Valor objetivo y exactitud

Para F32/F16, el objetivo es el racional exacto del valor IEEE almacenado. Para
Q4_0/Q8_0/Q6_K, es el racional exacto del valor F32 producido por el codec fijado
y verificado; no el peso previo a cuantización, que el archivo no contiene.
Convertir F32 a F64 es exacto y permite usar la API actual `from_f64_exact`.
Se rechazan NaN/inf; ambos ceros IEEE se normalizan al cero matemático.

Cada éxito requiere la derivación certificada, no una comparación flotante.
Un error de dominio, síntesis o presupuesto cancela la conversión completa.
No hay fallback silencioso a floats ni a la API aproximada. Una eventual modalidad
con pérdidas necesitaría perfil, métricas y criterios de aceptación separados.

### D3. Preparación acotada y publicación

Se procesa por bloques tensoriales con límites de entrada, términos, profundidad,
bits, trabajo de síntesis, bytes de salida y memoria residente. La caché opcional
de pesos repetidos tiene presupuesto propio. El payload v1 usa bloques `BMLS`
con índices y rangos comprobados; la deduplicación física entre bloques no se
presupone. Se escribe un temporal vecino al destino y se publica sin sobrescribir
solo tras terminar índices, metadatos e integridad. Cualquier fallo elimina el
temporal y deja intactos fuente y destino existente.

### D4. Orden y autosuficiencia

Los pesos siguen el orden plano GGML, con eje cero contiguo. La forma original se
guarda; un adaptador row-major invierte dimensiones, no valores. El contenedor
incluye tokenizer y metadatos necesarios para el perfil del modelo. El hash del
GGUF original sirve de procedencia; cargar y ejecutar no requiere ese archivo.

### D5. Integración en la ruta

Interfaz prevista: `bml-cli convert-weights -m modelo.gguf -o modelo.bmlgguf`,
`bml-cli inspect-weights -m modelo.bmlgguf` y consumo explícito del contenedor por
la ruta simbólica. Los dos comandos CLI están implementados. La API
`SymbolicCausalModel::open_bmlgguf` implementa inferencia causal con política
explícita `CheckedExactF32`: cada peso evaluado debe coincidir como racional con
el objetivo de su árbol. Las activaciones usan el perfil numérico RPN BML existente.
El comando CLI `run` exige `--weight-evaluation checked-exact-f32` junto con
`--max-evaluated-weights`; no selecciona esa frontera numérica automáticamente.
Las campañas reales siguen pendientes.
El cargador ofrece acceso a bloques `Bml` para matrices y proyecciones simbólicas.
Una evaluación numérica posterior debe declarar dominio/rango; exactitud del
almacenamiento no implica que todo término sea evaluable directamente en F64.

### D6. API y frontera de dependencias

[api.md](api.md) fija `bml-model-weights` como crate separado, con reader/writer
independientes del compilador/runtime y `gguf-import` optativo para conversión.
`WeightReader` posee un archivo y devuelve bloques BML propios; `WeightWriter`
solo publica tras completar la transacción. Los errores conservan causas tipadas
y ubicación tensorial. El reader GGUF sin features no adquiere dependencias nuevas.

## Risks / Trade-offs

- R1: expansión de árboles y síntesis no viable para pesos reales. Mitigación:
  presupuestos, diagnóstico localizado y campañas progresivas; no esconder fallos.
- R2: dominios numéricos parciales de BML. Mitigación: separar pruebas de valor
  exacto, ejecución simbólica y paridad de logits/tokens.
- R3: crecimiento de tamaño/memoria. Mitigación: bloques, mediciones de pico,
  límites previos a reserva y distinción entre preparación y ejecución.
- R4: deriva del decoder cuantizado. Mitigación: fijar revisión, dtype y política
  de decodificación en la procedencia y usar fixtures con oráculos independientes.
