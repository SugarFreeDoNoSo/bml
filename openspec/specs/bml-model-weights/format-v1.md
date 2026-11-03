# Formato binario BMLGGUF v1

Este documento es normativo para `bml-model-weights`. Define el formato antes de
implementar reader/writer. La extensión recomendada es `.bmlgguf`; la identificación
depende del magic, nunca de la extensión. Todos los enteros binarios son unsigned
little-endian. Los offsets son absolutos desde el primer byte del archivo.
Los rangos usan inicio inclusivo y final exclusivo; no hay alineación implícita.

## 1. Cabecera fija: 256 bytes

| Offset | Bytes | Campo | Valor/regla v1 |
|---:|---:|---|---|
| 0 | 8 | magic | `42 4d 4c 47 47 55 46 00` (`BMLGGUF\0`) |
| 8 | 2 | version | `1` |
| 10 | 1 | endian | `1` = little-endian |
| 11 | 1 | basis | `2` = BML base 2 |
| 12 | 4 | header_bytes | `256` |
| 16 | 8 | flags | `0`; otros bits se rechazan |
| 24 | 8 | file_bytes | longitud exacta, sin trailer |
| 32 | 8 | tensor_count | al menos 1 |
| 40 | 8 | block_count | al menos 1 |
| 48 | 8 | metadata_offset | `256` |
| 56 | 8 | metadata_bytes | longitud del JSON UTF-8 |
| 64 | 8 | tensors_offset | `metadata_offset + metadata_bytes` |
| 72 | 8 | tensors_bytes | longitud del índice tensorial |
| 80 | 8 | blocks_offset | `tensors_offset + tensors_bytes` |
| 88 | 8 | blocks_bytes | `80 * block_count` |
| 96 | 8 | payload_offset | `blocks_offset + blocks_bytes` |
| 104 | 8 | payload_bytes | `file_bytes - payload_offset` |
| 112 | 32 | source_sha256 | SHA-256 de todos los bytes del GGUF original |
| 144 | 32 | catalog_sha256 | SHA-256 del rango `[256, payload_offset)` |
| 176 | 32 | header_sha256 | SHA-256 de la cabecera de 256 bytes con este campo a cero |
| 208 | 48 | reserved | todos cero |

Las secciones son contiguas en ese orden, sin huecos, padding ni solapamientos.
Toda suma, resta, multiplicación y conversión a `usize` se comprueba antes de usar
el resultado. El tamaño real debe coincidir con `file_bytes`. El reader comprueba
la cabecera, sus límites y su hash antes de reservar o leer secciones variables.

## 2. Metadatos

La sección contiene un único objeto JSON UTF-8, sin BOM, con estas claves
obligatorias y ninguna otra en v1:

| Clave | Contenido |
|---|---|
| `schema` | número JSON entero `1` |
| `profile` | string `bml-exact-decoded-f32-v1` |
| `source_gguf_version` | número JSON entero `2` o `3` |
| `converter` | objeto con strings no vacíos `name`, `version`, `revision` |
| `codec` | objeto con strings no vacíos `name`, `version`, `revision` |
| `limits` | objeto de presupuestos descrito abajo |
| `gguf` | array de entradas tipadas que conserva todos los metadatos GGUF admitidos |

No se admiten claves duplicadas en ningún objeto. El orden de las claves JSON y
el whitespace no tienen significado; los hashes cubren los bytes almacenados,
sin recanonizar. Un writer emite claves en el orden de la tabla y JSON compacto
para reproducibilidad. `gguf` se ordena por bytes UTF-8 de la clave; sus claves
son únicas, no vacías y no contienen NUL. Los metadatos requeridos para arquitectura,
tokenizer y configuración deben existir y ser compatibles con el perfil de
inferencia admitido; preservarlos no implica admitir todas las arquitecturas.

Cada entrada es exactamente `{"key": STRING, "type": TYPE, "value": VALUE}`.
`TYPE` es un número JSON entero con los códigos GGUF siguientes. Los enteros
de VALUE se representan mediante strings decimales canónicos para evitar pérdida
en parsers JSON: sin `+`, sin ceros iniciales salvo `"0"`, sin `"-0"`.

| TYPE | Tipo | VALUE |
|---:|---|---|
| 0, 1 | u8, i8 | string decimal dentro del rango |
| 2, 3 | u16, i16 | string decimal dentro del rango |
| 4, 5 | u32, i32 | string decimal dentro del rango |
| 6 | f32 | 8 dígitos hex minúsculos del patrón IEEE, más significativo primero |
| 7 | bool | booleano JSON `true` o `false` |
| 8 | string | string JSON Unicode válido, sin sustituciones de bytes inválidos |
| 9 | array | `{"element_type": TYPE, "values": [VALUE, ...]}` |
| 10, 11 | u64, i64 | string decimal dentro del rango |
| 12 | f64 | 16 dígitos hex minúsculos del patrón IEEE, más significativo primero |

Los arrays son homogéneos y `element_type` no puede ser 9; los arrays anidados se
rechazan explícitamente. No se convierten silenciosamente tipos o strings inválidos.
Los patrones IEEE de metadatos se conservan; cada consumidor valida finitud/rango
cuando el campo tiene semántica numérica. Esto no permite pesos NaN/inf.

`limits` contiene exactamente las siguientes claves, cada una un string decimal
canónico u64 positivo: `max_weight_bits`, `max_weight_depth`,
`max_synthesis_steps_per_weight`, `max_synthesis_steps_total`, `max_block_terms`,
`max_block_bytes`, `max_output_bytes`, `max_resident_bytes`, `max_cache_bytes`.
Solo `max_cache_bytes` admite cero, que desactiva la caché. Estos son los límites
de conversión solicitados y no pueden ampliar los límites locales del reader.
Los límites de bytes/bits/cantidad derivados del archivo se verifican contra
ellos; los límites de trabajo/memoria de conversión son procedencia declarada,
no una prueba del trabajo realizado. El informe de conversión registra el consumo.

Los límites de pasos se aplican independientemente a síntesis y replay de
certificados: cada contador (por peso y acumulado) debe respetar el mismo límite
declarado. Una entrada de caché evita síntesis, pero no el replay de cada peso
emitido. El informe `synthesis_steps` cuenta solo los pasos de síntesis.

## 3. Índice tensorial: registros variables

Se recorren exactamente `tensor_count` registros, sin padding. La posición del
registro define su `tensor_id` (desde cero). El orden conserva el de los tensores
en el GGUF fuente. Cada registro empieza con estos 40 bytes:

| Offset relativo | Bytes | Campo |
|---:|---:|---|
| 0 | 4 | `name_bytes` |
| 4 | 4 | `rank` |
| 8 | 4 | `source_dtype` (código GGML) |
| 12 | 4 | reservado cero |
| 16 | 8 | `element_count` |
| 24 | 8 | `first_block` (índice en tabla de bloques) |
| 32 | 8 | `tensor_blocks` |

Siguen `rank` dimensiones u64 en orden GGML y luego `name_bytes` bytes UTF-8 del
nombre, sin terminador NUL. El registro mide `40 + 8*rank + name_bytes` bytes.
Los nombres son únicos, no vacíos y sin NUL. Se admite `1 <= rank <= 4`; todas las
dimensiones son positivas y su producto debe ser `element_count`.

Los códigos fuente admitidos en v1 son F32=0, F16=1, Q4_0=2, Q8_0=8 y Q6_K=14.
No describen el payload de salida, que siempre es BML. Dtypes ajenos se rechazan.
Para F32/F16 el objetivo es el valor IEEE exacto; para cuantizados es el F32 del
codec identificado. El bloque original cuantizado no se conserva como peso.

`tensor_blocks >= 1`. El primer tensor tiene `first_block=0`; cada tensor siguiente
empieza inmediatamente después de los bloques del anterior. El final del último
coincide con `block_count`. El parser debe terminar exactamente en `blocks_offset`.

## 4. Índice de bloques: registros fijos de 80 bytes

| Offset relativo | Bytes | Campo |
|---:|---:|---|
| 0 | 8 | `tensor_id` |
| 8 | 8 | `first_element` (índice plano dentro del tensor) |
| 16 | 8 | `term_count` |
| 24 | 8 | `data_offset` (absoluto) |
| 32 | 8 | `data_bytes` |
| 40 | 8 | `useful_bits` |
| 48 | 32 | `data_sha256` |

Los bloques se ordenan por tensor y elemento. Cada tensor comienza en elemento
cero y cada bloque comienza después del anterior; no se permiten omisiones,
duplicaciones ni bloques compartidos. La suma de `term_count` de cada tensor
coincide con su `element_count`; todo bloque tiene al menos un término.

El primer `data_offset` es `payload_offset`; los siguientes son el final del
bloque anterior. El último termina en `file_bytes`. Los hashes son SHA-256 de
todos los bytes del bloque, incluida su cabecera BMLS. El índice puede validarse
sin decodificar todos los pesos; antes de exponer un bloque se verifica su hash
y su contenido. Una inspección parcial declara qué bloques no ha verificado.

## 5. Payload de cada bloque

Cada bloque es un flujo BMLS v1 completo:

| Offset relativo | Bytes | Campo |
|---:|---:|---|
| 0 | 4 | ASCII `BMLS` |
| 4 | 1 | versión `1` |
| 5 | 1 | base `2` |
| 6 | 2 | reservado cero |
| 8 | 8 | count = `term_count` del índice |
| 16 | 8 | bit_length = `useful_bits` del índice |
| 24 | resto | códigos de árbol concatenados, MSB primero |

`C(1)=0`, `C(B(a,b))=1 C(a) C(b)`. Se conserva cada bit, incluidas hojas iniciales
y finales. `data_bytes = 24 + useful_bits/8 + (useful_bits%8 != 0)` con aritmética
comprobada. Los bits bajos de relleno son cero. Se consumen exactamente `count`
árboles completos y `bit_length` bits; la hoja tiene profundidad cero.

El orden plano sigue GGML (eje cero contiguo). Por ejemplo, `[1,2,1]` tiene
`term_count=3`, `useful_bits=5`, `data_bytes=25` y payload `0x40`.
Un tensor GGML `[3,2]` se adapta a row-major `[2,3]` sin permutar valores planos.

## 6. Integridad y exactitud: comprobaciones distintas

El writer calcula primero hashes de bloques, luego el hash de catálogo (que
incluye esos hashes), y finalmente el de cabecera. Modificar cualquier dato
requiere actualizar su cadena de hashes. SHA-256 detecta corrupción, no autentica
al productor ni demuestra la equivalencia matemática respecto al GGUF fuente.

La conversión certifica cada peso antes de almacenarlo. V1 guarda árboles, no
derivaciones; verificar independientemente exactitud respecto de una fuente
requiere esa fuente y recertificar/comparar objetivos. El reader comprueba
estructura e integridad y no declara por ello demostrada la conversión exacta.
No necesita la fuente para cargar ni ejecutar el archivo.

## 7. Límites de lectura predeterminados

Los siguientes límites son independientes de los metadatos del archivo. Una API
puede ofrecer límites explícitos menores o mayores representables; los campos
fijos, las reglas de versión y los controles de overflow siguen siendo obligatorios.
Una operación de CLI debe declarar cualquier ampliación del perfil predeterminado.

| Recurso | Límite predeterminado |
|---|---:|
| Tamaño total | 64 GiB = 68.719.476.736 bytes |
| Metadatos JSON | 64 MiB = 67.108.864 bytes |
| Entradas GGUF | 65.536 |
| Elementos de arrays de metadatos, sumados | 4.194.304 |
| Bytes UTF-8 de una clave o nombre tensorial | 4.096 |
| Bytes UTF-8 de un string de metadatos | 1.048.576 |
| Tensores | 65.536 |
| Bloques | 1.048.576 |
| Índices tensoriales y de bloques, sumados | 128 MiB = 134.217.728 bytes |
| Términos por bloque | 65.536 |
| Bytes de un bloque (incluida cabecera) | 8 MiB = 8.388.608 bytes |
| Bits por término expandido | 1.048.576 |
| Profundidad por término | 4.096 |
| Términos de todo el archivo | 4.294.967.296 |

Los límites son inclusivos. El parser JSON tiene profundidad máxima 8 contando
el objeto raíz como 1 y valida longitudes/conteos durante el parseo, no solo tras
materializar arrays. El reader procesa bloques sin reservar un `Vec<Bml>` para
todo el modelo. La suma de bits y los productos de formas usan aritmética
comprobada incluso en plataformas de 32 bits.

## 8. Orden de validación y publicación

1. Leer exactamente 256 bytes, comprobar magic/versión/reservados, hash y tamaño.
2. Comprobar límites y ecuaciones de secciones antes de leerlas.
3. Verificar el hash del catálogo, parsear JSON e índices con presupuestos.
4. Verificar unicidad, dtypes, formas, cobertura exacta de bloques y payload.
5. Para cada bloque solicitado, comprobar hash, cabecera, gramática, padding,
   conteos, profundidad y bits antes de devolver términos.
6. Para conversión, certificar todos los pesos y terminar el temporal vecino;
   publicar mediante una operación sin sobrescritura. No usar un rename que
   reemplace silenciosamente un destino que apareció durante la conversión.

Un modelo cargado puede mantener un mapa de archivo inmutable y decodificar
bloques a demanda. No se modificará ni truncará el archivo mientras esté mapeado.
Un error nunca entrega un bloque parcialmente válido ni un destino incompleto.
