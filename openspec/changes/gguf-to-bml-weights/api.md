# API pública prevista de bml-model-weights

Contrato normativo de la tarea 1.4. Los tipos, límites, errores, reader, writer y
`convert::from_gguf` están implementados en `crates/model-weights`. La revisión
de contabilidad conjunta de memoria y presupuestos de síntesis/replay se documenta
abajo. El formato en disco es [BMLGGUF v1](format-v1.md).

## 1. Paquete y dependencias

- Paquete `bml-model-weights`, módulo Rust `bml_model_weights`, directorio
  `crates/model-weights`. `#![deny(unsafe_code)]`.
- Features predeterminadas vacías. El núcleo contiene reader, writer, metadatos,
  límites y errores; depende de `bml-domain`, `sha2`, `serde` y `serde_json`.
  El reader usa serde/serde_json para un decoder de metadatos estricto y acotado.
  Los enums públicos no presuponen una representación JSON
  derivada automáticamente.
- `gguf-import = ["dep:bml-gguf"]` habilita el módulo `convert`. La dependencia
  opcional usa `default-features = false`, sin activar `tensor`.
- Ninguna dependencia normal/build hacia `bml-compiler`, `bml-runtime`,
  `bml-inference`, la fachada `bml` ni aplicaciones.
- `bml-gguf` no depende de este crate. Su configuración sin features sigue
  independiente de `bml-domain`, compilador, inferencia y runtime.
- CLI/compilador/runtime pueden depender del nuevo crate. El adaptador que
  interpreta la arquitectura y ejecuta inferencia vive en esos consumidores;
  el contenedor conserva metadatos sin incluir lógica del transformer.

Se comprobarán ambos grafos con `cargo metadata --locked` y las suites
`cargo test -p bml-gguf --no-default-features --locked` y
`cargo test -p bml-model-weights --no-default-features --locked`. La suite de
importación usará `cargo test -p bml-model-weights --features gguf-import --locked`.

## 2. Tipos y unidades

Los tipos de dominio públicos reutilizados son `Bml`, `CertifiedBml`,
`ExactSynthesisOptions`, `ExactError` y `StreamError` de `bml_domain::number`.
No se introduce una segunda representación de árbol.

- `TensorId(u64)` y `BlockId(u64)`: newtypes públicos, `Copy + Eq + Ord + Hash`;
  construir un ID no implica que exista. Cada acceso verifica su rango.
- `Digest([u8; 32])`: SHA-256 en bytes; presentación hexadecimal minúscula solo
  para informes. No se usa un string arbitrario como hash interno.
- `SourceDtype`: enum `F32`, `F16`, `Q4_0`, `Q8_0`, `Q6K`, con conversión
  comprobada a los códigos GGML de format-v1; no representa el dtype del payload.
- `TensorInfo`: campos `id: TensorId`, `name: String`, `ggml_dimensions: Vec<u64>`,
  `source_dtype: SourceDtype`, `element_count: u64`, `first_block: BlockId`,
  `block_count: u64`. Sus getters no modifican los índices internos.
- `BlockInfo`: `id`, `tensor_id`, `first_element`, `term_count`, `data_offset`,
  `data_bytes`, `useful_bits` y `digest`, con los tipos anteriores o u64.
- `DecodedBlock`: `id: BlockId`, `tensor_id: TensorId`, `first_element: u64`,
  `terms: Vec<Bml>`. Posee sus términos; no toma prestado el buffer de lectura.
- `WeightLocation`: `tensor_id: TensorId`, `tensor_name: String`,
  `flat_index: u64`, donde el índice es plano GGML desde cero.
- `FileSummary`: versión, base, tamaños, conteos y hash de fuente/catálogo de
  cabecera; es información validada estructuralmente, no una certificación numérica.

Los offsets/conteos persistidos y los presupuestos públicos usan u64; cualquier
paso a `usize` es fallible. Los índices de APIs no se truncan en hosts de 32 bits.

### Metadatos tipados

`Metadata` expone `schema`, `profile`, `source_gguf_version`, `converter`, `codec`,
`limits` y una lista ordenada `gguf` según format-v1. `ToolIdentity` contiene
`name`, `version`, `revision`. `GgufMetadataEntry` contiene `key` y `value`.

`MetadataValue` es un enum con variantes U8/I8/U16/I16/U32/I32/U64/I64, Bool,
String, `F32Bits(u32)`, `F64Bits(u64)` y Array. `MetadataArray` es un enum de
vectores homogéneos de esos tipos escalares, sin una variante Array de arrays.
La serialización tipada preserva rangos, patrones IEEE y arrays vacíos; no usa
`serde_json::Value` como representación pública ni pierde el tipo de un array vacío.

## 3. Límites

`ReadLimits` tiene un campo u64 por límite de la sección 7 de format-v1:
`max_file_bytes`, `max_metadata_bytes`, `max_metadata_entries`,
`max_metadata_array_elements`, `max_name_bytes`, `max_string_bytes`,
`max_tensors`, `max_blocks`, `max_index_bytes`, `max_block_terms`,
`max_block_bytes`, `max_weight_bits`, `max_weight_depth`, `max_total_terms`.
`Default` usa exactamente esa tabla. `validate()` comprueba coherencia sin
reservar memoria. El límite de profundidad JSON 8 y rank máximo 4 son reglas v1,
no presupuestos ampliables por la API.

`WriteOptions` contiene `limits: ReadLimits` para garantizar que el resultado
puede releerse bajo ese perfil, más los límites de conversión de format-v1.
`ConversionOptions` contiene `write: WriteOptions` y
`synthesis: ExactSynthesisOptions`. Ambos se validan al iniciar; prevalece el
menor de los límites aplicables a un peso. Los límites globales de trabajo,
salida y memoria se contabilizan acumulativamente con overflow comprobado.
La implementación debe instrumentar el consumo real de síntesis: multiplicar
el número de pesos por el máximo configurado no es una medición de consumo.

## 4. Reader y verificación

```rust,ignore
impl WeightReader {
    pub fn open(path: impl AsRef<Path>, limits: ReadLimits) -> Result<Self, Error>;
    pub fn summary(&self) -> &FileSummary;
    pub fn metadata(&self) -> &Metadata;
    pub fn tensors(&self) -> &[TensorInfo];
    pub fn tensor(&self, id: TensorId) -> Result<&TensorInfo, Error>;
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo>;
    pub fn block_info(&self, id: BlockId) -> Result<&BlockInfo, Error>;
    pub fn read_block(&mut self, id: BlockId) -> Result<DecodedBlock, Error>;
    pub fn verify_all(&mut self) -> Result<VerificationReport, Error>;
}
```

`WeightReader` posee un `std::fs::File` abierto en lectura y los índices acotados;
v1 no exige mmap ni punteros inseguros. `open` valida cabecera, catálogo,
metadatos, índices, rangos y tamaño, pero no lee todo el payload. `read_block`
comprueba hash y estructura antes de devolver un bloque completo. Usa `&mut self`
para hacer explícito el cursor y no promete lecturas concurrentes sobre un handle.
El usuario mantiene inmutable el archivo abierto; un handle no es un snapshot.

`VerificationReport` contiene `blocks_checked`, `terms_checked`, `bytes_checked`
y `all_payload_blocks_checked`. `verify_all` recorre todos los bloques en orden,
descarta los términos de cada bloque antes del siguiente y solo retorna éxito
con `all_payload_blocks_checked=true`. Un error no se convierte en informe exitoso.
El informe acredita estructura/integridad, nunca equivalencia con el GGUF fuente.
`bytes_checked` cuenta bytes de bloques BMLS, incluidas sus cabeceras; no vuelve
a sumar la cabecera del contenedor ni el catálogo ya comprobados por `open`.
Un inspector que solo llama `open` debe mostrar payload no verificado.

`read_block` no evalúa floats, no sintetiza, no reconoce expresiones ni transforma
el orden. Los consumidores pueden convertir dimensiones a row-major invirtiendo
solo la lista de dimensiones y decidir su propio presupuesto de materialización.

## 5. Writer transaccional

```rust,ignore
impl WeightWriter {
    pub fn create(
        destination: impl AsRef<Path>, metadata: Metadata,
        source_sha256: Digest, options: WriteOptions,
    ) -> Result<Self, Error>;
    pub fn begin_tensor(
        &mut self, name: &str, ggml_dimensions: &[u64], source_dtype: SourceDtype,
    ) -> Result<TensorId, Error>;
    pub fn write_block(&mut self, weights: &[CertifiedBml]) -> Result<BlockId, Error>;
    pub fn end_tensor(&mut self) -> Result<(), Error>;
    pub fn finish(self) -> Result<WriteReport, Error>;
    pub fn abort(self) -> Result<(), Error>;
}
```

Solo hay un tensor activo. `begin_tensor` valida nombre/forma y el estado;
`write_block` requiere tensor activo, bloque no vacío y capacidad restante.
El writer verifica las derivaciones bajo presupuesto antes de serializar los
términos; aceptar una lista de `Bml` sin certificar no es esta API. La certificación
demuestra el valor del término, no que el llamador eligió el peso fuente correcto.
`end_tensor` exige el conteo exacto. `finish` exige al menos un tensor completo y
ninguno activo, valida el archivo completo y publica sin sobrescribir.

Un error de una operación mutadora deja el writer abortado: no admite continuar
ni publicar. `abort` elimina temporales y devuelve errores de limpieza; `Drop`
hace limpieza best-effort sin panic, sin convertir un fallo en éxito. Ninguna
operación modifica el GGUF fuente. `create` no crea el destino final vacío.
Si otro proceso crea el destino antes de `finish`, este falla con DestinationExists.

La implementación publica mediante `std::fs::hard_link` desde un temporal vecino,
sin fallback a rename con reemplazo. Un filesystem sin soporte devuelve Io/Publish.
El spool se elimina antes del punto de publicación; el enlace temporal final se
limpia best-effort después y Drop reintenta. Tras publicar no se devuelve un error
de limpieza que pueda confundirse con ausencia del destino. No se promete fsync
del archivo/directorio ni durabilidad ante pérdida de alimentación.

`create` ordena las entradas GGUF por clave y fija `metadata.limits` a los límites
de conversión efectivamente solicitados. El writer cuenta pasos de replay por
bloque y acumulados, y aplica límites de bits, profundidad, términos, índices y
salida antes de escribir. La contabilidad conservadora de memoria incluye catálogo,
buffers y working sets acotados del replay; no representa una medición de RSS.

Para el layout de cabecera/catálogo antes del payload, el writer usa un spool de
bloques y ensambla el temporal final cuando conoce tamaños e índices. Los bytes
temporales en disco y su posible duplicación se contabilizan en `WriteReport`;
la memoria no crece con el payload completo. `max_output_bytes` limita el archivo
final; el espacio temporal requerido se documenta y un fallo de disco propaga Io.

`WriteReport`: ruta publicada, resumen del archivo, SHA-256 completo del resultado,
conteos y pico de bytes temporales. No contiene tasas de inferencia ni afirma
persistencia ante pérdida de alimentación: la publicación es atómica sin reemplazo,
y cualquier garantía adicional de durabilidad debe documentarse por plataforma.

## 6. Adaptador GGUF optativo

```rust,ignore
// Disponible solo con feature gguf-import.
pub fn convert::from_gguf(
    source: impl AsRef<Path>, destination: impl AsRef<Path>,
    options: ConversionOptions,
) -> Result<ConversionReport, Error>;
```

La función rechaza identidad fuente/destino, destinos existentes, tipos no
admitidos, dimensiones inválidas y valores no finitos. Extrae metadatos sin
perder sus tipos, decodifica por bloques, obtiene el objetivo racional exacto,
sintetiza/certifica y entrega pesos al writer. Los fallos posteriores al inicio
de un tensor incluyen `WeightLocation` cuando se puede identificar un peso.
El primer error sigue orden de tensor GGUF y de índice plano; no hay fallback.

`ConversionReport` contiene `write: WriteReport`, identidad y bytes de fuente,
conteos de pesos sintetizados/reutilizados, pasos reales de síntesis, bytes/bits
de árboles, picos de memoria contabilizada/caché, y duraciones de lectura/decoding,
síntesis/certificación y escritura/verificación. Duraciones son `Duration`, tamaños
y conteos u64. Memoria contabilizada no se etiqueta como RSS medido del proceso.
La cache tiene presupuesto explícito y conserva el objetivo exacto como clave;
el perfil y opciones de síntesis forman parte de su contexto.

### Contabilidad conjunta y semántica de trabajo v1

`max_synthesis_steps_total` limita **dos contadores independientes**: pasos emitidos
por síntesis en el convertidor y pasos de replay en el writer. Cada contador debe
ser menor o igual al límite; no se resta replay del saldo de síntesis. La misma
regla se aplica por peso. `ConversionReport.synthesis_steps` informa solo síntesis.
La caché ahorra síntesis, nunca replay: tres pesos `1` con caché emiten un paso de
síntesis y requieren tres de replay. Un límite global de dos rechaza el tercer
peso aunque los tres términos procedan de la misma entrada de caché.

El convertidor y el writer comparten `MemoryBudget`, que suma el working set
externo (metadatos GGUF, caché, certificados pendientes y scratch de síntesis)
al catálogo, buffers y scratch de replay del writer. El pico reportado incluye
ambos propietarios simultáneos y la verificación final. La creación limita el
encoder de metadatos al saldo disponible. Antes de `finish` se liberan la caché y
el reader GGUF; vaciar un bloque descarta también la capacidad de su vector.
Los errores de límite no actualizan el pico de operaciones aceptadas.

La contabilidad es conservadora: el catálogo del writer se carga a ocho veces
sus bytes serializados, los buffers de bloque a ocho veces sus bytes, y el scratch
de replay se suma explícitamente. Los certificados pendientes se cargan con un
factor dos para capacidad sobrante, más un margen fijo de 64 KiB. No son medidas
de RSS ni incluyen páginas del mmap, memoria del allocator o librerías del proceso.
Una admisión fallida de caché deja el peso sin cachear y restaura la reserva previa.

## 7. Errores públicos

`Error` y los enums de clasificación son `#[non_exhaustive]`, implementan
`Debug`, `Display` y `std::error::Error`. `Display` es diagnóstico no vacío;
los consumidores hacen matching sobre variantes, no parsing de strings.

| Variante Error | Datos públicos |
|---|---|
| `Io` | `operation: IoOperation`, `path: PathBuf`, `source: std::io::Error` |
| `Unsupported` | `kind: UnsupportedKind`, `value: u64` |
| `Format` | `kind: FormatKind`, `offset: Option<u64>`, `tensor: Option<TensorId>`, `block: Option<BlockId>` |
| `Integrity` | `scope: IntegrityScope`, `expected: Digest`, `actual: Digest` |
| `Limit` | `resource: Resource`, `limit: u64`, `required: Option<u64>`, `location: Option<WeightLocation>` |
| `Index` | `kind: IndexKind`, `index: u64`, `count: u64` |
| `State` | `operation: WriterOperation`, `state: WriterState` |
| `DestinationExists` | `path: PathBuf` |
| `SourceDestinationConflict` | `source: PathBuf`, `destination: PathBuf` |
| `Metadata` | `key: Option<String>`, `reason: MetadataReason` |
| `NonFiniteWeight` | `location: WeightLocation`, `decoded_f32_bits: u32` |
| `Synthesis` | `location: WeightLocation`, `source: ExactError` |
| `Stream` | `block: BlockId`, `source: StreamError` |
| `Gguf` (feature gguf-import) | `tensor: Option<String>`, `source: bml_gguf::GgufError` |

`Error::source()` retorna la causa original para Io/Synthesis/Stream/Gguf y None
para las demás variantes. Conserva `ExactError` para distinguir dominio, búsqueda,
entrada y presupuesto interno; no lo aplana a un string. `required=None` en Limit
significa overflow o tamaño no representable, nunca cero inventado.

Clasificaciones mínimas:

- `IoOperation`: Open, Read, Seek, CreateTemporary, Write, Flush, Publish, Cleanup.
- `UnsupportedKind`: Version, Endian, Basis, Flags, SourceDtype, GgufVersion.
- `FormatKind`: Magic, HeaderLength, Reserved, FileLength, SectionLayout,
  TensorShape, DuplicateTensor, BlockCoverage, BlockHeader, ArithmeticOverflow.
- `IntegrityScope`: Header, Catalog, Block(BlockId).
- `Resource`: un identificador por campo de ReadLimits y por límite adicional
  de síntesis, salida, memoria y caché; los nombres siguen los campos en PascalCase.
- `IndexKind`: Tensor, Block.
- `WriterOperation`: BeginTensor, WriteBlock, EndTensor, Finish.
- `WriterState`: Ready, TensorOpen, Aborted.
- `MetadataReason`: InvalidJson, DuplicateKey, UnknownField, MissingField,
  InvalidType, InvalidValue, InvalidUtf8, UnsupportedProfile, NestedArray.

Un error sintáctico/overflow no se clasifica como agotamiento de síntesis.
Los errores de bloques no exponen términos parciales. Un error de lectura no
invalida automáticamente el handle: una lectura posterior vuelve a verificar
el bloque solicitado; nunca usa un bloque parcialmente decodificado en caché.

## 8. Criterios de implementación

1. Compilar núcleo sin gguf-import; grafo sin compilador/runtime/inferencia.
2. Compilar importación optativa sin alterar dependencias de bml-gguf mínimo.
3. Probar typed source chains, campos de ubicación y prioridades de primer error.
4. Probar acceso inválido, carga parcial explícita y verificación integral acotada.
5. Probar estados inválidos del writer, fallo de síntesis/disco y carrera de destino.
6. Probar carga de bloques sin síntesis ni necesidad del GGUF fuente.

Estas comprobaciones pertenecen a las tareas 2–4; definir las firmas no acredita
su implementación ni permite marcar esas tareas como realizadas.
