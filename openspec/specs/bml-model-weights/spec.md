# bml-model-weights Specification

## Purpose

Convertir pesos GGUF admitidos en términos BML exactos y persistentes dentro de
un contenedor autónomo, con límites explícitos, integridad verificable,
publicación transaccional y carga reutilizable para inferencia simbólica.

## Requirements

### Requirement: Conversión explícita de pesos GGUF a BML
El convertidor SHALL producir un archivo `.bmlgguf` versionado cuyos pesos sean
términos de la gramática `T ::= 1 | B(T,T)` en base 2. SHALL incluir los pesos,
formas, nombres y metadatos necesarios para el perfil admitido. SHALL distinguir
este archivo de un GGUF renombrado y de un programa con pesos externos.

#### Scenario: Modelo convertido autónomo
- **WHEN** termina correctamente la conversión y se retira el GGUF original
- **THEN** el reader recupera todos los pesos y metadatos del contenedor convertido
- **AND** no necesita acceder al archivo original para cargar el modelo.

### Requirement: Valor exacto y política de entrada
Para F32/F16 el objetivo SHALL ser el racional exacto del valor IEEE almacenado.
Para pesos cuantizados SHALL ser el racional exacto del F32 producido por el
codec declarado y verificado. Cada término exitoso SHALL tener una justificación
exacta; SHALL rechazarse NaN/inf y normalizarse ambos ceros al cero matemático.

#### Scenario: Peso cuantizado
- **WHEN** se convierte un tensor cuantizado admitido
- **THEN** la procedencia identifica dtype, codec y política de decodificación
- **AND** la certificación se refiere al valor decodificado, no al peso anterior a la cuantización.

#### Scenario: Síntesis no certificada
- **WHEN** no se obtiene una representación exacta dentro del presupuesto
- **THEN** la conversión falla indicando tensor e índice y causa
- **AND** no guarda una aproximación ni un float como si fuera un peso BML exacto.

### Requirement: Formato acotado e integridad
El contenedor SHALL cumplir el contrato binario v1 de
`openspec/specs/bml-model-weights/format-v1.md`: cabecera de 256 bytes,
metadatos tipados, índice tensorial variable e índice de bloques de 80 bytes.
SHALL identificar BML base 2, usar bloques BMLS
versionados y validar hashes, conteos, formas, rangos y límites antes de reservar.
SHALL rechazar versiones desconocidas, truncamientos y solapamientos inválidos.

#### Scenario: Archivo manipulado
- **WHEN** un bloque tiene hash incorrecto, rango fuera del archivo o conteo incompatible
- **THEN** el reader devuelve un error explícito sin entregar una matriz parcial.

#### Scenario: Hashes y fronteras del archivo
- **WHEN** se valida un archivo v1
- **THEN** se comprueban el SHA-256 de cabecera con su propio campo a cero, el del catálogo y el de cada bloque antes de exponerlo
- **AND** las secciones y bloques cubren exactamente sus rangos contiguos, sin bytes extra, huecos ni alias.

#### Scenario: Límite inclusivo y presupuesto local
- **WHEN** un recurso ocupa exactamente el límite permitido o lo excede en uno
- **THEN** el primero puede aceptarse si cumple los demás invariantes y el segundo se rechaza antes de reservar
- **AND** los presupuestos declarados en el archivo no amplían los del reader.

### Requirement: Orden tensorial conservado
El convertidor SHALL preservar nombres, dimensiones GGML y secuencia plana de
pesos, con eje cero contiguo. El adaptador row-major SHALL invertir dimensiones
sin reordenar la secuencia plana. SHALL comprobar productos de dimensiones y
cantidad de términos con aritmética verificada.

#### Scenario: Tensor rectangular
- **WHEN** se convierte un tensor GGML de dimensiones `[3, 2]` con seis pesos distintos
- **THEN** el archivo conserva esos seis pesos en su orden original
- **AND** la vista row-major tiene forma `[2, 3]` y la misma secuencia plana.

### Requirement: Presupuestos y publicación transaccional
La conversión SHALL aplicar límites por peso y globales de síntesis, profundidad,
bits, cantidad, memoria y bytes de salida. SHALL procesar bloques acotados y
publicar el destino sin sobrescribir solo tras completar y verificar el archivo.
Un fallo SHALL preservar la fuente y cualquier destino existente.

#### Scenario: Presupuesto agotado a mitad del modelo
- **WHEN** falla la síntesis de un peso tras convertir bloques anteriores
- **THEN** no aparece un destino que pueda confundirse con una conversión completa
- **AND** el diagnóstico localiza tensor, índice y presupuesto agotado.

### Requirement: Carga de pesos preparados e inferencia
El cargador SHALL entregar bloques de términos BML sin repetir su síntesis.
La ruta simbólica SHALL consumir esos pesos con tokenizer/configuración propios
del contenedor. SHALL distinguir exactitud de almacenamiento y dominio/rango de
evaluación; no SHALL sustituir silenciosamente pesos por aproximaciones nativas.

#### Scenario: Inferencia desde el contenedor convertido
- **WHEN** se ejecuta un fixture admitido desde `.bmlgguf` sin su GGUF fuente
- **THEN** los términos preparados se reutilizan y los logits/tokens se contrastan con un oráculo independiente
- **AND** los fallos de evaluación se comunican como tales, sin atribuirlos a una conversión exacta exitosa.

### Requirement: Evidencia reproducible por fase
El informe SHALL identificar fuente, hashes, codec, presupuestos y comandos.
SHALL medir conversión, carga, tamaño, pico de memoria y ejecución por separado.
SHALL registrar modelos no convertidos o no ejecutados y no atribuir compresión,
paridad o aceleración a partir únicamente de la representación BML.

#### Scenario: Campaña real sin éxito completo
- **WHEN** un modelo real excede el presupuesto de conversión o el rango de evaluación
- **THEN** se conserva el diagnóstico reproducible y la fase alcanzada
- **AND** no se declara validada la conversión e inferencia completa de ese modelo.

### Requirement: API independiente y errores tipados
El crate `bml-model-weights` SHALL cumplir la API descrita en
`openspec/specs/bml-model-weights/api.md`. La importación GGUF SHALL ser una
feature optativa. El núcleo SHALL carecer de dependencias de compilador, runtime
e inferencia; el reader GGUF sin features SHALL conservar su independencia.
Los errores SHALL preservar causas tipadas y ubicación de pesos cuando esté disponible.

#### Scenario: Núcleo sin importación GGUF
- **WHEN** se compila el contenedor sin features
- **THEN** reader/writer de bloques BML están disponibles sin incorporar bml-gguf, compilador, runtime ni inferencia
- **AND** habilitar gguf-import añade el adaptador, sin introducir una dependencia inversa en bml-gguf.

#### Scenario: Fallo localizado de síntesis
- **WHEN** falla la síntesis de un peso durante la conversión
- **THEN** Error::Synthesis conserva tensor, índice plano y ExactError como causa consultable
- **AND** el writer aborta sin publicar un destino parcial.

#### Scenario: Inspección parcial explícita
- **WHEN** el consumidor abre el contenedor sin recorrer sus bloques
- **THEN** puede consultar índices y metadatos validados pero no declara verificado el payload completo
- **AND** verify_all solo informa éxito integral después de comprobar todos los bloques.
