## ADDED Requirements

### Requirement: Ingesta GGUF Zero-Copy
El sistema SHALL ingerir archivos GGUF mediante mapeo directo a memoria (`memmap2`), referenciando los tensores desde el disco al espacio de memoria de Rust sin copias a RAM.

#### Scenario: Apertura de un GGUF
- **WHEN** se abre un archivo GGUF válido
- **THEN** se mapea en memoria vía `mmap` y los tensores son accesibles como slices sobre el archivo mapeado

#### Scenario: Ausencia de copias a userspace
- **WHEN** se inspecciona con `strace` la apertura y lectura de tensores
- **THEN** no aparecen syscalls `read` de los tensores a buffers userspace; solo `mmap`

### Requirement: Decodificación de cabeceras y metadatos GGUF
El sistema SHALL decodificar la cabecera mágica y los metadatos del estándar GGUF.

#### Scenario: Cabecera mágica válida
- **WHEN** se abre un archivo GGUF
- **THEN** se valida la cabecera mágica y se rechaza el archivo si no coincide

#### Scenario: Metadatos accesibles
- **WHEN** se decodifica el archivo
- **THEN** los metadatos (versión, número de tensores, kv metadata) quedan accesibles como estructuras tipadas

### Requirement: Lifetime seguro del mmap
El sistema SHALL encapsular el mmap en un RAII guard con lifetime explícito ligado al parser, de forma que los tensores mapeados no sean accesibles tras cerrar el guard.

#### Scenario: Cierre del guard
- **WHEN** el guard del mmap se cierra
- **THEN** cualquier referencia a un tensor mapeado deja de compilar (lifetime agotado)

#### Scenario: Tensor válido mientras el guard vive
- **WHEN** el guard está abierto
- **THEN** los slices de tensores son válidos y legibles
