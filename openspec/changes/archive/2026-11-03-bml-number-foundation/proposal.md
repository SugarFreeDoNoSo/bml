## Why

El foco de BML es un dato numérico representado mediante una cadena booleana de
la constante `1` y el operador BML. Las APIs simbólica, tensorial y de GGUF
existentes ofrecen piezas útiles, pero aún falta ese tipo público y sus
operaciones sobre la representación.

## What Changes

- Añadir un tipo `Bml` en `bml-domain::number`, sin dependencias adicionales,
  con codificación compacta y validación de la gramática.
- Construir operaciones mediante reglas de reescritura en base 2; proporcionar
  evaluación real comprobada y síntesis de `f64` acotada y verificable.
- Crear la fachada `bml`: el escalar está disponible por defecto, mientras tensor
  y GGUF son capacidades optativas.
- Conservar los contratos de las APIs existentes y relegar la infraestructura
  tensorial, GGUF y de aplicaciones a consumidores de la biblioteca.

## Capabilities

### New Capabilities

- `bml-number`: representación, operaciones y conversiones comprobadas de `Bml`.

### Modified Capabilities

Ninguna especificación canónica se cambia en esta fase.

## Impact

`crates/domain`, nueva fachada `crates/bml`, workspace y pruebas de integración.
La representación y evaluación actuales de `symbolic`, los kernels `tensor` y
el adaptador GGUF mantienen sus contratos. Las aplicaciones y pruebas históricas
siguen siendo consumidores y evidencia independientes.
