## 1. Representación
- [x] 1.1 Implementar `bml-domain::number::Bml` con palabras canónicas y validación de la gramática.
- [x] 1.2 Probar ejemplos, orden, entradas malformadas, igualdad estructural y límites de tamaño.

## 2. Operaciones y conversiones
- [x] 2.1 Implementar reglas base 2 para `exp2`, `log2`, `neg`, `add`, `sub`, `mul`, `div`, `recip`, `pow` y `sqrt` con dominios explícitos.
- [x] 2.2 Implementar evaluación `to_f64` comprobada y síntesis `from_f64` acotada, sin pool.
- [x] 2.3 Validar valores, errores, límites y precisión contra oráculos independientes.
- [x] 2.4 Documentar trigonometría y evaluación compleja como fase posterior.

## 3. Fachada y documentación
- [x] 3.1 Crear crate `bml` con escalar predeterminado y features `tensor` y `gguf`.
- [x] 3.2 Probar features y compatibilidad de APIs existentes; actualizar documentación con el estado real.

## 4. Cierre
- [x] 4.1 `openspec validate bml-number-foundation` pasa sin errores (OpenSpec 1.13.1).
- [x] 4.2 `cargo test --workspace --locked` y el gate aplicable pasan.
- [x] 4.3 Commit y push del refactor, solicitados por el usuario.
