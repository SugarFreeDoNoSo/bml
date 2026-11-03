## Context

El dominio ya incluye `ClosedTerm = One | Op(ClosedTerm, ClosedTerm)` y una
interpretación real comprobada. El encoder heredado usa un pool de constantes,
por lo que no constituye una representación booleana pura de un `f64`. Las
identidades del artículo de EML requieren adaptación y validación para BML base 2.

## Goals / Non-Goals

- G1: representar árboles cerrados como bits canónicos, sin constantes externas.
- G2: construir operaciones por reglas sobre esos árboles, con errores explícitos.
- G3: evaluar y sintetizar valores reales bajo límites de recursos y precisión.
- G4: exponer una fachada escalar pequeña y capacidades tensor/GGUF optativas.
- NG1: garantizar conversión de todo `f64`, equivalencia numérica de árboles,
  estabilidad general, compresión o aceleración.
- NG2: implementar en esta fase trigonometría o evaluación compleja. Se investigarán
  posteriormente junto con ramas logarítmicas y reglas específicas de base 2.

## Decisions

- D1: recorrido prefijo; `0` es `One`, `1` es `Op` seguido de dos subárboles
  ordenados. Ejemplos: `0` = 1, `100` = 2 y `11000` = 4.
- D2: almacenar el entero sin signo de esa secuencia en palabras `u64` de menor
  peso primero, sin longitud persistida. El bit más significativo es el primer
  símbolo; cero representa la secuencia única `0`. Las palabras altas nulas se
  normalizan; se rechazan árboles incompletos o con símbolos sobrantes.
- D3: operaciones construyen términos por sustitución de reglas BML, sin calcular
  con floats para producir el resultado. La igualdad es estructural, no numérica.
- D4: `to_f64` comprueba dominio y rango de cada paso. `from_f64` hace síntesis
  acotada y comprueba el resultado según precisión explícita; informa errores de
  tamaño, precisión y dominio. No usa pool y no garantiza éxito para todo float.
- D5: ofrecer inicialmente `exp2`, `log2`, `neg`, `add`, `sub`, `mul`, `div`,
  `recip`, `pow` y `sqrt`, con dominios y restricciones explícitos y comprobados.
  Las identidades se validan independientemente para base 2 y no prometen que
  cualquier término construido pueda evaluarse en reales finitos.
- D6: fachada `bml` con escalar por defecto, `tensor` y `gguf` mediante features.
  Los contratos de tensor IEEE y GGUF no se redefinen por este tipo.

## Risks / Trade-offs

- R1: explosión del tamaño al expandir reglas. Limitar recursos y devolver error.
- R2: cancelación, overflow y dominio de intermedios. Verificar frente a oráculos
  independientes y devolver errores; no certificar precisión sin medirla.
- R3: una fórmula EML puede requerir complejos o no conservar ramas en base 2.
  Trigonometría y complejos requieren diseño y validación posteriores.
- R4: síntesis limitada puede fallar incluso para valores finitos; explicitar
  presupuesto y resultado en la API.
