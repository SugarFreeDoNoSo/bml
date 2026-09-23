## Context

La [revisión reproducible](../../../docs/architecture/REVIEW_2026-09-20.md)
identifica errores de RPN, contexto de constantes, dominio de multiplicación,
alignment GGUF, orden de Q4_0 y truncamiento. La referencia de arquitectura está en
[`TENSOR_LIBRARY.md`](../../../docs/architecture/TENSOR_LIBRARY.md) y la matriz de
pruebas en [`TESTING.md`](../../../docs/TESTING.md).

## Goals / Non-Goals

- G1: expresar con tipos la diferencia entre términos cerrados y abiertos.
- G2: conservar la gramática pura y definir evaluación parcial comprobada.
- G3: hacer verificable una biblioteca tensor/GGUF local antes de optimizar.
- G4: permitir comparar nativo, EML y BML bajo el mismo contrato.
- NG1: prometer síntesis exacta de cualquier `f64` mediante una conversión.
- NG2: implementar inferencia completa o distribución en esta primera etapa.
- NG3: reemplazar el almacenamiento de pesos por un árbol escalar por elemento.

## Decisions

- D1: `ClosedTerm` y `OpenTerm` con enum/Box para fórmulas pequeñas; la arena se
  reserva para compilación y grafos grandes.
- D2: el árbol es sintaxis; `Basis` fija la interpretación de todos sus nodos.
  El plan serializado futuro conserva base y política numérica.
- D3: `bind` permanece parcial y `collapse` devuelve una variable libre como
  error tipado. El conjunto completo se consulta con `free_variables`.
- D4: `evaluate_real` toma prestado el árbol. El contrato finito rechaza `log(0)`
  y se distingue de un evaluador complejo o de reales extendidos futuro.
- D5: se añade el módulo en `domain` antes de renombrar/mover crates, preservando
  consumidores y evitando mezclar reparación de semántica con movimientos masivos.
- D6: los kernels tensoriales conservan operaciones de alto nivel. Los backends
  experimentales se activan por capacidades, con dominios y errores conocidos.
- D7: las seis reproducciones son evidencia de deuda real; no se marcan resueltas
  por añadir documentación o tests que permitan `NaN`.

## Risks / Trade-offs

- R1: clonación, recorridos y destrucción recursiva de Box limitan profundidad.
  Se documenta el alcance y se exige arena antes de admitir grafos grandes.
- R2: el evaluador real estricto no reproduce todo el artículo. Las extensiones
  requieren políticas explícitas de ramas y no se activan automáticamente.
- R3: existen expectativas de aceleración sin evidencia. Se separa resultado
  formal, hipótesis de optimización y medición reproducible.
- R4: mover crates puede romper imports. Se migra por etapas con reexports
  transitorios y pruebas sobre la fachada.
