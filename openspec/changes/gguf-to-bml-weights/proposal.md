## Why

La ruta actual conserva pesos GGUF nativos o cuantizados. Incluso los artefactos
AOT simbólicos referencian esos pesos externos. Se necesita una conversión
explícita que produzca un modelo cuyos pesos persistidos sean términos BML,
para preparar las matrices una vez y reutilizarlas al cargar y ejecutar.

## What Changes

- Añadir la ruta GGUF → `.bmlgguf`, nombre provisional del contenedor de pesos
  BML; `.bmllm` queda como alternativa de nombre, no como segundo formato.
- Sintetizar cada peso finito mediante las APIs exactas acotadas y conservar
  nombres, formas, orden GGML, tokenizer, configuración y procedencia.
- Especificar contenedor versionado, índices tensoriales, bloques `BMLS`,
  integridad, límites y publicación transaccional del archivo.
- Añadir conversión e inspección CLI y un cargador independiente del GGUF fuente.
- Integrar los pesos preparados en la ruta simbólica de inferencia y medir
  conversión, carga, tamaño, memoria y ejecución por separado.

## Capabilities

### New Capabilities

- `bml-model-weights`: conversión GGUF a pesos BML persistentes, lectura acotada,
  inspección e integración con inferencia simbólica.

### Modified Capabilities

Ninguna capacidad canónica se reemplaza. La conversión se agrega a la ruta de
modelos y consume los contratos de síntesis exacta, almacenamiento y matrices.

## Impact

Reader/codecs GGUF en `bml-gguf`; síntesis y bloques BML en `bml-domain`;
contenedor/adaptador de modelos en un módulo o crate separado que pueda depender
de ambos; CLI, compilador y runtime como consumidores. `bml-gguf` sin features
sigue sin dependencias de dominio, compilador o runtime. La entrega requiere
evidencia nueva: los artefactos `.bmlgraph` y `.bmlres` actuales no la acreditan.
