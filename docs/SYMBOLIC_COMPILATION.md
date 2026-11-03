# Compilación GGUF simbólica

Las entradas `bml_compiler::gguf_compiler::compile_gguf` y `compile_gguf_fast`
construyen el mismo plan AOT causal de `symbolic_artifact`. El plan contiene
matmul, RMSNorm, RoPE, softmax, SwiGLU, escalado de atención y sumas residuales,
construidos mediante `BMLTransformer`, para todos los prefijos del contexto
declarado. La segunda función conserva su nombre histórico, sin promesa de tiempo.

## Migración de la API experimental

El resultado es ahora `symbolic_artifact::CompiledArtifact`. Sustituye al antiguo
`CompilationResult` escalar, que no representaba un transformer ejecutable.
Se consulta con `manifest()` y se publica con `publish(directory)`. Los programas
y su manifiesto están encapsulados para mantener juntas sus validaciones y hashes.
`CompilationResult`, `serialize_to_dir` y `load_from_dir` siguen disponibles para
contenedores escalares históricos; no son el formato causal simbólico.

```rust,ignore
use bml_compiler::{
    gguf_compiler::compile_gguf_with_limits,
    hardware::HardwareSpec,
    symbolic_kernels::KernelLimits,
};

let artifact = compile_gguf_with_limits(
    "model.gguf",
    &HardwareSpec::detect_local(),
    8, // contexto causal explícito
    KernelLimits::default(),
)?;
artifact.publish(std::path::Path::new("model.bmlgraph"))?;
```

Las dos entradas sin contexto explícito usan el límite del modelo. Para modelos
grandes se recomienda `compile_gguf_with_limits`: la compilación puede rechazar
el plan por presupuesto de nodos, opcodes, programas o bytes. No reduce el contexto
silenciosamente. El umbral de fragmento proviene de `HardwareSpec`; en esta API
prevalece sobre `KernelLimits::fragment_bytes`.

`symbolic_artifact::compile(source, context, limits)` permite construir sin
publicar; `compile_directory` combina ambas acciones. La publicación admite un
directorio nuevo o vacío y escribe el manifiesto al final. Un error de E/S puede
dejar archivos de programa sin manifiesto. Los pesos aprendidos se mantienen en
el GGUF externo, identificado por SHA-256; deben permanecer inmutables mientras
existan mappings prestados. La decodificación de filas sí utiliza memoria temporal.

El ejemplo exige contexto explícito:

```sh
cargo run -p bml-compiler --example compile_model --locked -- model.gguf 8 model.bmlgraph
cargo run -p bml-cli --locked -- run --model model.bmlgraph --prompt A --num-tokens 1
```

## Evidencia y alcance

`compiler_entrypoint_contracts` ejecuta ambos caminos contra el oráculo independiente
de una capa GQA con RoPE y MLP no nulo: prefill, tres posiciones y decode incremental.
La carga con `max_nodes=0` impide regenerar kernels. También verifica umbrales,
contexto, presupuestos, persistencia y ausencia de pesos aprendidos en los programas.

Esta migración elimina la ruta escalar placeholder. Las proyecciones desenrolladas,
el plan de prefijos y el rango numérico BML siguen limitando modelos reales.
Construir un plan no demuestra que un prompt concreto sea evaluable: los errores
numéricos son explícitos. La inferencia nativa mantiene un contrato distinto.
