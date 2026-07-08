# Roadmap: sub-fragmentación L1i + pesos BML + scheduler de waves

## Etapa 1: Sub-fragmentación L1i

**Objetivo:** partir cada fragmento de capa (168 MB) en sub-fragmentos de <30 KB
que caben en L1i, con cambio de hot loop en runtime.

### Tareas

1. **Nuevo tipo `SubFragment`** en `crates/compiler/src/distributed.rs`
   - `fragment_id`, `sub_id`, `layer_start`, `layer_end`
   - `ops: Vec<RpnOp>` (bytecode < 30 KB)
   - `weight_refs: Vec<(tensor_name, offset, len)>` (referencias a pesos, no copias)
   - `depends_on: Vec<sub_id>` (para el scheduler)

2. **Función `sub_fragment_layer()`** en `distributed.rs`
   - Toma un `DistributedFragment` (capa completa)
   - Lo parte en sub-fragmentos de ~30 KB de bytecode
   - Los pesos se referencian por offset (no se copian)
   - Retorna `Vec<SubFragment>`

3. **Serialización de sub-fragmentos** en `distributed.rs`
   - `serialize_sub_fragments(dir, &frag, &sub_frags)`
   - Cada sub-fragmento es un archivo `sub_N_M.bmlgraph`
   - El fragmento padre (`fragment_N.bmlgraph`) mantiene los pesos completos

4. **Runtime: cambio de hot loop** en `crates/runtime/src/runtime.rs`
   - `execute_sub_fragments(sub_frags, weights, ctx)`
   - Carga cada sub-fragmento en L1i secuencialmente
   - Ejecuta sin allocs, cambiando solo el slice de ops

5. **Tests**
   - Sub-fragmentación de capa sintética
   - Verificar que cada sub-fragmento < 30 KB
   - Ejecución secuencial de sub-fragmentos = ejecución de capa completa

### Commit
```
feat(distributed): sub-fragmentación L1i (<30 KB) con cambio de hot loop
```

---

## Etapa 3: Pesos como árboles BML nativos

**Objetivo:** integrar `RealEncoder` en el compilador distribuido para que
los pesos se almacenen como árboles BML (Const pool + NodeIds) en lugar
de blobs f32 opacos.

### Tareas

1. **`BmlWeightPool`** en `crates/compiler/src/distributed.rs`
   - Usa `RealEncoder` internamente
   - `encode(f32) -> ConstId` (deduplica)
   - `const_pool() -> &[f64]` (valores únicos)
   - `node_table() -> &[NodeData]` (árboles BML)
   - Estadísticas: n_unique, n_total, compression_ratio

2. **Integrar en `DistributedFragment`**
   - Campo `weight_pool: BmlWeightPool` (reemplaza `weights: Vec<f32>`)
   - Serialización: const_pool + node_table + weight_indices
   - Deserialización: reconstruir BmlWeightPool desde bytes

3. **Compilar GGUF con pesos BML**
   - `compile_distributed_bml(gguf_path, layers_per_fragment)`
   - Lee pesos Q4_0 → dequantiza → `BmlWeightPool::encode()`
   - Reporta estadísticas de compresión

4. **Test de roundtrip**
   - Codificar pesos Q4_0 → serializar → deserializar → verificar valores
   - Verificar que const_pool tiene ~14 entradas para Q4_0 puro

### Commit
```
feat(domain): pesos como árboles BML nativos con BmlWeightPool
```

---

## Etapa 2: Scheduler de waves paralelas/seriales

**Objetivo:** DAG de sub-fragmentos con dependencias, ejecutado en waves
paralelas separadas por barreras.

### Tareas

1. **`WaveScheduler`** en `crates/runtime/src/scheduler.rs`
   - Input: lista de sub-fragmentos con `depends_on`
   - Construye topological order
   - Identifica waves (conjuntos de sub-fragmentos sin dependencias entre sí)
   - `next_wave() -> Vec<sub_id>` (sub-fragmentos listos para ejecutar)
   - `mark_done(sub_id)` (libera dependientes)

2. **Ejecución paralela con barreras** en `runtime.rs`
   - `execute_with_scheduler(sub_frags, scheduler, n_cores)`
   - Cada wave se ejecuta en paralelo con threads
   - Barrera entre waves (esperar a que todos terminen)
   - Resultados intermedios via ResultBuffer

3. **Coordinador distribuido con waves**
   - `bml-cli distribute` usa el scheduler para asignar sub-fragmentos a nodos
   - Cada nodo ejecuta su wave en paralelo
   - Barrera distribuida via TCP (coordinador espera a todos)

4. **Tests**
   - DAG simple: A→B→C (todo serial, 3 waves de 1)
   - DAG paralelo: A,B → C (wave 1: A,B paralelo; wave 2: C)
   - Verificar que el orden topológico respeta dependencias

### Commit
```
feat(runtime): scheduler de waves paralelas/seriales con DAG de dependencias
```

---

## Etapa final: benchmarks actualizados

### Tareas

1. **Borrar resultados anteriores**
   - `rm docs/benchmarks/bml_results.*`
   - `rm docs/benchmarks/llamacpp_*.json`
   - `rm docs/benchmarks/REPORT.md`
   - `rm docs/benchmarks/HARDWARE.md`

2. **Re-ejecutar todo**
   - llama-bench (pp, tg, combined)
   - bml-bench (single, multicore)
   - Micro-benchmarks (criterion)
   - Medir tamaño del hot loop post-refactor

3. **Nuevo REPORT.md** con resultados actualizados
   - Incluir sección de sub-fragmentación L1i
   - Incluir sección de compresión BML de pesos
   - Incluir sección de scheduler de waves

### Commit
```
bench: resultados actualizados con sub-fragmentación + pesos BML + scheduler
```

---

## Resumen

| Etapa | Qué | Archivos nuevos/modificados | Líneas aprox |
|-------|-----|---------------------------|-------------|
| 1 | Sub-fragmentación L1i | `distributed.rs`, `runtime.rs` | ~300 |
| 3 | Pesos BML nativos | `distributed.rs`, `encoder.rs` | ~250 |
| 2 | Scheduler de waves | `scheduler.rs` (nuevo), `runtime.rs` | ~350 |
| Final | Benchmarks | `docs/benchmarks/*` | ~200 |
| **Total** | | | **~1100** |
