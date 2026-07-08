# Plan: distribución cross-machine con .bmlgraph

## Situación actual

### Lo que `.bmlgraph` almacena hoy

```
header.bmlgraph:
  [magic 4B][version 4B][n_fragments 4B]
  [arch_len 8B][arch_str][n_layers 4B][n_heads 4B][n_embd 4B][ctx 4B][vocab 4B]
  [const_pool_len 8B][const_pool: N × 8B f64]

fragment_N.bmlgraph:
  [n_ops 4B][ops: tag + args...]
```

**Contenido:** metadata del modelo + RPN ops (bytecode) + const_pool.
**NO contiene:** pesos del transformer (Q, K, V, MLP, embeddings, norm).

### Dónde están los pesos

Los pesos se cargan en runtime desde el GGUF original:
```rust
// InferenceCompiler::open() lee TODOS los pesos del GGUF a un Vec<f32>
// TinyLlama 1.1B Q4_0 → 1,034,518,528 valores → ~3.9 GB en f32
let compiler = InferenceCompiler::open(gguf_path)?;
```

### Lo que ya existe para distribución

| Componente | Archivo | Estado |
|-----------|---------|--------|
| Protocolo TCP raw | `runtime/src/net.rs` | ✅ framing, send/recv, NodeHandle |
| Work-stealing | `runtime/src/queue.rs` | ✅ WorkQueueSet, steal_from |
| Memoria compartida | `runtime/src/shm.rs` | ✅ /dev/shm same-machine |
| Ejecución paralela | `runtime/src/runtime.rs` | ✅ execute_fragments_parallel |
| Mensajes | `net.rs` | ✅ ExecuteFragment, ReportResult, StealWork, HealthCheck, Batch |

### Problema central

**El `.bmlgraph` actual es inútil sin el GGUF en la máquina.** Los fragmentos
contienen solo el bytecode RPN (pocos bytes), pero las operaciones del
transformer (matmul, attention, RoPE) se ejecutan en `InferenceCompiler` que
necesita los pesos en RAM. Los pesos NO están en el `.bmlgraph`.

---

## Arquitectura objetivo

```
                    ┌─────────────────┐
                    │  Coordinador    │
                    │  (bml-cli run)  │
                    └────┬────────────┘
                         │ TCP
            ┌────────────┼────────────┐
            │            │            │
     ┌──────▼──┐  ┌──────▼──┐  ┌──────▼──┐
     │ Nodo 0  │  │ Nodo 1  │  │ Nodo 2  │
     │frag 0-2 │  │frag 3-5 │  │frag 6-7 │
     │pesos 0-2│  │pesos 3-5│  │pesos 6-7│
     └─────────┘  └─────────┘  └─────────┘
```

Cada nodo:
1. Carga solo sus fragmentos (bytecode + pesos de esos fragmentos)
2. Ejecuta su porción del transformer
3. Reporta resultados al coordinador via TCP

---

## Lo que falta (7 items)

### 1. Embedir pesos en .bmlgraph

**Problema:** Los fragmentos actuales solo tienen bytecode RPN. Los pesos
(Q, K, V, MLP, embeddings, norm) se cargan desde el GGUF en runtime.

**Solución:** Cada fragmento `.bmlgraph` debe contener:
```
fragment_N.bmlgraph:
  [n_ops 4B]
  [ops...]
  [n_weights 8B]
  [weights: N × 4B f32]   ← NUEVO
  [weight_offsets: map de tensor_name → offset]  ← NUEVO
```

Cada fragmento es self-contained: bytecode + los pesos que necesita.

**Cambio necesario:**
- `Fragment` struct: agregar campo `weights: Vec<f32>`
- `serialize_to_dir`: escribir pesos por fragmento
- `load_from_dir`: leer pesos por fragmento
- `compile_gguf_fast`: particionar pesos por fragmento (no cargar todos en un pool global)

### 2. Fragmentación por capa (no por tamaño de bytecode)

**Problema:** La fragmentación actual (`fragment_program`) parte el programa
RPN por tamaño de bytecode (32 KB L1). Pero los pesos del transformer son
O(n_params) = GB, no caben en un fragmento de 32 KB.

**Solución:** Fragmentar por **capa del transformer** en lugar de por tamaño
de bytecode. Cada capa del transformer = un fragmento:
```
fragment_0.bmlgraph = capa 0 (Q, K, V, O, norm, MLP gate/up/down)
fragment_1.bmlgraph = capa 1
...
fragment_N.bmlgraph = capa N + final norm + lm_head
```

Cada fragmento tiene sus propios pesos (~180 MB para TinyLlama 1.1B Q4_0
por capa, 22 capas → ~3.9 GB total).

**Cambio necesario:**
- Nuevo `compile_gguf_distributed()` que lee el GGUF capa por capa
- Cada capa → un fragmento con bytecode + pesos de esa capa
- El bytecode usa `VarIndexed` para indexar pesos dentro del fragmento

### 3. Formato de fragmento distribuido (v2)

```
fragment_N.bmlgraph:
  [magic 4B][version 4B]
  [fragment_id 4B]
  [layer_range: start 4B, end 4B]  ← qué capas cubre
  [n_ops 4B][ops...]
  [n_weights 8B][weights: N × 4B f32]
  [n_tensor_maps 4B]
    [name_len 4B][name][offset 4B][n_rows 4B][n_cols 4B] × n_tensor_maps
  [vocab_subset: tokens relevantes para este fragmento]  ← opcional
```

Un nodo puede cargar un solo `fragment_N.bmlgraph` y tener todo lo que necesita.

### 4. Coordinador distribuido

**Problema:** No existe un coordinador que distribuya fragmentos a nodos.

**Solución:** Nuevo binario o subcomando `bml-cli distribute`:

```sh
bml-cli distribute \
  --graph modelo.bmlgraph/ \
  --nodes 192.168.1.10:9999,192.168.1.11:9999,192.168.1.12:9999 \
  -p "Hello" -n 64
```

El coordinador:
1. Carga el header.bmlgraph (metadata + const_pool)
2. Conecta a cada nodo via TCP (`NodeHandle::connect`)
3. Envía cada fragmento al nodo correspondiente (`send_fragment`)
4. Espera resultados (`recv_result`)
5. Ensambla el resultado final

**Cambio necesario:**
- Nuevo módulo `runtime/src/coordinator.rs`
- Lógica de asignación de fragmentos a nodos (round-robin o least-loaded)
- Manejo de fallas (retry en otro nodo)

### 5. Worker daemon en cada nodo

**Problema:** No existe un proceso que escuche TCP y ejecute fragmentos.

**Solución:** Nuevo binario `bml-worker`:

```sh
bml-worker --port 9999
```

El worker:
1. Escucha TCP en el puerto
2. Recibe `ExecuteFragment` con bytes del fragmento
3. Deserializa el fragmento (bytecode + pesos)
4. Lo carga en un `Runtime` local
5. Ejecuta y responde `ReportResult`

**Cambio necesario:**
- Nuevo crate `crates/worker/` con binario `bml-worker`
- TCP server loop (`TcpListener::accept`)
- Deserialización de fragmento desde bytes recibidos via TCP
- Pool de runtimes pre-asignados (uno por core)

### 6. Tokenización y sampling distribuidos

**Problema:** La tokenización y sampling se hacen en el coordinador, pero
la inferencia (forward pass) se distribuye. El forward pass necesita:
- Embedding lookup (token IDs → vectores)
- N capas de transformer
- lm_head → logits
- Sampling → next token ID

**Arquitectura:**

```
Coordinador:
  1. tokenizer.encode(prompt) → token_ids
  2. embedding_lookup(token_ids) → hidden_state  (en el coordinador, chico)
  3. Distribuir hidden_state a nodo 0
  4. Nodo 0 ejecuta capas 0-7 → envía hidden a nodo 1
  5. Nodo 1 ejecuta capas 8-14 → envía hidden a nodo 2
  6. Nodo 2 ejecuta capas 15-21 + final norm + lm_head → logits
  7. Coordinador recibe logits → sample → next token
  8. Repetir para cada token (autoregresivo)
```

**Cambio necesario:**
- El coordinador hace embedding lookup + sampling (liviano)
- Los nodos hacen forward pass de su rango de capas (pesado)
- Los resultados intermedios (hidden state) se pasan entre nodos via TCP
- El último nodo produce logits y los envía al coordinador

### 7. Streaming de pesos (lazy loading)

**Problema:** Cargar todos los pesos de un fragmento en RAM toma tiempo.

**Solución:** Usar mmap para los pesos del fragmento:
- El fragmento `.bmlgraph` se escribe con los pesos al final del archivo
- El worker hace mmap del archivo → cero copia a userspace
- Los pesos se leen via page cache del kernel (lazy)
- Solo las páginas accedidas se cargan en RAM

**Cambio necesario:**
- `Fragment` con path al archivo mmap'ed en lugar de Vec<f32> owned
- Usar `memmap2` (ya es dep del parser) para mapear pesos del fragmento

---

## Estado de implementación

### ✅ Completado

| Item | Descripción | Commit |
|------|-------------|--------|
| 1 | Embedir pesos en .bmlgraph | `806488f` |
| 2 | Fragmentación por capa | `806488f` |
| 3 | Formato fragmento v2 self-contained | `806488f` |
| 4 | Coordinador (bml-cli distribute) | `e850eb3` |
| 5 | Worker daemon (bml-worker) | `806488f` |

### Funciona end-to-end

```
bml-cli compile --distributed -m modelo.gguf -o modelo.bmlgraph
  → 23 fragmentos self-contained (168 MB c/u, 3.9 GB total)

bml-worker --port 9999 --fragment fragment_0.bmlgraph
  → Daemon TCP que ejecuta fragmentos

bml-cli distribute -m modelo.bmlgraph/ --nodes host1:9999,host2:9999 -p "Hello" -n 16
  → Coordinador que reparte fragmentos y orquesta inferencia
```

### ⏳ Pendiente

| Item | Descripción |
|------|-------------|
| 6 | Pipeline autoregresivo completo (embedding real, attention real, RoPE real en workers) |
| 7 | Lazy loading mmap de pesos |
