## 8. Scheduler con batching dinámico

- [ ] 8.1 Implementar `crates/runtime/src/scheduler.rs` con cola de requests (`crossbeam-channel`). (Deferred: el servidor actual usa backpressure HTTP simple. Scheduler más complejo puede agregarse cuando el throughput sea necesario.)
- [ ] 8.2 Implementar batching dinámico: agrupa N prompts en una ventana de 10ms. (Deferred.)
- [ ] 8.3 Implementar distribución de batches a nodos: round-robin o least-loaded. (Deferred.)
- [ ] 8.4 Implementar backpressure: si la cola está llena, retornar HTTP 429. (Implementado en bml-server con AtomicUsize y MAX_PENDING_REQUESTS=64.)
- [ ] 8.5 Pruebas de batching: 10 requests concurrentes se agrupan en 1-2 batches. (Deferred.)

## 9. Pruebas de concurrencia

- [ ] 9.1 Pruebas con `loom` de la cola lock-free. (La cola ya existe en queue.rs. Loom tests pueden agregarse.)
- [ ] 9.2 Pruebas con `loom` del work-stealing. (Work-stealing ya existe.)
- [ ] 9.3 Pruebas de append-only bajo estrés multicore. (Ya implementado en tests de runtime.)
- [ ] 9.4 Pruebas de que el hot loop no toca el código de red (aislamiento). (Hot loop + net están en módulos separados.)
- [ ] 9.5 Pruebas de backpressure: HTTP 429. (Implementado con MAX_PENDING_REQUESTS=64.)
- [ ] 9.6 Pruebas de cambio de hot loop: L1i no se contamina entre fragmentos. (Requiere PMU/hardware counters.)

## 10. Cierre

- [ ] 10.1 `openspec validate bml-inference-pipeline` pasa sin errores.
- [x] 10.2 `cargo test --workspace` pasa.
- [ ] 10.3 Commit y push.