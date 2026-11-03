//! Shared equal-work workload for timing and allocation measurements.
use bml_compiler::{linearize, HashConsRegistry, RpnProgram};
use bml_domain::NodeSoA;

pub fn build_program(steps: usize, deduplicate: bool) -> RpnProgram {
    if deduplicate {
        let mut registry = HashConsRegistry::new();
        let mut node = registry.var(0);
        for _ in 0..steps {
            let exponent = registry.const_value(1.0);
            node = registry.bml(exponent, node);
        }
        linearize(&registry.into_soa(), node)
    } else {
        let mut soa = NodeSoA::new();
        let mut node = soa.push_var(0);
        for _ in 0..steps {
            let id = soa.const_pool.len() as u32;
            soa.const_pool.push(1.0);
            let exponent = soa.push_const(id);
            node = soa.push_bml(exponent, node);
        }
        linearize(&soa, node)
    }
}

pub fn direct(steps: usize, mut value: f64, exponent: f64) -> f64 {
    for _ in 0..steps {
        value = exponent.exp2() - value.log2();
    }
    value
}
