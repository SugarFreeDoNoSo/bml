use bml_compiler::{fragment_program, linearize, BmlGraph, HashConsRegistry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // API histórica AST/SoA/RPN: distinta de los términos compactos Bml.
    let mut registry = HashConsRegistry::new();
    let one = registry.one();
    let two = registry.bml(one, one);
    let repeated = registry.bml(one, one);
    println!("Subexpresión compartida: {} (id={two})", two == repeated);
    let soa = registry.into_soa();
    let program = linearize(&soa, two);
    println!(
        "Nodos SoA: {}; instrucciones RPN: {}",
        soa.len(),
        program.len()
    );
    println!("Evaluación RPN: {}", program.evaluate());
    let graph = fragment_program(&program, 32);
    let bytes = graph.serialize();
    let restored = BmlGraph::deserialize(&bytes, 32)?;
    println!("Grafo serializado: {} bytes", bytes.len());
    println!("Evaluación tras restaurar: {}", restored.evaluate(0.0));
    Ok(())
}
