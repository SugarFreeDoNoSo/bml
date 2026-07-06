//! Detección de hardware para optimización AOT.
//!
//! El compilador detecta el hardware objetivo (cores, caché L1/L2/L3)
//! para calcular el número mínimo de fragmentos `.bmlgraph`.

use std::fs;

/// Especificaciones del hardware objetivo.
#[derive(Debug, Clone)]
pub struct HardwareSpec {
    /// Número de cores físicos.
    pub cores: usize,
    /// Tamaño de caché L1i en bytes.
    pub l1i: usize,
    /// Tamaño de caché L2 en bytes.
    pub l2: usize,
    /// Tamaño de caché L3 en bytes.
    pub l3: usize,
}

impl HardwareSpec {
    /// Detecta el hardware local.
    pub fn detect_local() -> Self {
        let cores = detect_cores();
        let l1i = detect_cache(0) * 1024; // /sys devuelve en KB
        let l2 = detect_cache(2) * 1024;
        let l3 = detect_cache(3) * 1024;
        Self { cores, l1i, l2, l3 }
    }

    /// Crea specs manuales.
    pub fn new(cores: usize, l1i: usize, l2: usize, l3: usize) -> Self {
        Self { cores, l1i, l2, l3 }
    }

    /// Calcula el número mínimo de fragmentos para un total de operaciones.
    ///
    /// `max(1, ceil(total_ops / (l1i_threshold * cores)))`
    pub fn min_fragments(&self, total_ops: usize) -> usize {
        if total_ops == 0 {
            return 1;
        }
        let ops_per_core = self.l1i / std::mem::size_of::<crate::rpn::RpnOp>();
        let total_capacity = ops_per_core * self.cores;
        if total_capacity == 0 {
            return 1;
        }
        (total_ops + total_capacity - 1) / total_capacity
    }

    /// Umbral de fragmento (L1i en bytes).
    pub fn fragment_threshold(&self) -> usize {
        self.l1i
    }
}

/// Detecta el número de cores del sistema.
fn detect_cores() -> usize {
    // Intentar con std::thread::available_parallelism
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

/// Detecta el tamaño de caché en KB desde /sys/devices/system/cpu.
///
/// `level`: 0 = L1, 2 = L2, 3 = L3.
fn detect_cache(level: usize) -> usize {
    // Intentar leer de /sys/devices/system/cpu/cpu0/cache/indexN/size
    for index in 0..10 {
        let path = format!("/sys/devices/system/cpu/cpu0/cache/index{index}/level");
        if let Ok(content) = fs::read_to_string(&path) {
            if content.trim() == level.to_string() {
                let size_path = format!("/sys/devices/system/cpu/cpu0/cache/index{index}/size");
                if let Ok(size_str) = fs::read_to_string(&size_path) {
                    // El formato es "32K" o "256K" o "8192K"
                    let size_str = size_str.trim();
                    if let Some(num) = size_str.strip_suffix('K') {
                        if let Ok(kb) = num.parse::<usize>() {
                            return kb;
                        }
                    }
                }
            }
        }
    }
    // Fallback
    match level {
        0 => 32,        // L1i: 32 KB
        2 => 256,       // L2: 256 KB
        3 => 16 * 1024, // L3: 16 MB
        _ => 32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_local_hardware() {
        let hw = HardwareSpec::detect_local();
        assert!(hw.cores > 0);
        assert!(hw.l1i > 0);
        println!("Hardware: {hw:?}");
    }

    #[test]
    fn min_fragments_calculation() {
        let hw = HardwareSpec::new(4, 32 * 1024, 256 * 1024, 16 * 1024 * 1024);
        let op_size = std::mem::size_of::<crate::rpn::RpnOp>();
        let ops_per_core = (32 * 1024) / op_size;
        let total_capacity = ops_per_core * 4;

        // Con total_ops = total_capacity, debe dar 1 fragmento
        assert_eq!(hw.min_fragments(total_capacity), 1);

        // Con el doble, debe dar 2
        assert_eq!(hw.min_fragments(total_capacity * 2), 2);

        // Con 0 ops, debe dar 1 (mínimo)
        assert_eq!(hw.min_fragments(0), 1);
    }

    #[test]
    fn fragment_threshold() {
        let hw = HardwareSpec::new(4, 32 * 1024, 256 * 1024, 16 * 1024 * 1024);
        assert_eq!(hw.fragment_threshold(), 32 * 1024);
    }
}
