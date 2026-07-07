//! Buffer circular pre-asignado para pasar resultados entre hot loops.
//!
//! Cada hot loop escribe su output a un slot del buffer, y el siguiente
//! hot loop lo lee. Cero allocs, cero copias.

/// Buffer circular de resultados entre hot loops.
///
/// Tiene `n_slots` slots, cada uno de `slot_size` elementos `f64`.
/// Todos pre-asignados al crear el buffer.
#[derive(Debug)]
pub struct ResultBuffer {
    /// Datos planos: slot[i] está en data[i*slot_size..(i+1)*slot_size].
    data: Vec<f64>,
    /// Tamaño de cada slot.
    slot_size: usize,
    /// Número de slots.
    n_slots: usize,
}

impl ResultBuffer {
    /// Crea un buffer con `n_slots` slots de `slot_size` elementos cada uno.
    pub fn new(n_slots: usize, slot_size: usize) -> Self {
        let total = n_slots * slot_size;
        Self {
            data: vec![0.0; total],
            slot_size,
            n_slots,
        }
    }

    /// Escribe un valor al slot en el offset dado.
    #[inline]
    pub fn write(&mut self, slot: u32, offset: u32, value: f64) {
        let idx = slot as usize * self.slot_size + offset as usize;
        if idx < self.data.len() {
            self.data[idx] = value;
        }
    }

    /// Lee un valor del slot en el offset dado.
    #[inline]
    pub fn read(&self, slot: u32, offset: u32) -> f64 {
        let idx = slot as usize * self.slot_size + offset as usize;
        if idx < self.data.len() {
            self.data[idx]
        } else {
            f64::NAN
        }
    }

    /// Lee un valor por índice absoluto (base + offset).
    #[inline]
    pub fn read_indexed(&self, base: u32, offset: u32) -> f64 {
        let idx = (base + offset) as usize;
        if idx < self.data.len() {
            self.data[idx]
        } else {
            f64::NAN
        }
    }

    /// Escribe un valor por índice absoluto (base + offset).
    #[inline]
    pub fn write_indexed(&mut self, base: u32, offset: u32, value: f64) {
        let idx = (base + offset) as usize;
        if idx < self.data.len() {
            self.data[idx] = value;
        }
    }

    /// Retorna un slice del slot dado.
    pub fn slot(&self, slot: usize) -> &[f64] {
        let start = slot * self.slot_size;
        let end = start + self.slot_size;
        &self.data[start..end]
    }

    /// Retorna un slice mutable del slot dado.
    pub fn slot_mut(&mut self, slot: usize) -> &mut [f64] {
        let start = slot * self.slot_size;
        let end = start + self.slot_size;
        &mut self.data[start..end]
    }

    /// Número de slots.
    pub fn n_slots(&self) -> usize {
        self.n_slots
    }

    /// Tamaño de cada slot.
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Total de elementos.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Acceso directo a todos los datos (para tests).
    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read_basic() {
        let mut buf = ResultBuffer::new(4, 2048);
        buf.write(0, 0, 1.5);
        buf.write(0, 1, 2.5);
        buf.write(1, 0, 3.5);

        assert_eq!(buf.read(0, 0), 1.5);
        assert_eq!(buf.read(0, 1), 2.5);
        assert_eq!(buf.read(1, 0), 3.5);
        assert_eq!(buf.read(1, 1), 0.0); // no escrito
    }

    #[test]
    fn write_read_indexed() {
        let mut buf = ResultBuffer::new(4, 2048);
        // base = slot 1 inicio = 2048
        let base = 2048;
        buf.write_indexed(base, 0, 10.0);
        buf.write_indexed(base, 1, 20.0);

        assert_eq!(buf.read_indexed(base, 0), 10.0);
        assert_eq!(buf.read_indexed(base, 1), 20.0);
        // Equivalente a slot 1
        assert_eq!(buf.read(1, 0), 10.0);
        assert_eq!(buf.read(1, 1), 20.0);
    }

    #[test]
    fn slot_access() {
        let mut buf = ResultBuffer::new(2, 4);
        let slot0 = buf.slot_mut(0);
        slot0[0] = 1.0;
        slot0[1] = 2.0;
        slot0[2] = 3.0;
        slot0[3] = 4.0;

        assert_eq!(buf.slot(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(buf.slot(1), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn out_of_bounds_returns_nan() {
        let buf = ResultBuffer::new(2, 4);
        assert!(buf.read(99, 0).is_nan());
        assert!(buf.read_indexed(99999, 0).is_nan());
    }

    #[test]
    fn zero_allocs() {
        // El buffer se pre-asigna al crear. Verificar que no crece.
        let mut buf = ResultBuffer::new(4, 2048);
        let len_before = buf.len();
        for i in 0..2048 {
            buf.write(0, i, i as f64);
        }
        assert_eq!(buf.len(), len_before);
    }

    #[test]
    fn pass_results_between_slots() {
        // Simular dos hot loops que se pasan resultados
        let mut buf = ResultBuffer::new(4, 8);

        // Hot loop 0 escribe al slot 0
        for i in 0..8 {
            buf.write(0, i, (i as f64) * 2.0);
        }

        // Hot loop 1 lee del slot 0, procesa, escribe al slot 1
        for i in 0..8 {
            let val = buf.read(0, i);
            buf.write(1, i, val + 1.0);
        }

        // Verificar
        for i in 0..8 {
            assert_eq!(buf.read(1, i), (i as f64) * 2.0 + 1.0);
        }
    }
}
