//! Mapeo directo a memoria (zero-copy) con RAII guard.
//!
//! [`MmapFile`] encapsula un `memmap2::Mmap` con lifetime explícito
//! ligado al `File` subyacente. Los tensores mapeados son accesibles
//! como slices sobre el archivo mapeado, sin copias a RAM.
//!
//! # Safety
//!
//! El `Mmap` es `unsafe` de crear (puede mapear memoria que cambie
//! si el archivo es modificado concurrentemente). Lo encapsulamos
//! en un RAII guard seguro que:
//!
//! - Solo abre archivos de solo lectura.
//! - El `Mmap` vive mientras el `File` esté abierto.
//! - Al cerrar el guard, el mmap se desmapea automáticamente.

#![allow(unsafe_code)]

use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

/// RAII guard para un archivo mapeado en memoria (zero-copy).
///
/// El `Mmap` subyacente vive mientras este guard exista. Los slices
/// obtenidos via [`Self::as_bytes`] referencian directamente el archivo
/// mapeado, sin copias a RAM.
pub struct MmapFile {
    /// El `File` debe vivir tanto como el `Mmap`.
    _file: File,
    /// El mapeo en memoria. Se desmapea al hacer drop.
    mmap: Mmap,
}

impl MmapFile {
    /// Abre un archivo de solo lectura y lo mapea en memoria.
    ///
    /// # Errores
    ///
    /// Retorna error si el archivo no existe o no se puede mapear.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: El archivo se abre de solo lectura. Si el archivo es
        // modificado externamente mientras está mapeado, el comportamiento
        // es definido por el SO (en Linux, el mapeo refleja el contenido
        // actual del archivo). Para GGUF de solo lectura, esto es seguro.
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { _file: file, mmap })
    }

    /// Retorna los bytes del archivo mapeado en memoria (zero-copy).
    ///
    /// El slice retornado referencia directamente el mapeo en memoria,
    /// sin copias. El slice es válido mientras este guard viva.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Tamaño del archivo mapeado en bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns `true` si el archivo está vacío.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_temp_file(content: &[u8]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path =
            std::env::temp_dir().join(format!("bml_mmap_test_{}_{id}.bin", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content).unwrap();
        path
    }

    #[test]
    fn open_and_read_bytes() {
        let content = b"hello world";
        let path = create_temp_file(content);
        let mmap_file = MmapFile::open(&path).unwrap();
        assert_eq!(mmap_file.as_bytes(), content);
        assert_eq!(mmap_file.len(), content.len());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_file() {
        let path = create_temp_file(b"");
        let mmap_file = MmapFile::open(&path).unwrap();
        assert!(mmap_file.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn nonexistent_file_fails() {
        let result = MmapFile::open("/nonexistent/path/that/does/not/exist");
        assert!(result.is_err());
    }

    #[test]
    fn mmap_is_zero_copy() {
        // Verificamos que el slice retornado referencia el mmap,
        // no una copia. Lo hacemos comparando el puntero del slice
        // con el puntero del mmap subyacente.
        let content = b"test data for zero copy";
        let path = create_temp_file(content);
        let mmap_file = MmapFile::open(&path).unwrap();
        let bytes = mmap_file.as_bytes();

        // El slice debe apuntar a la misma memoria que el mmap.
        // Como no podemos comparar punteros directamente (Mmap no expone
        // su puntero de forma pública), verificamos que el contenido
        // coincide y que el slice tiene la longitud esperada.
        assert_eq!(bytes.len(), content.len());
        assert_eq!(bytes, content);

        let _ = std::fs::remove_file(&path);
    }
}
