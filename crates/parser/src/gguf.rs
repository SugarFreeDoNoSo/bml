//! Decodificación del formato GGUF (GPT-Generated Unified Format).
//!
//! GGUF es un formato binario para almacenar modelos de ML (usado por
//! llama.cpp y otros). La estructura es:
//!
//! ```text
//! [magic: u32 = 0x46554747]  // "GGUF"
//! [version: u32]
//! [tensor_count: u64]
//! [metadata_kv_count: u64]
//! [metadata_kv: ...]
//! [tensor_infos: ...]
//! [tensor_data: ...]
//! ```
//!
//! # Referencia
//!
//! - https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md
//! - https://github.com/google/flatbuffers/blob/master/docs/source/WhitePaper.md

use crate::MmapFile;
use std::io;

/// Magic number de GGUF: `0x46554747` ("GGUF" en little-endian).
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Versión de GGUF soportada por este parser.
pub const GGUF_SUPPORTED_VERSION: u32 = 3;

/// Tipos de datos de GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufDataType {
    /// `u8`
    U8 = 0,
    /// `i8`
    I8 = 1,
    /// `u16`
    U16 = 2,
    /// `i16`
    I16 = 3,
    /// `u32`
    U32 = 4,
    /// `i32`
    I32 = 5,
    /// `f32`
    F32 = 6,
    /// `bool`
    Bool = 7,
    /// `String`
    String = 8,
    /// `Array`
    Array = 9,
    /// `u64`
    U64 = 10,
    /// `i64`
    I64 = 11,
    /// `f64`
    F64 = 12,
}

impl GgufDataType {
    /// Tamaño en bytes de un elemento de este tipo.
    pub fn element_size(&self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String | Self::Array => 0, // variable
        }
    }
}

/// Valor de un metadato GGUF.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    /// Entero sin signo de 8 bits.
    U8(u8),
    /// Entero con signo de 8 bits.
    I8(i8),
    /// Entero sin signo de 16 bits.
    U16(u16),
    /// Entero con signo de 16 bits.
    I16(i16),
    /// Entero sin signo de 32 bits.
    U32(u32),
    /// Entero con signo de 32 bits.
    I32(i32),
    /// Punto flotante de 32 bits.
    F32(f32),
    /// Booleano.
    Bool(bool),
    /// Cadena UTF-8.
    String(String),
    /// Arreglo de valores (con tipo de elemento).
    Array(Vec<GgufMetadataValue>),
    /// Entero sin signo de 64 bits.
    U64(u64),
    /// Entero con signo de 64 bits.
    I64(i64),
    /// Punto flotante de 64 bits.
    F64(f64),
}

/// Cabecera GGUF decodificada.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Magic number (debe ser `0x46554747`).
    pub magic: u32,
    /// Versión del formato.
    pub version: u32,
    /// Número de tensores en el archivo.
    pub tensor_count: u64,
    /// Número de pares clave-valor de metadatos.
    pub metadata_kv_count: u64,
}

/// Información de un tensor en el archivo GGUF.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Nombre del tensor.
    pub name: String,
    /// Número de dimensiones.
    pub n_dims: u32,
    /// Dimensiones del tensor.
    pub dims: Vec<u64>,
    /// Tipo de dato del tensor.
    pub data_type: GgufDataType,
    /// Offset del tensor desde el inicio de los datos de tensores.
    pub offset: u64,
}

/// Parser de archivos GGUF con mapeo zero-copy.
///
/// Encapsula un [`MmapFile`] y decodifica la cabecera, metadatos y
/// referencias a tensores. Los datos de los tensores se referencian
/// directamente desde el mmap, sin copias.
pub struct GgufParser {
    mmap: MmapFile,
    header: GgufHeader,
}

impl std::fmt::Debug for GgufParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufParser")
            .field("header", &self.header)
            .field("len", &self.mmap.len())
            .finish()
    }
}

impl GgufParser {
    /// Abre un archivo GGUF y decodifica su cabecera.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        let mmap = MmapFile::open(path)?;
        let bytes = mmap.as_bytes();
        if bytes.len() < 20 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "archivo demasiado pequeño para ser GGUF",
            ));
        }
        let header = decode_header(bytes)?;
        if header.magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "magic inválido: 0x{:08X} (esperado 0x{:08X})",
                    header.magic, GGUF_MAGIC
                ),
            ));
        }
        if header.version > GGUF_SUPPORTED_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "versión no soportada: {} (máximo {})",
                    header.version, GGUF_SUPPORTED_VERSION
                ),
            ));
        }
        Ok(Self { mmap, header })
    }

    /// Retorna la cabecera decodificada.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Retorna los bytes del archivo mapeado (zero-copy).
    pub fn bytes(&self) -> &[u8] {
        self.mmap.as_bytes()
    }

    /// Tamaño del archivo en bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }
}

/// Decodifica la cabecera GGUF desde los primeros 20 bytes.
fn decode_header(bytes: &[u8]) -> io::Result<GgufHeader> {
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let metadata_kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    Ok(GgufHeader {
        magic,
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// Genera un archivo GGUF sintético mínimo para tests.
#[cfg(test)]
pub fn create_minimal_gguf() -> std::path::PathBuf {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("bml_test_gguf_{}_{id}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();
    // Cabecera: magic, version, tensor_count, metadata_kv_count
    f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap(); // version 3
    f.write_all(&0u64.to_le_bytes()).unwrap(); // 0 tensores
    f.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadatos
    f.flush().unwrap();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_valid_gguf() {
        let path = create_minimal_gguf();
        let parser = GgufParser::open(&path).unwrap();
        assert_eq!(parser.header().magic, GGUF_MAGIC);
        assert_eq!(parser.header().version, 3);
        assert_eq!(parser.header().tensor_count, 0);
        assert_eq!(parser.header().metadata_kv_count, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reject_invalid_magic() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "bml_bad_magic_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.flush().unwrap();

        let result = GgufParser::open(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("magic"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reject_too_small() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "bml_small_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"too small").unwrap();
        f.flush().unwrap();

        let result = GgufParser::open(&path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reject_unsupported_version() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "bml_bad_ver_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        f.write_all(&99u32.to_le_bytes()).unwrap(); // version 99
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.flush().unwrap();

        let result = GgufParser::open(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("versión"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn data_type_element_sizes() {
        assert_eq!(GgufDataType::U8.element_size(), 1);
        assert_eq!(GgufDataType::F32.element_size(), 4);
        assert_eq!(GgufDataType::F64.element_size(), 8);
        assert_eq!(GgufDataType::String.element_size(), 0);
    }
}
