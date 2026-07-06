//! Decodificación del formato GGUF (GPT-Generated Unified Format).
//!
//! Estructura del archivo:
//!
//! ```text
//! [magic: u32 = 0x46554747]
//! [version: u32]
//! [tensor_count: u64]
//! [metadata_kv_count: u64]
//! [metadata_kv: ...]       // pares clave-valor con tipos
//! [tensor_infos: ...]      // nombre, dims, tipo, offset por tensor
//! [tensor_data: ...]       // datos binarios de los tensores
//! ```
//!
//! # Zero-Copy
//!
//! Los datos de los tensores se referencian directamente desde el mmap.
//! No hay copias a RAM.

use crate::MmapFile;
use std::collections::HashMap;
use std::io;

/// Magic number de GGUF: `0x46554747`.
pub const GGUF_MAGIC: u32 = 0x46554747;
pub const GGUF_SUPPORTED_VERSION: u32 = 3;

/// Tipos de datos de GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufDataType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
    // Cuantización GGUF
    Q4_0 = 14,
    Q4_1 = 15,
    Q5_0 = 16,
    Q5_1 = 17,
    Q8_0 = 18,
    Q8_1 = 19,
    Q2_K = 20,
    Q3_K = 21,
    Q4_K = 22,
    Q5_K = 23,
    Q6_K = 24,
    Q8_K = 25,
    Iq2Xxs = 28,
    Iq3Xxs = 29,
    Iq4Nl = 30,
    Iq3S = 31,
    Iq2S = 32,
    Iq4Xs = 33,
}

impl GgufDataType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::U16),
            3 => Some(Self::I16),
            4 => Some(Self::U32),
            5 => Some(Self::I32),
            6 => Some(Self::F32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::U64),
            11 => Some(Self::I64),
            12 => Some(Self::F64),
            14 => Some(Self::Q4_0),
            15 => Some(Self::Q4_1),
            16 => Some(Self::Q5_0),
            17 => Some(Self::Q5_1),
            18 => Some(Self::Q8_0),
            19 => Some(Self::Q8_1),
            20 => Some(Self::Q2_K),
            21 => Some(Self::Q3_K),
            22 => Some(Self::Q4_K),
            23 => Some(Self::Q5_K),
            24 => Some(Self::Q6_K),
            25 => Some(Self::Q8_K),
            28 => Some(Self::Iq2Xxs),
            29 => Some(Self::Iq3Xxs),
            30 => Some(Self::Iq4Nl),
            31 => Some(Self::Iq3S),
            32 => Some(Self::Iq2S),
            33 => Some(Self::Iq4Xs),
            _ => None,
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String | Self::Array => 0,
            // Tipos cuantizados: tamaño depende del bloque, no por elemento.
            _ => 0,
        }
    }

    /// Returns true si el tipo es cuantizado (no estándar).
    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::U8
                | Self::I8
                | Self::U16
                | Self::I16
                | Self::U32
                | Self::I32
                | Self::F32
                | Self::Bool
                | Self::String
                | Self::Array
                | Self::U64
                | Self::I64
                | Self::F64
        )
    }
}

/// Valor de un metadato GGUF.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(GgufDataType, Vec<GgufMetadataValue>),
}

/// Cabecera GGUF.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

/// Información de un tensor.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub data_type: GgufDataType,
    pub offset: u64,
}

/// Parser de archivos GGUF con mapeo zero-copy.
///
/// Decodifica cabecera, metadatos KV, tensor infos, y provee acceso
/// zero-copy a los datos de los tensores via slices sobre el mmap.
pub struct GgufParser {
    mmap: MmapFile,
    header: GgufHeader,
    /// Metadatos decodificados (clave -> valor).
    metadata: HashMap<String, GgufMetadataValue>,
    /// Información de cada tensor.
    tensor_infos: Vec<GgufTensorInfo>,
    /// Offset en el archivo donde empiezan los datos de los tensores.
    tensor_data_offset: usize,
}

impl std::fmt::Debug for GgufParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufParser")
            .field("header", &self.header)
            .field("metadata_count", &self.metadata.len())
            .field("tensor_count", &self.tensor_infos.len())
            .field("len", &self.mmap.len())
            .finish()
    }
}

impl GgufParser {
    /// Abre un archivo GGUF y decodifica todo: cabecera, metadatos, tensor infos.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        let mmap = MmapFile::open(path)?;
        let bytes = mmap.as_bytes();
        if bytes.len() < 24 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "archivo demasiado pequeño",
            ));
        }
        let header = decode_header(bytes)?;
        if header.magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("magic inválido: 0x{:08X}", header.magic),
            ));
        }
        if header.version > GGUF_SUPPORTED_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("versión no soportada: {}", header.version),
            ));
        }

        // Decodificar metadatos KV
        let mut offset = 24usize;
        let mut metadata = HashMap::new();
        for _ in 0..header.metadata_kv_count {
            let (key, value, new_offset) = decode_metadata_kv(bytes, offset)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            metadata.insert(key, value);
            offset = new_offset;
        }

        // Decodificar tensor infos
        let mut tensor_infos = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            let (info, new_offset) = decode_tensor_info(bytes, offset)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            tensor_infos.push(info);
            offset = new_offset;
        }

        // Alinear a 32 bytes (GGUF alignment)
        let tensor_data_offset = align(offset, 32);

        Ok(Self {
            mmap,
            header,
            metadata,
            tensor_infos,
            tensor_data_offset,
        })
    }

    /// Retorna la cabecera.
    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Retorna los metadatos decodificados.
    pub fn metadata(&self) -> &HashMap<String, GgufMetadataValue> {
        &self.metadata
    }

    /// Retorna la información de los tensores.
    pub fn tensor_infos(&self) -> &[GgufTensorInfo] {
        &self.tensor_infos
    }

    /// Retorna los bytes del archivo mapeado (zero-copy).
    pub fn bytes(&self) -> &[u8] {
        self.mmap.as_bytes()
    }

    /// Tamaño del archivo en bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Detecta la arquitectura del modelo leyendo `general.architecture`.
    pub fn architecture(&self) -> Option<&str> {
        match self.metadata.get("general.architecture") {
            Some(GgufMetadataValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Obtiene un metadato por clave.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata.get(key)
    }

    /// Busca un tensor por nombre.
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensor_infos.iter().find(|t| t.name == name)
    }

    /// Retorna un slice zero-copy a los datos de un tensor.
    ///
    /// El slice referencia directamente el mmap, sin copias.
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> Option<&[u8]> {
        let data_start = self.tensor_data_offset + info.offset as usize;
        let bytes = self.bytes();
        if data_start > bytes.len() {
            return None;
        }
        let elem_size = info.data_type.element_size();
        if elem_size > 0 {
            // Tipo estándar: calcular tamaño exacto
            let total_elems: usize = info.dims.iter().map(|d| *d as usize).product();
            let data_end = data_start + total_elems * elem_size;
            if data_end > bytes.len() {
                return None;
            }
            Some(&bytes[data_start..data_end])
        } else if info.data_type.is_quantized() {
            // Tipo cuantizado: dar el slice hasta el siguiente tensor o fin del archivo
            // El tamaño real depende del bloque de cuantización, que varía por tipo.
            // Por ahora, damos el slice hasta el fin del archivo.
            // El compilador deberá calcular el tamaño exacto al dequantizar.
            Some(&bytes[data_start..])
        } else {
            None // tipo variable (String, Array)
        }
    }
}

/// Alinea un offset al múltiplo más cercano de `alignment`.
fn align(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) / alignment * alignment
}

/// Decodifica la cabecera GGUF.
fn decode_header(bytes: &[u8]) -> io::Result<GgufHeader> {
    Ok(GgufHeader {
        magic: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        version: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        tensor_count: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        metadata_kv_count: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
    })
}

/// Decodifica un string GGUF: [u64 len][bytes].
fn decode_string(bytes: &[u8], offset: usize) -> Result<(String, usize), String> {
    if offset + 8 > bytes.len() {
        return Err("offset fuera de rango leyendo string len".into());
    }
    let len = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize;
    let str_start = offset + 8;
    if str_start + len > bytes.len() {
        return Err("offset fuera de rango leyendo string data".into());
    }
    let s = String::from_utf8_lossy(&bytes[str_start..str_start + len]).to_string();
    Ok((s, str_start + len))
}

/// Decodifica un par clave-valor de metadatos.
fn decode_metadata_kv(
    bytes: &[u8],
    offset: usize,
) -> Result<(String, GgufMetadataValue, usize), String> {
    let (key, offset) = decode_string(bytes, offset)?;
    if offset + 4 > bytes.len() {
        return Err("offset fuera de rango leyendo value type".into());
    }
    let value_type = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let offset = offset + 4;
    let (value, offset) = decode_metadata_value(bytes, offset, value_type)?;
    Ok((key, value, offset))
}

/// Decodifica un valor de metadato según su tipo.
fn decode_metadata_value(
    bytes: &[u8],
    offset: usize,
    type_id: u32,
) -> Result<(GgufMetadataValue, usize), String> {
    let dt = GgufDataType::from_u32(type_id).ok_or(format!("tipo desconocido: {type_id}"))?;
    match dt {
        GgufDataType::U8 => {
            if offset >= bytes.len() {
                return Err("offset fuera de rango U8".into());
            }
            Ok((GgufMetadataValue::U8(bytes[offset]), offset + 1))
        }
        GgufDataType::I8 => {
            if offset >= bytes.len() {
                return Err("offset fuera de rango I8".into());
            }
            Ok((GgufMetadataValue::I8(bytes[offset] as i8), offset + 1))
        }
        GgufDataType::U16 => {
            if offset + 2 > bytes.len() {
                return Err("offset fuera de rango U16".into());
            }
            Ok((
                GgufMetadataValue::U16(u16::from_le_bytes(
                    bytes[offset..offset + 2].try_into().unwrap(),
                )),
                offset + 2,
            ))
        }
        GgufDataType::I16 => {
            if offset + 2 > bytes.len() {
                return Err("offset fuera de rango I16".into());
            }
            Ok((
                GgufMetadataValue::I16(i16::from_le_bytes(
                    bytes[offset..offset + 2].try_into().unwrap(),
                )),
                offset + 2,
            ))
        }
        GgufDataType::U32 => {
            if offset + 4 > bytes.len() {
                return Err("offset fuera de rango U32".into());
            }
            Ok((
                GgufMetadataValue::U32(u32::from_le_bytes(
                    bytes[offset..offset + 4].try_into().unwrap(),
                )),
                offset + 4,
            ))
        }
        GgufDataType::I32 => {
            if offset + 4 > bytes.len() {
                return Err("offset fuera de rango I32".into());
            }
            Ok((
                GgufMetadataValue::I32(i32::from_le_bytes(
                    bytes[offset..offset + 4].try_into().unwrap(),
                )),
                offset + 4,
            ))
        }
        GgufDataType::F32 => {
            if offset + 4 > bytes.len() {
                return Err("offset fuera de rango F32".into());
            }
            Ok((
                GgufMetadataValue::F32(f32::from_le_bytes(
                    bytes[offset..offset + 4].try_into().unwrap(),
                )),
                offset + 4,
            ))
        }
        GgufDataType::Bool => {
            if offset >= bytes.len() {
                return Err("offset fuera de rango Bool".into());
            }
            Ok((GgufMetadataValue::Bool(bytes[offset] != 0), offset + 1))
        }
        GgufDataType::String => {
            let (s, offset) = decode_string(bytes, offset)?;
            Ok((GgufMetadataValue::String(s), offset))
        }
        GgufDataType::U64 => {
            if offset + 8 > bytes.len() {
                return Err("offset fuera de rango U64".into());
            }
            Ok((
                GgufMetadataValue::U64(u64::from_le_bytes(
                    bytes[offset..offset + 8].try_into().unwrap(),
                )),
                offset + 8,
            ))
        }
        GgufDataType::I64 => {
            if offset + 8 > bytes.len() {
                return Err("offset fuera de rango I64".into());
            }
            Ok((
                GgufMetadataValue::I64(i64::from_le_bytes(
                    bytes[offset..offset + 8].try_into().unwrap(),
                )),
                offset + 8,
            ))
        }
        GgufDataType::F64 => {
            if offset + 8 > bytes.len() {
                return Err("offset fuera de rango F64".into());
            }
            Ok((
                GgufMetadataValue::F64(f64::from_le_bytes(
                    bytes[offset..offset + 8].try_into().unwrap(),
                )),
                offset + 8,
            ))
        }
        GgufDataType::Array => {
            // Array: [u32 elem_type][u64 count][values...]
            if offset + 12 > bytes.len() {
                return Err("offset fuera de rango Array header".into());
            }
            let elem_type = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
            let count =
                u64::from_le_bytes(bytes[offset + 4..offset + 12].try_into().unwrap()) as usize;
            let mut offset = offset + 12;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                let (v, new_offset) = decode_metadata_value(bytes, offset, elem_type)?;
                values.push(v);
                offset = new_offset;
            }
            let elem_dt = GgufDataType::from_u32(elem_type)
                .ok_or(format!("tipo array desconocido: {elem_type}"))?;
            Ok((GgufMetadataValue::Array(elem_dt, values), offset))
        }
        // Tipos cuantizados no aparecen en metadatos, solo en tensores.
        _ => Err(format!("tipo de metadato no soportado: {type_id}")),
    }
}

/// Decodifica la información de un tensor.
fn decode_tensor_info(bytes: &[u8], offset: usize) -> Result<(GgufTensorInfo, usize), String> {
    let (name, offset) = decode_string(bytes, offset)?;
    if offset + 4 > bytes.len() {
        return Err("offset fuera de rango leyendo n_dims".into());
    }
    let n_dims = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let offset = offset + 4;
    let mut dims = Vec::with_capacity(n_dims as usize);
    let mut offset = offset;
    for _ in 0..n_dims {
        if offset + 8 > bytes.len() {
            return Err("offset fuera de rango leyendo dim".into());
        }
        dims.push(u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().unwrap(),
        ));
        offset += 8;
    }
    if offset + 4 > bytes.len() {
        return Err("offset fuera de rango leyendo data_type".into());
    }
    let data_type_raw = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let data_type = GgufDataType::from_u32(data_type_raw)
        .ok_or(format!("tipo tensor desconocido: {data_type_raw}"))?;
    offset += 4;
    if offset + 8 > bytes.len() {
        return Err("offset fuera de rango leyendo tensor offset".into());
    }
    let tensor_offset = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
    offset += 8;
    Ok((
        GgufTensorInfo {
            name,
            n_dims,
            dims,
            data_type,
            offset: tensor_offset,
        },
        offset,
    ))
}

// ===========================================================================
// Tests
// ===========================================================================

/// Genera un GGUF sintético con metadatos y un tensor.
pub fn create_gguf_with_metadata() -> std::path::PathBuf {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("bml_meta_gguf_{}_{id}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).unwrap();

    // Cabecera
    f.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
    f.write_all(&3u32.to_le_bytes()).unwrap();
    f.write_all(&1u64.to_le_bytes()).unwrap(); // 1 tensor
    f.write_all(&2u64.to_le_bytes()).unwrap(); // 2 metadatos

    // Metadato 1: general.architecture = "llama" (string)
    let key = b"general.architecture";
    f.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key).unwrap();
    f.write_all(&8u32.to_le_bytes()).unwrap(); // type = String
    let val = b"llama";
    f.write_all(&(val.len() as u64).to_le_bytes()).unwrap();
    f.write_all(val).unwrap();

    // Metadato 2: llama.context_length = 2048 (u32)
    let key2 = b"llama.context_length";
    f.write_all(&(key2.len() as u64).to_le_bytes()).unwrap();
    f.write_all(key2).unwrap();
    f.write_all(&4u32.to_le_bytes()).unwrap(); // type = U32
    f.write_all(&2048u32.to_le_bytes()).unwrap();

    // Tensor info: "token_embd.weight", 2 dims [4, 2], F32, offset 0
    let name = b"token_embd.weight";
    f.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
    f.write_all(name).unwrap();
    f.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
    f.write_all(&4u64.to_le_bytes()).unwrap(); // dim 0
    f.write_all(&2u64.to_le_bytes()).unwrap(); // dim 1
    f.write_all(&6u32.to_le_bytes()).unwrap(); // F32
    f.write_all(&0u64.to_le_bytes()).unwrap(); // offset

    // Alinear a 32 bytes
    let pos = f.metadata().unwrap().len() as usize;
    let aligned = align(pos, 32);
    let padding = aligned - pos;
    f.write_all(&vec![0u8; padding]).unwrap();

    // Tensor data: 4*2 = 8 floats = 32 bytes
    for i in 0..8 {
        f.write_all(&(i as f32).to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_with_metadata() {
        let path = create_gguf_with_metadata();
        let parser = GgufParser::open(&path).unwrap();

        assert_eq!(parser.header().tensor_count, 1);
        assert_eq!(parser.header().metadata_kv_count, 2);

        // Metadatos
        let arch = parser.architecture();
        assert_eq!(arch, Some("llama"));

        let ctx_len = parser.get_metadata("llama.context_length");
        match ctx_len {
            Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, 2048),
            _ => panic!("context_length no encontrado o tipo incorrecto"),
        }

        // Tensor info
        let infos = parser.tensor_infos();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].name, "token_embd.weight");
        assert_eq!(infos[0].n_dims, 2);
        assert_eq!(infos[0].dims, vec![4, 2]);
        assert_eq!(infos[0].data_type, GgufDataType::F32);

        // Tensor data (zero-copy)
        let info = &infos[0];
        let data = parser.tensor_data(info).unwrap();
        assert_eq!(data.len(), 32); // 8 * 4 bytes
        let first = f32::from_le_bytes(data[0..4].try_into().unwrap());
        assert_eq!(first, 0.0);
        let last = f32::from_le_bytes(data[28..32].try_into().unwrap());
        assert_eq!(last, 7.0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn find_tensor_by_name() {
        let path = create_gguf_with_metadata();
        let parser = GgufParser::open(&path).unwrap();
        let info = parser.find_tensor("token_embd.weight");
        assert!(info.is_some());
        assert_eq!(info.unwrap().dims, vec![4, 2]);

        let missing = parser.find_tensor("nonexistent");
        assert!(missing.is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn architecture_detection() {
        let path = create_gguf_with_metadata();
        let parser = GgufParser::open(&path).unwrap();
        assert_eq!(parser.architecture(), Some("llama"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tensor_data_zero_copy() {
        let path = create_gguf_with_metadata();
        let parser = GgufParser::open(&path).unwrap();
        let info = parser.find_tensor("token_embd.weight").unwrap();
        let data = parser.tensor_data(info).unwrap();
        // Verificar que podemos leer todos los elementos
        for i in 0..8 {
            let val = f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(val, i as f32);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_real_tinyllama() {
        // Intentar abrir el tinyllama real si está disponible
        let path = "/root/tinyllama.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: {path} no disponible");
            return;
        }
        let parser = GgufParser::open(path).unwrap();
        let arch = parser.architecture();
        println!("Architecture: {arch:?}");
        assert!(arch.is_some(), "debe detectar arquitectura");
        let infos = parser.tensor_infos();
        assert!(infos.len() > 0, "debe tener tensores");
        println!("Tensor count: {}", infos.len());
        // Verificar que podemos acceder a los datos del primer tensor
        let first = &infos[0];
        let data = parser.tensor_data(first);
        assert!(
            data.is_some(),
            "debe poder acceder a datos del primer tensor"
        );
        println!(
            "First tensor: {} ({} bytes)",
            first.name,
            data.unwrap().len()
        );
    }

    #[test]
    fn data_type_from_u32() {
        assert_eq!(GgufDataType::from_u32(0), Some(GgufDataType::U8));
        assert_eq!(GgufDataType::from_u32(6), Some(GgufDataType::F32));
        assert_eq!(GgufDataType::from_u32(99), None);
    }

    #[test]
    fn reject_invalid_magic() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!(
            "bml_bad_{:?}.bin",
            std::time::SystemTime::now().elapsed().unwrap().as_nanos()
        ));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.flush().unwrap();
        assert!(GgufParser::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
