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
//! No se copia el archivo completo a un buffer de usuario; las páginas usan RAM.

use crate::{GgufError, MmapFile};
use std::collections::HashMap;

/// Magic number de GGUF: `0x46554747`.
pub const GGUF_MAGIC: u32 = 0x46554747;
/// Highest supported little-endian GGUF version (v2 remains compatible).
pub const GGUF_SUPPORTED_VERSION: u32 = 3;

/// Tipos de datos de GGUF/GGML.
///
/// Sigue el mapeo oficial de ggml.h:
/// 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 6=Q5_0, 7=Q5_1, 8=Q8_0, 9=Q8_1,
/// 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K, 15=Q8_K,
/// 24=I8, 25=I16, 26=I32, 27=I64, 28=F64, 29=IQ1_M, 30=BF16
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)] // Preserve the official GGML format names.
pub enum GgufDataType {
    /// F32.
    F32 = 0,
    /// F16.
    F16 = 1,
    /// Q4_0.
    Q4_0 = 2,
    /// Q4_1.
    Q4_1 = 3,
    /// Q5_0.
    Q5_0 = 6,
    /// Q5_1.
    Q5_1 = 7,
    /// Q8_0.
    Q8_0 = 8,
    /// Q8_1.
    Q8_1 = 9,
    /// Q2_K.
    Q2_K = 10,
    /// Q3_K.
    Q3_K = 11,
    /// Q4_K.
    Q4_K = 12,
    /// Q5_K.
    Q5_K = 13,
    /// Q6_K.
    Q6_K = 14,
    /// Q8_K.
    Q8_K = 15,
    /// Iq2Xxs.
    Iq2Xxs = 16,
    /// Iq2Xs.
    Iq2Xs = 17,
    /// Iq3Xxs.
    Iq3Xxs = 18,
    /// Iq1S.
    Iq1S = 19,
    /// Iq4Nl.
    Iq4Nl = 20,
    /// Iq3S.
    Iq3S = 21,
    /// Iq2S.
    Iq2S = 22,
    /// Iq4Xs.
    Iq4Xs = 23,
    /// I8.
    I8 = 24,
    /// I16.
    I16 = 25,
    /// I32.
    I32 = 26,
    /// I64.
    I64 = 27,
    /// F64.
    F64 = 28,
    /// IQ1_M quantized blocks (metadata inspection only).
    Iq1M = 29,
    /// Bfloat16 storage (metadata inspection only).
    Bf16 = 30,
    // Tipos solo para metadatos (no tensores)
    /// String.
    String = 256, // originalmente 8 en GGUF metadata
    /// Bool.
    Bool = 257, // originalmente 7
    /// Array.
    Array = 258, // originalmente 9
    /// U8.
    U8 = 259, // originalmente 0 en metadatos
    /// U16.
    U16 = 260, // originalmente 2 en metadatos
    /// U32.
    U32 = 261, // originalmente 4 en metadatos
    /// U64.
    U64 = 262, // originalmente 10 en metadatos
}

impl GgufDataType {
    /// Mapea un tipo de tensor GGML (estándar ggml.h).
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::Iq2Xxs),
            17 => Some(Self::Iq2Xs),
            18 => Some(Self::Iq3Xxs),
            19 => Some(Self::Iq1S),
            20 => Some(Self::Iq4Nl),
            21 => Some(Self::Iq3S),
            22 => Some(Self::Iq2S),
            23 => Some(Self::Iq4Xs),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::Iq1M),
            30 => Some(Self::Bf16),
            _ => None,
        }
    }

    /// Mapea un tipo de metadato GGUF (valores 0-12 originales).
    pub fn from_metadata_u32(v: u32) -> Option<Self> {
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
            _ => None,
        }
    }

    /// Tamaño en bytes de un elemento de este tipo.
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::I16 | Self::U16 | Self::Bf16 => 2,
            Self::I8 | Self::U8 => 1,
            Self::I64 | Self::U64 | Self::F64 => 8,
            // Quantizers and metadata-only values have no scalar byte size.
            _ => 0,
        }
    }

    /// Returns true si el tipo es cuantizado (no estándar).
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_0
                | Self::Q8_1
                | Self::Q2_K
                | Self::Q3_K
                | Self::Q4_K
                | Self::Q5_K
                | Self::Q6_K
                | Self::Q8_K
                | Self::Iq2Xxs
                | Self::Iq2Xs
                | Self::Iq3Xxs
                | Self::Iq1S
                | Self::Iq4Nl
                | Self::Iq3S
                | Self::Iq2S
                | Self::Iq4Xs
                | Self::Iq1M
                | Self::F16
                | Self::Bf16
        )
    }
}

/// Valor de un metadato GGUF.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    /// U8.
    U8(u8),
    /// I8.
    I8(i8),
    /// U16.
    U16(u16),
    /// I16.
    I16(i16),
    /// U32.
    U32(u32),
    /// I32.
    I32(i32),
    /// F32.
    F32(f32),
    /// Bool.
    Bool(bool),
    /// String.
    String(String),
    /// U64.
    U64(u64),
    /// I64.
    I64(i64),
    /// F64.
    F64(f64),
    /// Array.
    Array(GgufDataType, Vec<GgufMetadataValue>),
}

/// Cabecera GGUF.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Magic value decoded in little-endian order.
    pub magic: u32,
    /// GGUF format version.
    pub version: u32,
    /// Number of tensor descriptors.
    pub tensor_count: u64,
    /// Number of metadata entries.
    pub metadata_kv_count: u64,
}

/// Información de un tensor.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// GGML dimensions, with the first dimension contiguous.
    pub dims: Vec<u64>,
    /// GGML storage dtype.
    pub data_type: GgufDataType,
    /// Byte offset relative to the aligned tensor data section.
    pub offset: u64,
}

/// Parser de archivos GGUF con mapeo zero-copy.
///
/// Decodifica cabecera, metadatos KV, tensor infos, y provee acceso
/// zero-copy a los datos de los tensores via slices sobre el mmap.
pub struct GgufReader {
    mmap: MmapFile,
    header: GgufHeader,
    /// Metadatos decodificados (clave -> valor).
    metadata: HashMap<String, GgufMetadataValue>,
    /// Información de cada tensor.
    tensor_infos: Vec<GgufTensorInfo>,
    /// Offset en el archivo donde empiezan los datos de los tensores.
    tensor_data_offset: usize,
}

impl std::fmt::Debug for GgufReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufReader")
            .field("header", &self.header)
            .field("metadata_count", &self.metadata.len())
            .field("tensor_count", &self.tensor_infos.len())
            .field("len", &self.mmap.len())
            .finish()
    }
}

impl GgufReader {
    /// Abre un archivo GGUF y decodifica todo: cabecera, metadatos, tensor infos.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, GgufError> {
        let mmap = MmapFile::open(path)?;
        let bytes = mmap.as_bytes();
        if bytes.len() < 24 {
            return Err(GgufError::Truncated {
                offset: 0,
                needed: 24,
                available: bytes.len(),
            });
        }
        let header = decode_header(bytes);
        if header.magic == GGUF_MAGIC && header.version.swap_bytes() == 3 {
            return Err(GgufError::UnsupportedEndianness);
        }
        if header.magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic(header.magic));
        }
        if !(2..=GGUF_SUPPORTED_VERSION).contains(&header.version) {
            return Err(GgufError::UnsupportedVersion(header.version));
        }

        // Decodificar metadatos KV
        let mut offset = 24usize;
        let mut metadata = HashMap::new();
        for _ in 0..header.metadata_kv_count {
            let (key, value, new_offset) = decode_metadata_kv(bytes, offset)
                .map_err(|reason| GgufError::InvalidMetadata { offset, reason })?;
            metadata.insert(key, value);
            offset = new_offset;
        }

        // Decodificar tensor infos
        // Decode incrementally: never allocate from untrusted counts.
        let mut tensor_infos = Vec::new();
        for _ in 0..header.tensor_count {
            let (info, new_offset) = decode_tensor_info(bytes, offset)
                .map_err(|reason| GgufError::InvalidTensorInfo { offset, reason })?;
            tensor_infos.push(info);
            offset = new_offset;
        }

        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(GgufMetadataValue::U32(n)) if *n != 0 && *n % 8 == 0 => *n as usize,
            _ => return Err(GgufError::InvalidAlignment),
        };
        let tensor_data_offset = offset
            .checked_add(alignment - 1)
            .map(|n| n / alignment * alignment)
            .ok_or(GgufError::SizeOverflow)?;
        let parser = Self {
            mmap,
            header,
            metadata,
            tensor_infos,
            tensor_data_offset,
        };
        for info in &parser.tensor_infos {
            if info.offset % alignment as u64 != 0 {
                return Err(GgufError::UnalignedTensor {
                    name: info.name.clone(),
                    offset: info.offset,
                    alignment,
                });
            }
            // Unknown quantizers can be inspected as metadata only.
            if supported_layout(info.data_type) && parser.tensor_data(info).is_none() {
                return Err(GgufError::InvalidTensorRange {
                    name: info.name.clone(),
                });
            }
        }
        Ok(parser)
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

    /// Whether the mapped file has no bytes (a valid GGUF is never empty).
    pub fn is_empty(&self) -> bool {
        // Construction rejects files shorter than the 24-byte header.
        false
    }

    /// Decode a named tensor into an owned f32 buffer, preserving GGML element order.
    /// Unsupported dtypes and missing tensors produce distinct typed errors.
    pub fn read_tensor_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let info = self
            .find_tensor(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_owned()))?;
        let data = self.tensor_data(info).ok_or_else(|| {
            if supported_layout(info.data_type) {
                GgufError::InvalidTensorRange {
                    name: name.to_owned(),
                }
            } else {
                GgufError::UnsupportedDataType(info.data_type)
            }
        })?;
        let elements = info.dims.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(usize::try_from(d).map_err(|_| GgufError::SizeOverflow)?)
                .ok_or(GgufError::SizeOverflow)
        })?;
        crate::decode_f32(info.data_type, data, elements)
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
    /// Soporta tipos escalares, Q4_0 y Q8_0; otros cuantizadores retornan None.
    /// Los bloques cuantizados deben estar completos en cada fila.
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> Option<&[u8]> {
        let data_start = self
            .tensor_data_offset
            .checked_add(usize::try_from(info.offset).ok()?)?;
        let data_end = data_start.checked_add(encoded_tensor_len(info)?)?;
        self.bytes().get(data_start..data_end)
    }
}

fn supported_layout(dtype: GgufDataType) -> bool {
    dtype.element_size() > 0 || matches!(dtype, GgufDataType::Q4_0 | GgufDataType::Q8_0)
}

fn encoded_tensor_len(info: &GgufTensorInfo) -> Option<usize> {
    if info.dims.is_empty() || info.dims.len() != info.n_dims as usize || info.dims.contains(&0) {
        return None;
    }
    let elements = info
        .dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(usize::try_from(d).ok()?))?;
    match info.data_type {
        GgufDataType::Q4_0 | GgufDataType::Q8_0 => {
            if !info.dims[0].is_multiple_of(32) {
                return None;
            }
            let block_bytes = if info.data_type == GgufDataType::Q4_0 {
                18
            } else {
                34
            };
            (elements / 32).checked_mul(block_bytes)
        }
        dtype if dtype.element_size() > 0 => elements.checked_mul(dtype.element_size()),
        _ => None,
    }
}

/// Alinea un offset al múltiplo más cercano de `alignment`.
fn align(offset: usize, alignment: usize) -> usize {
    offset.div_ceil(alignment) * alignment
}

/// Decodifica la cabecera GGUF.
fn decode_header(bytes: &[u8]) -> GgufHeader {
    GgufHeader {
        magic: u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        version: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        tensor_count: u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        metadata_kv_count: u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
    }
}

/// Decodifica un string GGUF: [u64 len][bytes].
fn decode_string(bytes: &[u8], offset: usize) -> Result<(String, usize), String> {
    if offset + 8 > bytes.len() {
        return Err("offset fuera de rango leyendo string len".into());
    }
    let len = usize::try_from(u64::from_le_bytes(
        bytes[offset..offset + 8].try_into().unwrap(),
    ))
    .map_err(|_| "string length overflow")?;
    let str_start = offset + 8;
    let end = str_start.checked_add(len).ok_or("string range overflow")?;
    let data = bytes.get(str_start..end).ok_or("string outside file")?;
    let s = std::str::from_utf8(data)
        .map_err(|_| "invalid UTF-8")?
        .to_owned();
    Ok((s, end))
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
    let dt =
        GgufDataType::from_metadata_u32(type_id).ok_or(format!("tipo desconocido: {type_id}"))?;
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
            match bytes[offset] {
                0 | 1 => Ok((GgufMetadataValue::Bool(bytes[offset] == 1), offset + 1)),
                _ => Err("invalid boolean value".into()),
            }
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
            let count = usize::try_from(u64::from_le_bytes(
                bytes[offset + 4..offset + 12].try_into().unwrap(),
            ))
            .map_err(|_| "array length overflow")?;
            let mut offset = offset + 12;
            if elem_type == 9 {
                return Err("nested arrays unsupported".into());
            }
            let mut values = Vec::new();
            for _ in 0..count {
                let (v, new_offset) = decode_metadata_value(bytes, offset, elem_type)?;
                values.push(v);
                offset = new_offset;
            }
            let elem_dt = GgufDataType::from_metadata_u32(elem_type)
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
    if !(1..=4).contains(&n_dims) {
        return Err("tensor rank must be 1..=4".into());
    }
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
    if dims.contains(&0)
        || dims
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .is_none()
    {
        return Err("invalid tensor dimensions".into());
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
    f.write_all(&0u32.to_le_bytes()).unwrap(); // F32 (GGML type 0)
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
        let parser = GgufReader::open(&path).unwrap();

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
        let parser = GgufReader::open(&path).unwrap();
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
        let parser = GgufReader::open(&path).unwrap();
        assert_eq!(parser.architecture(), Some("llama"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tensor_data_zero_copy() {
        let path = create_gguf_with_metadata();
        let parser = GgufReader::open(&path).unwrap();
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
    #[ignore = "requires BML_TEST_GGUF pointing to a TinyLlama GGUF"]
    fn open_real_tinyllama() {
        // Intentar abrir el tinyllama real si está disponible
        let model_path = std::env::var("BML_TEST_GGUF").expect("set BML_TEST_GGUF");
        let path = model_path.as_str();
        let parser = GgufReader::open(path).unwrap();
        let arch = parser.architecture();
        println!("Architecture: {arch:?}");
        assert!(arch.is_some(), "debe detectar arquitectura");
        let infos = parser.tensor_infos();
        assert!(!infos.is_empty(), "debe tener tensores");
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
        // Tipos de tensor (GGML)
        assert_eq!(GgufDataType::from_u32(0), Some(GgufDataType::F32));
        assert_eq!(GgufDataType::from_u32(2), Some(GgufDataType::Q4_0));
        assert_eq!(GgufDataType::from_u32(8), Some(GgufDataType::Q8_0));
        assert_eq!(GgufDataType::from_u32(99), None);
        // Tipos de metadatos
        assert_eq!(GgufDataType::from_metadata_u32(0), Some(GgufDataType::U8));
        assert_eq!(GgufDataType::from_metadata_u32(6), Some(GgufDataType::F32));
        assert_eq!(
            GgufDataType::from_metadata_u32(8),
            Some(GgufDataType::String)
        );
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
        assert!(GgufReader::open(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
