//! Compatibility imports for the reader extracted to `bml-gguf`.

pub use bml_gguf::gguf::{
    create_gguf_with_metadata, GgufDataType, GgufHeader, GgufMetadataValue, GgufTensorInfo,
    GGUF_MAGIC, GGUF_SUPPORTED_VERSION,
};

/// Legacy reader adapter, preserving `open`'s `std::io::Result` signature.
#[derive(Debug)]
pub struct GgufParser(bml_gguf::GgufReader);

impl GgufParser {
    /// Open a GGUF file. Prefer `bml_gguf::GgufReader` for typed errors.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<Self> {
        bml_gguf::GgufReader::open(path)
            .map(Self)
            .map_err(Into::into)
    }

    /// Legacy header accessor.
    pub fn header(&self) -> &GgufHeader {
        self.0.header()
    }
    /// Legacy metadata accessor.
    pub fn metadata(&self) -> &std::collections::HashMap<String, GgufMetadataValue> {
        self.0.metadata()
    }
    /// Legacy descriptor accessor.
    pub fn tensor_infos(&self) -> &[GgufTensorInfo] {
        self.0.tensor_infos()
    }
    /// Bytes borrowed from the reader's mapping.
    pub fn bytes(&self) -> &[u8] {
        self.0.bytes()
    }
    /// Mapped file length.
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Whether the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Legacy model architecture metadata accessor.
    pub fn architecture(&self) -> Option<&str> {
        self.0.architecture()
    }
    /// Legacy metadata lookup.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.0.get_metadata(key)
    }
    /// Legacy tensor descriptor lookup.
    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.0.find_tensor(name)
    }
    /// Legacy optional byte range accessor.
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> Option<&[u8]> {
        self.0.tensor_data(info)
    }
}

impl std::ops::Deref for GgufParser {
    type Target = bml_gguf::GgufReader;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
