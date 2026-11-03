//! Streaming JSON encoding in the field order specified by BMLGGUF v1.
use crate::*;
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Serialize, Serializer,
};
use std::io::Write;

struct Wire<'a>(&'a Metadata);
struct Tool<'a>(&'a ToolIdentity);
impl Serialize for Tool<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(3))?;
        m.serialize_entry("name", &self.0.name)?;
        m.serialize_entry("version", &self.0.version)?;
        m.serialize_entry("revision", &self.0.revision)?;
        m.end()
    }
}
struct Limits<'a>(&'a ConversionLimits);
impl Serialize for Limits<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(9))?;
        for (key, value) in [
            ("max_weight_bits", self.0.max_weight_bits),
            ("max_weight_depth", self.0.max_weight_depth),
            (
                "max_synthesis_steps_per_weight",
                self.0.max_synthesis_steps_per_weight,
            ),
            (
                "max_synthesis_steps_total",
                self.0.max_synthesis_steps_total,
            ),
            ("max_block_terms", self.0.max_block_terms),
            ("max_block_bytes", self.0.max_block_bytes),
            ("max_output_bytes", self.0.max_output_bytes),
            ("max_resident_bytes", self.0.max_resident_bytes),
            ("max_cache_bytes", self.0.max_cache_bytes),
        ] {
            m.serialize_entry(key, &value.to_string())?;
        }
        m.end()
    }
}
struct Entries<'a>(&'a [GgufMetadataEntry]);
struct Entry<'a>(&'a GgufMetadataEntry);
struct Value<'a>(&'a MetadataValue);
fn code(value: &MetadataValue) -> u8 {
    match value {
        MetadataValue::U8(_) => 0,
        MetadataValue::I8(_) => 1,
        MetadataValue::U16(_) => 2,
        MetadataValue::I16(_) => 3,
        MetadataValue::U32(_) => 4,
        MetadataValue::I32(_) => 5,
        MetadataValue::F32Bits(_) => 6,
        MetadataValue::Bool(_) => 7,
        MetadataValue::String(_) => 8,
        MetadataValue::Array(_) => 9,
        MetadataValue::U64(_) => 10,
        MetadataValue::I64(_) => 11,
        MetadataValue::F64Bits(_) => 12,
    }
}
impl Serialize for Entries<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for entry in self.0 {
            seq.serialize_element(&Entry(entry))?;
        }
        seq.end()
    }
}
impl Serialize for Entry<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(3))?;
        m.serialize_entry("key", &self.0.key)?;
        m.serialize_entry("type", &code(&self.0.value))?;
        m.serialize_entry("value", &Value(&self.0.value))?;
        m.end()
    }
}
struct Array<'a>(&'a MetadataArray);
struct ArrayValues<'a>(&'a MetadataArray);
impl Serialize for Array<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let code = match self.0 {
            MetadataArray::U8(_) => 0,
            MetadataArray::I8(_) => 1,
            MetadataArray::U16(_) => 2,
            MetadataArray::I16(_) => 3,
            MetadataArray::U32(_) => 4,
            MetadataArray::I32(_) => 5,
            MetadataArray::F32Bits(_) => 6,
            MetadataArray::Bool(_) => 7,
            MetadataArray::String(_) => 8,
            MetadataArray::U64(_) => 10,
            MetadataArray::I64(_) => 11,
            MetadataArray::F64Bits(_) => 12,
        };
        let mut m = s.serialize_map(Some(2))?;
        m.serialize_entry("element_type", &code)?;
        m.serialize_entry("values", &ArrayValues(self.0))?;
        m.end()
    }
}
impl Serialize for ArrayValues<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        macro_rules! sequence {
            ($v:expr,$convert:expr) => {{
                let mut seq = s.serialize_seq(Some($v.len()))?;
                for value in $v {
                    seq.serialize_element(&$convert(value))?;
                }
                seq.end()
            }};
        }
        match self.0 {
            MetadataArray::U8(v) => sequence!(v, |x: &u8| x.to_string()),
            MetadataArray::I8(v) => sequence!(v, |x: &i8| x.to_string()),
            MetadataArray::U16(v) => sequence!(v, |x: &u16| x.to_string()),
            MetadataArray::I16(v) => sequence!(v, |x: &i16| x.to_string()),
            MetadataArray::U32(v) => sequence!(v, |x: &u32| x.to_string()),
            MetadataArray::I32(v) => sequence!(v, |x: &i32| x.to_string()),
            MetadataArray::U64(v) => sequence!(v, |x: &u64| x.to_string()),
            MetadataArray::I64(v) => sequence!(v, |x: &i64| x.to_string()),
            MetadataArray::F32Bits(v) => sequence!(v, |x: &u32| format!("{x:08x}")),
            MetadataArray::F64Bits(v) => sequence!(v, |x: &u64| format!("{x:016x}")),
            MetadataArray::Bool(v) => v.serialize(s),
            MetadataArray::String(v) => v.serialize(s),
        }
    }
}
impl Serialize for Value<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            MetadataValue::U8(v) => v.to_string().serialize(s),
            MetadataValue::I8(v) => v.to_string().serialize(s),
            MetadataValue::U16(v) => v.to_string().serialize(s),
            MetadataValue::I16(v) => v.to_string().serialize(s),
            MetadataValue::U32(v) => v.to_string().serialize(s),
            MetadataValue::I32(v) => v.to_string().serialize(s),
            MetadataValue::U64(v) => v.to_string().serialize(s),
            MetadataValue::I64(v) => v.to_string().serialize(s),
            MetadataValue::F32Bits(v) => format!("{v:08x}").serialize(s),
            MetadataValue::F64Bits(v) => format!("{v:016x}").serialize(s),
            MetadataValue::Bool(v) => v.serialize(s),
            MetadataValue::String(v) => v.serialize(s),
            MetadataValue::Array(v) => Array(v).serialize(s),
        }
    }
}
impl Serialize for Wire<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(7))?;
        m.serialize_entry("schema", &self.0.schema)?;
        m.serialize_entry("profile", &self.0.profile)?;
        m.serialize_entry("source_gguf_version", &self.0.source_gguf_version)?;
        m.serialize_entry("converter", &Tool(&self.0.converter))?;
        m.serialize_entry("codec", &Tool(&self.0.codec))?;
        m.serialize_entry("limits", &Limits(&self.0.limits))?;
        m.serialize_entry("gguf", &Entries(&self.0.gguf))?;
        m.end()
    }
}
struct Bounded {
    bytes: Vec<u8>,
    cap: u64,
    error: Option<Error>,
}
impl Write for Bounded {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let needed = (self.bytes.len() as u64).checked_add(bytes.len() as u64);
        if let Err(e) = check_limit(Resource::MaxMetadataBytes, self.cap, needed) {
            self.error = Some(e);
            return Err(std::io::Error::other("metadata limit"));
        }
        self.bytes
            .try_reserve(bytes.len())
            .map_err(|_| std::io::Error::other("metadata allocation"))?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
pub(crate) fn encode(metadata: &Metadata, limits: &ReadLimits, cap: u64) -> Result<Vec<u8>, Error> {
    let mut out = Bounded {
        bytes: Vec::new(),
        cap: cap.min(limits.max_metadata_bytes),
        error: None,
    };
    serde_json::to_writer(&mut out, &Wire(metadata)).map_err(|_| {
        out.error.take().unwrap_or(Error::Metadata {
            key: None,
            reason: MetadataReason::InvalidValue,
        })
    })?;
    // The decoder is the single source of strict metadata constraints, including
    // sorted unique keys and source types. No temporary file exists at this point.
    crate::metadata::decode(&out.bytes, limits)?;
    Ok(out.bytes)
}
