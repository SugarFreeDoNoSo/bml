//! Strict typed JSON decoding with bounded nesting, arrays and duplicate detection.
use crate::*;
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use std::fmt;

fn bad(reason: MetadataReason) -> Error {
    Error::Metadata { key: None, reason }
}
struct Context<'a> {
    limits: &'a ReadLimits,
    elements: u64,
    error: Option<Error>,
}
impl Context<'_> {
    fn fail<E: de::Error>(&mut self, error: Error) -> E {
        self.error = Some(error);
        E::custom("BMLGGUF metadata contract")
    }
}
struct Seed<'a, 'b> {
    ctx: &'a mut Context<'b>,
    depth: usize,
    entries: bool,
}
impl<'de> DeserializeSeed<'de> for Seed<'_, '_> {
    type Value = Value;
    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        d.deserialize_any(self)
    }
}
struct ElementSeed<'a, 'b> {
    seed: Seed<'a, 'b>,
    entry_index: Option<u64>,
}
impl<'de> DeserializeSeed<'de> for ElementSeed<'_, '_> {
    type Value = Value;
    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        let ctx = &mut *self.seed.ctx;
        let (resource, limit, required) = if let Some(index) = self.entry_index {
            (
                Resource::MaxMetadataEntries,
                ctx.limits.max_metadata_entries,
                index.checked_add(1),
            )
        } else {
            let next = ctx.elements.checked_add(1);
            (
                Resource::MaxMetadataArrayElements,
                ctx.limits.max_metadata_array_elements,
                next,
            )
        };
        if let Err(error) = check_limit(resource, limit, required) {
            return Err(ctx.fail(error));
        }
        if self.entry_index.is_none() {
            ctx.elements = required.unwrap();
        }
        self.seed.deserialize(d)
    }
}
impl<'de> Visitor<'de> for Seed<'_, '_> {
    type Value = Value;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("bounded metadata JSON")
    }
    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Value, E> {
        Ok(Value::Bool(value))
    }
    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Value, E> {
        Ok(value.into())
    }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<Value, E> {
        // Signed metadata integers are canonical strings. Raw JSON numbers
        // are only used for nonnegative schema/version/type tags.
        Err(self.ctx.fail(bad(MetadataReason::InvalidType)))
    }
    fn visit_f64<E: de::Error>(self, _: f64) -> Result<Value, E> {
        Err(self.ctx.fail(bad(MetadataReason::InvalidType)))
    }
    fn visit_str<E: de::Error>(self, value: &str) -> Result<Value, E> {
        if let Err(e) = check_limit(
            Resource::MaxStringBytes,
            self.ctx.limits.max_string_bytes,
            Some(value.len() as u64),
        ) {
            return Err(self.ctx.fail(e));
        }
        Ok(Value::String(value.into()))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        if self.depth > 8 {
            return Err(self.ctx.fail(bad(MetadataReason::InvalidValue)));
        }
        let mut values = Vec::new();
        loop {
            // SeqAccess calls the seed only when another element exists: enforce
            // its budget before parsing/materializing even that first excess value.
            let Some(value) = seq.next_element_seed(ElementSeed {
                seed: Seed {
                    ctx: self.ctx,
                    depth: self.depth + 1,
                    entries: false,
                },
                entry_index: self.entries.then_some(values.len() as u64),
            })?
            else {
                break;
            };
            values.push(value);
        }
        Ok(Value::Array(values))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        if self.depth > 8 {
            return Err(self.ctx.fail(bad(MetadataReason::InvalidValue)));
        }
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if let Err(e) = check_limit(
                Resource::MaxNameBytes,
                self.ctx.limits.max_name_bytes,
                Some(key.len() as u64),
            ) {
                return Err(self.ctx.fail(e));
            }
            if values.contains_key(&key) {
                return Err(self.ctx.fail(bad(MetadataReason::DuplicateKey)));
            }
            let value = map.next_value_seed(Seed {
                ctx: self.ctx,
                depth: self.depth + 1,
                entries: self.depth == 1 && key == "gguf",
            })?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

fn object(value: Value) -> Result<Map<String, Value>, Error> {
    match value {
        Value::Object(v) => Ok(v),
        _ => Err(bad(MetadataReason::InvalidType)),
    }
}
fn take(m: &mut Map<String, Value>, key: &str) -> Result<Value, Error> {
    m.remove(key).ok_or_else(|| Error::Metadata {
        key: Some(key.into()),
        reason: MetadataReason::MissingField,
    })
}
fn done(m: Map<String, Value>) -> Result<(), Error> {
    if m.is_empty() {
        Ok(())
    } else {
        Err(bad(MetadataReason::UnknownField))
    }
}
fn string(value: Value) -> Result<String, Error> {
    match value {
        Value::String(s) => Ok(s),
        _ => Err(bad(MetadataReason::InvalidType)),
    }
}
fn number(value: Value) -> Result<u64, Error> {
    value
        .as_u64()
        .ok_or_else(|| bad(MetadataReason::InvalidType))
}
fn array(value: Value) -> Result<Vec<Value>, Error> {
    match value {
        Value::Array(v) => Ok(v),
        _ => Err(bad(MetadataReason::InvalidType)),
    }
}
fn integer<T: std::str::FromStr + ToString>(value: Value) -> Result<T, Error> {
    let text = string(value)?;
    let parsed: T = text
        .parse()
        .map_err(|_| bad(MetadataReason::InvalidValue))?;
    if parsed.to_string() != text {
        return Err(bad(MetadataReason::InvalidValue));
    }
    Ok(parsed)
}
fn hex(value: Value, digits: usize) -> Result<u64, Error> {
    let text = string(value)?;
    if text.len() != digits
        || !text
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(bad(MetadataReason::InvalidValue));
    }
    u64::from_str_radix(&text, 16).map_err(|_| bad(MetadataReason::InvalidValue))
}
fn scalar(kind: u64, value: Value) -> Result<MetadataValue, Error> {
    // typed handles array tags (including nested-array diagnostics) before
    // dispatching here. Only scalar tags reach this decoder.
    Ok(match kind {
        0 => MetadataValue::U8(integer(value)?),
        1 => MetadataValue::I8(integer(value)?),
        2 => MetadataValue::U16(integer(value)?),
        3 => MetadataValue::I16(integer(value)?),
        4 => MetadataValue::U32(integer(value)?),
        5 => MetadataValue::I32(integer(value)?),
        6 => MetadataValue::F32Bits(hex(value, 8)? as u32),
        7 => MetadataValue::Bool(
            value
                .as_bool()
                .ok_or_else(|| bad(MetadataReason::InvalidType))?,
        ),
        8 => MetadataValue::String(string(value)?),
        10 => MetadataValue::U64(integer(value)?),
        11 => MetadataValue::I64(integer(value)?),
        12 => MetadataValue::F64Bits(hex(value, 16)?),
        _ => return Err(bad(MetadataReason::InvalidType)),
    })
}
fn typed(kind: u64, value: Value) -> Result<MetadataValue, Error> {
    if kind != 9 {
        return scalar(kind, value);
    }
    let mut obj = object(value)?;
    let kind = number(take(&mut obj, "element_type")?)?;
    let values = array(take(&mut obj, "values")?)?;
    done(obj)?;
    macro_rules! vector {
        ($variant:ident) => {{
            let mut out = Vec::new();
            for value in values {
                if let MetadataValue::$variant(value) = scalar(kind, value)? {
                    out.push(value);
                } else {
                    return Err(bad(MetadataReason::InvalidType));
                }
            }
            MetadataArray::$variant(out)
        }};
    }
    Ok(MetadataValue::Array(match kind {
        0 => vector!(U8),
        1 => vector!(I8),
        2 => vector!(U16),
        3 => vector!(I16),
        4 => vector!(U32),
        5 => vector!(I32),
        6 => vector!(F32Bits),
        7 => vector!(Bool),
        8 => vector!(String),
        10 => vector!(U64),
        11 => vector!(I64),
        12 => vector!(F64Bits),
        9 => return Err(bad(MetadataReason::NestedArray)),
        _ => return Err(bad(MetadataReason::InvalidType)),
    }))
}
fn tool(value: Value) -> Result<ToolIdentity, Error> {
    let mut m = object(value)?;
    let out = ToolIdentity {
        name: string(take(&mut m, "name")?)?,
        version: string(take(&mut m, "version")?)?,
        revision: string(take(&mut m, "revision")?)?,
    };
    done(m)?;
    if [&out.name, &out.version, &out.revision]
        .iter()
        .any(|s| s.is_empty())
    {
        return Err(bad(MetadataReason::InvalidValue));
    }
    Ok(out)
}
fn conversion_limits(value: Value) -> Result<ConversionLimits, Error> {
    let mut m = object(value)?;
    let out = ConversionLimits {
        max_weight_bits: integer(take(&mut m, "max_weight_bits")?)?,
        max_weight_depth: integer(take(&mut m, "max_weight_depth")?)?,
        max_synthesis_steps_per_weight: integer(take(&mut m, "max_synthesis_steps_per_weight")?)?,
        max_synthesis_steps_total: integer(take(&mut m, "max_synthesis_steps_total")?)?,
        max_block_terms: integer(take(&mut m, "max_block_terms")?)?,
        max_block_bytes: integer(take(&mut m, "max_block_bytes")?)?,
        max_output_bytes: integer(take(&mut m, "max_output_bytes")?)?,
        max_resident_bytes: integer(take(&mut m, "max_resident_bytes")?)?,
        max_cache_bytes: integer(take(&mut m, "max_cache_bytes")?)?,
    };
    done(m)?;
    out.validate()?;
    Ok(out)
}
pub(crate) fn decode(bytes: &[u8], limits: &ReadLimits) -> Result<Metadata, Error> {
    std::str::from_utf8(bytes).map_err(|_| bad(MetadataReason::InvalidUtf8))?;
    let mut ctx = Context {
        limits,
        elements: 0,
        error: None,
    };
    let mut d = serde_json::Deserializer::from_slice(bytes);
    let value = Seed {
        ctx: &mut ctx,
        depth: 1,
        entries: false,
    }
    .deserialize(&mut d)
    .map_err(|_| {
        ctx.error
            .take()
            .unwrap_or_else(|| bad(MetadataReason::InvalidJson))
    })?;
    d.end().map_err(|_| bad(MetadataReason::InvalidJson))?;
    let mut m = object(value)?;
    let schema = number(take(&mut m, "schema")?)?;
    let profile = string(take(&mut m, "profile")?)?;
    let version = number(take(&mut m, "source_gguf_version")?)?;
    if schema != 1 || profile != "bml-exact-decoded-f32-v1" {
        return Err(bad(MetadataReason::UnsupportedProfile));
    }
    if !matches!(version, 2 | 3) {
        return Err(Error::Unsupported {
            kind: UnsupportedKind::GgufVersion,
            value: version,
        });
    }
    let converter = tool(take(&mut m, "converter")?)?;
    let codec = tool(take(&mut m, "codec")?)?;
    let conversion = conversion_limits(take(&mut m, "limits")?)?;
    let mut gguf: Vec<GgufMetadataEntry> = Vec::new();
    for entry in array(take(&mut m, "gguf")?)? {
        let mut entry = object(entry)?;
        let key = string(take(&mut entry, "key")?)?;
        check_limit(
            Resource::MaxNameBytes,
            limits.max_name_bytes,
            Some(key.len() as u64),
        )?;
        if key.is_empty() || key.contains('\0') {
            return Err(bad(MetadataReason::InvalidValue));
        }
        if gguf.last().is_some_and(|p| p.key >= key) {
            return Err(bad(MetadataReason::DuplicateKey));
        }
        let kind = number(take(&mut entry, "type")?)?;
        let value = typed(kind, take(&mut entry, "value")?)?;
        done(entry)?;
        gguf.push(GgufMetadataEntry { key, value });
    }
    done(m)?;
    Ok(Metadata {
        schema: 1,
        profile,
        source_gguf_version: version as u32,
        converter,
        codec,
        limits: conversion,
        gguf,
    })
}

#[cfg(test)]
mod parser_diagnostics {
    use super::*;

    #[test]
    fn unsupported_null_reports_the_bounded_json_expectation() {
        let limits = ReadLimits::default();
        let mut ctx = Context {
            limits: &limits,
            elements: 0,
            error: None,
        };
        let mut decoder = serde_json::Deserializer::from_slice(b"null");
        let error = Seed {
            ctx: &mut ctx,
            depth: 1,
            entries: false,
        }
        .deserialize(&mut decoder)
        .unwrap_err();
        assert!(error.to_string().contains("expected bounded metadata JSON"));
        assert!(matches!(
            decode(b"null", &limits),
            Err(Error::Metadata {
                reason: MetadataReason::InvalidJson,
                ..
            })
        ));
    }
}
