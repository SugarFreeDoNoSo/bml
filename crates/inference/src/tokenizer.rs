//! SentencePiece BPE for the GGUF `llama` tokenizer profile.
//!
//! Spaces are escaped as U+2581 without Unicode normalization or whitespace
//! collapsing. Adjacent pieces merge by score, with the leftmost pair winning
//! ties. Unknown pieces use their exact UTF-8 bytes, never a BOS replacement.
//! This profile requires the complete set of 256 SentencePiece byte tokens.

use crate::{Error, Result};
use bml_gguf::{GgufDataType, GgufMetadataValue, GgufReader};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

const NORMAL: i32 = 1;
const UNKNOWN: i32 = 2;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;
const BYTE: i32 = 6;
const SPACE: char = '\u{2581}';

/// Serializable vocabulary and explicit SentencePiece tokenization policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSpec {
    /// Token spellings in token-ID order.
    pub tokens: Vec<String>,
    /// Finite BPE merge scores, one per token; larger scores merge first.
    pub scores: Vec<f32>,
    /// SentencePiece types: normal=1, unknown=2, control=3, user-defined=4,
    /// unused=5 and byte=6.
    pub token_types: Vec<i32>,
    /// Beginning-of-sequence control token.
    pub bos_id: u32,
    /// End-of-sequence control token.
    pub eos_id: u32,
    /// Unknown token; required for vocabulary identity, not byte fallback.
    pub unk_id: u32,
    /// Whether encoding with special tokens prepends BOS.
    pub add_bos: bool,
    /// Whether encoding with special tokens appends EOS.
    pub add_eos: bool,
    /// Whether each ordinary text fragment after a special token gets a space.
    pub add_space_prefix: bool,
}

impl TokenizerSpec {
    /// Imports a `llama` SentencePiece vocabulary with required IDs and arrays.
    ///
    /// Absent optional flags use the SentencePiece defaults: BOS enabled, EOS
    /// disabled, and a space prefix enabled. Other normalization profiles are
    /// rejected. Construction of a [`Tokenizer`] validates all vocabulary data.
    pub fn from_gguf(reader: &GgufReader) -> Result<Self> {
        if !matches!(reader.get_metadata("tokenizer.ggml.model"), Some(GgufMetadataValue::String(model)) if model == "llama")
        {
            return Err(Error("only tokenizer.ggml.model=llama is supported".into()));
        }
        if let Some(value) = reader.get_metadata("tokenizer.ggml.pre") {
            if !matches!(value, GgufMetadataValue::String(name) if name == "default" || name == "llama")
            {
                return Err(Error("unsupported SentencePiece pre-tokenizer".into()));
            }
        }
        for key in [
            "tokenizer.ggml.remove_extra_whitespaces",
            "tokenizer.ggml.clean_spaces",
        ] {
            if optional_bool(reader, key, false)? {
                return Err(Error(format!("unsupported tokenizer normalization: {key}")));
            }
        }
        if reader
            .get_metadata("tokenizer.ggml.precompiled_charsmap")
            .is_some()
        {
            return Err(Error(
                "precompiled tokenizer normalization is unsupported".into(),
            ));
        }
        let tokens = array(reader, "tokenizer.ggml.tokens", GgufDataType::String)?
            .iter()
            .map(|value| match value {
                GgufMetadataValue::String(text) => Ok(text.clone()),
                _ => Err(Error("tokenizer.ggml.tokens must contain strings".into())),
            })
            .collect::<Result<Vec<_>>>()?;
        let scores = array(reader, "tokenizer.ggml.scores", GgufDataType::F32)?
            .iter()
            .map(|value| match value {
                GgufMetadataValue::F32(score) => Ok(*score),
                _ => Err(Error(
                    "tokenizer.ggml.scores must contain f32 scores".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let token_types = array(reader, "tokenizer.ggml.token_type", GgufDataType::I32)?
            .iter()
            .map(|value| match value {
                GgufMetadataValue::I32(kind) => Ok(*kind),
                _ => Err(Error(
                    "tokenizer.ggml.token_type must contain i32 types".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        let spec = Self {
            tokens,
            scores,
            token_types,
            bos_id: required_id(reader, "tokenizer.ggml.bos_token_id")?,
            eos_id: required_id(reader, "tokenizer.ggml.eos_token_id")?,
            unk_id: required_id(reader, "tokenizer.ggml.unknown_token_id")?,
            add_bos: optional_bool(reader, "tokenizer.ggml.add_bos_token", true)?,
            add_eos: optional_bool(reader, "tokenizer.ggml.add_eos_token", false)?,
            add_space_prefix: optional_bool(reader, "tokenizer.ggml.add_space_prefix", true)?,
        };
        // Validate imported metadata immediately, not only at first encoding.
        spec.validate()?;
        Ok(spec)
    }

    fn validate(&self) -> Result<()> {
        let count = self.tokens.len();
        if count == 0 || count > i32::MAX as usize {
            return Err(Error(
                "tokenizer vocabulary size is outside the supported range".into(),
            ));
        }
        if self.scores.len() != count || self.token_types.len() != count {
            return Err(Error(
                "tokenizer tokens, scores and types must have equal lengths".into(),
            ));
        }
        let mut names = std::collections::HashSet::with_capacity(count);
        let mut bytes = [false; 256];
        for (id, token) in self.tokens.iter().enumerate() {
            if token.is_empty() || !names.insert(token.as_str()) {
                return Err(Error(format!(
                    "empty or duplicate tokenizer spelling at ID {id}"
                )));
            }
            if !self.scores[id].is_finite() {
                return Err(Error(format!("non-finite tokenizer score at ID {id}")));
            }
            if !(NORMAL..=BYTE).contains(&self.token_types[id]) {
                return Err(Error(format!("unsupported tokenizer type at ID {id}")));
            }
            if self.token_types[id] == BYTE {
                let byte = parse_byte(token)
                    .ok_or_else(|| Error(format!("invalid byte token at ID {id}")))?;
                bytes[byte as usize] = true;
            }
        }
        if bytes.contains(&false) {
            return Err(Error(
                "SentencePiece requires all 256 <0xXX> byte tokens".into(),
            ));
        }
        for (name, id, kind) in [
            ("BOS", self.bos_id, CONTROL),
            ("EOS", self.eos_id, CONTROL),
            ("UNK", self.unk_id, UNKNOWN),
        ] {
            if self.token_types.get(id as usize) != Some(&kind) {
                return Err(Error(format!("invalid {name} token ID or type")));
            }
        }
        Ok(())
    }
}

fn array<'a>(
    reader: &'a GgufReader,
    key: &str,
    kind: GgufDataType,
) -> Result<&'a [GgufMetadataValue]> {
    match reader.get_metadata(key) {
        Some(GgufMetadataValue::Array(actual, values)) if *actual == kind => Ok(values),
        _ => Err(Error(format!(
            "missing or incorrectly typed tokenizer metadata: {key}"
        ))),
    }
}

fn required_id(reader: &GgufReader, key: &str) -> Result<u32> {
    match reader.get_metadata(key) {
        Some(GgufMetadataValue::U32(id)) => Ok(*id),
        _ => Err(Error(format!(
            "missing or incorrectly typed tokenizer ID: {key}"
        ))),
    }
}

fn optional_bool(reader: &GgufReader, key: &str, default: bool) -> Result<bool> {
    match reader.get_metadata(key) {
        Some(GgufMetadataValue::Bool(value)) => Ok(*value),
        None => Ok(default),
        _ => Err(Error(format!("tokenizer flag must be boolean: {key}"))),
    }
}

fn parse_byte(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6 || &bytes[..3] != b"<0x" || bytes[5] != b'>' {
        return None;
    }
    let nibble = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    };
    Some(nibble(bytes[3])? * 16 + nibble(bytes[4])?)
}

/// Validated SentencePiece BPE tokenizer with lossless byte-piece decoding.
#[derive(Debug)]
pub struct Tokenizer {
    spec: TokenizerSpec,
    ids: HashMap<String, u32>,
    byte_ids: [u32; 256],
    user_defined: Vec<u32>,
}

impl Tokenizer {
    /// Validates vocabulary metadata and prepares token and byte lookup tables.
    pub fn new(spec: TokenizerSpec) -> Result<Self> {
        spec.validate()?;
        let mut byte_ids = [0; 256];
        let mut user_defined = Vec::new();
        let mut ids = HashMap::with_capacity(spec.tokens.len());
        for (id, token) in spec.tokens.iter().enumerate() {
            let id = id as u32;
            ids.insert(token.clone(), id);
            if spec.token_types[id as usize] == BYTE {
                // The spelling has already passed validation.
                if let Some(byte) = parse_byte(token) {
                    byte_ids[byte as usize] = id;
                }
            }
            if spec.token_types[id as usize] == USER_DEFINED {
                user_defined.push(id);
            }
        }
        user_defined.sort_by(|&left, &right| {
            spec.tokens[right as usize]
                .len()
                .cmp(&spec.tokens[left as usize].len())
                .then(left.cmp(&right))
        });
        Ok(Self {
            spec,
            ids,
            byte_ids,
            user_defined,
        })
    }

    /// Returns the validated vocabulary and tokenization policy.
    pub fn spec(&self) -> &TokenizerSpec {
        &self.spec
    }

    /// Returns the number of tokens, including byte and special tokens.
    pub fn vocab_size(&self) -> usize {
        self.spec.tokens.len()
    }

    /// Encodes UTF-8 text using scored merges and exact byte fallback.
    ///
    /// `add_special` controls automatic BOS/EOS only. Control-token spellings
    /// in the input remain ordinary text, matching llama.cpp's
    /// `parse_special=false`; user-defined tokens are always recognized.
    /// Empty text never acquires a space prefix.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let mut output = Vec::new();
        if add_special && self.spec.add_bos {
            output.push(self.spec.bos_id);
        }
        let mut previous_special = true;
        for fragment in self.partition(text) {
            match fragment {
                Fragment::Special(id) => {
                    output.push(id);
                    previous_special = true;
                }
                Fragment::Text(start, end) => {
                    let mut normalized = String::new();
                    if previous_special && self.spec.add_space_prefix {
                        normalized.push(SPACE);
                    }
                    for character in text[start..end].chars() {
                        normalized.push(if character == ' ' { SPACE } else { character });
                    }
                    self.merge(&normalized, &mut output);
                    previous_special = false;
                }
            }
        }
        if add_special && self.spec.add_eos {
            output.push(self.spec.eos_id);
        }
        Ok(output)
    }

    /// Decodes complete text, returning an error for invalid token IDs or UTF-8.
    ///
    /// Use [`Self::decode_bytes`] for incomplete UTF-8 generated byte tokens.
    /// Space-prefix and special-token handling follow that method.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        String::from_utf8(self.decode_bytes(ids, skip_special)?).map_err(|error| {
            Error(format!(
                "decoded token bytes are not complete UTF-8: {error}"
            ))
        })
    }

    /// Returns one token's output bytes without removing a leading space.
    ///
    /// This is suitable for streaming: concatenate the returned bytes before
    /// decoding UTF-8, since a character may span multiple byte tokens. With
    /// `skip_special`, unknown/control tokens produce no bytes; user-defined
    /// tokens retain their literal spelling. Unused tokens produce no bytes.
    pub fn token_piece_bytes(&self, id: u32, skip_special: bool) -> Result<Vec<u8>> {
        let token = self
            .spec
            .tokens
            .get(id as usize)
            .ok_or_else(|| Error(format!("token ID {id} is outside the vocabulary")))?;
        match self.spec.token_types[id as usize] {
            NORMAL => Ok(token.replace(SPACE, " ").into_bytes()),
            BYTE => Ok(vec![
                parse_byte(token).ok_or_else(|| Error("invalid byte token".into()))?
            ]),
            UNKNOWN | CONTROL if skip_special => Ok(Vec::new()),
            UNKNOWN | CONTROL | USER_DEFINED => Ok(token.as_bytes().to_vec()),
            UNUSED => Ok(Vec::new()),
            _ => Err(Error("invalid tokenizer type".into())),
        }
    }

    /// Decodes a complete token sequence without losing invalid UTF-8 bytes.
    ///
    /// This follows llama.cpp detokenization: skip unknown/control tokens when
    /// requested, preserve user-defined spellings, and remove at most one
    /// leading space from the first piece when space-prefixing is enabled.
    /// Skipping an automatic initial BOS disables that space removal, so a
    /// BOS-prefixed sequence retains its SentencePiece leading space.
    /// Do not decode each generated token separately: prefix removal applies to
    /// the whole sequence, and UTF-8 characters can span several byte tokens.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Result<Vec<u8>> {
        for &id in ids {
            if id as usize >= self.spec.tokens.len() {
                return Err(Error(format!("token ID {id} is outside the vocabulary")));
            }
        }
        let mut ids = ids;
        let mut remove_space = self.spec.add_space_prefix;
        if skip_special && self.spec.add_bos && ids.first() == Some(&self.spec.bos_id) {
            ids = &ids[1..];
            remove_space = false;
        }
        if skip_special && self.spec.add_eos && ids.last() == Some(&self.spec.eos_id) {
            ids = &ids[..ids.len() - 1];
        }
        let mut output = Vec::new();
        for &id in ids {
            let piece = self.token_piece_bytes(id, skip_special)?;
            let offset = usize::from(remove_space && piece.first() == Some(&b' '));
            output.extend_from_slice(&piece[offset..]);
            remove_space = false;
        }
        Ok(output)
    }

    fn partition(&self, text: &str) -> Vec<Fragment> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut fragments = vec![Fragment::Text(0, text.len())];
        // As in SentencePiece's special partition, longer user-defined
        // spellings are selected before shorter spellings, even on overlap.
        for &id in &self.user_defined {
            let spelling = &self.spec.tokens[id as usize];
            let mut next = Vec::new();
            for fragment in fragments {
                if let Fragment::Text(start, end) = fragment {
                    let mut cursor = start;
                    for (relative, _) in text[start..end].match_indices(spelling) {
                        let position = start + relative;
                        if cursor < position {
                            next.push(Fragment::Text(cursor, position));
                        }
                        next.push(Fragment::Special(id));
                        cursor = position + spelling.len();
                    }
                    if cursor < end {
                        next.push(Fragment::Text(cursor, end));
                    }
                } else {
                    next.push(fragment);
                }
            }
            fragments = next;
        }
        fragments
    }

    fn merge(&self, text: &str, output: &mut Vec<u32>) {
        let mut symbols = text
            .char_indices()
            .enumerate()
            .map(|(index, (start, ch))| Symbol {
                start,
                end: start + ch.len_utf8(),
                previous: index.checked_sub(1),
                next: Some(index + 1),
                active: true,
            })
            .collect::<Vec<_>>();
        if let Some(last) = symbols.last_mut() {
            last.next = None;
        } else {
            return;
        }
        let mut candidates = BinaryHeap::new();
        for left in 0..symbols.len() - 1 {
            self.candidate(text, &symbols, left, left + 1, &mut candidates);
        }
        while let Some(pair) = candidates.pop() {
            let left = pair.left;
            let right = pair.right;
            if !symbols[left].active
                || !symbols[right].active
                || symbols[left].next != Some(right)
                || symbols[right].end - symbols[left].start != pair.length
            {
                continue;
            }
            symbols[left].end = symbols[right].end;
            symbols[left].next = symbols[right].next;
            symbols[right].active = false;
            if let Some(next) = symbols[left].next {
                symbols[next].previous = Some(left);
                self.candidate(text, &symbols, left, next, &mut candidates);
            }
            if let Some(previous) = symbols[left].previous {
                self.candidate(text, &symbols, previous, left, &mut candidates);
            }
        }
        for symbol in symbols.iter().filter(|symbol| symbol.active) {
            let piece = &text[symbol.start..symbol.end];
            if let Some(&id) = self.ids.get(piece) {
                output.push(id);
            } else {
                output.extend(
                    piece
                        .as_bytes()
                        .iter()
                        .map(|byte| self.byte_ids[*byte as usize]),
                );
            }
        }
    }

    fn candidate(
        &self,
        text: &str,
        symbols: &[Symbol],
        left: usize,
        right: usize,
        candidates: &mut BinaryHeap<Pair>,
    ) {
        let start = symbols[left].start;
        let end = symbols[right].end;
        if let Some(&id) = self.ids.get(&text[start..end]) {
            candidates.push(Pair {
                score: self.spec.scores[id as usize],
                left,
                right,
                length: end - start,
            });
        }
    }
}

enum Fragment {
    Text(usize, usize),
    Special(u32),
}

struct Symbol {
    start: usize,
    end: usize,
    previous: Option<usize>,
    next: Option<usize>,
    active: bool,
}

#[derive(PartialEq)]
struct Pair {
    score: f32,
    left: usize,
    right: usize,
    length: usize,
}

impl Eq for Pair {}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        // Scores are finite by construction. Signed zero ties numerically.
        let score_order = if self.score == other.score {
            Ordering::Equal
        } else {
            self.score.total_cmp(&other.score)
        };
        score_order
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| self.length.cmp(&other.length))
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
