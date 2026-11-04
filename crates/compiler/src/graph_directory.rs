//! Bounded decoder for the legacy directory container, distinct from BmlGraph's
//! single-file serialization. Parsing does not validate execution stack/loops.
use crate::{
    fragment::{BmlGraph, Fragment, BMLGRAPH_MAGIC, BMLGRAPH_SELECTION_VERSION, BMLGRAPH_VERSION},
    gguf_compiler::ModelConfig,
    rpn::RpnOp,
};
use std::{fs::File, io::Read, path::Path};

/// Admission bounds for encoded files and decoded collection sizes.
#[derive(Clone, Copy, Debug)]
pub struct GraphLoadLimits {
    /// Combined header and fragment file bytes, not a total RSS budget.
    pub max_total_bytes: usize,
    /// Maximum number of fragment files.
    pub max_fragments: usize,
    /// Maximum number of operations across all fragments.
    pub max_operations: usize,
    /// Maximum number of IEEE constants in the header.
    pub max_constants: usize,
}

impl Default for GraphLoadLimits {
    fn default() -> Self {
        Self {
            max_total_bytes: 64 * 1024 * 1024,
            max_fragments: 65_536,
            max_operations: 1_000_000,
            max_constants: 8_000_000,
        }
    }
}

struct Cursor<'a> {
    remaining: &'a [u8],
}
impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        if count > self.remaining.len() {
            return Err("truncated graph directory record".into());
        }
        let (value, tail) = self.remaining.split_at(count);
        self.remaining = tail;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn length(&mut self) -> Result<usize, String> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| "graph length exceeds address space".into())
    }
    fn finish(self) -> Result<(), String> {
        if self.remaining.is_empty() {
            Ok(())
        } else {
            Err("trailing bytes in graph directory record".into())
        }
    }
}

fn read_file(path: &Path, budget: &mut usize) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("graph metadata: {e}"))?;
    let declared = usize::try_from(metadata.len()).map_err(|_| "graph file size overflow")?;
    if !metadata.is_file() || declared > *budget {
        return Err("graph file exceeds remaining byte budget or is not a regular file".into());
    }
    read_contents(&mut file, declared, budget)
}

fn read_contents(
    reader: &mut impl Read,
    declared: usize,
    budget: &mut usize,
) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(declared)
        .map_err(|_| "graph allocation failed")?;
    let mut chunk = [0; 8192];
    loop {
        let count = reader
            .read(&mut chunk)
            .map_err(|e| format!("graph read: {e}"))?;
        if count == 0 {
            break;
        }
        // Check again while reading: the file may have grown since metadata().
        if count > *budget - bytes.len() {
            return Err("graph files exceed byte budget".into());
        }
        bytes
            .try_reserve(count)
            .map_err(|_| "graph allocation failed")?;
        bytes.extend_from_slice(&chunk[..count]);
    }
    *budget -= bytes.len();
    Ok(bytes)
}

#[cfg(test)]
mod bounded_read_tests {
    use super::read_contents;
    use std::io::Cursor;

    #[test]
    fn growth_after_metadata_is_bounded_and_failure_preserves_budget() {
        let mut budget = 8193;
        let mut input = Cursor::new(vec![0u8; 8194]);
        let error = read_contents(&mut input, 1, &mut budget).unwrap_err();
        assert!(error.contains("byte budget"));
        assert_eq!(budget, 8193);

        let mut input = Cursor::new(vec![0u8; 8193]);
        assert_eq!(
            read_contents(&mut input, 1, &mut budget).unwrap().len(),
            8193
        );
        assert_eq!(budget, 0);
    }
}

/// Load the legacy graph directory with default resource limits.
/// Constants remain in the separate return value, as in the historical API.
pub fn load_from_dir(input: &Path) -> Result<(BmlGraph, Vec<f64>, ModelConfig), String> {
    load_from_dir_with_limits(input, GraphLoadLimits::default())
}

/// Decode versioned files with bounded allocations and exact record consumption.
/// This checks framing and opcodes, not stack effects, loop budgets or model completeness.
pub fn load_from_dir_with_limits(
    input: &Path,
    limits: GraphLoadLimits,
) -> Result<(BmlGraph, Vec<f64>, ModelConfig), String> {
    let mut bytes_left = limits.max_total_bytes;
    let header = read_file(&input.join("header.bmlgraph"), &mut bytes_left)?;
    let mut cursor = Cursor { remaining: &header };
    if cursor.u32()? != BMLGRAPH_MAGIC {
        return Err("invalid graph directory magic".into());
    }
    let version = cursor.u32()?;
    if !matches!(version, BMLGRAPH_VERSION | BMLGRAPH_SELECTION_VERSION) {
        return Err("unsupported graph directory version".into());
    }
    let count = cursor.u32()? as usize;
    if count > limits.max_fragments {
        return Err("graph fragment limit exceeded".into());
    }
    let architecture_length = cursor.length()?;
    let architecture = std::str::from_utf8(cursor.take(architecture_length)?)
        .map_err(|_| "invalid graph architecture UTF-8")?
        .to_owned();
    let config = ModelConfig {
        architecture,
        n_layers: cursor.u32()?,
        n_heads: cursor.u32()?,
        n_embd: cursor.u32()?,
        context_length: cursor.u32()?,
        vocab_size: cursor.u32()?,
    };
    let constants = cursor.length()?;
    if constants > limits.max_constants || constants > cursor.remaining.len() / 8 {
        return Err("constant count exceeds budget or available bytes".into());
    }
    let mut pool = Vec::new();
    pool.try_reserve_exact(constants)
        .map_err(|_| "constant allocation failed")?;
    for _ in 0..constants {
        pool.push(f64::from_le_bytes(cursor.take(8)?.try_into().unwrap()));
    }
    cursor.finish()?;
    // Each fragment needs at least a four-byte operation count.
    if count > bytes_left / 4 {
        return Err("fragment count exceeds remaining byte budget".into());
    }
    let mut fragments = Vec::new();
    fragments
        .try_reserve_exact(count)
        .map_err(|_| "fragment allocation failed")?;
    let mut operations_left = limits.max_operations;
    for i in 0..count {
        let bytes = read_file(
            &input.join(format!("fragment_{i}.bmlgraph")),
            &mut bytes_left,
        )?;
        let mut cursor = Cursor { remaining: &bytes };
        let count = cursor.u32()? as usize;
        if count > operations_left || count > cursor.remaining.len() {
            return Err("operation count exceeds budget or available bytes".into());
        }
        operations_left -= count;
        let mut ops = Vec::new();
        ops.try_reserve_exact(count)
            .map_err(|_| "operation allocation failed")?;
        for _ in 0..count {
            let op = match cursor.take(1)?[0] {
                0 => RpnOp::One,
                1 => RpnOp::Bml,
                2 => RpnOp::Dup,
                3 => RpnOp::Loop {
                    count: cursor.u32()?,
                    body_len: cursor.u32()?,
                },
                4 => RpnOp::Var(cursor.u32()?),
                5 => RpnOp::Const(cursor.u32()?),
                6 => RpnOp::Zero,
                7 => RpnOp::VarIndexed {
                    base: cursor.u32()?,
                },
                8 => RpnOp::StoreResult {
                    slot: cursor.u32()?,
                },
                9 => RpnOp::FAdd,
                10 => RpnOp::FMul,
                16 if version >= BMLGRAPH_SELECTION_VERSION => RpnOp::FMax,
                11 => RpnOp::Pick {
                    depth: cursor.u32()?,
                },
                12 => RpnOp::Drop,
                13 => RpnOp::Swap,
                tag => return Err(format!("unknown graph opcode: {tag}")),
            };
            ops.push(op);
        }
        cursor.finish()?;
        fragments.push(Fragment { ops });
    }
    Ok((
        BmlGraph {
            const_pool: Vec::new(),
            fragments,
            threshold: 0,
        },
        pool,
        config,
    ))
}
