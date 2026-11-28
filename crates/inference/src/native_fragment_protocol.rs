//! Bounded private-stdio protocol for native F32 row-fragment executables.
//!
//! This diagnostic protocol is distinct from the symbolic worker protocol and
//! from the autonomous monolithic ELF. The executable authenticates its own
//! immutable embedded rows before emitting Ready. Its owner must authenticate
//! the executable and compare the complete Ready descriptor before sending data.
//! Activations/results retain their F32 bits; arithmetic is the model's shared
//! ordered F64 projection with its original final F32 cast, never a new backend.
//!
//! One process serves at most 256 requests and retains no activation or KV state.
//! Frames exclude their four-byte length prefix from the 256 KiB bound. Each
//! fragment has at most 32,768 rows, 4,096 columns, 64 MiB of embedded F32 payload
//! and 32 Mi logical products per request. The process owner separately enforces
//! deadlines, total model work and exact coverage of the output rows.

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

/// Version of the native row-fragment descriptor and wire envelope.
pub const VERSION: u32 = 1;
/// Largest frame body, excluding its four-byte length prefix.
pub const MAX_FRAME_BYTES: usize = 256 * 1024;
/// Largest admitted immutable weight payload per executable.
pub const MAX_WEIGHT_BYTES: usize = 64 * 1024 * 1024;
/// Maximum total vocabulary/row dimension of this diagnostic profile.
pub const MAX_ROWS: usize = 32_768;
/// Maximum activation width of this diagnostic profile.
pub const MAX_COLUMNS: usize = 4_096;
/// Upper bound on logical products evaluated by a fragment in one request.
pub const MAX_MACS: usize = 32 * 1024 * 1024;
/// A process accepts sequences 0 through 255, then only clean EOF.
pub const MAX_REQUESTS: u32 = 256;

const MAGIC: &[u8; 4] = b"BMLF";
const HEADER_BYTES: usize = 12;
const READY: u32 = 1;
const PROJECT: u32 = 2;
const OUTPUT: u32 = 3;
const MAX_DESCRIPTOR_BYTES: usize = 4 * 1024;

/// Compiled identity and exact row ownership of one native fragment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    /// Descriptor and protocol version; currently one.
    pub version: u32,
    /// Lowercase SHA-256 of the GGUF used at compilation time.
    pub source_sha256: String,
    /// Canonical model tensor name; not a filesystem path.
    pub tensor_name: String,
    /// First owned row, inclusive, in the original tensor.
    pub row_start: usize,
    /// Last owned row, exclusive, in the original tensor.
    pub row_end: usize,
    /// Complete row count of the original tensor.
    pub total_rows: usize,
    /// F32 activation width; GGML axis zero is contiguous.
    pub cols: usize,
    /// Lowercase SHA-256 of exactly the embedded little-endian F32 rows.
    pub weights_sha256: String,
}

impl Descriptor {
    /// Validate descriptor shape, quotas and canonical identity fields.
    ///
    /// This does not authenticate any weight bytes. The owner can use it while
    /// checking the expected Ready descriptor; workers must also call validate.
    pub fn validate_metadata(&self) -> Result<()> {
        self.layout().map(|_| ())
    }

    /// Authenticate the complete embedded payload and reject non-finite weights.
    pub fn validate(&self, weights: &[u8]) -> Result<()> {
        let (_, bytes) = self.layout()?;
        if weights.len() != bytes {
            return Err(Error("fragment weight payload length mismatch".into()));
        }
        if format!("{:x}", Sha256::digest(weights)) != self.weights_sha256 {
            return Err(Error("fragment weight SHA-256 mismatch".into()));
        }
        for chunk in weights.as_chunks::<4>().0 {
            if !f32::from_le_bytes(*chunk).is_finite() {
                return Err(Error("fragment weight is not finite".into()));
            }
        }
        Ok(())
    }

    fn layout(&self) -> Result<(usize, usize)> {
        if self.version != VERSION {
            return Err(Error("unsupported native fragment version".into()));
        }
        if !sha256_text(&self.source_sha256) || !sha256_text(&self.weights_sha256) {
            return Err(Error("fragment identity must be lowercase SHA-256".into()));
        }
        if self.tensor_name.is_empty()
            || self.tensor_name.len() > 128
            || !self
                .tensor_name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'.')
        {
            return Err(Error("invalid fragment tensor name".into()));
        }
        if self.total_rows == 0
            || self.total_rows > MAX_ROWS
            || self.cols == 0
            || self.cols > MAX_COLUMNS
            || self.row_start >= self.row_end
            || self.row_end > self.total_rows
        {
            return Err(Error("fragment dimensions or row coverage invalid".into()));
        }
        let rows = self
            .row_end
            .checked_sub(self.row_start)
            .ok_or_else(|| Error("fragment row count overflow".into()))?;
        let cells = rows
            .checked_mul(self.cols)
            .filter(|&count| count <= MAX_MACS)
            .ok_or_else(|| Error("fragment logical work exceeds budget".into()))?;
        let bytes = cells
            .checked_mul(4)
            .filter(|&count| count <= MAX_WEIGHT_BYTES)
            .ok_or_else(|| Error("fragment weight bytes exceed budget".into()))?;
        Ok((rows, bytes))
    }
}

fn sha256_text(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn buffer(length: usize) -> Result<Vec<u8>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| Error("native fragment buffer allocation failed".into()))?;
    result.resize(length, 0);
    Ok(result)
}

fn read_frame(reader: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut prefix = [0u8; 4];
    loop {
        match reader.read(&mut prefix[..1]) {
            Ok(0) => return Ok(None),
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    reader.read_exact(&mut prefix[1..])?;
    let length = u32::from_le_bytes(prefix) as usize;
    if !(HEADER_BYTES..=MAX_FRAME_BYTES).contains(&length) {
        return Err(Error("native fragment frame length exceeds bounds".into()));
    }
    let mut body = buffer(length)?;
    reader.read_exact(&mut body)?;
    Ok(Some(body))
}

fn write_frame(writer: &mut impl Write, kind: u32, payload: &[u8]) -> Result<()> {
    let length = payload
        .len()
        .checked_add(HEADER_BYTES)
        .filter(|&n| n <= MAX_FRAME_BYTES)
        .ok_or_else(|| Error("native fragment frame length exceeds bounds".into()))?;
    let length =
        u32::try_from(length).map_err(|_| Error("native fragment frame length overflow".into()))?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&kind.to_le_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(())
}

fn field(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| Error("native fragment field offset overflow".into()))?;
    let word: [u8; 4] = bytes
        .get(offset..end)
        .ok_or_else(|| Error("truncated native fragment field".into()))?
        .try_into()
        .map_err(|_| Error("truncated native fragment field".into()))?;
    Ok(u32::from_le_bytes(word))
}

fn payload(body: &[u8], kind: u32) -> Result<&[u8]> {
    if body.get(..4) != Some(MAGIC.as_slice())
        || field(body, 4)? != VERSION
        || field(body, 8)? != kind
    {
        return Err(Error("native fragment envelope mismatch".into()));
    }
    body.get(HEADER_BYTES..)
        .ok_or_else(|| Error("truncated native fragment envelope".into()))
}

fn values(bytes: &[u8], count: usize) -> Result<Vec<f32>> {
    if count.checked_mul(4) != Some(bytes.len()) {
        return Err(Error("native fragment vector length mismatch".into()));
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| Error("native fragment vector allocation failed".into()))?;
    for chunk in bytes.as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*chunk);
        if !value.is_finite() {
            return Err(Error("native fragment vector is not finite".into()));
        }
        result.push(value);
    }
    Ok(result)
}

/// Read Ready and require identity/shape equality with the authenticated plan.
pub fn handshake(reader: &mut impl Read, expected: &Descriptor) -> Result<()> {
    expected.validate_metadata()?;
    let frame =
        read_frame(reader)?.ok_or_else(|| Error("native fragment closed before Ready".into()))?;
    let data = payload(&frame, READY)?;
    if data.len() > MAX_DESCRIPTOR_BYTES {
        return Err(Error(
            "native fragment Ready descriptor exceeds budget".into(),
        ));
    }
    let actual: Descriptor = serde_json::from_slice(data)?;
    actual.validate_metadata()?;
    if actual != *expected {
        return Err(Error("native fragment Ready identity mismatch".into()));
    }
    Ok(())
}

/// Send one ordered projection request, retaining the exact F32 activation bits.
///
/// The owner increments sequence only after accepting the matching response.
/// Retrying after transport failure is not supported; discard that process.
pub fn write_request(writer: &mut impl Write, sequence: u32, input: &[f32]) -> Result<()> {
    if sequence >= MAX_REQUESTS {
        return Err(Error("native fragment request quota exhausted".into()));
    }
    if input.is_empty() || input.len() > MAX_COLUMNS || input.iter().any(|v| !v.is_finite()) {
        return Err(Error("invalid native fragment activation".into()));
    }
    let mut data = buffer(8 + input.len() * 4)?;
    data[..4].copy_from_slice(&sequence.to_le_bytes());
    data[4..8].copy_from_slice(&(input.len() as u32).to_le_bytes());
    for (slot, value) in data[8..].as_chunks_mut::<4>().0.iter_mut().zip(input) {
        slot.copy_from_slice(&value.to_bits().to_le_bytes());
    }
    write_frame(writer, PROJECT, &data)
}

/// Receive the complete finite output for exactly the expected owned row range.
pub fn read_response(
    reader: &mut impl Read,
    expected: &Descriptor,
    sequence: u32,
) -> Result<Vec<f32>> {
    expected.validate_metadata()?;
    if sequence >= MAX_REQUESTS {
        return Err(Error("native fragment request quota exhausted".into()));
    }
    let frame =
        read_frame(reader)?.ok_or_else(|| Error("native fragment closed before output".into()))?;
    let data = payload(&frame, OUTPUT)?;
    let rows = expected.row_end - expected.row_start;
    if field(data, 0)? != sequence
        || field(data, 4)? as usize != expected.row_start
        || field(data, 8)? as usize != expected.row_end
        || field(data, 12)? as usize != rows
    {
        return Err(Error(
            "native fragment output sequence or coverage mismatch".into(),
        ));
    }
    values(
        data.get(16..)
            .ok_or_else(|| Error("truncated native fragment output".into()))?,
        rows,
    )
}

/// Serve authenticated immutable rows on private stdin/stdout-like streams.
///
/// Clean EOF is accepted only between complete frames. Any malformed request,
/// arithmetic error or I/O failure terminates the session without an Output for
/// that request. Errors belong on the executable's stderr, never on this stream.
pub fn serve<R: Read, W: Write>(
    descriptor: &Descriptor,
    weights: &[u8],
    mut reader: R,
    mut writer: W,
) -> Result<()> {
    descriptor.validate(weights)?;
    let ready = serde_json::to_vec(descriptor)?;
    if ready.len() > MAX_DESCRIPTOR_BYTES {
        return Err(Error(
            "native fragment Ready descriptor exceeds budget".into(),
        ));
    }
    write_frame(&mut writer, READY, &ready)?;
    let rows = descriptor.row_end - descriptor.row_start;
    let mut sequence = 0;
    while let Some(frame) = read_frame(&mut reader)? {
        if sequence >= MAX_REQUESTS {
            return Err(Error("native fragment request quota exhausted".into()));
        }
        let data = payload(&frame, PROJECT)?;
        if field(data, 0)? != sequence || field(data, 4)? as usize != descriptor.cols {
            return Err(Error(
                "native fragment request sequence or width mismatch".into(),
            ));
        }
        let input = values(
            data.get(8..)
                .ok_or_else(|| Error("truncated native fragment request".into()))?,
            descriptor.cols,
        )?;
        let output = crate::model::project_f32_rows(weights, rows, descriptor.cols, &input)?;
        if output.len() != rows || output.iter().any(|v| !v.is_finite()) {
            return Err(Error(
                "native fragment kernel returned invalid output".into(),
            ));
        }
        let mut data = buffer(16 + rows * 4)?;
        data[..4].copy_from_slice(&sequence.to_le_bytes());
        data[4..8].copy_from_slice(&(descriptor.row_start as u32).to_le_bytes());
        data[8..12].copy_from_slice(&(descriptor.row_end as u32).to_le_bytes());
        data[12..16].copy_from_slice(&(rows as u32).to_le_bytes());
        for (slot, value) in data[16..].as_chunks_mut::<4>().0.iter_mut().zip(output) {
            slot.copy_from_slice(&value.to_bits().to_le_bytes());
        }
        write_frame(&mut writer, OUTPUT, &data)?;
        sequence += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn fixture() -> (Descriptor, Vec<u8>) {
        let weights = [1.0f32, -2.0, 3.0, -0.0, -0.0, -0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let descriptor = Descriptor {
            version: VERSION,
            source_sha256: "a".repeat(64),
            tensor_name: "output.weight".into(),
            row_start: 2,
            row_end: 4,
            total_rows: 5,
            cols: 3,
            weights_sha256: format!("{:x}", Sha256::digest(&weights)),
        };
        (descriptor, weights)
    }

    fn ready_bytes(descriptor: &Descriptor) -> Vec<u8> {
        let mut result = Vec::new();
        write_frame(&mut result, READY, &serde_json::to_vec(descriptor).unwrap()).unwrap();
        result
    }

    fn raw_request(sequence: u32, width: u32, bits: &[u32]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(sequence.to_le_bytes());
        data.extend(width.to_le_bytes());
        for b in bits {
            data.extend(b.to_le_bytes());
        }
        let mut frame = Vec::new();
        write_frame(&mut frame, PROJECT, &data).unwrap();
        frame
    }

    fn request_failure(descriptor: &Descriptor, weights: &[u8], input: &[u8]) -> String {
        let mut output = Vec::new();
        let error = serve(descriptor, weights, input, &mut output).unwrap_err();
        assert_eq!(
            output,
            ready_bytes(descriptor),
            "failed request emitted Output"
        );
        error.0
    }

    #[test]
    fn fragment_stdio_roundtrip_retains_signed_zero_and_separates_activations() {
        let (descriptor, weights) = fixture();
        let mut requests = Vec::new();
        write_request(&mut requests, 0, &[2.0, 3.0, 4.0]).unwrap();
        write_request(&mut requests, 1, &[1.0, 1.0, 1.0]).unwrap();
        let mut result = Vec::new();
        serve(&descriptor, &weights, &requests[..], &mut result).unwrap();
        let mut reader = Cursor::new(result);
        handshake(&mut reader, &descriptor).unwrap();
        let first = read_response(&mut reader, &descriptor, 0).unwrap();
        let second = read_response(&mut reader, &descriptor, 1).unwrap();
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![8.0f32.to_bits(), (-0.0f32).to_bits()]
        );
        assert_eq!(
            second.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![2.0f32.to_bits(), (-0.0f32).to_bits()]
        );
        assert!(read_frame(&mut reader).unwrap().is_none());
        assert_eq!(weights, fixture().1);
    }

    #[test]
    fn fragment_descriptor_rejects_identity_coverage_and_payload_changes_before_ready() {
        let (good, weights) = fixture();
        let mut invalid = Vec::new();
        for change in 0..11 {
            let mut d = good.clone();
            match change {
                0 => d.version = 2,
                1 => d.source_sha256 = "A".repeat(64),
                2 => d.weights_sha256 = "0".repeat(63),
                3 => d.tensor_name = "../output.weight".into(),
                4 => d.row_start = d.row_end,
                5 => d.row_end = d.total_rows + 1,
                6 => d.total_rows = 0,
                7 => d.total_rows = usize::MAX,
                8 => d.cols = 0,
                9 => d.cols = usize::MAX,
                _ => d.row_start = usize::MAX,
            }
            invalid.push(d);
        }
        for d in invalid {
            let mut output = Vec::new();
            assert!(serve(&d, &weights, &[][..], &mut output).is_err());
            assert!(output.is_empty());
        }
        for payload in [&weights[..weights.len() - 1], &[0u8; 24][..]] {
            let mut output = Vec::new();
            assert!(serve(&good, payload, &[][..], &mut output).is_err());
            assert!(output.is_empty());
        }
        let mut nonfinite = weights.clone();
        nonfinite[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        let mut d = good.clone();
        d.weights_sha256 = format!("{:x}", Sha256::digest(&nonfinite));
        assert!(d.validate(&nonfinite).unwrap_err().0.contains("not finite"));
    }

    #[test]
    fn fragment_descriptor_enforces_inclusive_resource_budgets_without_allocating_payloads() {
        let (mut d, _) = fixture();
        d.row_start = 0;
        d.row_end = MAX_ROWS;
        d.total_rows = MAX_ROWS;
        d.cols = 512;
        assert_eq!(d.layout().unwrap(), (MAX_ROWS, MAX_WEIGHT_BYTES));
        d.cols += 1;
        assert!(d.layout().unwrap_err().0.contains("weight bytes"));
        d.cols = 1024;
        assert!(d.layout().unwrap_err().0.contains("weight bytes"));
        d.cols = 1025;
        assert!(d.layout().unwrap_err().0.contains("logical work"));
        d.row_end = 1;
        d.cols = MAX_COLUMNS;
        assert!(d.validate_metadata().is_ok());
        d.cols += 1;
        assert!(d.validate_metadata().is_err());
    }

    #[test]
    fn fragment_requests_reject_wrong_version_sequence_width_nonfinite_and_trailing_bytes() {
        let (descriptor, weights) = fixture();
        let good = raw_request(0, 3, &[0, 0, 0]);
        let mut bad_version = good.clone();
        bad_version[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(request_failure(&descriptor, &weights, &bad_version).contains("envelope"));
        for bad in [
            raw_request(1, 3, &[0, 0, 0]),
            raw_request(0, 2, &[0, 0]),
            raw_request(0, 3, &[0, 0]),
            raw_request(0, 3, &[0, 0, 0, 0]),
            raw_request(0, 3, &[0, f32::NAN.to_bits(), 0]),
            raw_request(0, 3, &[0, f32::NEG_INFINITY.to_bits(), 0]),
        ] {
            request_failure(&descriptor, &weights, &bad);
        }
        let mut writer = Vec::new();
        assert!(write_request(&mut writer, 0, &[f32::NAN]).is_err());
        assert!(write_request(&mut writer, MAX_REQUESTS, &[1.0]).is_err());
        assert!(writer.is_empty());
    }

    #[test]
    fn fragment_frames_distinguish_clean_eof_from_truncation_and_bound_allocation() {
        let (descriptor, weights) = fixture();
        let mut output = Vec::new();
        serve(&descriptor, &weights, &[][..], &mut output).unwrap();
        assert_eq!(output, ready_bytes(&descriptor));
        let request = raw_request(0, 3, &[0, 0, 0]);
        for n in 1..request.len() {
            request_failure(&descriptor, &weights, &request[..n]);
        }
        for length in [
            0,
            (HEADER_BYTES - 1) as u32,
            (MAX_FRAME_BYTES + 1) as u32,
            u32::MAX,
        ] {
            request_failure(&descriptor, &weights, &length.to_le_bytes());
        }
        let mut frame = Vec::new();
        assert!(write_frame(&mut frame, PROJECT, &vec![0; MAX_FRAME_BYTES]).is_err());
        assert!(frame.is_empty());
    }

    #[test]
    fn fragment_finite_inputs_with_overflowing_f32_output_fail_without_an_output_frame() {
        let (mut descriptor, _) = fixture();
        let weights = f32::MAX.to_le_bytes();
        descriptor.row_end = descriptor.row_start + 1;
        descriptor.cols = 1;
        descriptor.weights_sha256 = format!("{:x}", Sha256::digest(weights));
        descriptor.validate(&weights).unwrap();
        request_failure(
            &descriptor,
            &weights,
            &raw_request(0, 1, &[f32::MAX.to_bits()]),
        );
    }

    #[test]
    fn fragment_handshake_and_outputs_are_bound_to_exact_identity_sequence_and_coverage() {
        let (descriptor, weights) = fixture();
        let mut expected = descriptor.clone();
        expected.source_sha256 = "b".repeat(64);
        assert!(handshake(&mut &ready_bytes(&descriptor)[..], &expected).is_err());
        let mut json = serde_json::to_value(&descriptor).unwrap();
        json["unexpected"] = 1.into();
        let mut extra = Vec::new();
        write_frame(&mut extra, READY, &serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(handshake(&mut &extra[..], &descriptor).is_err());
        let mut outputs = Vec::new();
        serve(
            &descriptor,
            &weights,
            &raw_request(0, 3, &[0, 0, 0])[..],
            &mut outputs,
        )
        .unwrap();
        let offset = ready_bytes(&descriptor).len();
        let good = &outputs[offset..];
        assert!(read_response(&mut &good[..], &descriptor, 1).is_err());
        for at in [20usize, 24, 28] {
            let mut bad = good.to_vec();
            bad[at..at + 4].copy_from_slice(&99u32.to_le_bytes());
            assert!(read_response(&mut &bad[..], &descriptor, 0).is_err());
        }
        let mut bad = good.to_vec();
        bad[32..36].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(read_response(&mut &bad[..], &descriptor, 0).is_err());
        let mut extra = good.to_vec();
        extra.extend(0u32.to_le_bytes());
        let length = field(&extra, 0).unwrap() + 4;
        extra[..4].copy_from_slice(&length.to_le_bytes());
        assert!(read_response(&mut &extra[..], &descriptor, 0).is_err());
    }

    #[test]
    fn fragment_sequence_replay_and_request_quota_are_not_silently_accepted() {
        let (descriptor, weights) = fixture();
        let first = raw_request(0, 3, &[0, 0, 0]);
        let mut output = Vec::new();
        let requests = [first.clone(), first].concat();
        assert!(serve(&descriptor, &weights, &requests[..], &mut output).is_err());
        let mut cursor = Cursor::new(output);
        handshake(&mut cursor, &descriptor).unwrap();
        read_response(&mut cursor, &descriptor, 0).unwrap();
        assert!(read_frame(&mut cursor).unwrap().is_none());
        let mut requests = Vec::new();
        for sequence in 0..MAX_REQUESTS {
            write_request(&mut requests, sequence, &[1.0, 1.0, 1.0]).unwrap();
        }
        let mut output = Vec::new();
        serve(&descriptor, &weights, &requests[..], &mut output).unwrap();
        let mut cursor = Cursor::new(output);
        handshake(&mut cursor, &descriptor).unwrap();
        for sequence in 0..MAX_REQUESTS {
            read_response(&mut cursor, &descriptor, sequence).unwrap();
        }
        assert!(read_frame(&mut cursor).unwrap().is_none());
        requests.extend(raw_request(MAX_REQUESTS, 3, &[0, 0, 0]));
        assert!(serve(&descriptor, &weights, &requests[..], std::io::sink())
            .unwrap_err()
            .0
            .contains("quota"));
    }
}
