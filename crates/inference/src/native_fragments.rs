//! Opt-in native ELF output-head fragments, independent of symbolic BMLX.
//!
//! Workers own only their immutable row ranges. The initial coordinator still
//! embeds the complete model, including a duplicate head; this is not full
//! model-memory partitioning. Private process pipes are a local transport only.

use crate::native_fragment_protocol::{self as wire, Descriptor};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Compiled identity of one sibling ELF and its exact row ownership.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FragmentExecutable {
    /// A simple sibling filename; paths and traversal are forbidden.
    pub file_name: String,
    /// SHA-256 of the entire executable, fixed in the coordinator source.
    pub sha256: String,
    /// Model, payload and matrix-range identity, also bound into the worker.
    pub descriptor: Descriptor,
}

/// Runtime admission for a generated worker; never executes the inspected file.
pub fn verify_elf(path: &Path, expected_hash: &str) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        Error(format!(
            "fragment executable missing or unreadable: {error}"
        ))
    })?;
    if !metadata.file_type().is_file() || metadata.len() > 128 * 1024 * 1024 {
        return Err(Error(
            "fragment executable file type or size rejected".into(),
        ));
    }
    let mut file = File::open(path)?;
    let mut header = [0u8; 20];
    file.read_exact(&mut header)?;
    if &header[..4] != b"\x7fELF"
        || header[4] != 2
        || header[5] != 1
        || header[6] != 1
        || !matches!(u16::from_le_bytes([header[16], header[17]]), 2 | 3)
        || u16::from_le_bytes([header[18], header[19]]) != 62
    {
        return Err(Error(
            "fragment must be a native little-endian AMD64 ELF".into(),
        ));
    }
    let mut hash = Sha256::new();
    hash.update(header);
    let mut block = [0; 65536];
    loop {
        let count = file.read(&mut block)?;
        if count == 0 {
            break;
        }
        hash.update(&block[..count]);
    }
    if format!("{:x}", hash.finalize()) != expected_hash {
        return Err(Error("fragment executable SHA-256 mismatch".into()));
    }
    Ok(())
}

pub(crate) fn layout(fragments: &[FragmentExecutable]) -> Result<(usize, usize)> {
    if fragments.is_empty() || fragments.len() > 8 {
        return Err(Error("expected one to eight output fragments".into()));
    }
    let first = &fragments[0].descriptor;
    if first.total_rows == 0
        || first.total_rows > 32768
        || first.cols == 0
        || first.cols > 4096
        || first
            .total_rows
            .checked_mul(first.cols)
            .is_none_or(|n| n > 32 * 1024 * 1024)
        || first
            .total_rows
            .checked_mul(first.cols)
            .and_then(|n| n.checked_mul(4))
            .is_none_or(|n| n > wire::MAX_WEIGHT_BYTES)
    {
        return Err(Error("fragment matrix dimensions exceed limits".into()));
    }
    let mut next = 0;
    for (i, fragment) in fragments.iter().enumerate() {
        let d = &fragment.descriptor;
        d.validate_metadata()?;
        if fragment.file_name != format!("fragment_{i}")
            || fragment.sha256.len() != 64
            || !fragment.sha256.bytes().all(|c| c.is_ascii_hexdigit())
            || d.version != 1
            || d.row_start != next
            || d.row_end <= next
            || d.row_end > first.total_rows
            || d.total_rows != first.total_rows
            || d.cols != first.cols
            || d.source_sha256 != first.source_sha256
            || d.tensor_name != first.tensor_name
        {
            return Err(Error(
                "fragment identity, order or exact coverage rejected".into(),
            ));
        }
        next = d.row_end;
    }
    if next != first.total_rows {
        return Err(Error("fragment output coverage is incomplete".into()));
    }
    Ok((first.total_rows, first.cols))
}

struct Process {
    child: Child,
    request: Option<Sender<(u32, Vec<f32>)>>,
    response: Receiver<Result<Vec<f32>>>,
    io: Option<JoinHandle<()>>,
}

impl Process {
    fn spawn(path: &Path, expected: &FragmentExecutable) -> Result<Self> {
        verify_elf(path, &expected.sha256)?;
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let mut input = child
            .stdin
            .take()
            .ok_or_else(|| Error("missing worker stdin".into()))?;
        let mut output = child
            .stdout
            .take()
            .ok_or_else(|| Error("missing worker stdout".into()))?;
        let (request, requests) = mpsc::channel::<(u32, Vec<f32>)>();
        let (responses, response) = mpsc::channel();
        let descriptor = expected.descriptor.clone();
        let io = std::thread::Builder::new()
            .name("bml-fragment-stdio".into())
            .spawn(move || {
                if let Err(error) = wire::handshake(&mut output, &descriptor) {
                    let _ = responses.send(Err(error));
                    return;
                }
                if responses.send(Ok(Vec::new())).is_err() {
                    return;
                }
                while let Ok((sequence, values)) = requests.recv() {
                    let result = wire::write_request(&mut input, sequence, &values)
                        .and_then(|()| wire::read_response(&mut output, &descriptor, sequence));
                    let failed = result.is_err();
                    if responses.send(result).is_err() || failed {
                        break;
                    }
                }
            });
        let io = match io {
            Ok(thread) => thread,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.into());
            }
        };
        let process = Self {
            child,
            request: Some(request),
            response,
            io: Some(io),
        };
        process.receive()?;
        Ok(process)
    }

    fn receive(&self) -> Result<Vec<f32>> {
        self.receive_before(Instant::now() + Duration::from_secs(20))
    }

    fn receive_before(&self, deadline: Instant) -> Result<Vec<f32>> {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| {
                Error("fragment aggregate exceeded 20 seconds; discard session".into())
            })?;
        self.response.recv_timeout(remaining).map_err(|_| {
            Error("fragment response failed or exceeded 20 seconds; discard session".into())
        })?
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        self.request.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(io) = self.io.take() {
            let _ = io.join();
        }
    }
}

/// One local set of persistent ELF workers, invalidated after any failed call.
pub struct ProjectionGroup {
    workers: Vec<Process>,
    rows: usize,
    cols: usize,
    sequence: u32,
    failed: bool,
}

impl ProjectionGroup {
    /// Verify full coverage before starting any child, then authenticate handshakes.
    pub fn spawn(directory: &Path, fragments: &[FragmentExecutable]) -> Result<Self> {
        let (rows, cols) = layout(fragments)?;
        // Admit every executable before the first child can run.
        for fragment in fragments {
            verify_elf(&directory.join(&fragment.file_name), &fragment.sha256)?;
        }
        let mut workers = Vec::new();
        for fragment in fragments {
            workers.push(Process::spawn(
                &directory.join(&fragment.file_name),
                fragment,
            )?);
        }
        Ok(Self {
            workers,
            rows,
            cols,
            sequence: 0,
            failed: false,
        })
    }

    /// Compute disjoint rows concurrently; concatenate in the compiled row order.
    /// No reduction is reassociated across fragments and no retry is performed.
    pub fn project(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.failed
            || self.sequence >= 256
            || input.len() != self.cols
            || input.iter().any(|v| !v.is_finite())
        {
            self.failed = true;
            self.workers.clear();
            return Err(Error(
                "fragment session, request budget or activation rejected".into(),
            ));
        }
        let result = (|| {
            let deadline = Instant::now() + Duration::from_secs(20);
            for worker in &self.workers {
                worker
                    .request
                    .as_ref()
                    .ok_or_else(|| Error("fragment closed".into()))?
                    .send((self.sequence, input.to_vec()))
                    .map_err(|_| Error("fragment request channel closed".into()))?;
            }
            let mut output = Vec::with_capacity(self.rows);
            for worker in &self.workers {
                output.extend(worker.receive_before(deadline)?);
            }
            if output.len() != self.rows || output.iter().any(|v| !v.is_finite()) {
                return Err(Error(
                    "fragment aggregate shape or numeric result rejected".into(),
                ));
            }
            Ok(output)
        })();
        if result.is_err() {
            self.failed = true;
            self.workers.clear();
        } else {
            self.sequence += 1;
        }
        result
    }
}

/// Entry point for generated workers; default stdin/stdout contain protocol frames.
/// Explicit `--tcp-test-listen` enables one bounded loopback-only test connection.
pub fn fragment_elf_main(descriptor_json: &str, weights: &[u8]) {
    let result = (|| {
        let args = std::env::args_os()
            .skip(1)
            .map(|arg| {
                arg.into_string()
                    .map_err(|_| Error("fragment argument must be UTF-8".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let descriptor: Descriptor = serde_json::from_str(descriptor_json)?;
        match args.as_slice() {
            [] => wire::serve(
                &descriptor,
                weights,
                std::io::stdin().lock(),
                std::io::stdout().lock(),
            ),
            [flag, address] if flag == "--tcp-test-listen" => {
                let address = crate::native_fragment_tcp::parse_loopback_endpoint(address, true)?;
                crate::native_fragment_tcp::serve_listener(&descriptor, weights, address)
            }
            _ => Err(Error(
                "fragment worker expects no arguments or --tcp-test-listen ADDRESS".into(),
            )),
        }
    })();
    if let Err(error) = result {
        eprintln!("bml-fragment: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fragments() -> Vec<FragmentExecutable> {
        (0..2)
            .map(|i| FragmentExecutable {
                file_name: format!("fragment_{i}"),
                sha256: "ab".repeat(32),
                descriptor: Descriptor {
                    version: 1,
                    source_sha256: "cd".repeat(32),
                    tensor_name: "output.weight".into(),
                    row_start: i * 2,
                    row_end: (i + 1) * 2,
                    total_rows: 4,
                    cols: 3,
                    weights_sha256: "ef".repeat(32),
                },
            })
            .collect()
    }

    #[test]
    fn coverage_is_ordered_complete_and_bound_to_one_model() {
        assert_eq!(layout(&fragments()).unwrap(), (4, 3));
        for case in 0..8 {
            let mut f = fragments();
            match case {
                0 => f[1].descriptor.row_start = 1,
                1 => f[1].descriptor.row_start = 3,
                2 => f[1].descriptor.row_end = 3,
                3 => f[1].descriptor.source_sha256 = "01".repeat(32),
                4 => f[1].descriptor.cols = 4,
                5 => f[1].file_name = "../fragment_1".into(),
                6 => f[1].descriptor.version = 2,
                _ => f[1].descriptor.tensor_name = "other.weight".into(),
            }
            assert!(layout(&f).is_err(), "case {case}");
        }
    }

    #[test]
    fn aggregate_weight_budget_is_checked_before_spawning() {
        let mut f = fragments();
        for (index, item) in f.iter_mut().enumerate() {
            item.descriptor.row_start = index * 16384;
            item.descriptor.row_end = (index + 1) * 16384;
            item.descriptor.total_rows = 32768;
            item.descriptor.cols = 1024;
            item.descriptor.validate_metadata().unwrap();
        }
        assert!(layout(&f).is_err()); // Individually legal 64 MiB + 64 MiB.
        for item in &mut f {
            item.descriptor.cols = 512;
        }
        assert_eq!(layout(&f).unwrap(), (32768, 512)); // Global 64 MiB inclusive.
    }
}
