//! Provision prevalidated shard bundles over the system SSH/SCP clients.
use std::{
    collections::HashSet,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};
const MAX_PLAN: u64 = 1024 * 1024;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    version: u32,
    artifact: String,
    artifact_sha256: String,
    workers: Vec<Assignment>,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Assignment {
    index: usize,
    layer_start: u32,
    layer_end: u32,
    directory: String,
    shard_sha256: String,
}

struct Snapshot(PathBuf);
impl Drop for Snapshot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn copy_bounded(source: &Path, destination: &Path, max_bytes: u64) -> Result<(), String> {
    let file = std::fs::File::open(source).map_err(|e| e.to_string())?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        return Err("deployment input exceeds its byte budget".into());
    }
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|e| e.to_string())?;
    let count = std::io::copy(&mut file.take(max_bytes.saturating_add(1)), &mut output)
        .map_err(|e| e.to_string())?;
    if count > max_bytes {
        return Err("deployment input grew past its byte budget".into());
    }
    Ok(())
}

fn destination(value: &str) -> Result<(&str, &str), String> {
    let (host, path) = value
        .split_once(':')
        .ok_or("destination must be host:/absolute/path")?;
    if host.is_empty()
        || host.starts_with('-')
        || host.bytes().filter(|&b| b == b'@').count() > 1
        || !host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'@'))
        || host.starts_with('@')
        || host.ends_with('@')
    {
        return Err("invalid SSH destination host".into());
    }
    if !path.starts_with('/')
        || path == "/"
        || path.ends_with('/')
        || !path
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'.' | b'_' | b'-'))
        || path
            .split('/')
            .skip(1)
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err("destination must be a safe absolute directory path".into());
    }
    Ok((host, path))
}

/// Verify the entire prepared plan locally before creating any remote directory.
/// Host keys and credentials are delegated to SSH; no interactive prompt is used.
pub fn deploy(prepared: &Path, nodes: &[String], max_bytes: u64) -> Result<(), String> {
    let file = std::fs::File::open(prepared.join("plan.json")).map_err(|e| e.to_string())?;
    if !file.metadata().map_err(|e| e.to_string())?.is_file() {
        return Err("plan must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_PLAN + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_PLAN {
        return Err("plan exceeds byte budget".into());
    }
    let plan: Plan = serde_json::from_slice(&bytes).map_err(|e| format!("invalid plan: {e}"))?;
    if plan.version != 1 || plan.artifact != "artifact" {
        return Err("invalid prepared plan".into());
    }
    let identity = plan.artifact_sha256.as_str();
    let artifact = prepared.join("artifact");
    let loaded =
        bml_compiler::symbolic_artifact::load_directory_authenticated(&artifact, identity)?;
    let entries = &plan.workers;
    if entries.is_empty() || entries.len() != nodes.len() {
        return Err("destination count must match prepared worker count".into());
    }
    let ranges = super::layer_ranges(loaded.manifest.config.n_layers, entries.len())?;
    if ranges.len() != entries.len() {
        return Err("worker count exceeds layer count".into());
    }
    let file_counts: Vec<_> = loaded
        .manifest
        .kernels
        .iter()
        .map(|kernel| kernel.programs.len())
        .collect();
    drop(loaded);
    let mut seen = HashSet::new();
    let mut validated = Vec::with_capacity(entries.len());
    for (index, ((entry, range), node)) in entries.iter().zip(ranges).zip(nodes).enumerate() {
        let name = format!("shard-{index}");
        let hash = entry.shard_sha256.as_str();
        if entry.index != index
            || entry.directory != name
            || entry.layer_start != range.start
            || entry.layer_end != range.end
        {
            return Err("invalid prepared assignment".into());
        }
        let (host, path) = destination(node)?;
        if !seen.insert((host.to_owned(), path.to_owned())) {
            return Err("duplicate destination".into());
        }
        let shard = prepared.join(&name);
        bml_compiler::symbolic_weight_shard::verify_layer_weights(
            &shard,
            hash,
            identity,
            range.clone(),
            max_bytes,
        )?;
        let shard = std::fs::canonicalize(shard).map_err(|e| e.to_string())?;
        validated.push((
            index,
            host.to_owned(),
            path.to_owned(),
            shard,
            hash.to_owned(),
            range,
        ));
    }
    // Stage only known files in a private directory. Validate the staged bytes
    // before any connection, so unrelated files and source changes cannot be sent.
    let temp = std::env::temp_dir().join(format!("bml-deploy-plan-{}", std::process::id()));
    let mut builder = std::fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(&temp)
        .map_err(|e| format!("create plan snapshot: {e}"))?;
    let snapshot = Snapshot(temp);
    let plan_file = snapshot.0.join("plan.json");
    std::fs::write(&plan_file, bytes).map_err(|e| e.to_string())?;
    let staged_artifact = snapshot.0.join("artifact");
    std::fs::create_dir(&staged_artifact).map_err(|e| e.to_string())?;
    copy_bounded(
        &artifact.join("symbolic.bmlgraph.json"),
        &staged_artifact.join("symbolic.bmlgraph.json"),
        16 * 1024 * 1024,
    )?;
    for (kernel, count) in file_counts.into_iter().enumerate() {
        for program in 0..count {
            let name = format!("kernel_{kernel}_{program}.bmlgraph");
            copy_bounded(
                &artifact.join(&name),
                &staged_artifact.join(&name),
                8 * 1024 * 1024,
            )?;
        }
    }
    bml_compiler::symbolic_artifact::load_directory_authenticated(&staged_artifact, identity)?;
    for (index, _, _, shard, hash, range) in &validated {
        let staged = snapshot.0.join(format!("shard-{index}"));
        std::fs::create_dir(&staged).map_err(|e| e.to_string())?;
        copy_bounded(
            &shard.join("weights.json"),
            &staged.join("weights.json"),
            16 * 1024 * 1024,
        )?;
        copy_bounded(
            &shard.join("weights.bin"),
            &staged.join("weights.bin"),
            max_bytes,
        )?;
        bml_compiler::symbolic_weight_shard::verify_layer_weights(
            &staged,
            hash,
            identity,
            range.clone(),
            max_bytes,
        )?;
    }
    let artifact = std::fs::canonicalize(&staged_artifact).map_err(|e| e.to_string())?;
    let plan_file = std::fs::canonicalize(&plan_file).map_err(|e| e.to_string())?;
    for (index, host, path, _, _, _) in validated {
        let shard = std::fs::canonicalize(snapshot.0.join(format!("shard-{index}")))
            .map_err(|e| e.to_string())?;
        let mkdir = Command::new("ssh")
            .args(["-o", "BatchMode=yes", "--", &host, "mkdir", "--", &path])
            .status()
            .map_err(|e| {
                format!(
                    "start ssh for worker {index}: {e}; earlier workers may already be provisioned"
                )
            })?;
        if !mkdir.success() {
            return Err(format!("worker {index}: remote directory creation failed; earlier workers may already be provisioned"));
        }
        let remote = format!("{host}:{path}");
        let copy = Command::new("scp")
            .args(["-r", "-o", "BatchMode=yes"])
            .arg(&plan_file)
            .arg(&artifact)
            .arg(&shard)
            .arg(&remote)
            .status()
            .map_err(|e| format!("start scp for worker {index}: {e}; remote directory {remote} remains and earlier workers may already be provisioned"))?;
        if !copy.success() {
            return Err(format!("worker {index}: copy failed; remote directory {remote} remains and earlier workers may already be provisioned"));
        }
        println!("worker={index} destination={remote}");
    }
    Ok(())
}
