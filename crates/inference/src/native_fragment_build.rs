//! Offline native ELF generation for a bounded output-head row partition.
//!
//! Each worker embeds only its own F32 rows. The coordinator deliberately keeps
//! the complete normalized model, including a duplicate of the output head.
//! This first vertical slice does not claim full model-memory partitioning.

use crate::model::Model;
use crate::native_fragment_protocol::Descriptor;
use crate::resources::{append_to_executable, write_model};
use crate::{Error, Result};
use bml_gguf::{GgufDataType, GgufReader};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::ops::Range;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

const TARGET: &str = "x86_64-unknown-linux-gnu";
const TOOLCHAIN: &str = "1.98.1";
const FLAGS: &str = "-Ctarget-cpu=x86-64\x1f-Ctarget-feature=-fma";

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_hash(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            return Ok(format!("{:x}", digest.finalize()));
        }
        digest.update(&buffer[..count]);
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn absolute_directory_path(path: &Path) -> Result<()> {
    if !path.is_absolute()
        || path.parent().is_none()
        || !path
            .components()
            .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
    {
        return Err(Error(
            "fragment output and target must be absolute non-root paths without traversal".into(),
        ));
    }
    let mut current = PathBuf::new();
    for part in path.components() {
        current.push(part.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(Error(
                    "fragment staging refuses symbolic-link path components".into(),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn partition(rows: usize, workers: usize) -> Result<Vec<Range<usize>>> {
    if rows == 0 || rows > 32768 || workers == 0 || workers > 8 || workers > rows {
        return Err(Error(
            "fragment partition requires 1..8 workers and 1..32768 rows".into(),
        ));
    }
    (0..workers)
        .map(|index| {
            let start = rows
                .checked_mul(index)
                .ok_or_else(|| Error("row overflow".into()))?
                / workers;
            let end = rows
                .checked_mul(index + 1)
                .ok_or_else(|| Error("row overflow".into()))?
                / workers;
            Ok(start..end)
        })
        .collect()
}

fn elf_amd64(path: &Path) -> Result<()> {
    let mut header = [0u8; 64];
    File::open(path)?.read_exact(&mut header)?;
    validate_elf_header(&header)
}

fn validate_elf_header(header: &[u8; 64]) -> Result<()> {
    let kind = u16::from_le_bytes([header[16], header[17]]);
    if &header[..4] != b"\x7fELF"
        || header[4] != 2
        || header[5] != 1
        || header[6] != 1
        || !matches!(kind, 2 | 3)
        || u16::from_le_bytes([header[18], header[19]]) != 62
        || u32::from_le_bytes(header[20..24].try_into().unwrap()) != 1
    {
        return Err(Error(
            "generated artifact is not a little-endian AMD64 ELF executable".into(),
        ));
    }
    Ok(())
}

fn copy_executable_new(source: &Path, destination: &Path) -> Result<()> {
    elf_amd64(source)?;
    let mut input = File::open(source)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    let expected = input.metadata()?.len();
    if std::io::copy(&mut input, &mut output)? != expected {
        return Err(Error("generated ELF changed during copy".into()));
    }
    output.set_permissions(input.metadata()?.permissions())?;
    output.sync_all()?;
    elf_amd64(destination)
}

fn source_snapshot(workspace: &Path) -> Result<BTreeMap<String, String>> {
    fn visit(workspace: &Path, path: &Path, hashes: &mut BTreeMap<String, String>) -> Result<()> {
        let metadata = fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() {
            return Err(Error("source snapshot refuses symbolic links".into()));
        }
        if metadata.is_dir() {
            for entry in fs::read_dir(path)? {
                visit(workspace, &entry?.path(), hashes)?;
            }
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            let relative = path
                .strip_prefix(workspace)
                .map_err(|error| Error(error.to_string()))?;
            hashes.insert(relative.to_string_lossy().into_owned(), file_hash(path)?);
        }
        Ok(())
    }
    let mut hashes = BTreeMap::new();
    for package in ["inference", "gguf"] {
        visit(
            workspace,
            &workspace.join("crates").join(package).join("src"),
            &mut hashes,
        )?;
        let relative = format!("crates/{package}/Cargo.toml");
        hashes.insert(relative.clone(), file_hash(&workspace.join(relative))?);
        let relative = format!("crates/{package}/build.rs");
        if workspace.join(&relative).exists() {
            hashes.insert(relative.clone(), file_hash(&workspace.join(relative))?);
        }
    }
    for relative in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        hashes.insert(relative.into(), file_hash(&workspace.join(relative))?);
    }
    Ok(hashes)
}

fn cargo_command(build: &Path, target: &Path) -> Command {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut command = Command::new(cargo);
    command
        .current_dir(build)
        .env("RUSTUP_TOOLCHAIN", TOOLCHAIN)
        .env("RUSTUP_AUTO_INSTALL", "0")
        .env("CARGO_TARGET_DIR", target)
        .env("CARGO_NET_OFFLINE", "true")
        .env("CARGO_INCREMENTAL", "0")
        .env("CARGO_PROFILE_RELEASE_DEBUG", "0")
        .env("RUSTFLAGS", "")
        .env("CARGO_ENCODED_RUSTFLAGS", FLAGS)
        .stdout(Stdio::null());
    command
}

fn checked_status(command: &mut Command, phase: &str) -> Result<()> {
    if !command.status()?.success() {
        return Err(Error(format!(
            "fragment {phase} failed; staging is preserved"
        )));
    }
    Ok(())
}

/// Compile a native F32 output-head partition and its autonomous coordinator.
///
/// `output` must be a new absolute directory. `target` is an explicit absolute
/// build-cache parent; a private child prevents collisions with other builds.
/// Builds are offline with the pinned toolchain and generic AMD64 without FMA.
/// Failed staging is retained and never reported as a successful artifact set.
pub fn compile(reader: &GgufReader, output: &Path, target: &Path, workers: usize) -> Result<Value> {
    if !cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
        return Err(Error("fragment AOT requires Linux x86_64 GNU".into()));
    }
    absolute_directory_path(output)?;
    absolute_directory_path(target)?;
    if output.exists() || fs::symlink_metadata(output).is_ok() {
        return Err(Error("fragment output already exists".into()));
    }
    if target == output || target.starts_with(output) || output.starts_with(target) {
        return Err(Error(
            "fragment output and build target must be separate trees".into(),
        ));
    }
    let model = Model::from_gguf(reader)?;
    let tensor = model.output_tensor();
    if tensor.dtype != GgufDataType::F32 || tensor.dims.len() != 2 {
        return Err(Error(
            "fragment AOT currently requires an F32 output head".into(),
        ));
    }
    let columns =
        usize::try_from(tensor.dims[0]).map_err(|_| Error("head width overflow".into()))?;
    let rows = usize::try_from(tensor.dims[1]).map_err(|_| Error("head row overflow".into()))?;
    let aggregate_macs = rows
        .checked_mul(columns)
        .filter(|&count| count <= crate::native_fragment_protocol::MAX_MACS)
        .ok_or_else(|| Error("full head exceeds the aggregate projection work limit".into()))?;
    let aggregate_weight_bytes = aggregate_macs
        .checked_mul(4)
        .filter(|&count| count <= crate::native_fragment_protocol::MAX_WEIGHT_BYTES)
        .ok_or_else(|| Error("full head exceeds the aggregate fragment weight limit".into()))?;
    if tensor.data.len() != aggregate_weight_bytes {
        return Err(Error("head byte count differs from dimensions".into()));
    }
    let ranges = partition(rows, workers)?;
    let row_bytes = columns
        .checked_mul(4)
        .ok_or_else(|| Error("head row bytes overflow".into()))?;
    let source_sha256 = sha256(reader.bytes());
    let mut fragments = Vec::new();
    for (index, range) in ranges.into_iter().enumerate() {
        let start = range
            .start
            .checked_mul(row_bytes)
            .ok_or_else(|| Error("head offset overflow".into()))?;
        let end = range
            .end
            .checked_mul(row_bytes)
            .ok_or_else(|| Error("head offset overflow".into()))?;
        let payload = tensor
            .data
            .get(start..end)
            .ok_or_else(|| Error("head rows outside tensor bytes".into()))?;
        let descriptor = Descriptor {
            version: 1,
            source_sha256: source_sha256.clone(),
            tensor_name: tensor.name.clone(),
            row_start: range.start,
            row_end: range.end,
            total_rows: rows,
            cols: columns,
            weights_sha256: sha256(payload),
        };
        descriptor.validate(payload)?;
        fragments.push((format!("fragment_{index}"), descriptor, payload));
    }
    let library = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = library
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| Error("BML source workspace unavailable".into()))?;
    let sources = source_snapshot(workspace)?;
    let rustc = Command::new("rustc")
        .env("RUSTUP_TOOLCHAIN", TOOLCHAIN)
        .env("RUSTUP_AUTO_INSTALL", "0")
        .args(["--version", "--verbose"])
        .output()?;
    let rustc_version =
        String::from_utf8(rustc.stdout).map_err(|error| Error(error.to_string()))?;
    if !rustc.status.success() || !rustc_version.starts_with("rustc 1.98.1 ") {
        return Err(Error(
            "fragment AOT requires the already-installed Rust 1.98.1 toolchain".into(),
        ));
    }
    fs::create_dir(output)?;
    let build = output.join("build");
    let run = output.join("run");
    fs::create_dir(&build)?;
    fs::create_dir(&run)?;
    fs::create_dir(build.join("src"))?;
    fs::create_dir(build.join("src/bin"))?;
    fs::create_dir(build.join("weights"))?;
    fs::create_dir_all(target)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error(error.to_string()))?
        .as_nanos();
    let isolated_target = target.join(format!("fragment-build-{}-{nonce}", std::process::id()));
    fs::create_dir(&isolated_target)?;
    let dependency_path = serde_json::to_string(&library.to_string_lossy())?;
    let cargo_manifest = format!(
        "[package]\nname = \"bml-native-fragment-generated\"\nversion = \"1.0.0\"\nedition = \"2021\"\n[workspace]\n[dependencies]\nbml-inference = {{ path = {dependency_path} }}\n[profile.release]\nopt-level = 3\nstrip = true\npanic = \"abort\"\ncodegen-units = 1\n"
    );
    write_new(&build.join("Cargo.toml"), cargo_manifest.as_bytes())?;
    write_new(
        &build.join("Cargo.lock"),
        &fs::read(workspace.join("Cargo.lock"))?,
    )?;
    write_new(
        &build.join("rust-toolchain.toml"),
        &fs::read(workspace.join("rust-toolchain.toml"))?,
    )?;
    let mut generated_sources = BTreeMap::new();
    for (name, descriptor, payload) in &fragments {
        write_new(&build.join("weights").join(format!("{name}.f32")), payload)?;
        let descriptor_json = serde_json::to_string(descriptor)?;
        let source = format!(
            "fn main() {{ bml_inference::native_fragments::fragment_elf_main({descriptor_json:?}, include_bytes!(\"../../weights/{name}.f32\")); }}\n"
        );
        write_new(
            &build.join("src/bin").join(format!("{name}.rs")),
            source.as_bytes(),
        )?;
        generated_sources.insert(format!("src/bin/{name}.rs"), sha256(source.as_bytes()));
    }
    // Lock resolution is local/offline. All actual compilation uses --locked.
    checked_status(
        cargo_command(&build, &isolated_target).args([
            "metadata",
            "--offline",
            "--format-version",
            "1",
        ]),
        "offline lock resolution",
    )?;
    let mut worker_build = cargo_command(&build, &isolated_target);
    worker_build.args([
        "build",
        "--release",
        "--locked",
        "--offline",
        "--target",
        TARGET,
    ]);
    for (name, _, _) in &fragments {
        worker_build.args(["--bin", name]);
    }
    checked_status(&mut worker_build, "worker compilation")?;
    let launchers = isolated_target.join(TARGET).join("release");
    let mut executables = Vec::new();
    let mut artifact_report = Vec::new();
    for (name, descriptor, payload) in &fragments {
        let destination = run.join(name);
        copy_executable_new(&launchers.join(name), &destination)?;
        let digest = file_hash(&destination)?;
        executables.push(json!({"file_name":name,"sha256":digest,"descriptor":descriptor}));
        artifact_report.push(json!({"file_name":name,"sha256":digest,"bytes":destination.metadata()?.len(),"weight_bytes":payload.len(),"descriptor":descriptor}));
    }
    let resource_path = build.join("model.bmlres");
    let mut resource = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&resource_path)?;
    let normalized = write_model(reader, &mut resource)?;
    resource.sync_all()?;
    let resource_sha256 = file_hash(&resource_path)?;
    let plan = model
        .plan()
        .iter()
        .map(|operation| format!("Operation::{operation:?}"))
        .collect::<Vec<_>>()
        .join(",\n");
    let executables_json = serde_json::to_string(&executables)?;
    let coordinator_source = format!(
        "use bml_inference::model::Operation;\nfn main() {{ bml_inference::driver::fragmented_embedded_main({resource_sha256:?}, &[{plan}], {executables_json:?}); }}\n"
    );
    write_new(
        &build.join("src/bin/coordinator.rs"),
        coordinator_source.as_bytes(),
    )?;
    generated_sources.insert(
        "src/bin/coordinator.rs".into(),
        sha256(coordinator_source.as_bytes()),
    );
    checked_status(
        cargo_command(&build, &isolated_target).args([
            "build",
            "--release",
            "--locked",
            "--offline",
            "--target",
            TARGET,
            "--bin",
            "coordinator",
        ]),
        "coordinator compilation",
    )?;
    let coordinator = run.join("coordinator");
    append_to_executable(launchers.join("coordinator"), &resource_path, &coordinator)?;
    elf_amd64(&coordinator)?;
    if source_snapshot(workspace)? != sources || sha256(reader.bytes()) != source_sha256 {
        return Err(Error(
            "source or model changed during fragment compilation; artifacts are not validated"
                .into(),
        ));
    }
    let report = json!({
        "format":"bml-native-head-fragments-v1",
        "target":TARGET,
        "cpu":"x86-64",
        "rustc_version":rustc_version,
        "build_flags":["--release","--locked","--offline","--target",TARGET,"-Ctarget-cpu=x86-64","-Ctarget-feature=-fma"],
        "source_sha256":source_sha256,
        "source_bytes":reader.bytes().len(),
        "normalized_source_sha256":normalized.source_sha256,
        "resource_sha256":resource_sha256,
        "resource_bytes":resource_path.metadata()?.len(),
        "head_rows":rows,"head_cols":columns,"workers":workers,
        "aggregate_head_macs_per_projection":aggregate_macs,
        "aggregate_fragment_weight_bytes":aggregate_weight_bytes,
        "fragments":artifact_report,
        "coordinator":{"file_name":"coordinator","sha256":file_hash(&coordinator)?,"bytes":coordinator.metadata()?.len()},
        "coordinator_retains_full_model_including_head":true,
        "generated_source_sha256":generated_sources,
        "generated_manifest_sha256":file_hash(&build.join("Cargo.toml"))?,
        "generated_toolchain_sha256":file_hash(&build.join("rust-toolchain.toml"))?,
        "generated_lock_sha256":file_hash(&build.join("Cargo.lock"))?,
        "source_files_sha256":sources,
        "execution":"not-run",
        "distributed_evidence":false
    });
    write_new(
        &output.join("build-report.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uneven_partition_has_exact_ordered_coverage() {
        assert_eq!(partition(7, 3).unwrap(), vec![0..2, 2..4, 4..7]);
        assert_eq!(partition(1, 1).unwrap(), vec![0..1]);
        for (rows, workers) in [(0, 1), (1, 0), (1, 2), (32769, 1), (32, 9)] {
            assert!(partition(rows, workers).is_err());
        }
    }

    #[test]
    fn staging_paths_require_explicit_non_root_absolute_locations() {
        for path in ["/", "relative", "/tmp/../target"] {
            assert!(absolute_directory_path(Path::new(path)).is_err());
        }
        assert!(absolute_directory_path(Path::new("/tmp/fragment-staging")).is_ok());
    }

    #[test]
    fn elf_check_rejects_other_architectures_and_object_files() {
        let mut header = [0; 64];
        header[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        header[16..18].copy_from_slice(&3u16.to_le_bytes());
        header[18..20].copy_from_slice(&62u16.to_le_bytes());
        header[20..24].copy_from_slice(&1u32.to_le_bytes());
        assert!(validate_elf_header(&header).is_ok());
        header[18..20].copy_from_slice(&183u16.to_le_bytes());
        assert!(validate_elf_header(&header).is_err());
        header[18..20].copy_from_slice(&62u16.to_le_bytes());
        header[16..18].copy_from_slice(&1u16.to_le_bytes());
        assert!(validate_elf_header(&header).is_err());
        header[16..18].copy_from_slice(&2u16.to_le_bytes());
        header[5] = 2;
        assert!(validate_elf_header(&header).is_err());
    }
}
