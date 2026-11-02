//! Independent byte fixtures for normalized resources and autonomous ELF loading.

use bml_gguf::{decode_f32, GgufDataType, GgufReader};
use bml_inference::resources::{append_to_executable, write_model, write_model_file, Artifact};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Files(PathBuf);

impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "bml_resources_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn resources(&self) -> Vec<u8> {
        let source = self.path("source.gguf");
        std::fs::write(&source, fixture()).unwrap();
        let reader = GgufReader::open(source).unwrap();
        let mut bytes = Vec::new();
        write_model(&reader, &mut bytes).unwrap();
        bytes
    }

    fn rejected(&self, bytes: &[u8]) -> String {
        let path = self.path(&format!("bad-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&path, bytes).unwrap();
        match Artifact::open(path) {
            Ok(_) => panic!("malformed resource was accepted"),
            Err(error) => error.to_string(),
        }
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn string(value: &str) -> Vec<u8> {
    let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
    bytes.extend(value.as_bytes());
    bytes
}

fn array(kind: u32, values: Vec<Vec<u8>>) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend((values.len() as u64).to_le_bytes());
    bytes.extend(values.into_iter().flatten());
    bytes
}

fn fixture() -> Vec<u8> {
    let mut tokens = vec!["<unk>".to_owned(), "<s>".into(), "</s>".into(), "▁".into()];
    tokens.extend((0..256).map(|byte| format!("<0x{byte:02X}>")));
    let vocab = tokens.len() as u64;
    let mut metadata = vec![
        ("general.architecture", 8u32, string("llama")),
        ("tokenizer.ggml.model", 8, string("llama")),
        (
            "tokenizer.ggml.tokens",
            9,
            array(8, tokens.iter().map(|token| string(token)).collect()),
        ),
        (
            "tokenizer.ggml.scores",
            9,
            array(
                6,
                tokens
                    .iter()
                    .map(|_| (-1.0f32).to_le_bytes().to_vec())
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type",
            9,
            array(
                5,
                (0..tokens.len())
                    .map(|i| {
                        let kind: i32 = match i {
                            0 => 2,
                            1 | 2 => 3,
                            3 => 1,
                            _ => 6,
                        };
                        kind.to_le_bytes().to_vec()
                    })
                    .collect(),
            ),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            6,
            1e-5f32.to_le_bytes().to_vec(),
        ),
        ("llama.rope.freq_base", 6, 10000.0f32.to_le_bytes().to_vec()),
    ];
    for (name, value) in [
        ("llama.embedding_length", 32u32),
        ("llama.block_count", 1),
        ("llama.attention.head_count", 4),
        ("llama.attention.head_count_kv", 2),
        ("llama.feed_forward_length", 32),
        ("llama.context_length", 8),
        ("llama.rope.dimension_count", 8),
        ("tokenizer.ggml.bos_token_id", 1),
        ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.unknown_token_id", 0),
    ] {
        metadata.push((name, 4, value.to_le_bytes().to_vec()));
    }
    for (name, value) in [
        ("tokenizer.ggml.add_bos_token", true),
        ("tokenizer.ggml.add_eos_token", false),
        ("tokenizer.ggml.add_space_prefix", true),
        ("tokenizer.ggml.remove_extra_whitespaces", false),
        ("tokenizer.ggml.byte_fallback", true),
    ] {
        metadata.push((name, 7, vec![u8::from(value)]));
    }
    let quantized = |rows: u64, value: u8| {
        let mut bytes = Vec::new();
        for _ in 0..rows {
            bytes.extend(0x3800u16.to_le_bytes()); // Independent half encoding: 0.5.
            bytes.extend([value; 16]);
        }
        bytes
    };
    let norm = || {
        (0..32)
            .flat_map(|_| 1.0f32.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut tensors = vec![
        (
            "token_embd.weight",
            vec![32, vocab],
            2u32,
            quantized(vocab, 0xc2),
        ),
        ("output_norm.weight", vec![32], 0, norm()),
        ("blk.0.attn_norm.weight", vec![32], 0, norm()),
        ("blk.0.ffn_norm.weight", vec![32], 0, norm()),
    ];
    for (name, rows) in [
        ("blk.0.attn_q.weight", 32u64),
        ("blk.0.attn_k.weight", 16),
        ("blk.0.attn_v.weight", 16),
        ("blk.0.attn_output.weight", 32),
        ("blk.0.ffn_gate.weight", 32),
        ("blk.0.ffn_up.weight", 32),
        ("blk.0.ffn_down.weight", 32),
    ] {
        tensors.push((name, vec![32, rows], 2, quantized(rows, 0x88)));
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (name, kind, value) in metadata {
        bytes.extend(string(name));
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    let mut data = Vec::new();
    for (name, dims, kind, payload) in tensors {
        data.resize(data.len().div_ceil(32) * 32, 0);
        bytes.extend(string(name));
        bytes.extend((dims.len() as u32).to_le_bytes());
        for dimension in dims {
            bytes.extend(dimension.to_le_bytes());
        }
        bytes.extend(kind.to_le_bytes());
        bytes.extend((data.len() as u64).to_le_bytes());
        data.extend(payload);
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend(data);
    bytes
}

fn change_manifest(bytes: &[u8], change: impl FnOnce(&mut Value)) -> Vec<u8> {
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let old_data = (48 + len).div_ceil(32) * 32;
    let mut value: Value = serde_json::from_slice(&bytes[48..48 + len]).unwrap();
    change(&mut value);
    let json = serde_json::to_vec(&value).unwrap();
    let mut changed = b"BMLRS001".to_vec();
    changed.extend((json.len() as u64).to_le_bytes());
    changed.extend(Sha256::digest(&json));
    changed.extend(json);
    changed.resize(changed.len().div_ceil(32) * 32, 0);
    changed.extend(&bytes[old_data..]);
    changed
}

fn elf_header() -> Vec<u8> {
    let mut bytes = vec![0u8; 96];
    bytes[..4].copy_from_slice(b"\x7fELF");
    bytes[4..7].copy_from_slice(&[2, 1, 1]);
    bytes[16..18].copy_from_slice(&2u16.to_le_bytes());
    bytes[18..20].copy_from_slice(&62u16.to_le_bytes());
    bytes
}

fn write(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn normalized_resources_preserve_compressed_rows_and_model_identity() {
    let files = Files::new();
    let bytes = files.resources();
    let path = files.path("model.bml");
    write(&path, &bytes);
    let artifact = Artifact::open(&path).unwrap();
    assert_eq!(
        artifact.resource_sha256(),
        format!("{:x}", Sha256::digest(&bytes))
    );
    assert_eq!(
        artifact.manifest().source_sha256,
        format!("{:x}", Sha256::digest(fixture()))
    );
    assert_eq!(artifact.manifest().limits.max_context_tokens, 8);
    assert_eq!(artifact.manifest().limits.kv_cache_bytes, 1024);
    let model = artifact.model().unwrap();
    assert_eq!(artifact.manifest().limits.scratch_payload_bytes_bound, 5472);
    assert_eq!(artifact.manifest().producer.package, "bml-inference");
    assert_eq!(
        artifact.manifest().producer.version,
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(artifact.manifest().execution_plan.version, 1);
    assert_eq!(artifact.manifest().execution_plan.operations, model.plan());
    assert_eq!(model.config().embedding_length, 32);
    assert_eq!(model.tokenizer().vocab_size(), 260);
    let embedding = model
        .tensors()
        .iter()
        .find(|tensor| tensor.name == "token_embd.weight")
        .unwrap();
    assert_eq!(embedding.dtype, GgufDataType::Q4_0);
    assert_eq!(embedding.data.len(), 260 * 18);
    assert_eq!(embedding.dims, [32, 260]);
    let first = decode_f32(embedding.dtype, &embedding.data[..18], 32).unwrap();
    assert_eq!(&first[..16], &[-3.0; 16]);
    assert_eq!(&first[16..], &[2.0; 16]);
    assert!(bytes.starts_with(b"BMLRS001"));
    assert!(!bytes.windows(4).any(|window| window == b"GGUF"));
}

#[test]
fn rejects_corrupt_truncated_or_oversized_resource_files() {
    let files = Files::new();
    let bytes = files.resources();
    for prefix in [0, 7, 15, 47, 48, bytes.len() - 1] {
        files.rejected(&bytes[..prefix]);
    }
    let mut changed = bytes.clone();
    changed[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(files.rejected(&changed).contains("manifest"));
    for at in [16, 48, bytes.len() - 1] {
        let mut changed = bytes.clone();
        changed[at] ^= 1;
        assert!(files.rejected(&changed).contains("checksum"));
    }
    let mut changed = bytes.clone();
    changed.push(0);
    assert!(files.rejected(&changed).contains("trailing"));
}

#[test]
fn rejects_rehashed_bad_ranges_shapes_types_duplicates_and_limits() {
    let files = Files::new();
    let bytes = files.resources();
    for (field, value) in [
        ("offset", serde_json::json!(u64::MAX)),
        ("length", serde_json::json!(u64::MAX)),
        ("dims", serde_json::json!([u64::MAX, u64::MAX])),
        ("dims", serde_json::json!([0, 260])),
        ("dims", serde_json::json!([31, 260])),
        ("dtype", serde_json::json!(12)),
        ("sha256", serde_json::json!("xyz")),
    ] {
        files.rejected(&change_manifest(&bytes, |manifest| {
            manifest["tensors"][0][field] = value;
        }));
    }
    files.rejected(&change_manifest(&bytes, |manifest| {
        manifest["tensors"][1]["offset"] = 0.into();
    }));
    files.rejected(&change_manifest(&bytes, |manifest| {
        manifest["tensors"][1]["name"] = manifest["tensors"][0]["name"].clone();
    }));
    files.rejected(&change_manifest(&bytes, |manifest| {
        manifest["limits"]["kv_cache_bytes"] = 1.into();
    }));
    files.rejected(&change_manifest(&bytes, |manifest| {
        manifest["limits"]["scratch_payload_bytes_bound"] = 1.into();
    }));
    files.rejected(&change_manifest(&bytes, |manifest| {
        manifest["config"]["head_count"] = 0.into();
    }));
    for field in [
        "format_version",
        "target",
        "abi",
        "profile",
        "source_sha256",
    ] {
        files.rejected(&change_manifest(&bytes, |manifest| {
            manifest[field] = serde_json::json!("unsupported");
        }));
    }
}

#[test]
fn rejects_unrecognized_execution_semantics_and_build_identity() {
    let files = Files::new();
    let bytes = files.resources();
    for (field, value) in [
        ("version", serde_json::json!(2)),
        ("activation_dtype", serde_json::json!("f16")),
        ("reduction_dtype", serde_json::json!("f32")),
        ("operations", serde_json::json!(["Embedding", "Logits"])),
    ] {
        files.rejected(&change_manifest(&bytes, |manifest| {
            manifest["execution_plan"][field] = value;
        }));
    }
    for (field, value) in [
        (
            "source_revision",
            serde_json::json!("/home/private/project"),
        ),
        (
            "rustc_version",
            serde_json::json!("rustc /home/private/rustc"),
        ),
        ("package", serde_json::json!("unknown-runtime")),
        ("version", serde_json::json!("../private")),
    ] {
        files.rejected(&change_manifest(&bytes, |manifest| {
            manifest["producer"][field] = value;
        }));
    }
    for section in ["config", "tokenizer", "execution_plan", "producer"] {
        files.rejected(&change_manifest(&bytes, |manifest| {
            manifest[section]["unknown_parameter"] = true.into();
        }));
    }
}

#[test]
fn resource_identity_covers_valid_changes_even_with_the_same_source_label() {
    let files = Files::new();
    let original = files.resources();
    let changed = change_manifest(&original, |manifest| {
        manifest["config"]["rms_norm_eps"] = serde_json::json!(0.0002);
    });
    let first = files.path("first.bml");
    let second = files.path("second.bml");
    write(&first, &original);
    write(&second, &changed);
    let first = Artifact::open(first).unwrap();
    let second = Artifact::open(second).unwrap();
    assert_eq!(
        first.manifest().source_sha256,
        second.manifest().source_sha256
    );
    assert_ne!(first.resource_sha256(), second.resource_sha256());
}

#[test]
fn executable_contains_normalized_model_and_needs_neither_source_file() {
    let files = Files::new();
    let resources = files.path("model.bml");
    let resource_bytes = files.resources();
    write(&resources, &resource_bytes);
    let launcher = files.path("launcher");
    write(&launcher, &elf_header());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&launcher, std::fs::Permissions::from_mode(0o751)).unwrap();
    }
    let binary = files.path("standalone");
    append_to_executable(&launcher, &resources, &binary).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&binary).unwrap().permissions().mode() & 0o777,
            0o751
        );
    }
    // Remove our own source files before opening the resulting resource appendix.
    std::fs::remove_file(resources).unwrap();
    std::fs::remove_file(launcher).unwrap();
    std::fs::remove_file(files.path("source.gguf")).unwrap();
    let artifact = Artifact::open(&binary).unwrap();
    assert_eq!(
        artifact.resource_sha256(),
        format!("{:x}", Sha256::digest(&resource_bytes))
    );
    assert_eq!(artifact.model().unwrap().config().vocab_size, 260);
    let bytes = std::fs::read(&binary).unwrap();
    let footer = bytes.len() - 64;
    assert_eq!(&bytes[footer..footer + 8], b"BMLAOT01");
    let offset = u64::from_le_bytes(bytes[footer + 16..footer + 24].try_into().unwrap()) as usize;
    assert_eq!(offset, 4096);
    assert_eq!(&bytes[offset..offset + 8], b"BMLRS001");
}

#[test]
fn failed_export_leaves_no_destination_or_temporary_and_allows_retry() {
    let files = Files::new();
    let source = files.path("source.gguf");
    let output = files.path("model.bml");
    let mut invalid = fixture();
    let architecture = invalid
        .windows(5)
        .position(|bytes| bytes == b"llama")
        .unwrap();
    invalid[architecture..architecture + 5].copy_from_slice(b"other");
    write(&source, &invalid);
    {
        let reader = GgufReader::open(&source).unwrap();
        assert!(write_model_file(&reader, &output).is_err());
    }
    assert!(!output.exists());
    assert_eq!(std::fs::read_dir(&files.0).unwrap().count(), 1);
    write(&source, &fixture());
    let reader = GgufReader::open(&source).unwrap();
    write_model_file(&reader, &output).unwrap();
    Artifact::open(&output).unwrap();
    let original = std::fs::read(&output).unwrap();
    assert!(write_model_file(&reader, &output).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), original);
    assert_eq!(std::fs::read_dir(&files.0).unwrap().count(), 2);
}

#[test]
fn concurrent_exports_publish_exactly_one_complete_model() {
    let files = Files::new();
    let source = files.path("source.gguf");
    let output = files.path("model.bml");
    write(&source, &fixture());
    let barrier = std::sync::Barrier::new(2);
    let results = std::thread::scope(|scope| {
        let export = || {
            let reader = GgufReader::open(&source).unwrap();
            barrier.wait();
            write_model_file(&reader, &output).is_ok()
        };
        let first = scope.spawn(export);
        let second = scope.spawn(export);
        [first.join().unwrap(), second.join().unwrap()]
    });
    assert_eq!(results.into_iter().filter(|&ok| ok).count(), 1);
    Artifact::open(&output).unwrap();
    assert_eq!(std::fs::read_dir(&files.0).unwrap().count(), 2);
}

#[test]
fn rejects_invalid_aot_footer_and_never_overwrites_an_existing_output() {
    let files = Files::new();
    let resources = files.path("model.bml");
    write(&resources, &files.resources());
    let launcher = files.path("launcher");
    write(&launcher, &elf_header());
    let binary = files.path("standalone");
    append_to_executable(&launcher, &resources, &binary).unwrap();
    let original = std::fs::read(&binary).unwrap();
    assert!(append_to_executable(&launcher, &resources, &binary).is_err());
    assert_eq!(std::fs::read(&binary).unwrap(), original);
    let footer = original.len() - 64;
    for at in [
        footer,
        footer + 8,
        footer + 16,
        footer + 24,
        footer + 32,
        4096,
    ] {
        let mut changed = original.clone();
        changed[at] ^= 1;
        files.rejected(&changed);
    }
    let mut changed = original.clone();
    changed[footer + 16..footer + 24].copy_from_slice(&u64::MAX.to_le_bytes());
    files.rejected(&changed);
    files.rejected(&original[..original.len() - 1]);
    let mut changed = original.clone();
    changed.push(0);
    files.rejected(&changed);
    write(&launcher, b"not an ELF");
    assert!(append_to_executable(&launcher, &resources, files.path("invalid")).is_err());
    assert!(!files.path("invalid").exists());
}
