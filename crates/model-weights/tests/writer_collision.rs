//! Separate test executable gives the temporary-name sequence a deterministic
//! initial value, without hooks in production or interference from other tests.
use bml_model_weights::*;

#[test]
fn temporary_name_collision_is_retried_without_touching_existing_file() {
    let dir = std::env::temp_dir().join(format!("bml-writer-collision-{}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(dir.clone());
    let collision = dir.join(format!(".bmlgguf-{}-0.tmp", std::process::id()));
    std::fs::write(&collision, b"existing unrelated data").unwrap();
    let identity = ToolIdentity {
        name: "fixture".into(),
        version: "1".into(),
        revision: "test".into(),
    };
    let metadata = Metadata {
        schema: 1,
        profile: "bml-exact-decoded-f32-v1".into(),
        source_gguf_version: 3,
        converter: identity.clone(),
        codec: identity,
        limits: ConversionLimits::default(),
        gguf: vec![],
    };
    let writer = WeightWriter::create(
        dir.join("out.bmlgguf"),
        metadata,
        Digest([0; 32]),
        Default::default(),
    )
    .unwrap();
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
    assert_eq!(
        std::fs::read(&collision).unwrap(),
        b"existing unrelated data"
    );
    writer.abort().unwrap();
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    assert_eq!(
        std::fs::read(collision).unwrap(),
        b"existing unrelated data"
    );
}
