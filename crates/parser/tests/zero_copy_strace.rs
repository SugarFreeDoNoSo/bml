//! mmap ownership check and opt-in syscall instrumentation of the parser.
use bml_parser::{create_gguf_with_metadata, GgufParser};
#[test]
fn parser_reads_tensor_data_from_mmap() {
    let path = create_gguf_with_metadata();
    let parser = GgufParser::open(&path).unwrap();
    let data = parser.tensor_data(&parser.tensor_infos()[0]).unwrap();
    let offset = data.as_ptr() as usize - parser.bytes().as_ptr() as usize;
    assert_eq!(data.as_ptr(), parser.bytes()[offset..].as_ptr());
    assert_eq!(data.len(), 32);
    assert!(offset + data.len() <= parser.bytes().len());
    for (i, b) in data.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(f32::from_le_bytes(*b), i as f32);
    }
    std::fs::remove_file(path).unwrap();
}
#[test]
#[ignore = "requires Linux, strace and ptrace permission"]
fn strace_shows_mmap_not_read() {
    let path = create_gguf_with_metadata();
    let output = std::process::Command::new("strace")
        .args(["-f", "-P"])
        .arg(&path)
        .args(["-e", "trace=read,readv,pread64,mmap", "--"])
        .arg(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "traced_parser_child"])
        .env("BML_TRACE_FIXTURE", &path)
        .output()
        .expect("strace must be installed");
    std::fs::remove_file(path).unwrap();
    let log = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "strace failed: {log}");
    assert!(log.contains("mmap("), "missing tensor mmap: {log}");
    for syscall in ["read(", "readv(", "pread64("] {
        assert!(!log.contains(syscall), "unexpected tensor read: {log}");
    }
}
#[test]
#[ignore = "child of strace_shows_mmap_not_read"]
fn traced_parser_child() {
    let path = std::env::var("BML_TRACE_FIXTURE").expect("run through strace parent");
    let p = GgufParser::open(path).unwrap();
    let d = p.tensor_data(&p.tensor_infos()[0]).unwrap();
    assert_eq!(f32::from_le_bytes(d[28..32].try_into().unwrap()), 7.0);
}
