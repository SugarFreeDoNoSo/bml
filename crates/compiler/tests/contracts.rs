//! Public regressions and independent numeric oracles for the six audited contracts.
use bml_compiler::gguf_compiler::read_tensor_f32;
use bml_compiler::{fragment_program, linearize, BmlGraph, HashConsRegistry, RpnOp, RpnProgram};
use bml_domain::{BMLTransformer, EvalContext, NodeSoA};
use bml_parser::GgufParser;
use proptest::prelude::*;

struct Fixture(std::path::PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
fn string(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}
fn fixture(a: u32, t: u32, d: &[u64], data: &[u8]) -> Fixture {
    fixture_named(a, t, d, data, "w")
}
fn fixture_named(alignment: u32, dtype: u32, dims: &[u64], data: &[u8], name: &str) -> Fixture {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ID: AtomicUsize = AtomicUsize::new(0);
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    string(&mut b, "general.alignment");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&alignment.to_le_bytes());
    string(&mut b, name);
    b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&dtype.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    let a = alignment.max(1) as usize;
    b.resize(b.len().div_ceil(a) * a, 0);
    b.extend_from_slice(data);
    let path = std::env::temp_dir().join(format!(
        "bml_contract_{}_{}.gguf",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, b).unwrap();
    Fixture(path)
}
fn q4_block() -> Vec<u8> {
    let mut d = 0x3c00u16.to_le_bytes().to_vec();
    d.extend((0u8..16).map(|j| j | ((15 - j) << 4)));
    d
}

#[test]
fn shared_value_is_not_replaced_by_the_stack_top() {
    let mut s = NodeSoA::new();
    let x = s.push_var(0);
    let y = s.push_var(1);
    let child = s.push_bml(x, y);
    let root = s.push_bml(child, x);
    assert_eq!(
        linearize(&s, root).evaluate_with_ctx(&EvalContext::new(&[1.0, 4.0], &[])),
        1.0
    );
}
#[test]
fn folded_constants_survive_compilation_and_serialization() {
    let mut r = HashConsRegistry::new();
    let one = r.one();
    let root = r.bml(one, one);
    let s = r.into_soa();
    let p = linearize(&s, root);
    assert_eq!(p.evaluate(), 2.0);
    let graph = fragment_program(&p, 32);
    let restored = BmlGraph::deserialize(&graph.serialize(), 32).unwrap();
    assert_eq!(restored.evaluate(0.0), 2.0);
    let mut runtime = bml_runtime::Runtime::new(64, 4);
    assert_eq!(runtime.execute(&p), 2.0);
    assert_eq!(runtime.execute_graph(&restored), 2.0);
}
#[test]
fn positive_product_below_one() {
    let mut t = BMLTransformer::new();
    let two = t.two();
    let half = t.recip(two);
    let root = t.mul(half, two);
    let s = t.into_soa();
    let nodes: Vec<_> = (0..s.len() as u32).map(|id| s.get(id)).collect();
    let result = bml_domain::ast::evaluate(&nodes, root, &EvalContext::new(&[], &[]));
    assert!((result - 1.0).abs() < 1e-12, "{result}");
}
#[test]
fn gguf_uses_declared_alignment() {
    for a in [8, 24, 32, 64, 128] {
        let data: Vec<_> = [1.0f32, -2.5]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let f = fixture(a, 0, &[2], &data);
        let p = GgufParser::open(&f.0).unwrap();
        assert_eq!(read_tensor_f32(&p, "w").unwrap(), [1.0, -2.5]);
        assert_eq!(p.tensor_data(&p.tensor_infos()[0]).unwrap().len(), 8);
    }
}
#[test]
fn q4_nibbles_are_two_halves() {
    let f = fixture(32, 2, &[32, 2], &q4_block().repeat(2));
    let p = GgufParser::open(&f.0).unwrap();
    let expected: Vec<f32> = (-8..8).chain((-8..8).rev()).map(|v| v as f32).collect();
    assert_eq!(read_tensor_f32(&p, "w").unwrap(), expected.repeat(2));
}
#[test]
fn incomplete_quantized_blocks_are_rejected() {
    for (dtype, size) in [(2, 18), (8, 34)] {
        for len in [0, 1, size - 1] {
            let f = fixture(32, dtype, &[32], &vec![0; len]);
            match GgufParser::open(&f.0) {
                Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData),
                Ok(p) => assert!(read_tensor_f32(&p, "w").is_none()),
            }
        }
    }
}
#[test]
fn gguf_rejects_invalid_alignment_and_shapes() {
    for (a, dims) in [
        (0, vec![32]),
        (3, vec![32]),
        (32, vec![0]),
        (32, vec![u64::MAX, 2]),
    ] {
        let f = fixture(a, 0, &dims, &[0; 128]);
        assert!(GgufParser::open(&f.0).is_err());
    }
    let f = fixture(32, 2, &[16, 2], &[0; 18]);
    assert!(GgufParser::open(&f.0).is_err());
}
#[test]
fn q8_signed_values_and_exact_slice() {
    let mut d = 0x3800u16.to_le_bytes().to_vec();
    d.extend((-16i8..16).map(|v| v as u8));
    d.extend([255; 10]);
    let f = fixture(32, 8, &[32], &d);
    let p = GgufParser::open(&f.0).unwrap();
    assert_eq!(p.tensor_data(&p.tensor_infos()[0]).unwrap().len(), 34);
    assert_eq!(
        read_tensor_f32(&p, "w").unwrap(),
        (-16..16).map(|v| v as f32 * 0.5).collect::<Vec<_>>()
    );
}
#[test]
fn graph_v1_is_readable_and_truncated_v2_is_rejected() {
    let mut v1 = [
        bml_compiler::BMLGRAPH_MAGIC.to_le_bytes(),
        1u32.to_le_bytes(),
        1u32.to_le_bytes(),
        1u32.to_le_bytes(),
    ]
    .concat();
    v1.push(0);
    assert_eq!(BmlGraph::deserialize(&v1, 32).unwrap().evaluate(0.0), 1.0);
    let mut p = RpnProgram::new();
    p.ops.push(RpnOp::Const(0));
    p.const_pool.push(3.25);
    let bytes = fragment_program(&p, 32).serialize();
    for end in 0..bytes.len() {
        assert!(BmlGraph::deserialize(&bytes[..end], 32).is_err());
    }
}
#[test]
fn alignment_handles_header_boundaries() {
    for a in [24, 32, 64] {
        for len in [1, 7, 38, 39, 63, 70] {
            let name = "w".repeat(len);
            let f = fixture_named(a, 0, &[1], &3.5f32.to_le_bytes(), &name);
            let p = GgufParser::open(&f.0).unwrap();
            assert_eq!(read_tensor_f32(&p, &name).unwrap(), [3.5]);
        }
    }
}
#[test]
fn quantization_scale_is_applied_and_f16_is_supported() {
    for (bits, scale) in [(0x3800u16, 0.5), (0xc000, -2.0), (0, 0.0)] {
        let mut d = q4_block();
        d[..2].copy_from_slice(&bits.to_le_bytes());
        let f = fixture(32, 2, &[32], &d);
        let p = GgufParser::open(&f.0).unwrap();
        let expected: Vec<f32> = (-8..8)
            .chain((-8..8).rev())
            .map(|x| x as f32 * scale)
            .collect();
        assert_eq!(read_tensor_f32(&p, "w").unwrap(), expected);
    }
    let d: Vec<_> = [0x3c00u16, 0xc000, 0x0001, 0x8000]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let f = fixture(32, 1, &[4], &d);
    let p = GgufParser::open(&f.0).unwrap();
    let v = read_tensor_f32(&p, "w").unwrap();
    assert_eq!(v, [1.0, -2.0, 2.0f32.powi(-24), -0.0]);
    assert_eq!(v[3].to_bits(), (-0.0f32).to_bits());
}
#[test]
fn explicit_constant_pool_and_initial_capacity() {
    let mut r = HashConsRegistry::with_capacity(128);
    assert!(r.soa().kinds.capacity() >= 128);
    let one = r.one();
    let two = r.bml(one, one);
    let v = r.var(0);
    let root = r.bml(v, two);
    let (s, pool) = r.into_soa_and_pool();
    assert!(s.const_pool.is_empty());
    assert_eq!(pool, [2.0]);
    assert_eq!(
        linearize(&s, root).evaluate_with_ctx(&EvalContext::new(&[2.0], &pool)),
        3.0
    );
}
#[test]
fn gguf_refuses_short_scalar_data_and_forged_tensor_ranges() {
    let f = fixture(32, 0, &[2], &[0; 7]);
    assert!(GgufParser::open(&f.0).is_err());
    let f = fixture(32, 0, &[2], &[0; 8]);
    let p = GgufParser::open(&f.0).unwrap();
    let valid = p.tensor_infos()[0].clone();
    let mut i = valid.clone();
    i.n_dims = 2;
    assert!(p.tensor_data(&i).is_none());
    let mut i = valid.clone();
    i.dims.clear();
    i.n_dims = 0;
    assert!(p.tensor_data(&i).is_none());
    let mut i = valid.clone();
    i.dims = vec![0];
    assert!(p.tensor_data(&i).is_none());
    let mut i = valid.clone();
    i.offset = u64::MAX;
    assert!(p.tensor_data(&i).is_none());
    let mut i = valid;
    i.dims = vec![u64::MAX];
    assert!(p.tensor_data(&i).is_none());
}
#[test]
fn unsupported_quantizers_are_metadata_only_but_shapes_must_be_valid() {
    let f = fixture(32, 3, &[32], &[]);
    let p = GgufParser::open(&f.0).unwrap();
    assert!(p.tensor_data(&p.tensor_infos()[0]).is_none());
    assert!(read_tensor_f32(&p, "w").is_none());
    for dims in [vec![0], vec![u64::MAX, 2]] {
        let f = fixture(32, 3, &dims, &[]);
        assert!(GgufParser::open(&f.0).is_err());
    }
}
#[test]
fn tensor_relative_offsets_are_aligned() {
    let f = fixture(32, 0, &[1], &[0; 128]);
    let mut b = std::fs::read(&f.0).unwrap();
    let k = 24 + 8 + "general.alignment".len() + 4 + 4 + 8 + 1 + 4 + 8 + 4;
    b[k..k + 8].copy_from_slice(&1u64.to_le_bytes());
    std::fs::write(&f.0, &b).unwrap();
    assert!(GgufParser::open(&f.0).is_err());
    b[k..k + 8].copy_from_slice(&32u64.to_le_bytes());
    std::fs::write(&f.0, &b).unwrap();
    assert!(GgufParser::open(&f.0).is_ok());
}
#[test]
fn metadata_arrays_and_malformed_lengths() {
    let f = fixture(32, 0, &[1], &[0; 4]);
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    string(&mut b, "values");
    b.extend_from_slice(&9u32.to_le_bytes());
    let k = b.len();
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&3u64.to_le_bytes());
    b.extend([4, 5, 6]);
    std::fs::write(&f.0, &b).unwrap();
    let p = GgufParser::open(&f.0).unwrap();
    match p.get_metadata("values").unwrap() {
        bml_parser::GgufMetadataValue::Array(_, v) => {
            use bml_parser::GgufMetadataValue::U8;
            assert!(matches!(v.as_slice(), [U8(4), U8(5), U8(6)]), "{v:?}");
        }
        _ => panic!("array expected"),
    };
    drop(p);
    b[k..k + 4].copy_from_slice(&9u32.to_le_bytes());
    std::fs::write(&f.0, &b).unwrap();
    assert!(GgufParser::open(&f.0).is_err());
    b[k..k + 4].copy_from_slice(&0u32.to_le_bytes());
    b[k + 4..k + 12].copy_from_slice(&u64::MAX.to_le_bytes());
    std::fs::write(&f.0, &b).unwrap();
    assert!(GgufParser::open(&f.0).is_err());
}
proptest! {
    #[test]
    fn arithmetic_matches_native_in_tested_domain(x in 0.25f64..2.0,y in 0.25f64..2.0) {
        let mut t=BMLTransformer::new();let a=t.var(0);let b=t.var(1);let root=t.mul(a,b);
        let actual=linearize(t.soa(),root).evaluate_with_ctx(&EvalContext::new(&[x,y],&[]));prop_assert!((actual-x*y).abs()<1e-9,"x={x} y={y} got {actual}");
    }
    #[test]
    fn signed_addition_matches_native(x in -2.0f64..2.0,y in -2.0f64..2.0) {
        let mut t=BMLTransformer::new();let a=t.var(0);let b=t.var(1);let root=t.add(a,b);
        let actual=linearize(t.soa(),root).evaluate_with_ctx(&EvalContext::new(&[x,y],&[]));prop_assert!((actual-(x+y)).abs()<1e-9,"x={x} y={y} got {actual}");
    }
    #[test]
    fn shared_dags_match_independent_numeric_oracle(edges in prop::collection::vec((0usize..40,0usize..40),1..40),x in 0.25f64..1.5) {
        let mut s=NodeSoA::new();s.push_var(0);s.push_one();s.push_zero();let mut values=vec![x,1.0,0.0];let mut root=0;
        for(a,b) in edges {let a=a%values.len();let b=b%values.len();root=s.push_bml(a as u32,b as u32);values.push(values[a].exp2()-values[b].log2());}
        let p=linearize(&s,root);let inputs=[x];let ctx=EvalContext::new(&inputs,&[]);let expected=*values.last().unwrap();let actual=p.evaluate_with_ctx(&ctx);
        prop_assert!(if expected.is_nan(){actual.is_nan()}else{actual.to_bits()==expected.to_bits()},"expected {expected}, got {actual}");prop_assert!(p.len()<=s.len()*5);
        let mut hot=bml_runtime::HotLoop::with_capacity(s.len()+2);let actual=hot.execute_with_ctx(&p,&ctx);
        prop_assert!(if expected.is_nan(){actual.is_nan()}else{actual.to_bits()==expected.to_bits()},"expected {expected}, got {actual}");
    }
}
