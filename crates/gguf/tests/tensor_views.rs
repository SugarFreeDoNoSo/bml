//! Newly reconstructed contract tests after the local-only #19 worktree was lost.
//! Fixtures are independent byte builders, not the production GGUF writer.
use bml_gguf::{GgufDataType, GgufError, GgufReader};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct StoredTensor<'a> {
    name: &'a str,
    dims: &'a [u64],
    dtype: u32,
    bytes: Vec<u8>,
}

struct Fixture(std::path::PathBuf);

impl Fixture {
    fn new(tensors: &[StoredTensor<'_>]) -> Self {
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend((tensors.len() as u64).to_le_bytes());
        header.extend(0u64.to_le_bytes());
        let mut payload = Vec::new();
        for tensor in tensors {
            payload.resize(payload.len().div_ceil(32) * 32, 0xa5);
            header.extend((tensor.name.len() as u64).to_le_bytes());
            header.extend(tensor.name.as_bytes());
            header.extend((tensor.dims.len() as u32).to_le_bytes());
            for &dimension in tensor.dims {
                header.extend(dimension.to_le_bytes());
            }
            header.extend(tensor.dtype.to_le_bytes());
            header.extend((payload.len() as u64).to_le_bytes());
            payload.extend(&tensor.bytes);
        }
        header.resize(header.len().div_ceil(32) * 32, 0x5a);
        header.extend(payload);
        header.extend([0x7e; 13]); // Must never leak into a tensor's byte view.
        let path = std::env::temp_dir().join(format!(
            "bml_reconstructed_views_{}_{}.gguf",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, header).unwrap();
        Self(path)
    }

    fn open(&self) -> GgufReader {
        GgufReader::open(&self.0).unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn views_borrow_the_readers_actual_metadata_and_exact_distinct_ranges() {
    let fixture = Fixture::new(&[
        StoredTensor {
            name: "matrix",
            dims: &[3, 2],
            dtype: 0,
            bytes: f32_bytes(&[1., 2., 3., 4., 5., 6.]),
        },
        StoredTensor {
            name: "half",
            dims: &[2],
            dtype: 1,
            bytes: vec![0, 0x3c, 0, 0xc0],
        },
    ]);
    let reader = fixture.open();
    let matrix = reader.tensor_view("matrix").unwrap();
    let half = reader.tensor_view("half").unwrap();
    for (name, view, length) in [("matrix", matrix, 24), ("half", half, 4)] {
        let descriptor = reader.find_tensor(name).unwrap();
        let bytes = reader.tensor_data(descriptor).unwrap();
        assert_eq!(view.name(), name);
        assert_eq!(view.name().as_ptr(), descriptor.name.as_ptr());
        assert_eq!(view.ggml_dims(), descriptor.dims);
        assert_eq!(view.ggml_dims().as_ptr(), descriptor.dims.as_ptr());
        assert_eq!(view.bytes().as_ptr(), bytes.as_ptr());
        assert_eq!(view.bytes(), bytes);
        assert_eq!(view.bytes().len(), length);
        assert!(format!("{view:?}").contains(name));
    }
    assert_eq!(matrix.dtype(), GgufDataType::F32);
    assert_eq!(half.dtype(), GgufDataType::F16);
    assert_ne!(matrix.bytes().as_ptr(), half.bytes().as_ptr());
    assert_eq!(matrix.decode_f32().unwrap(), [1., 2., 3., 4., 5., 6.]);
    assert_eq!(half.decode_f32().unwrap(), [1., -2.]);
    assert_eq!(
        reader.read_tensor_f32("matrix").unwrap(),
        [1., 2., 3., 4., 5., 6.]
    );
}

#[test]
fn missing_and_unknown_layout_views_return_distinct_typed_errors() {
    let fixture = Fixture::new(&[StoredTensor {
        name: "unknown_layout",
        dims: &[32],
        dtype: 6, // Q5_0 is identified, but its byte layout is not supported here.
        bytes: vec![0; 22],
    }]);
    let reader = fixture.open();
    assert!(matches!(
        reader.tensor_view("absent"),
        Err(GgufError::TensorNotFound(name)) if name == "absent"
    ));
    assert!(matches!(
        reader.tensor_view("unknown_layout"),
        Err(GgufError::UnsupportedDataType(GgufDataType::Q5_0))
    ));
}

#[test]
fn a_known_scalar_layout_can_be_inspected_without_claiming_f32_conversion() {
    let encoded = [i32::MIN.to_le_bytes(), i32::MAX.to_le_bytes()].concat();
    let fixture = Fixture::new(&[StoredTensor {
        name: "integer",
        dims: &[2],
        dtype: 26,
        bytes: encoded.clone(),
    }]);
    let reader = fixture.open();
    let view = reader.tensor_view("integer").unwrap();
    assert_eq!(view.dtype(), GgufDataType::I32);
    assert_eq!(view.ggml_dims(), [2]);
    assert_eq!(view.bytes(), encoded);
    assert!(matches!(
        view.decode_f32(),
        Err(GgufError::UnsupportedDataType(GgufDataType::I32))
    ));
}

#[cfg(feature = "tensor")]
mod tensor {
    use super::*;
    use bml_domain::tensor::{matmul, TensorError};

    #[test]
    fn asymmetric_matrix_conversion_reverses_axes_without_transposing_values() {
        let fixture = Fixture::new(&[StoredTensor {
            name: "matrix",
            dims: &[3, 2],
            dtype: 0,
            bytes: f32_bytes(&[1., 2., 3., 4., 5., 6.]),
        }]);
        let reader = fixture.open();
        let view = reader.tensor_view("matrix").unwrap();
        assert_eq!(view.ggml_dims(), [3, 2]);
        assert_eq!(view.row_major_shape().unwrap().dims(), [2, 3]);
        let tensor = view.to_tensor_f32().unwrap();
        assert_eq!(tensor.shape().dims(), [2, 3]);
        assert_eq!(tensor.as_slice(), [1., 2., 3., 4., 5., 6.]);
        assert_eq!(*tensor.view().get(&[1, 2]).unwrap(), 6.);
        assert_ne!(
            tensor.as_slice().as_ptr().cast::<u8>(),
            view.bytes().as_ptr()
        );
    }

    #[test]
    fn rectangular_matmul_matches_independent_known_answer() {
        let fixture = Fixture::new(&[
            StoredTensor {
                name: "a",
                dims: &[3, 2],
                dtype: 0,
                bytes: f32_bytes(&[1., 2., 3., 4., 5., 6.]),
            },
            StoredTensor {
                name: "b",
                dims: &[2, 3],
                dtype: 0,
                bytes: f32_bytes(&[7., 8., 9., 10., 11., 12.]),
            },
        ]);
        let reader = fixture.open();
        let a = reader.tensor_view("a").unwrap().to_tensor_f32().unwrap();
        let b = reader.tensor_view("b").unwrap().to_tensor_f32().unwrap();
        assert_eq!(a.shape().dims(), [2, 3]);
        assert_eq!(b.shape().dims(), [3, 2]);
        let product = matmul(&a.view(), &b.view()).unwrap();
        assert_eq!(product.shape().dims(), [2, 2]);
        assert_eq!(product.as_slice(), [58., 64., 139., 154.]);
    }

    #[test]
    fn rank_one_three_and_four_keep_flat_order_and_reverse_every_axis() {
        struct Layout<'a> {
            ggml: &'a [u64],
            row_major: &'a [usize],
            index: &'a [usize],
            flat: usize,
        }
        let layouts = [
            Layout {
                ggml: &[5],
                row_major: &[5],
                index: &[3],
                flat: 3,
            },
            Layout {
                ggml: &[2, 3, 4],
                row_major: &[4, 3, 2],
                index: &[1, 2, 1],
                flat: 11,
            },
            Layout {
                ggml: &[2, 3, 4, 5],
                row_major: &[5, 4, 3, 2],
                index: &[1, 2, 1, 0],
                flat: 38,
            },
        ];
        for Layout {
            ggml,
            row_major,
            index,
            flat,
        } in layouts
        {
            let count: usize = row_major.iter().product();
            let expected: Vec<f32> = (0..count).map(|n| n as f32).collect();
            let fixture = Fixture::new(&[StoredTensor {
                name: "ranked",
                dims: ggml,
                dtype: 0,
                bytes: f32_bytes(&expected),
            }]);
            let reader = fixture.open();
            let view = reader.tensor_view("ranked").unwrap();
            let tensor = view.to_tensor_f32().unwrap();
            assert_eq!(view.ggml_dims(), ggml);
            assert_eq!(tensor.shape().dims(), row_major);
            assert_eq!(tensor.as_slice(), expected);
            assert_eq!(*tensor.view().get(index).unwrap(), flat as f32);
        }
    }

    #[test]
    fn quantized_blocks_decode_to_row_major_shape_without_reordering_rows() {
        let mut q4 = vec![0, 0x3c];
        q4.extend([0xf0; 16]);
        q4.extend([0, 0x38]);
        q4.extend([0x79; 16]);
        let mut q8 = vec![0, 0x40];
        q8.extend([0xff; 32]);
        q8.extend([0, 0xbc]);
        q8.extend([3; 32]);
        let fixture = Fixture::new(&[
            StoredTensor {
                name: "q4",
                dims: &[32, 2],
                dtype: 2,
                bytes: q4,
            },
            StoredTensor {
                name: "q8",
                dims: &[32, 2],
                dtype: 8,
                bytes: q8,
            },
        ]);
        let reader = fixture.open();
        let q4 = reader.tensor_view("q4").unwrap().to_tensor_f32().unwrap();
        assert_eq!(q4.shape().dims(), [2, 32]);
        assert_eq!(&q4.as_slice()[..16], &[-8.; 16]);
        assert_eq!(&q4.as_slice()[16..32], &[7.; 16]);
        assert_eq!(&q4.as_slice()[32..48], &[0.5; 16]);
        assert_eq!(&q4.as_slice()[48..], &[-0.5; 16]);
        let q8 = reader.tensor_view("q8").unwrap().to_tensor_f32().unwrap();
        assert_eq!(q8.shape().dims(), [2, 32]);
        assert_eq!(&q8.as_slice()[..32], &[-2.; 32]);
        assert_eq!(&q8.as_slice()[32..], &[-3.; 32]);
    }

    #[test]
    fn explicit_f16_conversion_owns_values_after_reader_is_dropped() {
        let fixture = Fixture::new(&[StoredTensor {
            name: "half",
            dims: &[2, 3],
            dtype: 1,
            bytes: vec![0, 0x3c, 0, 0xc0, 0, 0x38, 0, 0x44, 0, 0x80, 0, 0x42],
        }]);
        let owned = {
            let reader = fixture.open();
            reader.tensor_view("half").unwrap().to_tensor_f32().unwrap()
        };
        assert_eq!(owned.shape().dims(), [3, 2]);
        assert_eq!(owned.as_slice(), [1., -2., 0.5, 4., -0., 3.]);
        assert_eq!(owned.as_slice()[4].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn unsupported_scalar_conversion_is_typed_not_an_implicit_cast() {
        let fixture = Fixture::new(&[StoredTensor {
            name: "integers",
            dims: &[1, 2],
            dtype: 26,
            bytes: vec![1, 0, 0, 0, 2, 0, 0, 0],
        }]);
        let reader = fixture.open();
        let view = reader.tensor_view("integers").unwrap();
        assert_eq!(view.row_major_shape().unwrap().dims(), [2, 1]);
        assert!(matches!(
            view.to_tensor_f32(),
            Err(GgufError::UnsupportedDataType(GgufDataType::I32))
        ));
    }

    #[test]
    fn tensor_bridge_errors_preserve_the_underlying_error_source() {
        let error = GgufError::from(TensorError::LengthMismatch {
            expected: 3,
            actual: 2,
        });
        assert!(matches!(
            error,
            GgufError::Tensor(TensorError::LengthMismatch {
                expected: 3,
                actual: 2
            })
        ));
        let source = std::error::Error::source(&error).unwrap();
        assert_eq!(
            source.downcast_ref::<TensorError>(),
            Some(&TensorError::LengthMismatch {
                expected: 3,
                actual: 2
            })
        );
    }
}
