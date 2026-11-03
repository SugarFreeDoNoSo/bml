//! Reader-bound views with an explicit optional row-major tensor bridge.
use crate::{GgufDataType, GgufError, GgufTensorInfo};

/// Borrowed metadata and encoded bytes belonging to one GGUF reader.
///
/// Obtain a view with [`crate::GgufReader::tensor_view`]. Its private fields
/// prevent pairing a descriptor from one reader with another reader's bytes.
/// GGML axis zero varies fastest; [`Self::ggml_dims`] preserves that convention.
/// No data is transposed, dequantized or allocated by creating a view.
///
/// A view cannot outlive the reader that owns its mapping and descriptor:
///
/// ```compile_fail
/// use bml_gguf::GgufReader;
/// let view = {
///     let reader = GgufReader::open("model.gguf").unwrap();
///     reader.tensor_view("weights").unwrap()
/// };
/// println!("{:?}", view.bytes());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct GgufTensorView<'a> {
    info: &'a GgufTensorInfo,
    data: &'a [u8],
}

impl<'a> GgufTensorView<'a> {
    pub(crate) fn new(info: &'a GgufTensorInfo, data: &'a [u8]) -> Self {
        Self { info, data }
    }

    /// Tensor name borrowed from the reader's descriptor.
    pub fn name(&self) -> &'a str {
        &self.info.name
    }

    /// Stored GGML dtype, unchanged by inspecting the view.
    pub fn dtype(&self) -> GgufDataType {
        self.info.data_type
    }

    /// GGML dimensions, with the first axis contiguous (fastest varying).
    pub fn ggml_dims(&self) -> &'a [u64] {
        &self.info.dims
    }

    /// Exact encoded range borrowed from the reader, excluding inter-tensor padding.
    ///
    /// These are bytes, not an aligned native `f32` slice. Endianness and
    /// quantization are handled only by an explicit decoding operation.
    pub fn bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Allocate and decode to f32 while preserving the GGML flat element order.
    ///
    /// Supports F32, F16, Q4_0, Q8_0 and Q6_K; other stored types produce
    /// [`GgufError::UnsupportedDataType`]. This does not attach a row-major shape.
    pub fn decode_f32(&self) -> Result<Vec<f32>, GgufError> {
        let elements = self.info.dims.iter().try_fold(1usize, |n, &dimension| {
            n.checked_mul(usize::try_from(dimension).map_err(|_| GgufError::SizeOverflow)?)
                .ok_or(GgufError::SizeOverflow)
        })?;
        crate::decode_f32(self.dtype(), self.data, elements)
    }

    /// Decode one axis-zero-contiguous row without expanding the whole tensor.
    /// Outer axes are flattened in GGML order. The reader has already validated
    /// row/block alignment; the row index is checked before slicing encoded bytes.
    pub fn decode_row_f32(&self, row: usize) -> Result<Vec<f32>, GgufError> {
        let invalid = || GgufError::InvalidTensorRange {
            name: self.name().into(),
        };
        let (&columns, outer) = self.info.dims.split_first().ok_or_else(invalid)?;
        let columns = usize::try_from(columns).map_err(|_| GgufError::SizeOverflow)?;
        let rows = outer.iter().try_fold(1usize, |n, &dimension| {
            n.checked_mul(usize::try_from(dimension).map_err(|_| GgufError::SizeOverflow)?)
                .ok_or(GgufError::SizeOverflow)
        })?;
        if columns == 0 || row >= rows || !self.data.len().is_multiple_of(rows) {
            return Err(invalid());
        }
        let stride = self.data.len() / rows;
        let start = row.checked_mul(stride).ok_or(GgufError::SizeOverflow)?;
        crate::decode_f32(self.dtype(), &self.data[start..start + stride], columns)
    }

    /// Convert GGML axis order to a row-major shape by reversing dimensions.
    ///
    /// Available with feature `tensor`. For example, GGML `[3, 2]` describes
    /// two rows of three contiguous values, so the row-major shape is `[2, 3]`.
    /// This allocates shape metadata only; it does not transpose the values.
    #[cfg(feature = "tensor")]
    pub fn row_major_shape(&self) -> Result<bml_domain::tensor::Shape, GgufError> {
        let dims = self
            .ggml_dims()
            .iter()
            .rev()
            .map(|&dimension| usize::try_from(dimension).map_err(|_| GgufError::SizeOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(bml_domain::tensor::Shape::new(dims)?)
    }

    /// Explicitly allocate/decode an owned row-major CPU tensor.
    ///
    /// Available with feature `tensor`. Dimensions are reversed from GGML
    /// order and values retain their original flat order: there is no implicit
    /// transpose or axis shuffle. Quantized storage is dequantized to f32.
    /// The returned tensor owns its buffer and can outlive the reader.
    #[cfg(feature = "tensor")]
    pub fn to_tensor_f32(&self) -> Result<bml_domain::tensor::Tensor<f32>, GgufError> {
        let shape = self.row_major_shape()?;
        let values = self.decode_f32()?;
        Ok(bml_domain::tensor::Tensor::from_vec(shape, values)?)
    }
}
