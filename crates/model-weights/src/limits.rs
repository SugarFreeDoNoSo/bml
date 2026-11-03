use crate::{Error, ExactSynthesisOptions, Resource, HEADER_BYTES};

/// Independent inclusive reader limits; file metadata cannot enlarge them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadLimits {
    pub max_file_bytes: u64,
    pub max_metadata_bytes: u64,
    pub max_metadata_entries: u64,
    pub max_metadata_array_elements: u64,
    pub max_name_bytes: u64,
    pub max_string_bytes: u64,
    pub max_tensors: u64,
    pub max_blocks: u64,
    pub max_index_bytes: u64,
    pub max_block_terms: u64,
    pub max_block_bytes: u64,
    pub max_weight_bits: u64,
    pub max_weight_depth: u64,
    pub max_total_terms: u64,
}
impl Default for ReadLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 64 << 30,
            max_metadata_bytes: 64 << 20,
            max_metadata_entries: 65_536,
            max_metadata_array_elements: 4_194_304,
            max_name_bytes: 4096,
            max_string_bytes: 1 << 20,
            max_tensors: 65_536,
            max_blocks: 1_048_576,
            max_index_bytes: 128 << 20,
            max_block_terms: 65_536,
            max_block_bytes: 8 << 20,
            max_weight_bits: 1_048_576,
            max_weight_depth: 4096,
            max_total_terms: 4_294_967_296,
        }
    }
}

/// Check an inclusive limit. None represents an unrepresentable required size.
pub fn check_limit(resource: Resource, limit: u64, required: Option<u64>) -> Result<(), Error> {
    if required.is_some_and(|n| n <= limit) {
        return Ok(());
    }
    Err(Error::Limit {
        resource,
        limit,
        required,
        location: None,
    })
}

impl ReadLimits {
    /// Validate configuration without allocating or reading data.
    pub fn validate(&self) -> Result<(), Error> {
        check_limit(
            Resource::MaxFileBytes,
            self.max_file_bytes,
            Some(HEADER_BYTES),
        )?;
        check_limit(Resource::MaxBlockBytes, self.max_block_bytes, Some(25))?;
        for (resource, limit) in [
            (Resource::MaxMetadataBytes, self.max_metadata_bytes),
            (Resource::MaxMetadataEntries, self.max_metadata_entries),
            (
                Resource::MaxMetadataArrayElements,
                self.max_metadata_array_elements,
            ),
            (Resource::MaxNameBytes, self.max_name_bytes),
            (Resource::MaxStringBytes, self.max_string_bytes),
            (Resource::MaxTensors, self.max_tensors),
            (Resource::MaxBlocks, self.max_blocks),
            (Resource::MaxIndexBytes, self.max_index_bytes),
            (Resource::MaxBlockTerms, self.max_block_terms),
            (Resource::MaxWeightBits, self.max_weight_bits),
            (Resource::MaxWeightDepth, self.max_weight_depth),
            (Resource::MaxTotalTerms, self.max_total_terms),
        ] {
            check_limit(resource, limit, Some(1))?;
        }
        for bytes in [
            self.max_metadata_bytes,
            self.max_index_bytes,
            self.max_block_bytes,
        ] {
            check_limit(Resource::MaxFileBytes, self.max_file_bytes, Some(bytes))?;
        }
        Ok(())
    }
}

/// Requested conversion budgets recorded as provenance in metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConversionLimits {
    pub max_weight_bits: u64,
    pub max_weight_depth: u64,
    pub max_synthesis_steps_per_weight: u64,
    pub max_synthesis_steps_total: u64,
    pub max_block_terms: u64,
    pub max_block_bytes: u64,
    pub max_output_bytes: u64,
    pub max_resident_bytes: u64,
    pub max_cache_bytes: u64,
}
impl Default for ConversionLimits {
    fn default() -> Self {
        Self {
            max_weight_bits: 1_048_576,
            max_weight_depth: 4096,
            max_synthesis_steps_per_weight: 4096,
            max_synthesis_steps_total: 1_000_000,
            max_block_terms: 65_536,
            max_block_bytes: 8 << 20,
            max_output_bytes: 64 << 30,
            max_resident_bytes: 256 << 20,
            max_cache_bytes: 0,
        }
    }
}
impl ConversionLimits {
    pub fn validate(&self) -> Result<(), Error> {
        for (resource, limit) in [
            (Resource::MaxWeightBits, self.max_weight_bits),
            (Resource::MaxWeightDepth, self.max_weight_depth),
            (
                Resource::MaxSynthesisStepsPerWeight,
                self.max_synthesis_steps_per_weight,
            ),
            (
                Resource::MaxSynthesisStepsTotal,
                self.max_synthesis_steps_total,
            ),
            (Resource::MaxBlockTerms, self.max_block_terms),
            (Resource::MaxResidentBytes, self.max_resident_bytes),
        ] {
            check_limit(resource, limit, Some(1))?;
        }
        check_limit(Resource::MaxBlockBytes, self.max_block_bytes, Some(25))?;
        check_limit(
            Resource::MaxOutputBytes,
            self.max_output_bytes,
            Some(HEADER_BYTES),
        )?;
        check_limit(
            Resource::MaxOutputBytes,
            self.max_output_bytes,
            Some(self.max_block_bytes),
        )?;
        check_limit(
            Resource::MaxResidentBytes,
            self.max_resident_bytes,
            Some(self.max_cache_bytes),
        )?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriteOptions {
    pub limits: ReadLimits,
    pub conversion: ConversionLimits,
}
impl WriteOptions {
    pub fn validate(&self) -> Result<(), Error> {
        self.limits.validate()?;
        self.conversion.validate()
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConversionOptions {
    pub write: WriteOptions,
    pub synthesis: ExactSynthesisOptions,
}
impl ConversionOptions {
    pub fn validate(&self) -> Result<(), Error> {
        self.write.validate()?;
        check_limit(
            Resource::MaxWeightBits,
            self.synthesis.max_bits as u64,
            Some(1),
        )?;
        check_limit(
            Resource::MaxSynthesisStepsPerWeight,
            self.synthesis.max_steps as u64,
            Some(1),
        )
    }
    /// Effective per-weight limits; global counters are enforced by the converter.
    pub fn effective_synthesis(&self) -> Result<ExactSynthesisOptions, Error> {
        self.validate()?;
        let mut options = self.synthesis;
        let bits = (options.max_bits as u64)
            .min(self.write.limits.max_weight_bits)
            .min(self.write.conversion.max_weight_bits);
        let steps =
            (options.max_steps as u64).min(self.write.conversion.max_synthesis_steps_per_weight);
        // Both minima are bounded by existing usize-valued options.
        options.max_bits = bits as usize;
        options.max_steps = steps as usize;
        Ok(options)
    }
}
