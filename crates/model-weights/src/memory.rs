//! Shared conservative owned-memory accounting, distinct from process RSS.
use crate::{check_limit, Error, Resource};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub(crate) struct MemoryBudget {
    external: AtomicU64,
    peak: AtomicU64,
}

impl MemoryBudget {
    #[cfg(any(feature = "gguf-import", test))]
    pub(crate) fn set_external(&self, bytes: u64) {
        self.external.store(bytes, Ordering::Relaxed);
    }
    pub(crate) fn external(&self) -> u64 {
        self.external.load(Ordering::Relaxed)
    }
    pub(crate) fn check(&self, writer: u64, cap: u64) -> Result<(), Error> {
        let total = self.external().checked_add(writer);
        check_limit(Resource::MaxResidentBytes, cap, total)?;
        self.peak.fetch_max(total.unwrap(), Ordering::Relaxed);
        Ok(())
    }
    #[cfg(any(feature = "gguf-import", test))]
    pub(crate) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simultaneous_owners_share_inclusive_cap_and_peak() {
        let budget = MemoryBudget::default();
        budget.set_external(60);
        budget.check(40, 100).unwrap();
        assert!(budget.check(41, 100).is_err());
        assert_eq!(budget.peak(), 100);
        budget.set_external(0);
        budget.check(80, 100).unwrap();
        assert_eq!(budget.peak(), 100);
        budget.set_external(u64::MAX);
        assert!(matches!(
            budget.check(1, u64::MAX),
            Err(Error::Limit { required: None, .. })
        ));
    }
}
