#![doc = include_str!("../README.md")]

mod blocks;
#[cfg(target_arch = "aarch64")]
pub mod neon;

pub use crate::blocks::{SimdBlock, SimdBlocks};

/// Extension methods for traversing `u8` slices with fixed-step, fixed-load semantics.
pub trait SimdTraverseExt {
    /// Returns an iterator over step-aligned blocks.
    ///
    /// Every yielded block guarantees that `LOAD` bytes starting at its offset are
    /// in bounds within the original slice.
    ///
    /// # Panics
    ///
    /// Panics if `STEP == 0` or `LOAD < STEP`.
    fn simd_blocks<const STEP: usize, const LOAD: usize>(&self) -> SimdBlocks<'_, STEP, LOAD>;

    /// Returns the suffix after the final yielded `LOAD`-byte window.
    ///
    /// If the iterator would yield no blocks, this returns the full original slice.
    ///
    /// # Panics
    ///
    /// Panics if `STEP == 0` or `LOAD < STEP`.
    fn simd_tail<const STEP: usize, const LOAD: usize>(&self) -> &[u8];

    /// Returns both the iterator and its matching non-overlapping tail.
    ///
    /// The returned tail is derived from the same internal traversal arithmetic as
    /// the iterator so their semantics cannot drift.
    ///
    /// # Panics
    ///
    /// Panics if `STEP == 0` or `LOAD < STEP`.
    fn simd_partition<const STEP: usize, const LOAD: usize>(
        &self,
    ) -> (SimdBlocks<'_, STEP, LOAD>, &[u8]);
}

impl SimdTraverseExt for [u8] {
    fn simd_blocks<const STEP: usize, const LOAD: usize>(&self) -> SimdBlocks<'_, STEP, LOAD> {
        self.simd_partition::<STEP, LOAD>().0
    }

    fn simd_tail<const STEP: usize, const LOAD: usize>(&self) -> &[u8] {
        self.simd_partition::<STEP, LOAD>().1
    }

    fn simd_partition<const STEP: usize, const LOAD: usize>(
        &self,
    ) -> (SimdBlocks<'_, STEP, LOAD>, &[u8]) {
        blocks::partition_slice::<STEP, LOAD>(self)
    }
}
