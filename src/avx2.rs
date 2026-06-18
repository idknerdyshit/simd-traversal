//! x86_64 AVX2 helpers built on top of [`SimdBlock`].
//!
//! The direct public helpers in this module are available when AVX2 is enabled
//! at compile time. With the `runtime-dispatch` feature, this module also
//! provides internal AVX2 routines used by [`crate::x86`].

#[cfg(any(target_feature = "avx2", doc))]
use crate::SimdBlock;
use core::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
};

/// Loads a 32-byte traversal block into an AVX2 register.
///
/// This is the lowest-level bridge from the crate's traversal semantics into
/// architecture-specific byte classification code.
#[cfg(any(target_feature = "avx2", doc))]
pub fn load_u8x32<const STEP: usize>(block: SimdBlock<'_, STEP, 32>) -> __m256i {
    unsafe {
        // SAFETY: `SimdBlock<'_, STEP, 32>` guarantees that 32 bytes starting
        // at `block.as_ptr()` are within the original slice. `_mm256_loadu_si256`
        // accepts unaligned input.
        load_ptr_u8x32(block.as_ptr())
    }
}

/// Returns a bitmask of lanes equal to `needle`.
///
/// Bit `i` in the returned mask corresponds to byte lane `i` in the loaded
/// 32-byte block.
#[cfg(any(target_feature = "avx2", doc))]
pub fn match_byte_mask_u8x32<const STEP: usize>(block: SimdBlock<'_, STEP, 32>, needle: u8) -> u32 {
    unsafe {
        // SAFETY: `SimdBlock<'_, STEP, 32>` guarantees that 32 bytes starting
        // at `block.as_ptr()` are within the original slice.
        match_byte_mask_ptr_u8x32(block.as_ptr(), needle)
    }
}

/// Returns `true` if any byte in the 32-byte block equals `needle`.
#[cfg(any(target_feature = "avx2", doc))]
pub fn any_byte_eq_u8x32<const STEP: usize>(block: SimdBlock<'_, STEP, 32>, needle: u8) -> bool {
    match_byte_mask_u8x32(block, needle) != 0
}

#[cfg(any(target_feature = "avx2", feature = "runtime-dispatch", doc))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn load_ptr_u8x32(ptr: *const u8) -> __m256i {
    unsafe {
        // SAFETY: the caller guarantees that `ptr` starts a readable 32-byte
        // region. `_mm256_loadu_si256` accepts unaligned input.
        _mm256_loadu_si256(ptr.cast::<__m256i>())
    }
}

#[cfg(any(target_feature = "avx2", feature = "runtime-dispatch", doc))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn match_byte_mask_ptr_u8x32(ptr: *const u8, needle: u8) -> u32 {
    let bytes = unsafe {
        // SAFETY: the caller guarantees that `ptr` starts a readable 32-byte
        // region.
        load_ptr_u8x32(ptr)
    };
    let needles = _mm256_set1_epi8(needle as i8);
    let matches = _mm256_cmpeq_epi8(bytes, needles);

    _mm256_movemask_epi8(matches) as u32
}
