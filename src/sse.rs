//! x86_64 SSE2 helpers built on top of [`SimdBlock`].
//!
//! These helpers consume the traversal crate's in-bounds load guarantee without
//! changing the architecture-neutral traversal API.

use crate::SimdBlock;
use core::arch::x86_64::{
    __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_set1_epi8,
};

/// Loads a 16-byte traversal block into an SSE2 register.
///
/// This is the lowest-level bridge from the crate's traversal semantics into
/// architecture-specific byte classification code.
pub fn load_u8x16<const STEP: usize>(block: SimdBlock<'_, STEP, 16>) -> __m128i {
    unsafe {
        // SAFETY: `SimdBlock<'_, STEP, 16>` guarantees that 16 bytes starting
        // at `block.as_ptr()` are within the original slice. `_mm_loadu_si128`
        // accepts unaligned input.
        load_ptr_u8x16(block.as_ptr())
    }
}

/// Returns a bitmask of lanes equal to `needle`.
///
/// Bit `i` in the returned mask corresponds to byte lane `i` in the loaded
/// 16-byte block.
pub fn match_byte_mask_u8x16<const STEP: usize>(block: SimdBlock<'_, STEP, 16>, needle: u8) -> u16 {
    unsafe {
        // SAFETY: `SimdBlock<'_, STEP, 16>` guarantees that 16 bytes starting
        // at `block.as_ptr()` are within the original slice.
        match_byte_mask_ptr_u8x16(block.as_ptr(), needle)
    }
}

/// Returns `true` if any byte in the 16-byte block equals `needle`.
pub fn any_byte_eq_u8x16<const STEP: usize>(block: SimdBlock<'_, STEP, 16>, needle: u8) -> bool {
    match_byte_mask_u8x16(block, needle) != 0
}

pub(crate) unsafe fn load_ptr_u8x16(ptr: *const u8) -> __m128i {
    unsafe {
        // SAFETY: the caller guarantees that `ptr` starts a readable 16-byte
        // region. `_mm_loadu_si128` accepts unaligned input.
        _mm_loadu_si128(ptr.cast::<__m128i>())
    }
}

pub(crate) unsafe fn match_byte_mask_ptr_u8x16(ptr: *const u8, needle: u8) -> u16 {
    let bytes = unsafe {
        // SAFETY: the caller guarantees that `ptr` starts a readable 16-byte
        // region.
        load_ptr_u8x16(ptr)
    };
    let needles = unsafe {
        // SAFETY: this module is only compiled on `x86_64`, where SSE2 is part
        // of the architectural baseline for these intrinsics.
        _mm_set1_epi8(needle as i8)
    };
    let matches = unsafe {
        // SAFETY: this module is only compiled on `x86_64`, where SSE2 is part
        // of the architectural baseline for these intrinsics.
        _mm_cmpeq_epi8(bytes, needles)
    };

    unsafe {
        // SAFETY: this module is only compiled on `x86_64`, where SSE2 is part
        // of the architectural baseline for these intrinsics.
        _mm_movemask_epi8(matches) as u16
    }
}
