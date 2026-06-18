//! Runtime-dispatched x86_64 helpers built on top of [`SimdBlock`].
//!
//! This module is available with the `runtime-dispatch` feature. It uses AVX2
//! when the running CPU supports it and falls back to SSE2 otherwise.

use crate::SimdBlock;

/// Returns a bitmask of lanes equal to `needle`.
///
/// Bit `i` in the returned mask corresponds to byte lane `i` in the 32-byte
/// block. The implementation uses AVX2 at runtime when available and otherwise
/// combines two SSE2 16-byte masks.
pub fn match_byte_mask_u8x32<const STEP: usize>(block: SimdBlock<'_, STEP, 32>, needle: u8) -> u32 {
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe {
            // SAFETY: runtime CPU detection confirmed AVX2 support, and
            // `SimdBlock<'_, STEP, 32>` guarantees that 32 bytes starting at
            // `block.as_ptr()` are within the original slice.
            crate::avx2::match_byte_mask_ptr_u8x32(block.as_ptr(), needle)
        }
    } else {
        let low = unsafe {
            // SAFETY: `SimdBlock<'_, STEP, 32>` guarantees that at least the
            // first 16 bytes starting at `block.as_ptr()` are readable.
            crate::sse::match_byte_mask_ptr_u8x16(block.as_ptr(), needle) as u32
        };
        let high = unsafe {
            // SAFETY: `SimdBlock<'_, STEP, 32>` guarantees that 32 bytes
            // starting at `block.as_ptr()` are readable, so advancing by 16
            // still leaves a readable 16-byte region.
            crate::sse::match_byte_mask_ptr_u8x16(block.as_ptr().add(16), needle) as u32
        };

        low | (high << 16)
    }
}

/// Returns `true` if any byte in the 32-byte block equals `needle`.
pub fn any_byte_eq_u8x32<const STEP: usize>(block: SimdBlock<'_, STEP, 32>, needle: u8) -> bool {
    match_byte_mask_u8x32(block, needle) != 0
}
