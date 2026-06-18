//! ARM64 NEON helpers built on top of [`SimdBlock`].
//!
//! These helpers consume the traversal crate's in-bounds load guarantee without
//! changing the architecture-neutral traversal API.

use crate::SimdBlock;
use core::arch::aarch64::{uint8x16_t, vceqq_u8, vdupq_n_u8, vld1q_u8, vst1q_u8};

/// Loads a 16-byte traversal block into a NEON register.
///
/// This is the lowest-level bridge from the crate's traversal semantics into
/// architecture-specific byte classification code.
pub fn load_u8x16<const STEP: usize>(block: SimdBlock<'_, STEP, 16>) -> uint8x16_t {
    unsafe {
        // SAFETY: `SimdBlock<'_, STEP, 16>` guarantees that 16 bytes starting
        // at `block.as_ptr()` are within the original slice.
        vld1q_u8(block.as_ptr())
    }
}

/// Returns a bitmask of lanes equal to `needle`.
///
/// Bit `i` in the returned mask corresponds to byte lane `i` in the loaded
/// 16-byte block.
pub fn match_byte_mask_u8x16<const STEP: usize>(block: SimdBlock<'_, STEP, 16>, needle: u8) -> u16 {
    let bytes = load_u8x16(block);
    let needles = unsafe {
        // SAFETY: this module is only compiled on `aarch64`, where NEON is part
        // of the architectural baseline for these intrinsics.
        vdupq_n_u8(needle)
    };
    let matches = unsafe {
        // SAFETY: this module is only compiled on `aarch64`, where NEON is part
        // of the architectural baseline for these intrinsics.
        vceqq_u8(bytes, needles)
    };
    let mut lanes = [0_u8; 16];

    unsafe {
        // SAFETY: `lanes` is a properly aligned, writable 16-byte output buffer
        // for the NEON vector produced by `vceqq_u8`.
        vst1q_u8(lanes.as_mut_ptr(), matches);
    }

    lanes.into_iter().enumerate().fold(
        0_u16,
        |mask, (index, lane)| {
            if lane != 0 { mask | (1 << index) } else { mask }
        },
    )
}

/// Returns `true` if any byte in the 16-byte block equals `needle`.
pub fn any_byte_eq_u8x16<const STEP: usize>(block: SimdBlock<'_, STEP, 16>, needle: u8) -> bool {
    match_byte_mask_u8x16(block, needle) != 0
}
