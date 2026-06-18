//! A byte scanner that uses the best available `simd-traverse` helper module.

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
use simd_traverse::SimdTraverseExt;

fn main() {
    let haystack = b"alpha,beta,gamma,delta,epsilon,zeta,eta,theta,tail";
    let offsets = find_byte_offsets(haystack, b',');

    println!("backend: {}", backend_name());
    println!("comma offsets: {offsets:?}");

    assert_eq!(offsets, scalar_find_byte_offsets(haystack, b','));
}

#[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
fn backend_name() -> &'static str {
    "x86 runtime dispatch"
}

#[cfg(all(
    target_arch = "x86_64",
    not(feature = "runtime-dispatch"),
    target_feature = "avx2"
))]
fn backend_name() -> &'static str {
    "x86_64 AVX2"
}

#[cfg(all(
    target_arch = "x86_64",
    not(feature = "runtime-dispatch"),
    not(target_feature = "avx2")
))]
fn backend_name() -> &'static str {
    "x86_64 SSE2"
}

#[cfg(target_arch = "aarch64")]
fn backend_name() -> &'static str {
    "aarch64 NEON"
}

#[cfg(all(target_arch = "arm", target_feature = "neon"))]
fn backend_name() -> &'static str {
    "arm NEON"
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
)))]
fn backend_name() -> &'static str {
    "scalar"
}

#[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
fn find_byte_offsets(haystack: &[u8], needle: u8) -> Vec<usize> {
    let mut offsets = Vec::new();
    let (blocks, tail) = haystack.simd_partition::<32, 32>();

    for block in blocks {
        push_mask_offsets(
            simd_traverse::x86::match_byte_mask_u8x32(block, needle),
            block.offset(),
            &mut offsets,
        );
    }

    push_tail_offsets(tail, haystack.len() - tail.len(), needle, &mut offsets);
    offsets
}

#[cfg(all(
    target_arch = "x86_64",
    not(feature = "runtime-dispatch"),
    target_feature = "avx2"
))]
fn find_byte_offsets(haystack: &[u8], needle: u8) -> Vec<usize> {
    let mut offsets = Vec::new();
    let (blocks, tail) = haystack.simd_partition::<32, 32>();

    for block in blocks {
        push_mask_offsets(
            simd_traverse::avx2::match_byte_mask_u8x32(block, needle),
            block.offset(),
            &mut offsets,
        );
    }

    push_tail_offsets(tail, haystack.len() - tail.len(), needle, &mut offsets);
    offsets
}

#[cfg(all(
    target_arch = "x86_64",
    not(feature = "runtime-dispatch"),
    not(target_feature = "avx2")
))]
fn find_byte_offsets(haystack: &[u8], needle: u8) -> Vec<usize> {
    let mut offsets = Vec::new();
    let (blocks, tail) = haystack.simd_partition::<16, 16>();

    for block in blocks {
        push_mask_offsets(
            simd_traverse::sse::match_byte_mask_u8x16(block, needle).into(),
            block.offset(),
            &mut offsets,
        );
    }

    push_tail_offsets(tail, haystack.len() - tail.len(), needle, &mut offsets);
    offsets
}

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
fn find_byte_offsets(haystack: &[u8], needle: u8) -> Vec<usize> {
    let mut offsets = Vec::new();
    let (blocks, tail) = haystack.simd_partition::<16, 16>();

    for block in blocks {
        push_mask_offsets(
            simd_traverse::neon::match_byte_mask_u8x16(block, needle).into(),
            block.offset(),
            &mut offsets,
        );
    }

    push_tail_offsets(tail, haystack.len() - tail.len(), needle, &mut offsets);
    offsets
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
)))]
fn find_byte_offsets(haystack: &[u8], needle: u8) -> Vec<usize> {
    scalar_find_byte_offsets(haystack, needle)
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
fn push_mask_offsets(mut mask: u32, block_offset: usize, offsets: &mut Vec<usize>) {
    while mask != 0 {
        let lane = mask.trailing_zeros() as usize;

        offsets.push(block_offset + lane);
        mask &= mask - 1;
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
fn push_tail_offsets(tail: &[u8], tail_offset: usize, needle: u8, offsets: &mut Vec<usize>) {
    offsets.extend(
        tail.iter()
            .enumerate()
            .filter_map(|(index, byte)| (*byte == needle).then_some(tail_offset + index)),
    );
}

fn scalar_find_byte_offsets(haystack: &[u8], needle: u8) -> Vec<usize> {
    haystack
        .iter()
        .enumerate()
        .filter_map(|(index, byte)| (*byte == needle).then_some(index))
        .collect()
}
