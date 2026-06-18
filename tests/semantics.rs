#![allow(missing_docs)]

use simd_traverse::SimdTraverseExt;

fn sample_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|value| (value % 251) as u8).collect()
}

#[test]
fn empty_slice_has_no_blocks_and_empty_tail() {
    let bytes = [];
    let offsets: Vec<_> = bytes
        .simd_blocks::<16, 32>()
        .map(|block| block.offset())
        .collect();

    assert!(offsets.is_empty());
    assert_eq!(bytes.simd_tail::<16, 32>(), &[]);
}

#[test]
fn shorter_than_load_yields_no_blocks_and_full_tail() {
    let bytes = sample_bytes(31);
    let offsets: Vec<_> = bytes
        .as_slice()
        .simd_blocks::<16, 32>()
        .map(|block| block.offset())
        .collect();

    assert!(offsets.is_empty());
    assert_eq!(bytes.as_slice().simd_tail::<16, 32>(), bytes.as_slice());
}

#[test]
fn exactly_load_yields_one_block_and_empty_tail() {
    let bytes = sample_bytes(32);
    let mut blocks = bytes.as_slice().simd_blocks::<16, 32>();
    let block = blocks.next().expect("expected one block");

    assert_eq!(block.offset(), 0);
    assert_eq!(block.load_slice(), bytes.as_slice());
    assert_eq!(block.load(), &bytes[..32]);
    assert_eq!(block.step_slice(), &bytes[..16]);
    assert_eq!(block.as_ptr(), block.load_slice().as_ptr());
    assert!(blocks.next().is_none());
    assert_eq!(bytes.as_slice().simd_tail::<16, 32>(), &[]);
}

#[test]
fn exactly_multiple_steps_with_step_equal_load() {
    let bytes = sample_bytes(96);
    let offsets: Vec<_> = bytes
        .as_slice()
        .simd_blocks::<32, 32>()
        .map(|block| {
            assert_eq!(block.step_slice(), block.load_slice());
            block.offset()
        })
        .collect();

    assert_eq!(offsets, vec![0, 32, 64]);
    assert_eq!(bytes.as_slice().simd_tail::<32, 32>(), &[]);
}

#[test]
fn non_multiple_lengths_keep_a_disjoint_tail() {
    let bytes = sample_bytes(70);
    let offsets: Vec<_> = bytes
        .as_slice()
        .simd_blocks::<16, 32>()
        .map(|block| block.offset())
        .collect();

    assert_eq!(offsets, vec![0, 16, 32]);
    assert_eq!(bytes.as_slice().simd_tail::<16, 32>(), &bytes[64..]);
}

#[test]
fn step_less_than_load_supports_overlapping_loads() {
    let bytes = sample_bytes(48);
    let blocks: Vec<_> = bytes.as_slice().simd_blocks::<16, 32>().collect();

    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].load_slice(), &bytes[..32]);
    assert_eq!(blocks[1].load_slice(), &bytes[16..48]);
    assert_eq!(blocks[0].step_slice(), &bytes[..16]);
    assert_eq!(blocks[1].step_slice(), &bytes[16..32]);
    assert_eq!(bytes.as_slice().simd_tail::<16, 32>(), &[]);
}

#[test]
fn tail_is_disjoint_from_all_load_ranges() {
    let bytes = sample_bytes(70);
    let tail = bytes.as_slice().simd_tail::<16, 32>();
    let tail_start = bytes.len() - tail.len();

    for block in bytes.as_slice().simd_blocks::<16, 32>() {
        let load_range = block.offset()..block.offset() + 32;
        assert!(load_range.end <= tail_start || tail_start <= load_range.start);
    }
}

#[test]
fn partition_matches_individual_calls() {
    let bytes = sample_bytes(70);
    let (blocks, tail) = bytes.as_slice().simd_partition::<16, 32>();
    let separate_offsets: Vec<_> = bytes
        .as_slice()
        .simd_blocks::<16, 32>()
        .map(|block| block.offset())
        .collect();
    let partition_offsets: Vec<_> = blocks.map(|block| block.offset()).collect();

    assert_eq!(partition_offsets, separate_offsets);
    assert_eq!(tail, bytes.as_slice().simd_tail::<16, 32>());
}

#[test]
fn iterator_size_hint_and_len_shrink_as_items_are_consumed() {
    let bytes = sample_bytes(70);
    let mut blocks = bytes.as_slice().simd_blocks::<16, 32>();

    assert_eq!(blocks.size_hint(), (3, Some(3)));
    assert_eq!(blocks.len(), 3);

    let first = blocks.next().expect("expected first block");
    assert_eq!(first.offset(), 0);
    assert_eq!(blocks.size_hint(), (2, Some(2)));
    assert_eq!(blocks.len(), 2);

    let second = blocks.next().expect("expected second block");
    assert_eq!(second.offset(), 16);
    assert_eq!(blocks.size_hint(), (1, Some(1)));
    assert_eq!(blocks.len(), 1);

    let third = blocks.next().expect("expected third block");
    assert_eq!(third.offset(), 32);
    assert_eq!(blocks.size_hint(), (0, Some(0)));
    assert_eq!(blocks.len(), 0);
    assert!(blocks.next().is_none());
    assert!(blocks.next().is_none());
}

macro_rules! assert_exhaustive_layout {
    ($step:literal, $load:literal) => {
        for len in 0..=128 {
            let bytes = sample_bytes(len);
            let slice = bytes.as_slice();
            let expected_offsets: Vec<_> = if len < $load {
                Vec::new()
            } else {
                (0..=len - $load).step_by($step).collect()
            };
            let expected_tail_start = expected_offsets.last().map_or(0, |offset| offset + $load);
            let expected_tail = &slice[expected_tail_start..];
            let (blocks, tail) = slice.simd_partition::<$step, $load>();
            let actual_offsets: Vec<_> = blocks
                .map(|block| {
                    assert!(block.offset() <= len.saturating_sub($load));
                    assert_eq!(
                        block.load_slice(),
                        &slice[block.offset()..block.offset() + $load]
                    );
                    assert_eq!(
                        block.step_slice(),
                        &slice[block.offset()..block.offset() + $step]
                    );
                    block.offset()
                })
                .collect();

            assert_eq!(
                actual_offsets, expected_offsets,
                "unexpected offsets for len={len}, STEP={}, LOAD={}",
                $step, $load
            );
            assert_eq!(
                tail, expected_tail,
                "unexpected tail for len={len}, STEP={}, LOAD={}",
                $step, $load
            );
        }
    };
}

#[test]
fn exhaustive_small_layouts_match_the_documented_invariant() {
    assert_exhaustive_layout!(1, 1);
    assert_exhaustive_layout!(1, 2);
    assert_exhaustive_layout!(1, 4);
    assert_exhaustive_layout!(2, 2);
    assert_exhaustive_layout!(2, 3);
    assert_exhaustive_layout!(2, 4);
    assert_exhaustive_layout!(3, 3);
    assert_exhaustive_layout!(3, 5);
    assert_exhaustive_layout!(4, 4);
    assert_exhaustive_layout!(4, 8);
    assert_exhaustive_layout!(5, 8);
    assert_exhaustive_layout!(8, 16);
    assert_exhaustive_layout!(16, 16);
    assert_exhaustive_layout!(16, 32);
    assert_exhaustive_layout!(31, 64);
    assert_exhaustive_layout!(32, 64);
}

#[test]
#[should_panic(expected = "STEP must be greater than zero")]
fn step_zero_panics_at_api_entry() {
    let bytes = sample_bytes(64);
    let _ = bytes.as_slice().simd_blocks::<0, 32>();
}

#[test]
#[should_panic(expected = "LOAD must be greater than or equal to STEP")]
fn load_less_than_step_panics_at_api_entry() {
    let bytes = sample_bytes(64);
    let _ = bytes.as_slice().simd_tail::<32, 16>();
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use simd_traverse::SimdTraverseExt;
    use simd_traverse::neon::{any_byte_eq_u8x16, load_u8x16, match_byte_mask_u8x16};

    #[test]
    fn neon_load_and_match_mask_cover_expected_lanes() {
        let bytes = *b"abcdefghijklmnop";
        let block = bytes.as_slice().simd_blocks::<16, 16>().next().unwrap();
        let _register = load_u8x16(block);

        assert_eq!(match_byte_mask_u8x16(block, b'a'), 0x0001);
        assert_eq!(match_byte_mask_u8x16(block, b'h'), 0x0080);
        assert_eq!(match_byte_mask_u8x16(block, b'p'), 0x8000);
        assert_eq!(match_byte_mask_u8x16(block, b'z'), 0x0000);
    }

    #[test]
    fn neon_any_byte_eq_reports_presence() {
        let bytes = *b"abcdefghijklmnop";
        let block = bytes.as_slice().simd_blocks::<16, 16>().next().unwrap();

        assert!(any_byte_eq_u8x16(block, b'j'));
        assert!(!any_byte_eq_u8x16(block, b'Z'));
    }
}
