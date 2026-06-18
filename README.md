# simd-traverse

`simd-traverse` is a small traversal toolkit for byte-oriented SIMD scanners and
parsers.

The core crate does not implement substring search or depend on SIMD
intrinsics. Its job is simply to walk a `&[u8]` in fixed-size steps while
preserving one useful invariant:

- Every yielded block at `offset` guarantees that `offset + LOAD <= slice.len()`.
- Callers can therefore treat `LOAD` bytes starting at that offset as an
  in-bounds load window from the original slice.
- Tail handling is explicit and separate.

## Example

```rust
use simd_traverse::SimdTraverseExt;

let haystack = b"0123456789abcdef0123456789ABCDEFtail";

let mut offsets = Vec::new();
for block in haystack.simd_blocks::<8, 16>() {
    offsets.push(block.offset());
    assert_eq!(block.load().len(), 16);
    assert_eq!(block.step_slice().len(), 8);
}

assert_eq!(offsets, vec![0, 8, 16]);
assert_eq!(haystack.simd_tail::<8, 16>(), b"tail");
```

## ARM64 NEON

On `aarch64`, the crate also exposes a small `neon` module that consumes the
traversal invariant for 16-byte NEON work.

```rust
#[cfg(target_arch = "aarch64")]
{
    use simd_traverse::neon::match_byte_mask_u8x16;
    use simd_traverse::SimdTraverseExt;

    let haystack = b"abcdefghijklmnop";
    let block = haystack.simd_blocks::<16, 16>().next().unwrap();

    assert_eq!(match_byte_mask_u8x16(block, b'a'), 0b0000_0000_0000_0001);
}
```

## Tail Semantics

The iterator yields step-aligned offsets `0, STEP, 2 * STEP, ...` while a
`LOAD`-byte window remains in bounds.

The tail is intentionally **disjoint** from all yielded load windows:

- If at least one block is yielded, the tail starts immediately after the final
  yielded `LOAD`-byte window.
- If no block is yielded, the tail is the full original slice.

This means overlapping loads may consume bytes that never appear in the tail.

```rust
use simd_traverse::SimdTraverseExt;

let haystack = b"0123456789abcdef0123456789ABCDEF";
let offsets: Vec<_> = haystack
    .simd_blocks::<16, 32>()
    .map(|block| block.offset())
    .collect();

assert_eq!(offsets, vec![0]);
assert_eq!(haystack.simd_tail::<16, 32>(), b"");
```

## Non-Goals

- Generic element types beyond `u8`
- Search algorithms, parsers, or tokenization frameworks
- Broad cross-platform SIMD coverage in v0.1
- Unsafe implementation tricks without a measured need beyond focused arch helpers
