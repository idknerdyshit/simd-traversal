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

The `examples/byte_scanner.rs` example shows a complete byte scanner that
exercises the arm/aarch64 NEON, SSE2, AVX2, runtime-dispatched x86, and scalar
fallback paths through `cfg` selection.

## ARM NEON

On `aarch64`, and on 32-bit `arm` when the `neon` target feature is enabled,
the crate exposes a small `neon` module that consumes the traversal invariant
for 16-byte NEON work.

```rust
#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "arm", target_feature = "neon")
))]
{
    use simd_traverse::neon::match_byte_mask_u8x16;
    use simd_traverse::SimdTraverseExt;

    let haystack = b"abcdefghijklmnop";
    let block = haystack.simd_blocks::<16, 16>().next().unwrap();

    assert_eq!(match_byte_mask_u8x16(block, b'a'), 0b0000_0000_0000_0001);
}
```

## x86_64 SSE2

On `x86_64`, the crate exposes a matching `sse` module for 16-byte SSE2 work.

```rust
#[cfg(target_arch = "x86_64")]
{
    use simd_traverse::sse::match_byte_mask_u8x16;
    use simd_traverse::SimdTraverseExt;

    let haystack = b"abcdefghijklmnop";
    let block = haystack.simd_blocks::<16, 16>().next().unwrap();

    assert_eq!(match_byte_mask_u8x16(block, b'a'), 0b0000_0000_0000_0001);
}
```

## x86_64 AVX2

When AVX2 is enabled at compile time, the crate exposes an `avx2` module for
32-byte work.

```rust
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
{
    use simd_traverse::avx2::match_byte_mask_u8x32;
    use simd_traverse::SimdTraverseExt;

    let haystack = b"abcdefghijklmnopqrstuvwxyz012345";
    let block = haystack.simd_blocks::<32, 32>().next().unwrap();

    assert_eq!(match_byte_mask_u8x32(block, b'a'), 0x0000_0001);
}
```

With the `runtime-dispatch` feature, the crate also exposes an `x86` module that
uses AVX2 when the running CPU supports it and falls back to SSE2 otherwise.
This feature enables `std`; the default crate remains `no_std`.

```rust
#[cfg(all(target_arch = "x86_64", feature = "runtime-dispatch"))]
{
    use simd_traverse::x86::match_byte_mask_u8x32;
    use simd_traverse::SimdTraverseExt;

    let haystack = b"abcdefghijklmnopqrstuvwxyz012345";
    let block = haystack.simd_blocks::<32, 32>().next().unwrap();

    assert_eq!(match_byte_mask_u8x32(block, b'5'), 0x8000_0000);
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
- SIMD operations beyond byte equality masks in v0.1
- Unsafe implementation tricks without a measured need beyond focused arch helpers
