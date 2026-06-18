use core::fmt;
use core::iter::FusedIterator;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TraversalLayout {
    block_count: usize,
    tail_start: usize,
}

impl TraversalLayout {
    fn new<const STEP: usize, const LOAD: usize>(len: usize) -> Self {
        validate_parameters::<STEP, LOAD>();

        if len < LOAD {
            return Self {
                block_count: 0,
                tail_start: 0,
            };
        }

        let last_start = len - LOAD;
        let block_count = 1 + (last_start / STEP);
        let last_offset = (block_count - 1) * STEP;

        Self {
            block_count,
            tail_start: last_offset + LOAD,
        }
    }
}

fn validate_parameters<const STEP: usize, const LOAD: usize>() {
    assert!(STEP > 0, "STEP must be greater than zero");
    assert!(LOAD >= STEP, "LOAD must be greater than or equal to STEP");
}

/// Iterator over step-aligned byte-slice offsets with guaranteed in-bounds load windows.
#[derive(Clone)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct SimdBlocks<'a, const STEP: usize, const LOAD: usize> {
    slice: &'a [u8],
    offset: usize,
    remaining: usize,
}

impl<'a, const STEP: usize, const LOAD: usize> SimdBlocks<'a, STEP, LOAD> {
    fn new(slice: &'a [u8], layout: TraversalLayout) -> Self {
        Self {
            slice,
            offset: 0,
            remaining: layout.block_count,
        }
    }
}

impl<'a, const STEP: usize, const LOAD: usize> fmt::Debug for SimdBlocks<'a, STEP, LOAD> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimdBlocks")
            .field("slice", &self.slice)
            .field("offset", &self.offset)
            .field("remaining", &self.remaining)
            .finish()
    }
}

impl<'a, const STEP: usize, const LOAD: usize> Iterator for SimdBlocks<'a, STEP, LOAD> {
    type Item = SimdBlock<'a, STEP, LOAD>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let block = SimdBlock {
            slice: self.slice,
            offset: self.offset,
        };

        self.offset += STEP;
        self.remaining -= 1;

        Some(block)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, const STEP: usize, const LOAD: usize> ExactSizeIterator for SimdBlocks<'a, STEP, LOAD> {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a, const STEP: usize, const LOAD: usize> FusedIterator for SimdBlocks<'a, STEP, LOAD> {}

/// A single step-aligned traversal position with an in-bounds `LOAD`-byte window.
#[derive(Clone, Copy, Debug)]
#[must_use = "blocks describe a guaranteed in-bounds load window"]
pub struct SimdBlock<'a, const STEP: usize, const LOAD: usize> {
    slice: &'a [u8],
    offset: usize,
}

impl<'a, const STEP: usize, const LOAD: usize> SimdBlock<'a, STEP, LOAD> {
    /// Returns the starting offset of this block within the original slice.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the guaranteed in-bounds `LOAD`-byte window for this block.
    ///
    /// # Panics
    ///
    /// Panics if the block's internal invariant is violated and the load window
    /// is not exactly `LOAD` bytes long. This should be unreachable through the
    /// public API.
    pub fn load(&self) -> &'a [u8; LOAD] {
        self.load_slice()
            .try_into()
            .expect("internal invariant violated: load slice must be LOAD bytes")
    }

    /// Returns the guaranteed in-bounds `LOAD`-byte window as a slice.
    pub fn load_slice(&self) -> &'a [u8] {
        &self.slice[self.offset..self.offset + LOAD]
    }

    /// Returns the logical `STEP`-sized sub-slice for this block.
    pub fn step_slice(&self) -> &'a [u8] {
        &self.slice[self.offset..self.offset + STEP]
    }

    /// Returns a pointer to the first byte of the block's load window.
    ///
    /// The returned pointer is valid for pointer arithmetic and may be used as
    /// the start of a `LOAD`-byte region within the original slice. Dereferencing
    /// that pointer remains unsafe and is the caller's responsibility.
    pub fn as_ptr(&self) -> *const u8 {
        self.load_slice().as_ptr()
    }
}

pub(crate) fn partition_slice<'a, const STEP: usize, const LOAD: usize>(
    slice: &'a [u8],
) -> (SimdBlocks<'a, STEP, LOAD>, &'a [u8]) {
    let layout = TraversalLayout::new::<STEP, LOAD>(slice.len());
    let blocks = SimdBlocks::<'a, STEP, LOAD>::new(slice, layout);
    let tail = &slice[layout.tail_start..];

    (blocks, tail)
}
