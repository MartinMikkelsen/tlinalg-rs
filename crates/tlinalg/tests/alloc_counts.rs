//! Steady-state heap allocations per batched call, counted with a global allocator.
//!
//! The output vector is reused with enough capacity, as a host recycling pooled buffers would.
//! What remains is the driver's per-call bookkeeping and the lane scratch, independent of the batch
//! size. This test pins those counts for one lane so a regression shows up as a number.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use strided_view::RawStridedRef;
use tlinalg::triangular_solve::{triangular_solve, TriangularSolveFlags};
use tlinalg::{LanePlan, Op, Parallel};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator unchanged; only counts calls.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded with the caller's layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded with the caller's pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Allocations made by one triangular solve over `batch` systems on one sequential lane.
fn allocations(left_side: bool, batch: usize) -> usize {
    let n = 6;
    let nrhs = 3;
    let a: Vec<f64> = (0..n * n)
        .map(|i| if i % (n + 1) == 0 { 4.0 } else { 0.1 })
        .collect();
    let (rows, cols) = if left_side { (n, nrhs) } else { (nrhs, n) };
    let b = vec![1.0_f64; rows * cols * batch];
    let flags = TriangularSolveFlags {
        left_side,
        lower: true,
        transpose_a: false,
        unit_diagonal: false,
    };
    let (a_dims, a_strides) = ([n, n, batch], [1, n as isize, 0]);
    let (b_dims, b_strides) = (
        [rows, cols, batch],
        [1, rows as isize, (rows * cols) as isize],
    );
    let a_view = RawStridedRef::new(&a, &a_dims, &a_strides, 0).unwrap();
    let b_view = RawStridedRef::new(&b, &b_dims, &b_strides, 0).unwrap();
    let mut x = Vec::with_capacity(rows * cols * batch);
    let call = |x: &mut Vec<f64>| {
        triangular_solve(
            Op::TriangularSolve,
            a_view,
            b_view,
            flags,
            x,
            Parallel::Sequential,
            LanePlan::sequential(),
        )
        .unwrap();
    };
    call(&mut x); // warm-up
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    call(&mut x);
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

#[test]
fn per_call_allocations_do_not_grow_with_the_batch() {
    let left = (allocations(true, 1), allocations(true, 64));
    let right = (allocations(false, 1), allocations(false, 64));
    // Driver bookkeeping, independent of the batch size: the normalised batch axes of `a` and `b`
    // (one small vector each) and the per-lane lists of the output and of the output tuple.
    assert_eq!(left, (4, 4), "left side: no per-item or scratch allocation");
    // Plus the right side's lane work matrix.
    assert_eq!(
        right,
        (5, 5),
        "right side: one work matrix per lane per call"
    );
}
