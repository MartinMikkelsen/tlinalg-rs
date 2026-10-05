//! Steady-state heap allocations per batched call, counted with a global allocator.
//!
//! The output vectors are reused with enough capacity, as a host recycling pooled buffers would.
//! The batch driver allocates nothing on one lane; what remains is each family's faer lane
//! scratch, independent of the batch size. This test pins those counts for one lane so a
//! regression shows up as a number.

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

/// Allocations made by the second of two identical calls (the first warms up).
fn steady(mut call: impl FnMut()) -> usize {
    call();
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    call();
    ALLOCATIONS.load(Ordering::Relaxed) - before
}

/// `batch` compact, well-conditioned `n x n` matrices.
fn batch_of(n: usize, batch: usize) -> Vec<f64> {
    (0..n * n * batch)
        .map(|i| {
            let (row, col) = (i % n, (i / n) % n);
            if row == col {
                4.0 + row as f64
            } else {
                0.1
            }
        })
        .collect()
}

/// Steady-state allocations per call of each family, one sequential lane, `n = 6`.
fn family_counts(batch: usize) -> [(&'static str, usize); 7] {
    let n = 6;
    let a = batch_of(n, batch);
    let (dims, strides) = ([n, n, batch], [1, n as isize, (n * n) as isize]);
    let view = RawStridedRef::new(&a, &dims, &strides, 0).unwrap();
    let (seq, plan) = (Parallel::Sequential, LanePlan::sequential());
    let cap = n * n * batch;
    let (mut v1, mut v2, mut v3) = (
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
        Vec::with_capacity(cap),
    );
    let cholesky = steady(|| {
        tlinalg::cholesky::cholesky(Op::Cholesky, view, &mut v1, seq, plan).unwrap();
    });
    let qr = steady(|| tlinalg::qr::qr(Op::Qr, view, &mut v1, &mut v2, seq, plan).unwrap());
    let eigh = steady(|| {
        tlinalg::eigh::eigh(Op::Eigh, view, &mut v1, &mut v2, seq, plan).unwrap();
    });
    let svd = steady(|| {
        tlinalg::svd::svd(Op::Svd, view, false, &mut v1, &mut v2, &mut v3, seq, plan).unwrap();
    });
    let (mut lu, mut piv, mut parity) = (a.clone(), vec![0; n * batch], vec![0.0; batch]);
    let packed_lu = steady(|| {
        lu.copy_from_slice(&a);
        tlinalg::packed_lu::factor(
            Op::LuFactor,
            n,
            n,
            &mut lu,
            &mut piv,
            &mut parity,
            seq,
            plan,
        )
        .unwrap();
    });
    [
        ("cholesky", cholesky),
        ("qr", qr),
        ("eigh", eigh),
        ("svd", svd),
        ("packed_lu factor", packed_lu),
        ("triangular_solve left", triangular(true, batch)),
        ("triangular_solve right", triangular(false, batch)),
    ]
}

/// Allocations made by one triangular solve over `batch` systems on one sequential lane.
fn triangular(left_side: bool, batch: usize) -> usize {
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
    steady(|| {
        triangular_solve(
            Op::TriangularSolve,
            a_view,
            b_view,
            flags,
            &mut x,
            Parallel::Sequential,
            LanePlan::sequential(),
        )
        .unwrap();
    })
}

/// The driver itself allocates nothing on one lane; what remains is each family's faer lane
/// scratch, built once per call and independent of the batch size.
#[test]
fn per_call_allocations_do_not_grow_with_the_batch() {
    let one = family_counts(1);
    let many = family_counts(64);
    for ((name, at_one), (_, at_many)) in one.iter().zip(&many) {
        println!("{name}: {at_one} (batch 1), {at_many} (batch 64)");
        assert_eq!(at_one, at_many, "{name}: allocations grow with the batch");
    }
    let expected = [
        ("cholesky", 2),
        ("qr", 5),
        ("eigh", 3),
        ("svd", 4),
        ("packed_lu factor", 5),
        ("triangular_solve left", 0),
        ("triangular_solve right", 1),
    ];
    assert_eq!(one, expected);
}
