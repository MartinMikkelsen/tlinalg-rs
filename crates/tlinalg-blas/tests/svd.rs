//! Behavioural tests for the LAPACK SVD kernel.
//!
//! Requires a linked LAPACK: run with `--features link-openblas`. The scratch comes from a
//! test-local workspace over plain vectors, which is also the smallest proof that the `Workspace`,
//! `IndexWorkspace` and capacity contracts are usable by a host that is not tenferro.

#![cfg(feature = "link-openblas")]

use core::mem::MaybeUninit;
use num_complex::Complex64;
use tlinalg_blas::svd::{svd_batch, SvdMode};
use tlinalg_traits::{IndexWorkspace, Op, Parallel, Workspace};

/// A host-shaped workspace over plain vectors: it records what it handed out so the test can check
/// that everything came back.
#[derive(Default)]
struct TestWorkspace {
    outstanding: usize,
}

impl Workspace<f64> for TestWorkspace {
    fn acquire_zeroed(&mut self, len: usize) -> Vec<f64> {
        self.outstanding += 1;
        vec![0.0; len]
    }
    fn acquire_capacity(&mut self, cap: usize) -> Vec<f64> {
        self.outstanding += 1;
        Vec::with_capacity(cap)
    }
    fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<f64>> {
        self.outstanding += 1;
        (0..len).map(|_| MaybeUninit::uninit()).collect()
    }
    fn release(&mut self, _buf: Vec<f64>) {
        self.outstanding -= 1;
    }
}

impl Workspace<Complex64> for TestWorkspace {
    fn acquire_zeroed(&mut self, len: usize) -> Vec<Complex64> {
        self.outstanding += 1;
        vec![Complex64::new(0.0, 0.0); len]
    }
    fn acquire_capacity(&mut self, cap: usize) -> Vec<Complex64> {
        self.outstanding += 1;
        Vec::with_capacity(cap)
    }
    fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<Complex64>> {
        self.outstanding += 1;
        (0..len).map(|_| MaybeUninit::uninit()).collect()
    }
    fn release(&mut self, _buf: Vec<Complex64>) {
        self.outstanding -= 1;
    }
}

impl IndexWorkspace for TestWorkspace {
    fn acquire_zeroed_index(&mut self, len: usize) -> Vec<i32> {
        self.outstanding += 1;
        vec![0; len]
    }
    fn release_index(&mut self, _buf: Vec<i32>) {
        self.outstanding -= 1;
    }
}

/// Column-major `m x n` matrix with distinct singular values.
fn matrix(m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|index| {
            let row = index % m;
            let col = index / m;
            if row == col {
                3.0 + row as f64
            } else {
                0.5 + ((row * 3 + col * 7) % 5) as f64 * 0.25
            }
        })
        .collect()
}

fn check(m: usize, n: usize, mode: SvdMode) {
    let k = m.min(n);
    let (u_cols, vt_rows) = match mode {
        SvdMode::Thin => (k, k),
        SvdMode::Full => (m, n),
        SvdMode::Values => (0, 0),
    };
    // Two matrices in the batch, the second different, so the loop and the once-per-batch
    // workspace are both exercised.
    let first = matrix(m, n);
    let second: Vec<f64> = first.iter().map(|v| v * 0.5 + 0.25).collect();
    let mut a = [first.clone(), second.clone()].concat();
    let batch = 2;
    let mut s = vec![0.0f64; k * batch];
    let mut u = vec![0.0f64; m * u_cols * batch];
    let mut vt = vec![0.0f64; vt_rows * n * batch];
    let mut workspace = TestWorkspace::default();
    svd_batch(
        Op::Svd,
        mode,
        m,
        n,
        &mut a,
        &mut s,
        &mut u,
        &mut vt,
        &mut workspace,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(workspace.outstanding, 0, "every buffer must come back");

    for (index, want) in [&first, &second].into_iter().enumerate() {
        // Values are non-increasing in each item's slice.
        let s_i = &s[index * k..(index + 1) * k];
        for pair in s_i.windows(2) {
            assert!(pair[0] >= pair[1], "{m}x{n} {mode:?}: {s_i:?} not sorted");
        }
        if mode == SvdMode::Values {
            continue;
        }
        let u_i = &u[index * m * u_cols..(index + 1) * m * u_cols];
        let vt_i = &vt[index * vt_rows * n..(index + 1) * vt_rows * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0;
                for j in 0..k {
                    acc += u_i[row + j * m] * s_i[j] * vt_i[j + col * vt_rows];
                }
                let expected = want[row + col * m];
                assert!(
                    (acc - expected).abs() < 1e-10,
                    "{m}x{n} {mode:?} item {index} entry {row},{col}: {acc} != {expected}"
                );
            }
        }
    }
}

#[test]
fn svd_reconstructs_for_every_mode_and_shape() {
    for (m, n) in [(4usize, 4usize), (6, 3), (3, 6), (1, 4)] {
        for mode in [SvdMode::Thin, SvdMode::Full, SvdMode::Values] {
            check(m, n, mode);
        }
    }
}

#[test]
fn complex_svd_reconstructs() {
    let (m, n) = (4usize, 3usize);
    let k = m.min(n);
    let a: Vec<Complex64> = matrix(m, n)
        .into_iter()
        .enumerate()
        .map(|(index, real)| Complex64::new(real, if index % 3 == 0 { 0.5 } else { -0.25 }))
        .collect();
    let reference = a.clone();
    let mut a = a;
    let mut s = vec![0.0f64; k];
    let mut u = vec![Complex64::new(0.0, 0.0); m * k];
    let mut vt = vec![Complex64::new(0.0, 0.0); k * n];
    let mut workspace = TestWorkspace::default();
    svd_batch(
        Op::Svd,
        SvdMode::Thin,
        m,
        n,
        &mut a,
        &mut s,
        &mut u,
        &mut vt,
        &mut workspace,
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(workspace.outstanding, 0);

    for row in 0..m {
        for col in 0..n {
            // `?gesdd` returns `Vᴴ` in `vt`, so `A = U diag(S) Vᴴ` multiplies without a further
            // conjugation.
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..k {
                acc += u[row + j * m] * s[j] * vt[j + col * k];
            }
            let expected = reference[row + col * m];
            assert!(
                (acc - expected).norm() < 1e-10,
                "complex entry {row},{col}: {acc} != {expected}"
            );
        }
    }
}
