//! Behavioural tests for the ported partial-pivot LU, the LU solve, and the full-pivot LU family.

mod common;

use common::*;
use num_complex::Complex64;
use strided_view::{RawStridedMut, RawStridedRef};
use tlinalg::full_piv_lu::{full_piv_lu, full_piv_lu_solve, FullPivLuFactors};
use tlinalg::lu::{lu, solve, LuFactors};
use tlinalg::FaerScalar;
use tlinalg::{Error, Op, Parallel};

/// Determinant sign of a permutation matrix, by counting inversions of its row map.
fn permutation_sign(p: &[Complex64], n: usize) -> f64 {
    let map: Vec<usize> = (0..n)
        .map(|row| (0..n).find(|&col| p[row + col * n].re == 1.0).unwrap())
        .collect();
    let mut inversions = 0;
    for i in 0..n {
        for j in i + 1..n {
            if map[i] > map[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn lu_reconstructs<T: TestScalar + FaerScalar>() {
    for (m, n) in [(4, 4), (5, 3), (3, 5), (1, 1)] {
        let a = matrix::<T>(m, n, 11);
        let (storage, strides, offset) = padded(&a, m, n);
        let dims = [m, n];
        let (mut p, mut l, mut u) = (Vec::new(), Vec::new(), Vec::new());
        let parity = lu(
            Op::Lu,
            m,
            n,
            RawStridedRef::new(&storage, &dims, &strides, offset).unwrap(),
            LuFactors {
                p: &mut p,
                l: &mut l,
                u: &mut u,
            },
            Parallel::Sequential,
        )
        .unwrap();
        let k = m.min(n);
        assert_eq!((p.len(), l.len(), u.len()), (m * m, m * k, k * n));
        let (p, l, u) = (widen(&p), widen(&l), widen(&u));
        for i in 0..k {
            assert_eq!(l[i + i * m], Complex64::new(1.0, 0.0), "unit diagonal");
        }
        let rebuilt = matmul(&matmul(&p, &l, m, m, k), &u, m, k, n);
        assert_close(&rebuilt, &widen(&a), T::TOL, &format!("P L U {m}x{n}"));
        assert_eq!(parity.to_c64().re, permutation_sign(&p, m), "parity");
    }
}
for_each_scalar!(lu_reconstruction, lu_reconstructs);

#[test]
fn lu_of_a_singular_matrix_is_not_an_error() {
    let a = [1.0_f64, 2.0, 2.0, 4.0];
    let (mut p, mut l, mut u) = (Vec::new(), Vec::new(), Vec::new());
    lu(
        Op::Lu,
        2,
        2,
        RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
        LuFactors {
            p: &mut p,
            l: &mut l,
            u: &mut u,
        },
        Parallel::Sequential,
    )
    .unwrap();
    assert_eq!(u[3], 0.0);
}

fn solve_residual<T: TestScalar + FaerScalar>() {
    let n = 4;
    let nrhs = 2;
    let a = matrix::<T>(n, n, 13);
    let b = matrix::<T>(n, nrhs, 17);
    for transpose_a in [false, true] {
        // Strided destination: leading dimension n + 2, offset 3.
        let (mut storage, strides, offset) = padded(&vec![T::default(); n * nrhs], n, nrhs);
        let dims = [n, nrhs];
        solve(
            Op::Solve,
            n,
            nrhs,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            Some(RawStridedRef::new(&b, &dims, &[1, n as isize], 0).unwrap()),
            RawStridedMut::new(&mut storage, &dims, &strides, offset).unwrap(),
            transpose_a,
            Parallel::Sequential,
        )
        .unwrap();
        let lda = n + 2;
        let x: Vec<Complex64> = (0..n * nrhs)
            .map(|i| storage[3 + i % n + (i / n) * lda].to_c64())
            .collect();
        let op_a = if transpose_a {
            transpose(&widen(&a), n, n)
        } else {
            widen(&a)
        };
        assert_close(
            &matmul(&op_a, &x, n, n, nrhs),
            &widen(&b),
            T::TOL,
            &format!("solve transpose={transpose_a}"),
        );
        // The padding outside the destination is untouched.
        assert!(storage[0].to_c64().re > 1.0e29);
    }

    // `None`: the destination already holds the right-hand side.
    let mut x = b.clone();
    solve(
        Op::Solve,
        n,
        nrhs,
        RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
        None,
        RawStridedMut::new(&mut x, &[n, nrhs], &[1, n as isize], 0).unwrap(),
        false,
        Parallel::Sequential,
    )
    .unwrap();
    assert_close(
        &matmul(&widen(&a), &widen(&x), n, n, nrhs),
        &widen(&b),
        T::TOL,
        "solve in place",
    );
}
for_each_scalar!(solve_residuals, solve_residual);

fn solve_singular_leaves_destination<T: TestScalar + FaerScalar>() {
    let a = narrow::<T>(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let b = matrix::<T>(2, 1, 1);
    let sentinel = T::from_c64(Complex64::new(7.0, 0.0));
    let mut out = vec![sentinel; 2];
    let err = solve(
        Op::Solve,
        2,
        1,
        RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
        Some(RawStridedRef::new(&b, &[2, 1], &[1, 2], 0).unwrap()),
        RawStridedMut::new(&mut out, &[2, 1], &[1, 2], 0).unwrap(),
        false,
        Parallel::Sequential,
    )
    .unwrap_err();
    assert_eq!(err, Error::Singular { op: Op::Solve });
    assert_eq!(out, vec![sentinel; 2], "destination untouched on failure");
}
for_each_scalar!(solve_singular, solve_singular_leaves_destination);

fn full_piv_reconstructs<T: TestScalar + FaerScalar>() {
    for n in [1, 4] {
        let a = matrix::<T>(n, n, 19);
        let (storage, strides, offset) = padded(&a, n, n);
        let dims = [n, n];
        let (mut p, mut l, mut u, mut q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let parity = full_piv_lu(
            Op::FullPivLu,
            n,
            RawStridedRef::new(&storage, &dims, &strides, offset).unwrap(),
            FullPivLuFactors {
                p: &mut p,
                l: &mut l,
                u: &mut u,
                q: &mut q,
            },
            Parallel::Sequential,
        )
        .unwrap();
        let (p, l, u, q) = (widen(&p), widen(&l), widen(&u), widen(&q));
        let rebuilt = matmul(&matmul(&matmul(&p, &l, n, n, n), &u, n, n, n), &q, n, n, n);
        assert_close(&rebuilt, &widen(&a), T::TOL, "P L U Q");
        let sign = permutation_sign(&p, n) * permutation_sign(&q, n);
        assert_eq!(parity.to_c64().re, sign, "parity");
    }
}
for_each_scalar!(full_piv_reconstruction, full_piv_reconstructs);

fn full_piv_solves<T: TestScalar + FaerScalar>() {
    let n = 4;
    let nrhs = 3;
    let a = matrix::<T>(n, n, 23);
    let b = matrix::<T>(n, nrhs, 29);
    for transpose_a in [false, true] {
        let mut x = b.clone();
        full_piv_lu_solve(
            Op::FullPivLuSolve,
            n,
            nrhs,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            &mut x,
            transpose_a,
            Parallel::Sequential,
        )
        .unwrap();
        let op_a = if transpose_a {
            transpose(&widen(&a), n, n)
        } else {
            widen(&a)
        };
        assert_close(
            &matmul(&op_a, &widen(&x), n, n, nrhs),
            &widen(&b),
            T::TOL,
            "full-pivot solve",
        );
    }

    // Rank one: the second pivot is effectively zero.
    let singular = narrow::<T>(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
    ]);
    let mut x = matrix::<T>(2, 1, 0);
    let before = x.clone();
    let err = full_piv_lu_solve(
        Op::FullPivLuSolve,
        2,
        1,
        RawStridedRef::new(&singular, &[2, 2], &[1, 2], 0).unwrap(),
        &mut x,
        false,
        Parallel::Sequential,
    )
    .unwrap_err();
    assert_eq!(
        err,
        Error::Singular {
            op: Op::FullPivLuSolve
        }
    );
    assert_eq!(x, before, "right-hand side untouched on failure");

    let err = full_piv_lu_solve(
        Op::FullPivLuSolve,
        2,
        2,
        RawStridedRef::new(&singular, &[2, 2], &[1, 2], 0).unwrap(),
        &mut x,
        false,
        Parallel::Sequential,
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidArgument { .. }));
}
for_each_scalar!(full_piv_solve, full_piv_solves);

#[test]
fn solve_on_a_supplied_pool_matches_sequential() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let n = 40;
    let a = matrix::<f64>(n, n, 3);
    let b = matrix::<f64>(n, 2, 4);
    let run = |par| {
        let mut x = b.clone();
        solve(
            Op::Solve,
            n,
            2,
            RawStridedRef::new(&a, &[n, n], &[1, n as isize], 0).unwrap(),
            None,
            RawStridedMut::new(&mut x, &[n, 2], &[1, n as isize], 0).unwrap(),
            false,
            par,
        )
        .unwrap();
        x
    };
    let sequential = run(Parallel::Sequential);
    let pooled = run(Parallel::Pool {
        pool: &pool,
        budget: core::num::NonZeroUsize::new(2).unwrap(),
    });
    assert_close(
        &widen(&pooled),
        &widen(&sequential),
        1e-12,
        "pool vs sequential",
    );
}
