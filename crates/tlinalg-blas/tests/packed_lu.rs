//! Behavioural tests for the LAPACK packed-LU kernels.
//!
//! They pin the format and the semantics the host depends on, and they deliberately mirror the
//! faer-backed implementation's tests: the same reconstruction check passing on both is what makes
//! the packed factors interchangeable.
//!
//! Requires a linked LAPACK: run with `--features link-openblas`.

#![cfg(feature = "link-openblas")]

use num_complex::Complex64;
use tlinalg_blas::lu::{factor_chunk, factor_solve_chunk, solve_prepared_chunk, validate_pivots};
use tlinalg_traits::{Error, Op, Parallel};

/// Full-rank column-major `n x n` matrix whose leading entry is zero, so pivoting happens.
fn matrix(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let row = index % n;
            let col = index / n;
            if row == 0 && col == 0 && n > 1 {
                0.0
            } else if row == col {
                5.0
            } else {
                (((row * 3 + col * 7) % 5) as f64) * 0.25 + 0.5
            }
        })
        .collect()
}

fn matvec<T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T> + Default>(
    a: &[T],
    n: usize,
    x: &[T],
) -> Vec<T> {
    let mut out = vec![T::default(); n];
    for row in 0..n {
        for col in 0..n {
            out[row] = out[row] + a[row + col * n] * x[col];
        }
    }
    out
}

/// Undo a one-based swap sequence on the rows of a compact column-major block.
fn undo_swaps(a: &mut [f64], n: usize, ipiv: &[i32]) {
    for (step, &pivot) in ipiv.iter().enumerate().rev() {
        let pivot = pivot as usize - 1;
        if pivot != step {
            for col in 0..n {
                a.swap(step + col * n, pivot + col * n);
            }
        }
    }
}

/// `L * U` from the packed factor, where `L` is unit lower and `U` is upper.
fn reconstruct_lu(packed: &[f64], n: usize) -> Vec<f64> {
    let mut lu = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..n {
            let mut acc = 0.0;
            for k in 0..=row.min(col) {
                let l = if row == k {
                    1.0
                } else if row > k {
                    packed[row + k * n]
                } else {
                    0.0
                };
                let u = if k <= col { packed[k + col * n] } else { 0.0 };
                acc += l * u;
            }
            lu[row + col * n] = acc;
        }
    }
    lu
}

fn factor_one(a: &[f64], n: usize) -> (Vec<f64>, Vec<i32>, f64) {
    let mut packed = a.to_vec();
    let mut pivots = vec![0i32; n];
    let mut parity = vec![0.0f64; 1];
    factor_chunk(
        Op::LuFactor,
        n,
        n,
        &mut packed,
        &mut pivots,
        &mut parity,
        Parallel::Sequential,
    )
    .unwrap();
    (packed, pivots, parity[0])
}

#[test]
fn factor_chunk_reproduces_the_input_and_the_parity() {
    for n in [1usize, 3, 5] {
        let a = matrix(n);
        let (packed, pivots, parity) = factor_one(&a, n);

        let mut rebuilt = reconstruct_lu(&packed, n);
        undo_swaps(&mut rebuilt, n, &pivots);
        for (index, (got, want)) in rebuilt.iter().zip(a.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "n={n} entry {index}: reconstructed {got} != input {want}"
            );
        }

        let swaps = pivots
            .iter()
            .enumerate()
            .filter(|(i, p)| **p as usize - 1 != *i)
            .count();
        let expected = if swaps % 2 == 1 { -1.0 } else { 1.0 };
        assert_eq!(parity, expected, "n={n} parity");
    }
}

#[test]
fn prepared_solve_recovers_a_known_solution_for_every_flag_combination() {
    let n = 4;
    let a = matrix(n);
    let (packed, pivots, _) = factor_one(&a, n);
    for (index, value) in (0..n).map(|i| packed[i + i * n]).enumerate() {
        assert!(value.abs() > 1e-8, "test matrix is singular at {index}");
    }
    let x_true = [1.0, -2.0, 0.5, 3.0];

    // Plain: A x = b.
    let mut out = matvec(&a, n, &x_true);
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &packed,
        &pivots,
        &mut out,
        false,
        false,
        Parallel::Sequential,
    )
    .unwrap();
    for (got, want) in out.iter().zip(x_true.iter()) {
        assert!((got - want).abs() < 1e-10, "plain: {got} != {want}");
    }

    // Transposed: A^T x = b.
    let mut a_t = vec![0.0f64; n * n];
    for row in 0..n {
        for col in 0..n {
            a_t[row + col * n] = a[col + row * n];
        }
    }
    let mut out = matvec(&a_t, n, &x_true);
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &packed,
        &pivots,
        &mut out,
        true,
        false,
        Parallel::Sequential,
    )
    .unwrap();
    for (got, want) in out.iter().zip(x_true.iter()) {
        assert!((got - want).abs() < 1e-10, "transposed: {got} != {want}");
    }

    // Two right-hand sides at once.
    let x2 = [-0.5, 1.5, 2.0, -1.0];
    let mut rhs = matvec(&a, n, &x_true);
    rhs.extend(matvec(&a, n, &x2));
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        2,
        &packed,
        &pivots,
        &mut rhs,
        false,
        false,
        Parallel::Sequential,
    )
    .unwrap();
    for (got, want) in rhs[..n].iter().zip(x_true.iter()) {
        assert!((got - want).abs() < 1e-10, "rhs 0: {got} != {want}");
    }
    for (got, want) in rhs[n..].iter().zip(x2.iter()) {
        assert!((got - want).abs() < 1e-10, "rhs 1: {got} != {want}");
    }
}

#[test]
fn conjugated_solve_solves_the_adjoint() {
    // `conj(A) x = b` is solved as `A conj(x) = conj(b)` and conjugated back, so the caller gets
    // the true solution of the conjugated system, not its conjugate.
    let n = 3;
    let a: Vec<Complex64> = matrix(n)
        .into_iter()
        .enumerate()
        .map(|(index, real)| Complex64::new(real, if index % 2 == 0 { 0.5 } else { -0.25 }))
        .collect();
    let mut packed = a.clone();
    let mut pivots = vec![0i32; n];
    let mut parity = vec![Complex64::new(0.0, 0.0); 1];
    factor_chunk(
        Op::LuFactor,
        n,
        n,
        &mut packed,
        &mut pivots,
        &mut parity,
        Parallel::Sequential,
    )
    .unwrap();

    let x_true: Vec<Complex64> = [1.0, -2.0, 0.5]
        .into_iter()
        .map(|re| Complex64::new(re, 0.25))
        .collect();
    let mut rhs: Vec<Complex64> = (0..n)
        .map(|row| {
            (0..n)
                .map(|col| a[row + col * n].conj() * x_true[col])
                .sum()
        })
        .collect();
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &packed,
        &pivots,
        &mut rhs,
        false,
        true,
        Parallel::Sequential,
    )
    .unwrap();
    for (got, want) in rhs.iter().zip(x_true.iter()) {
        assert!((got - want).norm() < 1e-10, "conjugated: {got} != {want}");
    }
}

#[test]
fn factor_solve_matches_factor_then_solve() {
    let n = 3;
    let a = matrix(n);
    let x_true = [1.0, -2.0, 0.5];
    let b = matvec(&a, n, &x_true);

    let (reference_packed, reference_pivots, _) = factor_one(&a, n);
    let mut expected = b.clone();
    solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &reference_packed,
        &reference_pivots,
        &mut expected,
        false,
        false,
        Parallel::Sequential,
    )
    .unwrap();

    let mut fused = a.clone();
    let mut pivots = vec![0i32; n];
    let mut solved = b.clone();
    factor_solve_chunk(
        Op::LuFactorSolve,
        n,
        1,
        &mut fused,
        &mut pivots,
        &mut solved,
        Parallel::Sequential,
    )
    .unwrap();

    assert_eq!(fused, reference_packed, "fused factors differ");
    for (got, want) in solved.iter().zip(expected.iter()) {
        assert!((got - want).abs() < 1e-12, "fused {got} != {want}");
    }
}

#[test]
fn singular_input_is_rejected_only_when_a_rhs_is_present() {
    let n = 2;
    let a = [1.0, 2.0, 1.0, 2.0];
    // Zero-column RHS: only factor.
    let mut packed = a;
    let mut pivots = vec![0i32; n];
    let mut no_rhs: Vec<f64> = Vec::new();
    factor_solve_chunk(
        Op::LuFactorSolve,
        n,
        0,
        &mut packed,
        &mut pivots,
        &mut no_rhs,
        Parallel::Sequential,
    )
    .unwrap();

    let mut packed = a;
    let mut pivots = vec![0i32; n];
    let mut rhs = vec![1.0f64; n];
    let error = factor_solve_chunk(
        Op::LuFactorSolve,
        n,
        1,
        &mut packed,
        &mut pivots,
        &mut rhs,
        Parallel::Sequential,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::Singular {
            op: Op::LuFactorSolve
        }
    ));
}

#[test]
fn invalid_pivots_are_rejected_before_any_write() {
    let n = 2;
    let (packed, mut pivots, _) = factor_one(&matrix(n), n);
    pivots[0] = 7;
    assert!(matches!(
        validate_pivots(Op::LuSolvePrepared, n, &pivots),
        Err(Error::InvalidArgument { role: "pivot", .. })
    ));
    let mut output = vec![1.0f64; n];
    let before = output.clone();
    let error = solve_prepared_chunk(
        Op::LuSolvePrepared,
        n,
        1,
        &packed,
        &pivots,
        &mut output,
        false,
        false,
        Parallel::Sequential,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidArgument { role: "pivot", .. }
    ));
    assert_eq!(output, before);
}

#[test]
fn empty_batches_are_accepted() {
    let n = 2;
    let mut lu: Vec<f64> = Vec::new();
    let mut pivots: Vec<i32> = Vec::new();
    let mut parity: Vec<f64> = Vec::new();
    factor_chunk(
        Op::LuFactor,
        n,
        n,
        &mut lu,
        &mut pivots,
        &mut parity,
        Parallel::Sequential,
    )
    .unwrap();
}
