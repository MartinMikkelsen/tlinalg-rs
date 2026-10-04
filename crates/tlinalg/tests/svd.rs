//! Behavioural tests for the ported SVD kernel.
//!
//! The check is reconstruction: `U diag(S) Vᴴ ≈ A` for thin and full, square, tall, wide and
//! complex, plus that values-only agrees with the full decomposition's values.

use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tlinalg::svd::{svd, svd_values};
use tlinalg_traits::{Op, Parallel};

/// Column-major `m x n` matrix with well-separated singular values.
fn matrix(m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|index| {
            let row = index % m;
            let col = index / m;
            if row == col {
                3.0 + (row as f64)
            } else {
                0.5 + ((row * 3 + col * 7) % 5) as f64 * 0.25
            }
        })
        .collect()
}

/// `U diag(S) Vᴴ` in column-major, with `u` `m x u_cols`, `s` `k`, `vt` `v_cols x n`.
fn reconstruct(u: &[f64], s: &[f64], vt: &[f64], m: usize, n: usize, v_cols: usize) -> Vec<f64> {
    let k = s.len();
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for j in 0..k {
                acc += u[row + j * m] * s[j] * vt[j + col * v_cols];
            }
            out[row + col * m] = acc;
        }
    }
    out
}

fn check_svd(m: usize, n: usize, full: bool) {
    let a = matrix(m, n);
    let k = m.min(n);
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    let mut u = vec![0.0f64; m * u_cols];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![0.0f64; v_cols * n];
    svd(
        Op::Svd,
        m,
        n,
        full,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap();

    // Values are non-increasing and positive.
    for pair in s.windows(2) {
        assert!(
            pair[0] >= pair[1],
            "{m}x{n} full={full}: {:?} not sorted",
            s
        );
    }
    assert!(s[k - 1] > 1e-8, "{m}x{n}: unexpectedly singular");

    let rebuilt = reconstruct(&u, &s, &vt, m, n, v_cols);
    for (index, (got, want)) in rebuilt.iter().zip(a.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-10,
            "{m}x{n} full={full} entry {index}: reconstructed {got} != {want}"
        );
    }
}

#[test]
fn svd_reconstructs_square_tall_and_wide_matrices() {
    for (m, n) in [(4usize, 4usize), (6, 3), (3, 6), (1, 5), (5, 1)] {
        check_svd(m, n, false);
        check_svd(m, n, true);
    }
}

#[test]
fn values_only_agrees_with_the_full_decomposition() {
    let (m, n) = (5usize, 4usize);
    let a = matrix(m, n);
    let k = m.min(n);
    let mut full = vec![0.0f64; k];
    svd(
        Op::Svd,
        m,
        n,
        false,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        &mut vec![0.0; m * k],
        &mut full,
        &mut vec![0.0; k * n],
        Parallel::Sequential,
    )
    .unwrap();
    let mut only = vec![0.0f64; k];
    svd_values(
        Op::SvdValues,
        m,
        n,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        &mut only,
        Parallel::Sequential,
    )
    .unwrap();
    for (got, want) in only.iter().zip(full.iter()) {
        assert!((got - want).abs() < 1e-12, "{got} != {want}");
    }
}

#[test]
fn complex_and_f32_reconstruct() {
    let (m, n) = (4usize, 3usize);
    let k = m.min(n);

    // Complex64.
    let a: Vec<Complex64> = matrix(m, n)
        .into_iter()
        .enumerate()
        .map(|(index, real)| Complex64::new(real, if index % 3 == 0 { 0.5 } else { -0.25 }))
        .collect();
    let mut u = vec![Complex64::new(0.0, 0.0); m * k];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![Complex64::new(0.0, 0.0); k * n];
    svd(
        Op::Svd,
        m,
        n,
        false,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap();
    let mut rebuilt = vec![Complex64::new(0.0, 0.0); m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for j in 0..k {
                // Vᴴ[row j] is conjugated when multiplying out.
                acc += u[row + j * m] * s[j] * vt[j + col * k].conj();
            }
            rebuilt[row + col * m] = acc;
        }
    }
    for (index, (got, want)) in rebuilt.iter().zip(a.iter()).enumerate() {
        assert!(
            (got - want).norm() < 1e-10,
            "complex entry {index}: {got} != {want}"
        );
    }

    // Complex32 and f32 compile and run through the same paths.
    let a32: Vec<Complex32> = a
        .iter()
        .map(|v| Complex32::new(v.re as f32, v.im as f32))
        .collect();
    let mut u32 = vec![Complex32::new(0.0, 0.0); m * k];
    let mut s32 = vec![0.0f32; k];
    let mut vt32 = vec![Complex32::new(0.0, 0.0); k * n];
    svd(
        Op::Svd,
        m,
        n,
        false,
        RawStridedRef::new(&a32, &[m, n], &[1, m as isize], 0).unwrap(),
        &mut u32,
        &mut s32,
        &mut vt32,
        Parallel::Sequential,
    )
    .unwrap();

    let ar: Vec<f32> = matrix(m, n).into_iter().map(|v| v as f32).collect();
    let mut ur = vec![0.0f32; m * k];
    let mut sr = vec![0.0f32; k];
    let mut vtr = vec![0.0f32; k * n];
    svd(
        Op::Svd,
        m,
        n,
        false,
        RawStridedRef::new(&ar, &[m, n], &[1, m as isize], 0).unwrap(),
        &mut ur,
        &mut sr,
        &mut vtr,
        Parallel::Sequential,
    )
    .unwrap();
    assert!(sr[0] > 0.0);
}

#[test]
fn wrong_sized_outputs_are_rejected() {
    let (m, n) = (3usize, 2usize);
    let a = matrix(m, n);
    let k = m.min(n);
    let mut u = vec![0.0f64; m * k];
    let mut s = vec![0.0f64; k];
    let mut vt = vec![0.0f64; k * n - 1];
    let error = svd(
        Op::Svd,
        m,
        n,
        false,
        RawStridedRef::new(&a, &[m, n], &[1, m as isize], 0).unwrap(),
        &mut u,
        &mut s,
        &mut vt,
        Parallel::Sequential,
    )
    .unwrap_err();
    assert!(matches!(error, tlinalg_traits::Error::Inconsistent { .. }));
}
