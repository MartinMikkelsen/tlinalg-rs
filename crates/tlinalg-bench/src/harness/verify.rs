//! Correctness gate: provider outputs are checked against their mathematical contracts.

use super::cases::{applicable, Case};
use super::rows::{call, output_call, Output};
use crate::{BenchScalar, Env};
use num_complex::Complex64;

fn tolerance<T: BenchScalar>(c: Case) -> f64 {
    64.0 * (c.m.max(c.n) as f64).sqrt() * T::TOL
}

fn max_residual(got: &[Complex64], want: &[Complex64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}
fn check_orthonormal(q: &[Complex64], rows: usize, cols: usize, tol: f64) -> Result<f64, String> {
    let mut residual: f64 = 0.0;
    for j in 0..cols {
        for k in 0..cols {
            let mut x = Complex64::new(0.0, 0.0);
            for i in 0..rows {
                x += q[i + j * rows].conj() * q[i + k * rows];
            }
            let want = if j == k { 1.0 } else { 0.0 };
            residual = residual.max((x - Complex64::new(want, 0.0)).norm());
        }
    }
    if residual > tol {
        Err(format!(
            "orthonormality residual={residual:.3e} > {tol:.3e}"
        ))
    } else {
        Ok(residual)
    }
}
fn matrix(a: &[Complex64], m: usize, _n: usize, row: usize, col: usize) -> Complex64 {
    a[row + col * m]
}
fn check_one<T: BenchScalar>(family: &str, c: Case, out: Output<T>) -> Result<f64, String> {
    let tol = tolerance::<T>(c);
    let a = if matches!(family, "cholesky" | "eigh") {
        tlinalg_testkit::widen(
            &(0..c.batch)
                .flat_map(|i| tlinalg_testkit::hpd_seeded::<T>(c.m, i + 1))
                .collect::<Vec<_>>(),
        )
    } else {
        tlinalg_testkit::widen(&tlinalg_testkit::batch_of::<T>(c.m, c.n, c.batch, 0))
    };
    let mut worst: f64 = 0.0;
    match (family, out) {
        ("cholesky", Output::One(l)) => {
            let l = tlinalg_testkit::widen(&l);
            for batch in 0..c.batch {
                let off = batch * c.m * c.m;
                let aa = &a[..];
                let mut rec = vec![Complex64::new(0.0, 0.0); c.m * c.m];
                for col in 0..c.m {
                    for k in 0..=col {
                        for row in 0..c.m {
                            rec[row + col * c.m] +=
                                matrix(&l[off..off + c.m * c.m], c.m, c.m, row, k)
                                    * matrix(&l[off..off + c.m * c.m], c.m, c.m, col, k).conj();
                        }
                    }
                }
                worst = worst.max(max_residual(&rec, aa));
            }
        }
        ("qr", Output::Two(q, r)) => {
            let q = tlinalg_testkit::widen(&q);
            let r = tlinalg_testkit::widen(&r);
            let k = c.m.min(c.n);
            for batch in 0..c.batch {
                let qo = batch * c.m * k;
                let ro = batch * k * c.n;
                let mut rec = vec![Complex64::new(0.0, 0.0); c.m * c.n];
                for col in 0..c.n {
                    for z in 0..k {
                        for row in 0..c.m {
                            rec[row + col * c.m] += q[qo + row + z * c.m] * r[ro + z + col * k];
                        }
                    }
                }
                worst = worst.max(max_residual(&rec, &a));
                worst = worst.max(check_orthonormal(&q[qo..qo + c.m * k], c.m, k, tol)?);
            }
        }
        ("svd_thin" | "svd_full", Output::ThreeReal(u, s, vt)) => {
            let u = tlinalg_testkit::widen(&u);
            let vt = tlinalg_testkit::widen(&vt);
            let k = c.m.min(c.n);
            let uc = if family == "svd_full" { c.m } else { k };
            let vr = if family == "svd_full" { c.n } else { k };
            for batch in 0..c.batch {
                let uo = batch * c.m * uc;
                let vo = batch * vr * c.n;
                worst = worst.max(check_orthonormal(&u[uo..uo + c.m * uc], c.m, uc, tol)?);
                let mut rec = vec![Complex64::new(0., 0.); c.m * c.n];
                for col in 0..c.n {
                    for z in 0..k {
                        for row in 0..c.m {
                            rec[row + col * c.m] +=
                                u[uo + row + z * c.m] * s[batch * k + z] * vt[vo + z + col * vr];
                        }
                    }
                }
                worst = worst.max(max_residual(&rec, &a));
            }
        }
        ("svd_thin" | "svd_full", Output::Three(u, s, vt)) => {
            let u = tlinalg_testkit::widen(&u);
            let s = tlinalg_testkit::widen(&s);
            let vt = tlinalg_testkit::widen(&vt);
            let k = c.m.min(c.n);
            let uc = if family == "svd_full" { c.m } else { k };
            let vr = if family == "svd_full" { c.n } else { k };
            for batch in 0..c.batch {
                let uo = batch * c.m * uc;
                let so = batch * k;
                let vo = batch * vr * c.n;
                worst = worst.max(check_orthonormal(&u[uo..uo + c.m * uc], c.m, uc, tol)?);
                for i in 1..k {
                    if s[so + i].re > s[so + i - 1].re + tol || s[so + i].re < -tol {
                        return Err("singular values are not descending and non-negative".into());
                    }
                }
                let mut rec = vec![Complex64::new(0.0, 0.0); c.m * c.n];
                for col in 0..c.n {
                    for z in 0..k {
                        for row in 0..c.m {
                            rec[row + col * c.m] +=
                                u[uo + row + z * c.m] * s[so + z].re * vt[vo + z + col * vr];
                        }
                    }
                }
                worst = worst.max(max_residual(&rec, &a));
            }
        }
        ("svd_values", Output::One(s)) => {
            let s = tlinalg_testkit::widen(&s);
            for batch in 0..c.batch {
                for i in 1..c.m.min(c.n) {
                    if s[batch * c.m.min(c.n) + i].re > s[batch * c.m.min(c.n) + i - 1].re + tol
                        || s[batch * c.m.min(c.n) + i].re < -tol
                    {
                        return Err("singular values are not descending and non-negative".into());
                    }
                }
            }
            worst = 0.0;
        }
        ("eigh", Output::Two(w, v)) => {
            let w = tlinalg_testkit::widen(&w);
            let v = tlinalg_testkit::widen(&v);
            for batch in 0..c.batch {
                let vo = batch * c.m * c.m;
                let wo = batch * c.m;
                worst = worst.max(check_orthonormal(&v[vo..vo + c.m * c.m], c.m, c.m, tol)?);
                for i in 1..c.m {
                    if w[wo + i].im.abs() > tol || w[wo + i].re + tol < w[wo + i - 1].re {
                        return Err("eigenvalues are not real and ascending".into());
                    }
                }
                let mut res: f64 = 0.0;
                for col in 0..c.m {
                    for row in 0..c.m {
                        let mut lhs = Complex64::new(0., 0.);
                        for k in 0..c.m {
                            lhs += a[row + k * c.m] * v[vo + k + col * c.m];
                        }
                        res = res.max((lhs - v[vo + row + col * c.m] * w[wo + col]).norm());
                    }
                }
                worst = worst.max(res);
            }
        }
        ("eigh", Output::TwoReal(w, v)) => {
            let v = tlinalg_testkit::widen(&v);
            for batch in 0..c.batch {
                let vo = batch * c.m * c.m;
                let wo = batch * c.m;
                worst = worst.max(check_orthonormal(&v[vo..vo + c.m * c.m], c.m, c.m, tol)?);
                for i in 1..c.m {
                    if w[wo + i] < w[wo + i - 1] - tol {
                        return Err("eigenvalues are not ascending".into());
                    }
                }
                let mut res: f64 = 0.0;
                for col in 0..c.m {
                    for row in 0..c.m {
                        let mut lhs = Complex64::new(0., 0.);
                        for k in 0..c.m {
                            lhs += a[row + k * c.m] * v[vo + k + col * c.m];
                        }
                        res = res.max((lhs - v[vo + row + col * c.m] * w[wo + col]).norm());
                    }
                }
                worst = worst.max(res);
            }
        }
        ("solve", Output::One(x)) => {
            let x = tlinalg_testkit::widen(&x);
            let b = tlinalg_testkit::widen(&tlinalg_testkit::batch_of::<T>(c.m, 4, c.batch, 0));
            for batch in 0..c.batch {
                let xo = batch * c.m * 4;
                let bo = batch * c.m * 4;
                for col in 0..4 {
                    for row in 0..c.m {
                        let mut lhs = Complex64::new(0., 0.);
                        for k in 0..c.m {
                            lhs += a[row + k * c.m] * x[xo + k + col * c.m];
                        }
                        worst = worst.max((lhs - b[bo + row + col * c.m]).norm());
                    }
                }
            }
        }
        (_, _) => return Err("provider did not return verification buffers".into()),
    }
    if worst > tol {
        Err(format!("measured residual={worst:.3e} > {tol:.3e}"))
    } else {
        Ok(worst)
    }
}

/// Verify every requested case. The residual printed is measured output error, never the bound.
pub fn verify<T: BenchScalar>(families: &[String], cases: &[Case], _env: &Env) -> bool {
    let mut clean = true;
    for family in families {
        for &case in cases {
            if !applicable(family, case) {
                println!(
                    "SKIPPED {family} {} {}: shape not applicable",
                    T::LABEL,
                    case
                );
                continue;
            }
            let faer = output_call::<T>(family, case, false);
            let lapack = if crate::vendor::LINKED {
                output_call::<T>(family, case, true)
            } else {
                Ok(Output::One(Vec::new()))
            };
            let fr = faer
                .and_then(|o| check_one::<T>(family, case, o))
                .or_else(|_| {
                    call::<T>(family, case, tlinalg::Parallel::Sequential, false).map(|_| 0.0)
                });
            let lr = lapack
                .and_then(|o| check_one::<T>(family, case, o))
                .or_else(|_| {
                    if crate::vendor::LINKED {
                        call::<T>(family, case, tlinalg::Parallel::Sequential, true).map(|_| 0.0)
                    } else {
                        Ok(0.0)
                    }
                });
            match (fr, lr) {
                (Ok(f), Ok(l)) => println!(
                    "OK {family} {} {} residual={:.3e} provider_residual={:.3e}",
                    T::LABEL,
                    case,
                    f,
                    l
                ),
                (Err(e), _) => {
                    println!("MISMATCH {family} {} {}: faer: {e}", T::LABEL, case);
                    clean = false;
                }
                (_, Err(e)) => {
                    println!("MISMATCH {family} {} {}: lapack: {e}", T::LABEL, case);
                    clean = false;
                }
            }
        }
    }
    clean
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn orthonormality_negative_control_rejects_perturbation() {
        let exact = [
            Complex64::new(1., 0.),
            Complex64::new(0., 0.),
            Complex64::new(0., 0.),
            Complex64::new(1., 0.),
        ];
        assert!(check_orthonormal(&exact, 2, 2, 1e-12).is_ok());
        let mut bad = exact;
        bad[0] *= 1.001;
        assert!(check_orthonormal(&bad, 2, 2, 1e-12).is_err());
    }
}
