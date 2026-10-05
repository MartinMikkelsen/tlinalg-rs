//! faer-backed general (non-Hermitian) eigendecomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry points
//! are per matrix; the host keeps its batch iteration.
//!
//! # Conventions
//!
//! The outputs are always complex: `f32`/`Complex32` input yields `Complex32`, `f64`/`Complex64`
//! input yields `Complex64`. For real input, faer returns real Schur-form eigenpairs; they are
//! converted exactly as before:
//!
//! * [`eig`] treats an eigenvalue whose imaginary part is at most `ε · max(|re|, 1)` as real (zero
//!   imaginary part, real eigenvector); otherwise it emits the conjugate pair `re ± i·im` with the
//!   eigenvectors `u_j ± i·u_{j+1}`.
//! * [`eig_values`] copies the real and imaginary parts as faer returned them, without that test.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::{Mat, MatRef};
use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;

use crate::scalar::{EigScalar, ScalarEntity};
use crate::util::{checked_product, mat_ref};
use crate::{faer_par, with_parallel, Error, FaerScalar, Op, Parallel, Result};

/// Eigenvalues and eigenvectors of one `n x n` matrix, `A V = V diag(w)`.
///
/// `values` (`n`) and `vectors` (column-major `n x n`) are cleared and filled by `push` in the
/// complex type of the input scalar.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `n x n` or the vector count
/// overflows, and [`Error::NonConvergence`] when faer fails to converge.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use strided_view::RawStridedRef;
/// use tlinalg::{eig::eig, Op, Parallel};
///
/// // Rotation by 90 degrees: eigenvalues ±i.
/// let a = [0.0_f64, 1.0, -1.0, 0.0];
/// let (mut w, mut v) = (Vec::<Complex64>::new(), Vec::new());
/// eig(Op::Eig, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w, &mut v, Parallel::Sequential).unwrap();
/// assert!((w[0].im.abs() - 1.0).abs() < 1e-12 && (w[0].conj() - w[1]).norm() < 1e-12);
/// ```
pub fn eig<T: FaerScalar>(
    op: Op,
    n: usize,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Complex>,
    vectors: &mut Vec<<T as ScalarEntity>::Complex>,
    par: Parallel<'_>,
) -> Result<()> {
    let mat = mat_ref(op, "input", &input, n, n)?;
    let _ = checked_product(op, "eigenvector matrix", &[n, n])?;
    values.clear();
    vectors.clear();
    // faer's eigensolvers do not accept an empty matrix; there is nothing to decompose.
    if n == 0 {
        return Ok(());
    }
    T::eig_into(op, mat, values, vectors, par)
}

/// Eigenvalues of one `n x n` matrix, without the eigenvectors.
///
/// `values` is cleared and filled with `n` values in the complex type of the input scalar.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `n x n`, and
/// [`Error::NonConvergence`] when faer fails to converge.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use strided_view::RawStridedRef;
/// use tlinalg::{eig::eig_values, Op, Parallel};
///
/// let a = [2.0_f64, 0.0, 0.0, 3.0];
/// let mut w = Vec::<Complex64>::new();
/// eig_values(Op::EigValues, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w, Parallel::Sequential).unwrap();
/// assert_eq!(w.len(), 2);
/// ```
pub fn eig_values<T: FaerScalar>(
    op: Op,
    n: usize,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Complex>,
    par: Parallel<'_>,
) -> Result<()> {
    let mat = mat_ref(op, "input", &input, n, n)?;
    values.clear();
    // faer's eigensolvers do not accept an empty matrix; there is nothing to decompose.
    if n == 0 {
        return Ok(());
    }
    T::eig_values_into(op, mat, values, par)
}

/// `|im| <= ε · max(|re|, 1)`, the test for treating a real-input eigenvalue as real.
fn imag_is_effectively_zero(real: f64, imag: f64, eps: f64) -> bool {
    imag.abs() <= eps * real.abs().max(1.0)
}

macro_rules! impl_eig_real {
    ($real:ty, $complex:ty) => {
        impl EigScalar for $real {
            fn eig_into(
                op: Op,
                mat: MatRef<'_, $real>,
                values: &mut Vec<$complex>,
                vectors: &mut Vec<$complex>,
                par: Parallel<'_>,
            ) -> Result<()> {
                let n = mat.nrows();
                let mut u_real = Mat::<$real>::zeros(n, n);
                let mut s_re = Diag::<$real>::zeros(n);
                let mut s_im = Diag::<$real>::zeros(n);
                let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$real>(
                    n,
                    ComputeEigenvectors::No,
                    ComputeEigenvectors::Yes,
                    faer_par(par),
                    Default::default(),
                ));
                with_parallel(par, |par| {
                    faer::linalg::evd::evd_real(
                        mat,
                        s_re.as_mut(),
                        s_im.as_mut(),
                        None,
                        Some(u_real.as_mut()),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| Error::NonConvergence { op })
                })?;
                let mut j = 0;
                while j < n {
                    if j + 1 >= n
                        || imag_is_effectively_zero(
                            s_re[j] as f64,
                            s_im[j] as f64,
                            <$real>::EPSILON as f64,
                        )
                    {
                        values.push(<$complex>::new(s_re[j], 0.0));
                        for i in 0..n {
                            vectors.push(<$complex>::new(u_real[(i, j)], 0.0));
                        }
                        j += 1;
                    } else {
                        values.push(<$complex>::new(s_re[j], s_im[j]));
                        values.push(<$complex>::new(s_re[j], -s_im[j]));
                        for i in 0..n {
                            vectors.push(<$complex>::new(u_real[(i, j)], u_real[(i, j + 1)]));
                        }
                        for i in 0..n {
                            vectors.push(<$complex>::new(u_real[(i, j)], -u_real[(i, j + 1)]));
                        }
                        j += 2;
                    }
                }
                Ok(())
            }

            fn eig_values_into(
                op: Op,
                mat: MatRef<'_, $real>,
                values: &mut Vec<$complex>,
                par: Parallel<'_>,
            ) -> Result<()> {
                let n = mat.nrows();
                let mut s_re = Diag::<$real>::zeros(n);
                let mut s_im = Diag::<$real>::zeros(n);
                let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$real>(
                    n,
                    ComputeEigenvectors::No,
                    ComputeEigenvectors::No,
                    faer_par(par),
                    Default::default(),
                ));
                with_parallel(par, |par| {
                    faer::linalg::evd::evd_real(
                        mat,
                        s_re.as_mut(),
                        s_im.as_mut(),
                        None,
                        None,
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| Error::NonConvergence { op })
                })?;
                for j in 0..n {
                    values.push(<$complex>::new(s_re[j], s_im[j]));
                }
                Ok(())
            }
        }
    };
}

macro_rules! impl_eig_complex {
    ($complex:ty, $entity:ty) => {
        impl EigScalar for $complex {
            fn eig_into(
                op: Op,
                mat: MatRef<'_, $entity>,
                values: &mut Vec<$complex>,
                vectors: &mut Vec<$complex>,
                par: Parallel<'_>,
            ) -> Result<()> {
                let n = mat.nrows();
                let mut u = Mat::<$entity>::zeros(n, n);
                let mut s = Diag::<$entity>::zeros(n);
                let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$entity>(
                    n,
                    ComputeEigenvectors::No,
                    ComputeEigenvectors::Yes,
                    faer_par(par),
                    Default::default(),
                ));
                with_parallel(par, |par| {
                    faer::linalg::evd::evd_cplx(
                        mat,
                        s.as_mut(),
                        None,
                        Some(u.as_mut()),
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| Error::NonConvergence { op })
                })?;
                for j in 0..n {
                    values.push(<$complex>::new(s[j].re, s[j].im));
                }
                for col in 0..n {
                    for row in 0..n {
                        let value = u[(row, col)];
                        vectors.push(<$complex>::new(value.re, value.im));
                    }
                }
                Ok(())
            }

            fn eig_values_into(
                op: Op,
                mat: MatRef<'_, $entity>,
                values: &mut Vec<$complex>,
                par: Parallel<'_>,
            ) -> Result<()> {
                let n = mat.nrows();
                let mut s = Diag::<$entity>::zeros(n);
                let mut mem = MemBuffer::new(faer::linalg::evd::evd_scratch::<$entity>(
                    n,
                    ComputeEigenvectors::No,
                    ComputeEigenvectors::No,
                    faer_par(par),
                    Default::default(),
                ));
                with_parallel(par, |par| {
                    faer::linalg::evd::evd_cplx(
                        mat,
                        s.as_mut(),
                        None,
                        None,
                        par,
                        MemStack::new(&mut mem),
                        Default::default(),
                    )
                    .map_err(|_| Error::NonConvergence { op })
                })?;
                for j in 0..n {
                    values.push(<$complex>::new(s[j].re, s[j].im));
                }
                Ok(())
            }
        }
    };
}

impl_eig_real!(f32, Complex32);
impl_eig_real!(f64, Complex64);
impl_eig_complex!(Complex32, faer::c32);
impl_eig_complex!(Complex64, faer::c64);
