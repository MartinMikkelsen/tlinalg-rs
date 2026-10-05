//! faer-backed Hermitian (self-adjoint) eigendecomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry points
//! are per matrix; the host keeps its batch iteration.
//!
//! The decomposition reads the **lower** triangle and returns non-decreasing eigenvalues. As with
//! [`crate::svd`], [`eigh`] carries the (real) eigenvalues **in the scalar type itself** — complex
//! values with a zero imaginary part for the complex scalars — because that is what the
//! pre-extraction code produced; [`eigh_values`] returns them in the real type, matching the
//! values-only path.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::ComputeEigenvectors;
use faer::Mat;
use strided_view::RawStridedRef;

use crate::scalar::ScalarEntity;
use crate::util::{checked_product, mat_ref, push_mat};
use crate::{faer_par, with_parallel, Error, FaerScalar, Op, Parallel, Result};

/// Eigenvalues of one `n x n` Hermitian matrix, without the vectors.
///
/// `values` is cleared and filled with `n` non-decreasing real eigenvalues.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `n x n`, and
/// [`Error::NonConvergence`] when faer fails to converge.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{eigh::eigh_values, Op, Parallel};
///
/// let a = [2.0_f64, 0.0, 0.0, 1.0];
/// let mut w = Vec::new();
/// eigh_values(Op::EighValues, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w, Parallel::Sequential).unwrap();
/// assert_eq!(w, [1.0, 2.0]);
/// ```
pub fn eigh_values<T: FaerScalar>(
    op: Op,
    n: usize,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<<T as ScalarEntity>::Real>,
    par: Parallel<'_>,
) -> Result<()> {
    let mat = mat_ref(op, "input", &input, n, n)?;
    values.clear();
    // faer's eigensolvers do not accept an empty matrix; there is nothing to decompose.
    if n == 0 {
        return Ok(());
    }
    let mut diag = Diag::<T::Entity>::zeros(n);
    let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<T::Entity>(
        n,
        ComputeEigenvectors::No,
        faer_par(par),
        Default::default(),
    ));
    with_parallel(par, |par| {
        faer::linalg::evd::self_adjoint_evd(
            mat,
            diag.as_mut(),
            None,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| Error::NonConvergence { op })
    })?;
    for index in 0..n {
        values.push(T::real_from_entity(diag[index]));
    }
    Ok(())
}

/// Eigendecomposition of one `n x n` Hermitian matrix, `A = V diag(w) Vᴴ`.
///
/// `values` is cleared and filled with the `n` non-decreasing eigenvalues in the scalar type (zero
/// imaginary part for the complex scalars); `vectors` with the column-major `n x n` eigenvectors.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `n x n` or the vector count
/// overflows, and [`Error::NonConvergence`] when faer fails to converge.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{eigh::eigh, Op, Parallel};
///
/// let a = [2.0_f64, 0.0, 0.0, 1.0];
/// let (mut w, mut v) = (Vec::new(), Vec::new());
/// eigh(Op::Eigh, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut w, &mut v, Parallel::Sequential).unwrap();
/// assert_eq!(w, [1.0, 2.0]);
/// assert_eq!(v.len(), 4);
/// ```
pub fn eigh<T: FaerScalar>(
    op: Op,
    n: usize,
    input: RawStridedRef<'_, T>,
    values: &mut Vec<T>,
    vectors: &mut Vec<T>,
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
    let mut diag = Diag::<T::Entity>::zeros(n);
    let mut v_mat = Mat::<T::Entity>::zeros(n, n);
    let mut mem = MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<T::Entity>(
        n,
        ComputeEigenvectors::Yes,
        faer_par(par),
        Default::default(),
    ));
    with_parallel(par, |par| {
        faer::linalg::evd::self_adjoint_evd(
            mat,
            diag.as_mut(),
            Some(v_mat.as_mut()),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map_err(|_| Error::NonConvergence { op })
    })?;
    for index in 0..n {
        values.push(T::from_real(T::real_from_entity(diag[index])));
    }
    push_mat(vectors, v_mat.as_ref());
    Ok(())
}
