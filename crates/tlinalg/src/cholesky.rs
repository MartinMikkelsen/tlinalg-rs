//! faer-backed Cholesky factorization.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry point is
//! per matrix; the host keeps its batch iteration.
//!
//! The factorization reads the **lower** triangle of `A` and returns the lower-triangular `L` with
//! `A = L Lᴴ`, column-major, zero above the diagonal. Faer's work matrix and scratch are
//! operation-local, as before, so this family needs no [`crate::Workspace`].

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::Mat;
use strided_view::RawStridedRef;

use crate::util::{checked_product, mat_ref, push_masked};
use crate::{faer_par, with_parallel, Error, FaerScalar, Op, Parallel, Result};

/// Cholesky factor of one `n x n` Hermitian positive-definite matrix.
///
/// `l` is cleared and filled by `push` with the column-major `n x n` factor.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `n x n`, and
/// [`Error::NonConvergence`] when the matrix is not numerically positive definite (the
/// classification the pre-extraction code reported).
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{cholesky::cholesky, Op, Parallel};
///
/// let a = [4.0_f64, 2.0, 2.0, 3.0];
/// let mut l = Vec::new();
/// cholesky(Op::Cholesky, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut l, Parallel::Sequential).unwrap();
/// assert!((l[0] - 2.0).abs() < 1e-12 && l[2] == 0.0);
/// ```
pub fn cholesky<T: FaerScalar>(
    op: Op,
    n: usize,
    input: RawStridedRef<'_, T>,
    l: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    let mat = mat_ref(op, "input", &input, n, n)?;
    let _ = checked_product(op, "L", &[n, n])?;
    l.clear();
    let mut work = Mat::<T::Entity>::zeros(n, n);
    work.copy_from(mat);
    let mut mem = MemBuffer::new(
        faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T::Entity>(
            n,
            faer_par(par),
            Default::default(),
        ),
    );
    with_parallel(par, |par| {
        faer::linalg::cholesky::llt::factor::cholesky_in_place(
            work.as_mut(),
            Default::default(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .map(|_| ())
        .map_err(|_| Error::NonConvergence { op })
    })?;
    push_masked(l, work.as_ref(), n, n, |row, col| row >= col);
    Ok(())
}
