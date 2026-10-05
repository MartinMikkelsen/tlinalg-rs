//! faer-backed batched Cholesky factorization.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! The factorization reads the **lower** triangle of each `A` and returns the lower-triangular `L`
//! with `A = L Lᴴ`, column-major, zero above the diagonal. Faer's work matrix and scratch are lane
//! scratch, reused for every item of the lane.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Mat, MatRef};
use strided_view::RawStridedRef;

use crate::batch::{self, out, BatchedRef, Push};
use crate::util::{checked_product, push_masked};
use crate::{Error, FaerScalar, LanePlan, Op, Parallel, Result};

struct CholeskyScratch<E: faer::traits::ComplexField> {
    work: Mat<E>,
    mem: MemBuffer,
}

fn cholesky_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    l: &mut impl Push<T>,
    scratch: &mut CholeskyScratch<T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let n = mat.nrows();
    scratch.work.copy_from(mat);
    faer::linalg::cholesky::llt::factor::cholesky_in_place(
        scratch.work.as_mut(),
        Default::default(),
        par,
        MemStack::new(&mut scratch.mem),
        Default::default(),
    )
    .map_err(|_| Error::NonConvergence { op })?;
    push_masked(l, scratch.work.as_ref(), n, n, |row, col| row >= col);
    Ok(())
}

/// Cholesky factors of every `n x n` Hermitian positive-definite matrix of a batch.
///
/// `input` is `[n, n, b_1, ..., b_B]`; `l` receives `n * n` elements per item.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` is not a batch of square matrices, and
/// [`Error::NonConvergence`] for the lowest-indexed item that is not numerically positive definite
/// (the classification the pre-extraction code reported). `l` is empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{cholesky::cholesky, LanePlan, Op, Parallel};
///
/// let a = [4.0_f64, 2.0, 2.0, 3.0];
/// let mut l = Vec::new();
/// cholesky(
///     Op::Cholesky, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), &mut l,
///     Parallel::Sequential, LanePlan::sequential(),
/// ).unwrap();
/// assert!((l[0] - 2.0).abs() < 1e-12 && l[2] == 0.0);
/// ```
pub fn cholesky<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    l: &mut Vec<T>,
    par: Parallel<'_>,
    plan: LanePlan<'_>,
) -> Result<()> {
    l.clear();
    let input = BatchedRef::square(op, "input", input)?;
    let n = input.rows();
    let l_len = checked_product(op, "L", &[n, n])?;
    batch::run(
        op,
        input.batch(),
        par,
        plan,
        &mut (out(l, l_len),),
        |par| CholeskyScratch {
            work: Mat::<T::Entity>::zeros(n, n),
            mem: MemBuffer::new(
                faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch::<T::Entity>(
                    n,
                    par,
                    Default::default(),
                ),
            ),
        },
        |index, (l,), scratch, par| cholesky_item::<T>(op, input.item(index), l, scratch, par),
    )
}
