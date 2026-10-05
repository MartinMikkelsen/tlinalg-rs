//! Batched Cholesky factorization on LAPACK (`?potrf`).
//!
//! Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0).

use strided_view::RawStridedRef;

use crate::batch::{batch_len, clear_on_error, Input};
use crate::common::{check_info, check_scratch, checked_product, dim_i32};
use crate::{Error, LapackScalar, Op, Result, Workspace};

/// Lower Cholesky factors of every Hermitian positive definite matrix of a batch.
///
/// `a` has dims `[n, n, batch...]` and is read through its lower triangle; it is not modified.
/// `out` is cleared and receives `n * n` elements per matrix, compact column-major in batch order,
/// with `L` in the lower triangle and zeros above it. One destructible `n x n` copy is acquired
/// from `workspace` for the whole call, reused per matrix and released on success.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a non-square or malformed operand or a dimension outside the
/// LAPACK `i32` range, and [`Error::NonConvergence`] for the first matrix (in batch order) that is
/// not positive definite. On error `out` is left empty.
pub fn cholesky<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    out: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    out.clear();
    let result = run(op, a, out, workspace);
    clear_on_error(result, || out.clear())
}

fn run<T, W>(op: Op, a: RawStridedRef<'_, T>, out: &mut Vec<T>, workspace: &mut W) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    let input = Input::new(op, "input", a)?;
    let n = input.square(op)?;
    let n_i32 = dim_i32(op, n)?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let count = input.layout.count;
    if matrix_len == 0 || count == 0 {
        return Ok(());
    }
    out.reserve(batch_len(op, matrix_len, count)?);
    let mut factor = workspace.acquire_capacity(matrix_len);
    for offset in input.layout.offsets() {
        factor.clear();
        input.gather(offset, &mut factor);
        check_scratch(op, factor.len(), matrix_len)?;
        let mut info = 0;
        // SAFETY: `factor` is a mutable compact column-major `n x n` matrix.
        unsafe {
            T::potrf(b'L', n_i32, &mut factor, n_i32, &mut info);
        }
        if info > 0 {
            return Err(Error::NonConvergence { op });
        }
        // The routine name is the one the host reported for every scalar type before the move.
        check_info(op, "dpotrf", info)?;
        for col in 0..n {
            out.extend(core::iter::repeat_n(T::default(), col));
            out.extend_from_slice(&factor[col + col * n..(col + 1) * n]);
        }
    }
    workspace.release(factor);
    Ok(())
}
