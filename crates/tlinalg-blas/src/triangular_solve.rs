//! Batched triangular solve on LAPACK/BLAS.
//!
//! Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0): a left-side solve
//! is one `?trtrs` per matrix, a right-side solve one `cblas_?trsm`, in a serial loop.

use strided_view::RawStridedRef;

use crate::batch::{batch_len, clear_on_error, Input};
use crate::common::{check_info, checked_product, dim_i32};
use crate::{Error, LapackScalar, Op, Result, Workspace};

/// Which triangular system to solve.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TriangularSolveOptions {
    /// Solve `op(A) X = B` when true, `X op(A) = B` when false.
    pub left_side: bool,
    /// Read the lower triangle of `A` when true, the upper one when false.
    pub lower: bool,
    /// Use `Aᵀ` (not `Aᴴ`) instead of `A`.
    pub transpose_a: bool,
    /// Treat the diagonal of `A` as all ones without reading it.
    pub unit_diagonal: bool,
}

/// Solve every triangular system of a batch.
///
/// `a` has dims `[n, n, batch...]` and `b` dims `[rows, cols, batch...]` with the same batch dims;
/// `rows == n` for a left-side solve and `cols == n` for a right-side one. Neither is modified.
/// `out` is cleared and receives `rows * cols` elements per system, compact column-major in batch
/// order. `A` is read in place when its layout is LAPACK-readable (unit row stride); otherwise one
/// compact `n x n` copy is acquired from `workspace` for the whole call and released on success.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for malformed or mismatched operands or a dimension outside the
/// LAPACK `i32` range, [`Error::Singular`] when a right-side, non-unit-diagonal matrix has an
/// exactly zero diagonal entry (BLAS `?trsm` would divide by it), and [`Error::NonConvergence`] for
/// a positive `?trtrs` `info`; always for the first failing system in batch order. On error `out`
/// is left empty.
pub fn triangular_solve<T, W>(
    op: Op,
    options: TriangularSolveOptions,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    out: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    out.clear();
    let result = run(op, options, a, b, out, workspace);
    clear_on_error(result, || out.clear())
}

fn run<T, W>(
    op: Op,
    options: TriangularSolveOptions,
    a: RawStridedRef<'_, T>,
    b: RawStridedRef<'_, T>,
    out: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    let a = Input::new(op, "a", a)?;
    let b = Input::new(op, "b", b)?;
    let n = a.square(op)?;
    let (rows, cols) = (b.layout.rows, b.layout.cols);
    let rhs_core_dim = if options.left_side { rows } else { cols };
    if rhs_core_dim != n {
        return Err(Error::InvalidArgument {
            op,
            role: "shape",
            detail: format!("the right-hand side core dim {rhs_core_dim} does not match {n}"),
        });
    }
    a.layout.same_batch(op, &b.layout)?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[rows, cols])?;
    let n_i32 = dim_i32(op, n)?;
    let rows_i32 = dim_i32(op, rows)?;
    let cols_i32 = dim_i32(op, cols)?;
    let count = b.layout.count;
    if matrix_len == 0 || rhs_len == 0 || count == 0 {
        return Ok(());
    }
    out.reserve(batch_len(op, rhs_len, count)?);
    let TriangularSolveOptions {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    } = options;
    let mut scratch = if a.layout.lapack_lda().is_none() {
        Some(workspace.acquire_capacity(matrix_len))
    } else {
        None
    };
    // INVARIANT: both operands describe `count` matrices in the same batch order. Each provider
    // call reads its own triangle and overwrites only its own block of `out`. The serial loop is
    // intentional: the BLAS/LAPACK provider owns threading.
    for (a_offset, b_offset) in a.layout.offsets().zip(b.layout.offsets()) {
        let (matrix, lda): (&[T], usize) = match scratch.as_mut() {
            Some(compact) => {
                compact.clear();
                a.gather(a_offset, compact);
                (compact.as_slice(), n)
            }
            None => a.in_place(a_offset).expect("layout checked above"),
        };
        let start = out.len();
        b.gather(b_offset, out);
        let rhs = &mut out[start..];
        let lda_i32 = dim_i32(op, lda)?;
        if left_side {
            let mut info = 0;
            // SAFETY: `matrix` holds an `n x n` triangle with leading dimension `lda` and `rhs` a
            // compact `n x cols` right-hand side.
            unsafe {
                T::trtrs(
                    if lower { b'L' } else { b'U' },
                    if transpose_a { b'T' } else { b'N' },
                    if unit_diagonal { b'U' } else { b'N' },
                    n_i32,
                    cols_i32,
                    matrix,
                    lda_i32,
                    rhs,
                    n_i32,
                    &mut info,
                );
            }
            check_info(op, "trtrs", info)?;
        } else {
            if !unit_diagonal && (0..n).any(|idx| matrix[idx + idx * lda] == T::default()) {
                return Err(Error::Singular { op });
            }
            // SAFETY: `matrix` holds an `n x n` triangle with leading dimension `lda` and `rhs` a
            // compact `rows x n` matrix.
            unsafe {
                T::trsm_right(
                    lower,
                    transpose_a,
                    unit_diagonal,
                    rows_i32,
                    n_i32,
                    matrix,
                    lda_i32,
                    rhs,
                    rows_i32,
                );
            }
        }
    }
    if let Some(compact) = scratch {
        workspace.release(compact);
    }
    Ok(())
}
