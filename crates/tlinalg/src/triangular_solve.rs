//! faer-backed triangular solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry point is
//! per matrix; the host keeps its batch iteration.
//!
//! # Boundary
//!
//! The right-hand side is an **owned** column-major buffer the solve consumes, because every
//! triangular solve overwrites its right-hand side: the caller decides once where those elements
//! come from (a compact tensor, or a borrowed view gathered straight into a pooled buffer). The
//! result is returned as a buffer too:
//!
//! * a left-side solve `A X = B` overwrites `rhs` in place and returns it;
//! * a right-side solve `X A = B` is the left-side solve of the transposed system, so the
//!   right-hand side is transposed into a buffer from the [`Workspace`], solved, and transposed
//!   back into a second one. The two consumed buffers are [`Workspace::release`]d, exactly as the
//!   pre-extraction code returned them to the host pool.

use faer::{MatMut, MatRef};
use strided_view::RawStridedRef;

use crate::util::{invalid, mat_ref};
use crate::{with_parallel, FaerScalar, Op, Parallel, Result, Workspace};

/// Which triangle and which variant of the coefficient matrix a triangular solve uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TriangularSolveFlags {
    /// Solve `A X = B` when `true`, `X A = B` when `false`.
    pub left_side: bool,
    /// `A` is stored in its lower triangle when `true`, its upper triangle when `false`.
    pub lower: bool,
    /// Use `Aᵀ` instead of `A`.
    pub transpose_a: bool,
    /// Treat the diagonal of `A` as ones without reading it.
    pub unit_diagonal: bool,
}

/// Dispatch faer's triangular solve from the triangle/transpose/unit flags.
///
/// Transposing `A` swaps which triangle is stored, so the four faer routines cover all eight flag
/// combinations once that flip is applied.
fn solve_in_place<E: faer::traits::ComplexField>(
    a: MatRef<'_, E>,
    rhs: MatMut<'_, E>,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
    par: faer::Par,
) {
    let effective_lower = lower != transpose_a;
    let a = if transpose_a { a.transpose() } else { a };
    match (effective_lower, unit_diagonal) {
        (true, false) => {
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(a, rhs, par);
        }
        (true, true) => {
            faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place(a, rhs, par);
        }
        (false, false) => {
            faer::linalg::triangular_solve::solve_upper_triangular_in_place(a, rhs, par);
        }
        (false, true) => {
            faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place(a, rhs, par);
        }
    }
}

/// Transpose a column-major `rows x cols` buffer into a capacity buffer from the workspace.
fn transpose_into<T: FaerScalar, W: Workspace<T> + ?Sized>(
    workspace: &mut W,
    data: &[T],
    rows: usize,
    cols: usize,
) -> Vec<T> {
    let mut transposed = workspace.acquire_capacity(data.len());
    for j in 0..rows {
        for i in 0..cols {
            transposed.push(data[j + i * rows]);
        }
    }
    transposed
}

/// Solve one triangular system with an `n x n` coefficient matrix.
///
/// `rhs` is the column-major `b_rows x b_cols` right-hand side and is consumed. The returned buffer
/// is the column-major solution: `n x b_cols` for a left-side solve, `b_rows x n` for a right-side
/// one.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `a` does not describe `n x n`, when
/// `rhs` does not hold `b_rows * b_cols` elements, or when the right-hand side's solved dimension is
/// not `n` (`b_rows` for a left-side solve, `b_cols` for a right-side one). A host that reports a
/// shape mismatch for that case checks it before calling. On error `rhs` is dropped, not released.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::triangular_solve::{triangular_solve, TriangularSolveFlags};
/// use tlinalg::{Op, Parallel, Workspace};
///
/// struct Heap;
/// impl Workspace<f64> for Heap {
///     fn acquire_zeroed(&mut self, len: usize) -> Vec<f64> { vec![0.0; len] }
///     fn acquire_capacity(&mut self, cap: usize) -> Vec<f64> { Vec::with_capacity(cap) }
///     fn acquire_uninit(&mut self, len: usize) -> Vec<core::mem::MaybeUninit<f64>> {
///         (0..len).map(|_| core::mem::MaybeUninit::uninit()).collect()
///     }
///     fn release(&mut self, _buf: Vec<f64>) {}
/// }
///
/// // Lower-triangular A = [[2, 0], [1, 1]], B = [2, 3].
/// let a = [2.0_f64, 1.0, 0.0, 1.0];
/// let flags = TriangularSolveFlags { left_side: true, lower: true, transpose_a: false, unit_diagonal: false };
/// let x = triangular_solve(
///     Op::TriangularSolve, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     vec![2.0, 3.0], 2, 1, flags, &mut Heap, Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
// INVARIANT: the argument list mirrors the host core it replaces — coefficient descriptor, owned
// right-hand side with its shape, flags, scratch source and token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn triangular_solve<T: FaerScalar, W: Workspace<T> + ?Sized>(
    op: Op,
    n: usize,
    a: RawStridedRef<'_, T>,
    mut rhs: Vec<T>,
    b_rows: usize,
    b_cols: usize,
    flags: TriangularSolveFlags,
    workspace: &mut W,
    par: Parallel<'_>,
) -> Result<Vec<T>> {
    let a_mat = mat_ref(op, "A", &a, n, n)?;
    if b_rows.checked_mul(b_cols) != Some(rhs.len()) {
        return Err(invalid(
            op,
            "configuration",
            format!(
                "right-hand side holds {} elements, expected {b_rows}x{b_cols}",
                rhs.len()
            ),
        ));
    }
    let TriangularSolveFlags {
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    } = flags;
    if left_side {
        if b_rows != n {
            return Err(invalid(
                op,
                "configuration",
                format!("right-hand side has {b_rows} rows, expected {n}"),
            ));
        }
        let matrix = MatMut::from_column_major_slice_mut(T::entity_slice_mut(&mut rhs), n, b_cols);
        with_parallel(par, |par| {
            solve_in_place(a_mat, matrix, lower, transpose_a, unit_diagonal, par)
        });
        Ok(rhs)
    } else {
        if b_cols != n {
            return Err(invalid(
                op,
                "configuration",
                format!("right-hand side has {b_cols} columns, expected {n}"),
            ));
        }
        // Right-side solve `X A = B` is the left-side solve of the transposed system, so the RHS is
        // transposed in and out and the triangle/transpose flags flip once.
        let nrhs = b_rows;
        let mut transposed = transpose_into(workspace, &rhs, nrhs, n);
        workspace.release(rhs);
        let matrix =
            MatMut::from_column_major_slice_mut(T::entity_slice_mut(&mut transposed), n, nrhs);
        with_parallel(par, |par| {
            solve_in_place(a_mat, matrix, lower, !transpose_a, unit_diagonal, par)
        });
        let result = transpose_into(workspace, &transposed, n, nrhs);
        workspace.release(transposed);
        Ok(result)
    }
}
