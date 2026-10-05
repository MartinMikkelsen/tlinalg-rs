//! faer-backed full-pivot LU factorization and solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry points
//! are per matrix; the host keeps its batch iteration.
//!
//! The factorization never fails on an exactly singular input; the solve rejects an effectively
//! singular factor (a pivot at or below `ε · max|pivot|`), exactly as before. Faer's work matrix,
//! permutation vectors and scratch are operation-local.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Mat, MatMut};
use strided_view::RawStridedRef;

use crate::scalar::ScalarEntity;
use crate::util::{checked_product, invalid, mat_ref, push_masked, push_permutation};
use crate::{faer_par, with_parallel, Error, FaerScalar, Op, Parallel, Result};

/// The explicit factors of a full-pivot LU decomposition, as caller-provided buffers.
///
/// Each vector is cleared and then filled by `push`, column-major `n x n`.
#[derive(Debug)]
pub struct FullPivLuFactors<'a, T> {
    /// The row permutation `P`, with `A = P L U Q`.
    pub p: &'a mut Vec<T>,
    /// The unit-lower-triangular factor `L`.
    pub l: &'a mut Vec<T>,
    /// The upper-triangular factor `U`.
    pub u: &'a mut Vec<T>,
    /// The column permutation `Q`.
    pub q: &'a mut Vec<T>,
}

/// Full-pivot LU of one `n x n` matrix, `A = P L U Q`.
///
/// Returns the parity of the transposition count (`+1` or `-1` in the scalar type).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `n x n` or an output size
/// overflows.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::full_piv_lu::{full_piv_lu, FullPivLuFactors};
/// use tlinalg::{Op, Parallel};
///
/// let a = [1.0_f64, 0.0, 0.0, 2.0];
/// let (mut p, mut l, mut u, mut q) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
/// full_piv_lu(
///     Op::FullPivLu, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     FullPivLuFactors { p: &mut p, l: &mut l, u: &mut u, q: &mut q }, Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(u[0], 2.0);
/// ```
pub fn full_piv_lu<T: FaerScalar>(
    op: Op,
    n: usize,
    input: RawStridedRef<'_, T>,
    factors: FullPivLuFactors<'_, T>,
    par: Parallel<'_>,
) -> Result<T> {
    let mat = mat_ref(op, "input", &input, n, n)?;
    let _ = checked_product(op, "matrix", &[n, n])?;
    let FullPivLuFactors { p, l, u, q } = factors;
    p.clear();
    l.clear();
    u.clear();
    q.clear();

    let mut lu = Mat::<T::Entity>::zeros(n, n);
    lu.copy_from(mat);
    let mut row_perm = vec![0usize; n];
    let mut row_perm_inv = vec![0usize; n];
    let mut col_perm = vec![0usize; n];
    let mut col_perm_inv = vec![0usize; n];
    let mut mem = MemBuffer::new(
        faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T::Entity>(
            n,
            n,
            faer_par(par),
            Default::default(),
        ),
    );
    let transpositions = with_parallel(par, |par| {
        faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .0
        .transposition_count
    });

    push_permutation(p, &row_perm_inv);
    for col in 0..n {
        for row in 0..n {
            l.push(if row == col {
                T::parity(false)
            } else if row > col {
                T::from_entity(lu[(row, col)])
            } else {
                T::default()
            });
        }
    }
    push_masked(u, lu.as_ref(), n, n, |row, col| row <= col);
    push_permutation(q, &col_perm_inv);
    Ok(T::parity(transpositions % 2 != 0))
}

/// Solve one linear system through a full-pivot LU of `A`, in place.
///
/// `a` is `n x n`; `rhs` is the compact column-major `n x nrhs` right-hand side and is overwritten
/// with the solution of `A X = B` (or `Aᵀ X = B` when `transpose_a`).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `a` does not describe `n x n` or `rhs` does not hold
/// `n * nrhs` elements, and [`Error::Singular`] when a pivot is at or below `ε · max|pivot|` in
/// magnitude. `rhs` is untouched on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{full_piv_lu::full_piv_lu_solve, Op, Parallel};
///
/// let a = [2.0_f64, 0.0, 0.0, 4.0];
/// let mut x = [2.0_f64, 8.0];
/// full_piv_lu_solve(
///     Op::FullPivLuSolve, 2, 1, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     &mut x, false, Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
pub fn full_piv_lu_solve<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    a: RawStridedRef<'_, T>,
    rhs: &mut [T],
    transpose_a: bool,
    par: Parallel<'_>,
) -> Result<()> {
    let a_mat = mat_ref(op, "A", &a, n, n)?;
    if n.checked_mul(nrhs) != Some(rhs.len()) {
        return Err(invalid(
            op,
            "configuration",
            format!(
                "right-hand side holds {} elements, expected {n}x{nrhs}",
                rhs.len()
            ),
        ));
    }

    let mut lu = Mat::<T::Entity>::zeros(n, n);
    lu.copy_from(a_mat);
    let mut row_perm = vec![0usize; n];
    let mut row_perm_inv = vec![0usize; n];
    let mut col_perm = vec![0usize; n];
    let mut col_perm_inv = vec![0usize; n];
    let mut mem = MemBuffer::new(
        faer::linalg::lu::full_pivoting::factor::lu_in_place_scratch::<usize, T::Entity>(
            n,
            n,
            faer_par(par),
            Default::default(),
        ),
    );
    with_parallel(par, |par| {
        let (_, row_perm_ref, col_perm_ref) = faer::linalg::lu::full_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            &mut col_perm,
            &mut col_perm_inv,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
        let max_diagonal = (0..n)
            .map(|i| T::entity_magnitude(lu[(i, i)]))
            .fold(0.0, f64::max);
        for i in 0..n {
            if T::entity_magnitude(lu[(i, i)]) <= <T as ScalarEntity>::EPSILON * max_diagonal {
                return Err(Error::Singular { op });
            }
        }

        let matrix = MatMut::from_column_major_slice_mut(T::entity_slice_mut(rhs), n, nrhs);
        let mut mem = MemBuffer::new(
            faer::linalg::lu::full_pivoting::solve::solve_in_place_scratch::<usize, T::Entity>(
                n, nrhs, par,
            ),
        );
        let stack = MemStack::new(&mut mem);
        if transpose_a {
            faer::linalg::lu::full_pivoting::solve::solve_transpose_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                matrix,
                par,
                stack,
            );
        } else {
            faer::linalg::lu::full_pivoting::solve::solve_in_place(
                lu.as_ref(),
                lu.as_ref(),
                row_perm_ref,
                col_perm_ref,
                matrix,
                par,
                stack,
            );
        }
        Ok(())
    })
}
