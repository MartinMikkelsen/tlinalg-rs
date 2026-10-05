//! faer-backed partial-pivot LU factorization and the LU-based linear solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry points
//! are per matrix; the host keeps its batch iteration. The packed (LAPACK-format) LU family used by
//! prepared solves lives in [`crate::packed_lu`]; this module is the explicit-factor `P, L, U`
//! decomposition and the one-shot solve.
//!
//! Faer's work matrices, permutation vectors and scratch are operation-local, as before.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::Mat;
use strided_view::{RawStridedMut, RawStridedRef};

use crate::util::{checked_product, mat_mut, mat_ref, push_masked, push_permutation};
use crate::{faer_par, with_parallel, Error, FaerScalar, Op, Parallel, Result};

/// The explicit factors of a partial-pivot LU decomposition, as caller-provided buffers.
///
/// Each vector is cleared and then filled by `push`, column-major.
#[derive(Debug)]
pub struct LuFactors<'a, T> {
    /// The `m x m` row permutation `P`, with `A = P L U`.
    pub p: &'a mut Vec<T>,
    /// The unit-lower-trapezoidal `m x min(m, n)` factor `L`.
    pub l: &'a mut Vec<T>,
    /// The upper-trapezoidal `min(m, n) x n` factor `U`.
    pub u: &'a mut Vec<T>,
}

/// Partial-pivot LU of one `m x n` matrix, `A = P L U`.
///
/// Returns the permutation parity (`+1` or `-1` in the scalar type). An exactly singular input is
/// not an error.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `m x n` or an output size
/// overflows.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::lu::{lu, LuFactors};
/// use tlinalg::{Op, Parallel};
///
/// let a = [1.0_f64, 3.0, 2.0, 4.0];
/// let (mut p, mut l, mut u) = (Vec::new(), Vec::new(), Vec::new());
/// let parity = lu(
///     Op::Lu, 2, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     LuFactors { p: &mut p, l: &mut l, u: &mut u }, Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(parity, -1.0);
/// assert_eq!(u[0], 3.0);
/// ```
pub fn lu<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    input: RawStridedRef<'_, T>,
    factors: LuFactors<'_, T>,
    par: Parallel<'_>,
) -> Result<T> {
    let mat = mat_ref(op, "input", &input, m, n)?;
    let k = m.min(n);
    let _ = checked_product(op, "permutation matrix", &[m, m])?;
    let _ = checked_product(op, "L", &[m, k])?;
    let _ = checked_product(op, "U", &[k, n])?;
    let LuFactors { p, l, u } = factors;
    p.clear();
    l.clear();
    u.clear();

    let mut lu = Mat::<T::Entity>::zeros(m, n);
    lu.copy_from(mat);
    let mut perm = vec![0usize; m];
    let mut perm_inv = vec![0usize; m];
    let mut mem = MemBuffer::new(
        faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T::Entity>(
            m,
            n,
            faer_par(par),
            Default::default(),
        ),
    );
    let transpositions = with_parallel(par, |par| {
        faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut perm,
            &mut perm_inv,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        )
        .0
        .transposition_count
    });

    push_permutation(p, &perm_inv);
    for col in 0..k {
        for row in 0..m {
            l.push(if row == col {
                T::parity(false)
            } else if row > col {
                T::from_entity(lu[(row, col)])
            } else {
                T::default()
            });
        }
    }
    push_masked(u, lu.as_ref(), k, n, |row, col| row <= col);
    Ok(T::parity(transpositions % 2 != 0))
}

/// Solve one linear system `A X = B` (or `Aᵀ X = B`) through a partial-pivot LU of `A`, writing
/// the solution into a caller-owned destination.
///
/// `a` is `n x n`. `out` is the `n x nrhs` destination. When `rhs` is `Some(b)`, `b` (`n x nrhs`) is
/// copied into `out` first; when it is `None`, `out` already holds the right-hand side.
///
/// # Failure behaviour
///
/// The destination is written only after the factorization and the singularity check succeed, so
/// a failed call leaves `out` exactly as it was.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when a descriptor has the wrong shape, and [`Error::Singular`] when the
/// LU factor has an exactly zero pivot.
///
/// # Examples
///
/// ```
/// use strided_view::{RawStridedMut, RawStridedRef};
/// use tlinalg::{lu::solve, Op, Parallel};
///
/// let a = [2.0_f64, 0.0, 0.0, 4.0];
/// let b = [2.0_f64, 8.0];
/// let mut x = [0.0_f64; 2];
/// solve(
///     Op::Solve, 2, 1,
///     RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     Some(RawStridedRef::new(&b, &[2, 1], &[1, 2], 0).unwrap()),
///     RawStridedMut::new(&mut x, &[2, 1], &[1, 2], 0).unwrap(),
///     false, Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(x, [1.0, 2.0]);
/// ```
// INVARIANT: the argument list mirrors the host solve it replaces — coefficient, optional source,
// destination, transpose flag and token are distinct operands.
#[allow(clippy::too_many_arguments)]
pub fn solve<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    a: RawStridedRef<'_, T>,
    rhs: Option<RawStridedRef<'_, T>>,
    mut out: RawStridedMut<'_, T>,
    transpose_a: bool,
    par: Parallel<'_>,
) -> Result<()> {
    let a_mat = mat_ref(op, "A", &a, n, n)?;
    let source = match rhs.as_ref() {
        Some(b) => Some(mat_ref(op, "B", b, n, nrhs)?),
        None => None,
    };
    let mut destination = mat_mut(op, "out", &mut out, n, nrhs)?;

    let mut lu = Mat::<T::Entity>::zeros(n, n);
    lu.copy_from(a_mat);
    let mut row_perm = vec![0usize; n];
    let mut row_perm_inv = vec![0usize; n];
    let mut mem = MemBuffer::new(
        faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T::Entity>(
            n,
            n,
            faer_par(par),
            Default::default(),
        ),
    );
    with_parallel(par, |par| {
        let (_, perm) = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            lu.as_mut(),
            &mut row_perm,
            &mut row_perm_inv,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
        for i in 0..n {
            if lu[(i, i)] == <T::Entity as Default>::default() {
                return Err(Error::Singular { op });
            }
        }

        if let Some(source) = source {
            destination.copy_from(source);
        }
        let mut mem = MemBuffer::new(if transpose_a {
            faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place_scratch::<
                usize,
                T::Entity,
            >(n, nrhs, par)
        } else {
            faer::linalg::lu::partial_pivoting::solve::solve_in_place_scratch::<usize, T::Entity>(
                n, nrhs, par,
            )
        });
        let stack = MemStack::new(&mut mem);
        if transpose_a {
            faer::linalg::lu::partial_pivoting::solve::solve_transpose_in_place(
                lu.as_ref(),
                lu.as_ref(),
                perm,
                destination.as_mut(),
                par,
                stack,
            );
        } else {
            faer::linalg::lu::partial_pivoting::solve::solve_in_place(
                lu.as_ref(),
                lu.as_ref(),
                perm,
                destination.as_mut(),
                par,
                stack,
            );
        }
        Ok(())
    })
}
