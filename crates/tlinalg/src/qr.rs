//! faer-backed thin QR and column-pivoted (rank-revealing) QR.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry points
//! are per matrix; the host keeps its batch iteration.
//!
//! Both return the **raw** faer factors: no sign/phase gauge is applied, matching the
//! pre-extraction `qr` and `rank_revealing_qr` paths (the positive-diagonal gauge of the compact
//! Householder family is applied by the host when it extracts `Q` and `R`). For the rank-revealing
//! variant the rank decision (tolerances, the non-finite screen, the all-zero shortcut) stays in
//! the host; [`magnitude`] is the diagonal measure it was computed with.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Conj, Mat};
use strided_view::RawStridedRef;

use crate::util::{checked_product, invalid, mat_ref, push_masked, push_mat};
use crate::{with_parallel, FaerScalar, Op, Parallel, Result};

/// Thin QR of one `m x n` matrix, `A = Q R`.
///
/// `q` is cleared and filled with the column-major `m x min(m, n)` factor with orthonormal columns;
/// `r` with the column-major upper-trapezoidal `min(m, n) x n` factor.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when the descriptor does not describe
/// `m x n` or an output size overflows.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{qr::qr, Op, Parallel};
///
/// let a = [3.0_f64, 4.0];
/// let (mut q, mut r) = (Vec::new(), Vec::new());
/// qr(Op::Qr, 2, 1, RawStridedRef::new(&a, &[2, 1], &[1, 2], 0).unwrap(), &mut q, &mut r, Parallel::Sequential).unwrap();
/// assert!((r[0].abs() - 5.0).abs() < 1e-12);
/// ```
pub fn qr<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    input: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    let mat = mat_ref(op, "input", &input, m, n)?;
    let k = m.min(n);
    let _ = checked_product(op, "Q", &[m, k])?;
    let _ = checked_product(op, "R", &[k, n])?;
    q.clear();
    r.clear();
    let block_size =
        faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T::Entity>(m, n);
    let mut work = Mat::<T::Entity>::zeros(m, n);
    work.copy_from(mat);
    let mut coeff = Mat::<T::Entity>::zeros(block_size, k);
    let mut q_mat = Mat::<T::Entity>::identity(m, k);
    with_parallel(par, |par| {
        let mut mem = MemBuffer::new(
            faer::linalg::qr::no_pivoting::factor::qr_in_place_scratch::<T::Entity>(
                m,
                n,
                block_size,
                par,
                Default::default(),
            ),
        );
        faer::linalg::qr::no_pivoting::factor::qr_in_place(
            work.as_mut(),
            coeff.as_mut(),
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
        let mut mem = MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<
                T::Entity,
            >(m, block_size, k),
        );
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            work.as_ref().subcols(0, k),
            coeff.as_ref(),
            Conj::No,
            q_mat.as_mut(),
            par,
            MemStack::new(&mut mem),
        );
    });
    push_mat(q, q_mat.as_ref());
    push_masked(r, work.as_ref(), k, n, |row, col| row <= col);
    Ok(())
}

/// The magnitude a rank decision compares against its tolerance: `|x|` for the real scalars and
/// `hypot(re, im)` evaluated in `f64` for the complex ones.
///
/// A host reading the diagonal of the `R` returned by [`rank_revealing_qr`] gets exactly the values
/// the pre-extraction rank decision used.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// assert_eq!(tlinalg::qr::magnitude(Complex64::new(3.0, 4.0)), 5.0);
/// assert_eq!(tlinalg::qr::magnitude(-2.0_f32), 2.0);
/// ```
#[must_use]
pub fn magnitude<T: FaerScalar>(value: T) -> f64 {
    T::entity_magnitude(T::entity_slice(core::slice::from_ref(&value))[0])
}

/// Column-pivoted QR of one `m x n` matrix, `A P = Q R`.
///
/// `q` (`m x min(m, n)`) and `r` (`min(m, n) x n`, upper trapezoidal) are cleared and filled by
/// `push`, column-major. The returned vector is the forward column permutation in the public
/// gather convention: column `j` of `A P` is column `permutation[j]` of `A`.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when the descriptor does not describe
/// `m x n`, an output size overflows, or a column index does not fit `i64`.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{qr::rank_revealing_qr, Op, Parallel};
///
/// let a = [1.0_f64, 0.0, 0.0, 5.0];
/// let (mut q, mut r) = (Vec::new(), Vec::new());
/// let perm = rank_revealing_qr(
///     Op::RankRevealingQr, 2, 2, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(),
///     &mut q, &mut r, Parallel::Sequential,
/// ).unwrap();
/// assert_eq!(perm, [1, 0]);
/// ```
pub fn rank_revealing_qr<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    input: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<Vec<i64>> {
    let mat = mat_ref(op, "input", &input, m, n)?;
    let k = m.min(n);
    let _ = checked_product(op, "Q", &[m, k])?;
    let _ = checked_product(op, "R", &[k, n])?;
    q.clear();
    r.clear();
    let block_size =
        faer::linalg::qr::no_pivoting::factor::recommended_block_size::<T::Entity>(m, n);
    let mut work = Mat::<T::Entity>::zeros(m, n);
    work.copy_from(mat);
    let mut coeff = Mat::<T::Entity>::zeros(block_size, k);
    let mut permutation = vec![0usize; n];
    let mut inverse_permutation = vec![0usize; n];
    let mut q_mat = Mat::<T::Entity>::identity(m, k);
    with_parallel(par, |par| {
        // Following faer 0.24's public column-pivoted QR factor API; the provider's forward
        // permutation is the public gather convention.
        let mut mem = MemBuffer::new(
            faer::linalg::qr::col_pivoting::factor::qr_in_place_scratch::<usize, T::Entity>(
                m,
                n,
                block_size,
                par,
                Default::default(),
            ),
        );
        faer::linalg::qr::col_pivoting::factor::qr_in_place(
            work.as_mut(),
            coeff.as_mut(),
            &mut permutation,
            &mut inverse_permutation,
            par,
            MemStack::new(&mut mem),
            Default::default(),
        );
        let mut mem = MemBuffer::new(
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<
                T::Entity,
            >(m, block_size, k),
        );
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
            work.as_ref().subcols(0, k),
            coeff.as_ref(),
            Conj::No,
            q_mat.as_mut(),
            par,
            MemStack::new(&mut mem),
        );
    });
    push_mat(q, q_mat.as_ref());
    push_masked(r, work.as_ref(), k, n, |row, col| row <= col);
    // Collecting a `Vec<usize>` into a `Vec<i64>` of the same layout reuses the allocation, as the
    // pre-extraction conversion did.
    permutation
        .into_iter()
        .map(|column| {
            i64::try_from(column)
                .map_err(|_| invalid(op, "configuration", "column permutation exceeds i64 range"))
        })
        .collect()
}
