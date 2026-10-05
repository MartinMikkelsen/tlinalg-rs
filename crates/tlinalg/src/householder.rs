//! faer-backed compact Householder QR primitives.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). These are the two
//! numerical kernels of the compact (incremental) Householder QR state; the state operations built
//! on them — factor, append, from-factors, `R` and `Q` extraction with the positive-diagonal gauge —
//! stay in the host.
//!
//! # Compact state
//!
//! A compact factor stores, column-major `rows x cols`, `R` on and above the diagonal and the
//! Householder vectors below it (each with an implicit unit head). The coefficients are
//! `coeff[j] = 1 / τ_j` for faer's reflector `H_j = I - v_j v_jᴴ / τ_j`, a real value carried in the
//! scalar type. Faer's denominator form is rebuilt transiently where a reflector is applied.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::prelude::ReborrowMut;
use faer::{Conj, Mat, MatMut, MatRef};

use crate::util::{checked_product, invalid};
use crate::{with_parallel, FaerScalar, Op, Parallel, Result};

/// Factor a compact column-major `rows x cols` matrix in place into the compact Householder state.
///
/// On return `data` holds `R` on and above the diagonal and the reflector vectors below it, and
/// `coeff` (cleared, then filled by `push`) holds the `min(rows, cols)` coefficients.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `data` does not hold
/// `rows * cols` elements.
///
/// # Examples
///
/// ```
/// use tlinalg::{householder::compact_factor, Op, Parallel};
///
/// let mut a = [3.0_f64, 4.0];
/// let mut coeff = Vec::new();
/// compact_factor(Op::HouseholderQr, &mut a, 2, 1, &mut coeff, Parallel::Sequential).unwrap();
/// assert!((a[0].abs() - 5.0).abs() < 1e-12);
/// assert_eq!(coeff.len(), 1);
/// ```
pub fn compact_factor<T: FaerScalar>(
    op: Op,
    data: &mut [T],
    rows: usize,
    cols: usize,
    coeff: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    let expected = checked_product(op, "matrix", &[rows, cols])?;
    if data.len() != expected {
        return Err(invalid(
            op,
            "configuration",
            "matrix: input buffer length does not match dimensions",
        ));
    }
    coeff.clear();
    let k = rows.min(cols);
    if k == 0 {
        return Ok(());
    }
    let mut qr = MatMut::from_column_major_slice_mut(T::entity_slice_mut(data), rows, cols);
    with_parallel(par, |par| {
        for j in 0..k {
            let mut column = qr.rb_mut().col_mut(j).subrows_mut(j, rows - j);
            let (mut head, tail) = column.rb_mut().split_at_row_mut(1);
            let info = faer::linalg::householder::make_householder_in_place(&mut head[0], tail);
            let beta = head[0];
            head[0] = head_one::<T>();
            // Faer stores H = I - vv^H/tau; compact state stores 1/tau.
            coeff.push(T::from_real(T::recip_real(info.tau)));
            if j + 1 < cols {
                let mut basis = Mat::<T::Entity>::zeros(rows - j, 1);
                basis[(0, 0)] = head_one::<T>();
                for row in 1..rows - j {
                    basis[(row, 0)] = qr[(j + row, j)];
                }
                let factor = Mat::from_fn(1, 1, |_, _| T::entity_from_real(info.tau));
                let mut mem = MemBuffer::new(
                    faer::linalg::householder::apply_block_householder_on_the_left_in_place_scratch::<
                        T::Entity,
                    >(rows - j, 1, cols - j - 1),
                );
                faer::linalg::householder::apply_block_householder_on_the_left_in_place_with_conj(
                    basis.as_ref(),
                    factor.as_ref(),
                    Conj::No,
                    qr.rb_mut().submatrix_mut(j, j + 1, rows - j, cols - j - 1),
                    par,
                    MemStack::new(&mut mem),
                );
            }
            qr[(j, j)] = beta;
        }
    });
    Ok(())
}

/// The faer entity one, the implicit head of every stored reflector.
fn head_one<T: FaerScalar>() -> T::Entity {
    T::entity_from_real(one_real::<T>())
}

/// The real one, without a numeric-literal bound on the real type.
fn one_real<T: FaerScalar>() -> T::Real {
    T::parity(false).real_part()
}

/// Apply the first `k` reflectors of a compact state to a compact column-major `rows x cols`
/// matrix in place: `C ← Q C`, or `C ← Qᴴ C` when `transpose`.
///
/// `a` is the compact `rows x a_cols` state and `coeff` its `k` coefficients.
///
/// # Errors
///
/// [`Error::InvalidArgument`](crate::Error::InvalidArgument) when `k` exceeds `rows` or `a_cols`,
/// `coeff` does not hold `k` values, or a buffer length does not match its dimensions.
///
/// # Examples
///
/// ```
/// use tlinalg::householder::{apply_reflectors, compact_factor};
/// use tlinalg::{Op, Parallel};
///
/// let mut a = [3.0_f64, 4.0];
/// let mut coeff = Vec::new();
/// compact_factor(Op::HouseholderQr, &mut a, 2, 1, &mut coeff, Parallel::Sequential).unwrap();
/// // Q applied to e1 is the first column of Q, which is ±(3, 4)/5.
/// let mut c = [1.0_f64, 0.0];
/// apply_reflectors(Op::HouseholderQrQColumns, &a, 1, &coeff, &mut c, 2, 1, 1, false, Parallel::Sequential).unwrap();
/// assert!((c[0].abs() - 0.6).abs() < 1e-12);
/// ```
// INVARIANT: these buffers and dimensions mirror the host reflector ABI the call replaces.
#[allow(clippy::too_many_arguments)]
pub fn apply_reflectors<T: FaerScalar>(
    op: Op,
    a: &[T],
    a_cols: usize,
    coeff: &[T],
    c: &mut [T],
    rows: usize,
    cols: usize,
    k: usize,
    transpose: bool,
    par: Parallel<'_>,
) -> Result<()> {
    if k > rows || k > a_cols || coeff.len() != k {
        return Err(invalid(
            op,
            "configuration",
            "dimensions: reflector count exceeds matrix dimensions",
        ));
    }
    if a.len() != checked_product(op, "A", &[rows, a_cols])?
        || c.len() != checked_product(op, "C", &[rows, cols])?
    {
        return Err(invalid(
            op,
            "configuration",
            "matrix: input buffer length does not match dimensions",
        ));
    }
    if rows == 0 || cols == 0 || k == 0 {
        return Ok(());
    }
    let basis = MatRef::from_column_major_slice(T::entity_slice(a), rows, a_cols).subcols(0, k);
    // Rebuild faer's denominator form transiently from the state coefficients.
    let factors = Mat::from_fn(1, k, |_, col| {
        T::entity_from_real(T::recip_real(coeff[col].real_part()))
    });
    let mut matrix = MatMut::from_column_major_slice_mut(T::entity_slice_mut(c), rows, cols);
    let scratch = if transpose {
        faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<T::Entity>(rows, 1, cols)
    } else {
        faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<
            T::Entity,
        >(rows, 1, cols)
    };
    let mut mem = MemBuffer::new(scratch);
    with_parallel(par, |par| {
        let stack = MemStack::new(&mut mem);
        if transpose {
            // `Qᴴ`: the conjugate of the transposed sequence. The real scalars are their own
            // conjugate, so this is the plain transpose for them, as before.
            faer::linalg::householder::apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                basis, factors.as_ref(), Conj::Yes, matrix.as_mut(), par, stack,
            );
        } else {
            faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                basis, factors.as_ref(), Conj::No, matrix.as_mut(), par, stack,
            );
        }
    });
    Ok(())
}
