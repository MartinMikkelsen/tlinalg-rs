//! Batched packed partial-pivot LU on LAPACK: factor, prepared solve, and fused factor+solve.
//!
//! Moved from `tenferro-linalg`'s LAPACK provider (same project, MIT OR Apache-2.0). The kernels
//! keep the LAPACK packed format — unit-lower `L` below the diagonal, `U` on and above it, one-based
//! row-swap pivots — so the factors stay interchangeable with the faer-backed implementation.
//!
//! # Host contract
//!
//! Every entry point operates on one **chunk** of whole matrices, exactly as the faer-backed
//! implementation does, so the host can drive either implementation with the same partition. The
//! parallelism token is accepted for interface parity and **ignored**: LAPACK owns its threading,
//! and the batch loop here is serial on purpose.
//!
//! Nothing in this family takes pooled buffers, so it needs no [`tlinalg_traits::Workspace`].

use tlinalg_traits::{Error, Op, Parallel, Result};

use crate::LapackScalar;

/// A shape product overflowed `usize`.
fn checked_product(op: Op, role: &'static str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| Error::InvalidArgument {
            op,
            role: "configuration",
            detail: format!("{role} element count overflows usize"),
        })
}

/// A dimension LAPACK cannot express in its `i32` interface.
fn dim_i32(op: Op, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument {
        op,
        role: "dimension",
        detail: format!("dimension {value} exceeds the LAPACK i32 range"),
    })
}

/// Translate a LAPACK `info` into this crate's error vocabulary.
///
/// `info < 0` is an illegal argument; `info > 0` is a numerical failure. Callers pass
/// `info.min(0)` where a positive value is meaningful rather than a failure — an exactly singular
/// `?getrf` is not an error, and the packed factors stay valid for a caller that checks the `U`
/// diagonal.
fn check_info(op: Op, routine: &'static str, info: i32) -> Result<()> {
    if info < 0 {
        return Err(Error::InvalidArgument {
            op,
            role: "lapack_argument",
            detail: format!("LAPACK {routine} argument {} had an illegal value", -info),
        });
    }
    if info > 0 {
        return Err(Error::NonConvergence { op });
    }
    Ok(())
}

/// Reject a pivot outside `1..=n`.
///
/// `?getrs` applies `ipiv` through `?laswp` without bounds checks, so every stored pivot must be a
/// one-based row index in `1..=n` before the call. A host that splits a batch into chunks can call
/// this on the whole batch first.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] with role `"pivot"`.
pub fn validate_pivots(op: Op, n: usize, ipiv: &[i32]) -> Result<()> {
    for &pivot_one_based in ipiv {
        let in_range = usize::try_from(pivot_one_based)
            .map(|pivot| (1..=n).contains(&pivot))
            .unwrap_or(false);
        if !in_range {
            return Err(Error::InvalidArgument {
                op,
                role: "pivot",
                detail: format!("LU pivot index {pivot_one_based} is outside 1..={n}"),
            });
        }
    }
    Ok(())
}

/// Check that every `(per_matrix_len, buffer_len)` pair describes the same number of matrices.
fn check_batches(op: Op, buffers: [(usize, usize); 3], batch: usize) -> Result<()> {
    for ((per_matrix, len), what) in buffers
        .into_iter()
        .zip(["packed LU", "pivots", "rhs batch"])
    {
        if len != checked_product(op, what, &[per_matrix, batch])? {
            return Err(Error::Inconsistent {
                op,
                detail: "packed LU, pivot, and rhs buffers describe different batches",
            });
        }
    }
    Ok(())
}

/// Factor one matrix in place and write its parity.
fn factor_one<T: LapackScalar>(
    op: Op,
    m_i32: i32,
    n_i32: i32,
    matrix: &mut [T],
    ipiv: &mut [i32],
    parity: Option<&mut T>,
    reject_singular: bool,
) -> Result<()> {
    let mut info = 0;
    // SAFETY: `matrix` is a compact column-major `m x n` block of this call's `lu`, `ipiv` holds
    // `min(m, n)` pivots, and `m_i32`/`n_i32` were derived from those lengths.
    unsafe {
        T::getrf(m_i32, n_i32, matrix, m_i32, ipiv, &mut info);
    }
    check_info(op, "getrf", info.min(0))?;
    if reject_singular && info > 0 {
        return Err(Error::Singular { op });
    }
    if let Some(parity) = parity {
        let swaps = ipiv
            .iter()
            .enumerate()
            .filter(|(index, pivot)| **pivot != (*index as i32 + 1))
            .count();
        *parity = if swaps.is_multiple_of(2) {
            T::one()
        } else {
            T::minus_one()
        };
    }
    Ok(())
}

/// Factor every compact column-major `m x n` matrix of `lu` in place.
///
/// `pivots` receives `min(m, n)` one-based pivots per matrix and `parity` one permutation parity per
/// matrix. Exactly singular matrices are **not** an error: `?getrf` reports them through a positive
/// `info`, and the packed factors stay usable.
///
/// # Errors
///
/// Returns [`Error::Inconsistent`] when the buffers describe different batches,
/// [`Error::InvalidArgument`] for a dimension outside the LAPACK `i32` range or an illegal LAPACK
/// argument, and [`Error::NonConvergence`] for a positive `info`.
pub fn factor_chunk<T: LapackScalar>(
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
    _par: Parallel<'_>,
) -> Result<()> {
    let k = m.min(n);
    let matrix_len = checked_product(op, "matrix shape", &[m, n])?;
    let batch = parity.len();
    check_batches(
        op,
        [(matrix_len, lu.len()), (k, pivots.len()), (1, parity.len())],
        batch,
    )?;
    if batch == 0 || matrix_len == 0 {
        return Ok(());
    }
    let m_i32 = dim_i32(op, m)?;
    let n_i32 = dim_i32(op, n)?;
    // INVARIANT: the buffers were checked above to hold exactly `batch` matrices, pivot vectors and
    // parities, and the early return gives `matrix_len > 0`, hence `k > 0`, so the chunk iterators
    // stay in lockstep. The serial loop is intentional: LAPACK owns threading inside `?getrf`, and
    // one call per matrix writes straight into the caller's buffers without scratch.
    for ((matrix, ipiv), parity) in lu
        .chunks_exact_mut(matrix_len)
        .zip(pivots.chunks_exact_mut(k))
        .zip(parity.iter_mut())
    {
        factor_one::<T>(op, m_i32, n_i32, matrix, ipiv, Some(parity), false)?;
    }
    Ok(())
}

/// Solve `op(A) X = B` for every matrix of the chunk from packed `?getrf` factors.
///
/// `output` enters holding the compact column-major RHS batch and leaves holding the solution.
/// `op(A)` is `A`, `A^T`, `A^H`, or `conj(A)` from the flags: the first three map onto `?getrs` with
/// `trans = N/T/C`, and `conj(A) x = b` is solved as `A conj(x) = conj(b)`.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for a pivot outside `1..=n`, a dimension outside the LAPACK
/// `i32` range, inconsistent buffer lengths, or an illegal LAPACK argument, and
/// [`Error::NonConvergence`] for a positive `info`. Pivots are validated for the whole chunk
/// **before** any output is written.
// INVARIANT: the argument list mirrors the host call it replaces; each value is a distinct operand
// or flag, so grouping them would add a wrapper without removing an argument.
#[allow(clippy::too_many_arguments)]
pub fn solve_prepared_chunk<T: LapackScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
    _par: Parallel<'_>,
) -> Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 || rhs_len == 0 {
        return Ok(());
    }
    let batch = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch,
    )?;
    validate_pivots(op, n, pivots)?;
    let n_i32 = dim_i32(op, n)?;
    let nrhs_i32 = dim_i32(op, nrhs)?;
    let (trans, conjugate_rhs) = match (transpose_a, conjugate_a) {
        (false, false) => (b'N', false),
        (true, false) => (b'T', false),
        (true, true) => (b'C', false),
        (false, true) => (b'N', true),
    };
    if conjugate_rhs {
        T::conj_in_place(output);
    }
    // INVARIANT: the buffers were checked above to hold exactly `batch` nonempty matrices, pivot
    // vectors and RHS blocks, and every pivot is in `1..=n`, so each `?getrs` reads only its own
    // factors and writes only its own RHS block.
    for ((matrix, ipiv), rhs) in packed_lu
        .chunks_exact(matrix_len)
        .zip(pivots.chunks_exact(n))
        .zip(output.chunks_exact_mut(rhs_len))
    {
        let mut info = 0;
        // SAFETY: `matrix` and `ipiv` are this item's matching `?getrf` factor, `rhs` is its
        // `n x nrhs` right-hand side, and the dimensions were derived from those lengths.
        unsafe {
            T::getrs(
                trans, n_i32, nrhs_i32, matrix, n_i32, ipiv, rhs, n_i32, &mut info,
            );
        }
        check_info(op, "getrs", info)?;
    }
    if conjugate_rhs {
        T::conj_in_place(output);
    }
    Ok(())
}

/// Factor and solve `A X = B` for every matrix of the chunk, keeping the packed factors.
///
/// `packed_lu` enters holding the compact `A` batch and leaves holding the packed factors;
/// `pivots` receives the one-based pivots; `output` enters holding the RHS batch and leaves holding
/// `X`. One `?getrf` and one `?getrs` per matrix, with no scratch because the factors are an output.
///
/// # Errors
///
/// Returns [`Error::Singular`] when a matrix is exactly singular and there is a nonempty RHS to
/// solve, [`Error::Inconsistent`] for inconsistent buffers, [`Error::InvalidArgument`] for a
/// dimension outside the LAPACK `i32` range or an illegal LAPACK argument, and
/// [`Error::NonConvergence`] for a positive `info` during a solve.
// INVARIANT: the argument list mirrors the host call it replaces (see `solve_prepared_chunk`).
#[allow(clippy::too_many_arguments)]
pub fn factor_solve_chunk<T: LapackScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
    _par: Parallel<'_>,
) -> Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 {
        return Ok(());
    }
    let batch = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch,
    )?;
    let n_i32 = dim_i32(op, n)?;
    let nrhs_i32 = dim_i32(op, nrhs)?;
    if rhs_len == 0 {
        // Nothing to solve: only factor, matching `factor_chunk` on singular input.
        for (matrix, ipiv) in packed_lu
            .chunks_exact_mut(matrix_len)
            .zip(pivots.chunks_exact_mut(n))
        {
            factor_one::<T>(op, n_i32, n_i32, matrix, ipiv, None, false)?;
        }
        return Ok(());
    }
    for ((matrix, ipiv), rhs) in packed_lu
        .chunks_exact_mut(matrix_len)
        .zip(pivots.chunks_exact_mut(n))
        .zip(output.chunks_exact_mut(rhs_len))
    {
        factor_one::<T>(op, n_i32, n_i32, matrix, ipiv, None, true)?;
        let mut info = 0;
        // SAFETY: this item's factor, pivots and right-hand side, as the prepared solve validates.
        unsafe {
            T::getrs(
                b'N', n_i32, nrhs_i32, matrix, n_i32, ipiv, rhs, n_i32, &mut info,
            );
        }
        check_info(op, "getrs", info)?;
    }
    Ok(())
}
