//! Batched compact Householder QR kernels on LAPACK: factor (`?geqrf`) and reflector application
//! (`?ormqr`/`?unmqr`).
//!
//! Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0). These are the two
//! numerical kernels under the host's compact/incremental QR state; the state bookkeeping (append,
//! from-factors, `R` and `Q`-column extraction) stays in the host.
//!
//! This is an in-place route: the matrices are compact, column-major and batch-contiguous
//! `&mut [T]` buffers covering the whole batch, and the workspace is queried once per call and
//! reused for every item. The host passes [`Op::HouseholderFactor`] and [`Op::HouseholderApply`],
//! whose names are the ones these kernels reported before the move.

use crate::common::{
    check_info, check_len, check_query_info, check_scratch, checked_product, dim_i32, work_len,
};
use crate::{Error, LapackScalar, Op, Result, Workspace};

/// The number of whole `item_len` blocks in `len`, rejecting a remainder.
fn whole_items(op: Op, role: &'static str, len: usize, item_len: usize) -> Result<usize> {
    if !len.is_multiple_of(item_len) {
        return Err(Error::InvalidArgument {
            op,
            role,
            detail: format!("expected a multiple of {item_len} elements, got {len}"),
        });
    }
    Ok(len / item_len)
}

/// Factor every compact column-major `rows x cols` matrix of the batch `data` in place.
///
/// `data` leaves holding, per matrix, the compact reflectors below the diagonal and `R` on and
/// above it. `tau` is cleared and receives `min(rows, cols)` coefficients per matrix. The query slot
/// and one work buffer are acquired from `workspace` for the whole call and released on success. An
/// empty matrix (or an empty batch) leaves `tau` empty without touching `workspace`.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `data` is not whole matrices (role `"matrix"`), for a dimension
/// outside the LAPACK `i32` range or for an illegal LAPACK argument, [`Error::InvalidWorkspace`]
/// for an unusable workspace size, and [`Error::NonConvergence`] for a positive `info`; always for
/// the first failing matrix in batch order. On error `tau` is left empty.
pub fn factor<T, W>(
    op: Op,
    rows: usize,
    cols: usize,
    data: &mut [T],
    tau: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    tau.clear();
    let result = factor_run(op, rows, cols, data, tau, workspace);
    if result.is_err() {
        tau.clear();
    }
    result
}

fn factor_run<T, W>(
    op: Op,
    rows: usize,
    cols: usize,
    data: &mut [T],
    tau: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    let k = rows.min(cols);
    let matrix_len = checked_product(op, "matrix", &[rows, cols])?;
    if matrix_len == 0 {
        check_len(op, "matrix", data.len(), 0)?;
        return Ok(());
    }
    let count = whole_items(op, "matrix", data.len(), matrix_len)?;
    if count == 0 {
        return Ok(());
    }
    let rows_i32 = dim_i32(op, rows)?;
    let cols_i32 = dim_i32(op, cols)?;
    // The coefficients are written by LAPACK through a slice, so they are initialised first.
    tau.resize(
        checked_product(op, "coefficients", &[k, count])?,
        T::default(),
    );
    let mut query = workspace.acquire_zeroed(1);
    check_scratch(op, query.len(), 1)?;
    let mut info = 0;
    // SAFETY: the first matrix covers `rows * cols` elements, `tau` has at least `k` entries, and
    // `lwork = -1` writes only the query slot.
    unsafe {
        T::geqrf(
            rows_i32,
            cols_i32,
            &mut data[..matrix_len],
            rows_i32,
            &mut tau[..k],
            &mut query,
            -1,
            &mut info,
        );
    }
    check_query_info(op, T::GEQRF, info)?;
    let lwork = work_len(op, T::GEQRF, T::work_query_len(query[0]))?;
    let mut work = workspace.acquire_zeroed(lwork as usize);
    check_scratch(op, work.len(), lwork as usize)?;
    // INVARIANT: `data` holds `count` whole matrices and `tau` `count` coefficient blocks; the
    // workspace depends only on the shape and is reused. LAPACK owns threading.
    for (matrix, coefficients) in data
        .chunks_exact_mut(matrix_len)
        .zip(tau.chunks_exact_mut(k))
    {
        // SAFETY: as for the query, with the queried workspace length.
        unsafe {
            T::geqrf(
                rows_i32,
                cols_i32,
                matrix,
                rows_i32,
                coefficients,
                &mut work,
                lwork,
                &mut info,
            );
        }
        check_info(op, T::GEQRF, info)?;
    }
    workspace.release(query);
    workspace.release(work);
    Ok(())
}

/// Apply the first `k` compact reflectors of each `a` to the matching `c` from the left, in place,
/// for every item of a batch.
///
/// Per item, `a` holds the reflectors in a column-major `m x a_cols` matrix, `tau` `k`
/// coefficients, and `c` a column-major `m x p` matrix; the three buffers are batch-contiguous and
/// describe the same number of items. With `transpose`, `Qᴴ` (`Qᵀ` for real scalars) is applied
/// instead of `Q`. The query slot and one work buffer come from `workspace` and are released after
/// the last application **whether or not it succeeded**, as before the move.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `k` exceeds `a_cols` or `m` (role `"dimensions"`), when the
/// buffers are not whole items or disagree about the item count (roles `"A"`, `"tau"`, `"C"`),
/// for a dimension outside the LAPACK `i32` range or for an illegal LAPACK argument,
/// [`Error::InvalidWorkspace`] for an unusable workspace size, and [`Error::NonConvergence`] for a
/// positive `info`; always for the first failing item in batch order. On error the items before the
/// failing one have been transformed.
// INVARIANT: the argument list mirrors the host call it replaces; each value is a distinct operand.
#[allow(clippy::too_many_arguments)]
pub fn apply_reflectors<T, W>(
    op: Op,
    m: usize,
    a_cols: usize,
    p: usize,
    k: usize,
    transpose: bool,
    a: &[T],
    tau: &[T],
    c: &mut [T],
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    if k > a_cols || k > m {
        return Err(Error::InvalidArgument {
            op,
            role: "dimensions",
            detail: "reflector count exceeds matrix dimensions".to_owned(),
        });
    }
    let a_len = checked_product(op, "A", &[m, a_cols])?;
    let c_len = checked_product(op, "C", &[m, p])?;
    if m == 0 || p == 0 || k == 0 {
        return Ok(());
    }
    let count = whole_items(op, "C", c.len(), c_len)?;
    check_len(op, "A", a.len(), checked_product(op, "A", &[a_len, count])?)?;
    check_len(
        op,
        "tau",
        tau.len(),
        checked_product(op, "tau", &[k, count])?,
    )?;
    if count == 0 {
        return Ok(());
    }
    let m_i32 = dim_i32(op, m)?;
    let p_i32 = dim_i32(op, p)?;
    let k_i32 = dim_i32(op, k)?;
    let mut query = workspace.acquire_zeroed(1);
    check_scratch(op, query.len(), 1)?;
    let mut info = 0;
    let trans = if transpose { T::ADJOINT } else { b'N' };
    // SAFETY: the first item's `A`, `tau` and `C` satisfy the LAPACK dimensions; `lwork = -1`
    // writes only the query slot.
    unsafe {
        T::ormqr(
            b'L',
            trans,
            m_i32,
            p_i32,
            k_i32,
            &a[..a_len],
            m_i32,
            &tau[..k],
            &mut c[..c_len],
            m_i32,
            &mut query,
            -1,
            &mut info,
        );
    }
    check_query_info(op, T::ORMQR, info)?;
    let lwork = work_len(op, T::ORMQR, T::work_query_len(query[0]))?;
    let mut work = workspace.acquire_zeroed(lwork as usize);
    check_scratch(op, work.len(), lwork as usize)?;
    let mut result = Ok(());
    for ((reflectors, coefficients), target) in a
        .chunks_exact(a_len)
        .zip(tau.chunks_exact(k))
        .zip(c.chunks_exact_mut(c_len))
    {
        // SAFETY: the checked dimensions and the queried workspace cover the whole application;
        // `A` and `tau` are read-only and `C` is uniquely mutable.
        unsafe {
            T::ormqr(
                b'L',
                trans,
                m_i32,
                p_i32,
                k_i32,
                reflectors,
                m_i32,
                coefficients,
                target,
                m_i32,
                &mut work,
                lwork,
                &mut info,
            );
        }
        result = check_info(op, T::ORMQR, info);
        if result.is_err() {
            break;
        }
    }
    workspace.release(query);
    workspace.release(work);
    result
}
