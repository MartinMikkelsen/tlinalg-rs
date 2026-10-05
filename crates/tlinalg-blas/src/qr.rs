//! Batched thin QR and rank-revealing (column-pivoted) QR on LAPACK.
//!
//! Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0). Both entry points
//! query their workspace once per call and reuse it for every matrix. The rank decision of the
//! rank-revealing QR stays in the host, which reads it off the returned `R` diagonal (see
//! [`crate::magnitude`]).

use strided_view::RawStridedRef;

use crate::batch::{batch_len, Input};
use crate::common::{
    check_info, check_query_info, check_scratch, checked_product, dim_i32, work_len,
};
use crate::{Error, LapackScalar, NonFiniteRole, Op, Result, Workspace};

/// Thin QR, `A = Q R`, of every matrix of a batch.
///
/// `a` has dims `[m, n, batch...]` and is not modified; `k = min(m, n)`. `q` is cleared and receives
/// the `m x k` orthonormal factors, `r` the `k x n` upper-trapezoidal factors, both compact
/// column-major in batch order. One destructible copy, `tau` and one work buffer (sized by both
/// queries) are acquired from `workspace` for the whole call and released on success.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a malformed operand, a dimension outside the LAPACK `i32` range or
/// an illegal LAPACK argument, [`Error::InvalidWorkspace`] for an unusable workspace size, and
/// [`Error::NonConvergence`] for a positive `info`; always for the first failing matrix in batch
/// order. On error `q` and `r` are left empty.
pub fn qr<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    q.clear();
    r.clear();
    let result = qr_run(op, a, q, r, workspace);
    if result.is_err() {
        q.clear();
        r.clear();
    }
    result
}

fn qr_run<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    q: &mut Vec<T>,
    r: &mut Vec<T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T>,
{
    let input = Input::new(op, "input", a)?;
    let (m, n) = (input.layout.rows, input.layout.cols);
    let k = m.min(n);
    let count = input.layout.count;
    let (mi, ni, ki) = (dim_i32(op, m)?, dim_i32(op, n)?, dim_i32(op, k)?);
    let matrix_len = checked_product(op, "matrix", &[m, n])?;
    let q_len = checked_product(op, "Q matrix", &[m, k])?;
    let r_len = checked_product(op, "R matrix", &[k, n])?;
    if count == 0 || matrix_len == 0 {
        return Ok(());
    }
    q.reserve(batch_len(op, q_len, count)?);
    r.reserve(batch_len(op, r_len, count)?);
    let mut packed = workspace.acquire_capacity(matrix_len);
    let mut offsets = input.layout.offsets();
    let first = offsets.next().unwrap_or_default();
    input.gather(first, &mut packed);
    let mut tau = workspace.acquire_zeroed(k);
    check_scratch(op, tau.len(), k)?;
    // A stack slot, as before the move: the queries write one value each.
    let mut query = [T::default()];
    let mut info = 0;
    // SAFETY: `packed` is a compact `m x n` matrix, `tau` has `k` entries, and `lwork = -1` writes
    // only the query slot.
    unsafe {
        T::geqrf(mi, ni, &mut packed, mi, &mut tau, &mut query, -1, &mut info);
    }
    check_info(op, T::GEQRF, info)?;
    let factor_len = work_len(op, "QR workspace", T::work_query_len(query[0]))?;
    // SAFETY: the first `m * k` entries hold `k` reflectors of an `m x k` matrix and `k <= m`;
    // `lwork = -1` writes only the query slot.
    unsafe {
        T::orgqr(
            mi,
            ki,
            ki,
            &mut packed[..q_len],
            mi,
            &tau,
            &mut query,
            -1,
            &mut info,
        );
    }
    check_info(op, T::ORGQR, info)?;
    let lwork = factor_len.max(work_len(op, "QR workspace", T::work_query_len(query[0]))?);
    let mut work = workspace.acquire_zeroed(lwork as usize);
    check_scratch(op, work.len(), lwork as usize)?;
    // INVARIANT: the reduced `Q` occupies the first `m * k` entries of `packed`, `k <= n`. Every
    // matrix shares the queried dimensions and scratch, and the scratch never aliases the input.
    // LAPACK owns threading; the batch loop only prepares provider calls.
    for offset in core::iter::once(first).chain(offsets) {
        packed.clear();
        input.gather(offset, &mut packed);
        // SAFETY: as for the queries, with the queried workspace length.
        unsafe {
            T::geqrf(
                mi,
                ni,
                &mut packed,
                mi,
                &mut tau,
                &mut work,
                lwork,
                &mut info,
            );
        }
        check_info(op, T::GEQRF, info)?;
        push_leading_upper(&packed, m, k, n, r);
        // SAFETY: as for the queries, with the queried workspace length.
        unsafe {
            T::orgqr(
                mi,
                ki,
                ki,
                &mut packed[..q_len],
                mi,
                &tau,
                &mut work,
                lwork,
                &mut info,
            );
        }
        check_info(op, T::ORGQR, info)?;
        q.extend_from_slice(&packed[..q_len]);
    }
    workspace.release(packed);
    workspace.release(tau);
    workspace.release(work);
    Ok(())
}

/// Push the leading `k x n` upper triangle of a column-major factor with leading dimension `m`.
fn push_leading_upper<T: LapackScalar>(data: &[T], m: usize, k: usize, n: usize, out: &mut Vec<T>) {
    for col in 0..n {
        let diag = k.min(col + 1);
        out.extend_from_slice(&data[col * m..col * m + diag]);
        out.extend(core::iter::repeat_n(T::default(), k - diag));
    }
}

/// The batched outputs of [`rank_revealing_qr`], each cleared and then filled in batch order.
#[derive(Debug)]
pub struct RankRevealingQrOutputs<'o, T> {
    /// `m x k` orthonormal factors, column-major.
    pub q: &'o mut Vec<T>,
    /// `k x n` upper-trapezoidal factors with non-increasing diagonal magnitude, column-major.
    pub r: &'o mut Vec<T>,
    /// `n` zero-based column indices per matrix: column `j` of `A P` is column `permutation[j]`
    /// of `A`.
    pub permutation: &'o mut Vec<i64>,
}

impl<T> RankRevealingQrOutputs<'_, T> {
    fn clear(&mut self) {
        self.q.clear();
        self.r.clear();
        self.permutation.clear();
    }
}

/// Column-pivoted QR, `A P = Q R`, of every matrix of a batch with `?geqp3`.
///
/// `a` has dims `[m, n, batch...]` and is not modified; `k = min(m, n)`. An all-zero matrix is not
/// factored: it gets the leading `k` identity columns as `Q`, a zero `R` and the identity
/// permutation, exactly the result the host built for it before the move (its rank is zero by the
/// host's rule). The `?geqp3` and `?orgqr`/`?ungqr` workspaces are queried once (on the first
/// matrix that is factored), acquired once (with the `2n` real `rwork` for complex scalars), reused
/// for every matrix and released on success.
///
/// # Errors
///
/// [`Error::NonFinite`] with [`NonFiniteRole::Input`] for a non-finite entry,
/// [`Error::InvalidArgument`] for a malformed operand, a dimension outside the LAPACK `i32` range,
/// an illegal LAPACK argument or an invalid `?geqp3` permutation (role `"column_permutation"`),
/// [`Error::InvalidWorkspace`] for an unusable workspace size, and [`Error::NonConvergence`] for a
/// positive `info`; always for the first failing matrix in batch order. On error every output is
/// empty.
pub fn rank_revealing_qr<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    mut outputs: RankRevealingQrOutputs<'_, T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real>,
{
    outputs.clear();
    let result = rrqr_run(op, a, &mut outputs, workspace);
    if result.is_err() {
        outputs.clear();
    }
    result
}

/// The `?geqp3` and `?orgqr` scratch of one call, sized by the first factored matrix.
struct RrqrScratch<T, R> {
    geqp3_lwork: i32,
    orgqr_lwork: i32,
    work: Vec<T>,
    rwork: Vec<R>,
}

fn rrqr_run<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    out: &mut RankRevealingQrOutputs<'_, T>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real>,
{
    let input = Input::new(op, "input", a)?;
    let (m, n) = (input.layout.rows, input.layout.cols);
    let k = m.min(n);
    let count = input.layout.count;
    let (mi, ni, ki) = (dim_i32(op, m)?, dim_i32(op, n)?, dim_i32(op, k)?);
    let matrix_len = checked_product(op, "matrix", &[m, n])?;
    let q_len = checked_product(op, "Q matrix", &[m, k])?;
    let r_len = checked_product(op, "R matrix", &[k, n])?;
    if count == 0 {
        return Ok(());
    }
    out.q.reserve(batch_len(op, q_len, count)?);
    out.r.reserve(batch_len(op, r_len, count)?);
    out.permutation.reserve(batch_len(op, n, count)?);
    let mut qr = workspace.acquire_capacity(matrix_len);
    let mut tau = vec![T::default(); k];
    let mut jpvt = vec![0_i32; n];
    // Reused per item by the permutation check.
    let mut seen = vec![false; n];
    let mut scratch: Option<RrqrScratch<T, T::Real>> = None;
    for offset in input.layout.offsets() {
        qr.clear();
        input.gather(offset, &mut qr);
        if qr.iter().any(|&value| !value.is_finite()) {
            return Err(Error::NonFinite {
                op,
                role: NonFiniteRole::Input,
            });
        }
        if qr.iter().all(|&value| value.magnitude() == 0.0) {
            let start = out.q.len();
            out.q.resize(start + q_len, T::default());
            for diagonal in 0..k {
                out.q[start + diagonal + diagonal * m] = T::one();
            }
            out.r.resize(out.r.len() + r_len, T::default());
            out.permutation.extend((0..n).map(|column| column as i64));
            continue;
        }
        jpvt.fill(0);
        if scratch.is_none() {
            scratch = Some(rrqr_scratch(
                op, mi, ni, ki, &mut qr, &mut jpvt, &mut tau, workspace,
            )?);
        }
        let Some(RrqrScratch {
            geqp3_lwork,
            orgqr_lwork,
            work,
            rwork,
        }) = scratch.as_mut()
        else {
            unreachable!("the scratch was just built");
        };
        let mut info = 0;
        // SAFETY: the dimensions and workspaces were validated by the queries on this shape.
        unsafe {
            T::geqp3(
                mi,
                ni,
                &mut qr,
                mi,
                &mut jpvt,
                &mut tau,
                work,
                *geqp3_lwork,
                rwork,
                &mut info,
            );
        }
        check_info(op, T::GEQP3, info)?;
        push_leading_upper(&qr, m, k, n, out.r);
        let start = out.q.len();
        out.q.extend_from_slice(&qr[..q_len]);
        // SAFETY: the `m x k` block holds `k` reflectors with `tau`; the query sized `work`.
        unsafe {
            T::orgqr(
                mi,
                ki,
                ki,
                &mut out.q[start..],
                mi,
                &tau,
                work,
                *orgqr_lwork,
                &mut info,
            );
        }
        check_info(op, T::ORGQR, info)?;
        normalize_into(op, &jpvt, &mut seen, out.permutation)?;
    }
    workspace.release(qr);
    if let Some(RrqrScratch { work, rwork, .. }) = scratch {
        workspace.release(work);
        if T::COMPLEX {
            workspace.release(rwork);
        }
    }
    Ok(())
}

/// Query both routines on the first factored matrix and acquire one work buffer for both.
#[allow(clippy::too_many_arguments)]
fn rrqr_scratch<T, W>(
    op: Op,
    mi: i32,
    ni: i32,
    ki: i32,
    qr: &mut [T],
    jpvt: &mut [i32],
    tau: &mut [T],
    workspace: &mut W,
) -> Result<RrqrScratch<T, T::Real>>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real>,
{
    let n = ni as usize;
    let rwork_len = if T::COMPLEX {
        checked_product(op, "GEQP3 real workspace", &[2, n])?
    } else {
        0
    };
    let mut rwork: Vec<T::Real> = if T::COMPLEX {
        workspace.acquire_zeroed(rwork_len)
    } else {
        Vec::new()
    };
    check_scratch(op, rwork.len(), rwork_len)?;
    let mut query = [T::default()];
    let mut info = 0;
    // SAFETY: the matrix, pivot, tau, real-work and query lengths follow the checked dimensions;
    // `lwork = -1` writes only the query slot.
    unsafe {
        T::geqp3(
            mi, ni, qr, mi, jpvt, tau, &mut query, -1, &mut rwork, &mut info,
        );
    }
    check_query_info(op, T::GEQP3, info)?;
    let geqp3_lwork = work_len(op, T::GEQP3, T::work_query_len(query[0]))?;
    // SAFETY: the first `m * k` entries of `qr` stand in for the `Q` block; `lwork = -1` writes
    // only the query slot and leaves `qr` untouched.
    unsafe {
        T::orgqr(mi, ki, ki, qr, mi, tau, &mut query, -1, &mut info);
    }
    check_query_info(op, T::ORGQR, info)?;
    let orgqr_lwork = work_len(op, T::ORGQR, T::work_query_len(query[0]))?;
    let len = geqp3_lwork.max(orgqr_lwork) as usize;
    let work: Vec<T> = workspace.acquire_zeroed(len);
    check_scratch(op, work.len(), len)?;
    Ok(RrqrScratch {
        geqp3_lwork,
        orgqr_lwork,
        work,
        rwork,
    })
}

/// Append a one-based `?geqp3` permutation as zero-based `i64` indices, rejecting anything that is
/// not a permutation.
fn normalize_into(
    op: Op,
    permutation: &[i32],
    seen: &mut [bool],
    out: &mut Vec<i64>,
) -> Result<()> {
    let n = permutation.len();
    seen.fill(false);
    for &column in permutation {
        let zero_based = column
            .checked_sub(1)
            .and_then(|value| usize::try_from(value).ok());
        let Some(zero_based) = zero_based.filter(|&value| value < n && !seen[value]) else {
            return Err(Error::InvalidArgument {
                op,
                role: "column_permutation",
                detail: "LAPACK GEQP3 returned an invalid pivot permutation".to_owned(),
            });
        };
        seen[zero_based] = true;
        out.push(zero_based as i64);
    }
    Ok(())
}
