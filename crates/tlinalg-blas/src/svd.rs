//! Batched singular value decomposition on LAPACK.
//!
//! Moved from `tenferro-linalg`'s LAPACK provider (same project, MIT OR Apache-2.0). Two policies
//! from there are preserved exactly, because they are what the mechanism counts measure:
//!
//! * the workspace is queried **once per batch** and acquired once, then reused across the **serial**
//!   item loop — LAPACK owns threading inside each call, and the extraction must not turn one query
//!   per batch into one per matrix;
//! * `iwork` (`?gesdd`) and the complex `rwork` have lengths the caller computes, not queries, so
//!   they are acquired once per batch too.
//!
//! The factors use the same conventions as the faer-backed implementation: `s` holds `min(m, n)`
//! real singular values, `u` is `m x u_cols` and `vt` is `vt_rows x n`, both column-major, with
//! `vt` holding `Vᴴ`.

use tlinalg_traits::{Error, IndexWorkspace, Op, Parallel, Result, Workspace};

use crate::LapackScalar;

/// Which singular factors to compute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SvdMode {
    /// Thin factors: `U` is `m x k` and `Vᴴ` is `k x n`.
    Thin,
    /// Full factors: `U` is `m x m` and `Vᴴ` is `n x n`.
    Full,
    /// Singular values only.
    Values,
}

impl SvdMode {
    /// The LAPACK `jobz` letter.
    fn job(self) -> u8 {
        match self {
            Self::Thin => b'S',
            Self::Full => b'A',
            Self::Values => b'N',
        }
    }

    /// `(U columns, Vᴴ rows)`; zero in values-only mode.
    fn factor_dims(self, m: usize, n: usize) -> (usize, usize) {
        let k = m.min(n);
        match self {
            Self::Thin => (k, k),
            Self::Full => (m, n),
            Self::Values => (0, 0),
        }
    }
}

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

fn dim_i32(op: Op, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument {
        op,
        role: "dimension",
        detail: format!("dimension {value} exceeds the LAPACK i32 range"),
    })
}

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

/// The `lwork` LAPACK reports, as an `i32`.
fn work_len(op: Op, routine: &'static str, query: f64) -> Result<i32> {
    if !(query.is_finite() && query >= 1.0) {
        return Err(Error::InvalidWorkspace {
            op,
            library: "LAPACK",
            routine,
            detail: format!("returned invalid workspace size {query}"),
        });
    }
    dim_i32(op, query.ceil() as usize)
}

/// The integer workspace length the selected driver needs.
///
/// `?gesdd` needs `8 * max(k, 1)`; `?gesvd` has no integer workspace at all.
#[cfg(not(feature = "provider-inject"))]
fn iwork_len(op: Op, k: usize) -> Result<usize> {
    checked_product(op, "integer workspace", &[8, k.max(1)])
}

#[cfg(feature = "provider-inject")]
fn iwork_len(_op: Op, _k: usize) -> Result<usize> {
    Ok(0)
}

/// The real workspace length the selected driver needs for complex input.
///
/// `?gesvd` takes a fixed `5 * max(k, 1)`; `?gesdd` needs the length LAPACK documents, which grows
/// with the shape unless the larger dimension exceeds a crossover.
#[cfg(feature = "provider-inject")]
fn complex_rwork_len(op: Op, _jobz: u8, m: usize, n: usize) -> Result<usize> {
    checked_product(op, "real workspace", &[5, m.min(n).max(1)])
}

#[cfg(not(feature = "provider-inject"))]
fn complex_rwork_len(op: Op, jobz: u8, m: usize, n: usize) -> Result<usize> {
    let mn = m.min(n);
    let mx = m.max(n);
    if jobz == b'N' {
        return checked_product(op, "real workspace", &[5, mn.max(1)]);
    }
    let threshold = checked_product(op, "workspace crossover", &[10, mn])?;
    let square_term = checked_product(op, "real workspace square term", &[5, mn, mn])?;
    let linear_term = checked_product(op, "real workspace linear term", &[5, mn])?;
    let small_shape_len =
        square_term
            .checked_add(linear_term)
            .ok_or_else(|| Error::InvalidArgument {
                op,
                role: "configuration",
                detail: "real workspace length overflows usize".to_owned(),
            })?;
    if mx > threshold {
        return Ok(small_shape_len);
    }
    let rectangular_term = checked_product(op, "real workspace rectangular term", &[2, mx, mn])?;
    let second_square_term =
        checked_product(op, "real workspace secondary square term", &[2, mn, mn])?;
    let large_shape_len = rectangular_term
        .checked_add(second_square_term)
        .and_then(|len| len.checked_add(mn))
        .ok_or_else(|| Error::InvalidArgument {
            op,
            role: "configuration",
            detail: "real workspace length overflows usize".to_owned(),
        })?;
    Ok(small_shape_len.max(large_shape_len))
}

/// Per-matrix buffer lengths of one batched call.
struct SvdLayout {
    batch: usize,
    k: usize,
    a_len: usize,
    u_len: usize,
    vt_len: usize,
    ldu: i32,
    ldvt: i32,
}

impl SvdLayout {
    fn new(op: Op, mode: SvdMode, m: usize, n: usize, lens: [usize; 4]) -> Result<Self> {
        let [a, s, u, vt] = lens;
        let k = m.min(n);
        let (u_cols, vt_rows) = mode.factor_dims(m, n);
        let a_len = checked_product(op, "matrix", &[m, n])?;
        let u_len = checked_product(op, "left singular vectors", &[m, u_cols])?;
        let vt_len = checked_product(op, "right singular vectors", &[vt_rows, n])?;
        let batch = a.checked_div(a_len).unwrap_or(0);
        let consistent = a_len != 0
            && a == checked_product(op, "matrix batch", &[a_len, batch])?
            && s == checked_product(op, "singular value batch", &[k, batch])?
            && u == checked_product(op, "left factor batch", &[u_len, batch])?
            && vt == checked_product(op, "right factor batch", &[vt_len, batch])?;
        if !consistent {
            return Err(Error::Inconsistent {
                op,
                detail: "SVD buffers describe different batches",
            });
        }
        let (ldu, ldvt) = if mode == SvdMode::Values {
            (1, 1)
        } else {
            (dim_i32(op, m)?, dim_i32(op, vt_rows)?)
        };
        Ok(Self {
            batch,
            k,
            a_len,
            u_len,
            vt_len,
            ldu,
            ldvt,
        })
    }

    fn chunk<'a, T, R>(
        &self,
        index: usize,
        a: &'a mut [T],
        s: &'a mut [R],
        u: &'a mut [T],
        vt: &'a mut [T],
    ) -> (&'a mut [T], &'a mut [R], &'a mut [T], &'a mut [T]) {
        (
            &mut a[index * self.a_len..(index + 1) * self.a_len],
            &mut s[index * self.k..(index + 1) * self.k],
            &mut u[index * self.u_len..(index + 1) * self.u_len],
            &mut vt[index * self.vt_len..(index + 1) * self.vt_len],
        )
    }
}

/// Compute the singular value decomposition of every `m x n` matrix of a batch.
///
/// `a` enters holding the compact column-major input batch, and holds the destroyed input on
/// return. `s`, `u` and `vt` receive the factors. Every buffer is the caller's, so the host owns
/// allocation, placement and tensor construction.
///
/// # Errors
///
/// Returns [`Error::Inconsistent`] when the buffers describe different batches,
/// [`Error::InvalidArgument`] for a dimension outside the LAPACK `i32` range or an illegal LAPACK
/// argument, [`Error::InvalidWorkspace`] for an unusable workspace size, and
/// [`Error::NonConvergence`] when LAPACK fails to converge.
// INVARIANT: the argument list mirrors the tenferro call it replaces; each value is a distinct
// operand of one decomposition, so grouping them would add a wrapper without removing an argument.
#[allow(clippy::too_many_arguments)]
pub fn svd_batch<T, W>(
    op: Op,
    mode: SvdMode,
    m: usize,
    n: usize,
    a: &mut [T],
    s: &mut [T::Real],
    u: &mut [T],
    vt: &mut [T],
    workspace: &mut W,
    _par: Parallel<'_>,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real> + IndexWorkspace,
{
    let layout = SvdLayout::new(op, mode, m, n, [a.len(), s.len(), u.len(), vt.len()])?;
    if layout.batch == 0 {
        return Ok(());
    }
    let job = mode.job();
    let m_i32 = dim_i32(op, m)?;
    let n_i32 = dim_i32(op, n)?;
    let (ldu, ldvt) = (layout.ldu, layout.ldvt);
    let iwork_len = iwork_len(op, layout.k)?;
    // `?gesvd` has no integer workspace at all, so it must not touch the pool for one: taking an
    // integer buffer and returning it would churn the retained capacity of a pool it never used.
    let mut iwork = if iwork_len > 0 {
        workspace.acquire_zeroed_index(iwork_len)
    } else {
        Vec::new()
    };
    // The real routines have no real workspace at all; the complex ones need one of a computed
    // length, held for the whole batch alongside `work` and `iwork`.
    let complex = std::mem::size_of::<T>() != std::mem::size_of::<T::Real>();
    let rwork_len = if complex {
        complex_rwork_len(op, job, m, n)?
    } else {
        0
    };
    let mut rwork = if rwork_len > 0 {
        workspace.acquire_zeroed(rwork_len)
    } else {
        Vec::new()
    };
    // A stack slot, not a heap buffer: the workspace query writes one value and the caller-visible
    // allocation count must not grow because of it.
    let mut query = [T::default(); 1];
    let mut info = 0;
    {
        let (a0, s0, u0, vt0) = layout.chunk(0, a, s, u, vt);
        // SAFETY: every buffer is sized for this item as documented above.
        unsafe {
            T::svd_driver(
                job, job, m_i32, n_i32, a0, m_i32, s0, u0, ldu, vt0, ldvt, &mut query, -1,
                &mut rwork, &mut iwork, &mut info,
            );
        }
    }
    check_info(op, "svd(work query)", info)?;
    let lwork = work_len(op, T::routine_name(), T::work_query_len(query[0]))?;
    let mut work = workspace.acquire_zeroed(lwork as usize);
    // A `Workspace` implementation is safe code and may return anything, but LAPACK writes into
    // these buffers up to the lengths it was told, so the host's promise is checked here rather
    // than trusted at a raw FFI boundary.
    if work.len() < lwork as usize || iwork.len() < iwork_len || rwork.len() < rwork_len {
        return Err(Error::Inconsistent {
            op,
            detail: "the workspace returned fewer elements than the routine requires",
        });
    }
    // INVARIANT: every buffer was checked to hold `layout.batch` blocks and the workspace depends
    // only on `(mode, m, n)`, so it is reused across the batch. The serial loop is intentional:
    // LAPACK owns threading inside each call.
    for index in 0..layout.batch {
        let (a_i, s_i, u_i, vt_i) = layout.chunk(index, a, s, u, vt);
        // SAFETY: every buffer is sized for this item as documented above.
        unsafe {
            T::svd_driver(
                job, job, m_i32, n_i32, a_i, m_i32, s_i, u_i, ldu, vt_i, ldvt, &mut work, lwork,
                &mut rwork, &mut iwork, &mut info,
            );
        }
        check_info(op, T::routine_name(), info)?;
    }
    workspace.release(work);
    if iwork_len > 0 {
        workspace.release_index(iwork);
    }
    if rwork_len > 0 {
        workspace.release(rwork);
    }
    Ok(())
}
