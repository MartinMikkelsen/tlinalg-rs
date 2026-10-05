//! Batched Hermitian eigendecomposition on LAPACK: `?syevd` for real scalars, `?heev` for complex.
//!
//! Moved from tenferro-linalg's LAPACK backend (same project, MIT OR Apache-2.0). The workspace is
//! queried once per call and reused across the serial loop, with the acquisition and release order
//! the host used.

use strided_view::RawStridedRef;

use crate::batch::{batch_len, Input};
use crate::common::{
    check_info, check_query_info, check_scratch, checked_product, dim_i32, work_len,
};
use crate::{Error, IndexWorkspace, LapackScalar, Op, Result, Workspace};

fn invalid_iwork(op: Op, routine: &'static str, detail: String) -> Error {
    Error::InvalidWorkspace {
        op,
        library: "LAPACK",
        routine,
        detail,
    }
}

/// Eigenvalues, and with `vectors` the eigenvectors, of every Hermitian matrix of a batch.
///
/// `a` has dims `[n, n, batch...]`, is read through its lower triangle and is not modified.
/// `values` is cleared and receives `n` non-decreasing real eigenvalues per matrix. When `vectors`
/// is `Some`, it is cleared and receives the `n x n` eigenvector matrices, compact column-major in
/// batch order (LAPACK runs in place on them); otherwise one `n x n` copy is acquired from
/// `workspace` and destroyed per matrix. The LAPACK workspace is queried once: for real scalars
/// `work` and the queried integer `iwork` (released in that order), for complex scalars the
/// `max(3n - 2, 1)` real `rwork` and `work` (released in that order).
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a malformed operand, a dimension outside the LAPACK `i32` range or
/// an illegal LAPACK argument, [`Error::InvalidWorkspace`] for an unusable (integer) workspace size,
/// and [`Error::NonConvergence`] for a positive `info`; always for the first failing matrix in
/// batch order. On error `values` and `vectors` are left empty.
pub fn eigh<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    values: &mut Vec<T::Real>,
    mut vectors: Option<&mut Vec<T>>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real> + IndexWorkspace,
{
    values.clear();
    if let Some(vectors) = vectors.as_deref_mut() {
        vectors.clear();
    }
    let result = run(op, a, values, vectors.as_deref_mut(), workspace);
    if result.is_err() {
        values.clear();
        if let Some(vectors) = vectors {
            vectors.clear();
        }
    }
    result
}

fn run<T, W>(
    op: Op,
    a: RawStridedRef<'_, T>,
    values: &mut Vec<T::Real>,
    mut vectors: Option<&mut Vec<T>>,
    workspace: &mut W,
) -> Result<()>
where
    T: LapackScalar,
    W: Workspace<T> + Workspace<T::Real> + IndexWorkspace,
{
    let input = Input::new(op, "input", a)?;
    let n = input.square(op)?;
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let count = input.layout.count;
    if matrix_len == 0 || count == 0 {
        return Ok(());
    }
    let n_i32 = dim_i32(op, n)?;
    let jobz = if vectors.is_some() { b'V' } else { b'N' };
    let routine = T::EIGH;
    // The values are written by LAPACK through a slice, so they are initialised first.
    values.resize(batch_len(op, n, count)?, <T::Real>::default());
    if let Some(vectors) = vectors.as_deref_mut() {
        vectors.reserve(batch_len(op, matrix_len, count)?);
    }
    let mut scratch: Option<Vec<T>> = match vectors {
        Some(_) => None,
        None => Some(workspace.acquire_capacity(matrix_len)),
    };

    // The complex driver needs a real workspace of a computed length, acquired before the query.
    let rwork_len = if T::COMPLEX {
        checked_product(op, "real workspace", &[3, n])?
            .checked_sub(2)
            .unwrap_or(1)
            .max(1)
    } else {
        0
    };
    let mut rwork: Vec<T::Real> = if T::COMPLEX {
        workspace.acquire_zeroed(rwork_len)
    } else {
        Vec::new()
    };
    check_scratch(op, rwork.len(), rwork_len)?;

    let mut state: Option<(i32, i32, Vec<T>, Vec<i32>)> = None;
    for (item, offset) in input.layout.offsets().enumerate() {
        let matrix: &mut [T] = match (vectors.as_deref_mut(), scratch.as_mut()) {
            (Some(out), _) => {
                let start = out.len();
                input.gather(offset, out);
                &mut out[start..]
            }
            (None, Some(compact)) => {
                compact.clear();
                input.gather(offset, compact);
                compact.as_mut_slice()
            }
            (None, None) => unreachable!("values-only calls hold a scratch matrix"),
        };
        let vals = &mut values[item * n..(item + 1) * n];
        if state.is_none() {
            let mut query = [T::default(); 1];
            let mut iquery = [0_i32; 1];
            let mut info = 0;
            // SAFETY: `matrix` is a mutable column-major `n x n` matrix and `vals` holds `n`
            // values; `lwork = liwork = -1` writes only the query slots.
            unsafe {
                T::eigh_driver(
                    jobz,
                    n_i32,
                    matrix,
                    n_i32,
                    vals,
                    &mut query,
                    -1,
                    &mut rwork,
                    &mut iquery,
                    -1,
                    &mut info,
                );
            }
            check_query_info(op, routine, info)?;
            let lwork = work_len(op, routine, T::work_query_len(query[0]))?;
            let (liwork, liwork_capacity) = if T::COMPLEX {
                (0, 0)
            } else {
                let liwork = iquery[0];
                if liwork < 1 {
                    return Err(invalid_iwork(
                        op,
                        routine,
                        format!("invalid integer workspace size {liwork}"),
                    ));
                }
                let capacity = usize::try_from(liwork).map_err(|_| {
                    invalid_iwork(
                        op,
                        routine,
                        format!("integer workspace size {liwork} does not fit usize"),
                    )
                })?;
                (liwork, capacity)
            };
            let work: Vec<T> = workspace.acquire_zeroed(lwork as usize);
            let iwork = if T::COMPLEX {
                Vec::new()
            } else {
                workspace.acquire_zeroed_index(liwork_capacity)
            };
            check_scratch(op, work.len(), lwork as usize)?;
            check_scratch(op, iwork.len(), liwork_capacity)?;
            state = Some((lwork, liwork, work, iwork));
        }
        let Some((lwork, liwork, work, iwork)) = state.as_mut() else {
            unreachable!("the workspace was just queried");
        };
        let mut info = 0;
        // SAFETY: the dimensions and workspace lengths come from the validated shape and the
        // query on this shape.
        unsafe {
            T::eigh_driver(
                jobz, n_i32, matrix, n_i32, vals, work, *lwork, &mut rwork, iwork, *liwork,
                &mut info,
            );
        }
        check_info(op, routine, info)?;
    }
    if let Some((_, _, work, iwork)) = state {
        if T::COMPLEX {
            workspace.release(rwork);
            workspace.release(work);
        } else {
            workspace.release(work);
            workspace.release_index(iwork);
        }
    }
    if let Some(compact) = scratch {
        workspace.release(compact);
    }
    Ok(())
}
