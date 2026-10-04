//! faer-backed singular value decomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry point is
//! **per matrix**, matching what the host did before: the host keeps its own batch iteration, so
//! extracting this family does not quietly become a batching rewrite.
//!
//! # Boundary
//!
//! The input is a borrowed [`RawStridedRef`] and the outputs are caller-provided vectors, because
//! the host owns tensors, placement and allocation. The vectors are **cleared and then filled by
//! `push`**, which is what the pre-extraction code did: taking a `&mut [T]` instead would force the
//! host to initialize the buffer before the kernel overwrote it, adding a write pass the previous
//! implementation did not have.
//!
//! Internally faer still works in its own `Mat`/`Diag`/`MemBuffer` storage: that scratch is
//! operation-local and stays native, exactly as before. This family therefore needs no
//! [`tlinalg_traits::Workspace`].
//!
//! # Conventions
//!
//! `u` is `m x u_cols` column-major, `vt` is `v_cols x n` column-major and holds `Vᴴ` (not `V`),
//! and `s` holds `min(m, n)` singular values in non-increasing order **in the scalar type itself**.
//! For the complex scalars that means complex values with a zero imaginary part, which is what the
//! pre-extraction code produced and what its callers' plumbing expects; [`svd_values`] returns the
//! real values instead, matching the values-only path. `full` selects the square unitary factors
//! `u_cols = m`, `v_cols = n`; otherwise `u_cols = v_cols = min(m, n)`.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::ComputeSvdVectors;
use faer::{Mat, MatRef};

use strided_view::RawStridedRef;
use tlinalg_traits::{Error, Op, Parallel, Result};

use crate::scalar::ScalarEntity;
use crate::{faer_par, with_parallel, FaerScalar};

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

/// Singular values of one `m x n` matrix, without the vectors.
///
/// # Errors
///
/// As [`svd`]; `u` and `vt` are not needed, so only the descriptor, the value buffer and the
/// parallelism are validated.
pub fn svd_values<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    input: RawStridedRef<'_, T>,
    s: &mut Vec<<T as ScalarEntity>::Real>,
    par: Parallel<'_>,
) -> Result<()> {
    let k = m.min(n);
    if input.dims() != [m, n] {
        return Err(Error::InvalidArgument {
            op,
            role: "configuration",
            detail: format!("input describes {:?}, expected {m}x{n}", input.dims()),
        });
    }
    s.clear();
    if m == 0 || n == 0 {
        return Ok(());
    }
    let faer_par = faer_par(par);
    let mut s_diag = Diag::<T::Entity>::zeros(k);
    let mut mem = MemBuffer::new(faer::linalg::svd::svd_scratch::<T::Entity>(
        m,
        n,
        ComputeSvdVectors::No,
        ComputeSvdVectors::No,
        faer_par,
        Default::default(),
    ));
    // SAFETY: as in `svd`: the descriptor was validated by construction and its dims were checked.
    let mat: MatRef<'_, T::Entity> = unsafe {
        MatRef::from_raw_parts(
            input.ptr().cast::<T::Entity>(),
            m,
            n,
            input.strides()[0],
            input.strides()[1],
        )
    };
    with_parallel(par, |par| {
        let stack = MemStack::new(&mut mem);
        faer::linalg::svd::svd(
            mat,
            s_diag.as_mut(),
            None,
            None,
            par,
            stack,
            Default::default(),
        )
        .map_err(|_| Error::NonConvergence { op })
    })?;
    for index in 0..k {
        s.push(<T as ScalarEntity>::real_from_entity(s_diag[index]));
    }
    Ok(())
}

/// Singular value decomposition of one `m x n` matrix.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when the descriptor does not describe `m x n`,
/// [`Error::Inconsistent`] when an output buffer is the wrong size, and [`Error::NonConvergence`]
/// when faer fails to converge.
// INVARIANT: the argument list mirrors the tenferro call it replaces — descriptor, three output
// buffers, mode and token are distinct operands of one decomposition, so grouping them would add a
// wrapper without removing an argument.
#[allow(clippy::too_many_arguments)]
pub fn svd<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    full: bool,
    input: RawStridedRef<'_, T>,
    u: &mut Vec<T>,
    s: &mut Vec<T>,
    vt: &mut Vec<T>,
    par: Parallel<'_>,
) -> Result<()> {
    let k = m.min(n);
    // Full mode returns the square unitary factors `U (m x m)` and `V (n x n)`; thin mode keeps the
    // leading `k` vectors. The singular-value count is `k` in both modes.
    let (u_cols, v_cols, vectors) = if full {
        (m, n, ComputeSvdVectors::Full)
    } else {
        (k, k, ComputeSvdVectors::Thin)
    };
    if input.dims() != [m, n] {
        return Err(Error::InvalidArgument {
            op,
            role: "configuration",
            detail: format!("input describes {:?}, expected {m}x{n}", input.dims()),
        });
    }
    // The sizes are computed here only so an overflow is reported as a configuration error rather
    // than as a failed reservation; the vectors themselves are filled below.
    let _ = checked_product(op, "U", &[m, u_cols])?;
    let _ = checked_product(op, "VH", &[v_cols, n])?;
    u.clear();
    s.clear();
    vt.clear();
    if m == 0 || n == 0 {
        return Ok(());
    }

    let faer_par = faer_par(par);
    let mut u_mat = Mat::<T::Entity>::zeros(m, u_cols);
    let mut v_mat = Mat::<T::Entity>::zeros(n, v_cols);
    let mut s_diag = Diag::<T::Entity>::zeros(k);
    let mut mem = MemBuffer::new(faer::linalg::svd::svd_scratch::<T::Entity>(
        m,
        n,
        vectors,
        vectors,
        faer_par,
        Default::default(),
    ));

    // SAFETY: `RawStridedRef::new` validated that every reachable offset lies inside the borrowed
    // data, and `dims` was checked against `m`/`n` above, so the descriptor describes exactly the
    // `m x n` matrix faer is told to read. The pointer cast is the layout-preserving one asserted in
    // `crate::scalar`.
    let mat: MatRef<'_, T::Entity> = unsafe {
        MatRef::from_raw_parts(
            input.ptr().cast::<T::Entity>(),
            m,
            n,
            input.strides()[0],
            input.strides()[1],
        )
    };

    with_parallel(par, |par| {
        let stack = MemStack::new(&mut mem);
        faer::linalg::svd::svd(
            mat,
            s_diag.as_mut(),
            Some(u_mat.as_mut()),
            Some(v_mat.as_mut()),
            par,
            stack,
            Default::default(),
        )
        .map_err(|_| Error::NonConvergence { op })
    })?;

    // Column-major `U`, `min(m, n)` real singular values, then column-major `Vᴴ`, pushed in the
    // order the previous implementation produced them.
    for col in 0..u_cols {
        for row in 0..m {
            u.push(T::from_entity(u_mat[(row, col)]));
        }
    }
    for index in 0..k {
        // A real singular value carried in the scalar type: zero imaginary part for the complex
        // scalars, which is the shape the pre-extraction callers consumed.
        s.push(T::from_entity(s_diag[index]));
    }
    for col in 0..n {
        for row in 0..v_cols {
            // `V` is transposed into `Vᴴ`, so the complex scalars conjugate here. The real ones are
            // their own conjugate, which is why the two implementations share this call.
            vt.push(T::from_entity_conj(v_mat[(col, row)]));
        }
    }
    Ok(())
}
