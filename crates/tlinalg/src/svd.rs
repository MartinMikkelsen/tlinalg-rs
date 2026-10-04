//! faer-backed singular value decomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The entry point is
//! **per matrix**, matching what the host did before: the host keeps its own batch iteration, so
//! extracting this family does not quietly become a batching rewrite.
//!
//! # Boundary
//!
//! The input is a borrowed [`RawStridedRef`] and the outputs are caller-provided slices, because the
//! host owns tensors, placement and allocation. Internally faer still works in its own
//! `Mat`/`Diag`/`MemBuffer` storage: that scratch is operation-local and stays native, exactly as
//! before. This family therefore needs no [`tlinalg_traits::Workspace`].
//!
//! # Conventions
//!
//! `u` is `m x u_cols` column-major, `s` holds `min(m, n)` **real** singular values in
//! non-increasing order, and `vt` is `v_cols x n` column-major and holds `Vᴴ` (not `V`). `full`
//! selects the square unitary factors `u_cols = m`, `v_cols = n`; otherwise `u_cols = v_cols =
//! min(m, n)`.

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

/// Check that one output buffer holds exactly the described matrix or vector.
fn check_len(op: Op, role: &'static str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        return Ok(());
    }
    Err(Error::Inconsistent {
        op,
        detail: match role {
            "left singular vectors" => "U does not hold m x u_cols",
            "singular values" => "S does not hold min(m, n)",
            _ => "VH does not hold v_cols x n",
        },
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
    s: &mut [<T as ScalarEntity>::Real],
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
    check_len(op, "singular values", k, s.len())?;
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
    for (index, slot) in s.iter_mut().enumerate() {
        *slot = <T as ScalarEntity>::real_from_entity(s_diag[index]);
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
    u: &mut [T],
    s: &mut [<T as ScalarEntity>::Real],
    vt: &mut [T],
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
    check_len(
        op,
        "left singular vectors",
        checked_product(op, "U", &[m, u_cols])?,
        u.len(),
    )?;
    check_len(op, "singular values", k, s.len())?;
    check_len(
        op,
        "right singular vectors",
        checked_product(op, "VH", &[v_cols, n])?,
        vt.len(),
    )?;
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

    // The outputs are `T` and faer's matrices are `T::Entity`, which share a layout, so write
    // through the layout-preserving view rather than converting element by element.
    let u_out = T::entity_slice_mut(u);
    for col in 0..u_cols {
        for row in 0..m {
            u_out[row + col * m] = u_mat[(row, col)];
        }
    }
    for (index, slot) in s.iter_mut().enumerate() {
        // faer returns a real singular value through the entity type, so take its real part.
        *slot = <T as ScalarEntity>::real_from_entity(s_diag[index]);
    }
    let vt_out = T::entity_slice_mut(vt);
    for col in 0..n {
        for row in 0..v_cols {
            vt_out[row + col * v_cols] = v_mat[(col, row)];
        }
    }
    Ok(())
}
