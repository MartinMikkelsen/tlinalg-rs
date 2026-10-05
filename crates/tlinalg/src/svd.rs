//! faer-backed batched singular value decomposition.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0).
//!
//! # Boundary
//!
//! The input is a borrowed rank-`2 + B` [`RawStridedRef`] `[m, n, b_1, ..., b_B]`; the outputs are
//! caller-provided vectors, **cleared and then filled** with compact column-major items in batch
//! order (see `docs/design/batched-api.md`). Faer's `Mat`/`Diag`/`MemBuffer` storage is lane
//! scratch, sized once per lane and reused for every item of the lane.
//!
//! # Conventions
//!
//! Per item, `u` is `m x u_cols` column-major, `vt` is `v_cols x n` column-major and holds `Vᴴ`
//! (not `V`), and `s` holds `min(m, n)` singular values in non-increasing order **in the scalar type
//! itself**. For the complex scalars that means complex values with a zero imaginary part, which is
//! what the pre-extraction code produced and what its callers' plumbing expects; [`svd_values`]
//! returns the real values instead, matching the values-only path. `full` selects the square
//! unitary factors `u_cols = m`, `v_cols = n`; otherwise `u_cols = v_cols = min(m, n)`. A full SVD
//! of an empty matrix returns identity `U` and `Vᴴ`.

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::svd::ComputeSvdVectors;
use faer::{Mat, MatRef};

use strided_view::RawStridedRef;

use crate::batch::{self, out, BatchedRef, Push};
use crate::scalar::ScalarEntity;
use crate::util::{checked_product, push_identity};
use crate::{Error, FaerScalar, LanePlan, Op, Parallel, Result};

/// One lane's faer storage for `m x n` decompositions.
struct SvdScratch<E: faer::traits::ComplexField> {
    u: Mat<E>,
    v: Mat<E>,
    s: Diag<E>,
    mem: MemBuffer,
}

impl<E: faer::traits::ComplexField> SvdScratch<E> {
    fn new(m: usize, n: usize, vectors: ComputeSvdVectors, par: faer::Par) -> Self {
        let k = m.min(n);
        let (u_cols, v_cols) = match vectors {
            ComputeSvdVectors::Full => (m, n),
            ComputeSvdVectors::Thin => (k, k),
            ComputeSvdVectors::No => (0, 0),
        };
        // An empty matrix is never decomposed, so it needs no faer scratch.
        let req = if k == 0 {
            faer::dyn_stack::StackReq::EMPTY
        } else {
            faer::linalg::svd::svd_scratch::<E>(m, n, vectors, vectors, par, Default::default())
        };
        Self {
            u: Mat::zeros(m, u_cols),
            v: Mat::zeros(n, v_cols),
            s: Diag::zeros(k),
            mem: MemBuffer::new(req),
        }
    }
}

/// Singular values of one matrix, pushed in the real type.
fn svd_values_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    s: &mut impl Push<<T as ScalarEntity>::Real>,
    scratch: &mut SvdScratch<T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let k = mat.nrows().min(mat.ncols());
    if k == 0 {
        return Ok(());
    }
    scratch
        .s
        .as_mut()
        .fill(<T::Entity as faer::traits::ComplexField>::zero_impl());
    faer::linalg::svd::svd(
        mat,
        scratch.s.as_mut(),
        None,
        None,
        par,
        MemStack::new(&mut scratch.mem),
        Default::default(),
    )
    .map_err(|_| Error::NonConvergence { op })?;
    for index in 0..k {
        s.push(<T as ScalarEntity>::real_from_entity(scratch.s[index]));
    }
    Ok(())
}

/// SVD of one matrix, pushed as `U`, `S` and `Vᴴ`.
fn svd_item<T: FaerScalar>(
    op: Op,
    mat: MatRef<'_, T::Entity>,
    full: bool,
    (u, s, vt): (&mut impl Push<T>, &mut impl Push<T>, &mut impl Push<T>),
    scratch: &mut SvdScratch<T::Entity>,
    par: faer::Par,
) -> Result<()> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let k = m.min(n);
    if k == 0 {
        if full {
            push_identity(u, m);
            push_identity(vt, n);
        }
        return Ok(());
    }
    let (u_cols, v_cols) = if full { (m, n) } else { (k, k) };
    // The pre-extraction code handed faer freshly zeroed outputs; the lane buffers are reset to the
    // same state so reuse cannot leak a previous item into this one.
    let zero = <T::Entity as faer::traits::ComplexField>::zero_impl();
    scratch.u.as_mut().fill(zero);
    scratch.v.as_mut().fill(zero);
    scratch.s.as_mut().fill(zero);
    faer::linalg::svd::svd(
        mat,
        scratch.s.as_mut(),
        Some(scratch.u.as_mut()),
        Some(scratch.v.as_mut()),
        par,
        MemStack::new(&mut scratch.mem),
        Default::default(),
    )
    .map_err(|_| Error::NonConvergence { op })?;

    // Column-major `U`, `min(m, n)` real singular values, then column-major `Vᴴ`, pushed in the
    // order the previous implementation produced them.
    for col in 0..u_cols {
        for row in 0..m {
            u.push(T::from_entity(scratch.u[(row, col)]));
        }
    }
    for index in 0..k {
        // A real singular value carried in the scalar type: zero imaginary part for the complex
        // scalars, which is the shape the pre-extraction callers consumed.
        s.push(T::from_entity(scratch.s[index]));
    }
    for col in 0..n {
        for row in 0..v_cols {
            // `V` is transposed into `Vᴴ`, so the complex scalars conjugate here. The real ones are
            // their own conjugate, which is why the two implementations share this call.
            vt.push(T::from_entity_conj(scratch.v[(col, row)]));
        }
    }
    Ok(())
}

/// Singular values of every matrix of a batch, without the vectors.
///
/// `input` is `[m, n, b_1, ..., b_B]`; `s` receives `min(m, n)` real values per item.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2, and [`Error::NonConvergence`] for the
/// lowest-indexed item faer fails to converge on. `s` is empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{svd::svd_values, LanePlan, Op, Parallel};
///
/// // Two 2x2 diagonal matrices, batch-contiguous.
/// let a = [3.0_f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 5.0];
/// let mut s = Vec::new();
/// svd_values(
///     Op::SvdValues, RawStridedRef::new(&a, &[2, 2, 2], &[1, 2, 4], 0).unwrap(), &mut s,
///     Parallel::Sequential, LanePlan::sequential(),
/// ).unwrap();
/// assert_eq!(s, [3.0, 1.0, 5.0, 2.0]);
/// ```
pub fn svd_values<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    s: &mut Vec<<T as ScalarEntity>::Real>,
    par: Parallel<'_>,
    plan: LanePlan<'_>,
) -> Result<()> {
    s.clear();
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    batch::run(
        op,
        input.batch(),
        par,
        plan,
        &mut (out(s, m.min(n)),),
        |par| SvdScratch::<T::Entity>::new(m, n, ComputeSvdVectors::No, par),
        |index, (s,), scratch, par| svd_values_item::<T>(op, input.item(index), s, scratch, par),
    )
}

/// Singular value decomposition of every matrix of a batch.
///
/// `input` is `[m, n, b_1, ..., b_B]`. Per item, `u` receives `m x u_cols`, `s` `min(m, n)` and
/// `vt` `v_cols x n` elements (see the module conventions).
///
/// # Errors
///
/// [`Error::InvalidArgument`] when `input` has rank below 2 or an output size overflows, and
/// [`Error::NonConvergence`] for the lowest-indexed item faer fails to converge on. The outputs are
/// empty on error.
///
/// # Examples
///
/// ```
/// use strided_view::RawStridedRef;
/// use tlinalg::{svd::svd, LanePlan, Op, Parallel};
///
/// let a = [3.0_f64, 0.0, 0.0, 1.0];
/// let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
/// svd(
///     Op::Svd, RawStridedRef::new(&a, &[2, 2], &[1, 2], 0).unwrap(), false,
///     &mut u, &mut s, &mut vt, Parallel::Sequential, LanePlan::sequential(),
/// ).unwrap();
/// assert_eq!(s, [3.0, 1.0]);
/// ```
// INVARIANT: descriptor, mode, three output buffers, token and plan are distinct operands of one
// batched decomposition; grouping them would add a wrapper without removing an argument.
#[allow(clippy::too_many_arguments)]
pub fn svd<T: FaerScalar>(
    op: Op,
    input: RawStridedRef<'_, T>,
    full: bool,
    u: &mut Vec<T>,
    s: &mut Vec<T>,
    vt: &mut Vec<T>,
    par: Parallel<'_>,
    plan: LanePlan<'_>,
) -> Result<()> {
    u.clear();
    s.clear();
    vt.clear();
    let input = BatchedRef::new(op, "input", input)?;
    let (m, n) = (input.rows(), input.cols());
    let k = m.min(n);
    let (u_cols, v_cols, vectors) = if full {
        (m, n, ComputeSvdVectors::Full)
    } else {
        (k, k, ComputeSvdVectors::Thin)
    };
    let u_len = checked_product(op, "U", &[m, u_cols])?;
    let vt_len = checked_product(op, "VH", &[v_cols, n])?;
    batch::run(
        op,
        input.batch(),
        par,
        plan,
        &mut (out(u, u_len), out(s, k), out(vt, vt_len)),
        |par| SvdScratch::<T::Entity>::new(m, n, vectors, par),
        |index, (u, s, vt), scratch, par| {
            svd_item::<T>(op, input.item(index), full, (u, s, vt), scratch, par)
        },
    )
}
