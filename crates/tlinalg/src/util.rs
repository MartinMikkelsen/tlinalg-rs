//! Shared argument checks and descriptor-to-faer conversions for the per-matrix kernels.

use faer::{MatMut, MatRef};
use strided_view::{RawStridedMut, RawStridedRef};

use crate::scalar::ScalarEntity;
use crate::{Error, Op, Result};

/// A caller-supplied argument was invalid.
pub(crate) fn invalid(op: Op, role: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidArgument {
        op,
        role,
        detail: detail.into(),
    }
}

/// A shape product overflowed `usize`.
pub(crate) fn checked_product(op: Op, role: &'static str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            invalid(
                op,
                "configuration",
                format!("{role} element count overflows usize"),
            )
        })
}

/// Reject a descriptor that does not describe an `m x n` matrix.
fn check_dims(op: Op, role: &'static str, dims: &[usize], m: usize, n: usize) -> Result<()> {
    if dims != [m, n] {
        return Err(invalid(
            op,
            "configuration",
            format!("{role} describes {dims:?}, expected {m}x{n}"),
        ));
    }
    Ok(())
}

/// Borrow an `m x n` input descriptor as a faer matrix over the scalar's faer entity.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `m x n`.
pub(crate) fn mat_ref<'a, T: ScalarEntity>(
    op: Op,
    role: &'static str,
    input: &RawStridedRef<'a, T>,
    m: usize,
    n: usize,
) -> Result<MatRef<'a, T::Entity>> {
    check_dims(op, role, input.dims(), m, n)?;
    // SAFETY: `RawStridedRef::new` validated that every offset reachable from `ptr()` through
    // `dims`/`strides` lies inside the borrowed data, and `dims` was checked to be `[m, n]`, so the
    // descriptor describes exactly the `m x n` matrix faer is told to read, for the borrow `'a`. An
    // empty descriptor yields a dangling, aligned pointer that faer never dereferences. The pointer
    // cast is the layout-preserving one asserted in `crate::scalar`.
    Ok(unsafe {
        MatRef::from_raw_parts(
            input.ptr().cast::<T::Entity>(),
            m,
            n,
            input.strides()[0],
            input.strides()[1],
        )
    })
}

/// Borrow an `m x n` output descriptor as a mutable faer matrix over the scalar's faer entity.
///
/// # Errors
///
/// [`Error::InvalidArgument`] when the descriptor does not describe `m x n`.
pub(crate) fn mat_mut<'a, T: ScalarEntity>(
    op: Op,
    role: &'static str,
    output: &'a mut RawStridedMut<'_, T>,
    m: usize,
    n: usize,
) -> Result<MatMut<'a, T::Entity>> {
    check_dims(op, role, output.dims(), m, n)?;
    let (rs, cs) = (output.strides()[0], output.strides()[1]);
    // SAFETY: as in `mat_ref`, with exclusive access from the `&mut` borrow of the descriptor, whose
    // data is a `&mut [T]`. A descriptor whose strides alias two elements (a zero stride on a
    // non-trivial extent) would let faer write one location twice; the host never builds one, and
    // the caller contract says so.
    Ok(
        unsafe {
            MatMut::from_raw_parts_mut(output.as_mut_ptr().cast::<T::Entity>(), m, n, rs, cs)
        },
    )
}

/// Push the column-major `rows x cols` leading block of `mat`, keeping only the entries the
/// predicate selects and writing zero elsewhere.
///
/// One pass, by `push`: the pre-extraction code zero-initialized and then scattered, so this writes
/// each element once instead of twice and allocates the same single buffer.
pub(crate) fn push_masked<T: ScalarEntity>(
    out: &mut Vec<T>,
    mat: MatRef<'_, T::Entity>,
    rows: usize,
    cols: usize,
    keep: impl Fn(usize, usize) -> bool,
) {
    for col in 0..cols {
        for row in 0..rows {
            out.push(if keep(row, col) {
                T::from_entity(mat[(row, col)])
            } else {
                T::default()
            });
        }
    }
}

/// Push the whole column-major contents of `mat`.
pub(crate) fn push_mat<T: ScalarEntity>(out: &mut Vec<T>, mat: MatRef<'_, T::Entity>) {
    for col in 0..mat.ncols() {
        for row in 0..mat.nrows() {
            out.push(T::from_entity(mat[(row, col)]));
        }
    }
}

/// Push the `n x n` permutation matrix with a one at `(row, perm[row])`, column-major.
///
/// `perm_inv` is the inverse permutation, so column `col` has its one in row `perm_inv[col]`.
pub(crate) fn push_permutation<T: ScalarEntity>(out: &mut Vec<T>, perm_inv: &[usize]) {
    let n = perm_inv.len();
    let one = T::parity(false);
    for &one_row in perm_inv {
        for row in 0..n {
            out.push(if row == one_row { one } else { T::default() });
        }
    }
}
