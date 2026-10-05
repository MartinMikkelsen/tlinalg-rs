//! Argument checks and `info` translation shared by the kernels moved from tenferro's LAPACK backend.
//!
//! The texts reproduce the ones the host produced before the move, because the host forwards
//! `role` and `detail` into its own payloads verbatim.

use crate::{Error, Op, Result};

/// A shape product, with the host's overflow text (`role "shape"`).
pub(crate) fn checked_product(op: Op, role: &'static str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| Error::InvalidArgument {
            op,
            role: "shape",
            detail: format!("{role} element count overflow"),
        })
}

/// A dimension LAPACK cannot express in its `i32` interface.
pub(crate) fn dim_i32(op: Op, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::InvalidArgument {
        op,
        role: "dimension",
        detail: format!("dimension {value} exceeds LAPACK i32 range"),
    })
}

/// Translate a LAPACK `info`: negative is an illegal argument, positive a numerical failure.
pub(crate) fn check_info(op: Op, routine: &'static str, info: i32) -> Result<()> {
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

/// [`check_info`] for a workspace query: the routine is reported as `"<routine>(work query)"`.
pub(crate) fn check_query_info(op: Op, routine: &'static str, info: i32) -> Result<()> {
    if info < 0 {
        return Err(Error::InvalidArgument {
            op,
            role: "lapack_argument",
            detail: format!(
                "LAPACK {routine}(work query) argument {} had an illegal value",
                -info
            ),
        });
    }
    if info > 0 {
        return Err(Error::NonConvergence { op });
    }
    Ok(())
}

/// The `lwork` a workspace query reported, as an `i32`.
pub(crate) fn work_len(op: Op, routine: &'static str, query: f64) -> Result<i32> {
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

/// Reject a buffer whose length is not the one its dimensions describe.
pub(crate) fn check_len(op: Op, role: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::InvalidArgument {
            op,
            role,
            detail: format!("expected {expected} elements, got {actual}"),
        });
    }
    Ok(())
}

/// The host's workspace length check before a raw FFI call.
///
/// A `Workspace` implementation is safe code and may return anything, but LAPACK writes into these
/// buffers up to the lengths it was told, so the promise is checked rather than trusted.
pub(crate) fn check_scratch(op: Op, actual: usize, required: usize) -> Result<()> {
    if actual < required {
        return Err(Error::Inconsistent {
            op,
            detail: "the workspace returned fewer elements than the routine requires",
        });
    }
    Ok(())
}
