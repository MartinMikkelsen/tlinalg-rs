//! Tensor-free numerical interface shared by the `tlinalg` implementations.
//!
//! This crate owns the vocabulary that crosses the boundary between a host (which owns tensors,
//! allocation, dtype dispatch, placement and execution context) and a numerical implementation
//! (which owns the kernels and their per-item scratch). It deliberately contains no kernels and no
//! tensor types.
//!
//! # Conventions
//!
//! Every implementation must reproduce these, because callers and their tests depend on them:
//!
//! | Operation | Convention |
//! |---|---|
//! | `svd` | returns `U`, non-increasing **real** `S`, and `Vᴴ` (not `V`) |
//! | `eigh` | reads the lower triangle, returns non-decreasing **real** values |
//! | `cholesky` | reads the lower triangle |
//! | factorizations | do **not** fail on exactly singular input unless the table below says so |
//! | solves | report [`Error::Singular`] for a singular factor |
//!
//! Failure behaviour is **per provider** and must not be normalized:
//!
//! | Route | Exactly singular input |
//! |---|---|
//! | LAPACK `getrf` (partial-pivot LU) | not an error |
//! | LAPACK `getc2` (full-pivot LU) | [`Error::Singular`] |
//! | faer partial-pivot LU in `solve` | [`Error::Singular`] |
//! | faer full-pivot LU | not an error |
//! | LAPACK `potrf` / non-convergent drivers | [`Error::NonConvergence`] |
//!
//! # What is not here
//!
//! Dtype dispatch and the deliberate "unsupported for this dtype" cases stay in the host: this
//! crate never names a host `DType`. Terminal storage—tensor construction, placement tagging, and
//! the error wrapper that carries an implementation error into the host's error type—also stays in
//! the host.

#![warn(missing_docs)]

pub mod error;
pub mod lane;
pub mod parallel;
pub mod scratch;

pub use error::{Error, NonFiniteRole, Op, Result};
pub use lane::LanePlan;
pub use parallel::Parallel;
pub use scratch::{IndexWorkspace, Scalar, Workspace};

mod sealed {
    /// Seals [`crate::Scalar`].
    pub trait Sealed {}
}
