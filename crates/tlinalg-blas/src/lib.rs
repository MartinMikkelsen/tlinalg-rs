//! LAPACK/BLAS-backed tensor-free linear algebra.
//!
//! This crate owns the vendor calls and their argument marshalling. It does not own tensors,
//! allocation, dtype dispatch, placement, or the execution context, and it never touches vendor
//! threading: LAPACK and BLAS own their own parallelism, so no entry point takes a parallelism
//! token.
//!
//! # Vocabulary
//!
//! The crate is self-contained: [`Error`], [`Op`], [`Workspace`], [`IndexWorkspace`] and [`Scalar`]
//! are its own. The interface a host requires of a linear-algebra provider lives in that host
//! (tenferro's CPU linalg kernels), which adapts these functions and maps [`Error`] onto its own
//! errors. Nothing here is shared with the faer-backed `tlinalg` crate.
//!
//! # Conventions
//!
//! | Operation | Convention |
//! |---|---|
//! | `svd` | returns `U`, non-increasing **real** `S`, and `Vᴴ` (not `V`) |
//! | `eigh` | reads the lower triangle, returns non-decreasing **real** values |
//! | `cholesky` | reads the lower triangle |
//! | `getrf` (partial-pivot LU, packed or explicit) | exactly singular input is **not** an error |
//! | `getrf` inside `solve` | exactly singular input is [`Error::Singular`] |
//! | `getc2` (full-pivot LU and its solve) | exactly singular input is [`Error::Singular`] |
//! | `potrf`, `trtrs` and non-convergent drivers | a positive `info` is [`Error::NonConvergence`] |
//!
//! # Threading
//!
//! Every batch loop in this crate is serial, and the vendor call inside it does the threading. That
//! is deliberate: a Rayon fan-out around a vendor call would fight the vendor's own pool, so the
//! host is expected to place vendor work rather than parallelise inside it.

#![warn(missing_docs)]

pub mod cholesky;
pub mod eig;
pub mod eigh;
pub mod error;
pub mod full_piv_lu;
pub mod householder;
pub mod lu;
pub mod qr;
pub mod scratch;
pub mod solve;
pub mod svd;
pub mod triangular_solve;

mod batch;
mod common;
#[doc(hidden)]
pub mod symbols;

pub use error::{Error, NonFiniteRole, Op, Result};
pub use scratch::{IndexWorkspace, Scalar, Workspace};

mod sealed {
    /// Seals [`crate::Scalar`].
    pub trait Sealed {}
}

/// Link the vendor libraries for this crate's own tests.
///
/// Tenferro supplies the symbols through its own provider features; this is only so that
/// `cargo test --features link-openblas` can run the checks in this repository.
#[cfg(feature = "link-openblas")]
extern crate blas_src as _;
#[cfg(feature = "link-openblas")]
extern crate lapack_src as _;

/// A scalar this crate has LAPACK bindings for.
///
/// Sealed through `symbols::Symbols`: the four scalars are the only implementors.
pub trait LapackScalar: Scalar + Default + PartialEq + symbols::Symbols {}

/// `|value|` widened to `f64`, as the rank-revealing QR rank rule measures the `R` diagonal.
pub fn magnitude<T: LapackScalar>(value: T) -> f64 {
    value.magnitude()
}

impl LapackScalar for f32 {}
impl LapackScalar for f64 {}
impl LapackScalar for num_complex::Complex32 {}
impl LapackScalar for num_complex::Complex64 {}
