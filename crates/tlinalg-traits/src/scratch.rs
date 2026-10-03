//! Scratch acquisition.
//!
//! Implementations obtain every reusable buffer through [`Workspace`]. The host implements it over
//! its own pool, so the pool's identity, retention policy and steady-state allocation behaviour are
//! preserved and no second allocator is introduced.
//!
//! # Ownership
//!
//! Acquisition **transfers ownership** of a buffer to the caller, and [`Workspace::release`] returns
//! it. This mirrors the host pool: several buffers may be live at once (a thin SVD holds four), a
//! buffer that becomes a result is simply not released, and a miss allocates. A borrowing "lease"
//! contract would break both the simultaneous holds and the result hand-off.
//!
//! # Initialization
//!
//! [`Workspace::acquire_zeroed`] guarantees every element is zero. [`Workspace::acquire_uninit`] is
//! the full-overwrite path: the contents are unspecified and reading an element before writing it is
//! undefined behaviour. An implementation must pick the same variant a call site uses today; the two
//! are not interchangeable.
//!
//! # Concurrency
//!
//! A workspace is **not** used across lanes. Hosts acquire batch and output buffers before fanning
//! out, and per-lane scratch is allocated per lane. `Workspace` therefore has no `Sync` requirement
//! and no concurrent acquisition path.
//!
//! # Example
//!
//! ```
//! use core::mem::MaybeUninit;
//! use tlinalg_traits::{Scalar, Workspace};
//!
//! struct Owned;
//! impl Workspace<f64> for Owned {
//!     fn acquire_zeroed(&mut self, len: usize) -> Vec<f64> { vec![0.0; len] }
//!     fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<f64>> {
//!         (0..len).map(|_| MaybeUninit::uninit()).collect()
//!     }
//!     fn release(&mut self, _buf: Vec<f64>) {}
//! }
//!
//! let mut ws = Owned;
//! assert_eq!(ws.acquire_zeroed(2), [0.0, 0.0]);
//! ```

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};

/// Element types an implementation supports.
///
/// Sealed: only `f32`, `f64`, `Complex32` and `Complex64` implement it. Complex values are
/// `num_complex::Complex`.
pub trait Scalar: crate::sealed::Sealed + Copy + Send + Sync + 'static {}

impl crate::sealed::Sealed for f32 {}
impl crate::sealed::Sealed for f64 {}
impl crate::sealed::Sealed for Complex32 {}
impl crate::sealed::Sealed for Complex64 {}

impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for Complex32 {}
impl Scalar for Complex64 {}

/// Scratch and buffer acquisition for one scalar type.
///
/// Implemented by the host (per scalar) over its buffer pool. See the module documentation for the
/// ownership, initialization and concurrency contracts.
pub trait Workspace<T: Scalar> {
    /// Acquire an owned buffer of `len` elements, every element zeroed.
    ///
    /// # Contract
    ///
    /// Reading any element before writing it yields zero.
    fn acquire_zeroed(&mut self, len: usize) -> Vec<T>;

    /// Acquire an owned buffer of `len` elements whose contents are unspecified.
    ///
    /// # Contract
    ///
    /// The caller must write every element before reading any, and must [`Workspace::release`] it or
    /// transfer it into a result. Reading an unwritten element is undefined behaviour.
    fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<T>>;

    /// Return a buffer to the host's pool.
    ///
    /// # Contract
    ///
    /// `buf` was obtained from this workspace. A buffer that becomes a result is **not** released.
    fn release(&mut self, buf: Vec<T>);
}
