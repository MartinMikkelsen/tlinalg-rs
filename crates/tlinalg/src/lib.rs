//! faer-backed, tensor-free linear-algebra kernels.
//!
//! This crate owns the numerical kernels, the batch loop over them, their scratch, and the small
//! vocabulary its callers use to drive them: borrowed strided I/O, [`Parallel`] and typed [`Error`].
//! It does not own tensors, allocation policy, dtype dispatch, placement, or the execution context;
//! a host supplies operands, output vectors and one [`Parallel`] token. Kernel scratch is per lane.
//!
//! The interface a host requires of its linear-algebra providers belongs to that host (tenferro
//! defines it); this crate is one provider and defines only the types its own entry points take.
//! The LAPACK provider (`tlinalg-blas`) is independent of this crate.
//!
//! # Batches
//!
//! Every entry point is batched (`docs/design/batched-api.md` in the repository): an input is a
//! rank-`2 + B` [`strided_view::RawStridedRef`] `[rows, cols, b_1, ..., b_B]`, outputs are compact
//! column-major items in batch order (first batch axis fastest). The library selects Auto lanes
//! from the shape and [`Parallel`] resource. A failing call returns the error of its
//! lowest-indexed failing item and leaves every library-created output vector empty.
//!
//! # Conventions
//!
//! Callers and their tests depend on these:
//!
//! | Operation | Convention |
//! |---|---|
//! | `svd` | returns `U`, non-increasing singular values, and `Vᴴ` (not `V`) |
//! | `eigh` | reads the lower triangle, returns non-decreasing **real** values |
//! | `cholesky` | reads the lower triangle |
//! | factorizations | do **not** fail on exactly singular input unless the table below says so |
//! | solves | report [`Error::Singular`] for a singular factor |
//!
//! Failure behaviour of this provider (it is deliberately not normalized against other providers):
//!
//! | Route | Exactly singular input |
//! |---|---|
//! | partial-pivot LU in `solve` | [`Error::Singular`] |
//! | full-pivot LU | not an error |
//!
//! Dtype dispatch and the deliberate "unsupported for this dtype" cases stay in the host: this
//! crate never names a host `DType`. Terminal storage—tensor construction, placement tagging, and
//! the error wrapper that carries an error into the host's error type—also stays in the host.
//!
//! Parallelism is faer's own: faer takes a thread count and runs on the current rayon registry, so
//! an implementation **installs the caller's pool** for the duration of a call and derives
//! `faer::Par` from the pool-clamped requested budget. faer's count is a hint, not a strict
//! active-thread bound (see [`Parallel`]); outer lanes are bounded and run sequential children.
//! No pool is created and the ambient pool is never selected. Effective width one stays on the caller.

#![warn(missing_docs)]

pub mod cholesky;
pub mod eig;
pub mod eigh;
pub mod error;
pub mod full_piv_lu;
pub mod householder;
mod lane;
pub mod lu;
pub mod packed_lu;
pub mod parallel;
pub mod qr;
pub mod scratch;
pub mod svd;
pub mod triangular_solve;

mod batch;
mod scalar;
mod util;

pub use error::{Error, NonFiniteRole, Op, Result};
pub use parallel::Parallel;
pub use scalar::FaerScalar;
pub use scratch::Scalar;

mod sealed {
    /// Seals [`crate::Scalar`].
    pub trait Sealed {}
}

/// Run `f` on the caller's pool, with faer parallelism derived from the caller's budget.
///
/// [`Parallel::Sequential`] runs on the calling thread with `faer::Par::Seq`.
/// [`Parallel::Pool`] installs the supplied pool, so the pool is selected by the value the caller
/// passed and not by whatever thread the call happens to run on. Installing a pool the caller is
/// already inside runs the closure in place, so a host that has already entered its domain pays
/// nothing.
///
/// Clamp the requested faer count to pool width. This is a hint, not a hard active-thread bound.
/// Effective width one stays on the caller without installing a pool.
pub(crate) fn with_parallel<R>(par: Parallel<'_>, f: impl FnOnce(faer::Par) -> R + Send) -> R
where
    R: Send,
{
    match par.bounded() {
        Parallel::Sequential => f(faer::Par::Seq),
        Parallel::Pool { pool, budget } => pool.install(|| f(faer::Par::rayon(budget.get()))),
    }
}

#[cfg(test)]
mod tests {
    use super::{with_parallel, Parallel};
    use core::num::NonZeroUsize;

    /// The pool the caller passes is the pool the work runs on: not the ambient one, and not a
    /// pool discovered from the thread the call happens to start on.
    #[test]
    fn the_supplied_pool_is_the_one_installed() {
        let two = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("a 2-thread pool");
        let three = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .expect("a 3-thread pool");
        let observed_two = with_parallel(
            Parallel::Pool {
                pool: &two,
                budget: NonZeroUsize::new(2).unwrap(),
            },
            |_| rayon::current_num_threads(),
        );
        let observed_three = with_parallel(
            Parallel::Pool {
                pool: &three,
                budget: NonZeroUsize::new(3).unwrap(),
            },
            |_| rayon::current_num_threads(),
        );
        assert_eq!(observed_two, 2);
        assert_eq!(observed_three, 3);
    }
}
