//! faer-backed, tensor-free linear-algebra kernels.
//!
//! This crate owns the per-item numerical kernels, their scratch, and the small vocabulary its
//! callers use to drive them: borrowed strided I/O, [`Parallel`], [`LanePlan`], [`Workspace`] and
//! the typed [`Error`]. It does not own tensors, allocation policy, dtype dispatch, placement, or the
//! execution context; a host supplies borrowed operands, a [`Parallel`] token, a host-resolved
//! [`LanePlan`], and (where a kernel needs pooled buffers) a [`Workspace`].
//!
//! The interface a host requires of its linear-algebra providers belongs to that host (tenferro
//! defines it); this crate is one provider and defines only the types its own entry points take.
//! The LAPACK provider (`tlinalg-blas`) is independent of this crate.
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
//! `faer::Par` from the caller's budget. It never creates a pool and never falls back to the
//! ambient one.

#![warn(missing_docs)]

pub mod cholesky;
pub mod eig;
pub mod eigh;
pub mod error;
pub mod full_piv_lu;
pub mod householder;
pub mod lane;
pub mod lu;
pub mod packed_lu;
pub mod parallel;
pub mod qr;
pub mod scratch;
pub mod svd;
pub mod triangular_solve;

mod scalar;
mod util;

pub use error::{Error, NonFiniteRole, Op, Result};
pub use lane::LanePlan;
pub use parallel::Parallel;
pub use scalar::FaerScalar;
pub use scratch::{IndexWorkspace, Scalar, Workspace};

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
/// A budget larger than the pool is the caller's business: faer is told the budget and bounds its
/// own fan-out by it.
pub(crate) fn with_parallel<R>(par: Parallel<'_>, f: impl FnOnce(faer::Par) -> R + Send) -> R
where
    R: Send,
{
    match par {
        Parallel::Sequential => f(faer::Par::Seq),
        Parallel::Pool { pool, budget } => pool.install(|| f(faer::Par::rayon(budget.get()))),
    }
}

/// The `faer::Par` a token denotes, for sizing work that does not run yet.
///
/// Scratch sizing depends on the thread count faer will use, so a caller that builds scratch
/// before entering the pool needs the same mapping [`with_parallel`] applies.
pub(crate) fn faer_par(par: Parallel<'_>) -> faer::Par {
    match par {
        Parallel::Sequential => faer::Par::Seq,
        Parallel::Pool { budget, .. } => faer::Par::rayon(budget.get()),
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
