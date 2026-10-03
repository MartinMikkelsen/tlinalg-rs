//! faer-backed implementation of the [`tlinalg_traits`] interface.
//!
//! This crate owns the per-item numerical kernels and their scratch. It does not own tensors,
//! allocation policy, dtype dispatch, placement, or the execution context; a host supplies borrowed
//! operands, a [`tlinalg_traits::Parallel`] token, a host-resolved [`tlinalg_traits::LanePlan`], and
//! (where a kernel needs pooled buffers) a [`tlinalg_traits::Workspace`].
//!
//! Parallelism is faer's own: faer takes a thread count and runs on the current rayon registry, so
//! an implementation **installs the caller's pool** for the duration of a call and derives
//! `faer::Par` from the caller's budget. It never creates a pool and never falls back to the
//! ambient one.

#![warn(missing_docs)]

pub mod packed_lu;

mod scalar;

pub use scalar::FaerScalar;

use tlinalg_traits::Parallel;

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
