//! Parallelism handed to a numerical implementation.
//!
//! A host decides parallelism; an implementation consumes it. Implementations must not read global
//! or ambient state to discover how many threads they may use.

use core::num::NonZeroUsize;

use rayon::ThreadPool;

/// The parallelism an implementation may use for one call.
///
/// [`Parallel::Sequential`] means "no implementation-owned parallelism": run the work on the
/// calling thread, equivalent to a sequential provider policy. It is not "one thread of some pool".
///
/// [`Parallel::Pool`] names the pool the host has already installed the call in, together with the
/// thread budget for this call.
///
/// # Budget contract
///
/// `budget` is an upper bound the **implementation** must honour for the parallelism it starts
/// itself. Selecting the pool does not impose it: `ThreadPool::install` runs work on that pool, and
/// the pool may be larger than `budget`. Work the implementation fans out over — over the items of
/// one chunk, say — must fit in `budget` threads (a count-based provider policy already does so).
///
/// The batch's outer fan-out is not the implementation's to bound or to re-derive: it is the number
/// of tasks the host asked for in [`crate::LanePlan`], which the host resolves from this same
/// budget. A batched entry point runs exactly that many tasks.
///
/// # Example
///
/// ```
/// use tlinalg::Parallel;
/// assert!(matches!(Parallel::Sequential, Parallel::Sequential));
/// ```
#[derive(Clone, Copy, Debug)]
pub enum Parallel<'a> {
    /// Run on the calling thread with no implementation-owned parallelism.
    Sequential,
    /// Run on `pool`, using at most `budget` threads.
    Pool {
        /// Pool the host has already entered for this call.
        pool: &'a ThreadPool,
        /// Maximum threads this call may use.
        budget: NonZeroUsize,
    },
}

impl Parallel<'_> {
    /// The thread budget, or `1` for [`Parallel::Sequential`].
    #[must_use]
    pub fn budget(self) -> NonZeroUsize {
        match self {
            Self::Sequential => NonZeroUsize::MIN,
            Self::Pool { budget, .. } => budget,
        }
    }
}
