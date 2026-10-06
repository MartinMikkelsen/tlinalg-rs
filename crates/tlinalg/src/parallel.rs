//! The host supplies a resource; tlinalg decides batch lanes and requested faer parallelism.
//! No global or ambient pool is selected.

use core::num::NonZeroUsize;
use rayon::ThreadPool;

/// The caller-owned numerical resource for one call.
///
/// [`Parallel::Sequential`] runs on the calling thread without implementation-owned fan-out.
/// [`Parallel::Pool`] names a pool and requested budget; the caller need not enter it first.
///
/// # Budget contract
///
/// tlinalg resolves outer batch lanes using `min(budget, pool.current_num_threads())`. Their count
/// never exceeds that width, and their items are sequential. Otherwise one lane passes the
/// effective width to faer on the supplied pool. Effective width one runs on the calling thread
/// without entering a pool.
///
/// **faer's count is a hint, not a strict active-thread bound.** faer 0.24.4 can expose more
/// intra-item numerical tasks than requested (wide-RHS recursion and rounded-up split counts).
/// `budget` bounds tlinalg's outer lanes and the count requested from faer, not every active native
/// numerical thread. Admission across simultaneous calls belongs to the resource owner.
/// Vendor BLAS/LAPACK threading is outside this token's contract.
///
/// # Examples
///
/// ```
/// use tlinalg::Parallel;
/// assert_eq!(Parallel::Sequential.budget().get(), 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub enum Parallel<'a> {
    /// Run on the calling thread with no implementation-owned parallelism.
    Sequential,
    /// Use `pool`; bound outer lanes by `budget` and pass it as a faer count hint.
    Pool {
        /// Caller-owned pool selected for numerical execution.
        pool: &'a ThreadPool,
        /// Outer-lane ceiling and requested intra-item count; see the budget contract.
        budget: NonZeroUsize,
    },
}

impl Parallel<'_> {
    /// Clamp the request to the named pool and avoid pool entry at effective width one.
    pub(crate) fn bounded(self) -> Self {
        match self {
            Self::Pool { pool, budget } => {
                let width = budget.get().min(pool.current_num_threads());
                match NonZeroUsize::new(width).filter(|width| width.get() > 1) {
                    Some(budget) => Self::Pool { pool, budget },
                    None => Self::Sequential,
                }
            }
            Self::Sequential => Self::Sequential,
        }
    }

    /// The caller's requested budget (not pool-clamped), or `1` for [`Parallel::Sequential`].
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(tlinalg::Parallel::Sequential.budget().get(), 1);
    /// ```
    #[must_use]
    pub fn budget(self) -> NonZeroUsize {
        match self {
            Self::Sequential => NonZeroUsize::MIN,
            Self::Pool { budget, .. } => budget,
        }
    }
}
