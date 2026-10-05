//! Batch lane policy, computed by the host.
//!
//! A host owns the decision of how a batch is split and how much parallelism each piece may use;
//! this module only carries that decision to the implementation. An implementation must **not**
//! re-derive lane counts from the thread budget: today's hosts select lanes from a resolved batch
//! policy (including forced strategies and thresholds) that a budget alone cannot express.
//!
//! Every batched entry point takes a `LanePlan` next to the call's [`Parallel`]:
//!
//! * `lanes <= 1` — the whole batch runs in order on one lane with `item_parallel`;
//! * `lanes > 1` — the batch is split into `batch.div_ceil(lanes)`-item contiguous chunks, each a
//!   task on the call's pool (or run one after another when the call is
//!   [`Parallel::Sequential`]). A lane always runs its items with [`Parallel::Sequential`], so it
//!   never nests a second fan-out; `item_parallel` applies to the single-lane case only.

use crate::Parallel;

/// How a host has decided to run one batch.
///
/// # Fields
///
/// * `lanes` — how many contiguous, disjoint chunks the host will actually produce. The host
///   partitions with `chunk = batch.div_ceil(lanes)`, so the last chunk may be shorter and `lanes`
///   is an upper bound on the number of chunks, not an exact count. An implementation driven
///   per-chunk must not assume `batch / lanes`.
/// * `item_parallel` — the parallelism for the items of a single-lane run. This can be
///   [`Parallel::Sequential`] even when the surrounding call is parallel, because a forced
///   sequential batch strategy resolves to per-item sequential execution deliberately. A
///   multi-lane run ignores it and runs every lane sequentially.
///
/// # Example
///
/// ```
/// use tlinalg::{LanePlan, Parallel};
/// let plan = LanePlan { lanes: 1, item_parallel: Parallel::Sequential };
/// assert_eq!(plan.lanes, 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct LanePlan<'a> {
    /// Number of contiguous chunks the host will produce (upper bound when `div_ceil` rounds).
    pub lanes: usize,
    /// Parallelism for work inside a single chunk.
    pub item_parallel: Parallel<'a>,
}

impl<'a> LanePlan<'a> {
    /// One lane, items run sequentially on the calling thread.
    ///
    /// ```
    /// assert!(tlinalg::LanePlan::sequential().is_single_lane());
    /// ```
    #[must_use]
    pub const fn sequential() -> Self {
        Self {
            lanes: 1,
            item_parallel: Parallel::Sequential,
        }
    }

    /// One lane, items run in order with `item_parallel`.
    ///
    /// ```
    /// use tlinalg::{LanePlan, Parallel};
    /// assert!(LanePlan::single(Parallel::Sequential).is_single_lane());
    /// ```
    #[must_use]
    pub const fn single(item_parallel: Parallel<'a>) -> Self {
        Self {
            lanes: 1,
            item_parallel,
        }
    }

    /// Whether the host chose a single lane, i.e. the caller drives the whole batch.
    #[must_use]
    pub fn is_single_lane(self) -> bool {
        self.lanes <= 1
    }
}
