//! Shared helpers for the per-family kernel tests. The provider-neutral part (scalar conversion,
//! deterministic matrices, reference arithmetic, batched layouts) lives in `tlinalg-testkit`; this
//! module re-exports it and adds what is specific to `tlinalg` (its scalar bound, pools and lane
//! plans).

#![allow(dead_code, unused_imports)]

pub mod single;

/// In scope for its methods (`to_c64`, `from_c64`) on every scalar, including output types such as
/// `ScalarEntity::Complex` that only carry the shared bound.
pub use tlinalg_testkit::TestScalar as SharedTestScalar;
pub use tlinalg_testkit::{
    adjoint, assert_close, batch_buf, broadcast_buf, for_each_scalar, hermitian, hpd, identity,
    matmul, matrix, narrow, padded, transpose, widen, BatchBuf, Layout,
};

/// A scalar the faer tests instantiate a kernel for: the shared test scalar, as a `tlinalg` scalar.
pub trait TestScalar: tlinalg_testkit::TestScalar + tlinalg::Scalar {}

impl<T: tlinalg_testkit::TestScalar + tlinalg::Scalar> TestScalar for T {}

/// A placeholder for the scratch argument the pre-batching triangular solve took.
#[derive(Default)]
pub struct CountingWorkspace;

/// A two-thread pool, and the parallel token and a three-lane plan over it.
pub fn lanes_pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
}

pub fn pool_token(pool: &rayon::ThreadPool) -> tlinalg::Parallel<'_> {
    tlinalg::Parallel::Pool {
        pool,
        budget: core::num::NonZeroUsize::new(2).unwrap(),
    }
}

pub fn three_lanes() -> tlinalg::LanePlan<'static> {
    tlinalg::LanePlan {
        lanes: 3,
        item_parallel: tlinalg::Parallel::Sequential,
    }
}
