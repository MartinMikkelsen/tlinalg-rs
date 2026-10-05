//! Shared helpers for the family tests: a host-shaped workspace over plain vectors for every
//! scalar, plus the provider-neutral helpers re-exported from `tlinalg-testkit` (dense column-major
//! reference arithmetic in `Complex64`, deterministic matrices, compact strided operands).

#![allow(dead_code, unused_imports)]

use core::mem::MaybeUninit;
use tlinalg_blas::{IndexWorkspace, LapackScalar, Scalar, Workspace};

/// In scope for its methods (`to_c64`, `from_c64`) on every scalar, including output types that
/// only carry the shared bound.
pub use tlinalg_testkit::TestScalar as SharedTestScalar;
pub use tlinalg_testkit::{
    adjoint, assert_close, batch_of, identity, matmul, matrix, transpose, widen, Complex32,
    Complex64, Strided,
};

/// A workspace that counts what it hands out and what comes back.
#[derive(Default)]
pub struct TestWorkspace {
    pub acquired: usize,
    pub released: usize,
}

impl TestWorkspace {
    pub fn outstanding(&self) -> usize {
        self.acquired - self.released
    }
}

impl<T: Scalar + Default> Workspace<T> for TestWorkspace {
    fn acquire_zeroed(&mut self, len: usize) -> Vec<T> {
        self.acquired += 1;
        vec![T::default(); len]
    }
    fn acquire_capacity(&mut self, cap: usize) -> Vec<T> {
        self.acquired += 1;
        Vec::with_capacity(cap)
    }
    fn acquire_uninit(&mut self, len: usize) -> Vec<MaybeUninit<T>> {
        self.acquired += 1;
        (0..len).map(|_| MaybeUninit::uninit()).collect()
    }
    fn release(&mut self, _buf: Vec<T>) {
        self.released += 1;
    }
}

impl IndexWorkspace for TestWorkspace {
    fn acquire_zeroed_index(&mut self, len: usize) -> Vec<i32> {
        self.acquired += 1;
        vec![0; len]
    }
    fn release_index(&mut self, _buf: Vec<i32>) {
        self.released += 1;
    }
}

/// A scalar the LAPACK tests instantiate a kernel for: the shared test scalar, as a LAPACK scalar.
/// The suite compares with `LOOSE_TOL`, since every check runs through a vendor kernel.
pub trait TestScalar: tlinalg_testkit::TestScalar + LapackScalar {}

impl<T: tlinalg_testkit::TestScalar + LapackScalar> TestScalar for T {}
