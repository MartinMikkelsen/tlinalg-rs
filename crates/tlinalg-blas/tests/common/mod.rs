//! Shared helpers for the family tests: a host-shaped workspace over plain vectors for every scalar,
//! and dense column-major arithmetic carried out in `Complex64` so one checker serves all four
//! scalar types.

#![allow(dead_code)]

use core::mem::MaybeUninit;
use num_complex::{Complex32, Complex64};
use tlinalg_blas::{IndexWorkspace, LapackScalar, Scalar, Workspace};

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

/// A scalar the tests can build from and compare in `Complex64`.
pub trait TestScalar: LapackScalar + core::fmt::Debug {
    const TOL: f64;
    const IS_COMPLEX: bool;
    fn from_c64(value: Complex64) -> Self;
    fn to_c64(self) -> Complex64;
}

impl TestScalar for f32 {
    const TOL: f64 = 2e-3;
    const IS_COMPLEX: bool = false;
    fn from_c64(value: Complex64) -> Self {
        value.re as f32
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl TestScalar for f64 {
    const TOL: f64 = 1e-9;
    const IS_COMPLEX: bool = false;
    fn from_c64(value: Complex64) -> Self {
        value.re
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl TestScalar for Complex32 {
    const TOL: f64 = 2e-3;
    const IS_COMPLEX: bool = true;
    fn from_c64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
}

impl TestScalar for Complex64 {
    const TOL: f64 = 1e-9;
    const IS_COMPLEX: bool = true;
    fn from_c64(value: Complex64) -> Self {
        value
    }
    fn to_c64(self) -> Complex64 {
        self
    }
}

/// Convert a buffer to `Complex64`.
pub fn c64<T: TestScalar>(data: &[T]) -> Vec<Complex64> {
    data.iter().map(|&value| value.to_c64()).collect()
}

/// A deterministic, well-conditioned column-major `m x n` matrix; complex for complex scalars.
pub fn matrix<T: TestScalar>(m: usize, n: usize, seed: usize) -> Vec<T> {
    (0..m * n)
        .map(|index| {
            let row = index % m;
            let col = index / m;
            let re = if row == col {
                4.0 + row as f64
            } else {
                0.5 + ((row * 3 + col * 7 + seed * 5) % 7) as f64 * 0.25 - 0.75
            };
            let im = if T::IS_COMPLEX {
                ((row + 2 * col + seed) % 5) as f64 * 0.2 - 0.4
            } else {
                0.0
            };
            T::from_c64(Complex64::new(re, im))
        })
        .collect()
}

/// `a (m x k) * b (k x n)`, column-major.
pub fn matmul(a: &[Complex64], b: &[Complex64], m: usize, k: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for inner in 0..k {
            let scale = b[inner + col * k];
            for row in 0..m {
                out[row + col * m] += a[row + inner * m] * scale;
            }
        }
    }
    out
}

/// The conjugate transpose of a column-major `m x n` matrix.
pub fn adjoint(a: &[Complex64], m: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for row in 0..m {
            out[col + row * n] = a[row + col * m].conj();
        }
    }
    out
}

/// The transpose (no conjugation) of a column-major `m x n` matrix.
pub fn transpose(a: &[Complex64], m: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for row in 0..m {
            out[col + row * n] = a[row + col * m];
        }
    }
    out
}

/// Assert two buffers agree entrywise within `tol` relative to their scale.
pub fn assert_close(actual: &[Complex64], expected: &[Complex64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let scale = expected.iter().map(|v| v.norm()).fold(1.0f64, f64::max);
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() <= tol * scale,
            "{what}: entry {index}: {a} != {e}"
        );
    }
}

/// The identity, column-major `n x n`.
pub fn identity(n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        out[i + i * n] = Complex64::new(1.0, 0.0);
    }
    out
}

/// An owned strided operand for building `RawStridedRef`/`RawStridedMut` descriptors.
pub struct Strided<T> {
    pub data: Vec<T>,
    pub dims: Vec<usize>,
    pub strides: Vec<isize>,
}

impl<T: Copy> Strided<T> {
    /// A compact column-major batch `[rows, cols, batch]` (no batch axis when `batch` is `None`).
    pub fn compact(data: Vec<T>, rows: usize, cols: usize, batch: Option<usize>) -> Self {
        let mut dims = vec![rows, cols];
        let mut strides = vec![1, rows as isize];
        if let Some(batch) = batch {
            dims.push(batch);
            strides.push((rows * cols) as isize);
        }
        Self {
            data,
            dims,
            strides,
        }
    }

    pub fn r(&self) -> strided_view::RawStridedRef<'_, T> {
        strided_view::RawStridedRef::new(&self.data, &self.dims, &self.strides, 0).unwrap()
    }

    pub fn m(&mut self) -> strided_view::RawStridedMut<'_, T> {
        strided_view::RawStridedMut::new(&mut self.data, &self.dims, &self.strides, 0).unwrap()
    }
}

/// `count` compact `rows x cols` matrices with distinct seeds.
pub fn batch_of<T: TestScalar>(rows: usize, cols: usize, count: usize, seed: usize) -> Vec<T> {
    (0..count)
        .flat_map(|item| matrix::<T>(rows, cols, seed + item))
        .collect()
}
