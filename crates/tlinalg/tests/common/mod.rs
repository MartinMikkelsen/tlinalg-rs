//! Shared helpers for the per-family kernel tests: scalar conversion, deterministic matrices, and
//! dense reference arithmetic carried out in `Complex64`.
//!
//! Provider-neutral on purpose: nothing here names faer or a provider type, only the four scalar
//! types and the `Workspace` contract, so these helpers can move into a shared cross-provider
//! testkit unchanged.

#![allow(dead_code)]

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};
use tlinalg::Workspace;

/// A scalar the tests instantiate a kernel for.
pub trait TestScalar: tlinalg::Scalar + Default + PartialEq + core::fmt::Debug {
    /// Whether the scalar is complex.
    const COMPLEX: bool;
    /// Residual tolerance for this precision.
    const TOL: f64;
    /// Narrow a reference value into this scalar (the imaginary part is dropped for real types).
    fn from_c64(value: Complex64) -> Self;
    /// Widen this scalar into the reference type.
    fn to_c64(self) -> Complex64;
}

impl TestScalar for f32 {
    const COMPLEX: bool = false;
    const TOL: f64 = 1e-4;
    fn from_c64(value: Complex64) -> Self {
        value.re as f32
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl TestScalar for f64 {
    const COMPLEX: bool = false;
    const TOL: f64 = 1e-10;
    fn from_c64(value: Complex64) -> Self {
        value.re
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl TestScalar for Complex32 {
    const COMPLEX: bool = true;
    const TOL: f64 = 1e-4;
    fn from_c64(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
}

impl TestScalar for Complex64 {
    const COMPLEX: bool = true;
    const TOL: f64 = 1e-10;
    fn from_c64(value: Complex64) -> Self {
        value
    }
    fn to_c64(self) -> Complex64 {
        self
    }
}

/// Widen a buffer.
pub fn widen<T: TestScalar>(data: &[T]) -> Vec<Complex64> {
    data.iter().map(|&value| value.to_c64()).collect()
}

/// Narrow a buffer.
pub fn narrow<T: TestScalar>(data: &[Complex64]) -> Vec<T> {
    data.iter().map(|&value| T::from_c64(value)).collect()
}

/// A deterministic, well-conditioned column-major `m x n` matrix (diagonally dominant on its
/// leading square block), with imaginary parts for the complex scalars.
pub fn matrix<T: TestScalar>(m: usize, n: usize, seed: usize) -> Vec<T> {
    (0..m * n)
        .map(|index| {
            let row = index % m;
            let col = index / m;
            let re = if row == col {
                4.0 + row as f64
            } else {
                0.25 + ((row * 3 + col * 7 + seed) % 5) as f64 * 0.3 - 0.6
            };
            let im = if T::COMPLEX {
                ((row * 5 + col * 2 + seed) % 7) as f64 * 0.1 - 0.3
            } else {
                0.0
            };
            T::from_c64(Complex64::new(re, im))
        })
        .collect()
}

/// A Hermitian positive-definite `n x n` matrix `B Bᴴ + n I`.
pub fn hpd<T: TestScalar>(n: usize) -> Vec<T> {
    let b = widen(&matrix::<T>(n, n, 1));
    let mut a = matmul(&b, &adjoint(&b, n, n), n, n, n);
    for i in 0..n {
        a[i + i * n] += Complex64::new(n as f64, 0.0);
    }
    narrow(&a)
}

/// A Hermitian `n x n` matrix (not necessarily definite).
pub fn hermitian<T: TestScalar>(n: usize) -> Vec<T> {
    let b = widen(&matrix::<T>(n, n, 2));
    let bh = adjoint(&b, n, n);
    let a: Vec<Complex64> = b.iter().zip(&bh).map(|(x, y)| (x + y) * 0.5).collect();
    narrow(&a)
}

/// Column-major `(m x k) (k x n)`.
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

/// Conjugate transpose of a column-major `m x n` matrix.
pub fn adjoint(a: &[Complex64], m: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for row in 0..m {
            out[col + row * n] = a[row + col * m].conj();
        }
    }
    out
}

/// Plain transpose of a column-major `m x n` matrix.
pub fn transpose(a: &[Complex64], m: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for row in 0..m {
            out[col + row * n] = a[row + col * m];
        }
    }
    out
}

/// Identity `n x n`.
pub fn identity(n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        out[i + i * n] = Complex64::new(1.0, 0.0);
    }
    out
}

/// Assert two buffers agree elementwise to `tol` relative to the larger magnitude (or 1).
pub fn assert_close(got: &[Complex64], want: &[Complex64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().map(|v| v.norm()).fold(1.0, f64::max);
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).norm() <= tol * scale,
            "{what}: element {index} is {g}, expected {w}"
        );
    }
}

/// Embed a column-major `m x n` matrix in a padded buffer: leading dimension `m + 2`, starting at
/// offset `3`, with poison everywhere else. Returns `(storage, strides, offset)`.
pub fn padded<T: TestScalar>(a: &[T], m: usize, n: usize) -> (Vec<T>, [isize; 2], isize) {
    let lda = m + 2;
    let offset = 3usize;
    let poison = T::from_c64(Complex64::new(1.0e30, -1.0e30));
    let mut storage = vec![poison; offset + lda * n.max(1)];
    for col in 0..n {
        for row in 0..m {
            storage[offset + row + col * lda] = a[row + col * m];
        }
    }
    (storage, [1, lda as isize], offset as isize)
}

/// A workspace over the heap that counts its traffic.
#[derive(Default)]
pub struct CountingWorkspace {
    pub acquired: usize,
    pub released: usize,
}

impl<T: tlinalg::Scalar + Default> Workspace<T> for CountingWorkspace {
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

/// Instantiate a generic test body as `$name::{f32, f64, c32, c64}` tests.
#[macro_export]
macro_rules! for_each_scalar {
    ($name:ident, $body:ident) => {
        mod $name {
            #[test]
            fn f32() {
                super::$body::<f32>();
            }
            #[test]
            fn f64() {
                super::$body::<f64>();
            }
            #[test]
            fn c32() {
                super::$body::<num_complex::Complex32>();
            }
            #[test]
            fn c64() {
                super::$body::<num_complex::Complex64>();
            }
        }
    };
}
