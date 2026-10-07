//! Provider-neutral test helpers shared by the `tlinalg` and `tlinalg-blas` test suites and the
//! cross-provider parity tests.
//!
//! Dev-only and unpublished. Nothing here names a provider: the helpers are generic over the four
//! scalar types through [`TestScalar`], carry dense reference arithmetic in [`Complex64`], and build
//! batched strided operands. A provider's own suite adds its scalar bound on top (for example
//! `T: TestScalar + tlinalg::FaerScalar`).
//!
//! This crate must never become a dependency of either provider: it is a `[dev-dependencies]`
//! entry only, so the providers' library graphs stay independent of each other and of it.

#![allow(clippy::needless_range_loop)]

pub mod alloc;

pub use num_complex::{Complex32, Complex64};
use strided_view::{RawStridedMut, RawStridedRef};

/// A scalar the tests instantiate a kernel for.
pub trait TestScalar:
    Copy + Default + PartialEq + core::fmt::Debug + Send + Sync + 'static
{
    /// Whether the scalar is complex.
    const COMPLEX: bool;
    /// Residual tolerance for a single factorization at this precision.
    const TOL: f64;
    /// Looser tolerance for longer pipelines (vendor kernels, solves through a factorization,
    /// cross-provider comparisons).
    const LOOSE_TOL: f64;
    /// Narrow a reference value into this scalar (the imaginary part is dropped for real types).
    fn from_c64(value: Complex64) -> Self;
    /// Widen this scalar into the reference type.
    fn to_c64(self) -> Complex64;
}

impl TestScalar for f32 {
    const COMPLEX: bool = false;
    const TOL: f64 = 1e-4;
    const LOOSE_TOL: f64 = 2e-3;
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
    const LOOSE_TOL: f64 = 1e-9;
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
    const LOOSE_TOL: f64 = 2e-3;
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
    const LOOSE_TOL: f64 = 1e-9;
    fn from_c64(value: Complex64) -> Self {
        value
    }
    fn to_c64(self) -> Complex64 {
        self
    }
}

/// Widen a buffer into the reference type.
pub fn widen<T: TestScalar>(data: &[T]) -> Vec<Complex64> {
    data.iter().map(|&value| value.to_c64()).collect()
}

/// Narrow a reference buffer into `T`.
pub fn narrow<T: TestScalar>(data: &[Complex64]) -> Vec<T> {
    data.iter().map(|&value| T::from_c64(value)).collect()
}

/// A deterministic, well-conditioned column-major `m x n` matrix (diagonally dominant on its
/// leading square block), with imaginary parts for the complex scalars. Distinct `seed`s give
/// distinct matrices.
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

/// A deterministic column-major `m x n` matrix whose partial and full pivoting both produce
/// **non-symmetric** permutations.
///
/// Small entries from [`matrix`] (scaled by `0.05`) plus one dominant entry per leading index `i`,
/// at row `(i + 1) mod m` and column `i`, with well-separated magnitudes in a scrambled
/// order. A diagonally dominant matrix pivots trivially (identity or a single swap, both
/// symmetric), which cannot tell `P A = L U` from `A = P L U`; this one can.
pub fn pivoting<T: TestScalar>(m: usize, n: usize, seed: usize) -> Vec<T> {
    const MAGNITUDES: [f64; 5] = [30.0, 10.0, 50.0, 20.0, 40.0];
    let mut a = widen(&matrix::<T>(m, n, seed));
    for value in &mut a {
        *value *= 0.05;
    }
    if m > 0 && n > 0 {
        for i in 0..m.min(n) {
            let magnitude = MAGNITUDES[i % 5] * (1 + i / 5) as f64;
            let phase = if T::COMPLEX {
                Complex64::new(0.8, 0.6)
            } else {
                Complex64::new(1.0, 0.0)
            };
            a[(i + 1) % m + i * m] = phase * magnitude;
        }
    }
    narrow(&a)
}

/// Whether a column-major `n x n` matrix equals its transpose.
pub fn is_symmetric(a: &[Complex64], n: usize) -> bool {
    (0..n).all(|col| (0..n).all(|row| a[row + col * n] == a[col + row * n]))
}

/// `count` compact `rows x cols` matrices with consecutive seeds starting at `seed`, back to back.
pub fn batch_of<T: TestScalar>(rows: usize, cols: usize, count: usize, seed: usize) -> Vec<T> {
    (0..count)
        .flat_map(|item| matrix::<T>(rows, cols, seed + item))
        .collect()
}

/// A Hermitian positive-definite `n x n` matrix `B Bᴴ + n I`, `B = matrix(n, n, seed)`.
pub fn hpd_seeded<T: TestScalar>(n: usize, seed: usize) -> Vec<T> {
    let b = widen(&matrix::<T>(n, n, seed));
    let mut a = matmul(&b, &adjoint(&b, n, n), n, n, n);
    for i in 0..n {
        a[i + i * n] += Complex64::new(n as f64, 0.0);
    }
    narrow(&a)
}

/// A Hermitian positive-definite `n x n` matrix.
pub fn hpd<T: TestScalar>(n: usize) -> Vec<T> {
    hpd_seeded(n, 1)
}

/// A Hermitian `n x n` matrix (not necessarily definite), `(B + Bᴴ) / 2`.
pub fn hermitian_seeded<T: TestScalar>(n: usize, seed: usize) -> Vec<T> {
    let b = widen(&matrix::<T>(n, n, seed));
    let bh = adjoint(&b, n, n);
    let a: Vec<Complex64> = b.iter().zip(&bh).map(|(x, y)| (x + y) * 0.5).collect();
    narrow(&a)
}

/// A Hermitian `n x n` matrix (not necessarily definite).
pub fn hermitian<T: TestScalar>(n: usize) -> Vec<T> {
    hermitian_seeded(n, 2)
}

/// A rank-deficient `n x n` matrix (`n >= 2`): `matrix(n, n, seed)` with its last column replaced
/// by a copy of its first, so it is exactly singular in any precision.
pub fn singular<T: TestScalar>(n: usize, seed: usize) -> Vec<T> {
    let mut a = matrix::<T>(n, n, seed);
    for row in 0..n {
        a[row + (n - 1) * n] = a[row];
    }
    a
}

/// splitmix64 mapped to `[-1/2, 1/2)`.
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

/// Column-major `n x n` unitary matrix (orthogonal when `complex` is false): pseudo-random
/// columns orthonormalized by two passes of modified Gram-Schmidt.
fn unitary(n: usize, complex: bool, state: &mut u64) -> Vec<Complex64> {
    let mut q: Vec<Complex64> = (0..n * n)
        .map(|_| {
            let re = uniform(state);
            Complex64::new(re, if complex { uniform(state) } else { 0.0 })
        })
        .collect();
    for col in 0..n {
        for _ in 0..2 {
            for prev in 0..col {
                let dot: Complex64 = (0..n)
                    .map(|row| q[row + prev * n].conj() * q[row + col * n])
                    .sum();
                for row in 0..n {
                    let update = dot * q[row + prev * n];
                    q[row + col * n] -= update;
                }
            }
        }
        let norm = (0..n)
            .map(|row| q[row + col * n].norm_sqr())
            .sum::<f64>()
            .sqrt();
        for row in 0..n {
            q[row + col * n] /= norm;
        }
    }
    q
}

/// Column-major square matrix `U diag(spectrum) Vᴴ` with pseudo-random unitary `U` and `V`
/// (orthogonal for the real scalars), so its singular values are `spectrum`.
pub fn with_singular_values<T: TestScalar>(spectrum: &[f64], seed: u64) -> Vec<T> {
    let n = spectrum.len();
    let mut state = seed;
    let left = unitary(n, T::COMPLEX, &mut state);
    let right = unitary(n, T::COMPLEX, &mut state);
    let mut a = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for (j, &value) in spectrum.iter().enumerate() {
            if value == 0.0 {
                continue;
            }
            let weight = value * right[col + j * n].conj();
            for row in 0..n {
                a[row + col * n] += left[row + j * n] * weight;
            }
        }
    }
    narrow(&a)
}

/// Singular values of a rank-`n / 2` matrix that are clustered: three distinct values close to
/// one, then ones, then zeros. For `n = 160` and `f64` or `Complex64`, the divide-and-conquer
/// bidiagonal SVD of faer 0.24.4 returns `2^-13` in place of the first zero and factors that
/// reproduce [`with_singular_values`] of this spectrum only to `1.4e-5`
/// (<https://github.com/tensor4all/tlinalg-rs/issues/13>).
pub fn clustered_spectrum(n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| match index {
            0 => 1.0037,
            1 => 1.00024,
            2 => 1.000002,
            index if index < n / 2 => 1.0,
            _ => 0.0,
        })
        .collect()
}

/// Column-major `(m x k) (k x n)` in the reference type.
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

/// Plain transpose (no conjugation) of a column-major `m x n` matrix.
pub fn transpose(a: &[Complex64], m: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for col in 0..n {
        for row in 0..m {
            out[col + row * n] = a[row + col * m];
        }
    }
    out
}

/// The `n x n` identity in the reference type.
pub fn identity(n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        out[i + i * n] = Complex64::new(1.0, 0.0);
    }
    out
}

/// Assert two buffers agree elementwise to `tol` relative to the larger magnitude of `want` (or 1).
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
    let mut storage = vec![poison::<T>(); offset + lda * n.max(1)];
    for col in 0..n {
        for row in 0..m {
            storage[offset + row + col * lda] = a[row + col * m];
        }
    }
    (storage, [1, lda as isize], offset as isize)
}

/// The value written outside every item of a test buffer, so a kernel that reads out of its item
/// produces an obviously wrong result.
pub fn poison<T: TestScalar>() -> T {
    T::from_c64(Complex64::new(1.0e30, -1.0e30))
}

/// Instantiate a generic test body as `$name::{f32, f64, c32, c64}` tests.
///
/// The body is named relative to the invoking module (`super::$body`).
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
                super::$body::<$crate::Complex32>();
            }
            #[test]
            fn c64() {
                super::$body::<$crate::Complex64>();
            }
        }
    };
}

/// How a test lays out the batch axes of a strided batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Items back to back, compact: every batch axis coalesces into one.
    Compact,
    /// Padded leading dimension, gaps between items and between axes: no axis merges.
    Gapped,
    /// Gapped, with the batch axes' strides in reverse order (the last batch axis is fastest in
    /// memory).
    Transposed,
}

/// An owned strided `[m, n, batch...]` buffer, poison outside the items.
#[derive(Clone, Debug)]
pub struct BatchBuf<T> {
    /// Backing storage.
    pub storage: Vec<T>,
    /// `[m, n, batch...]`.
    pub dims: Vec<usize>,
    /// Element strides, one per dim.
    pub strides: Vec<isize>,
    /// Element offset of item 0's `(0, 0)`.
    pub offset: isize,
}

impl<T> BatchBuf<T> {
    /// Borrow as an input descriptor.
    pub fn view(&self) -> RawStridedRef<'_, T> {
        RawStridedRef::new(&self.storage, &self.dims, &self.strides, self.offset)
            .expect("a test batch buffer describes its own storage")
    }

    /// Borrow as an output descriptor.
    pub fn view_mut(&mut self) -> RawStridedMut<'_, T> {
        RawStridedMut::new(&mut self.storage, &self.dims, &self.strides, self.offset)
            .expect("a test batch buffer describes its own storage")
    }

    /// The batch dims (everything after the two matrix dims).
    pub fn batch_dims(&self) -> &[usize] {
        &self.dims[2..]
    }

    /// The number of items.
    pub fn count(&self) -> usize {
        self.dims[2..].iter().product()
    }

    /// The element offset of item `index` (first batch axis fastest).
    pub fn item_offset(&self, mut index: usize) -> usize {
        let mut offset = self.offset;
        for (&dim, &stride) in self.dims[2..].iter().zip(&self.strides[2..]) {
            offset += (index % dim) as isize * stride;
            index /= dim;
        }
        offset as usize
    }

    /// Item `index` gathered into a compact column-major buffer.
    pub fn item(&self, index: usize) -> Vec<T>
    where
        T: Copy,
    {
        let base = self.item_offset(index);
        let (m, n) = (self.dims[0], self.dims[1]);
        let mut out = Vec::with_capacity(m * n);
        for col in 0..n {
            for row in 0..m {
                let at =
                    base as isize + row as isize * self.strides[0] + col as isize * self.strides[1];
                out.push(self.storage[at as usize]);
            }
        }
        out
    }
}

/// Lay out `items` (compact `m x n` each) over `batch_dims`.
pub fn batch_buf<T: TestScalar>(
    items: &[Vec<T>],
    m: usize,
    n: usize,
    batch_dims: &[usize],
    layout: Layout,
) -> BatchBuf<T> {
    let count: usize = batch_dims.iter().product();
    assert_eq!(items.len(), count);
    let (lda, gap, axis_gap, offset) = match layout {
        Layout::Compact => (m, 0, 0, 0),
        Layout::Gapped | Layout::Transposed => (m + 1, 2, 3, 2),
    };
    let item_span = lda * n + gap;
    let mut batch_strides = vec![0isize; batch_dims.len()];
    let order: Vec<usize> = match layout {
        Layout::Transposed => (0..batch_dims.len()).rev().collect(),
        _ => (0..batch_dims.len()).collect(),
    };
    let mut stride = item_span.max(1);
    for &axis in &order {
        batch_strides[axis] = stride as isize;
        stride = stride * batch_dims[axis].max(1) + axis_gap;
    }
    let mut buf = BatchBuf {
        storage: vec![poison::<T>(); offset + stride + lda * n + 1],
        dims: [&[m, n][..], batch_dims].concat(),
        strides: [&[1, lda as isize][..], &batch_strides].concat(),
        offset: offset as isize,
    };
    for (index, item) in items.iter().enumerate() {
        let base = buf.item_offset(index);
        for col in 0..n {
            for row in 0..m {
                buf.storage[base + row + col * lda] = item[row + col * m];
            }
        }
    }
    buf
}

/// One compact `m x n` matrix broadcast over `batch_dims`: every batch stride is `0`, so every item
/// reads the same storage (the host's way of expressing torch-style batch broadcasting).
pub fn broadcast_buf<T: TestScalar>(
    item: &[T],
    m: usize,
    n: usize,
    batch_dims: &[usize],
) -> BatchBuf<T> {
    assert_eq!(item.len(), m * n);
    BatchBuf {
        storage: item.to_vec(),
        dims: [&[m, n][..], batch_dims].concat(),
        strides: [&[1, m as isize][..], &vec![0isize; batch_dims.len()]].concat(),
        offset: 0,
    }
}

/// An owned compact strided operand, `[rows, cols]` or `[rows, cols, batch]`.
#[derive(Clone, Debug)]
pub struct Strided<T> {
    /// Backing storage.
    pub data: Vec<T>,
    /// Dims.
    pub dims: Vec<usize>,
    /// Element strides.
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

    /// Borrow as an input descriptor.
    pub fn r(&self) -> RawStridedRef<'_, T> {
        RawStridedRef::new(&self.data, &self.dims, &self.strides, 0)
            .expect("a compact operand describes its own storage")
    }

    /// Borrow as an output descriptor.
    pub fn m(&mut self) -> RawStridedMut<'_, T> {
        RawStridedMut::new(&mut self.data, &self.dims, &self.strides, 0)
            .expect("a compact operand describes its own storage")
    }
}
