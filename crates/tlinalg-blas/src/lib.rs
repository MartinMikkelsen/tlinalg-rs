//! LAPACK/BLAS-backed implementation of the [`tlinalg_traits`] interface.
//!
//! This crate owns the vendor calls and their argument marshalling. It does not own tensors,
//! allocation, dtype dispatch, placement, or the execution context, and it never touches vendor
//! threading: LAPACK and BLAS own their own parallelism, so a [`tlinalg_traits::Parallel`] token is
//! accepted for interface parity and ignored.
//!
//! # Threading
//!
//! Every batch loop in this crate is serial, and the vendor call inside it does the threading. That
//! is deliberate: a Rayon fan-out around a vendor call would fight the vendor's own pool, so the
//! host is expected to place vendor work rather than parallelise inside it.

#![warn(missing_docs)]

pub mod lu;

/// Link the vendor libraries for this crate's own tests.
///
/// Tenferro supplies the symbols through its own provider features; this is only so that
/// `cargo test --features link-openblas` can run the checks in this repository.
#[cfg(feature = "link-openblas")]
extern crate blas_src as _;
#[cfg(feature = "link-openblas")]
extern crate lapack_src as _;

/// The vendor symbols a scalar needs, sealed away from the public bound.
mod symbols {
    /// The LAPACK routines and scalar literals this crate needs.
    ///
    /// `pub` so it can be a supertrait of [`crate::LapackScalar`] while staying inside this private
    /// module: callers cannot implement it, and its signatures stay free to change.
    pub trait Symbols: tlinalg_traits::Scalar + Default + PartialEq {
        /// `1` in this scalar.
        fn one() -> Self;
        /// `-1` in this scalar.
        fn minus_one() -> Self;
        /// `?getrf`: factor one column-major `m x n` matrix in place.
        fn getrf(m: i32, n: i32, data: &mut [Self], lda: i32, ipiv: &mut [i32], info: &mut i32);
        /// `?getrs`: solve `op(A) X = B` from `?getrf` factors.
        #[allow(clippy::too_many_arguments)]
        fn getrs(
            trans: u8,
            n: i32,
            nrhs: i32,
            a: &[Self],
            lda: i32,
            ipiv: &[i32],
            b: &mut [Self],
            ldb: i32,
            info: &mut i32,
        );
        /// Conjugate every element in place; a no-op for real scalars.
        fn conj_in_place(_data: &mut [Self]) {}
    }

    macro_rules! impl_real_symbols {
        ($scalar:ty, $getrf:path, $getrs:path) => {
            impl Symbols for $scalar {
                fn one() -> Self {
                    1.0
                }

                fn minus_one() -> Self {
                    -1.0
                }

                fn getrf(
                    m: i32,
                    n: i32,
                    data: &mut [Self],
                    lda: i32,
                    ipiv: &mut [i32],
                    info: &mut i32,
                ) {
                    // SAFETY: callers validate `m`, `n` and `lda`, provide a mutable column-major
                    // `lda x n` matrix, `min(m, n)` pivots, and live `info`.
                    unsafe {
                        $getrf(m, n, data, lda, ipiv, info);
                    }
                }

                fn getrs(
                    trans: u8,
                    n: i32,
                    nrhs: i32,
                    a: &[Self],
                    lda: i32,
                    ipiv: &[i32],
                    b: &mut [Self],
                    ldb: i32,
                    info: &mut i32,
                ) {
                    // SAFETY: `a` holds a prior `getrf` factorization, `ipiv` matches it, `b` is a
                    // mutable `ldb x nrhs` right-hand side, and dimensions are validated.
                    unsafe {
                        $getrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
                    }
                }
            }
        };
    }

    macro_rules! impl_complex_symbols {
        ($scalar:ty, $getrf:path, $getrs:path) => {
            impl Symbols for $scalar {
                fn one() -> Self {
                    Self::new(1.0, 0.0)
                }

                fn minus_one() -> Self {
                    Self::new(-1.0, 0.0)
                }

                fn getrf(
                    m: i32,
                    n: i32,
                    data: &mut [Self],
                    lda: i32,
                    ipiv: &mut [i32],
                    info: &mut i32,
                ) {
                    // SAFETY: as in the real case.
                    unsafe {
                        $getrf(m, n, data, lda, ipiv, info);
                    }
                }

                fn getrs(
                    trans: u8,
                    n: i32,
                    nrhs: i32,
                    a: &[Self],
                    lda: i32,
                    ipiv: &[i32],
                    b: &mut [Self],
                    ldb: i32,
                    info: &mut i32,
                ) {
                    // SAFETY: as in the real case.
                    unsafe {
                        $getrs(trans, n, nrhs, a, lda, ipiv, b, ldb, info);
                    }
                }

                fn conj_in_place(data: &mut [Self]) {
                    for value in data {
                        *value = value.conj();
                    }
                }
            }
        };
    }

    impl_real_symbols!(f32, lapack::sgetrf, lapack::sgetrs);
    impl_real_symbols!(f64, lapack::dgetrf, lapack::dgetrs);
    impl_complex_symbols!(num_complex::Complex32, lapack::cgetrf, lapack::cgetrs);
    impl_complex_symbols!(num_complex::Complex64, lapack::zgetrf, lapack::zgetrs);
}

/// A scalar this crate has LAPACK bindings for.
///
/// Sealed through `symbols::Symbols`: the four scalars are the only implementors.
pub trait LapackScalar: tlinalg_traits::Scalar + Default + PartialEq + symbols::Symbols {}

impl LapackScalar for f32 {}
impl LapackScalar for f64 {}
impl LapackScalar for num_complex::Complex32 {}
impl LapackScalar for num_complex::Complex64 {}
