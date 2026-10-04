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
pub mod svd;

/// Link the vendor libraries for this crate's own tests.
///
/// Tenferro supplies the symbols through its own provider features; this is only so that
/// `cargo test --features link-openblas` can run the checks in this repository.
#[cfg(feature = "link-openblas")]
extern crate blas_src as _;
#[cfg(feature = "link-openblas")]
extern crate lapack_src as _;

/// The vendor symbols a scalar needs, sealed away from the public bound.
#[doc(hidden)]
pub mod symbols {
    /// The LAPACK routines and scalar literals this crate needs.
    ///
    /// `pub` only so [`crate::LapackScalar`] can name its associated types; sealed by
    /// [`tlinalg_traits::Scalar`], which no new type can implement, and by the orphan rule for the
    /// existing ones.
    pub trait Symbols: tlinalg_traits::Scalar + Default + PartialEq {
        /// The real scalar this type's singular values live in.
        type Real: tlinalg_traits::Scalar + Default + PartialEq;

        /// `1` in this scalar.
        fn one() -> Self;
        /// `-1` in this scalar.
        fn minus_one() -> Self;
        /// The `lwork` a workspace query returned, as a real scalar.
        fn work_query_len(query: Self) -> f64;

        /// The LAPACK routine the selected driver calls, for diagnostics.
        fn routine_name() -> &'static str;
        /// `?getrf`: factor one column-major `m x n` matrix in place.
        ///
        /// # Safety
        ///
        /// `data` holds a mutable column-major `lda x n` matrix, `ipiv` holds at least
        /// `min(m, n)` pivots, and `m`, `n`, `lda` describe them.
        unsafe fn getrf(
            m: i32,
            n: i32,
            data: &mut [Self],
            lda: i32,
            ipiv: &mut [i32],
            info: &mut i32,
        );
        /// `?getrs`: solve `op(A) X = B` from `?getrf` factors.
        ///
        /// # Safety
        ///
        /// `a` and `ipiv` are a prior matching `?getrf` factorization, `b` holds a mutable
        /// `ldb x nrhs` right-hand side, and the dimensions describe them.
        #[allow(clippy::too_many_arguments)]
        unsafe fn getrs(
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

        /// The SVD driver for one column-major `m x n` matrix, in place.
        ///
        /// `jobu` and `jobvt` are the two job letters `?gesvd` takes; `?gesdd` shares one letter, so
        /// it is given `jobu`. Which routine is called is a compiled-in choice: without
        /// `provider-inject` this is `?gesdd`, and with it `?gesvd`, because the injected LAPACK
        /// symbol set exports `?gesvd` only.
        ///
        /// `rwork` is empty for the real routines and required by the complex ones; `iwork` is
        /// required by `?gesdd` and ignored by `?gesvd`.
        #[allow(clippy::too_many_arguments)]
        /// # Safety
        ///
        /// Every buffer holds what the selected LAPACK routine documents for these dimensions:
        /// `a` a mutable `m x n` matrix, `work` at least the queried `lwork`, `rwork` the
        /// documented real workspace, `iwork` the integer workspace, and `u`/`vt` the factors.
        unsafe fn svd_driver(
            jobu: u8,
            jobvt: u8,
            m: i32,
            n: i32,
            a: &mut [Self],
            lda: i32,
            s: &mut [Self::Real],
            u: &mut [Self],
            ldu: i32,
            vt: &mut [Self],
            ldvt: i32,
            work: &mut [Self],
            lwork: i32,
            rwork: &mut [Self::Real],
            iwork: &mut [i32],
            info: &mut i32,
        );
    }

    macro_rules! impl_real_symbols {
        ($scalar:ty, $getrf:path, $getrs:path, $gesdd:path, $gesvd:path, $gesdd_routine:literal, $gesvd_routine:literal) => {
            impl Symbols for $scalar {
                type Real = $scalar;

                fn one() -> Self {
                    1.0
                }

                fn minus_one() -> Self {
                    -1.0
                }

                fn work_query_len(query: Self) -> f64 {
                    query as f64
                }

                #[cfg(not(feature = "provider-inject"))]
                fn routine_name() -> &'static str {
                    $gesdd_routine
                }

                #[cfg(feature = "provider-inject")]
                fn routine_name() -> &'static str {
                    $gesvd_routine
                }

                unsafe fn getrf(
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

                unsafe fn getrs(
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

                unsafe fn svd_driver(
                    jobu: u8,
                    jobvt: u8,
                    m: i32,
                    n: i32,
                    a: &mut [Self],
                    lda: i32,
                    s: &mut [Self::Real],
                    u: &mut [Self],
                    ldu: i32,
                    vt: &mut [Self],
                    ldvt: i32,
                    work: &mut [Self],
                    lwork: i32,
                    _rwork: &mut [Self::Real],
                    iwork: &mut [i32],
                    info: &mut i32,
                ) {
                    // The driver is a compiled-in choice: the injected LAPACK symbol set exports
                    // `?gesvd` but not `?gesdd`.
                    #[cfg(feature = "provider-inject")]
                    let _ = iwork;
                    #[cfg(not(feature = "provider-inject"))]
                    let _ = jobvt;
                    // SAFETY: callers validate the dimensions and layouts and supply buffers of the
                    // lengths LAPACK documents; `lwork = -1` with a one-element `work` is the
                    // workspace query.
                    #[cfg(not(feature = "provider-inject"))]
                    unsafe {
                        $gesdd(
                            jobu, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info,
                        );
                    }
                    #[cfg(feature = "provider-inject")]
                    unsafe {
                        $gesvd(
                            jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info,
                        );
                    }
                }
            }
        };
    }

    macro_rules! impl_complex_symbols {
        ($scalar:ty, $real:ty, $getrf:path, $getrs:path, $gesdd:path, $gesvd:path, $gesdd_routine:literal, $gesvd_routine:literal) => {
            impl Symbols for $scalar {
                type Real = $real;

                fn one() -> Self {
                    Self::new(1.0, 0.0)
                }

                fn minus_one() -> Self {
                    Self::new(-1.0, 0.0)
                }

                fn work_query_len(query: Self) -> f64 {
                    query.re as f64
                }

                #[cfg(not(feature = "provider-inject"))]
                fn routine_name() -> &'static str {
                    $gesdd_routine
                }

                #[cfg(feature = "provider-inject")]
                fn routine_name() -> &'static str {
                    $gesvd_routine
                }

                unsafe fn getrf(
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

                unsafe fn getrs(
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

                unsafe fn svd_driver(
                    jobu: u8,
                    jobvt: u8,
                    m: i32,
                    n: i32,
                    a: &mut [Self],
                    lda: i32,
                    s: &mut [Self::Real],
                    u: &mut [Self],
                    ldu: i32,
                    vt: &mut [Self],
                    ldvt: i32,
                    work: &mut [Self],
                    lwork: i32,
                    rwork: &mut [Self::Real],
                    iwork: &mut [i32],
                    info: &mut i32,
                ) {
                    // The driver is a compiled-in choice, as in the real case.
                    #[cfg(feature = "provider-inject")]
                    let _ = iwork;
                    #[cfg(not(feature = "provider-inject"))]
                    let _ = jobvt;
                    // SAFETY: as in the real case; the complex routines additionally require
                    // `rwork` of the length LAPACK documents.
                    #[cfg(not(feature = "provider-inject"))]
                    unsafe {
                        $gesdd(
                            jobu, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork,
                            info,
                        );
                    }
                    #[cfg(feature = "provider-inject")]
                    unsafe {
                        $gesvd(
                            jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork,
                            info,
                        );
                    }
                }
            }
        };
    }

    impl_real_symbols!(
        f32,
        lapack::sgetrf,
        lapack::sgetrs,
        lapack::sgesdd,
        lapack::sgesvd,
        "sgesdd",
        "sgesvd"
    );
    impl_real_symbols!(
        f64,
        lapack::dgetrf,
        lapack::dgetrs,
        lapack::dgesdd,
        lapack::dgesvd,
        "dgesdd",
        "dgesvd"
    );
    impl_complex_symbols!(
        num_complex::Complex32,
        f32,
        lapack::cgetrf,
        lapack::cgetrs,
        lapack::cgesdd,
        lapack::cgesvd,
        "cgesdd",
        "cgesvd"
    );
    impl_complex_symbols!(
        num_complex::Complex64,
        f64,
        lapack::zgetrf,
        lapack::zgetrs,
        lapack::zgesdd,
        lapack::zgesvd,
        "zgesdd",
        "zgesvd"
    );
}

/// A scalar this crate has LAPACK bindings for.
///
/// Sealed through `symbols::Symbols`: the four scalars are the only implementors.
pub trait LapackScalar: tlinalg_traits::Scalar + Default + PartialEq + symbols::Symbols {}

impl LapackScalar for f32 {}
impl LapackScalar for f64 {}
impl LapackScalar for num_complex::Complex32 {}
impl LapackScalar for num_complex::Complex64 {}
