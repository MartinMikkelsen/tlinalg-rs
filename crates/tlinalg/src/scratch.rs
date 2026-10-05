//! The scalar types the kernels support.

use num_complex::{Complex32, Complex64};

/// Element types an implementation supports.
///
/// Sealed: only `f32`, `f64`, `Complex32` and `Complex64` implement it. Complex values are
/// `num_complex::Complex`.
pub trait Scalar: crate::sealed::Sealed + Copy + Send + Sync + 'static {}

impl crate::sealed::Sealed for f32 {}
impl crate::sealed::Sealed for f64 {}
impl crate::sealed::Sealed for Complex32 {}
impl crate::sealed::Sealed for Complex64 {}

impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for Complex32 {}
impl Scalar for Complex64 {}
