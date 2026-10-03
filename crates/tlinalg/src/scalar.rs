//! Scalar types a faer-backed kernel runs on.
//!
//! The public bound is deliberately small: a kernel only needs a scalar that this crate has an
//! implementation for. The faer entity and the layout-preserving casts are crate-internal, so
//! callers never depend on how a scalar is presented to faer.
//!
//! The complex casts are moved from `tenferro-linalg`'s faer backend (same project, MIT OR
//! Apache-2.0), where they reinterpret `num_complex::Complex` as the layout-identical faer scalar.

use num_complex::{Complex32, Complex64};

/// A [`tlinalg_traits::Scalar`] this crate implements kernels for.
///
/// Sealed through `ScalarEntity`, which lives in this private module: implementors outside this
/// crate cannot name the faer entity, so the cast surface stays internal.
pub trait FaerScalar: tlinalg_traits::Scalar + Default + PartialEq + ScalarEntity {}

/// The faer-facing half of a [`FaerScalar`], crate-internal despite being `pub` in this private
/// module.
pub trait ScalarEntity: tlinalg_traits::Scalar + Default + PartialEq {
    /// The faer scalar sharing this type's memory layout.
    type Entity: faer::traits::ComplexField + Copy + PartialEq + Default;

    /// Reinterpret a slice as the faer entity type.
    fn entity_slice(data: &[Self]) -> &[Self::Entity];
    /// Reinterpret a mutable slice as the faer entity type.
    fn entity_slice_mut(data: &mut [Self]) -> &mut [Self::Entity];
    /// The permutation parity scalar: `+1`, or `-1` when `odd`.
    fn parity(odd: bool) -> Self;
}

macro_rules! impl_real_scalar {
    ($scalar:ty) => {
        impl FaerScalar for $scalar {}

        impl ScalarEntity for $scalar {
            type Entity = $scalar;

            fn entity_slice(data: &[Self]) -> &[Self::Entity] {
                data
            }

            fn entity_slice_mut(data: &mut [Self]) -> &mut [Self::Entity] {
                data
            }

            fn parity(odd: bool) -> Self {
                if odd {
                    -1.0
                } else {
                    1.0
                }
            }
        }
    };
}

macro_rules! impl_complex_scalar {
    ($scalar:ty, $entity:ty) => {
        // The casts below are unsound unless the two types have the same size, alignment and field
        // offsets; these const assertions are the whole proof, and they run at compile time.
        const _: () = {
            assert!(core::mem::size_of::<$scalar>() == core::mem::size_of::<$entity>());
            assert!(core::mem::align_of::<$scalar>() == core::mem::align_of::<$entity>());
            assert!(core::mem::offset_of!($scalar, re) == core::mem::offset_of!($entity, re));
            assert!(core::mem::offset_of!($scalar, im) == core::mem::offset_of!($entity, im));
        };

        impl FaerScalar for $scalar {}

        impl ScalarEntity for $scalar {
            type Entity = $entity;

            fn entity_slice(data: &[Self]) -> &[Self::Entity] {
                // SAFETY: the const assertions above pin size, alignment and field offsets, so both
                // types represent one complex scalar over the same real type with the same layout.
                unsafe { core::slice::from_raw_parts(data.as_ptr().cast::<$entity>(), data.len()) }
            }

            fn entity_slice_mut(data: &mut [Self]) -> &mut [Self::Entity] {
                // SAFETY: as above; the mutable receiver guarantees exclusive access.
                unsafe {
                    core::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<$entity>(), data.len())
                }
            }

            fn parity(odd: bool) -> Self {
                Self::new(if odd { -1.0 } else { 1.0 }, 0.0)
            }
        }
    };
}

impl_real_scalar!(f32);
impl_real_scalar!(f64);
impl_complex_scalar!(Complex32, faer::c32);
impl_complex_scalar!(Complex64, faer::c64);
