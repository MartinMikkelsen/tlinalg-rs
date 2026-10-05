//! Scalar types a faer-backed kernel runs on.
//!
//! The public bound is deliberately small: a kernel only needs a scalar that this crate has an
//! implementation for. The faer entity and the layout-preserving casts are crate-internal, so
//! callers never depend on how a scalar is presented to faer.
//!
//! The complex casts are moved from `tenferro-linalg`'s faer backend (same project, MIT OR
//! Apache-2.0), where they reinterpret `num_complex::Complex` as the layout-identical faer scalar.

use num_complex::{Complex32, Complex64};

/// A [`crate::Scalar`] this crate implements kernels for.
///
/// Sealed through `ScalarEntity`, which lives in this private module: implementors outside this
/// crate cannot name the faer entity, so the cast surface stays internal.
pub trait FaerScalar: crate::Scalar + Default + PartialEq + ScalarEntity {}

/// The faer-facing half of a [`FaerScalar`], crate-internal despite being `pub` in this private
/// module.
pub trait ScalarEntity: crate::Scalar + Default + PartialEq {
    /// The faer scalar sharing this type's memory layout.
    type Entity: faer::traits::ComplexField + Copy + PartialEq + Default;

    /// The real scalar this type's singular values and eigenvalues live in.
    ///
    /// A host allocates those outputs, so this is a [`crate::Scalar`] too.
    type Real: crate::Scalar + Default + PartialEq;

    /// Reinterpret a slice as the faer entity type.
    fn entity_slice(data: &[Self]) -> &[Self::Entity];
    /// Reinterpret a mutable slice as the faer entity type.
    fn entity_slice_mut(data: &mut [Self]) -> &mut [Self::Entity];
    /// The permutation parity scalar: `+1`, or `-1` when `odd`.
    fn parity(odd: bool) -> Self;

    /// This scalar from its faer entity; the inverse of [`ScalarEntity::entity_slice`].
    fn from_entity(entity: Self::Entity) -> Self;

    /// This scalar from the conjugate of its faer entity.
    ///
    /// Transposing a factor into its adjoint needs this: the complex scalars conjugate, the real
    /// ones are already their own conjugate.
    fn from_entity_conj(entity: Self::Entity) -> Self;

    /// The real part of a scalar that is known to be real.
    ///
    /// A decomposition returns real singular values through the entity type, so this is the identity
    /// for the real scalars and the real part for the complex ones.
    fn real_from_entity(entity: Self::Entity) -> Self::Real;
}

macro_rules! impl_real_scalar {
    ($scalar:ty) => {
        impl FaerScalar for $scalar {}

        impl ScalarEntity for $scalar {
            type Entity = $scalar;
            type Real = $scalar;

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

            fn from_entity(entity: Self::Entity) -> Self {
                entity
            }

            fn from_entity_conj(entity: Self::Entity) -> Self {
                entity
            }

            fn real_from_entity(entity: Self::Entity) -> Self::Real {
                entity
            }
        }
    };
}

macro_rules! impl_complex_scalar {
    ($scalar:ty, $entity:ty, $real:ty) => {
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
            type Real = $real;

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

            fn from_entity(entity: Self::Entity) -> Self {
                Self::new(entity.re, entity.im)
            }

            fn from_entity_conj(entity: Self::Entity) -> Self {
                Self::new(entity.re, -entity.im)
            }

            fn real_from_entity(entity: Self::Entity) -> Self::Real {
                entity.re
            }
        }
    };
}

impl_real_scalar!(f32);
impl_real_scalar!(f64);
impl_complex_scalar!(Complex32, faer::c32, f32);
impl_complex_scalar!(Complex64, faer::c64, f64);
