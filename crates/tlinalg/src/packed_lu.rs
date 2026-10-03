//! Batched packed partial-pivot LU: factor, prepared solve, and fused factor+solve.
//!
//! Moved from `tenferro-linalg`'s faer backend (same project, MIT OR Apache-2.0). The kernels keep
//! the LAPACK packed format — unit-lower `L` below the diagonal, `U` on and above it, one-based
//! row-swap pivots — so the factors stay interchangeable with a LAPACK implementation.
//!
//! # Host contract
//!
//! Every entry point operates on one **chunk** of whole matrices: `lu` holds
//! `matrices * m * n` elements, and the same whole-matrix count applies to the pivots, the parity
//! and any right-hand side. The host owns the batch split and the fan-out: it partitions the batch
//! with `chunk = batch.div_ceil(lanes)` and calls these functions once per chunk with
//! [`tlinalg_traits::Parallel::Sequential`], or once with the resolved item policy when it decided
//! on a single lane. Nothing here re-derives lanes from a thread count, and nothing here validates
//! the whole batch — each call validates only the buffer lengths it receives.
//!
//! Scratch is native and per chunk: four permutation vectors and one faer `MemBuffer`, reused
//! across every matrix of the chunk. Nothing in this family takes pooled buffers, so it needs no
//! [`tlinalg_traits::Workspace`].

use core::marker::PhantomData;

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::prelude::ReborrowMut;
use faer::{Conj, MatMut, MatRef};

use tlinalg_traits::{Error, Op, Parallel, Result};

use crate::{faer_par, with_parallel, FaerScalar};

/// A caller-supplied argument was invalid.
fn invalid(op: Op, role: &'static str, detail: impl Into<String>) -> Error {
    Error::InvalidArgument {
        op,
        role,
        detail: detail.into(),
    }
}

/// A shape product overflowed `usize`.
fn checked_product(op: Op, role: &'static str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| {
            invalid(
                op,
                "configuration",
                format!("{role} element count overflows usize"),
            )
        })
}

/// `A` is exactly singular and the caller asked for a solve.
fn singular(op: Op) -> Error {
    Error::Singular { op }
}

/// Reusable per-chunk state for factoring a run of `m x n` matrices.
///
/// Typed by the scalar it was sized for, so mixing a scratch with another scalar is a compile
/// error. The shape is not part of the type: [`FactorScratch::new`] records `m` and `n` and every
/// entry point rejects a call whose shape disagrees, before touching the buffers.
pub struct FactorScratch<T: FaerScalar> {
    m: usize,
    n: usize,
    k: usize,
    perm: Vec<usize>,
    perm_inv: Vec<usize>,
    current: Vec<usize>,
    position: Vec<usize>,
    mem: MemBuffer,
    scalar: PhantomData<T>,
}

impl<T: FaerScalar> FactorScratch<T> {
    /// Size the scratch for `m x n` matrices and the given parallelism.
    ///
    /// The parallelism participates because faer sizes its scratch per thread.
    #[must_use]
    pub fn new(m: usize, n: usize, par: Parallel<'_>) -> Self {
        let par = faer_par(par);
        Self {
            m,
            n,
            k: m.min(n),
            perm: vec![0; m],
            perm_inv: vec![0; m],
            current: vec![0; m],
            position: vec![0; m],
            mem: MemBuffer::new(
                faer::linalg::lu::partial_pivoting::factor::lu_in_place_scratch::<usize, T::Entity>(
                    m,
                    n,
                    par,
                    Default::default(),
                ),
            ),
            scalar: PhantomData,
        }
    }

    /// Reject a call whose shape disagrees with the shape this scratch was sized for.
    ///
    /// The scratch carries the permutation vectors and the faer buffer for exactly one shape, so a
    /// mismatch has to be refused instead of producing an out-of-range index inside faer.
    fn check_shape(&self, op: Op, m: usize, n: usize) -> Result<()> {
        if (self.m, self.n) == (m, n) {
            return Ok(());
        }
        Err(invalid(
            op,
            "configuration",
            format!(
                "scratch was sized for {}x{} but the call uses {m}x{n}",
                self.m, self.n
            ),
        ))
    }

    /// Factor one compact column-major matrix in place and write its one-based swap sequence into
    /// `ipiv`. Returns whether the permutation is odd.
    fn factor(
        &mut self,
        par: faer::Par,
        matrix: MatMut<'_, T::Entity>,
        ipiv: &mut [i32],
        op: Op,
    ) -> Result<bool> {
        let stack = MemStack::new(&mut self.mem);
        let info = faer::linalg::lu::partial_pivoting::factor::lu_in_place(
            matrix,
            &mut self.perm,
            &mut self.perm_inv,
            par,
            stack,
            Default::default(),
        )
        .0;

        // faer returns `perm` with `(P A)[i, :] = A[perm[i], :]`. Replay it as the LAPACK swap
        // sequence: at step `i`, swap row `i` with the current position of row `perm[i]`.
        // `current`/`position` track the running permutation and its inverse, so each step is O(1).
        for (idx, (slot, pos)) in self
            .current
            .iter_mut()
            .zip(self.position.iter_mut())
            .enumerate()
        {
            *slot = idx;
            *pos = idx;
        }
        // INVARIANT: `check_shape` fixed this scratch to the call's `(m, n)`, so `self.k ==
        // m.min(n)`. `perm` is a permutation of `0..m` (faer contract) and the caller passed
        // `ipiv.len() == k <= m`, so every index below is in bounds.
        for (step, slot) in ipiv.iter_mut().enumerate().take(self.k) {
            let wanted = self.perm[step];
            let pivot = self.position[wanted];
            if pivot >= self.m {
                return Err(invalid(op, "configuration", "invalid row permutation"));
            }
            let displaced = self.current[step];
            self.current.swap(step, pivot);
            self.position[wanted] = step;
            self.position[displaced] = pivot;
            *slot = i32::try_from(pivot + 1)
                .map_err(|_| invalid(op, "configuration", "pivot index exceeds i32 range"))?;
        }
        Ok(info.transposition_count % 2 == 1)
    }
}

/// Check that every `(per_matrix_len, buffer_len)` pair describes the same number of matrices.
fn check_batches(op: Op, buffers: [(usize, usize); 3], batch: usize) -> Result<()> {
    for ((per_matrix, len), what) in buffers
        .into_iter()
        .zip(["packed LU", "pivots", "rhs batch"])
    {
        if len != checked_product(op, what, &[per_matrix, batch])? {
            return Err(Error::Inconsistent {
                op,
                detail: "packed LU, pivot, and batch buffers describe different batches",
            });
        }
    }
    Ok(())
}

/// Factor every compact column-major `m x n` matrix of `lu` in place.
///
/// `pivots` receives `min(m, n)` one-based pivots per matrix and `parity` one permutation parity
/// per matrix. Exactly singular matrices are **not** an error, matching LAPACK `?getrf` with
/// positive `info`.
///
/// `scratch` must have been built by [`FactorScratch::new`] for the same `m`, `n` and scalar.
///
/// # Errors
///
/// Returns [`Error::Inconsistent`] when the buffers describe different batches,
/// [`Error::InvalidArgument`] when the scratch shape disagrees, and [`Error::InvalidArgument`] when
/// faer returns an invalid row permutation.
// INVARIANT: the argument list mirrors the host call site, which splits a batch into three
// separate chunk iterators plus an already-resolved policy. Grouping them into a struct would add a
// wrapper per chunk without removing any argument (the host holds the three slices separately).
#[allow(clippy::too_many_arguments)]
pub fn factor_chunk<T: FaerScalar>(
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
    par: Parallel<'_>,
    scratch: &mut FactorScratch<T>,
) -> Result<()> {
    let k = m.min(n);
    let matrix_len = checked_product(op, "matrix shape", &[m, n])?;
    let batch = parity.len();
    check_batches(
        op,
        [(matrix_len, lu.len()), (k, pivots.len()), (1, parity.len())],
        batch,
    )?;
    if matrix_len == 0 || batch == 0 {
        return Ok(());
    }
    scratch.check_shape(op, m, n)?;
    with_parallel(par, |par| {
        // INVARIANT: lengths were checked above and `matrix_len > 0` implies `k > 0`, so the three
        // chunk iterators yield exactly `batch` aligned items. faer owns any threading inside
        // `lu_in_place`, so this loop stays serial and reuses one scratch set.
        for ((matrix, ipiv), parity) in lu
            .chunks_exact_mut(matrix_len)
            .zip(pivots.chunks_exact_mut(k))
            .zip(parity.iter_mut())
        {
            let mat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(matrix), m, n);
            let odd = scratch.factor(par, mat, ipiv, op)?;
            *parity = T::parity(odd);
        }
        Ok(())
    })
}

/// Apply a one-based LAPACK swap sequence to the rows of a compact column-major `n x nrhs` block,
/// forward (`P b`) or in reverse (`P^T b`).
fn apply_row_swaps<T: Copy>(rhs: &mut [T], n: usize, nrhs: usize, ipiv: &[i32], reverse: bool) {
    let swap = |rhs: &mut [T], step: usize, pivot_one_based: i32| {
        // INVARIANT: callers validated every pivot in `1..=n` first.
        let pivot = pivot_one_based as usize - 1;
        if pivot != step {
            for col in 0..nrhs {
                rhs.swap(step + col * n, pivot + col * n);
            }
        }
    };
    if reverse {
        for (step, &pivot) in ipiv.iter().enumerate().rev() {
            swap(rhs, step, pivot);
        }
    } else {
        for (step, &pivot) in ipiv.iter().enumerate() {
            swap(rhs, step, pivot);
        }
    }
}

fn validate_pivots(op: Op, n: usize, ipiv: &[i32]) -> Result<()> {
    for &pivot_one_based in ipiv {
        let in_range = usize::try_from(pivot_one_based)
            .map(|pivot| (1..=n).contains(&pivot))
            .unwrap_or(false);
        if !in_range {
            return Err(invalid(
                op,
                "pivot",
                format!("LU pivot index {pivot_one_based} is outside 1..={n}"),
            ));
        }
    }
    Ok(())
}

/// Solve `op(A) x = b` for one matrix from packed factors, in place.
///
/// With `P A = L U`: `A x = b` is `x = U^-1 L^-1 P b`, and `A^T x = b` is `x = P^T L^-T U^-T b`.
/// Conjugation conjugates `L` and `U` implicitly.
// INVARIANT: the flags mirror the host's solve attributes one-to-one.
fn solve_one<T: FaerScalar>(
    par: faer::Par,
    (n, nrhs): (usize, usize),
    matrix: &[T],
    ipiv: &[i32],
    rhs: &mut [T],
    (transpose_a, conjugate_a): (bool, bool),
) {
    let conj = if conjugate_a { Conj::Yes } else { Conj::No };
    let lu = MatRef::from_column_major_slice(T::entity_slice(matrix), n, n);
    if transpose_a {
        {
            let mut x = MatMut::from_column_major_slice_mut(T::entity_slice_mut(rhs), n, nrhs);
            let lu_t = lu.transpose();
            faer::linalg::triangular_solve::solve_lower_triangular_in_place_with_conj(
                lu_t,
                conj,
                x.rb_mut(),
                par,
            );
            faer::linalg::triangular_solve::solve_unit_upper_triangular_in_place_with_conj(
                lu_t, conj, x, par,
            );
        }
        apply_row_swaps(rhs, n, nrhs, ipiv, true);
    } else {
        apply_row_swaps(rhs, n, nrhs, ipiv, false);
        let mut x = MatMut::from_column_major_slice_mut(T::entity_slice_mut(rhs), n, nrhs);
        faer::linalg::triangular_solve::solve_unit_lower_triangular_in_place_with_conj(
            lu,
            conj,
            x.rb_mut(),
            par,
        );
        faer::linalg::triangular_solve::solve_upper_triangular_in_place_with_conj(lu, conj, x, par);
    }
}

/// Solve `op(A) X = B` for every matrix of the chunk from packed partial-pivot factors.
///
/// `output` enters holding the compact column-major RHS batch and leaves holding the solution. The
/// factors must be nonsingular.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for a pivot outside `1..=n` and [`Error::Inconsistent`] when
/// the buffers describe different batches. Pivots are validated for the whole chunk **before** any
/// output is written.
// INVARIANT: the argument list mirrors the host call site (see `factor_chunk`).
#[allow(clippy::too_many_arguments)]
pub fn solve_prepared_chunk<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
    par: Parallel<'_>,
) -> Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 || rhs_len == 0 {
        return Ok(());
    }
    let batch = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch,
    )?;
    validate_pivots(op, n, pivots)?;
    with_parallel(par, |par| {
        // INVARIANT: lengths were checked above, so the chunk iterators yield exactly `batch`
        // aligned nonempty items, and all pivots are in range for `apply_row_swaps`.
        for ((matrix, ipiv), rhs) in packed_lu
            .chunks_exact(matrix_len)
            .zip(pivots.chunks_exact(n))
            .zip(output.chunks_exact_mut(rhs_len))
        {
            solve_one::<T>(
                par,
                (n, nrhs),
                matrix,
                ipiv,
                rhs,
                (transpose_a, conjugate_a),
            );
        }
        Ok(())
    })
}

/// Factor and solve `A X = B` for every matrix of the chunk, keeping the packed factors.
///
/// `packed_lu` enters holding the compact `A` batch and leaves holding the packed factors;
/// `pivots` receives one-based pivots; `output` enters holding the RHS batch and leaves holding `X`.
///
/// # Errors
///
/// Returns [`Error::Singular`] when a factor has an exactly zero `U` diagonal and there is a
/// nonempty RHS to solve, [`Error::Inconsistent`] when the buffers describe different batches, and
/// [`Error::InvalidArgument`] when the scratch shape disagrees. A zero-column RHS only factors,
/// matching [`factor_chunk`] on singular input.
// INVARIANT: the argument list mirrors the host call site (see `factor_chunk`).
#[allow(clippy::too_many_arguments)]
pub fn factor_solve_chunk<T: FaerScalar>(
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
    par: Parallel<'_>,
    scratch: &mut FactorScratch<T>,
) -> Result<()> {
    let matrix_len = checked_product(op, "matrix", &[n, n])?;
    let rhs_len = checked_product(op, "rhs", &[n, nrhs])?;
    if matrix_len == 0 {
        return Ok(());
    }
    let batch = packed_lu.len() / matrix_len;
    check_batches(
        op,
        [
            (matrix_len, packed_lu.len()),
            (n, pivots.len()),
            (rhs_len, output.len()),
        ],
        batch,
    )?;
    scratch.check_shape(op, n, n)?;
    let zero = T::default();
    with_parallel(par, |par| {
        // INVARIANT: lengths were checked above; a zero-column RHS yields empty RHS blocks, which
        // `chunks_mut` cannot express with a zero chunk size, so the RHS block is sliced by offset
        // instead. faer owns any threading inside the factorization and the triangular solves.
        for (index, (matrix, ipiv)) in packed_lu
            .chunks_exact_mut(matrix_len)
            .zip(pivots.chunks_exact_mut(n))
            .enumerate()
        {
            {
                let mat = MatMut::from_column_major_slice_mut(T::entity_slice_mut(matrix), n, n);
                scratch.factor(par, mat, ipiv, op)?;
            }
            if rhs_len > 0 {
                if (0..n).any(|i| matrix[i + i * n] == zero) {
                    return Err(singular(op));
                }
                let start = index * rhs_len;
                let rhs = &mut output[start..start + rhs_len];
                solve_one::<T>(par, (n, nrhs), matrix, ipiv, rhs, (false, false));
            }
        }
        Ok(())
    })
}
