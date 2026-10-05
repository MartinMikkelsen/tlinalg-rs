//! The shared batch driver: strided batch iteration, lane fan-out, and output assembly.
//!
//! Every public entry point of this crate is batched (see `docs/design/batched-api.md`). This module
//! is the one place that
//!
//! * walks the batch dimensions of a rank-`2 + B` strided descriptor ([`BatchedRef`],
//!   [`BatchedMut`]),
//! * splits the batch into the host-chosen lanes and runs them on the caller's pool ([`run`]),
//! * assembles the compact, batch-contiguous outputs without a zero-fill pass ([`Out`]) or hands
//!   out disjoint chunks of a caller-owned compact buffer ([`InPlace`]).
//!
//! # Output soundness
//!
//! A [`Out`] output is cleared, reserved, and its spare capacity is split into one disjoint
//! `MaybeUninit` chunk per lane. Each lane writes its chunk strictly sequentially through a [`Sink`],
//! so "the sink reached the end of its chunk" is exactly "every element of the chunk was written".
//! The driver checks that for every lane and calls `set_len` only when **every** lane succeeded and
//! filled its chunk; otherwise the vector stays at length zero, so a caller never observes a
//! partially initialised vector.

use core::mem::MaybeUninit;

use faer::{MatMut, MatRef};
use strided_view::{RawStridedMut, RawStridedRef};

use crate::scalar::ScalarEntity;
use crate::util::{checked_product, invalid};
use crate::{with_parallel, Error, LanePlan, Op, Parallel, Result};

/// The batch count of a descriptor's trailing dimensions.
fn batch_count(op: Op, batch_dims: &[usize]) -> Result<usize> {
    checked_product(op, "batch", batch_dims)
}

/// Batch axes after normalisation: extent-one axes dropped and mergeable neighbours coalesced.
///
/// Adjacent axes `i`, `i + 1` (first batch axis fastest) merge when `stride[i + 1] == stride[i] *
/// dim[i]`, so a compact batch — the common case — collapses to one axis and an item's offset is a
/// single multiply. Only a genuinely non-mergeable view keeps several axes. A stride-0 axis (host
/// broadcasting) is kept like any other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BatchAxes {
    axes: Vec<(usize, isize)>,
}

impl BatchAxes {
    pub(crate) fn new(dims: &[usize], strides: &[isize]) -> Self {
        let mut axes: Vec<(usize, isize)> = Vec::with_capacity(dims.len());
        for (&dim, &stride) in dims.iter().zip(strides) {
            if dim == 1 {
                continue;
            }
            match axes.last_mut() {
                Some((last_dim, last_stride))
                    if last_stride.checked_mul(*last_dim as isize) == Some(stride) =>
                {
                    *last_dim *= dim;
                }
                _ => axes.push((dim, stride)),
            }
        }
        Self { axes }
    }

    /// Element offset of batch item `index`.
    ///
    /// One axis (the compact case) is a single multiply. Several axes — a non-mergeable view —
    /// decompose `index` once per item; that is `O(B)` integer work against an `O(n³)` kernel, and
    /// it keeps each item independent of the order lanes visit them.
    pub(crate) fn offset(&self, index: usize) -> isize {
        match self.axes.as_slice() {
            [] => 0,
            [(_, stride)] => index as isize * stride,
            axes => {
                let mut rest = index;
                let mut offset = 0isize;
                for &(dim, stride) in axes {
                    offset += (rest % dim) as isize * stride;
                    rest /= dim;
                }
                offset
            }
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.axes.len()
    }
}

/// A borrowed rank-`2 + B` input: `[rows, cols, b_1, ..., b_B]`.
#[derive(Clone)]
pub(crate) struct BatchedRef<'a, T> {
    input: RawStridedRef<'a, T>,
    axes: BatchAxes,
    rows: usize,
    cols: usize,
    batch: usize,
}

impl<'a, T: ScalarEntity> BatchedRef<'a, T> {
    /// Validate the rank of `input` and read its matrix and batch shape.
    pub(crate) fn new(op: Op, role: &'static str, input: RawStridedRef<'a, T>) -> Result<Self> {
        let dims = input.dims();
        if dims.len() < 2 {
            return Err(invalid(
                op,
                "configuration",
                format!("{role} has rank {}, expected at least 2", dims.len()),
            ));
        }
        Ok(Self {
            rows: dims[0],
            cols: dims[1],
            batch: batch_count(op, &dims[2..])?,
            axes: BatchAxes::new(&dims[2..], &input.strides()[2..]),
            input,
        })
    }

    /// Validate as [`BatchedRef::new`] and require a square matrix shape.
    pub(crate) fn square(op: Op, role: &'static str, input: RawStridedRef<'a, T>) -> Result<Self> {
        let this = Self::new(op, role, input)?;
        if this.rows != this.cols {
            return Err(invalid(
                op,
                "configuration",
                format!(
                    "{role} is {}x{}, expected a square matrix",
                    this.rows, this.cols
                ),
            ));
        }
        Ok(this)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    pub(crate) fn batch(&self) -> usize {
        self.batch
    }

    pub(crate) fn batch_dims(&self) -> &'a [usize] {
        &self.input.dims()[2..]
    }

    /// Batch item `index` as a faer matrix over the scalar's faer entity.
    ///
    /// `index` must be below [`BatchedRef::batch`].
    pub(crate) fn item(&self, index: usize) -> MatRef<'a, T::Entity> {
        debug_assert!(index < self.batch);
        let strides = self.input.strides();
        let offset = self.axes.offset(index);
        // SAFETY: `RawStridedRef::new` validated that every offset reachable from `ptr()` through
        // `dims`/`strides` lies inside the borrowed data, for the borrow `'a`. Batch item `index`
        // (below the batch count) is the sub-rectangle at the batch offset computed above, so every
        // element faer reads is one of those validated offsets. `wrapping_offset` keeps the
        // pointer's provenance; for an empty matrix `ptr()` is dangling and aligned, and faer never
        // dereferences a matrix with a zero dimension. The cast is the layout-preserving one
        // asserted in `crate::scalar`.
        unsafe {
            MatRef::from_raw_parts(
                self.input.ptr().wrapping_offset(offset).cast::<T::Entity>(),
                self.rows,
                self.cols,
                strides[0],
                strides[1],
            )
        }
    }
}

/// Reject a second operand whose batch shape differs from the first's.
pub(crate) fn same_batch(
    op: Op,
    role: &'static str,
    expected: &[usize],
    got: &[usize],
) -> Result<()> {
    if expected != got {
        return Err(invalid(
            op,
            "configuration",
            format!("{role} has batch shape {got:?}, expected {expected:?}"),
        ));
    }
    Ok(())
}

/// Whether a strided layout maps distinct indices to distinct offsets.
///
/// Sufficient test: after dropping extent-one dimensions and sorting by absolute stride, each stride
/// exceeds the span of every smaller dimension. A layout that fails it may still be injective, but
/// the hosts this crate serves never build one, so rejecting it costs nothing.
fn is_injective(dims: &[usize], strides: &[isize]) -> bool {
    if dims.contains(&0) {
        return true;
    }
    let mut axes: Vec<(usize, usize)> = dims
        .iter()
        .zip(strides)
        .filter(|(&dim, _)| dim > 1)
        .map(|(&dim, &stride)| (stride.unsigned_abs(), dim))
        .collect();
    axes.sort_unstable();
    let mut span = 0usize;
    for (stride, dim) in axes {
        if stride <= span {
            return false;
        }
        span = match (dim - 1)
            .checked_mul(stride)
            .and_then(|s| s.checked_add(span))
        {
            Some(span) => span,
            None => return false,
        };
    }
    true
}

/// A raw pointer that lanes may share because they touch disjoint elements through it.
#[derive(Clone, Copy)]
struct SharedPtr<T>(*mut T);

// SAFETY: the pointer is only dereferenced through `BatchedMut::item`, whose contract gives each
// batch item to one lane at a time, and an injective layout makes distinct items disjoint.
unsafe impl<T: Send> Send for SharedPtr<T> {}
// SAFETY: as above.
unsafe impl<T: Send> Sync for SharedPtr<T> {}

/// A borrowed, mutable rank-`2 + B` destination with an injective layout.
pub(crate) struct BatchedMut<'a, T> {
    base: SharedPtr<T>,
    dims: &'a [usize],
    strides: &'a [isize],
    axes: BatchAxes,
    batch: usize,
}

impl<'a, T: ScalarEntity> BatchedMut<'a, T> {
    /// Validate the rank and injectivity of `output` and take exclusive access to it.
    pub(crate) fn new(
        op: Op,
        role: &'static str,
        output: &'a mut RawStridedMut<'_, T>,
    ) -> Result<Self> {
        let dims = output.dims();
        let strides = output.strides();
        if dims.len() < 2 {
            return Err(invalid(
                op,
                "configuration",
                format!("{role} has rank {}, expected at least 2", dims.len()),
            ));
        }
        if !is_injective(dims, strides) {
            return Err(invalid(
                op,
                "configuration",
                format!("{role} layout aliases two elements"),
            ));
        }
        Ok(Self {
            batch: batch_count(op, &dims[2..])?,
            base: SharedPtr(output.as_mut_ptr()),
            axes: BatchAxes::new(&dims[2..], &strides[2..]),
            dims,
            strides,
        })
    }

    pub(crate) fn rows(&self) -> usize {
        self.dims[0]
    }

    pub(crate) fn cols(&self) -> usize {
        self.dims[1]
    }

    pub(crate) fn batch_dims(&self) -> &'a [usize] {
        &self.dims[2..]
    }

    /// Batch item `index` as a mutable faer matrix.
    ///
    /// # Safety
    ///
    /// `index` is below the batch count, and no other live reference (from this or any other call)
    /// covers item `index` for as long as the returned matrix is used.
    pub(crate) unsafe fn item(&self, index: usize) -> MatMut<'_, T::Entity> {
        debug_assert!(index < self.batch);
        let offset = self.axes.offset(index);
        // SAFETY: `RawStridedMut::new` validated every reachable offset against the exclusively
        // borrowed data; item `index` is a sub-rectangle of those offsets, and the injective layout
        // (checked in `new`) makes it disjoint from every other item. The caller guarantees no other
        // reference covers it. Empty matrices are never dereferenced; see `BatchedRef::item`.
        unsafe {
            MatMut::from_raw_parts_mut(
                self.base.0.wrapping_offset(offset).cast::<T::Entity>(),
                self.dims[0],
                self.dims[1],
                self.strides[0],
                self.strides[1],
            )
        }
    }
}

/// Sequential writer over one lane's uninitialised output chunk.
pub(crate) struct Sink<'a, U> {
    buf: &'a mut [MaybeUninit<U>],
    pos: usize,
}

impl<U> Sink<'_, U> {
    /// Write the next element. Panics (safely) if the kernel writes past its chunk.
    pub(crate) fn push(&mut self, value: U) {
        self.buf[self.pos].write(value);
        self.pos += 1;
    }

    fn is_full(&self) -> bool {
        self.pos == self.buf.len()
    }
}

/// Something a per-item kernel can append output elements to: a `Vec` or a lane [`Sink`].
///
/// `pub` only so the crate-internal `EigScalar` can name it; this module is private.
pub trait Push<U> {
    /// Append one element.
    fn push(&mut self, value: U);
}

impl<U> Push<U> for Vec<U> {
    fn push(&mut self, value: U) {
        Vec::push(self, value);
    }
}

impl<U> Push<U> for Sink<'_, U> {
    fn push(&mut self, value: U) {
        Sink::push(self, value);
    }
}

/// One lane's chunk of a caller-owned compact buffer.
pub(crate) struct Chunk<'a, U> {
    slice: &'a mut [U],
    item_len: usize,
    first: usize,
}

impl<U> Chunk<'_, U> {
    /// The elements of batch item `index` (a global index inside this lane's range).
    pub(crate) fn item(&mut self, index: usize) -> &mut [U] {
        let start = (index - self.first) * self.item_len;
        &mut self.slice[start..start + self.item_len]
    }
}

/// Split `len` elements starting at `ptr` into `lanes` consecutive pieces of `piece` elements (the
/// last may be shorter), as slices of lifetime `'a`.
///
/// # Safety
///
/// `ptr..ptr + len` is valid for writes (and reads, once initialised) for `'a`, and nothing else
/// accesses it while any returned slice is alive.
unsafe fn split_raw<'a, X>(
    ptr: *mut X,
    len: usize,
    piece: usize,
    lanes: usize,
) -> Vec<&'a mut [X]> {
    (0..lanes)
        .map(|lane| {
            let start = (lane * piece).min(len);
            let end = (start + piece).min(len);
            // SAFETY: `start..end` lies in `0..len` and distinct lanes get disjoint ranges; the
            // caller guarantees validity and exclusivity for `'a`.
            unsafe { core::slice::from_raw_parts_mut(ptr.add(start), end - start) }
        })
        .collect()
}

/// The outputs of one batched call, split per lane.
pub(crate) trait Outputs {
    /// One lane's view of every output.
    type Lane: Send;

    /// Clear and reserve the vector outputs, check the in-place ones.
    fn prepare(&mut self, op: Op, batch: usize) -> Result<()>;

    /// Split into one lane view per `chunk` items (`batch.div_ceil(chunk)` lanes).
    ///
    /// # Safety
    ///
    /// [`Outputs::prepare`] succeeded for this `batch`, and every returned lane is dropped before
    /// `self` is used again (the lanes alias its buffers).
    unsafe fn split(&mut self, batch: usize, chunk: usize) -> Vec<Self::Lane>;

    /// Whether a lane wrote every element it owns.
    fn lane_complete(lane: &Self::Lane) -> bool;

    /// Publish the written vector outputs.
    ///
    /// # Safety
    ///
    /// Every lane returned by [`Outputs::split`] for this `batch` completed (see
    /// [`Outputs::lane_complete`]) and has been dropped.
    unsafe fn commit(&mut self, batch: usize);
}

/// A vector output, cleared and then filled with `item_len` elements per batch item.
pub(crate) struct Out<'a, U> {
    vec: &'a mut Vec<U>,
    item_len: usize,
}

/// A vector output of `item_len` elements per batch item.
pub(crate) fn out<U>(vec: &mut Vec<U>, item_len: usize) -> Out<'_, U> {
    Out { vec, item_len }
}

impl<'a, U: Send> Outputs for Out<'a, U> {
    type Lane = Sink<'a, U>;

    fn prepare(&mut self, op: Op, batch: usize) -> Result<()> {
        self.vec.clear();
        let total = checked_product(op, "output", &[self.item_len, batch])?;
        self.vec.reserve(total);
        Ok(())
    }

    unsafe fn split(&mut self, batch: usize, chunk: usize) -> Vec<Sink<'a, U>> {
        let total = self.item_len * batch;
        let ptr = self.vec.spare_capacity_mut().as_mut_ptr();
        // SAFETY: `prepare` cleared the vector and reserved `total`, so its spare capacity starts at
        // `ptr` and covers `total` elements; the caller keeps the vector untouched while the lanes
        // live.
        unsafe { split_raw(ptr, total, chunk * self.item_len, batch.div_ceil(chunk)) }
            .into_iter()
            .map(|buf| Sink { buf, pos: 0 })
            .collect()
    }

    fn lane_complete(lane: &Sink<'a, U>) -> bool {
        lane.is_full()
    }

    unsafe fn commit(&mut self, batch: usize) {
        // SAFETY: `prepare` reserved `item_len * batch`, `split` handed the first `item_len * batch`
        // spare elements out as disjoint lane chunks covering that range exactly, and the caller
        // guarantees every lane wrote its whole chunk (each `Sink` writes sequentially, so a full
        // sink means every element was initialised) and is gone.
        unsafe { self.vec.set_len(self.item_len * batch) };
    }
}

/// A caller-owned compact buffer of `item_len` elements per batch item, updated in place.
pub(crate) struct InPlace<'a, U> {
    slice: &'a mut [U],
    item_len: usize,
    role: &'static str,
}

/// A compact in-place buffer of `item_len` elements per batch item.
pub(crate) fn in_place<'a, U>(
    slice: &'a mut [U],
    item_len: usize,
    role: &'static str,
) -> InPlace<'a, U> {
    InPlace {
        slice,
        item_len,
        role,
    }
}

impl<'a, U: Send> Outputs for InPlace<'a, U> {
    type Lane = Chunk<'a, U>;

    fn prepare(&mut self, op: Op, batch: usize) -> Result<()> {
        if self.slice.len() != checked_product(op, self.role, &[self.item_len, batch])? {
            return Err(Error::Inconsistent {
                op,
                detail: "batch buffers describe different batches",
            });
        }
        Ok(())
    }

    unsafe fn split(&mut self, batch: usize, chunk: usize) -> Vec<Chunk<'a, U>> {
        let item_len = self.item_len;
        let len = self.slice.len();
        // SAFETY: the slice is exclusively borrowed for `'a` and `prepare` checked it holds
        // `item_len * batch` elements; the caller keeps it untouched while the lanes live.
        unsafe {
            split_raw(
                self.slice.as_mut_ptr(),
                len,
                chunk * item_len,
                batch.div_ceil(chunk),
            )
        }
        .into_iter()
        .enumerate()
        .map(|(lane, slice)| Chunk {
            slice,
            item_len,
            first: lane * chunk,
        })
        .collect()
    }

    fn lane_complete(_: &Chunk<'a, U>) -> bool {
        true
    }

    unsafe fn commit(&mut self, _: usize) {}
}

/// No outputs beyond what the item closure captures itself (a [`BatchedMut`] destination).
impl Outputs for () {
    type Lane = ();

    fn prepare(&mut self, _: Op, _: usize) -> Result<()> {
        Ok(())
    }

    unsafe fn split(&mut self, batch: usize, chunk: usize) -> Vec<()> {
        vec![(); batch.div_ceil(chunk)]
    }

    fn lane_complete(_: &()) -> bool {
        true
    }

    unsafe fn commit(&mut self, _: usize) {}
}

macro_rules! impl_outputs_tuple {
    ($($name:ident $index:tt),+) => {
        impl<$($name: Outputs),+> Outputs for ($($name,)+) {
            type Lane = ($($name::Lane,)+);

            fn prepare(&mut self, op: Op, batch: usize) -> Result<()> {
                $(self.$index.prepare(op, batch)?;)+
                Ok(())
            }

            #[allow(non_snake_case)]
            unsafe fn split(&mut self, batch: usize, chunk: usize) -> Vec<Self::Lane> {
                // SAFETY: forwarded; the caller's guarantee covers every component.
                $(let mut $name = unsafe { self.$index.split(batch, chunk) }.into_iter();)+
                (0..batch.div_ceil(chunk))
                    .map(|_| ($($name.next().expect("every output splits into the same lanes"),)+))
                    .collect()
            }

            fn lane_complete(lane: &Self::Lane) -> bool {
                true $(&& $name::lane_complete(&lane.$index))+
            }

            unsafe fn commit(&mut self, batch: usize) {
                // SAFETY: forwarded; the caller's guarantee covers every component.
                $(unsafe { self.$index.commit(batch) };)+
            }
        }
    };
}

impl_outputs_tuple!(A 0);
impl_outputs_tuple!(A 0, B 1);
impl_outputs_tuple!(A 0, B 1, C 2);
impl_outputs_tuple!(A 0, B 1, C 2, D 3);
impl_outputs_tuple!(A 0, B 1, C 2, D 3, E 4);

/// Run `item` for every batch item, on the host's lane plan.
///
/// * `plan.lanes <= 1`: one lane, run in order on `plan.item_parallel`.
/// * `plan.lanes > 1`: the batch is split into `chunk = batch.div_ceil(lanes)` contiguous chunks.
///   With [`Parallel::Pool`] the pool is installed and each chunk is a `rayon::scope` task; each
///   lane runs its items with [`Parallel::Sequential`], so a lane never nests a second fan-out
///   (this is the semantics of the host's outer-lane children). With [`Parallel::Sequential`] the
///   chunks run one after another on the calling thread.
///
/// `make_scratch` builds one lane's scratch, reused for every item of that lane.
///
/// Each lane stops at its own first failure and is not stopped by another lane's. The returned
/// error is the lowest-indexed failing lane's, which is the lowest-indexed failing item, so the
/// result does not depend on scheduling. On error every [`Out`] output is left empty.
pub(crate) fn run<O, S, MK, F>(
    op: Op,
    batch: usize,
    par: Parallel<'_>,
    plan: LanePlan<'_>,
    outputs: &mut O,
    make_scratch: MK,
    item: F,
) -> Result<()>
where
    O: Outputs,
    MK: Fn(faer::Par) -> S + Sync,
    F: Fn(usize, &mut O::Lane, &mut S, faer::Par) -> Result<()> + Sync,
{
    outputs.prepare(op, batch)?;
    if batch == 0 {
        return Ok(());
    }
    let lanes = plan.lanes.clamp(1, batch);
    let chunk = batch.div_ceil(lanes);
    let result = {
        // SAFETY: `prepare` succeeded above, and every lane is consumed inside this block, before
        // `outputs` is touched again by `commit`.
        let lane_outputs = unsafe { outputs.split(batch, chunk) };
        let run_lane = |lane: usize, mut out: O::Lane, item_par: Parallel<'_>| -> Result<()> {
            let start = lane * chunk;
            let end = (start + chunk).min(batch);
            with_parallel(item_par, |faer_par| {
                let mut scratch = make_scratch(faer_par);
                for index in start..end {
                    item(index, &mut out, &mut scratch, faer_par)?;
                }
                if O::lane_complete(&out) {
                    Ok(())
                } else {
                    Err(Error::Inconsistent {
                        op,
                        detail: "a kernel did not fill its output",
                    })
                }
            })
        };
        if lane_outputs.len() == 1 {
            let only = lane_outputs.into_iter().next().expect("one lane");
            run_lane(0, only, plan.item_parallel)
        } else {
            match par {
                Parallel::Pool { pool, .. } => {
                    let mut results: Vec<Result<()>> =
                        (0..lane_outputs.len()).map(|_| Ok(())).collect();
                    let run_lane = &run_lane;
                    pool.install(|| {
                        rayon::scope(|scope| {
                            for ((lane, out), slot) in
                                lane_outputs.into_iter().enumerate().zip(results.iter_mut())
                            {
                                scope.spawn(move |_| {
                                    *slot = run_lane(lane, out, Parallel::Sequential);
                                });
                            }
                        });
                    });
                    results.into_iter().collect::<Result<()>>()
                }
                Parallel::Sequential => lane_outputs
                    .into_iter()
                    .enumerate()
                    .try_for_each(|(lane, out)| run_lane(lane, out, Parallel::Sequential)),
            }
        }
    };
    if result.is_ok() {
        // SAFETY: every lane returned `Ok`, a lane returns `Ok` only after `O::lane_complete`
        // confirmed it filled its outputs, and every lane was moved into (and dropped by) its run.
        unsafe { outputs.commit(batch) };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::{is_injective, out, run, BatchAxes};
    use crate::{Error, LanePlan, Op, Parallel};
    use core::num::NonZeroUsize;

    /// Items 2 and 5 fail with distinguishable errors; whatever the scheduling, the error of the
    /// lowest-indexed failing item is returned and the output stays empty.
    #[test]
    fn the_lowest_failing_item_wins_and_outputs_stay_empty() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let par = Parallel::Pool {
            pool: &pool,
            budget: NonZeroUsize::new(3).unwrap(),
        };
        for lanes in [1, 2, 3, 4, 8] {
            for _ in 0..20 {
                let mut values = vec![9usize; 3];
                let plan = LanePlan {
                    lanes,
                    item_parallel: Parallel::Sequential,
                };
                let err = run(
                    Op::Svd,
                    8,
                    par,
                    plan,
                    &mut (out(&mut values, 1),),
                    |_| (),
                    |index, (values,), (), _| {
                        if index == 2 || index == 5 {
                            return Err(Error::Inconsistent {
                                op: Op::Svd,
                                detail: if index == 2 { "two" } else { "five" },
                            });
                        }
                        values.push(index);
                        Ok(())
                    },
                )
                .unwrap_err();
                assert_eq!(
                    err,
                    Error::Inconsistent {
                        op: Op::Svd,
                        detail: "two"
                    },
                    "lanes={lanes}"
                );
                assert!(values.is_empty());
            }
        }
    }

    #[test]
    fn a_kernel_that_writes_too_little_is_an_error() {
        let mut values: Vec<u8> = Vec::new();
        let err = run(
            Op::Svd,
            2,
            Parallel::Sequential,
            LanePlan::sequential(),
            &mut (out(&mut values, 2),),
            |_| (),
            |_, (values,), (), _| {
                values.push(1);
                Ok(())
            },
        );
        assert!(matches!(err, Err(Error::Inconsistent { .. })));
        assert!(values.is_empty());
    }

    #[test]
    fn injectivity() {
        assert!(is_injective(&[2, 3], &[1, 2]));
        assert!(is_injective(&[2, 3, 4], &[1, 4, 12]));
        assert!(!is_injective(&[2, 3], &[1, 1]));
        assert!(!is_injective(&[2, 2, 2], &[1, 2, 0]));
        assert!(is_injective(&[2, 1, 2], &[1, 0, 2]));
        assert!(is_injective(&[0, 2], &[0, 0]));
    }

    #[test]
    fn offsets_walk_the_first_batch_dim_fastest() {
        let axes = BatchAxes::new(&[2, 3], &[10, 100]);
        assert_eq!(axes.len(), 2, "not mergeable");
        assert_eq!(axes.offset(0), 0);
        assert_eq!(axes.offset(1), 10);
        assert_eq!(axes.offset(2), 100);
        assert_eq!(axes.offset(5), 210);
    }

    #[test]
    fn compact_axes_coalesce_and_unit_axes_drop() {
        let axes = BatchAxes::new(&[2, 1, 3, 4], &[9, 77, 18, 54]);
        assert_eq!(axes.len(), 1);
        assert_eq!(axes.offset(23), 23 * 9);
        assert_eq!(BatchAxes::new(&[1, 1], &[5, 6]).len(), 0);
        // Broadcast axis: stride 0 is kept and never merges with a non-zero neighbour.
        let broadcast = BatchAxes::new(&[3, 2], &[0, 4]);
        assert_eq!(broadcast.len(), 2);
        assert_eq!(broadcast.offset(4), 4);
        assert_eq!(BatchAxes::new(&[3, 2], &[0, 0]).offset(5), 0);
    }
}
