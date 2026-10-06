# tlinalg batched API (torch-like) — decision 2026-10-05

The maintainer's intent: the whole tlinalg (and tlinalg-blas) public API is **torch-like batched**.
A host calls each family **once per batch**; the library owns the batch loop (and, for tlinalg, the
batch-direction fan-out). Per-matrix entry points become crate-private helpers.

## Shape of every entry point

* Input: one `RawStridedRef<'_, T>` of rank `2 + B`, dims `[rows, cols, b_1, ..., b_B]`, arbitrary
  (validated, non-negative) strides. Matrix dims first, batch dims after (tenferro's column-major
  convention). `B = 0` is a single matrix. Batch iteration over arbitrary-rank strided batch dims is
  the library's job (shared helper), so the host never packs a batch just to call us.
* Outputs: compact column-major per item, items contiguous in batch order (item `i` occupies
  `[i*item_len, (i+1)*item_len)`). Output `Vec`s are **cleared and then filled**; total length is
  `item_len * batch`. No zero-fill pass: sequential paths push; parallel paths write into
  `spare_capacity_mut()` per lane (disjoint `MaybeUninit` chunks) and `set_len` once after every lane
  succeeded. On error the `Vec`s are left empty (cleared) — never partially initialised.
* In-place / direct-output routes (solve into a host view, compact Householder data, packed LU) take
  a batched `RawStridedMut` / compact `&mut [T]` covering the whole batch, and keep their existing
  "unchanged on failure" guarantees per item where they had them.
* Per-item scalar results (LU parity, RRQR permutation, …) are batch-length outputs too.

## Parallelism (tlinalg only)

* One argument: `par: Parallel<'_>` (Sequential or caller-owned pool + requested budget).
  **tlinalg owns Auto lane selection**, not the host. `LanePlan` is crate-private; there is no
  forced-policy or benchmark escape hatch. See [library-owned-lanes.md](library-owned-lanes.md).
* Effective requested width is `min(budget, pool.current_num_threads())`. Width one executes on
  the calling thread, without installing a pool.
* Ordinary families fan out only when `max(rows, cols) <= 64` and the batch has at least one item
  per effective worker (and at least two workers). RHS/reflector target extents are included.
  Packed LU's three routes retain their existing no-size-cutoff policy.
* Outer tasks are contiguous chunks of `batch.div_ceil(width)` items on the supplied pool, using
  `in_place_scope`; every outer child is sequential. Their count never exceeds effective width.
  Otherwise one lane runs the batch in order, passing effective width to faer as an intra-item hint.
* **faer 0.24.4 does not guarantee a strict active-thread bound from that hint.** Its wide-RHS
  recursion and rounded-up splitting can exceed the requested count. This accepted limitation is
  not hidden by serial fallback or replacement pools; no hard bound on all native work is claimed.
* Scratch is per lane and reused across the items of that lane (faer `MemBuffer` sized once for the
  item shape; host `Workspace` where the family already uses it).
* Errors: each lane stops at its own first failure and does **not** stop because another lane
  failed; the returned error is the one from the lowest-indexed failing lane, i.e. the
  lowest-indexed failing item — deterministic regardless of scheduling.

## tlinalg-blas

Same batched signatures, minus `par`/`plan` (LAPACK threading is vendor-owned): a sequential loop
over the batch, with **one** workspace query + acquisition per batch call outside the loop, reused
for every item (as `svd_batch` already does). Injected-symbol (`provider-inject`) support unchanged.

## Scope

Applies to every family, including the already-extracted SVD and packed LU (packed LU's chunk API
becomes the batched API; its fan-out moves from tenferro into tlinalg).

## Host (tenferro) keeps

Resource ownership/admission, dtype/placement dispatch, tensor construction, QR gauges, RRQR rank
decision, Householder append/from-factors/gemm, negative-stride rejection, error mapping.

## Behaviour change to record

Families other than packed LU previously looped sequentially over the batch in tenferro. Batched
extraction enabled outer fan-out; the single-token revision moves the existing default Auto policy
into tlinalg. Host forced strategies/custom thresholds are removed from this provider's API.
Integrated tenferro migration and its performance measurements remain a separate follow-up.

## Addendum: multiple batch axes (2026-10-05)

* Keep `B >= 0` batch axes. One shared helper normalises them first: drop size-1 axes and
  **coalesce mergeable adjacent axes** (`stride[i+1] == stride[i] * dim[i]`), so the common compact
  case collapses to a single axis with a plain stride step — no per-item multi-index arithmetic.
  Only genuinely non-mergeable views pay for the multi-index walk (an odometer, not div/mod per item).
* Broadcasting: every operand of a call shares the **same batch dims**. The host expresses
  torch-style batch broadcasting by giving an input a **stride of 0** on a batch axis
  (e.g. one `A` against many `B` in solve/triangular_solve/lu_solve). Inputs may have stride-0
  batch axes; the kernel needs no broadcasting logic of its own.
* Mutable descriptors (in-place / direct output) must **not** self-overlap across batch items:
  reject stride 0 on a batch axis of size > 1 and any overlapping item ranges with
  `Error::InvalidArgument`. `RawStridedMut::new` only checks bounds, and parallel lanes writing an
  aliased item would be a data race, so this check is a soundness requirement, not a nicety.
  Overlap between a mutable output and an input is likewise rejected unless the family documents
  in-place semantics.
* Outputs created by the library (`Vec`s) are always compact with one flattened batch extent; the
  host reshapes them to the batch shape it owns.

## Implementation notes (tlinalg)

Recorded with the implementation; they refine, not replace, the contract above.

* **Shared driver.** `crates/tlinalg/src/batch.rs` is the only batch loop: descriptor validation
  and batch-axis normalisation (`BatchedRef`, `BatchedMut`, `BatchAxes`), lane split and fan-out
  (`run`), and output assembly (`Out` for library-created vectors, `InPlace` for host-owned compact
  buffers).
* **Lane parallelism.** The private resolver selects outer lanes only for a pool with effective
  width greater than one. Each child uses `Parallel::Sequential`; a single lane uses the sole call
  token. A sequential resource never resolves to multiple lanes.
* **Item offsets.** After normalisation a compact batch has one axis and an item offset is one
  multiply. A non-mergeable view decomposes the item index per item (`O(B)` integer work next to an
  `O(n³)` kernel) instead of carrying an odometer: items are addressed by index from independent
  lanes, so a per-item computation keeps them order-independent.
* **Destination aliasing.** A mutable descriptor is rejected unless its whole layout (matrix and
  batch axes) is injective under a sufficient test: after dropping extent-one axes and sorting by
  absolute stride, every stride exceeds the span of the smaller axes. Input/output overlap cannot
  be expressed: inputs borrow `&[T]` and destinations `&mut [T]`.
* **No driver allocation.** Normalised batch axes are stored inline (up to 8 axes; heap only
  beyond). Lane views are derived from the lane index on demand, never collected into a list, and a
  multi-lane run records the lowest failing lane in a mutex instead of a per-lane result vector. On
  one lane the driver allocates nothing; `tests/alloc_counts.rs` pins the remaining per-call
  allocations of compact Householder, Cholesky, QR, rank-revealing QR, LU solve, full-pivot LU
  solve, eigh, SVD, packed LU and triangular solve (their faer lane scratch), which do not grow
  with the batch. The families the table does not name are not pinned yet.
* **Output assembly.** Library-created vectors are cleared, reserved, and their spare capacity is
  split into disjoint per-lane `MaybeUninit` chunks written sequentially. `set_len` runs only after
  every lane returned `Ok` with its chunk full; otherwise the vectors stay empty.
* **Scratch.** faer work matrices and `MemBuffer`s are built once per lane and reused; buffers
  whose prior contents faer could observe (decomposition outputs, QR block coefficients, the thin
  `Q` seed) are reset per item to the state the pre-batching code allocated them in.
* **Empty `full` SVD.** For a matrix with a zero dimension in `full` mode the two providers differ,
  as they did before the extraction: faer emits the identity `U` and `Vᴴ` blocks while `tlinalg-blas`
  returns empty factors and leaves those blocks to the host.
