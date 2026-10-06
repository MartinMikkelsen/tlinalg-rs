# Library-owned native lane scheduling

Owning-layer prerequisite for tensor4all/tenferro-rs#2004, U6. This supersedes the host-policy parts of `batched-api.md`; tensor formats, numerical leaves, output/error guarantees and the independent vendor provider stay unchanged.

## Contract and accepted faer limitation

Every native batched entry point takes one `Parallel` token, never a second executable lane/resource token. `LanePlan` and its module become crate-private. The private plan carries only a lane count, not a pool or lifetime. No compatibility shim, public policy escape hatch or new dependency. Tenferro's pinned adapter is migrated in the subsequent tenferro PR, not partially edited here.

The effective requested width is `min(budget, pool.current_num_threads())`; Sequential and effective width one execute on the calling thread without installing a pool. Keep `Parallel::budget()` as the caller's requested value; derive effective width privately. tlinalg's own outer numerical lane count is strictly bounded by effective width. Outer children are sequential, so tlinalg does not add nested parallel fan-out.

**faer's intra-item count is a hint, not a strict active-thread bound.** `faer::Par::degree()` calls it the number of threads that "should ideally execute" an operation. With faer 0.24.4, triangular-solve RHS recursion ignores split tokens and `join_raw` gives both children a rounded-up half. Thus even a pool-clamped faer count can expose more active numerical tasks than requested. The maintainer explicitly accepted proceeding with this limitation recorded. Do not claim a hard bound on all native numerical execution. Pass the effective requested count and install only the supplied pool; no silent serial downgrade, replacement pool, spindle integration or upstream patch. The pool's physical worker count remains its bound, not the hint. See the work log for source evidence and a synthetic scheduling reproducer, which is not a measured full-solver result.

Admission across concurrent calls belongs to the resource owner; this PR introduces no per-call admission framework. Vendor BLAS/LAPACK threading stays vendor-owned and is not governed by `Parallel`.

## Auto policy, not a new tuning experiment

Move the current tenferro default Auto policy into tlinalg, retaining private `AUTO_FAN_OUT_MAX_ITEM_DIM = 64`. Ordinary families provide `max(rows, cols)` from validated operands (including target columns for solves and reflector application); their batch fans out only for small items with at least one item per effective worker and at least two workers. Packed LU's three routes retain their no-size-cutoff Auto policy. Actual contiguous lane count is `batch.div_ceil(batch.div_ceil(requested_lanes))`, never above batch or effective width. Otherwise one lane uses the bounded count hint for faer's intra-item work. Choose from routine dimensions, not caller-supplied diagnostic `Op`.

Keep the existing cutoff rather than inventing uncalibrated work/flop constants. tenferro#2000's work-model tuning is a tlinalg-owned follow-up, not implemented by moving ownership. Forced host strategies/custom thresholds are not reproduced as a new public surface.

`batch::run` remains the single batch loop, resolving a private lane count from the token and a size hint: `Some(max_item_dim)` for ordinary families, `None` for packed LU. No extra per-item allocation or output initialization. Use `pool.in_place_scope` for outer tasks; faer's numerical closure uses `pool.install`. These are library execution, not installation of a host operation continuation.

## Scope and integration

Update every native entry point, executable docs, native tests, parity calls and benchmarks. Preserve library-created outputs' empty-on-error and deterministic lowest failing item; direct/in-place output guarantees stay per item. Vendor signatures, serial batch loop, queried-once workspace and threading are unchanged.

Benchmarks use the ordinary API only: Sequential (`faer-1lane`) and explicit Pool+budget (`faer-Nt`), no fake host plan. Overhead baselines use explicit 1T settings and report effective pool width; 4T/8T are separate scheduling coverage, not default overhead baselines. No speedup or completed integrated-tenferro performance claim in this prerequisite PR.

## Acceptance

- Native calls compile with only Parallel; no public LanePlan remains.
- Driver tests check width one/no pool dispatch, supplied-pool identity, budgets 1/2/3/4 on a larger pool, oversized budget, empty/single item, 64/65 cutoff, packed exception, same/foreign-pool entry, uneven chunking, exact once-per-item execution and bounded **outer** concurrency. Do not mistake observing a faer count for proving its active-thread bound.
- Production numerical tests cover four scalar types, batch strides/broadcast, both triangular-solve sides, 64/65 RHS and reflector extent boundaries, LU solve without RHS, and all three packed routes above 64. Replace impossible three-lane/budget-two fixtures without dropping partial-mutation assertions.
- Existing allocation gates stay unchanged; scratch allocations do not grow with batch length.
- Repository fmt, clippy (unlinked and linked/injected), debug/CI tests (unlinked and linked/injected), release native tests, docs with warnings denied and Rust 1.89 check. Completed independent gpt-6.1-sol pre/post review and coherent self-review before one PR, no merge.

## Provenance

The cutoff/default conditions are adapted from same-project `tenferro-rs` commit `527f58cd`, `crates/tenferro-linalg/src/cpu/tlinalg.rs:42-115` and `crates/tenferro-cpu/src/batch_policy.rs:78-97,312-316` (MIT OR Apache-2.0). Preserve existing numerical sources/notices; no new third-party numerical implementation or copied tests.
