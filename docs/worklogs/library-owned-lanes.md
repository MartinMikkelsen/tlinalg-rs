# U6 library-owned lanes

## Scope and state

User requested the next tenferro#2004 step through PR creation after the prepared contraction prerequisites merged. Selected rollout step 4: tlinalg single-token native scheduling, budget enforcement and library-owned Auto lanes; later host adapter migration stays separate. Branch `library-owned-lanes` from fetched `origin/main`, base `2734716b5b224a67672aef075df6eb90a6ea5ed7`. At the end of the initial investigation, no production changes, commits, upstream draft, upstream patch or PR had been created. The maintainer subsequently accepted proceeding with the limitation recorded; current implementation/validation status is below.

`kache doctor --verify` passed (3,607 valid entries). Baseline `CARGO_BUILD_JOBS=16 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo test -p tlinalg` passed, including 24 doctests; log `/tmp/tlinalg-lanes-baseline.log`. This is a baseline, not validation of an implemented U6 change. Existing untracked `.codegraph/` and `bench-tlinalg-*.log` files were preserved.

## Completed independent pre-review

Fresh read-only `pi --no-session --no-extensions --tools read,grep,find,ls --model openai/gpt-6.1-sol:high` reviewed the proposal, actual shared driver/native routines and host default policy. Request `/tmp/tlinalg-lanes-pre-review.md`, completed report `/tmp/tlinalg-lanes-pre-review-out.md`.

BLOCKER: passing a bounded faer count does not establish the intended strict intra-item numerical concurrency bound in faer 0.24.4. IMPORTANT: acceptance must test real routine shape hints and numerical routes, not only the private resolver. MINORs: replace the impossible existing three-lane/budget-two fixture without weakening mutation guarantees; preserve `Parallel::budget()` as the requested ceiling. This initial review stopped implementation under the original hard-bound proposal. That proposal was subsequently revised with maintainer approval; see the accepted disposition below.

The independent exploration child hit its tool budget; it was not counted as completed review. Main directly read the required downstream policy/source instead.

## Verified blocking source evidence

Locked faer is 0.24.4, source at Cargo registry `faer-0.24.4`:

- `src/linalg/triangular_solve.rs:511-528`: wide-RHS split for `k > 64 && n <= 128`; both children ignore `join_raw`'s reduced token and reuse the original `par`. The unit-diagonal sibling does the same.
- `src/utils/mod.rs:25-38`: `join_raw` gives both children `ceil(n_threads/2)`. Repeated splitting can expose four sequential leaves under an odd budget of three.
- `spindle-0.2.6/src/lib.rs:556-565`: absent a spindle lock, the fork uses ordinary Rayon iteration. tlinalg establishes no spindle lock. Merely clamping faer's requested count to pool width does not cure either defect.

Read-only fetch of canonical Codeberg `main` (`https://codeberg.org/sarah-quinones/faer/raw/branch/main/faer/src/{utils/mod.rs,linalg/triangular_solve.rs}`) during this investigation still showed both patterns; a simple assumption that upstream main already fixed them was not supported. These are mutable-branch observations, not an exact tested upstream revision or a full faer audit.

## Local mechanism reproducer

Isolated safe diagnostic harness `/tmp/tlinalg-faer-budget-probe/{Cargo.toml,src/main.rs}`, pinned faer `=0.24.4`, explicit eight-worker Rayon pool. Command:

```
CARGO_BUILD_JOBS=16 cargo run --offline \
  --manifest-path /tmp/tlinalg-faer-budget-probe/Cargo.toml \
  --target-dir /home/shinaoka/projects/tensor4all/tlinalg-rs/target
```

Full output `/tmp/tlinalg-faer-budget-probe.log`:

| Requested budget | Recursive mode | Observed peak active leaves |
|---|---|---:|
| 1 | honor split token | 1 |
| 2 | honor split token | 2 |
| 2 | reuse original token | 8 |
| 3 | honor split token | 4 |

The harness calls faer's actual `join_raw` with independent synthetic leaf bodies (atomic active/peak counters and a sleep to expose overlap). It reproduces the scheduling mechanism, **not** the production triangular solver's measured peak concurrency. No timing, speedup, numerical-error, Miri, sanitizer or integrated-host claim follows from it.

## Initial stop (superseded by maintainer decision)

Do not implement a false hard-budget promise, silently serialize undersized budgets, create replacement pools per call, or expand this into an executor framework. The preferred prerequisite is a correction in faer's owning scheduling/triangular-solve layer, including an audit of its other count-based fan-out before relying on a strict bound. Shared provenance rules require explicit user permission **before preparing an upstream-facing issue draft or patch**, and separate explicit permission before submission. No such upstream preparation has begun.

## Accepted disposition and implementation

The maintainer said to record the point and proceed. Re-reading faer's `Par::degree()` documentation (`src/lib.rs:952-956`) confirmed that its count is an ideal requested count, not a strict active-thread promise. Treat the ignored split token as a likely implementation issue, but do not claim exceeding the hint is necessarily a public-contract violation. No numerical-result bug was demonstrated.

Revised contract: tlinalg strictly bounds its own outer lanes by `min(budget, pool width)` and runs sequential children. A single lane passes that effective count to faer as a hint on the supplied pool; **no hard active-thread bound for intra-item execution is promised**. Effective width one stays on the caller. No replacement pool, retry, spindle integration, new dependency or upstream preparation. #2000 work-model calibration and the tenferro adapter remain separate follow-ups.

Completed gpt-6.1-sol design delta pre-review `/tmp/tlinalg-lanes-pre-review-delta-out.md` closed the blocker under this explicit accepted contract. Its work-log status cleanup finding is addressed here.

Implementation removes public LanePlan and the second token from all 19 native numerical entry points, retaining the existing Auto cutoff/packed exception. The sole shared driver resolves private resource-free lane counts and uses the selected pool's `in_place_scope`. Updated caller tests, parity and ordinary-API benchmarks. Existing unsafe output ownership and numerical leaves were not expanded. Public `Parallel` and durable design docs explicitly record the faer limitation.

The implementation worker's initial tool-budget stop and follow-up timeout were not counted as completed work/review. Main integrated its partial caller edits, corrected an unresolved `par::single` call and stale argument/prose, removed a duplicate weaker numerical test, and ran the actual repository checks. The worker unexpectedly ran Cargo despite instructions not to; main checked no owned Cargo/rustc process remained before continuing sequential validation.

Focused driver/production tests passed: budgets 1/2/3/4 and oversized requests, empty/single/uneven batches, exact once-per-item execution, caller-thread width one, same/foreign pool identity, count hints, 64/65 RHS and reflector state/target boundaries, and every packed route above64 with numerical assertions. A test-only caller-thread lane-count recorder makes real production shape decisions observable; it is absent from release code.

Negative control: temporarily removing pool-width clamping made the focused scheduling test fail (`requested99` passed instead of the expected bounded hint). Restoring the clamp passed the workspace gate. Logs `/tmp/tlinalg-lanes-clamp-red.log` and `/tmp/tlinalg-lanes-debug.log`. `cargo test --workspace` passed, including 22 native doctests; the existing testkit allocator doctest remains ignored, unchanged.

## Local validation and review handoff

All applicable local gates passed with `CARGO_BUILD_JOBS=16`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`:

- `cargo fmt --all -- --check` and staged `git diff --check`.
- `cargo clippy --workspace --all-targets -- -D warnings`.
- Same clippy with `tlinalg-blas/link-openblas,tlinalg-blas/provider-inject,tlinalg-parity/link-openblas,tlinalg-bench/link-openblas`.
- `cargo test --workspace` (unlinked debug).
- `cargo test --workspace --profile ci --features tlinalg-blas/link-openblas,tlinalg-parity/link-openblas`.
- Same CI-profile gate also enabling `tlinalg-blas/provider-inject`.
- `cargo test -p tlinalg --release`, including existing allocation gates and 22 doctests.
- `cargo +1.89.0 check --workspace --all-targets`.
- `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps`.

Both linked/injected parity runs executed all 52 tests. Logs `/tmp/tlinalg-lanes-{debug,clippy,clippy-linked,ci-linked,ci-injected,release,msrv,docs}.log`.

The initial linked clippy attempt failed when the OpenBLAS build's C linker could not find `-lgfortran`. GNU Fortran9 and its library were already installed. Setting command-local `LIBRARY_PATH=/usr/lib/gcc/x86_64-linux-gnu/9` resolved the search path; no installation or shared configuration change. Preserve the initial failure log `/tmp/tlinalg-lanes-clippy-linked-initial.log`. Vendor thread settings above are explicit, but no vendor/runtime thread query, timing/speedup or integrated-host performance measurement is claimed.

Main completed a coherent full-diff self-review, checking every native shape hint, shared scheduling/output/error path, unchanged allocation assertions, four-dtype numerical test migration, vendor isolation, documentation and source provenance.

Completed independent gpt-6.1-sol post-review `/tmp/tlinalg-lanes-post-review-out.md` found no BLOCKER/IMPORTANT findings. Its optional Delete-list cleanup was verified: two now-identical packed-LU test forwarding wrappers were replaced by import aliases, retaining the descriptor-building prepared-solve helper. Relevant final native debug/release tests, linked-injected all-target clippy, Rust1.89 all-target check, docs and fmt passed after cleanup; logs `/tmp/tlinalg-lanes-{native,release,clippy-linked,msrv,docs}-final.log`. Final resolver computes chunk geometry only once by retaining the private requested lane upper bound; focused-final tests prove the same actual lane decisions.

Ready for one owning-layer PR. Hosted CI has not yet been asserted green; no merge authorization is implied. No benchmark timing/scaling experiment, integrated tenferro migration, Miri or sanitizer proof is claimed.
