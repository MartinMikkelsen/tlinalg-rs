# AGENTS.md

Guidance for agents working in this repository.

Before acting, read the latest shared tensor4all agent rules:
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`.
If the network is unavailable, use the sibling checkout `../tensor4all-agent-rules/rules/index.md`.
Load only the rule files the task needs. If neither source is available, continue and state that
the shared rules were unavailable.

## Repository-specific rules

* `tlinalg` is a provider: tensor-free kernels plus the small vocabulary its own entry points take
  (`Parallel`, `LanePlan`, `Scalar`, `Error`, `Op`). It contains no tensor types.
* Every public entry point is batched (`docs/design/batched-api.md`); per-matrix kernels stay
  crate-private, and `crates/tlinalg/src/batch.rs` is the only batch loop.
* The interface a host requires of its linear-algebra providers belongs to the host (tenferro
  defines it and adapts each provider to it). Do not reintroduce a shared trait crate here, and do
  not shape these types for another provider: `tlinalg-blas` (joining this workspace as a sibling
  crate) never depends on `tlinalg`, nor `tlinalg` on it.
* The public types are published behaviour: changing `Parallel`, `LanePlan`, `Error`, or the
  batched contract in `docs/design/batched-api.md` is a contract change and needs the design
  reviewed first.
* Keep SIMD and reusable strided kernels in `strided-rs`, not here.
* Do not add a provider registry, autotuner, or automatic provider switching. Implementations are
  selected by the host.
* Do not widen the interface to make a host-side detail easier. The host adapts.
