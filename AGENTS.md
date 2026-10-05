# AGENTS.md

Guidance for agents working in this repository.

Before acting, read the latest shared tensor4all agent rules:
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`.
If the network is unavailable, use the sibling checkout `../tensor4all-agent-rules/rules/index.md`.
Load only the rule files the task needs. If neither source is available, continue and state that
the shared rules were unavailable.

## Repository-specific rules

* `tlinalg-blas` owns its vocabulary (`Error`, `Op`, `Workspace`, `IndexWorkspace`, `Scalar`);
  there is no shared interface crate. The interface tenferro requires lives in tenferro, which
  adapts these functions. Changing a public signature or `Error` is a contract change for that
  adapter and needs the design reviewed first.
* Every public entry point is batched: one call per batch over a rank `2 + B` strided operand, a
  serial loop, and one workspace query and acquisition per call. No parallelism token: LAPACK and
  BLAS own their threading.
* Keep SIMD and reusable strided kernels in `strided-rs`, not here.
* Do not add a provider registry, autotuner, or automatic provider switching. Implementations are
  selected by the host.
* Do not widen the interface to make a host-side detail easier. The host adapts.
