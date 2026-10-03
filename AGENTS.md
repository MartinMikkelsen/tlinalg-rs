# AGENTS.md

Guidance for agents working in this repository.

Before acting, read the latest shared tensor4all agent rules:
`https://github.com/tensor4all/tensor4all-agent-rules/blob/main/rules/index.md`.
If the network is unavailable, use the sibling checkout `../tensor4all-agent-rules/rules/index.md`.
Load only the rule files the task needs. If neither source is available, continue and state that
the shared rules were unavailable.

## Repository-specific rules

* `tlinalg-traits` is an interface crate. It contains no numerical kernels and no tensor types;
  keep it that way.
* The interface is published behaviour: changing `Parallel`, `LanePlan`, `Workspace` or `Error`
  is a contract change and needs the design reviewed first.
* Keep SIMD and reusable strided kernels in `strided-rs`, not here.
* Do not add a provider registry, autotuner, or automatic provider switching. Implementations are
  selected by the host.
* Do not widen the interface to make a host-side detail easier. The host adapts.
