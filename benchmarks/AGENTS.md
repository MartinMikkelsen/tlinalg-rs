<!-- Adapted from tensor4all/tprims-rs benchmarks/AGENTS.md. -->

# AGENTS.md — benchmarks/

Guidance in the repository root `AGENTS.md` applies in full. The rules that are specific to
measuring:

## Critical Rules

- **Never run measurements in parallel**, and **never build while a measurement is running.**
  Run one thread budget after another (1T first), one process at a time. Concurrent runs
  interfere and produce numbers that look like results.
- **Record the measured commit, the core set, the vendor and its thread budget for every published
  number**, next to the number. A measurement without them is not evidence.
- **Always pin CPU cores** with `taskset`, including at 1T, inside one L3 domain — and check that
  those cores *and their SMT siblings* were idle before and after. `scripts/idle_cpus.py check`
  does the expansion; `scripts/pinned.sh` applies it and discards a spoiled run.
- **Correctness precedes timing.** `tlbench verify` must report a clean grid for the sizes and
  thread counts of a run before any timing is taken from it.
- **Priming is time-based** (at least 0.5 s of wall time per arm) and identical on every arm. A
  fixed call count is not a substitute: after an idle gate this host reads low for the first second
  or two of sustained AVX-512 work, which is easily mistaken for a kernel defect. The reference
  harness in `tprims-rs` records 500 ms while doing a single warm-up call; do not copy that.
- **Campaign results go to [tlinalg-benchmark](https://github.com/tensor4all/tlinalg-benchmark)**,
  not here. That repository owns the revision pin, the declarations, the run manifests, the
  published reports and the staleness index. An ad-hoc run kept here, or a table copied out of a
  campaign report without its commit, is not evidence.

## Scope

- `tlbench` measures the two providers' public batched entry points directly, the way a host calls
  them after it has selected its execution pool. Do not widen a provider's API for the harness, do
  not add a provider registry or an autotuner, and do not move lane policy out of the library.
- Keep SIMD and reusable strided kernels in `strided-rs`.
