# tlinalg-rs

Tensor-free linear algebra for the Tensor4all stack.

`tlinalg` is the numerical layer that tenferro's CPU linear algebra is being extracted into. It owns
the kernels and the batch/scheduling behaviour; the host owns tensors, allocation, dtype dispatch,
placement, execution context and error wrapping.

## Crates

| Crate | Role |
|---|---|
| `tlinalg` | The faer-backed provider, with the batch-direction lane fan-out on the caller's rayon pool. |
| `tlinalg-blas` | The LAPACK/BLAS provider: vendor calls and their argument marshalling, a serial batch loop, vendor-owned threading. |
| `tlinalg-testkit` | Dev-only, unpublished: provider-neutral test helpers (scalar test trait, generators, reference arithmetic, batched layouts, counting allocator). |
| `tlinalg-parity` | Dev-only, unpublished: the same batched cases through both providers, compared gauge-aware. |

The two providers are siblings: neither depends on the other, and each owns its own vocabulary.
The dev-only crates depend on the providers, never the other way round (`tlinalg-testkit` is only
ever a `[dev-dependencies]` entry of a provider).

`tlinalg` is the faer-backed provider. It owns batched kernels (SVD, packed LU, Cholesky,
triangular solve, LU and solve, full-pivot LU, QR and column-pivoted QR, Hermitian and general
eigendecompositions, compact Householder QR), the batch loop and lane fan-out over them, and the
vocabulary its entry points take: borrowed strided I/O, parallelism (`Parallel`), lane policy
(`LanePlan`) and typed errors (`Error`). No tensor types.

`tlinalg-blas` has the same batched shape without a parallelism token: LAPACK and BLAS own their
threading, the batch loop is serial, and scratch comes from a host `Workspace`, queried and acquired
once per call and reused for every item. It was developed as `tensor4all/tlinalg-blas-rs` and merged
here with its history.

The interface a host requires of its linear-algebra providers is defined by the host: tenferro owns
it and adapts each provider to it.

## Contracts

Every entry point is batched, torch-style: one call per batch over a rank-`2 + B` strided
descriptor, with the library owning the batch loop and the lane fan-out. The contract is
[`docs/design/batched-api.md`](docs/design/batched-api.md).

The crate documentation is the specification for the rest:

* numerical conventions and failure behaviour — crate root of `tlinalg`;
* the parallelism and budget contract — `tlinalg::Parallel`;
* the batch lane contract — `tlinalg::LanePlan`;
* output assembly and lane scratch — `docs/design/batched-api.md`;
* the error vocabulary — `tlinalg::Error`.

## Status

Extraction in progress. Nothing is published; `publish = false` is set deliberately until the
interface and the package names settle.

## Build

```sh
cargo fmt --all -- --check
cargo clippy -j 16 --workspace --all-targets --features tlinalg-blas/link-openblas,tlinalg-parity/link-openblas -- -D warnings
cargo test -j 16 --workspace --features tlinalg-blas/link-openblas,tlinalg-parity/link-openblas
cargo test -j 16 --workspace --features tlinalg-blas/link-openblas,tlinalg-blas/provider-inject,tlinalg-parity/link-openblas
```

`link-openblas` exists only so `tlinalg-blas` has an executable check of its own; it builds OpenBLAS
from source through `openblas-src`. Tenferro selects the vendor and the injected-symbol path
itself, so the feature is not part of the implementation contract. Without it the workspace still
builds and lints; the LAPACK tests are skipped. `tlinalg-parity/link-openblas` turns on the
cross-provider parity suite, which needs both providers to run.

## Parity

`tlinalg-parity` runs every family (SVD thin/full/values, Cholesky, triangular solve in all flag
combinations, LU, solve, full-pivot LU and its solve, QR, column-pivoted QR, `eigh`/`eigvalsh`,
`eig`/`eigvals`, packed LU factor/prepared/fused solve, and the compact Householder pair) through
both providers for `f32`, `f64`, `Complex32` and `Complex64`, over batch shapes `[]`, `[3]` and
`[2, 3]` (gapped, so no axis merges, and with transposed batch strides), plus a stride-0 broadcast
coefficient for the binary families. What is unique is compared directly (singular values,
eigenvalues as a multiset, Cholesky factors, solutions, permutation-convention LU factors, QR factors
after fixing the `R` diagonal phase); the rest is checked by reconstruction against each
provider's documented convention.

## License

MIT OR Apache-2.0.
