# tlinalg-blas-rs

LAPACK/BLAS-backed implementation of the `tlinalg` tensor-free numerical interface
([`tlinalg-traits`](https://github.com/tensor4all/tlinalg-rs)).

## Crates

| Crate | Role |
|---|---|
| `tlinalg-blas` | The LAPACK/BLAS implementation: vendor calls and their argument marshalling, behind the same contract as the faer-backed implementation. |

## Contract

Like the faer-backed implementation, this crate owns the kernels and nothing else. The host supplies
borrowed operands, a `Parallel` token and a host-resolved `LanePlan`; the host keeps policy, error
classification, placement, allocation and session entry.

Two properties are deliberate here:

- **Read-only threading.** LAPACK and BLAS own their own parallelism, so the `Parallel` token is
  accepted for interface parity and **ignored**, no Rayon fan-out is created around a vendor batch,
  and the batch loops are serial.
- **No scratch.** The packed-LU family takes no pooled buffers, so it does not need `Workspace`.

`validate_pivots` is public so a host can validate a whole batch before splitting it into chunks: an
invalid pivot must be reported before any chunk mutates its output.

## Build and test

```sh
cargo fmt --all -- --check
cargo clippy -j 16 --all-targets --features link-openblas -- -D warnings
cargo test -j 16 --workspace --features link-openblas
```

`link-openblas` exists only so this crate has an executable check of its own; it builds OpenBLAS from
source through `openblas-src`. Tenferro selects the vendor and the injected-symbol path itself, so
the feature is not part of the implementation contract.

## Status

Only the packed-LU family (`lu_factor`, `lu_solve_prepared`, `lu_factor_solve`) is extracted so far.
Full-pivot LU, ordinary LU/solve, triangular solve, Cholesky, QR, `eigh`, `eig` and the Householder
family are later slices of [tenferro-rs#1956](https://github.com/tensor4all/tenferro-rs/issues/1956).

Nothing is published; `publish = false` until the interface and the package names settle.

## License

MIT OR Apache-2.0.
