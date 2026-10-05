# tlinalg-blas-rs

LAPACK/BLAS-backed tensor-free linear algebra kernels for the Tensor4all stack.

## Crates

| Crate | Role |
|---|---|
| `tlinalg-blas` | The LAPACK/BLAS implementation: vendor calls and their argument marshalling, behind the same contract as the faer-backed implementation. |

## Contract

The crate owns the vendor calls, their argument marshalling and the batch loop; the host keeps
policy, error classification, placement, tensor construction and session entry. It owns its own
vocabulary (`Error`, `Op`, `Workspace`, `IndexWorkspace`, `Scalar`); the interface tenferro requires
lives in tenferro.

- **Batched.** Every family is called once per batch with a rank `2 + B` strided operand
  `[rows, cols, batch...]` (stride-0 batch axes broadcast an input). Outputs are compact and
  batch-contiguous, cleared and then filled, and left empty on error; the error is the one of the
  first failing item in batch order.
- **Vendor-owned threading.** No entry point takes a parallelism token; the batch loop is serial.
- **One workspace query per call.** Scratch comes from the host's `Workspace`, is acquired once per
  call and reused for every item.

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

Every CPU family tenferro's LAPACK route uses is here: SVD, packed and explicit LU, solve (owned and
direct-output), full-pivot LU and its solve, triangular solve, Cholesky, QR, rank-revealing QR,
`eigh`, `eig`, and the compact Householder kernels. Nothing is published; `publish = false`.

## License

MIT OR Apache-2.0.
