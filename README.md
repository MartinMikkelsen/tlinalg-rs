# tlinalg-rs

Tensor-free linear algebra for the Tensor4all stack.

`tlinalg` is the numerical layer that tenferro's CPU linear algebra is being extracted into. It owns
the kernels and the batch/scheduling behaviour; the host owns tensors, allocation, dtype dispatch,
placement, execution context and error wrapping.

## Crate

`tlinalg` is the faer-backed provider. It owns the per-item kernels (SVD, packed LU, Cholesky,
triangular solve, LU and solve, full-pivot LU, QR and column-pivoted QR, Hermitian and general
eigendecompositions, compact Householder QR) and the vocabulary its entry points take: borrowed
strided I/O, scratch acquisition (`Workspace`), parallelism (`Parallel`), lane policy (`LanePlan`)
and typed errors (`Error`). No tensor types.

The interface a host requires of its linear-algebra providers is defined by the host: tenferro owns
it and adapts each provider to it. The LAPACK/BLAS provider currently lives in
[`tlinalg-blas-rs`](https://github.com/tensor4all/tlinalg-blas-rs) and is joining this workspace as a
sibling crate; the two providers do not depend on each other.

## Contracts

The crate documentation is the specification:

* numerical conventions and failure behaviour — crate root of `tlinalg`;
* the parallelism and budget contract — `tlinalg::Parallel`;
* the batch lane contract — `tlinalg::LanePlan`;
* buffer ownership and initialization — `tlinalg::Workspace`;
* the error vocabulary — `tlinalg::Error`.

## Status

Extraction in progress. Nothing is published; `publish = false` is set deliberately until the
interface and the package names settle.

## Build

```sh
cargo fmt --all -- --check
cargo clippy -j 16 --workspace --all-targets -- -D warnings
cargo test -j 16 --workspace
```

## License

MIT OR Apache-2.0.
