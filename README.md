# tlinalg-rs

Tensor-free linear algebra for the Tensor4all stack.

`tlinalg` is the numerical layer that tenferro's CPU linear algebra is being extracted into. It owns
the kernels and the batch/scheduling behaviour; the host owns tensors, allocation, dtype dispatch,
placement, execution context and error wrapping.

## Crates

| Crate | Role |
|---|---|
| `tlinalg-traits` | The vocabulary that crosses the boundary: borrowed strided I/O, scratch acquisition, parallelism, lane policy, typed errors. No kernels, no tensor types. |

The faer-backed implementation (`tlinalg`) lands in this repository. The LAPACK/BLAS implementation
lives in [`tlinalg-blas-rs`](https://github.com/tensor4all/tlinalg-blas-rs); both share
`tlinalg-traits` and neither depends on the other, nor on `tprims`.

## Contracts

`tlinalg-traits` is the frozen interface, so its documentation is the specification:

* numerical conventions and per-provider failure behaviour — crate root of `tlinalg-traits`;
* the parallelism and budget contract — `tlinalg_traits::Parallel`;
* the batch lane contract — `tlinalg_traits::LanePlan`;
* buffer ownership and initialization — `tlinalg_traits::Workspace`;
* the error vocabulary — `tlinalg_traits::Error`.

## Status

Interface extraction in progress. Nothing is published; `publish = false` is set deliberately until
the interface and the package names settle.

## Build

```sh
cargo fmt --all -- --check
cargo clippy -j 16 --workspace --all-targets -- -D warnings
cargo test -j 16 --workspace
```

## License

MIT OR Apache-2.0.
