# tlinalg-blas-rs

LAPACK/BLAS-backed implementation of the `tlinalg` tensor-free numerical interface
(`tlinalg-traits`).

It shares the interface with the faer-backed `tlinalg` implementation and must not impose Rayon
threading on vendor kernels: vendor-owned threading, one workspace query per batch with reuse across
a serial batch, and the existing vendor admission rules are preserved.

Not started yet. The faer-backed extraction lands first in
[`tlinalg-rs`](https://github.com/tensor4all/tlinalg-rs).

## License

MIT OR Apache-2.0.
