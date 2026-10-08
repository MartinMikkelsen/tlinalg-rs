# SVD: checking faer's divide-and-conquer path

## Purpose

faer 0.24.4's divide-and-conquer bidiagonal SVD, which its default parameters select once the
smaller dimension reaches 128, returns a spurious singular value and inaccurate factors for some
rank-deficient matrices with clustered singular values, without an error (issue #13). This log
compares three kernels for `crates/tlinalg/src/svd.rs` and records what each costs:

- **default**: faer's default parameters everywhere. The file at commit `8fef99c`. Inaccurate on
  the matrices above.
- **always QR**: `recursion_threshold: usize::MAX` everywhere. The default file with the patch at
  the end of this log applied.
- **checked**: the kernel this log is committed with. A decomposition with vectors whose smaller
  dimension is at least 128 uses the defaults, is checked by the Frobenius norm of
  `A - U diag(S) Vᴴ` against `24 * sqrt(max(m, n)) * epsilon` times that of `A`, and is repeated
  with the QR algorithm when the check fails or faer reports an error. A decomposition without
  vectors always uses the QR algorithm.

## Procedure

All three kernels were timed with the benchmark harness of this commit, which has the large,
rectangular, full-factor, clustered and complex values-only cases that `8fef99c` lacks. One binary
was built per kernel by replacing the kernel file only:

```
git show 8fef99c:crates/tlinalg/src/svd.rs > /tmp/svd_default.rs
cp crates/tlinalg/src/svd.rs /tmp/svd_checked.rs
git show 8fef99c:crates/tlinalg/src/svd.rs > crates/tlinalg/src/svd.rs
git apply --unidiff-zero always-qr.patch          # the patch at the end of this log
cp crates/tlinalg/src/svd.rs /tmp/svd_always_qr.rs
for kernel in default always_qr checked; do
    cp /tmp/svd_$kernel.rs crates/tlinalg/src/svd.rs
    cargo bench -p tlinalg-bench --bench kernels --no-run   # prints the path of the binary
    cp target/release/deps/kernels-<hash> /tmp/kernels_$kernel
done
cp /tmp/svd_checked.rs crates/tlinalg/src/svd.rs
export TLINALG_BENCH_THREADS=4
for round in 1 2 3; do
    for kernel in default always_qr checked; do
        /tmp/kernels_$kernel --bench '^svd' --save-baseline ${kernel}_r$round
    done
done
```

The three binaries ran interleaved so that drift of the machine affects them alike. Each
Criterion measurement is 10 samples; `svd-recursion-threshold-medians.csv` holds the median of
every case, row, kernel and round in milliseconds. Machine: Apple M3 Pro (5 performance and 6
efficiency cores), aarch64-apple-darwin, rustc 1.98.1, faer 0.24.4, no vendor BLAS.

## Inputs

- `svd`, `svd_full`, `svdvals`: `Batch::general`, a deterministic matrix with a growing diagonal
  and periodic off-diagonal entries (`tlinalg_testkit::matrix`). It is well conditioned and its
  singular values are not clustered, so the checked kernel never repeats on it.
- `svd_clustered`: `Batch::clustered`, the rank-deficient matrix with clustered singular values of
  the regression test. At size 160 the default path is inaccurate for `f64` and `Complex64` and
  the checked kernel repeats the decomposition; at size 256 it does not.

`n1024xb1` is one 1024 x 1024 matrix, `t1024x256` one tall 1024 x 256 matrix, and `w256x1024` one
wide 256 x 1024 matrix.

## Results

Fastest of the three rounds, in milliseconds; each ratio is to the default kernel in the same
row.

| Factors | Case | One lane: default | checked | ratio | always QR | ratio | Pool: default | checked | ratio | always QR | ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| thin, `f64` | `n128xb1` | 1.75 | 1.85 | 1.06 | 2.21 | 1.26 | 1.37 | 1.44 | 1.05 | 2.21 | 1.61 |
| thin, `f64` | `n256xb1` | 9.63 | 10.34 | 1.07 | 14.50 | 1.51 | 7.30 | 7.57 | 1.04 | 13.49 | 1.85 |
| thin, `f64` | `n512xb1` | 55.60 | 60.92 | 1.10 | 98.91 | 1.78 | 34.31 | 35.92 | 1.05 | 86.00 | 2.51 |
| thin, `f64` | `n1024xb1` | 363 | 405 | 1.12 | 736 | 2.03 | 165 | 181 | 1.10 | 584 | 3.54 |
| thin, `f64` | `t1024x256` | 18.81 | 21.66 | 1.15 | 23.43 | 1.25 | 12.30 | 13.39 | 1.09 | 18.02 | 1.46 |
| thin, `f64` | `w256x1024` | 18.95 | 22.02 | 1.16 | 23.57 | 1.24 | 12.58 | 13.44 | 1.07 | 18.13 | 1.44 |
| full, `f64` | `n256xb1` | 9.63 | 10.35 | 1.08 | 14.36 | 1.49 | 7.30 | 7.56 | 1.04 | 13.15 | 1.80 |
| full, `f64` | `n1024xb1` | 362 | 409 | 1.13 | 719 | 1.99 | 169 | 180 | 1.07 | 577 | 3.41 |
| full, `f64` | `t1024x256` | 36.08 | 39.98 | 1.11 | 40.88 | 1.13 | 18.77 | 21.23 | 1.13 | 23.99 | 1.28 |
| full, `f64` | `w256x1024` | 36.37 | 38.99 | 1.07 | 41.27 | 1.13 | 18.65 | 19.87 | 1.07 | 24.91 | 1.34 |
| thin, `Complex64` | `n128xb1` | 3.36 | 3.81 | 1.13 | 3.75 | 1.12 | 2.46 | 2.62 | 1.06 | 3.11 | 1.26 |
| thin, `Complex64` | `n256xb1` | 21.07 | 24.45 | 1.16 | 25.76 | 1.22 | 13.68 | 14.66 | 1.07 | 19.35 | 1.41 |
| thin, `Complex64` | `n1024xb1` | 1039 | 1248 | 1.20 | 1388 | 1.34 | 358 | 425 | 1.19 | 771 | 2.16 |
| thin, `Complex64` | `t1024x256` | 61.45 | 76.60 | 1.25 | 66.74 | 1.09 | 29.00 | 33.97 | 1.17 | 35.45 | 1.22 |
| thin, `Complex64` | `w256x1024` | 61.68 | 76.55 | 1.24 | 67.21 | 1.09 | 29.36 | 34.07 | 1.16 | 35.76 | 1.22 |
| full, `Complex64` | `n256xb1` | 21.02 | 24.89 | 1.18 | 26.03 | 1.24 | 13.56 | 14.86 | 1.10 | 19.71 | 1.45 |
| full, `Complex64` | `n1024xb1` | 1036 | 1270 | 1.23 | 1414 | 1.36 | 358 | 424 | 1.18 | 781 | 2.18 |
| full, `Complex64` | `t1024x256` | 138 | 153 | 1.11 | 144 | 1.04 | 52.24 | 57.78 | 1.11 | 59.46 | 1.14 |
| full, `Complex64` | `w256x1024` | 139 | 156 | 1.12 | 143 | 1.03 | 53.18 | 58.48 | 1.10 | 59.12 | 1.11 |
| values only, `f64` | `n128xb1` | 1.21 | 1.17 | 0.97 | 1.09 | 0.91 | 0.99 | 1.18 | 1.19 | 1.14 | 1.15 |
| values only, `f64` | `n256xb1` | 6.17 | 6.22 | 1.01 | 6.01 | 0.97 | 5.57 | 6.35 | 1.14 | 6.09 | 1.09 |
| values only, `f64` | `n512xb1` | 33.25 | 34.92 | 1.05 | 34.13 | 1.03 | 26.27 | 31.55 | 1.20 | 30.74 | 1.17 |
| values only, `f64` | `n1024xb1` | 206 | 218 | 1.06 | 212 | 1.03 | 115 | 137 | 1.19 | 136 | 1.18 |
| values only, `f64` | `t1024x256` | 9.84 | 9.94 | 1.01 | 9.40 | 0.96 | 8.57 | 9.44 | 1.10 | 9.15 | 1.07 |
| values only, `f64` | `w256x1024` | 9.98 | 10.07 | 1.01 | 9.85 | 0.99 | 8.69 | 9.60 | 1.10 | 9.24 | 1.06 |
| values only, `Complex64` | `n128xb1` | 1.92 | 1.88 | 0.98 | 1.88 | 0.98 | 1.71 | 1.89 | 1.11 | 1.89 | 1.11 |
| values only, `Complex64` | `n256xb1` | 11.40 | 11.43 | 1.00 | 11.48 | 1.01 | 10.12 | 10.82 | 1.07 | 10.84 | 1.07 |
| values only, `Complex64` | `n1024xb1` | 519 | 521 | 1.00 | 530 | 1.02 | 210 | 228 | 1.09 | 233 | 1.11 |
| values only, `Complex64` | `t1024x256` | 27.28 | 27.05 | 0.99 | 27.42 | 1.01 | 19.09 | 19.10 | 1.00 | 19.41 | 1.02 |
| values only, `Complex64` | `w256x1024` | 26.59 | 27.20 | 1.02 | 27.34 | 1.03 | 18.20 | 19.17 | 1.05 | 19.46 | 1.07 |
| thin, clustered, `f64` | `n160xb1` | 2.05 | 4.62 | 2.25 | 2.49 | 1.21 | 1.56 | 3.76 | 2.40 | 2.24 | 1.43 |
| thin, clustered, `f64` | `n256xb1` | 7.36 | 8.05 | 1.09 | 10.60 | 1.44 | 5.53 | 5.77 | 1.04 | 9.64 | 1.74 |
| thin, clustered, `Complex64` | `n160xb1` | 5.23 | 12.11 | 2.31 | 5.97 | 1.14 | 3.50 | 8.36 | 2.39 | 4.62 | 1.32 |
| thin, clustered, `Complex64` | `n256xb1` | 19.07 | 22.53 | 1.18 | 22.57 | 1.18 | 12.00 | 13.29 | 1.11 | 16.49 | 1.37 |

## Variation between rounds

For the cases of size 128 and above, the slowest round of a case took a median of 1.08 times as
long as its fastest round on one lane (90th percentile 1.12, largest 1.56), and 1.10 times on the
four-thread pool (90th percentile 1.22, largest 3.45). In the individual rounds the ratio of the
checked to the default kernel for the thin 1024 x 1024 SVD was 1.12, 1.17 and 1.12 for `f64` and
1.24, 1.11 and 1.22 for `Complex64` on one lane. Differences of less than about 10% between
kernels are not resolved, and a single pool figure can be off by much more.

## Conclusions

- The check costs 6% to 16% of a decomposition for `f64` and 11% to 25% for `Complex64` on one
  lane, and 4% to 19% on the pool.
- When the check fails, the decomposition runs twice: 2.3 times the default on one lane at size
  160, for both scalar types.
- Always using the QR algorithm costs real input the most: up to 2.0 times the default for a
  square `f64` matrix on one lane and about 3.5 times on the pool at size 1024. For `Complex64`
  it costs up to 1.36 times on one lane and 2.2 times on the pool.
- For tall and wide `Complex64` matrices with a small dimension of 256, always using the QR
  algorithm (1.03 to 1.09 on one lane) is cheaper than checking (1.11 to 1.25), and for the
  128 x 128 `Complex64` matrix the two cost the same. For every other case measured the checked
  kernel is the faster of the two on matrices that pass the check.
- Values-only decompositions, which use the QR algorithm in the checked kernel, are within 6% of
  the default on one lane for `f64` and `Complex64`, and up to 20% slower on the pool for `f64`.

## Remaining constraints

- Values-only decompositions are not checked; they avoid the divide-and-conquer path instead.
- The failure was observed for `f64` and `Complex64`. The clustered spectrum did not trigger it
  for `f32` and `Complex32`, and no bound is known on the sizes or spectra that trigger it: a
  160 x 160 matrix fails where 128 x 128 and 256 x 256 matrices with the same spectrum do not.
- The bound of the check, `24 * sqrt(max(m, n)) * epsilon`, rests on residuals of 10 to 60 machine
  epsilons measured for accurate decompositions of sizes up to 1024. A larger accurate
  decomposition whose residual exceeded it would be repeated needlessly, not returned wrong.
- The `lapack` rows were not run: building OpenBLAS from source failed on this machine.
- Once faer's divide-and-conquer SVD is fixed, the check and the QR parameters can go; the
  regression test `clustered_singular_values_of_a_rank_deficient_matrix` then has to pass with
  faer's defaults.
- `eigh` passes faer's default parameters too and was not examined.

## The always-QR kernel

The patch to `crates/tlinalg/src/svd.rs` at `8fef99c` that gives the always-QR kernel:

```diff
diff --git a/crates/tlinalg/src/svd.rs b/crates/tlinalg/src/svd.rs
index 10598f4..41629e0 100644
--- a/crates/tlinalg/src/svd.rs
+++ b/crates/tlinalg/src/svd.rs
@@ -24 +24 @@ use faer::dyn_stack::{MemBuffer, MemStack};
-use faer::linalg::svd::ComputeSvdVectors;
+use faer::linalg::svd::{ComputeSvdVectors, SvdParams};
@@ -33,0 +34,15 @@ use crate::{Error, FaerScalar, Op, Parallel, Result};
+/// The parameters every decomposition here hands to faer.
+///
+/// faer's defaults switch from the QR algorithm to a divide-and-conquer bidiagonal SVD once the
+/// smaller dimension reaches `recursion_threshold` (128). In faer 0.24.4 that path returns a
+/// spurious singular value and inaccurate factors for some rank-deficient matrices with clustered
+/// singular values, without reporting an error (see the test
+/// `clustered_singular_values_of_a_rank_deficient_matrix`). The threshold is raised so that the QR
+/// algorithm is used at every size.
+fn params<E: faer::traits::ComplexField>() -> faer::Spec<SvdParams, E> {
+    faer::Spec::new(SvdParams {
+        recursion_threshold: usize::MAX,
+        ..<SvdParams as faer::Auto<E>>::auto()
+    })
+}
+
@@ -54 +69 @@ impl<E: faer::traits::ComplexField> SvdScratch<E> {
-            faer::linalg::svd::svd_scratch::<E>(m, n, vectors, vectors, par, Default::default())
+            faer::linalg::svd::svd_scratch::<E>(m, n, vectors, vectors, par, params())
@@ -88 +103 @@ fn svd_values_item<T: FaerScalar>(
-        Default::default(),
+        params(),
@@ -129 +144 @@ fn svd_item<T: FaerScalar>(
-        Default::default(),
+        params(),
```
