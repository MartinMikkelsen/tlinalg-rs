use crate::batch::{out, run};
use crate::{Op, Parallel};
use core::num::NonZeroUsize;
use core::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::cell::Cell;
use strided_view::{RawStridedMut, RawStridedRef};

thread_local! {
    // The driver records on its caller before dispatch: real API shape decisions are observable
    // without changing release code or interfering with tests on other threads.
    static LAST_LANES: Cell<usize> = const { Cell::new(0) };
}

pub(in crate::batch) fn record_lanes(lanes: usize) {
    LAST_LANES.set(lanes);
}

fn token(pool: &rayon::ThreadPool, budget: usize) -> Parallel<'_> {
    Parallel::Pool {
        pool,
        budget: NonZeroUsize::new(budget).unwrap(),
    }
}

#[test]
fn driver_bounds_lanes_and_hints_and_visits_each_item_once() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    for (batch, budget, dim, lanes, hint) in [
        (0, 4, Some(1), 0, 0),
        (1, 4, Some(1), 1, 4),
        (12, 1, Some(1), 1, 1),
        (12, 2, Some(1), 2, 1),
        (12, 3, Some(1), 3, 1),
        (12, 4, Some(1), 4, 1),
        (16, 99, Some(1), 8, 1),
        (3, 4, Some(1), 1, 4),
        (5, 4, Some(1), 3, 1),
        (8, 4, Some(64), 4, 1),
        (8, 4, Some(65), 1, 4),
        (8, 4, None, 4, 1),
    ] {
        let mut values = vec![99];
        let scratch = AtomicUsize::new(0);
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let seen: Vec<_> = (0..batch).map(|_| AtomicUsize::new(0)).collect();
        let caller = std::thread::current().id();
        LAST_LANES.set(0);
        run(
            Op::Svd,
            batch,
            token(&pool, budget),
            dim,
            &mut (out(&mut values, 1),),
            |par| {
                scratch.fetch_add(1, SeqCst);
                assert_eq!(par.degree(), hint);
            },
            |index, (values,), (), par| {
                assert_eq!(par.degree(), hint);
                if budget == 1 {
                    assert_eq!(std::thread::current().id(), caller);
                } else {
                    assert!(
                        pool.current_thread_index().is_some(),
                        "wrong numerical pool"
                    );
                }
                let n = active.fetch_add(1, SeqCst) + 1;
                peak.fetch_max(n, SeqCst);
                if lanes > 1 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                seen[index].fetch_add(1, SeqCst);
                values.push(index);
                active.fetch_sub(1, SeqCst);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            LAST_LANES.get(),
            lanes,
            "batch={batch}, budget={budget}, dim={dim:?}"
        );
        assert_eq!(scratch.load(SeqCst), lanes);
        assert_eq!(values, (0..batch).collect::<Vec<_>>());
        assert!(seen.iter().all(|n| n.load(SeqCst) == 1));
        assert_eq!(active.load(SeqCst), 0);
        // This probes driver callback concurrency, not faer's internal numerical concurrency.
        assert!(peak.load(SeqCst) <= budget.min(8));
    }
}

#[test]
fn effective_one_stays_on_caller_and_nested_work_uses_only_selected_pool() {
    let selected = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let foreign = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let check = |par, expected_hint: usize, stays: bool| {
        let caller = std::thread::current().id();
        crate::with_parallel(par, |faer_par| {
            assert_eq!(faer_par.degree(), expected_hint);
            if stays {
                assert_eq!(std::thread::current().id(), caller);
            } else {
                assert!(selected.current_thread_index().is_some());
                assert!(foreign.current_thread_index().is_none());
            }
        });
    };
    check(token(&selected, 1), 1, true);
    check(token(&one, 4), 1, true);
    foreign.install(|| check(token(&selected, 1), 1, true));
    foreign.install(|| check(token(&selected, 3), 3, false));
    selected.install(|| check(token(&selected, 3), 3, false));
    foreign.install(|| {
        let mut values = Vec::new();
        run(
            Op::Qr,
            8,
            token(&selected, 2),
            Some(1),
            &mut (out(&mut values, 1),),
            |_| (),
            |index, (values,), (), par| {
                assert_eq!(par, faer::Par::Seq);
                assert!(selected.current_thread_index().is_some());
                assert!(foreign.current_thread_index().is_none());
                values.push(index);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(values, (0..8).collect::<Vec<_>>());
    });
    assert_eq!(
        token(&one, 99).budget().get(),
        99,
        "public accessor preserves requested ceiling"
    );
}

fn diagonal(n: usize, batch: usize) -> Vec<f64> {
    let mut data = vec![0.0; n * n * batch];
    for item in data.chunks_exact_mut(n * n) {
        for i in 0..n {
            item[i + i * n] = 1.0;
        }
    }
    data
}

#[test]
fn real_solve_routes_include_rhs_extent_even_without_a_source() {
    use crate::triangular_solve::{triangular_solve, TriangularSolveFlags};
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let batch = 4;
    let n = 2;
    let a = diagonal(n, batch);
    let adims = [n, n, batch];
    let astrides = [1, n as isize, (n * n) as isize];
    for width in [64, 65] {
        for left in [false, true] {
            let (rows, cols) = if left { (n, width) } else { (width, n) };
            let b = vec![3.0; rows * cols * batch];
            let dims = [rows, cols, batch];
            let strides = [1, rows as isize, (rows * cols) as isize];
            let mut x = Vec::new();
            triangular_solve(
                Op::Eig,
                RawStridedRef::new(&a, &adims, &astrides, 0).unwrap(),
                RawStridedRef::new(&b, &dims, &strides, 0).unwrap(),
                TriangularSolveFlags {
                    left_side: left,
                    lower: true,
                    transpose_a: false,
                    unit_diagonal: false,
                },
                &mut x,
                token(&pool, 2),
            )
            .unwrap();
            assert_eq!(LAST_LANES.get(), if width == 64 { 2 } else { 1 });
            assert_eq!(x, b);
        }
        let dims = [n, width, batch];
        let strides = [1, n as isize, (n * width) as isize];
        let mut x = vec![3.0; n * width * batch];
        crate::lu::solve(
            Op::Solve,
            RawStridedRef::new(&a, &adims, &astrides, 0).unwrap(),
            None,
            RawStridedMut::new(&mut x, &dims, &strides, 0).unwrap(),
            false,
            token(&pool, 2),
        )
        .unwrap();
        assert_eq!(LAST_LANES.get(), if width == 64 { 2 } else { 1 });
        assert!(x.iter().all(|&v| v == 3.0));
    }
}

#[test]
fn real_reflector_routes_include_both_state_and_target_extent() {
    use crate::householder::{apply_reflectors, ReflectorShape};
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for width in [64, 65] {
        for (a_cols, cols) in [(width, 1), (1, width)] {
            let batch = 4;
            let rows = 2;
            let mut a = vec![0.0; rows * a_cols * batch];
            for item in a.chunks_exact_mut(rows * a_cols) {
                item[0] = 1.0;
            }
            let mut c = vec![3.0; rows * cols * batch];
            apply_reflectors(
                Op::HouseholderQrQColumns,
                ReflectorShape {
                    rows,
                    a_cols,
                    cols,
                    k: 1,
                },
                batch,
                &a,
                &[2.0; 4],
                &mut c,
                false,
                token(&pool, 2),
            )
            .unwrap();
            assert_eq!(LAST_LANES.get(), if width == 64 { 2 } else { 1 });
            for (index, &value) in c.iter().enumerate() {
                assert_eq!(value, if index % 2 == 0 { -3.0 } else { 3.0 });
            }
        }
    }
}

#[test]
fn all_real_packed_routes_keep_the_large_item_exception() {
    use crate::packed_lu::{factor, factor_solve, solve_prepared};
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let (n, batch, nrhs) = (65, 4, 2);
    let mut lu = diagonal(n, batch);
    let mut pivots = vec![0; n * batch];
    let mut parity = vec![0.0; batch];
    factor(
        Op::LuFactor,
        n,
        n,
        &mut lu,
        &mut pivots,
        &mut parity,
        token(&pool, 2),
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 2);
    assert_eq!(lu, diagonal(n, batch));
    assert_eq!(parity, vec![1.0; batch]);
    let dims = [n, n, batch];
    let strides = [1, n as isize, (n * n) as isize];
    let pdims = [n, batch];
    let pstrides = [1, n as isize];
    let mut rhs = vec![3.0; n * nrhs * batch];
    solve_prepared(
        Op::LuSolvePrepared,
        RawStridedRef::new(&lu, &dims, &strides, 0).unwrap(),
        RawStridedRef::new(&pivots, &pdims, &pstrides, 0).unwrap(),
        nrhs,
        &mut rhs,
        false,
        false,
        token(&pool, 2),
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 2);
    assert!(rhs.iter().all(|&v| v == 3.0));
    let mut lu = diagonal(n, batch);
    factor_solve(
        Op::LuFactorSolve,
        n,
        nrhs,
        &mut lu,
        &mut pivots,
        &mut rhs,
        token(&pool, 2),
    )
    .unwrap();
    assert_eq!(LAST_LANES.get(), 2);
    assert!(rhs.iter().all(|&v| v == 3.0));
}
