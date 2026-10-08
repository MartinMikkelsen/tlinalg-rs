#!/usr/bin/env python3
"""Tests for idle_cpus.py against a synthetic /proc and /sys tree.

Adapted from `tensor4all/tprims-rs` `benchmarks/scripts/test_idle_cpus.py`
(MIT, Copyright (c) 2026 Lukas Devos and tensor4all contributors), plus a case
for the sibling expansion this copy adds.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("idle_cpus", Path(__file__).with_name("idle_cpus.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def tree(root: Path, n: int, l3: int, smt: bool = False) -> None:
    for c in range(n):
        d = root / f"sys/devices/system/cpu/cpu{c}"
        (d / "cache/index3").mkdir(parents=True)
        lo = c // l3 * l3
        (d / "cache/index3/shared_cpu_list").write_text(f"{lo}-{lo + l3 - 1}\n")
        (d / "topology").mkdir()
        sib = f"{c - c % 2},{c - c % 2 + 1}" if smt else str(c)
        (d / "topology/thread_siblings_list").write_text(sib + "\n")


def stat(root: Path, busy: dict[int, int], n: int, tick: int) -> None:
    lines = ["cpu 0 0 0 0 0 0 0 0"]
    for c in range(n):
        b = busy.get(c, 0) * tick
        lines.append(f"cpu{c} {b} 0 0 {100 * tick - b} 0 0 0 0 0 0")
    (root / "proc").mkdir(exist_ok=True)
    (root / "proc/stat").write_text("\n".join(lines) + "\n")


def run(root: Path, argv: list[str], busy: dict[int, int], n: int) -> int:
    state = {"tick": 1}
    stat(root, busy, n, state["tick"])

    def sleep(_s: float) -> None:
        state["tick"] = 2
        stat(root, busy, n, state["tick"])

    return m.main(argv, root=root, sleep=sleep)


def test_parse_cpu_list() -> None:
    assert m.parse_cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]


def test_pick_skips_busy_cores_and_stays_in_one_l3(capsys=None) -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        tree(root, 16, 8)
        # Domain 0 has two busy cores (50%), domain 1 is idle.
        rc = run(root, ["pick", "4", "--seconds", "0"], {1: 50, 2: 50}, 16)
        assert rc == 0
        frac = m.busy_fractions(root, 0, sleep=lambda s: None)
        assert m.pick(root, 4, {**frac, 1: 0.5, 2: 0.5}, 0.05) == [8, 9, 10, 11]


def test_pick_fails_when_no_domain_is_idle_enough() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        tree(root, 8, 8)
        assert run(root, ["pick", "8", "--seconds", "0"], {3: 90}, 8) == 2


def test_pick_uses_one_thread_per_physical_core() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        tree(root, 8, 8, smt=True)
        assert m.pick(root, 4, {c: 0.0 for c in range(8)}, 0.05) == [0, 2, 4, 6]
        assert m.pick(root, 5, {c: 0.0 for c in range(8)}, 0.05) is None


def test_check_reports_busy_cpus() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        tree(root, 8, 8)
        assert run(root, ["check", "0-3", "--seconds", "0"], {}, 8) == 0
        assert run(root, ["check", "0-3", "--seconds", "0"], {2: 40}, 8) == 1


def test_check_sees_a_busy_sibling_of_an_idle_core() -> None:
    """A busy SMT sibling contends with the physical core that is being measured."""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        tree(root, 8, 8, smt=True)
        assert m.with_siblings(root, [0, 2]) == [0, 1, 2, 3]
        assert run(root, ["check", "0", "--seconds", "0"], {}, 8) == 0
        assert run(root, ["check", "0", "--seconds", "0"], {1: 60}, 8) == 1


def test_missing_proc_stat_means_pinning_unavailable() -> None:
    with tempfile.TemporaryDirectory() as t:
        assert m.main(["check", "0", "--seconds", "0"], root=Path(t)) == 3


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in tests:
        f()
    print(f"test_idle_cpus: {len(tests)} tests passed")
