#!/usr/bin/env python3
"""Pick idle CPUs inside one L3 domain, or check that given CPUs are idle.

Adapted from `tensor4all/tprims-rs` `benchmarks/scripts/idle_cpus.py`
(MIT, Copyright (c) 2026 Lukas Devos and tensor4all contributors). The one
deliberate difference is in `check`: it expands the given CPUs with their SMT
siblings, because a busy sibling contends with the physical core that is being
measured. Upstream checked only the listed logical CPUs.

PERFORMANCE_TIPS.md (Performance-Sensitive Tests And Benchmarks) requires the
measured process to be pinned to cores of one L3 domain that are idle
immediately before and after each measurement; "idle" is a `/proc/stat` busy
fraction over a few seconds. This script implements that observation on
Linux with the standard library only.

    idle_cpus.py pick N [--seconds S] [--max-busy F]   # prints e.g. 8,9,10,11
    idle_cpus.py check CPUS [--seconds S] [--max-busy F]

`pick` exits 2 when no L3 domain has N idle CPUs; `check` exits 1 when a CPU
or one of its siblings is busy. Both exit 3 off Linux (CPU affinity is
Linux-only): record that pinning was unavailable instead.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

DEFAULT_SECONDS = 3.0
DEFAULT_MAX_BUSY = 0.05


def parse_cpu_list(text: str) -> list[int]:
    """`0-3,8,10-11` -> [0, 1, 2, 3, 8, 10, 11]."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def read_times(root: Path) -> dict[int, tuple[int, int]]:
    """Per-CPU (busy, total) jiffies from `proc/stat`."""
    out: dict[int, tuple[int, int]] = {}
    for line in (root / "proc/stat").read_text().splitlines():
        head, *fields = line.split()
        if not head.startswith("cpu") or head == "cpu":
            continue
        v = [int(x) for x in fields]
        idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
        total = sum(v[:8])  # guest time is already inside user/nice
        out[int(head[3:])] = (total - idle, total)
    return out


def busy_fractions(root: Path, seconds: float, sleep=time.sleep) -> dict[int, float]:
    a = read_times(root)
    sleep(seconds)
    b = read_times(root)
    frac: dict[int, float] = {}
    for cpu, (busy1, tot1) in b.items():
        busy0, tot0 = a.get(cpu, (busy1, tot1))
        dt = tot1 - tot0
        frac[cpu] = (busy1 - busy0) / dt if dt > 0 else 0.0
    return frac


def l3_domains(root: Path, cpus: list[int]) -> list[list[int]]:
    """Distinct L3 domains (cache index3 shared_cpu_list), lowest CPU first."""
    seen: dict[tuple[int, ...], None] = {}
    for cpu in cpus:
        p = root / f"sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"
        dom = tuple(parse_cpu_list(p.read_text())) if p.exists() else (cpu,)
        seen.setdefault(dom, None)
    return sorted((list(d) for d in seen), key=lambda d: d[0])


def siblings(root: Path, cpu: int) -> list[int]:
    """The hardware threads sharing `cpu`'s physical core, `cpu` itself included."""
    p = root / f"sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    return parse_cpu_list(p.read_text()) if p.exists() else [cpu]


def with_siblings(root: Path, cpus: list[int]) -> list[int]:
    """`cpus` expanded with every sibling, so a busy sibling is not invisible."""
    out: set[int] = set()
    for cpu in cpus:
        out.update(siblings(root, cpu))
    return sorted(out)


def primary_threads(root: Path, cpus: list[int]) -> list[int]:
    """One hardware thread per physical core (the lowest sibling)."""
    keep = []
    for cpu in cpus:
        p = root / f"sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        sib = parse_cpu_list(p.read_text()) if p.exists() else [cpu]
        if cpu == min(sib):
            keep.append(cpu)
    return keep


def pick(root: Path, n: int, frac: dict[int, float], max_busy: float) -> list[int] | None:
    """The first n idle primary CPUs of the L3 domain with the most idle ones."""
    best: list[int] | None = None
    for dom in l3_domains(root, sorted(frac)):
        idle = [c for c in primary_threads(root, dom) if frac.get(c, 1.0) <= max_busy]
        if len(idle) >= n and (best is None or len(idle) > len(best)):
            best = idle
    return None if best is None else best[:n]


def main(argv: list[str] | None = None, root: Path = Path("/"), sleep=time.sleep) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pick", "check"):
        p = sub.add_parser(name)
        p.add_argument("arg", help="number of CPUs (pick) or a CPU list (check)")
        p.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
        p.add_argument("--max-busy", type=float, default=DEFAULT_MAX_BUSY)
    a = ap.parse_args(argv)
    if not (root / "proc/stat").exists() or (root == Path("/") and not sys.platform.startswith("linux")):
        print("idle_cpus: not Linux; CPU pinning is unavailable on this host", file=sys.stderr)
        return 3
    frac = busy_fractions(root, a.seconds, sleep)
    if a.cmd == "pick":
        got = pick(root, int(a.arg), frac, a.max_busy)
        if got is None:
            print(f"idle_cpus: no L3 domain has {a.arg} idle CPUs (max busy {a.max_busy})", file=sys.stderr)
            return 2
        print(",".join(map(str, got)))
        return 0
    requested = parse_cpu_list(a.arg)
    cpus = with_siblings(root, requested)
    busy = {c: frac.get(c, 1.0) for c in cpus if frac.get(c, 1.0) > a.max_busy}
    for c in cpus:
        mark = "" if c in requested else " (sibling)"
        print(f"cpu{c} busy={frac.get(c, 1.0):.3f}{mark}", file=sys.stderr)
    if busy:
        print(f"idle_cpus: busy CPUs {sorted(busy)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
