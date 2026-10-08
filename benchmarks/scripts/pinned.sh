#!/usr/bin/env bash
# Run one measurement pinned to CPUS, valid only if CPUS were idle right
# before and right after it (PERFORMANCE_TIPS.md, Performance-Sensitive Tests
# And Benchmarks). The command's stdout is printed only for a valid run; a run
# spoiled by other load is retried up to PINNED_RETRIES times (default 3),
# waiting PINNED_WAIT seconds (default 30) for the cores to become idle.
#
#   pinned.sh CPUS -- COMMAND [ARGS...]
#
# Exits 75 when no valid run was obtained. CPU affinity is Linux-only: on
# other hosts the command runs unpinned and a note goes to stderr.
#
# Adapted from `tensor4all/tprims-rs` `benchmarks/scripts/pinned.sh`
# (MIT, Copyright (c) 2026 Lukas Devos and tensor4all contributors). The
# effective idle window, busy threshold and retry budget are echoed to stderr,
# so the guard log of a run records the policy that was actually in force
# rather than the one the manifest assumes.
set -euo pipefail
cpus=$1; shift
[[ ${1:-} == -- ]] && shift
here=$(cd "$(dirname "$0")" && pwd)
seconds=${PINNED_IDLE_SECONDS:-3}
max_busy=${PINNED_MAX_BUSY:-0.05}
retries=${PINNED_RETRIES:-3}
wait=${PINNED_WAIT:-30}
check() { python3 "$here/idle_cpus.py" check "$cpus" --seconds "$seconds" --max-busy "$max_busy" 2>/dev/null; }
if [[ $(uname -s) != Linux ]] || [[ ! -r /proc/stat ]] || ! command -v taskset >/dev/null; then
    echo "pinned.sh: CPU pinning unavailable on this host (Linux only); running unpinned" >&2
    exec "$@"
fi
echo "pinned.sh: cpus=$cpus idle_window=${seconds}s max_busy=$max_busy retries=$retries" >&2
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
for attempt in $(seq 1 "$retries"); do
    if ! check; then
        echo "pinned.sh: cpus $cpus busy before run (attempt $attempt); waiting" >&2
        sleep "$wait"; continue
    fi
    taskset -c "$cpus" "$@" > "$tmp"
    if check; then
        cat "$tmp"
        echo "pinned.sh: cpus $cpus idle before and after (attempt $attempt)" >&2
        exit 0
    fi
    echo "pinned.sh: cpus $cpus busy after run (attempt $attempt); discarding it" >&2
done
echo "pinned.sh: no valid run on cpus $cpus" >&2
exit 75
