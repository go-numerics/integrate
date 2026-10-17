#!/bin/sh
# Interleaves Go and SciPy timings of the same problems, ROUNDS times, and
# prints the median of each side per problem. The host is shared: run it when
# `uptime` shows a low load, and quote the load with the numbers.
#
# Usage: [ROUNDS=5] bench/compare.sh [name...]   (no names: every problem)
# Go runs on one thread (-cpu 1), as SciPy does; SciPy and NumPy are pinned.
set -eu
ROUNDS=${ROUNDS:-5}
out=$(mktemp -d)
cd "$(dirname "$0")/.."
pattern=.
if [ $# -gt 0 ]; then
  pattern="^($(echo "$@" | tr ' ' '|'))\$"
fi
GOWORK=off go test -c -o "$out/integrate.test" .
uptime
for r in $(seq 1 "$ROUNDS"); do
  "$out/integrate.test" -test.run '^$' -test.bench "$pattern" -test.benchtime 1s -test.count 1 -test.cpu 1 |
    awk '/^Benchmark/ { sub(/^Benchmark/, "", $1); sub(/-[0-9]+$/, "", $1); print $1, $3 }' >> "$out/go.txt"
  uv run -q --with scipy==1.18.1 --with numpy==2.5.4 python bench/scipy_bench.py "$@" >> "$out/py.txt"
done
uptime
python3 - "$out" <<'PY'
import collections, statistics, sys
d = sys.argv[1]
def load(f, col):
    m = collections.defaultdict(list)
    for line in open(f):
        parts = line.split()
        m[parts[0]].append(float(parts[col]))
    return m
g, p, cb = load(d + "/go.txt", 1), load(d + "/py.txt", 1), load(d + "/py.txt", 2)
print("| problem | Go median | SciPy median | SciPy / Go | of which Python callbacks | SciPy without callbacks / Go |")
print("|---|---:|---:|---:|---:|---:|")
def fmt(ns):
    if ns < 1e3:
        return f"{ns:.0f} ns"
    return f"{ns/1e3:.1f} µs" if ns < 1e6 else f"{ns/1e6:.2f} ms"
for k in g:
    gm, pm, cm = statistics.median(g[k]), statistics.median(p[k]), statistics.median(cb[k])
    def ratio(r):
        return f"{r:.0f}×" if r >= 10 else f"{r:.2g}×"
    print(f"| {k} | {fmt(gm)} | {fmt(pm)} | {ratio(pm/gm)} | {fmt(cm)} | {ratio((pm-cm)/gm)} |")
PY
