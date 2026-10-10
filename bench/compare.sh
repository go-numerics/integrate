#!/bin/sh
# Interleaves Go and SciPy timings of the same problems, ROUNDS times, and
# prints the median of each side per problem. The host is shared: run it when
# `uptime` shows a low load, and quote the load with the numbers.
set -eu
ROUNDS=${ROUNDS:-5}
out=$(mktemp -d)
cd "$(dirname "$0")/.."
uptime
for r in $(seq 1 "$ROUNDS"); do
  go test -run '^$' -bench . -benchtime 1s -count 1 . |
    awk '/^Benchmark/ { sub(/^Benchmark/, "", $1); sub(/-[0-9]+$/, "", $1); print $1, $3 }' >> "$out/go.txt"
  uv run -q --with scipy --with numpy python bench/scipy_bench.py >> "$out/py.txt"
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
    return f"{ns/1e3:.1f} µs" if ns < 1e6 else f"{ns/1e6:.2f} ms"
for k in g:
    gm, pm, cm = statistics.median(g[k]), statistics.median(p[k]), statistics.median(cb[k])
    print(f"| {k} | {fmt(gm)} | {fmt(pm)} | {pm/gm:.0f}× | {fmt(cm)} | {(pm-cm)/gm:.0f}× |")
PY
