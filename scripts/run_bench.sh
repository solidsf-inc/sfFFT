#!/usr/bin/env bash
# Full sweep (L = 128 .. 8192), batch 8 x 768 channels, then the plot.
# Usage: scripts/run_bench.sh [B] [H] [only_L]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x ./sffft ] || make sffft
out=results/$(hostname -s)_$(date +%Y%m%d_%H%M%S).csv
./sffft "${1:-8}" "${2:-768}" "${3:-0}" | tee "$out"
echo "wrote $out"
if python3 -c "import matplotlib, numpy" 2>/dev/null; then
  python3 scripts/plot.py "$out" docs/benchmark_local.png "${1:-8}" "${2:-768}" && echo "wrote docs/benchmark_local.png"
fi
