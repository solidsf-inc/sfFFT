#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
scratch_root="${SOLIDSF_SCRATCH:-$HOME/solidsf-scratch}"
mkdir -p "$scratch_root/staging"
work=$(mktemp -d "$scratch_root/staging/sffft-precision-test.XXXXXX")
trap 'rm -rf "$work"' EXIT

check_exit() {
  local expected=$1 actual
  shift
  if ./sffft "$@" > "$work/check.csv" 2>&1; then actual=0; else actual=$?; fi
  if [ "$actual" != "$expected" ]; then
    cat "$work/check.csv"
    printf 'FAIL: expected exit %s, got %s: %s\n' "$expected" "$actual" "$*"
    exit 1
  fi
}

check_exit 0 --help
for value in nan inf 0 -1 1e-999 garbage; do check_exit 1 --max-error "$value"; done
check_exit 1 --max-error
check_exit 1 --io unknown
check_exit 1 --seed 0
check_exit 1 --seed -1
check_exit 1 --input-scale inf
check_exit 1 --input-scale 0
check_exit 1 --allocator unknown
check_exit 1 --allocator
check_exit 1 0 1 128
check_exit 1 1 0 128
check_exit 1 1 1 129
check_exit 1 2147483647 2 128
check_exit 2 1 1 128 --max-error 1e-8 --io fp32
grep -q '^NO_MATCH,128,fp32,' "$work/check.csv"
check_exit 2 1 1 128 --max-error 1e-3 --io bf16
grep -q '^NO_MATCH,128,bf16,' "$work/check.csv"

for seed in 42 12345; do
  ./sffft 1 1 --max-error 1e-3 --io fp32 --seed "$seed" > "$work/seed.csv"
  awk -F, '
    $1=="SELECT" { n++; if ($6+0 > $7+0 || $8!="fp32") bad++ }
    END { if(n!=7 || bad) exit 1 }
  ' "$work/seed.csv"
done

./sffft 1 1 128 --max-error 1e-3 --io fp32 --input-scale 15000 > "$work/range.csv"
awk -F, '
  $1=="RESULT" && $3~/fp16acc/ && $6~/nan|inf/ { rejected++ }
  $1=="SELECT" { n++; if ($3!="fused_fp32io" || $6+0 > $7+0) bad++ }
  END { if(n!=1 || bad || !rejected) exit 1 }
' "$work/range.csv"
check_exit 2 1 1 128 --max-error 1e-3 --io fp32 --input-scale 25000
grep -q '^NO_MATCH,128,fp32,' "$work/check.csv"
printf 'PASS: CLI validation, unattainable targets, I/O constraints, two seeds, and FP16 overflow rejection\n'

for allocator in device managed mapped; do
  ./sffft 1 1 --max-error 1e-3 --io fp32 --allocator "$allocator" > "$work/allocator.csv"
  awk -F, -v allocator="$allocator" '
    $1=="SELECT" { n++; if ($6+0 > $7+0 || $8!="fp32") bad++ }
    $1=="MEMORY" { m++; if ($3!=allocator || (allocator!="device" && ($7+0!=0 || $8+0!=0 || $5+0!=0))) bad++ }
    END { if(n!=7 || m!=7 || bad) exit 1 }
  ' "$work/allocator.csv"
done
printf 'PASS: device, managed, and mapped allocations preserve error budgets and shared CPU views avoid explicit copies\n'
