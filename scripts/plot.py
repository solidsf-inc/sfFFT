"""Plot a results CSV from ./sffft: throughput vs L and speedup over cuFFT.

Usage: python3 scripts/plot.py results/gb10_dgx_spark.csv docs/benchmark.png [B H]
"""
import collections, csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

src = sys.argv[1] if len(sys.argv) > 1 else "results/gb10_dgx_spark.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/benchmark.png"
B = int(sys.argv[3]) if len(sys.argv) > 3 else 8
H = int(sys.argv[4]) if len(sys.argv) > 4 else 768

best = collections.defaultdict(dict)   # method -> L -> best ms over plans
gpu = ""
for r in csv.reader(open(src)):
    if r and r[0].startswith("# ") and not gpu and "RESULT" not in r[0]:
        gpu = r[0][2:].split(" sm_")[0]
    if not r or r[0] != "RESULT":
        continue
    L, m, ms = int(r[1]), r[2], float(r[4])
    best[m][L] = min(ms, best[m].get(L, 1e30))

Ls = sorted(best["cufft_pipeline_fp32"])
col = lambda m: np.array([best[m][L] for L in Ls])
pipe, core = col("cufft_pipeline_fp32"), col("cufft_core_fp32")
f32, b16 = col("fused_fp32io"), col("fused_bf16io")
tput = lambda ms: B * H * np.array(Ls) / ms / 1e6   # G output samples / s

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), dpi=140)
a = ax[0]
a.plot(Ls, tput(pipe), "o-", c="#888", lw=2, label="cuFFT pipeline (pad + R2C + mul + C2R + slice)")
a.plot(Ls, tput(core), "s--", c="#bbb", lw=2, label="cuFFT core only (R2C + mul + C2R)")
a.plot(Ls, tput(f32), "o-", c="#1f5fbf", lw=2.5, label="fused tensor-core kernel, fp32 I/O")
a.plot(Ls, tput(b16), "D-", c="#e8590c", lw=2.5, label="fused tensor-core kernel, bf16 I/O")
a.set_xscale("log", base=2); a.set_yscale("log")
a.set_xticks(Ls); a.set_xticklabels(Ls)
a.yaxis.set_major_locator(FixedLocator([2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40]))
a.yaxis.set_major_formatter(ScalarFormatter()); a.yaxis.set_minor_formatter(NullFormatter())
a.set_xlabel("sequence length L (causal convolution, FFT size 2L)")
a.set_ylabel("throughput, G output samples / s (higher is better)")
a.set_title(f"Long causal convolution throughput{' on ' + gpu if gpu else ''}")
a.grid(True, which="both", alpha=.3); a.legend(fontsize=8.5, loc="lower left")

a = ax[1]; x = np.arange(len(Ls)); w = .27
for i, (y, lab, c) in enumerate([(pipe / f32, "fp32 I/O vs cuFFT pipeline", "#1f5fbf"),
                                 (pipe / b16, "bf16 I/O vs cuFFT pipeline", "#e8590c"),
                                 (core / f32, "fp32 I/O vs cuFFT core only", "#7fa7e0")]):
    bars = a.bar(x + (i - 1) * w, y, w, label=lab, color=c)
    a.bar_label(bars, fmt="%.1fx", fontsize=7.5, padding=1)
a.axhline(1, c="k", lw=.8)
a.set_xticks(x); a.set_xticklabels(Ls)
a.set_xlabel("sequence length L"); a.set_ylabel("speedup (x)")
a.set_title("Speedup of the fused four-step FFT-conv kernel")
a.legend(fontsize=8.5); a.grid(True, axis="y", alpha=.3)
fig.suptitle(f"Batch {B} x {H} channels = {B*H} sequences. Fused kernel: fp16 tensor cores, fp32 accumulate. "
             "Rel. error vs cuFFT fp32 output: ~4e-4 (fp32 I/O), ~2.4e-3 (bf16 I/O).", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(dst)
print("L, speedup_fp32_vs_pipeline, speedup_bf16_vs_pipeline, speedup_fp32_vs_core, fused_fp32_ms")
for L, p, c, a1, b1 in zip(Ls, pipe, core, f32, b16):
    print(L, f"{p/a1:.2f}", f"{p/b1:.2f}", f"{c/a1:.2f}", f"{a1:.4f}")
