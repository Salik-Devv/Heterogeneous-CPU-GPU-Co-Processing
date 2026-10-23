import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── load data ─────────────────────────────────────────────────────────────────
CSV = "../results/gemm_results(kaggle).csv"   
df  = pd.read_csv(CSV)

# ── colour / style palette ────────────────────────────────────────────────────
IMPL_ORDER  = ["CPU_Serial", "CPU_OMP", "GPU_Naive", "GPU_Tiled", "GPU_cuBLAS"]
IMPL_LABELS = ["CPU Serial", "CPU OMP", "GPU Naive", "GPU Tiled", "GPU cuBLAS"]
IMPL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

N_VALUES    = [512, 1024, 2048]
N_COLORS    = {"512": "#E377C2", "1024": "#7F7F7F", "2048": "#17BECF"}
N_MARKERS   = {"512": "o", "1024": "s", "2048": "^"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — GFLOPS vs N  (bar chart, log-y)
# ══════════════════════════════════════════════════════════════════════════════

# aggregate: mean ± std per (implementation, N)
base = df[df["implementation"].isin(IMPL_ORDER)]
stats = (base
         .groupby(["implementation", "N"])["gflops"]
         .agg(mean="mean", std="std")
         .reset_index())

n_groups  = len(N_VALUES)
n_impls   = len(IMPL_ORDER)
bar_width = 0.15
group_gap = 0.05
x_centres = np.arange(n_groups)          # one tick per N

for i, (impl, label, color) in enumerate(zip(IMPL_ORDER, IMPL_LABELS, IMPL_COLORS)):
    offset = (i - n_impls / 2 + 0.5) * (bar_width + group_gap)
    sub    = stats[stats["implementation"] == impl].set_index("N")

    means  = [sub.loc[n, "mean"] if n in sub.index else 0 for n in N_VALUES]
    stds   = [sub.loc[n, "std"]  if n in sub.index else 0 for n in N_VALUES]
    stds   = [s if not np.isnan(s) else 0 for s in stds]   # single-run → 0

    ax1.bar(
        x_centres + offset, means,
        width=bar_width, label=label, color=color, alpha=0.85,
        yerr=stds, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )

ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f"{v:g}"
))
ax1.set_xticks(x_centres)
ax1.set_xticklabels([f"N = {n}" for n in N_VALUES], fontsize=12)
ax1.set_xlabel("Matrix size  N × N", fontsize=13)
ax1.set_ylabel("GFLOPS  (log scale)", fontsize=13)
ax1.set_title("GEMM Performance: GFLOPS vs N", fontsize=14, fontweight="bold")
ax1.legend(title="Implementation", fontsize=10, title_fontsize=10,
           loc="upper left", framealpha=0.9)
ax1.grid(axis="y", which="both", linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

# annotate peak values
for i, (impl, color) in enumerate(zip(IMPL_ORDER, IMPL_COLORS)):
    offset = (i - n_impls / 2 + 0.5) * (bar_width + group_gap)
    sub    = stats[stats["implementation"] == impl].set_index("N")
    for j, n in enumerate(N_VALUES):
        if n in sub.index:
            val = sub.loc[n, "mean"]
            if val > 300:           # only annotate tall bars to avoid clutter
                ax1.text(
                    j + offset, val * 1.12,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold",
                )


print("Saved  →  gemm_gflops_vs_N.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Hetero: wall-time vs CPU fraction  (line chart)
# ══════════════════════════════════════════════════════════════════════════════

hetero = df[df["implementation"] == "CPU_GPU_Hetero"].copy()

# aggregate per (N, cpu_fraction)
ht = (hetero
      .groupby(["N", "cpu_fraction"])["wall_total_ms"]
      .agg(mean="mean", std="std")
      .reset_index())
ht["std"] = ht["std"].fillna(0)

for n in N_VALUES:
    sub   = ht[ht["N"] == n].sort_values("cpu_fraction")
    fracs = sub["cpu_fraction"].values
    means = sub["mean"].values
    stds  = sub["std"].values
    color = N_COLORS[str(n)]
    mk    = N_MARKERS[str(n)]

    ax2.plot(fracs, means, marker=mk, linewidth=2.0,
             markersize=7, color=color, label=f"N = {n}")
    ax2.fill_between(fracs, means - stds, means + stds,
                     alpha=0.18, color=color)

ax2.set_xlabel("CPU fraction  (0 = GPU only,  1 = CPU only)", fontsize=13)
ax2.set_ylabel("Wall time  (ms)", fontsize=13)
ax2.set_title("Hetero GEMM: Wall Time vs CPU–GPU Split Ratio",
              fontsize=14, fontweight="bold")
ax2.set_xticks(np.arange(0, 1.01, 0.1))
ax2.set_xticklabels([f"{x:.1f}" for x in np.arange(0, 1.01, 0.1)], fontsize=10)
ax2.legend(title="Matrix size", fontsize=11, title_fontsize=11,
           loc="upper left", framealpha=0.9)
ax2.grid(linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

# mark optimal (minimum mean) for each N
for n in N_VALUES:
    sub      = ht[ht["N"] == n].sort_values("cpu_fraction")
    best_idx = sub["mean"].idxmin()
    best_row = sub.loc[best_idx]
    color    = N_COLORS[str(n)]
    ax2.axvline(best_row["cpu_fraction"], color=color,
                linestyle=":", linewidth=1.3, alpha=0.7)
    ax2.annotate(
        f"min @ {best_row['cpu_fraction']:.1f}\n({best_row['mean']:.1f} ms)",
        xy=(best_row["cpu_fraction"], best_row["mean"]),
        xytext=(best_row["cpu_fraction"] + 0.04, best_row["mean"] * 1.07),
        fontsize=8, color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
    )

fig.tight_layout()
fig.savefig("combined_gemm_plots.png", dpi=150)
print("Saved  →  gemm_hetero_walltime.png")

plt.show()
