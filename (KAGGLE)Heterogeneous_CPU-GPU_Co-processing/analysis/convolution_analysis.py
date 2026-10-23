import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("../results/conv_results(kaggle).csv")
df.columns = df.columns.str.strip()

# ── Filter only Hetero split-ratio experiments ─────────────────────────────────
hetero = df[df["implementation"] == "CPU_GPU_Hetero"].copy()

# Group by cpu_fraction → mean & std of compute_time_ms
grouped = (
    hetero.groupby("cpu_fraction")["compute_time_ms"]
    .agg(mean="mean", std="std", count="count")
    .reset_index()
)

# GPU idle time = max(cpu_part_ms, gpu_kernel_ms) - min(...) -- i.e. the slower
# leg has to wait for the faster one.  The bottleneck column tells us who waits.
# GPU idle = how long GPU sits idle = max(0, cpu_part_ms - gpu_kernel_ms)
hetero["gpu_idle_ms"] = np.maximum(0, hetero["cpu_part_ms"] - hetero["gpu_kernel_ms"])

idle_grouped = (
    hetero.groupby("cpu_fraction")["gpu_idle_ms"]
    .agg(mean="mean", std="std")
    .reset_index()
)

# ── Shared style ───────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
GRAY   = "#6B7280"
LIGHT  = "#DBEAFE"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Convolution: Hetero Split-Ratio Analysis  (5824 × 3264, n = 7 runs/point)",
             fontsize=13, fontweight="bold", y=1.01)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Line chart: CPU fraction vs Compute wall time
# ══════════════════════════════════════════════════════════════════════════════
ax1 = axes[0]

x  = grouped["cpu_fraction"].values
y  = grouped["mean"].values
sd = grouped["std"].fillna(0).values

ax1.fill_between(x, y - sd, y + sd, color=LIGHT, alpha=0.7, label="±1 SD")
ax1.plot(x, y, color=BLUE, linewidth=2.5, marker="o", markersize=7,
         markerfacecolor="white", markeredgewidth=2, label="Mean compute time")

# Annotate min point (GPU-only, cpu_fraction = 0)
min_idx = np.argmin(y)
ax1.annotate(
    f"  GPU-only\n  {y[min_idx]:.1f} ms",
    xy=(x[min_idx], y[min_idx]),
    xytext=(x[min_idx] + 0.07, y[min_idx] + 15),
    fontsize=8.5, color=BLUE,
    arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2),
)

ax1.set_xlabel("CPU Fraction", fontsize=11)
ax1.set_ylabel("Compute Wall Time (ms)", fontsize=11)
ax1.set_title("(a)Compute Time vs Split Ratio", fontsize=11, pad=8)
ax1.set_xlim(-0.02, 1.02)
ax1.set_xticks(np.arange(0, 1.1, 0.1))
ax1.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 1.1, 0.1)], fontsize=9)
ax1.legend(fontsize=9, framealpha=0.9)

# Trend annotation
ax1.text(0.55, 0.18, "GPU dominates —\nany CPU involvement hurts",
         transform=ax1.transAxes, fontsize=8.5, color=GRAY,
         ha="center", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.8))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Bar chart: CPU fraction vs GPU idle time
# ══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]

xi   = idle_grouped["cpu_fraction"].values
yi   = idle_grouped["mean"].values
sdi  = idle_grouped["std"].fillna(0).values

bar_width = 0.07
bars = ax2.bar(xi, yi, width=bar_width, color=RED, alpha=0.82,
               yerr=sdi, capsize=4, error_kw=dict(elinewidth=1.2, ecolor="#7F1D1D"),
               label="Mean GPU idle time")

# Colour gradient: darker = higher idle
norm_yi = yi / (yi.max() + 1e-9)
for bar, nv in zip(bars, norm_yi):
    bar.set_alpha(0.45 + 0.55 * nv)

ax2.set_xlabel("CPU Fraction", fontsize=11)
ax2.set_ylabel("GPU Idle Time (ms)", fontsize=11)
ax2.set_title("(b) GPU Idle Time vs Split Ratio", fontsize=11, pad=8)
ax2.set_xlim(-0.06, 1.06)
ax2.set_xticks(xi)
ax2.set_xticklabels([f"{v:.1f}" for v in xi], fontsize=9, rotation=45)
ax2.legend(fontsize=9, framealpha=0.9)

# Annotate worst offender
max_idx = np.argmax(yi)
ax2.annotate(
    f"Peak idle\n{yi[max_idx]:.1f} ms",
    xy=(xi[max_idx], yi[max_idx]),
    xytext=(xi[max_idx] + 0.12, yi[max_idx] * 0.85),
    fontsize=8.5, color=RED,
    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
)

ax2.text(0.5, 0.88,
         "GPU sits idle waiting\nfor CPU to finish its rows",
         transform=ax2.transAxes, fontsize=8.5, color=GRAY,
         ha="center", style="italic",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, lw=0.8))

# ── Save ───────────────────────────────────────────────────────────────────────
plt.tight_layout()
out_path = "convolution_figures.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close()
