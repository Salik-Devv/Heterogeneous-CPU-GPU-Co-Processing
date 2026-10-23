import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("../results/prefix_results(kaggle).csv")
df.columns = df.columns.str.strip()

# ── Shared style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.color"       : "#E5E7EB",
    "grid.linewidth"   : 0.8,
    "axes.axisbelow"   : True,
})

SIZES   = [65536, 1048576, 4194304, 16777216, 67108864, 134217728]
LABELS  = ["64K", "1M", "4M", "16M", "64M", "128M"]
N_SIZES = len(SIZES)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_SERIAL  = "#6B7280"   # grey
C_OMP8    = "#F59E0B"   # amber
C_OMP16   = "#10B981"   # emerald
C_GPU     = "#2563EB"   # blue
C_H2D     = "#93C5FD"   # light blue  (H2D transfer)
C_D2H     = "#1D4ED8"   # dark blue   (D2H transfer)
C_SYNC    = "#EF4444"   # red         (sync overhead)
C_GCPU    = "#A78BFA"   # violet      (GPU compute)
C_CCPU    = "#F97316"   # orange      (CPU compute)
C_LINES   = ["#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6","#6B7280"]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Bandwidth bar chart
# ══════════════════════════════════════════════════════════════════════════════
def make_fig5():
    # --- Gather stats ---------------------------------------------------------
    # CPU Serial (threads=1)
    ser = df[(df["implementation"]=="CPU_Serial") & (df["threads"]==1)]
    ser_g = ser.groupby("N")["bandwidth_gb_s"].agg(mean="mean", std="std").reindex(SIZES)

    # CPU-OMP 8T
    omp8 = df[(df["implementation"]=="CPU_OMP") & (df["threads"]==8)]
    omp8_g = omp8.groupby("N")["bandwidth_gb_s"].agg(mean="mean", std="std").reindex(SIZES)

    # CPU-OMP 16T
    omp16 = df[(df["implementation"]=="CPU_OMP") & (df["threads"]==16)]
    omp16_g = omp16.groupby("N")["bandwidth_gb_s"].agg(mean="mean", std="std").reindex(SIZES)

    # GPU — exclude obvious outlier (N=65536 cold start gives very low bw, keep; but N=16777216 has 6795ms outlier)
    gpu_all = df[df["implementation"]=="GPU"].copy()
    gpu_clean = gpu_all[gpu_all["gpu_kernel_ms"] < 200]  # drop the 6795ms runaway
    gpu_g = gpu_clean.groupby("N")["bandwidth_gb_s"].agg(mean="mean", std="std").reindex(SIZES)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(N_SIZES)
    w = 0.19
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    groups  = [
        (ser_g,  C_SERIAL, "CPU Serial"),
        (omp8_g, C_OMP8,   "CPU-OMP 8T"),
        (omp16_g,C_OMP16,  "CPU-OMP 16T"),
        (gpu_g,  C_GPU,    "GPU"),
    ]

    for (grp, col, lbl), off in zip(groups, offsets):
        means = grp["mean"].values.astype(float)
        stds  = grp["std"].fillna(0).values.astype(float)
        bars = ax.bar(x + off, means, w, color=col, alpha=0.88, label=lbl,
                      yerr=stds, capsize=3,
                      error_kw=dict(elinewidth=1.1, ecolor="#374151", capthick=1.1))

    # Annotate GPU peak
    gpu_peak_idx = int(np.nanargmax(gpu_g["mean"].values))
    ax.annotate(
        f"GPU peak\n{gpu_g['mean'].iloc[gpu_peak_idx]:.1f} GB/s",
        xy=(x[gpu_peak_idx] + offsets[3], gpu_g["mean"].iloc[gpu_peak_idx]),
        xytext=(x[gpu_peak_idx] + offsets[3] + 0.45,
                gpu_g["mean"].iloc[gpu_peak_idx] + 2),
        fontsize=8.5, color=C_GPU,
        arrowprops=dict(arrowstyle="-|>", color=C_GPU, lw=1.2),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_xlabel("Array Size N (elements)", fontsize=11)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=11)
    ax.set_title("Prefix Sum: GPU vs CPU Bandwidth", fontsize=12, pad=9)
    ax.legend(fontsize=9, framealpha=0.9, ncol=4, loc="upper left")

    ax.text(0.98, 0.96,
            "Memory-bound: roofline = PCIe bandwidth\n(not compute)",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
            color="#374151", style="italic",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#9CA3AF", lw=0.8))

    plt.tight_layout()
    fig.savefig("prefix_sum_bandwidth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Fig 5 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Stacked bar: sync overhead vs N
# ══════════════════════════════════════════════════════════════════════════════
def make_fig6(ax):
    hetero0 = df[(df["implementation"]=="CPU_GPU_Hetero") & (df["cpu_fraction"]==0.0)]
    g = hetero0.groupby("N")[["gpu_kernel_ms","cpu_part_ms","sync_overhead_ms","h2d_ms","d2h_ms"]].mean().reindex(SIZES)

    h2d   = g["h2d_ms"].values
    gpu_k = g["gpu_kernel_ms"].values
    sync  = g["sync_overhead_ms"].values
    d2h   = g["d2h_ms"].values

    x = np.arange(N_SIZES)
    w = 0.55

    b1 = ax.bar(x, h2d,   w, color=C_H2D,  alpha=0.9, label="H2D Transfer")
    b2 = ax.bar(x, gpu_k, w, bottom=h2d,              color=C_GCPU, alpha=0.9, label="GPU Compute")
    b3 = ax.bar(x, sync,  w, bottom=h2d+gpu_k,        color=C_SYNC, alpha=0.88, label="Sync Overhead")
    b4 = ax.bar(x, d2h,   w, bottom=h2d+gpu_k+sync,   color=C_D2H,  alpha=0.9, label="D2H Transfer")

    hetero0_std = hetero0.groupby("N")[["gpu_kernel_ms","sync_overhead_ms","h2d_ms","d2h_ms"]].std().reindex(SIZES)
    total_mean = h2d + gpu_k + sync + d2h
    total_std  = (hetero0_std["h2d_ms"]**2 + hetero0_std["gpu_kernel_ms"]**2 +
                  hetero0_std["sync_overhead_ms"]**2 + hetero0_std["d2h_ms"]**2).pow(0.5).values

    ax.errorbar(x, total_mean, yerr=total_std, fmt="none",
                ecolor="#111827", elinewidth=1.2, capsize=4, capthick=1.2)

    for i, n in enumerate(SIZES):
        pct = 100 * sync[i] / total_mean[i]
        if pct > 8:
            ax.text(x[i], total_mean[i] + total_std[i] + 0.8,
                    f"{pct:.0f}%\nsync", ha="center", va="bottom",
                    fontsize=7.5, color="#991B1B")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_xlabel("Array Size N (elements)", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("Prefix Sum: Sync Overhead vs N\n(Hetero GPU-only mode, cpu_fraction = 0)", fontsize=12, pad=9)

    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")

    ax.text(0.98, 0.96,
            "Serial dependency cost grows with N\n— unique bottleneck of prefix-sum hetero",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
            color="#374151", style="italic",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#9CA3AF", lw=0.8))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Hetero wall time vs split ratio
# ══════════════════════════════════════════════════════════════════════════════
def make_fig7(ax):
    hetero = df[df["implementation"]=="CPU_GPU_Hetero"]
    grp = hetero.groupby(["N","cpu_fraction"])["wall_total_ms"].agg(mean="mean", std="std").reset_index()

    for color, N in zip(C_LINES, SIZES):
        sub = grp[grp["N"]==N].sort_values("cpu_fraction")
        frac = sub["cpu_fraction"].values
        mean = sub["mean"].values
        std  = sub["std"].fillna(0).values
        label = LABELS[SIZES.index(N)]

        ax.fill_between(frac, mean-std, mean+std, color=color, alpha=0.12)

        ax.plot(frac, mean, color=color, linewidth=2.2,
                marker="o", markersize=5, markerfacecolor="white",
                markeredgewidth=1.8, label=label)

        min_i = np.argmin(mean)

        ax.plot(frac[min_i], mean[min_i],
                marker="*", markersize=11,
                color=color, zorder=5)

    ax.set_xlabel("CPU Fraction", fontsize=11)
    ax.set_ylabel("Wall Time (ms)", fontsize=11)
    ax.set_title("Prefix Sum: Hetero Wall Time vs Split Ratio", fontsize=12, pad=9)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0, 1.1, 0.1)], fontsize=9)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    star_patch = plt.Line2D([0],[0],
                             marker="*",
                             color="gray",
                             linestyle="None",
                             markersize=10,
                             label="Optimum split (★)")

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles + [star_patch],
              labels + ["Optimum split (★)"],
              title="Array Size N",
              fontsize=9,
              title_fontsize=9,
              framealpha=0.9,
              loc="upper left",
              ncol=2)

    ax.text(0.98, 0.97,
            "GPU-only (0%) best for large N\nSmall N: sync overhead dominates",
            transform=ax.transAxes,
            fontsize=8.5,
            ha="right",
            va="top",
            color="#374151",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="white",
                      ec="#9CA3AF",
                      lw=0.8))


# ══════════════════════════════════════════════════════════════════════════════
# Generate figures
# ══════════════════════════════════════════════════════════════════════════════

# Keep Fig 5 separate
make_fig5()

# Combine Fig 6 and Fig 7 side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

make_fig6(ax1)
make_fig7(ax2)

fig.tight_layout()
fig.savefig("combined_fig6_fig7.png", dpi=180, bbox_inches="tight")

plt.show()

print("Combined Fig 6 + Fig 7 saved.")
