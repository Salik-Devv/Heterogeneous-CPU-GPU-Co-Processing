import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("../results/spmv_results.csv")
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

SIZES  = [65536, 131072, 262144]
LABELS = ["64K×64K\n(2M nnz)", "128K×128K\n(4M nnz)", "256K×256K\n(8M nnz)"]
N_SIZES = len(SIZES)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_SERIAL  = "#6B7280"
C_OMP16   = "#10B981"
C_GNAIVE  = "#F59E0B"
C_GWARP   = "#2563EB"
C_H2D     = "#93C5FD"
C_D2H     = "#1D4ED8"
C_GCPU    = "#A78BFA"
C_CCPU    = "#F97316"
C_LINES   = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#6B7280"]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A — Bandwidth bar chart (GB/s)
# ══════════════════════════════════════════════════════════════════════════════
def make_figA():
    ser   = df[(df["implementation"] == "CPU_Serial") & (df["threads"] == 1)]
    omp16 = df[(df["implementation"] == "CPU_OMP")    & (df["threads"] == 16)]
    gnaive = df[df["implementation"] == "GPU_Naive"]
    gwarp  = df[df["implementation"] == "GPU_Warp"]

    def stats(subset):
        return subset.groupby("M")["bandwidth_GBs"].agg(mean="mean", std="std").reindex(SIZES)

    ser_g    = stats(ser)
    omp16_g  = stats(omp16)
    gnaive_g = stats(gnaive)
    gwarp_g  = stats(gwarp)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(N_SIZES)
    w = 0.19
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

    groups = [
        (ser_g,    C_SERIAL,  "CPU Serial"),
        (omp16_g,  C_OMP16,   "CPU-OMP 16T"),
        (gnaive_g, C_GNAIVE,  "GPU Naive"),
        (gwarp_g,  C_GWARP,   "GPU Warp"),
    ]

    for (grp, col, lbl), off in zip(groups, offsets):
        means = grp["mean"].values.astype(float)
        stds  = grp["std"].fillna(0).values.astype(float)

        ax.bar(x + off, means, w,
               color=col,
               alpha=0.88,
               label=lbl,
               yerr=stds,
               capsize=3,
               error_kw=dict(
                   elinewidth=1.1,
                   ecolor="#374151",
                   capthick=1.1
               ))

    peak_idx = int(np.nanargmax(gwarp_g["mean"].values))

    ax.annotate(
        f"GPU Warp peak\n{gwarp_g['mean'].iloc[peak_idx]:.1f} GB/s",
        xy=(x[peak_idx] + offsets[3],
            gwarp_g["mean"].iloc[peak_idx]),
        xytext=(x[peak_idx] + offsets[3] + 0.4,
                gwarp_g["mean"].iloc[peak_idx] + 8),
        fontsize=8.5,
        color=C_GWARP,
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_GWARP,
            lw=1.2
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_xlabel("Matrix Size M×N (nnz = 32 per row)", fontsize=11)
    ax.set_ylabel("Effective Bandwidth (GB/s)", fontsize=11)
    ax.set_title("SpMV: Effective Bandwidth vs Matrix Size", fontsize=12, pad=9)

    ax.legend(fontsize=9,
              framealpha=0.9,
              ncol=4,
              loc="upper left")

    ax.text(
        0.98, 0.96,
        "Memory-bound kernel: warp-level coalescing\nyields ~2× bandwidth over naive CSR",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right",
        va="top",
        color="#374151",
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="#9CA3AF",
            lw=0.8
        )
    )

    plt.tight_layout()
    fig.savefig("spmv_bandwidth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Fig A saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B — GFLOPS bar chart
# ══════════════════════════════════════════════════════════════════════════════
def make_figB():
    ser    = df[(df["implementation"] == "CPU_Serial") & (df["threads"] == 1)]
    omp16  = df[(df["implementation"] == "CPU_OMP")    & (df["threads"] == 16)]
    gnaive = df[df["implementation"] == "GPU_Naive"]
    gwarp  = df[df["implementation"] == "GPU_Warp"]

    def stats(subset):
        return subset.groupby("M")["gflops"].agg(mean="mean", std="std").reindex(SIZES)

    ser_g    = stats(ser)
    omp16_g  = stats(omp16)
    gnaive_g = stats(gnaive)
    gwarp_g  = stats(gwarp)

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(N_SIZES)
    w = 0.19
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

    groups = [
        (ser_g,    C_SERIAL,  "CPU Serial"),
        (omp16_g,  C_OMP16,   "CPU-OMP 16T"),
        (gnaive_g, C_GNAIVE,  "GPU Naive"),
        (gwarp_g,  C_GWARP,   "GPU Warp"),
    ]

    for (grp, col, lbl), off in zip(groups, offsets):
        means = grp["mean"].values.astype(float)
        stds  = grp["std"].fillna(0).values.astype(float)

        bars = ax.bar(
            x + off,
            means,
            w,
            color=col,
            alpha=0.88,
            label=lbl,
            yerr=stds,
            capsize=3,
            error_kw=dict(
                elinewidth=1.1,
                ecolor="#374151",
                capthick=1.1
            )
        )

        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{m:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=col,
                    fontweight="bold"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_xlabel("Matrix Size M×N (nnz = 32 per row)", fontsize=11)
    ax.set_ylabel("GFLOP/s (2·nnz / compute_time)", fontsize=11)
    ax.set_title("SpMV: Compute Throughput (GFLOP/s) vs Matrix Size", fontsize=12, pad=9)

    ax.legend(fontsize=9,
              framealpha=0.9,
              ncol=4,
              loc="upper right")

    ax.text(
        0.02, 0.96,
        "SpMV is memory-bound; GFLOP/s tracks\nbandwidth utilisation, not FPU saturation",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="top",
        color="#374151",
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="#9CA3AF",
            lw=0.8
        )
    )

    plt.tight_layout()
    fig.savefig("spmv_gflops.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Fig B saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C — Hetero wall time vs CPU fraction
# ══════════════════════════════════════════════════════════════════════════════
def make_figC(ax):
    hetero = df[df["implementation"] == "CPU_GPU_Hetero"]

    grp = hetero.groupby(
        ["M", "cpu_fraction"]
    )["wall_total_ms"].agg(
        mean="mean",
        std="std"
    ).reset_index()

    SIZE_LABELS = ["64K×64K", "128K×128K", "256K×256K"]

    for color, M, lbl in zip(C_LINES, SIZES, SIZE_LABELS):
        sub = grp[grp["M"] == M].sort_values("cpu_fraction")

        frac = sub["cpu_fraction"].values
        mean = sub["mean"].values
        std  = sub["std"].fillna(0).values

        ax.fill_between(
            frac,
            mean - std,
            mean + std,
            color=color,
            alpha=0.12
        )

        ax.plot(
            frac,
            mean,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=lbl
        )

        min_i = np.argmin(mean)

        ax.plot(
            frac[min_i],
            mean[min_i],
            marker="*",
            markersize=11,
            color=color,
            zorder=5
        )

        ax.annotate(
            f"min @ {frac[min_i]:.1f}\n({mean[min_i]:.1f} ms)",
            xy=(frac[min_i], mean[min_i]),
            xytext=(frac[min_i] + 0.06, mean[min_i] + 0.5),
            fontsize=7.5,
            color=color,
        )

    ax.set_xlabel("CPU Fraction  (0 = GPU only,  1 = CPU only)", fontsize=11)
    ax.set_ylabel("Wall Time (ms)", fontsize=11)

    ax.set_title(
        "SpMV: Hetero Wall Time vs CPU-GPU Split Ratio",
        fontsize=12,
        pad=9
    )

    ax.set_xlim(-0.02, 1.02)

    ax.set_xticks(np.arange(0, 1.1, 0.1))

    ax.set_xticklabels(
        [f"{v:.1f}" for v in np.arange(0, 1.1, 0.1)],
        fontsize=9
    )

    star_patch = plt.Line2D(
        [0],
        [0],
        marker="*",
        color="gray",
        linestyle="None",
        markersize=10,
        label="Optimum split (★)"
    )

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(
        handles + [star_patch],
        labels + ["Optimum split (★)"],
        title="Matrix Size",
        fontsize=9,
        title_fontsize=9,
        framealpha=0.9,
        loc="upper left",
        ncol=2
    )

    ax.text(
        0.98, 0.97,
        "GPU-only (0%) best for all sizes\nCPU rows become bottleneck above ~10%",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right",
        va="top",
        color="#374151",
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="#9CA3AF",
            lw=0.8
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D — GPU breakdown
# ══════════════════════════════════════════════════════════════════════════════
def make_figD(ax):
    gwarp = df[df["implementation"] == "GPU_Warp"]

    g = gwarp.groupby("M")[[
        "h2d_ms",
        "gpu_kernel_ms",
        "d2h_ms"
    ]].mean().reindex(SIZES)

    g_std = gwarp.groupby("M")[[
        "h2d_ms",
        "gpu_kernel_ms",
        "d2h_ms"
    ]].std().reindex(SIZES)

    h2d   = g["h2d_ms"].values
    gpu_k = g["gpu_kernel_ms"].values
    d2h   = g["d2h_ms"].values

    total_mean = h2d + gpu_k + d2h

    total_std  = (
        g_std["h2d_ms"]**2 +
        g_std["gpu_kernel_ms"]**2 +
        g_std["d2h_ms"]**2
    ).pow(0.5).values

    x = np.arange(N_SIZES)
    w = 0.45

    ax.bar(
        x,
        h2d,
        w,
        color=C_H2D,
        alpha=0.9,
        label="H2D Transfer"
    )

    ax.bar(
        x,
        gpu_k,
        w,
        bottom=h2d,
        color=C_GCPU,
        alpha=0.9,
        label="GPU Kernel"
    )

    ax.bar(
        x,
        d2h,
        w,
        bottom=h2d + gpu_k,
        color=C_D2H,
        alpha=0.9,
        label="D2H Transfer"
    )

    ax.errorbar(
        x,
        total_mean,
        yerr=total_std,
        fmt="none",
        ecolor="#111827",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2
    )

    for i in range(N_SIZES):
        pct = 100 * gpu_k[i] / total_mean[i]

        ax.text(
            x[i],
            total_mean[i] + total_std[i] + 0.05,
            f"kernel\n{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#1E3A5F"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)

    ax.set_xlabel(
        "Matrix Size M×N (nnz = 32 per row)",
        fontsize=11
    )

    ax.set_ylabel("Time (ms)", fontsize=11)

    ax.set_title(
        "SpMV GPU Warp: Time Breakdown\n(H2D + Kernel + D2H)",
        fontsize=12,
        pad=9
    )

    ax.legend(fontsize=9,
              framealpha=0.9,
              loc="upper left")

    ax.text(
        0.98, 0.96,
        "PCIe transfer dominates end-to-end;\nkernel is <5% of wall time",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="right",
        va="top",
        color="#374151",
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="white",
            ec="#9CA3AF",
            lw=0.8
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# Generate figures
# ══════════════════════════════════════════════════════════════════════════════

# Keep A and B separate
make_figA()
make_figB()

# Combine C and D side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

make_figC(ax1)
make_figD(ax2)

fig.tight_layout()
fig.savefig("combined_figC_figD.png", dpi=180, bbox_inches="tight")

plt.show()

print("\nCombined Fig C + Fig D saved.")
