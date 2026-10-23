import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# LOAD CSV
# ==================================================
df = pd.read_csv("spmv_results(kaggle).csv")
df.columns = df.columns.str.strip()

# ==================================================
# FILTER FOR LARGEST MATRIX (256K x 256K, 8M nnz)
# ==================================================
TARGET_M = 262144
df = df[df["M"] == TARGET_M].copy()

# ==================================================
# CREATE CONFIG NAMES
# ==================================================
def config_name(row):
    impl = str(row["implementation"]).strip()

    if impl == "CPU_Serial":
        return "CPU Serial"

    elif impl == "CPU_OMP":
        return "CPU-OMP 16T"

    elif impl == "GPU_Naive":
        return "GPU Naive"

    elif impl == "GPU_Warp":
        return "GPU Warp"

    elif impl == "CPU_GPU_Hetero":
        pct = int(round(row["cpu_fraction"] * 100))
        if pct == 0:
            return "Hetero 0% CPU (GPU-only)"
        elif pct == 100:
            return "Hetero 100% CPU"
        else:
            return f"Hetero {pct}% CPU"

    return "Unknown"

df["Config"] = df.apply(config_name, axis=1)

# ==================================================
# KEEP ONLY RELEVANT ROWS
# ==================================================
wanted = [
    "CPU Serial",
    "CPU-OMP 4T",
    "GPU Naive",
    "GPU Warp",
    "Hetero 10% CPU",
    "Hetero 20% CPU",
    "Hetero 30% CPU",
    "Hetero 40% CPU",
    "Hetero 50% CPU",
]

df = df[df["Config"].isin(wanted)]

# ==================================================
# CALCULATE MEAN + STD  (wall time & bandwidth only)
# ==================================================
summary = df.groupby("Config").agg(
    Time_mean=("wall_total_ms",  "mean"),
    Time_std= ("wall_total_ms",  "std"),
    BW_mean=  ("bandwidth_GBs",  "mean"),
    BW_std=   ("bandwidth_GBs",  "std"),
).reset_index()

summary["order"] = summary["Config"].apply(
    lambda x: wanted.index(x) if x in wanted else 99)
summary = summary.sort_values("order").drop(columns="order")

summary["Wall Time (ms)"] = summary["Time_mean"].round(3)
summary["±σ (ms)"]        = summary["Time_std"].round(3)
summary["BW (GB/s)"]      = summary["BW_mean"].round(2)
summary["±σ (GB/s)"]      = summary["BW_std"].round(2)

table_df = summary[["Config", "Wall Time (ms)", "±σ (ms)", "BW (GB/s)", "±σ (GB/s)"]]

# Terminal preview
print(table_df.to_string(index=False))

# ==================================================
# GENERATE TABLE IMAGE
# ==================================================
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.axis("off")

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.1)

# ── Colour scheme ──────────────────────────────────────────────────────────────
HDR_COLOR    = "#d9e1f2"   # blue-grey header (matches prefix_table.py)
GPU_COLOR    = "#dbeafe"   # light blue  — pure GPU rows
CPU_COLOR    = "#f3f4f6"   # light grey  — CPU rows
HETERO_COLOR = "#fef9c3"   # light amber — hetero rows
BEST_COLOR   = "#bbf7d0"   # light green — best performing row

# Find best wall-time row (index in table, 1-based because row 0 = header)
best_time_idx = table_df["Wall Time (ms)"].idxmin()
best_row_pos  = table_df.index.get_loc(best_time_idx) + 1   # +1 for header offset

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#9CA3AF")
    cell.set_linewidth(0.6)

    if row == 0:
        cell.set_facecolor(HDR_COLOR)
        cell.set_text_props(weight="bold")
        continue

    config = table_df.iloc[row - 1]["Config"]

    if row == best_row_pos:
        cell.set_facecolor(BEST_COLOR)
    elif "Hetero" in config:
        cell.set_facecolor(HETERO_COLOR)
    elif "GPU" in config:
        cell.set_facecolor(GPU_COLOR)
    else:
        cell.set_facecolor(CPU_COLOR)

plt.title(
    "TABLE V.  SpMV AT M=N=256K, nnz=8M  (MEAN ± 1 SD, n=8 RUNS)\n"
    "Best wall-time row highlighted in green",
    fontsize=13,
    fontweight="bold",
    pad=18,
)

plt.tight_layout()
plt.savefig("spmv_table.png", dpi=300, bbox_inches="tight")
print("\nspmv_table.png saved.")
