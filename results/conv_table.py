import pandas as pd
import matplotlib.pyplot as plt
 
# =====================================================
# LOAD CSV
# =====================================================
df = pd.read_csv("convolution_results.csv")
 
# Clean column names
df.columns = df.columns.str.strip()
 
# =====================================================
# FILTER IMAGE SIZE
# Since your CSV uses width + height (NOT N)
# =====================================================
TARGET_W = 5824
TARGET_H = 3264
 
df = df[(df["width"] == TARGET_W) & (df["height"] == TARGET_H)].copy()
 
# =====================================================
# CREATE CONFIG NAMES
# =====================================================
def config_name(row):
    impl = str(row["implementation"]).strip()
 
    if impl == "CPU_Serial":
        return "CPU Serial"
 
    elif impl == "CPU_OMP":
        return f"CPU-OMP {int(row['threads'])}T"
 
    elif impl == "GPU_Naive":
        return "GPU Naive"
 
    elif impl == "GPU_Tiled":
        return "GPU Tiled"
 
    elif impl == "CPU_GPU_Hetero":
        pct = int(row["cpu_fraction"] * 100)
        return f"Hetero {pct}% CPU"
 
    return "Unknown"
 
df["Config"] = df.apply(config_name, axis=1)
 
# =====================================================
# CONFIGS TO SHOW
# =====================================================
wanted = [
    "CPU Serial",
    "CPU-OMP 16T",
    "GPU Naive",
    "GPU Tiled",
    "Hetero 10% CPU",
    "Hetero 20% CPU",
    "Hetero 30% CPU",
    "Hetero 40% CPU",
    "Hetero 50% CPU",
]
 
 
df = df[df["Config"].isin(wanted)]
 
# =====================================================
# CALCULATE MEAN + STD
# =====================================================
summary = df.groupby("Config").agg(
    Time_mean=("wall_total_ms", "mean"),
    Time_std=("wall_total_ms", "std"),
    MP_mean=("throughput_mpx_per_sec", "mean"),
    MP_std=("throughput_mpx_per_sec", "std")
).reset_index()
 
# =====================================================
# ORDER ROWS
# =====================================================
summary["order"] = summary["Config"].apply(lambda x: wanted.index(x))
summary = summary.sort_values("order").drop(columns="order")
 
# =====================================================
# ROUND
# =====================================================
summary["Time (ms)"] = summary["Time_mean"].round(3)
summary["±σ"] = summary["Time_std"].round(3)
summary["MP/s"] = summary["MP_mean"].round(1)
summary["±σ "] = summary["MP_std"].round(1)
 
table_df = summary[["Config", "Time (ms)", "±σ", "MP/s", "±σ "]]
 
print(table_df.to_string(index=False))
 
# =====================================================
# DRAW TABLE
# =====================================================
fig_height = max(6, len(table_df) * 0.42)
 
fig, ax = plt.subplots(figsize=(11, fig_height))
ax.axis("off")
 
table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="center"
)
 
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.7)
 
# ── Colour scheme ──────────────────────────────────────────────────────────────
HDR_COLOR    = "#d9e1f2"   # blue-grey header
GPU_COLOR    = "#dbeafe"   # light blue  — pure GPU rows
CPU_COLOR    = "#f3f4f6"   # light grey  — CPU rows
HETERO_COLOR = "#fef9c3"   # light amber — hetero rows
BEST_COLOR   = "#bbf7d0"   # light green — best wall-time row
 
best_time_idx = table_df["Time (ms)"].idxmin()
best_row_pos  = table_df.index.get_loc(best_time_idx) + 1  # +1 for header
 
# Styling
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
    "TABLE III. CONVOLUTION PERFORMANCE (5824×3264, MEAN ± 1 SD)\n"
    "Best wall-time row highlighted in green",
    fontsize=13,
    fontweight="bold",
    pad=20
)
 
plt.tight_layout()
plt.savefig("convolution_table.png", dpi=300, bbox_inches="tight")
plt.show()
