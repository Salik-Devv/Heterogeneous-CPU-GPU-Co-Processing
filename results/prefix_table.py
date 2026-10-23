import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# LOAD CSV
# ==================================================
df = pd.read_csv("prefix_sum_results.csv")

# Clean headers
df.columns = df.columns.str.strip()

# ==================================================
# FILTER FOR N = 16M
# ==================================================
TARGET_N = 16777216
df = df[df["N"] == TARGET_N].copy()

# ==================================================
# CREATE CONFIG NAMES
# ==================================================
def config_name(row):
    impl = str(row["implementation"]).strip()

    if impl == "CPU_Serial":
        return "CPU Serial"

    elif impl == "CPU_OMP":
        return f"CPU-OMP {int(row['threads'])}T"

    elif impl == "GPU":
        return "GPU"

    elif impl == "CPU_GPU_Hetero":
        pct = int(row["cpu_fraction"] * 100)

        if pct == 0:
            return "GPU"
        elif pct == 100:
            return f"CPU-OMP {int(row['threads'])}T"
        else:
            return f"Hetero {pct}% CPU"

    return "Unknown"

# Create column
df["Config"] = df.apply(config_name, axis=1)

# ==================================================
# KEEP ONLY REQUIRED ROWS
# ==================================================
wanted = [
    "CPU Serial",
    "CPU-OMP 1T",
    "CPU-OMP 8T",
    "CPU-OMP 16T",
    "GPU",
    "Hetero 10% CPU",
    "Hetero 20% CPU",
    "Hetero 30% CPU",
    "Hetero 40% CPU",
    "Hetero 50% CPU"
]

df = df[df["Config"].isin(wanted)]

# ==================================================
# CALCULATE MEAN + STD
# ==================================================
summary = df.groupby("Config").agg(
    Time_mean=("wall_total_ms", "mean"),
    Time_std=("wall_total_ms", "std"),
    BW_mean=("bandwidth_gb_s", "mean"),
    BW_std=("bandwidth_gb_s", "std")
).reset_index()

# Reorder rows
summary["order"] = summary["Config"].apply(lambda x: wanted.index(x))
summary = summary.sort_values("order").drop(columns="order")

# Round values
summary["Time (ms)"] = summary["Time_mean"].round(3)
summary["±σ"] = summary["Time_std"].round(3)
summary["BW (GB/s)"] = summary["BW_mean"].round(2)
summary["±σ "] = summary["BW_std"].round(2)

table_df = summary[["Config", "Time (ms)", "±σ", "BW (GB/s)", "±σ "]]

# Print terminal output
print(table_df.to_string(index=False))

# ==================================================
# GENERATE TABLE IMAGE
# ==================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

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
    "TABLE IV. PREFIX SUM AT N=16M (MEAN ± 1 SD, n=8 RUNS)\n"
    "Best wall-time row highlighted in green",
    fontsize=14,
    fontweight="bold",
    pad=20
)

plt.tight_layout()
plt.savefig("prefix_sum_table.png", dpi=300, bbox_inches="tight")
plt.show()