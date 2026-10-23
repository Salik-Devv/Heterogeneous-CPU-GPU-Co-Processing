import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# LOAD CSV
# =====================================================
df = pd.read_csv("gemm_results.csv")   # <-- change filename if needed
df.columns = df.columns.str.strip()

# =====================================================
# CONFIG LABELS
# =====================================================
def config_name(row):
    impl = str(row["implementation"]).strip()

    if impl == "CPU_Serial":
        return "CPU Serial"

    elif impl == "CPU_OMP":
        return f"CPU-OMP {int(row['threads'])}T"

    elif impl == "GPU_Tiled":
        return "GPU Tiled"

    elif impl == "GPU_cuBLAS":
        return "GPU cuBLAS"

    elif impl == "CPU_GPU_Hetero":
        pct = int(row["cpu_fraction"] * 100)
        return f"Hetero {pct}%"

    return "Unknown"

df["Config"] = df.apply(config_name, axis=1)

# =====================================================
# MATRIX SIZES TO SHOW
# =====================================================
sizes = [512, 1024, 2048]

# =====================================================
# CONFIGS TO SHOW
# =====================================================
wanted = [
    "CPU Serial",
    "CPU-OMP 16T",
    "GPU Tiled",
    "GPU cuBLAS",
    "Hetero 10%",
    "Hetero 20%",
    "Hetero 30%",
    "Hetero 40%",
    "Hetero 100%"
]

# =====================================================
# BUILD TABLE
# =====================================================
rows = []

for n in sizes:
    row_data = {"N": n}

    temp = df[df["N"] == n]

    for cfg in wanted:
        sub = temp[temp["Config"] == cfg]

        mean = sub["wall_total_ms"].mean()
        std = sub["wall_total_ms"].std()

        row_data[cfg] = f"{mean:,.1f}±{std:,.1f}"

    rows.append(row_data)

table_df = pd.DataFrame(rows)

print(table_df.to_string(index=False))

# =====================================================
# DRAW TABLE
# =====================================================
fig, ax = plt.subplots(figsize=(13, 3.8))
ax.axis("off")

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)

# ── Colour scheme ──────────────────────────────────────────────────────────────
HDR_COLOR    = "#d9e1f2"   # blue-grey header
GPU_COLOR    = "#dbeafe"   # light blue  — GPU columns
CPU_COLOR    = "#f3f4f6"   # light grey  — CPU columns
HETERO_COLOR = "#fef9c3"   # light amber — hetero columns
BEST_COLOR   = "#bbf7d0"   # light green — best (min) cell per data row

# Map column index → category
col_names = list(table_df.columns)  # ["N", "CPU Serial", "CPU-OMP 16T", ...]

def col_category(name):
    if "Hetero" in name:
        return "hetero"
    elif "GPU" in name:
        return "gpu"
    elif name == "N":
        return "header"
    else:
        return "cpu"

# For each data row, find the column index with the minimum mean wall time
def best_col_for_row(row_values):
    # row_values is a list of strings like "5.3±1.2"; skip "N" col (index 0)
    min_val = float("inf")
    min_idx = -1
    for i, v in enumerate(row_values[1:], start=1):
        try:
            mean = float(str(v).split("±")[0].replace(",", ""))
            if mean < min_val:
                min_val = mean
                min_idx = i
        except ValueError:
            pass
    return min_idx

best_cols = [best_col_for_row(list(table_df.iloc[r])) for r in range(len(table_df))]

# Styling
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#9CA3AF")
    cell.set_linewidth(0.6)

    if row == 0:
        cell.set_facecolor(HDR_COLOR)
        cell.set_text_props(weight="bold")
        continue

    # Check if this cell is the best in its data row
    data_row = row - 1
    if col == best_cols[data_row]:
        cell.set_facecolor(BEST_COLOR)
        continue

    cat = col_category(col_names[col])
    if cat == "hetero":
        cell.set_facecolor(HETERO_COLOR)
    elif cat == "gpu":
        cell.set_facecolor(GPU_COLOR)
    else:
        cell.set_facecolor(CPU_COLOR)

plt.title(
    "TABLE II. GEMM WALL TIME (MS, MEAN ± 1 SD, N=8 RUNS)\n"
    "Best cell per row highlighted in green",
    fontsize=13,
    fontweight="bold",
    pad=20
)

plt.tight_layout()
plt.savefig("gemm_table.png", dpi=300, bbox_inches="tight")
plt.show()