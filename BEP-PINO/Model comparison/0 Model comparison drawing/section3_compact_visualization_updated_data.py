# -*- coding: utf-8 -*-
"""
Section 3 compact visualization package for the updated Table 2 data.

The plotting style of Figs. 4–6 is retained:
- Fig. 4: normalized multi-metric performance matrix;
- Fig. 5: response-resolved 3D relative-L2 bar chart;
- Fig. 6: accuracy-efficiency trade-off with an inset loss comparison.

All generated figures and the cleaned CSV file are saved in the same folder
as this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, LogFormatterMathtext
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


try:
    SAVE_DIR = Path(__file__).resolve().parent
except NameError:
    SAVE_DIR = Path.cwd()


# ============================================================
# User-controllable settings
# ============================================================
FONT_SIZE = 16
MAIN_FONT = "Times New Roman"

# Fig. 6 main-axis range
FIG6_YLIM = (4.5e-3, 7.5e-2)
FIG6_YTICKS = [5e-3, 1e-2, 2e-2, 5e-2]

# Fig. 6 inset-axis range
LOSS_INSET_YLIM = (5e-4, 1.0e-1)
LOSS_INSET_YTICKS = [1e-3, 1e-2, 1e-1]


# ============================================================
# Updated data
# ============================================================
models = ["BEP-PINO", "ME-BEP-PINO", "HO-BEP-PINO", "BEP-PDON"]

df = pd.DataFrame({
    "Model": models,
    "Epoch": [1100, 1100, 3500, 1100],
    "Training time (s)": [63.79, 243.69, 165.78, 30.56],
    "Loss": [5.96e-04, 1.40e-02, 6.61e-02, 3.34e-03],

    "u MAE": [2.24e-06, 6.61e-06, 1.95e-05, 6.64e-06],
    "phi MAE": [1.27e-05, 6.07e-05, 7.23e-05, 2.66e-05],
    "M MAE": [1.09e-04, 1.25e-03, 6.33e-04, 2.47e-04],
    "Q MAE": [4.11e-04, 4.85e-03, 4.23e-03, 1.76e-03],

    "u epsL2": [4.47e-03, 1.37e-02, 3.89e-02, 1.30e-02],
    "phi epsL2": [6.85e-03, 3.09e-02, 3.91e-02, 1.52e-02],
    "M epsL2": [9.78e-03, 9.55e-02, 6.17e-02, 2.06e-02],
    "Q epsL2": [4.46e-03, 4.14e-02, 4.39e-02, 2.06e-02],

    "u R2": [0.999931, 0.999422, 0.994725, 0.999101],
    "phi R2": [0.999945, 0.998974, 0.998223, 0.999721],
    "M R2": [0.999897, 0.988260, 0.995638, 0.999569],
    "Q R2": [0.999979, 0.997666, 0.997414, 0.999477],

    # The following averages use the values reported in the updated table.
    "Average MAE": [1.34e-04, 1.54e-03, 1.24e-03, 5.10e-04],
    "Average epsL2": [6.39e-03, 4.54e-02, 4.59e-02, 1.73e-02],
    "Average R2": [0.999938, 0.996081, 0.996500, 0.999467],
})

df["1 - Average R2"] = 1.0 - df["Average R2"]

df.to_csv(
    SAVE_DIR / "table2_updated_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Global plotting style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [MAIN_FONT, "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": MAIN_FONT,
    "mathtext.it": MAIN_FONT + ":italic",
    "mathtext.bf": MAIN_FONT + ":bold",
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "axes.linewidth": 1.0,
    "figure.dpi": 170,
    "savefig.dpi": 900,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

palette = {
    "BEP-PINO": "#081D58",
    "ME-BEP-PINO": "#AEE0D2",
    "HO-BEP-PINO": "#FFFFD9",
    "BEP-PDON": "#44B7C4",
    "Loss": "#F0DAD5",
}

matrix_cmap = LinearSegmentedColormap.from_list(
    "bp_pfno_matrix",
    #[ "#3858C0", "#F0DAD5", "#C50303"],
    [ "#FFFFD9", "#44B7C4", "#081D58"],
)


def save_all(fig, name, bbox_inches="tight", pad=0.1):
    """Save PNG and editable SVG outputs beside this script."""
    fig.savefig(
        SAVE_DIR / f"{name}.png",
        bbox_inches=bbox_inches,
        pad_inches=pad,
        facecolor="white",
    )
    fig.savefig(
        SAVE_DIR / f"{name}.svg",
        bbox_inches=bbox_inches,
        pad_inches=pad,
        facecolor="white",
    )
    plt.close(fig)


# ============================================================
# Fig. 4: normalized multi-metric performance matrix
# ============================================================
metric_cols = [
    "Training time (s)",
    "Loss",
    "Average MAE",
    "Average epsL2",
    "1 - Average R2",
]

metric_labels = [
    "Training\ntime",
    "Final\nloss",
    "Average\n$\\overline{\\mathrm{MAE}}$",
    "Average\n$\\overline{\\varepsilon}_{L_2}$",
    "1 - Average$\\overline{R^2}$",
]

metric_matrix = df[metric_cols].to_numpy(dtype=float) 
log_metric_matrix = np.log10(metric_matrix)

# Column-wise log normalization
normalized_matrix = (
    log_metric_matrix - log_metric_matrix.min(axis=0)
) / (
    log_metric_matrix.max(axis=0)
    - log_metric_matrix.min(axis=0)
    + 1e-12
)

fig, ax = plt.subplots(figsize=(11.5, 3))

image = ax.imshow(
    normalized_matrix,
    cmap=matrix_cmap,
    aspect="auto",
    alpha=0.96,
    vmin=0,
    vmax=1,
)

ax.set_xticks(np.arange(len(metric_cols)))
ax.set_xticklabels(metric_labels)
ax.set_yticks(np.arange(len(models)))
ax.set_yticklabels(models, rotation=30)

ax.set_title("Normalized multi-metric performance matrix")
#ax.set_xlabel("Metric")

# Cell boundaries
ax.set_xticks(np.arange(-0.5, len(metric_cols), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
ax.grid(
    which="minor",
    color=(0, 0, 0, 0.35),
    linestyle="-",
    linewidth=0.9,
)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(axis="both", labelsize=FONT_SIZE)

# Original values displayed in each cell
for row_index in range(normalized_matrix.shape[0]):
    for col_index in range(normalized_matrix.shape[1]):
        value = df.iloc[row_index][metric_cols[col_index]]
        label = f"{value:.1f}" if value >= 1 else f"{value:.1e}"

        rgba = image.cmap(image.norm(normalized_matrix[row_index, col_index]))
        r, g, b = rgba[:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "white" if luminance < 0.4 else "black"
        ax.text(
            col_index,
            row_index,
            label,
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
            color=text_color,
        )

colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.02)
colorbar.set_label(
    "Normalized score\n(log-normalized)",
    fontsize=FONT_SIZE,
)
colorbar.ax.tick_params(labelsize=FONT_SIZE)

fig.tight_layout()
save_all(fig, "Fig4_multimetric_performance_matrix")


response_labels = [r"$u$", r"$\phi$", r"$M$", r"$Q$"]
eps_cols = ["u epsL2", "phi epsL2", "M epsL2", "Q epsL2"]

depth_order = ["BEP-PINO", "BEP-PDON", "ME-BEP-PINO", "HO-BEP-PINO"]
depth_df = df.set_index("Model").loc[depth_order].reset_index()
eps_matrix = depth_df[eps_cols].to_numpy(dtype=float)

fig = plt.figure(figsize=(12.0, 7.2))
ax = fig.add_subplot(111, projection="3d")

xpos, ypos = np.meshgrid(
    np.arange(len(response_labels)),
    np.arange(len(depth_order)),
)
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos, dtype=float)

dx = np.full_like(xpos, 0.55, dtype=float)
dy = np.full_like(ypos, 0.55, dtype=float)
dz = eps_matrix.ravel()

bar_colors = []
for model_name in depth_order:
    bar_colors.extend(
        [palette[model_name]] * len(response_labels)
    )

ax.bar3d(
    xpos,
    ypos,
    zpos,
    dx,
    dy,
    dz,
    color=bar_colors,
    edgecolor=(0, 0, 0, 0.65),  
    linewidth=0.6,              
    shade=False,
    alpha=1.0,
)

ax.set_xticks(np.arange(len(response_labels)) + 0.28)
ax.set_xticklabels(response_labels, fontsize=FONT_SIZE+1)

ax.set_yticks(np.arange(len(depth_order)) + 0.28)
ax.yaxis.set_major_formatter(FixedFormatter(depth_order))
ax.set_yticklabels(
    depth_order,
    fontsize=FONT_SIZE,
    rotation=0,
    ha="left",
    va="center",
)

ax.tick_params(axis="x", pad=1)
ax.tick_params(axis="y", pad=1)

ax.set_zlabel(
    r"Relative L2 error ($\overline{\varepsilon}_{L_2}$)",
    labelpad=8,
    fontsize=FONT_SIZE+2,
)

# Remove grids and panes, while retaining the XYZ axis lines.
ax.grid(False)

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis._axinfo["grid"]["linewidth"] = 0
    axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    axis._axinfo["tick"]["inward_factor"] = 0
    axis._axinfo["tick"]["outward_factor"] = 0.25

for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_facecolor((1, 1, 1, 0))
    axis.pane.set_edgecolor((1, 1, 1, 0))

ax.xaxis.line.set_color("black")
ax.yaxis.line.set_color("black")
ax.zaxis.line.set_color("black")

ax.xaxis.line.set_linewidth(1.0)
ax.yaxis.line.set_linewidth(1.0)
ax.zaxis.line.set_linewidth(1.0)

ax.tick_params(
    axis="both",
    which="both",
    length=0,
    labelsize=FONT_SIZE,
)
ax.tick_params(
    axis="z",
    which="both",
    length=0,
    labelsize=FONT_SIZE+2,
)

# The updated maximum epsL2 is 0.0955.
ax.set_zlim(0.0, 0.10)
ax.set_zticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax.set_zticklabels(
    ["0.00", "0.02", "0.04", "0.06", "0.08", "0.10"],
    fontsize=FONT_SIZE,
)

ax.view_init(elev=20, azim=-57)

fig.subplots_adjust(
    left=0.02,
    right=0.85,
    bottom=0.08,
    top=0.90,
)
fig.tight_layout()

save_all(
    fig,
    "Fig5_response_resolved_epsL2_3D",
    bbox_inches=None,
)


# ============================================================
# Fig. 6: accuracy-efficiency trade-off
# ============================================================
fig, ax = plt.subplots(figsize=(8.0, 5.5))

for _, row in df.iterrows():
    ax.scatter(
        row["Training time (s)"],
        row["Average epsL2"],
        s=240,
        color=palette[row["Model"]],
        edgecolor=(0, 0, 0, 0.75),
        linewidth=0.9,
        alpha=0.82,
        zorder=3,
    )

# Updated annotation positions for the new data distribution.
label_offsets = {
    "BEP-PINO": (-30, -24),
    "ME-BEP-PINO": (-75, -24),
    "HO-BEP-PINO": (-30, 12),
    "BEP-PDON": (-15, -24),
}

for _, row in df.iterrows():
    offset_x, offset_y = label_offsets[row["Model"]]
    ax.annotate(
        row["Model"],
        (
            row["Training time (s)"],
            row["Average epsL2"],
        ),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        fontsize=FONT_SIZE,
    )

ax.set_yscale("log")
ax.set_xlim(20, 260)
ax.set_ylim(*FIG6_YLIM)

ax.yaxis.set_major_locator(FixedLocator(FIG6_YTICKS))
ax.yaxis.set_major_formatter(
    FixedFormatter([
        f"{value:.3f}" if value < 0.01 else f"{value:.2f}"
        for value in FIG6_YTICKS
    ])
)

ax.set_xlabel("Training time (s)", fontsize=FONT_SIZE+4)
ax.set_ylabel(
    r"Average $\overline{\varepsilon}_{L_2}$",
    fontsize=FONT_SIZE+4,
)
ax.set_title(
    "Accuracy-efficiency trade-off",
    fontsize=FONT_SIZE,
)

ax.grid(
    True,
    which="both",
    linewidth=0.45,
    alpha=0.25,
)
ax.tick_params(axis="both", labelsize=FONT_SIZE+2)


# ------------------------------------------------------------
# Inset: updated single-loss comparison
# ------------------------------------------------------------
inset = ax.inset_axes([0.35, 0.15, 0.6, 0.50])

for spine in inset.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("black")
    spine.set_linewidth(1.0)

x = np.arange(len(models))
bar_width = 0.55

inset.bar(
    x,
    df["Loss"],
    bar_width,
    color=palette["Loss"],
    edgecolor=(0, 0, 0, 0.55),
    linewidth=0.5,
    alpha=0.78,
    label="Loss",
)

inset.set_yscale("log")
inset.set_ylim(*LOSS_INSET_YLIM)
inset.yaxis.set_major_locator(
    FixedLocator(LOSS_INSET_YTICKS)
)
inset.yaxis.set_major_formatter(
    LogFormatterMathtext(base=10)
)

inset.set_xticks(x)
inset.set_xticklabels(
    ["BEP-PINO", "ME-BEP-PINO", "HO-BEP-PINO", "BEP-PDON"],
    fontsize=FONT_SIZE - 2,
    rotation=15,
)

inset.tick_params(
    axis="y",
    labelsize=FONT_SIZE,
)
inset.set_title(
    "Final loss comparison",
    fontsize=FONT_SIZE,
)

inset.grid(
    True,
    which="both",
    axis="y",
    linewidth=0.35,
    alpha=0.28,
)

inset.legend(
    frameon=False,
    fontsize=FONT_SIZE - 2,
    loc="upper right",
    bbox_to_anchor=(0.38, 0.95),
    handlelength=1,
    handleheight=0.8,
    labelspacing=0.1,
)

fig.tight_layout()
save_all(fig, "Fig6_accuracy_efficiency_tradeoff")
