# ============================================================ 
# BP-PFNO Chapter 3.3: Plotting for Ablation and Parameter Tests
# File: BP-PFNO_3.3_Ablation_and_Param_Plot.py
# Function: read CSV/logs and generate publication-style figures only.
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


RUN_ROOT = Path("./BP_PFNO_3p3_runs_no_dynamic_weights")
PLOT_ROOT = RUN_ROOT / "plots"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'figure.dpi': 500,
    'savefig.dpi': 500,
    'font.size': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16
})

TOTAL_ADAM_COLOR = "#C0321A"
TOTAL_LBFGS_COLOR = "#218D42"
COMPONENT_COLORS = {
    'geo':   "#8074C8",
    'const': "#7895C1",
    'eq1':   "#59A14F",
    'eq2':   "#EF8B67",
}

RESPONSE_LABELS = ["$u$", "$\\phi$", "$M$", "$Q$"]
RESPONSE_KEYS = ["u", "phi", "M", "Q"]

ABLATION_ORDER = [
    "Base",
    "Abl-BC-Soft",
    "Abl-BC-None",
    "Abl-Scale-None",
    "Abl-Scale-Half",
    "Abl-Scale-Double",
    "Abl-LoadScale-Off",
    "Abl-Norm-Off",
    "Abl-Precond-Off",
    "Abl-Precond-Off-DW"
]


def savefig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), dpi=500, bbox_inches="tight")
    plt.close()


def load_summary():
    summary_path = RUN_ROOT / "ALL_results_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"未找到 {summary_path}，请先运行 BP-PFNO_3.3_Ablation_and_Param_EVA.py")
    return pd.read_csv(summary_path)


def aggregate_mean_std(df, group_col, value_cols):
    mean_df = df.groupby(group_col, observed=True)[value_cols].mean().reset_index()
    std_df = df.groupby(group_col, observed=True)[value_cols].std().reset_index()
    for col in value_cols:
        if col in std_df.columns:
            std_df[col] = std_df[col].fillna(0.0)
    return mean_df, std_df


def plot_standardized_training_history(
    logs_path,
    out_prefix
):
    logs = torch.load(
        logs_path,
        weights_only=False
    )

    components = [
        ('geo', 'Geometric'),
        ('const', 'Constitutive'),
        ('eq1', 'Equilibrium 1'),
        ('eq2', 'Equilibrium 2')
    ]

    def stages(
        adam_key,
        lbfgs_key,
        label,
        adam_color=None,
        lbfgs_color=None
    ):
        adam = np.asarray(
            logs.get(adam_key, []),
            dtype=float
        )
        lbfgs = np.asarray(
            logs.get(lbfgs_key, []),
            dtype=float
        )

        if adam.size:
            plt.semilogy(
                np.arange(adam.size),
                adam,
                color=adam_color,
                linewidth=1.8,
                linestyle='-',
                label=f'Adam {label}'
            )

        if lbfgs.size:
            plt.semilogy(
                np.arange(
                    adam.size,
                    adam.size + lbfgs.size
                ),
                lbfgs,
                color=lbfgs_color,
                linewidth=2.0,
                linestyle='--',
                label=f'L-BFGS {label}'
            )

    def has_values(*keys):
        for key in keys:
            arr = np.asarray(
                logs.get(key, []),
                dtype=float
            )
            if arr.size:
                return True
        return False

    for (
        title,
        adam_key,
        lbfgs_key,
        file_name
    ) in [
        (
            'Total Objective Loss',
            'loss_history',
            'lbfgs_loss_history',
            'total_objective_loss'
        ),
        (
            'Physics Total Loss',
            'raw_total_loss_history',
            'lbfgs_raw_total_loss_history',
            'physics_total_loss'
        )
    ]:
        plt.figure(figsize=(10, 6))
        stages(
            adam_key,
            lbfgs_key,
            title,
            adam_color=TOTAL_ADAM_COLOR,
            lbfgs_color=TOTAL_LBFGS_COLOR
        )
        plt.title(title)
        plt.xlabel('Optimization Step')
        plt.ylabel('Loss (Log Scale)')
        plt.grid(
            True,
            which="both",
            ls="--",
            alpha=0.5
        )
        plt.legend(frameon=False)
        savefig(
            f"{out_prefix}_{file_name}"
        )

    plt.figure(figsize=(10, 6))
    for key, label in components:
        stages(
            f'loss_{key}_history',
            f'lbfgs_loss_{key}_history',
            label,
            adam_color=COMPONENT_COLORS[key],
            lbfgs_color=COMPONENT_COLORS[key]
        )

    plt.title(
        'Raw Physics Loss Components'
    )
    plt.xlabel('Optimization Step')
    plt.ylabel('Loss (Log Scale)')
    plt.grid(
        True,
        which="both",
        ls="--",
        alpha=0.5
    )
    plt.legend(frameon=False)
    savefig(
        f"{out_prefix}_physics_loss_components"
    )

    if has_values(
        'weighted_geo_history',
        'weighted_const_history',
        'weighted_eq1_history',
        'weighted_eq2_history',
        'lbfgs_weighted_geo_history',
        'lbfgs_weighted_const_history',
        'lbfgs_weighted_eq1_history',
        'lbfgs_weighted_eq2_history'
    ):
        plt.figure(figsize=(10, 6))
        for key, label in components:
            stages(
                f'weighted_{key}_history',
                f'lbfgs_weighted_{key}_history',
                label,
                adam_color=COMPONENT_COLORS[key],
                lbfgs_color=COMPONENT_COLORS[key]
            )
        plt.title('Weighted Physics Loss Components')
        plt.xlabel('Optimization Step')
        plt.ylabel('Weighted Loss (Log Scale)')
        plt.grid(
            True,
            which="both",
            ls="--",
            alpha=0.5
        )
        plt.legend(frameon=False)
        savefig(
            f"{out_prefix}_weighted_physics_loss_components"
        )

    weight_history = np.asarray(
        logs.get('weight_history', []),
        dtype=float
    )
    if (
        weight_history.ndim == 2 and
        weight_history.shape[0] > 0 and
        weight_history.shape[1] == 4
    ):
        plt.figure(figsize=(10, 6))
        for idx, (key, label) in enumerate(components):
            plt.plot(
                weight_history[:, idx],
                color=COMPONENT_COLORS[key],
                linewidth=2.0,
                label=label
            )
        plt.xlabel('Adam Epoch')
        plt.ylabel('Dynamic Weight Value')
        plt.title('Evolution of Dynamic Weights')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=False)
        savefig(
            f"{out_prefix}_dynamic_weights"
        )


def plot_selected_training_histories():
    selected = [
        "Base",
        "Abl-BC-Soft",
        "Abl-LoadScale-Off",
        "Abl-Norm-Off",
        "Abl-Precond-Off",
        "Abl-Precond-Off-DW"
    ]

    for exp_id in selected:
        candidates = sorted(
            RUN_ROOT.glob(
                f"**/{exp_id}/seed_*/logs.pth"
            )
        )
        if not candidates:
            continue

        logs_path = candidates[0]
        out_dir = (
            PLOT_ROOT /
            "training_history" /
            exp_id
        )
        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        plot_standardized_training_history(
            logs_path,
            out_dir / exp_id
        )



def plot_ablation_average_metrics(df):
    ablation = df[df["exp_group"] == "ablation"].copy()
    if ablation.empty:
        return

    order = ABLATION_ORDER
    ablation["exp_id"] = pd.Categorical(ablation["exp_id"], categories=order, ordered=True)
    ablation = ablation.sort_values("exp_id")

    mean_df, std_df = aggregate_mean_std(ablation, "exp_id", ["avg_rel_l2", "avg_mae", "avg_r2"])
    mean_df = mean_df.dropna(subset=["exp_id"])
    std_df = std_df.dropna(subset=["exp_id"])

    x = np.arange(len(mean_df))
    plt.figure(figsize=(13.5, 5))
    plt.bar(x, mean_df["avg_rel_l2"].values, yerr=std_df["avg_rel_l2"].values, capsize=4, alpha=0.85)
    plt.xticks(x, mean_df["exp_id"].astype(str).values, rotation=25, ha="right")
    plt.ylabel(r'Average $\varepsilon_{L_2}$')
    plt.title('Ablation Study of BP-PFNO Modules')
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    savefig(PLOT_ROOT / "ablation_avg_rel_l2")

    plt.figure(figsize=(13.5, 5))
    one_minus_r2 = 1.0 - mean_df["avg_r2"].values
    one_minus_r2_std = std_df["avg_r2"].values
    plt.bar(x, one_minus_r2, yerr=one_minus_r2_std, capsize=4, alpha=0.85)
    plt.xticks(x, mean_df["exp_id"].astype(str).values, rotation=25, ha="right")
    plt.ylabel(r'$1-\overline{R^2}$')
    plt.title('Ablation Study: Coefficient of Determination')
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    savefig(PLOT_ROOT / "ablation_one_minus_r2")


def plot_ablation_response_breakdown(df):
    ablation = df[df["exp_group"] == "ablation"].copy()
    if ablation.empty:
        return

    order = ABLATION_ORDER
    ablation = ablation[ablation["exp_id"].isin(order)].copy()
    if ablation.empty:
        return

    rel_cols = [f"{k}_rel_l2" for k in RESPONSE_KEYS]
    agg = ablation.groupby("exp_id")[rel_cols].mean().reindex(order).dropna(how="all")

    x = np.arange(len(agg.index))
    width = 0.18
    plt.figure(figsize=(14.5, 5.5))
    for i, (key, label) in enumerate(zip(RESPONSE_KEYS, RESPONSE_LABELS)):
        plt.bar(x + (i - 1.5) * width, agg[f"{key}_rel_l2"].values, width=width, label=label)
    plt.xticks(x, agg.index.astype(str), rotation=25, ha="right")
    plt.ylabel(r'Relative $L_2$ error')
    plt.title('Response-wise Error Breakdown in Ablation Study')
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=False, ncol=4)
    savefig(PLOT_ROOT / "ablation_response_rel_l2_breakdown")



def plot_ablation_performance_matrix(df):
    ablation = df[df["exp_group"] == "ablation"].copy()
    if ablation.empty:
        return

    order = ABLATION_ORDER
    metrics = [
        "train_time_s", "lbfgs_final_physics_loss",
        "avg_mae", "avg_rel_l2", "avg_r2"
    ]
    labels = [
        "Training\ntime", "Final raw\nphysics loss",
        "Average\n$\\overline{\\mathrm{MAE}}$",
        "Average\n$\\overline{\\varepsilon}_{L_2}$",
        "Average\n$\\overline{R}^{\\,2}$"
    ]

    agg = ablation.groupby("exp_id")[metrics].mean().reindex(order).dropna(how="all")
    if agg.empty:
        return

    values = agg.copy()
    values["avg_r2"] = 1.0 - values["avg_r2"]
    values = values.replace([np.inf, -np.inf], np.nan).fillna(values.max(numeric_only=True))

    arr = values.values.astype(float)
    arr[:, :-1] = np.maximum(arr[:, :-1], 1e-20)
    arr[:, -1] = np.maximum(arr[:, -1], 1e-20)
    log_arr = np.log10(arr)


    norm_arr = np.zeros_like(log_arr)
    for j in range(log_arr.shape[1]):
        col = log_arr[:, j]
        denom = np.nanmax(col) - np.nanmin(col)
        norm_arr[:, j] = 0.0 if denom < 1e-12 else (col - np.nanmin(col)) / denom


    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_ablation",
        ["#FFFFD9", "#44B7C4", "#081D58"]
    )
    plt.figure(figsize=(13.5, 5.8))
    im = plt.imshow(norm_arr, aspect="auto", cmap=custom_cmap)
    plt.colorbar(im, label='Normalized score')
    plt.yticks(np.arange(len(agg.index)), agg.index.astype(str), rotation=25)
    plt.xticks(np.arange(len(labels)), labels)
    plt.title('Normalized Multi-metric Performance Matrix for Ablation Study')

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            text_val = values.iloc[i, j]
            if metrics[j] == "train_time_s":
                txt = f"{text_val:.1f}"
            elif metrics[j] == "avg_r2":
                txt = f"{1.0 - text_val:.4f}"
            else:
                txt = f"{text_val:.1e}"
            rgba = im.cmap(im.norm(norm_arr[i, j]))
            r, g, b = rgba[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"

            plt.text(
                j, i, txt,
                ha="center",
                va="center",
                fontsize=16,
                color=text_color
            )

    savefig(PLOT_ROOT / "ablation_performance_matrix")



def plot_parameter_sensitivity(df):

    epoch_color = "#C50303"          
    epoch2_color = "#5C5959"         
    epoch_shadow_color = "#F39678"  

    sensor_color = "#081D58" 
    sensor2_color = "#5C5959"  
    sensor_shadow_color = "#56BEC6" 

    shadow_alpha = 0.25

    line_width = 1.6
    error_width = 1.1
    marker_size = 4
    cap_size = 3

    figure_size = (10, 3.8)
    legend_n = 3


    sensor = df[df["exp_group"] == "sensor"].copy()
    epoch = df[df["exp_group"] == "epoch"].copy()

    if sensor.empty or epoch.empty:
        print("Sensor or epoch sensitivity data are missing.")
        return

    sensor_agg = (
        sensor.groupby("m_sensors")["avg_rel_l2"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna(0.0)
        .sort_values("m_sensors")
    )

    epoch_agg = (
        epoch.groupby("adam_epochs")["avg_rel_l2"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna(0.0)
        .sort_values("adam_epochs")
    )

    sensor_x = sensor_agg["m_sensors"].to_numpy(dtype=float)
    sensor_mean = sensor_agg["mean"].to_numpy(dtype=float)
    sensor_std = sensor_agg["std"].to_numpy(dtype=float)

    epoch_x = epoch_agg["adam_epochs"].to_numpy(dtype=float)
    epoch_mean = epoch_agg["mean"].to_numpy(dtype=float)
    epoch_std = epoch_agg["std"].to_numpy(dtype=float)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figure_size,
        sharey=True
    )

    legend_label = "Mean $\\pm$ SD (n={})".format(legend_n)

    axes[0].fill_between(
        epoch_x,
        np.maximum(epoch_mean - epoch_std, 0.0),
        epoch_mean + epoch_std,
        color=epoch_shadow_color,
        alpha=shadow_alpha,
        linewidth=0,
        zorder=1
    )

    axes[0].errorbar(
        epoch_x,
        epoch_mean,
        yerr=epoch_std,
        color=epoch_color,
        ecolor=epoch2_color,
        marker="o",
        markersize=marker_size,
        markerfacecolor="white",
        markeredgecolor=epoch2_color,
        markeredgewidth=1.1,
        linewidth=line_width,
        elinewidth=error_width,
        capsize=cap_size,
        capthick=error_width,
        label=legend_label,
        zorder=2
    )

    axes[0].set_xlabel("Adam training epochs")
    axes[0].set_ylabel(
        r"Average $\overline{\varepsilon}_{L_2}$"
    )
    axes[0].set_xticks(epoch_x)

    axes[1].fill_between(
        sensor_x,
        np.maximum(sensor_mean - sensor_std, 0.0),
        sensor_mean + sensor_std,
        color=sensor_shadow_color,
        alpha=shadow_alpha,
        linewidth=0,
        zorder=1
    )

    axes[1].errorbar(
        sensor_x,
        sensor_mean,
        yerr=sensor_std,
        color=sensor_color,
        ecolor=sensor2_color,
        marker="o",
        markersize=marker_size,
        markerfacecolor="white",
        markeredgecolor=sensor2_color,
        markeredgewidth=1.1,
        linewidth=line_width,
        elinewidth=error_width,
        capsize=cap_size,
        capthick=error_width,
        label=legend_label,
        zorder=2
    )

    axes[1].set_xlabel("Number of load sensors")
    axes[1].set_xticks(sensor_x)

    for ax, panel_label in zip(axes, [" ", " "]):
        ax.text(
            0.025,
            0.96,
            panel_label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left"
        )

        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=10
        )

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.7,
            color="#BFBFBF",
            alpha=0.55
        )

        ax.set_axisbelow(True)
        ax.margins(x=0.04)

        ax.tick_params(
            axis="both",
            direction="out",
            labelsize=13,
            width=0.8,
            length=3.5
        )

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    axes[0].ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(-2, -2),
        useMathText=True
    )
    axes[0].yaxis.get_offset_text().set_fontsize(10)
    plt.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.22,
        top=0.95,
        wspace=0.12
    )

    savefig(PLOT_ROOT / "parameter_sensitivity_combined")

def plot_sensor_sensitivity(df):
    sensor = df[df["exp_group"] == "sensor"].copy()
    if sensor.empty:
        return

    agg = sensor.groupby("m_sensors").agg(
        avg_rel_l2_mean=("avg_rel_l2", "mean"),
        avg_rel_l2_std=("avg_rel_l2", "std"),
        avg_mae_mean=("avg_mae", "mean"),
        train_time_mean=("train_time_s", "mean"),
        train_time_std=("train_time_s", "std")
    ).reset_index().fillna(0.0)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        agg["m_sensors"], agg["avg_rel_l2_mean"],
        yerr=agg["avg_rel_l2_std"], marker="o", linewidth=2, capsize=4
    )
    plt.xlabel('Number of Load Sensors')
    plt.ylabel(r'Average $\varepsilon_{L_2}$')
    plt.title('Sensitivity to Load Sensor Number')
    plt.grid(True, linestyle='--', alpha=0.5)
    savefig(PLOT_ROOT / "sensor_sensitivity_avg_rel_l2")

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        agg["m_sensors"], agg["train_time_mean"],
        yerr=agg["train_time_std"], marker="s", linewidth=2, capsize=4
    )
    plt.xlabel('Number of Load Sensors')
    plt.ylabel('Training Time (s)')
    plt.title('Training Cost versus Load Sensor Number')
    plt.grid(True, linestyle='--', alpha=0.5)
    savefig(PLOT_ROOT / "sensor_sensitivity_training_time")


def plot_epoch_sensitivity(df):
    epoch = df[df["exp_group"] == "epoch"].copy()
    if epoch.empty:
        return

    agg = epoch.groupby("adam_epochs").agg(
        avg_rel_l2_mean=("avg_rel_l2", "mean"),
        avg_rel_l2_std=("avg_rel_l2", "std"),
        adam_loss_mean=("adam_final_loss", "mean"),
        lbfgs_loss_mean=("lbfgs_final_loss", "mean"),
        train_time_mean=("train_time_s", "mean"),
        train_time_std=("train_time_s", "std")
    ).reset_index().fillna(0.0)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        agg["adam_epochs"], agg["avg_rel_l2_mean"],
        yerr=agg["avg_rel_l2_std"], marker="o", linewidth=2, capsize=4
    )
    plt.xlabel('Adam Epochs')
    plt.ylabel(r'Average $\varepsilon_{L_2}$')
    plt.title('Sensitivity to Adam Training Epochs')
    plt.grid(True, linestyle='--', alpha=0.5)
    savefig(PLOT_ROOT / "epoch_sensitivity_avg_rel_l2")

    plt.figure(figsize=(10, 6))
    plt.semilogy(agg["adam_epochs"], agg["adam_loss_mean"], marker="o", linewidth=2, label='Adam final loss')
    plt.semilogy(agg["adam_epochs"], agg["lbfgs_loss_mean"], marker="s", linewidth=2, linestyle='--', label='L-BFGS final loss')
    plt.xlabel('Adam Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Final Loss versus Adam Training Epochs')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend(frameon=False)
    savefig(PLOT_ROOT / "epoch_sensitivity_final_losses")

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        agg["adam_epochs"], agg["train_time_mean"],
        yerr=agg["train_time_std"], marker="s", linewidth=2, capsize=4
    )
    plt.xlabel('Adam Epochs')
    plt.ylabel('Training Time (s)')
    plt.title('Training Cost versus Adam Epochs')
    plt.grid(True, linestyle='--', alpha=0.5)
    savefig(PLOT_ROOT / "epoch_sensitivity_training_time")


def plot_accuracy_efficiency_tradeoff(df):
    selected = df[df["exp_group"].isin(["ablation", "sensor", "epoch"])].copy()
    if selected.empty:
        return

    agg = selected.groupby(["exp_group", "exp_id"]).agg(
        train_time_s=("train_time_s", "mean"),
        avg_rel_l2=("avg_rel_l2", "mean")
    ).reset_index()

    plt.figure(figsize=(10, 6))
    markers = {"ablation": "o", "sensor": "s", "epoch": "^"}
    for group, sub in agg.groupby("exp_group"):
        plt.scatter(sub["train_time_s"], sub["avg_rel_l2"], s=80, marker=markers.get(group, "o"), label=group)
        for _, row in sub.iterrows():
            label = str(row["exp_id"]).replace("Abl_", "").replace("Abl-", "")
            plt.text(row["train_time_s"], row["avg_rel_l2"], label, fontsize=8, alpha=0.85)
    plt.xlabel('Training Time (s)')
    plt.ylabel(r'Average $\varepsilon_{L_2}$')
    plt.title('Accuracy-efficiency Trade-off')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=False)
    savefig(PLOT_ROOT / "accuracy_efficiency_tradeoff")


def _find_prediction_file(exp_id):
    candidates = sorted(RUN_ROOT.glob(f"**/{exp_id}/seed_*/canonical_predictions.npz"))
    return candidates[0] if candidates else None


def _load_prediction_records(exp_id):
    path = _find_prediction_file(exp_id)
    if path is None:
        return None, None
    data = np.load(path, allow_pickle=True)
    return np.asarray(data["x"], dtype=float), list(data["records"])


def _plot_response_group(exp_ids, group_name):
    loaded = []
    x_ref = None
    for exp_id in exp_ids:
        x, records = _load_prediction_records(exp_id)
        if records is None:
            continue
        loaded.append((exp_id, records))
        if x_ref is None:
            x_ref = x
    if not loaded:
        return

    n_cases = len(loaded[0][1])
    for case_idx in range(n_cases):
        rec0 = loaded[0][1][case_idx]
        case_id = rec0["case_id"]
        P = float(rec0["P"])
        xp = float(rec0["xp"])

        plt.figure(figsize=(10, 3))
        plt.plot(x_ref, rec0["q_domain"], linewidth=2.5, label=fr'$P$={P:.2f}, $x_p$={xp:.2f}')
        plt.fill_between(x_ref, rec0["q_domain"], alpha=0.12)
        plt.hlines(0, 0, 1.0, color='black', linewidth=2)
        plt.xlabel('Beam Position $(x)$')
        plt.ylabel('Load Input $q(x)$')
        plt.title(f'Gaussian Localized Load: {group_name} - {case_id}')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(frameon=False)
        savefig(PLOT_ROOT / 'representative_response' / group_name / f'{case_id}_load')

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
        axes = axes.flatten()
        quantity_info = [
            ('u', 'Deflection $(u)$'),
            ('phi', 'Rotation $(\\phi)$'),
            ('M', 'Moment $(M)$'),
            ('Q', 'Shear Force $(Q)$'),
        ]
        for ax, (key, ylabel) in zip(axes, quantity_info):
            ax.plot(x_ref, rec0[f'{key}_fem'], color='black', linewidth=3.0, alpha=0.35, label='FEM')
            for exp_id, records in loaded:
                rec = records[case_idx]
                label = exp_id.replace('Abl_', '').replace('Abl-', '')
                ax.plot(x_ref, rec[f'{key}_pred'], linestyle='--', linewidth=1.8, label=label)
            if key == 'M':
                ax.invert_yaxis()
            ax.hlines(0, 0, 1.0, color='black', linewidth=1.8, alpha=0.75)
            ax.set_xlabel('Beam Position $(x)$')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel)
            ax.grid(True, linestyle='--', alpha=0.4)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)), frameon=False)
        fig.suptitle(f'Representative Response Comparison: {group_name} - {case_id}', y=1.02)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = PLOT_ROOT / 'representative_response' / group_name / f'{case_id}_responses'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix('.png'), dpi=500, bbox_inches='tight')
        fig.savefig(out_path.with_suffix('.pdf'), dpi=500, bbox_inches='tight')
        plt.close(fig)


def plot_representative_response_comparison():
    groups = {
        'ablation_key_modules': [
            'Base', 'Abl-BC-None', 'Abl-LoadScale-Off', 'Abl-Precond-Off',
            'Abl-Norm-Off'
        ],
        'precond_dynamic_pair': [
            'Base', 'Abl-Precond-Off', 'Abl-Precond-Off-DW'
        ],
        'sensor_resolution': ['Base', 'Sensor_11', 'Sensor_51', 'Sensor_81'],
        'epoch_budget': ['Base', 'Epoch_200', 'Epoch_600', 'Epoch_1000'],
    }
    for group_name, exp_ids in groups.items():
        _plot_response_group(exp_ids, group_name)

def export_publication_tables(df):
    table_dir = RUN_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    main_cols = [
        "exp_group", "exp_id", "seed", "m_sensors", "adam_epochs",
        "use_dynamic_weights", "loss_weighting",
        "train_time_s", "adam_final_loss", "adam_final_physics_loss",
        "adam_final_bc_loss",
        "lbfgs_final_loss", "lbfgs_final_physics_loss",
        "lbfgs_final_bc_loss",
        "avg_mae", "avg_rel_l2", "avg_r2",
        "u_rel_l2", "phi_rel_l2", "M_rel_l2", "Q_rel_l2",
        "geo_res", "const_res", "eq1_res", "eq2_res"
    ]
    existing_cols = [c for c in main_cols if c in df.columns]
    df[existing_cols].to_csv(table_dir / "Table_3_3_all_runs_summary.csv", index=False, encoding="utf-8-sig")

    group_cols = ["exp_group", "exp_id"]
    value_cols = [
        "train_time_s", "adam_final_loss", "adam_final_physics_loss",
        "adam_final_bc_loss",
        "lbfgs_final_loss", "lbfgs_final_physics_loss",
        "lbfgs_final_bc_loss",
        "avg_mae", "avg_rel_l2", "avg_r2",
        "u_rel_l2", "phi_rel_l2", "M_rel_l2", "Q_rel_l2"
    ]
    value_cols = [c for c in value_cols if c in df.columns]
    agg_mean = df.groupby(group_cols)[value_cols].mean().reset_index()
    agg_std = df.groupby(group_cols)[value_cols].std().reset_index()
    for col in value_cols:
        if col in agg_std.columns:
            agg_std[col] = agg_std[col].fillna(0.0)
    agg_mean.to_csv(table_dir / "Table_3_3_mean_summary.csv", index=False, encoding="utf-8-sig")
    agg_std.to_csv(table_dir / "Table_3_3_std_summary.csv", index=False, encoding="utf-8-sig")


def main():
    df = load_summary()
    export_publication_tables(df)

    plot_selected_training_histories()
    plot_ablation_average_metrics(df)
    plot_ablation_response_breakdown(df)
    plot_ablation_performance_matrix(df)
    plot_parameter_sensitivity(df)
    plot_sensor_sensitivity(df)
    plot_epoch_sensitivity(df)
    plot_accuracy_efficiency_tradeoff(df)
    plot_representative_response_comparison()

    print("\n Finish：")
    print(f"  Figures: {PLOT_ROOT}")
    print(f"  Tables : {RUN_ROOT / 'tables'}")


if __name__ == '__main__':
    main()
