from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUPPLEMENTARY_DIR = OUTPUT_DIR / "supplementary"

DISPLAY_MODEL_NAME = "BE-PFNO"
RESPONSE_KEYS = ("u", "phi", "M", "Q")
RESPONSE_TITLES = {
    "u": "Deflection $u$",
    "phi": "Rotation $\\varphi$",
    "M": "Moment $M$",
    "Q": "Shear force $Q$",
}
CASE_COLORS = ("#4583B6", "#B02425", "#218D42")

CUSTOM_SCENE_LABELS = {
    "SS-linear": "Simply supported linear load",
    "SS-quadratic": "Simply supported quadratic load",
    "SS-concentrated": "Simply supported Gaussian local load",
    "FF-linear": "Fixed-fixed linear load",
    "FF-quadratic": "Fixed-fixed quadratic load",
    "FF-concentrated": "Fixed-fixed Gaussian local load",
    "CF-linear": "Cantilever linear load",
    "CF-quadratic": "Cantilever quadratic load",
    "CF-concentrated": "Cantilever Gaussian local load",
    "FF-settlement": "Fixed-fixed left settlement",
    "FF-rotation": "Fixed-fixed left rotation",
}
SCENE_ID_TO_DISPLAY_NAME = {
    "ss_linear": "SS-linear",
    "ss_quadratic": "SS-quadratic",
    "ss_concentrated": "SS-concentrated",
    "ff_linear": "FF-linear",
    "ff_quadratic": "FF-quadratic",
    "ff_concentrated": "FF-concentrated",
    "cf_linear": "CF-linear",
    "cf_quadratic": "CF-quadratic",
    "cf_concentrated": "CF-concentrated",
    "settlement": "FF-settlement",
    "rotation": "FF-rotation",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot section 4 multi-scenario BE-PFNO results.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: str) -> float:
    return float(value) if value not in {"", None} else float("nan")


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.dpi": 500,
        "savefig.dpi": 500,
        "font.size": 10,
    })



def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def apply_readable_ylim(ax: plt.Axes, *arrays: np.ndarray, include_zero: bool = False) -> None:
    values = np.concatenate([np.ravel(array) for array in arrays])
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.concatenate([values, np.array([0.0])])
    if values.size == 0:
        return
    ymin = float(np.min(values))
    ymax = float(np.max(values))
    span = ymax - ymin
    abs_scale = float(np.max(np.abs(values)))
    min_span = max(0.12 * abs_scale, 1e-6)
    if span < min_span:
        center = 0.5 * (ymin + ymax)
        half_span = 0.5 * min_span
        ax.set_ylim(center - half_span, center + half_span)


def draw_beam_boundary_axis(ax: plt.Axes, boundary_condition: str, color: str = "black") -> None:
    beam_left = 0.0
    beam_right = 1.0
    x_margin = 0.05
    ax.hlines(0.0, beam_left, beam_right, color=color, linewidth=2.4, zorder=8)
    if boundary_condition == "simply-supported":
        ax.plot([beam_left, beam_right], [0.0, 0.0], linestyle="", marker="^",
                markersize=7.0, color=color, zorder=20, clip_on=False)
    elif boundary_condition == "cantilever":
        ax.plot([beam_left], [0.0], linestyle="", marker="s",
                markersize=6.5, color=color, zorder=20, clip_on=False)
    else:
        ax.plot([beam_left, beam_right], [0.0, 0.0], linestyle="", marker="s",
                markersize=6.5, color=color, zorder=20, clip_on=False)
    ax.set_xlim(beam_left - x_margin, beam_right + x_margin)
    ax.set_xticks(np.linspace(beam_left, beam_right, 6))


def group_curves_by_scene(curves: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    ordered_scene_ids: list[str] = []
    for item in curves:
        scene_id = item["scene_id"]
        if scene_id not in grouped:
            grouped[scene_id] = []
            ordered_scene_ids.append(scene_id)
        grouped[scene_id].append(item)
    for scene_id in ordered_scene_ids:
        grouped[scene_id].sort(key=lambda item: int(item.get("representative_case_index", 1)))
    return [grouped[scene_id] for scene_id in ordered_scene_ids]


def plot_representative_comparison(curves: list[dict[str, Any]], figure_dir: Path) -> Path:
    selected_groups = group_curves_by_scene(curves)
    if not selected_groups:
        raise ValueError("No representative curves are available for plotting.")

    fig, axes = plt.subplots(
        len(selected_groups),          # 行：不同边界-荷载场景
        len(RESPONSE_KEYS),            # 列：不同响应场
        figsize=(3.2 * len(RESPONSE_KEYS), 1.45 * len(selected_groups)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )

    for row, scene_group in enumerate(selected_groups):
        scene_item = scene_group[0]
        for col, response in enumerate(RESPONSE_KEYS):
            ax = axes[row, col]
            ylim_series: list[np.ndarray] = []
            for case_pos, item in enumerate(scene_group):
                x = np.asarray(item["x"], dtype=float)
                fem = np.asarray(item["fem"][response], dtype=float)
                pred = np.asarray(item["pred"][response], dtype=float)
                color = CASE_COLORS[case_pos % len(CASE_COLORS)]
                case_index = int(item.get("representative_case_index", case_pos + 1))
                show_legend_label = row == 0 and col == 0
                ax.plot(
                    x, fem, color=color, alpha=0.35, linewidth=3.2,
                    label=f"Case {case_index} FEM" if show_legend_label else None,
                )
                ax.plot(
                    x, pred, color=color, linewidth=1.3, linestyle="--",
                    label=f"Case {case_index} {DISPLAY_MODEL_NAME}" if show_legend_label else None,
                )
                ylim_series.extend([fem, pred])
            apply_readable_ylim(ax, *ylim_series, include_zero=True)
            draw_beam_boundary_axis(ax, scene_item["boundary_condition"])
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
            # 第一列标注工况名称
            if col == 0:
                scene_name = SCENE_ID_TO_DISPLAY_NAME.get(
                    scene_item["scene_id"],
                    scene_item["scene_id"],
                )

                ax.set_ylabel(scene_name, fontsize=14)      

            # 第一行标注响应场名称
            if row == 0:
                ax.set_title(RESPONSE_TITLES[response], fontsize=14)
            if row == len(selected_groups) - 1:
                ax.set_xlabel("Beam Position ($x$)", fontsize=14)
            if response == "M":
                ax.invert_yaxis()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.025))
    fig.suptitle("Representative multi-scenario response comparison", y=1.055, fontsize=14)
    path = figure_dir / "section4_multiscenario_fem_comparison.png"
    save_figure(fig, path)
    return path


def plot_rel_l2_heatmap(summary_rows: list[dict[str, str]], figure_dir: Path) -> Path:
    scene_labels = [SCENE_ID_TO_DISPLAY_NAME.get(row["scene_id"], row["scene_id"],) for row in summary_rows]
    matrix = np.array(
        [[to_float(row[f"{response}_mean_rel_l2"]) for response in RESPONSE_KEYS] for row in summary_rows],
        dtype=float,
    )
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    vmin = max(float(np.min(positive)) if positive.size else 1e-8, 1e-8)
    vmax = max(float(np.max(positive)) if positive.size else 1.0, vmin * 10.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.0), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_xticks(np.arange(len(RESPONSE_KEYS)))
    ax.set_xticklabels(["$u$", "$\phi$", "$M$", "$Q$"], fontsize=12)
  
    ax.set_yticks(np.arange(len(scene_labels)))
    ax.set_yticklabels(scene_labels, fontsize=12)
    ax.set_title(r"LHS-100 Average$\overline{\varepsilon}_{L_2}$ by condition and response", fontsize=12)
    ax.set_xlabel("Response component", fontsize=12)
    ax.set_ylabel("Boundary-load condition", fontsize=12)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = f"{value:.1e}" if np.isfinite(value) else "NA"
            rgba = im.cmap(im.norm(value)) if np.isfinite(value) else (1.0, 1.0, 1.0, 1.0)
            r, g, b = rgba[:3]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.4 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=12, color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Average$\overline{\varepsilon}_{L_2}$ (log scale)", fontsize=12)
    cbar.ax.tick_params(labelsize=12) 
    path = figure_dir / "section4_lhs100_rel_l2_heatmap.png"
    save_figure(fig, path)
    return path


def plot_summary_table(summary_rows: list[dict[str, str]], figure_dir: Path) -> Path:
    headers = ["Scenario", "Mean Rel-L2", "Std", "Max", "Max BC violation"]
    table_rows = []
    for row in summary_rows:
        table_rows.append([
            row["scene_label"],
            f"{to_float(row['mean_rel_l2_all']):.3e}",
            f"{to_float(row['std_rel_l2_all']):.3e}",
            f"{to_float(row['max_rel_l2_all']):.3e}",
            f"{to_float(row['max_boundary_violation']):.3e}",
        ])

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.42, 0.16, 0.12, 0.12, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D9E2F3")
        if row_idx == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(weight="bold", color="white", ha="center")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#F5F8FC")
        if col_idx > 0 and row_idx > 0:
            cell.set_text_props(ha="right")

    ax.set_title("LHS-100 scenario-level Relative $L^2$ summary", fontsize=14, pad=16)
    path = figure_dir / "section4_lhs100_summary_table.png"
    save_figure(fig, path)
    return path


def plot_supplementary_curves(curves: list[dict[str, Any]], supplementary_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for item in curves:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), constrained_layout=True)
        flat_axes = axes.flatten()
        x = np.asarray(item["x"], dtype=float)
        for ax, response in zip(flat_axes, RESPONSE_KEYS):
            fem = np.asarray(item["fem"][response], dtype=float)
            pred = np.asarray(item["pred"][response], dtype=float)
            err = np.abs(pred - fem)
            ax.plot(x, fem, color="#1F4E79", linewidth=1.9, label="Hermite FEM")
            ax.plot(x, pred, color="#C0321A", linestyle="--", linewidth=1.5, label=DISPLAY_MODEL_NAME)
            ax.fill_between(x, 0.0, err, color="#F4B183", alpha=0.25, label="Abs. error")
            ax.set_title(RESPONSE_TITLES[response], fontsize=8)
            ax.set_xlabel("Beam coordinate $x/L$", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
            if response == "M":
                ax.invert_yaxis()
        handles, labels = flat_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(f"{item['scene_label']} | {item['parameter_label']}", y=1.055, fontsize=8)
        case_index = int(item.get("representative_case_index", 1))
        path = supplementary_dir / f"supplementary_{item['scene_id']}_case{case_index}_representative_comparison.png"
        save_figure(fig, path)
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    supplementary_dir = output_dir / "supplementary"
    figure_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    summary_path = output_dir / "section4_lhs100_summary.csv"
    curves_path = output_dir / "section4_representative_curves.json"
    if not summary_path.exists() or not curves_path.exists():
        raise FileNotFoundError("Run run_multiscenario_lhs100.py before plotting.")

    summary_rows = read_csv_rows(summary_path)
    curves = load_json(curves_path)

    generated = [
        plot_representative_comparison(curves, figure_dir),
        plot_rel_l2_heatmap(summary_rows, figure_dir),
        plot_summary_table(summary_rows, figure_dir),
    ]
    supplementary = plot_supplementary_curves(curves, supplementary_dir)

    manifest = {
        "model_label": DISPLAY_MODEL_NAME,
        "main_figures": [str(path.relative_to(output_dir)) for path in generated],
        "supplementary_figures": [str(path.relative_to(output_dir)) for path in supplementary],
    }
    (output_dir / "section4_figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Generated main figures:")
    for path in generated:
        print(f"  {path}")
    print(f"Generated {len(supplementary)} supplementary figures.")


if __name__ == "__main__":
    main()
