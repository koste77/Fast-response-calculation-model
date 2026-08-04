import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

TRAINING_SCRIPT = "BEP-PINO_Model_cantilever_quadratic.py"
TARGET_RELATIVE_DIR = Path("BEP-PINO Model comparison under multiple operating conditions/Cantilever/Cantilever quadratic load")
MODEL_SAVE_PATH = "BEP-PINO_Model_Quadratic_2Param_cantilever.pth"
LOGS_SAVE_PATH = "BEP-PINO_Logs_Quadratic_2Param_cantilever.pth"
LOAD_KIND = "quadratic"
BC_TYPE = "cantilever"
BOUNDARY_TITLE = "Cantilever"
LOAD_TITLE = "Symmetric Quadratic Load"
INPUT_YLABEL = "Load Input $q(x)$"


def resolve_base_dir():
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    if (cwd / TRAINING_SCRIPT).exists():
        return cwd
    for root in (cwd, *cwd.parents):
        candidate = root / TARGET_RELATIVE_DIR
        if (candidate / TRAINING_SCRIPT).exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot locate {TARGET_RELATIVE_DIR / TRAINING_SCRIPT} from {cwd}")


BASE_DIR = resolve_base_dir()


def load_training_module():
    script_path = BASE_DIR / TRAINING_SCRIPT
    spec = importlib.util.spec_from_file_location("bp_pfno_training_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module specification for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training_module = load_training_module()
FNO_Combined = training_module.FNO_Combined

model_path = BASE_DIR / MODEL_SAVE_PATH
logs_path = BASE_DIR / LOGS_SAVE_PATH
print(f"▶ Loading model data: {model_path.name} ...")
model_data = torch.load(model_path, weights_only=False)
logs_data = torch.load(logs_path, weights_only=False)

L_val = model_data["L_val"]
EI_val = model_data["EI_val"]
m_sensors = model_data["m_sensors"]
model = FNO_Combined(
    m_sensors=m_sensors,
    modes=model_data["modes"],
    width=model_data["width"],
    hidden_dim=model_data.get("hidden_dim", 128),
)
model.load_state_dict(model_data["model_state_dict"])
model.eval()

loss_history = logs_data["loss_history"]
raw_total_loss_history = np.asarray(logs_data["raw_total_loss_history"], dtype=float)
lbfgs_loss_history = logs_data["lbfgs_loss_history"]


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.dpi": 500,
    "savefig.dpi": 500,
    "font.size": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})


def plot_standardized_training_history(logs):
    components = [
        ("geo", "Geometric"),
        ("const", "Constitutive"),
        ("eq1", "Equilibrium 1"),
        ("eq2", "Equilibrium 2"),
    ]
    total_adam_color = "#C0321A"
    total_lbfgs_color = "#218D42"
    component_colors = {
        "geo": "#8074C8",
        "const": "#7895C1",
        "eq1": "#59A14F",
        "eq2": "#EF8B67",
    }

    def stages(adam_key, lbfgs_key, label, adam_color=None, lbfgs_color=None):
        adam = np.asarray(logs.get(adam_key, []), dtype=float)
        lbfgs = np.asarray(logs.get(lbfgs_key, []), dtype=float)
        if adam.size:
            plt.semilogy(
                np.arange(adam.size), adam, color=adam_color,
                linewidth=1.8, linestyle="-", label=f"Adam {label}",
            )
        if lbfgs.size:
            plt.semilogy(
                np.arange(adam.size, adam.size + lbfgs.size), lbfgs,
                color=lbfgs_color, linewidth=2.0, linestyle="--",
                label=f"L-BFGS {label}",
            )

    plt.figure(figsize=(10, 6))
    stages(
        "loss_history", "lbfgs_loss_history", "Total Physics Loss",
        adam_color=total_adam_color, lbfgs_color=total_lbfgs_color,
    )
    plt.title("Total Physics Loss")
    plt.ylabel("Loss (Log Scale)", fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for key, label in components:
        stages(
            f"loss_{key}_history", f"lbfgs_loss_{key}_history", label,
            adam_color=component_colors[key], lbfgs_color=component_colors[key],
        )
    plt.title("Physics Loss Components")
    plt.ylabel("Loss (Log Scale)", fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_standardized_training_history(logs_data)

plt.figure(figsize=(10, 6))
plt.semilogy(loss_history, color="#C0321A", linewidth=1.5, label="Total Physics Loss")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss (Log Scale)", fontsize=16)
plt.title("Training Convergence Curve (FNO Model)", fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(raw_total_loss_history, color="#C0321A", linewidth=1.5, label="raw_total_loss_history")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss (Log Scale)", fontsize=16)
plt.title("Training Convergence Curve (FNO Model)", fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
len_adam = len(loss_history)
len_lbfgs = len(lbfgs_loss_history)
x_adam = np.arange(len_adam)
x_lbfgs = np.arange(len_adam, len_adam + len_lbfgs)
plt.semilogy(x_adam, loss_history, color="#C0321A", linewidth=1.5, label="Adam Stage (Epochs)")
plt.semilogy(x_lbfgs, lbfgs_loss_history, color="#218D42", linewidth=2.0, label="L-BFGS Stage (Evaluations)")
plt.axvline(x=len_adam, color="gray", linestyle="--", linewidth=1.5, label="Optimizer Switch")
ax = plt.gca()
trans = ax.get_xaxis_transform()
plt.text(
    len_adam * 0.5, 0.1, "Global Search\n(1st Order)",
    horizontalalignment="center", color="#C0321A", fontsize=14,
    alpha=0.8, transform=trans,
)
plt.text(
    len_adam + len_lbfgs * 0.5, 0.1, "Fine-tuning\n(2nd Order)",
    horizontalalignment="center", color="#218D42", fontsize=14,
    alpha=0.8, transform=trans,
)
plt.xlabel("Optimization Steps (Adam Epochs + L-BFGS Evaluations)", fontsize=16)
plt.ylabel("Total PDE Loss (Log Scale)", fontsize=16)
plt.title("Complete Training Convergence: Adam $\\rightarrow$ L-BFGS", fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc="upper right", fontsize=14)
plt.tight_layout()
plt.show()


def hermite_shape_functions(s, Le):
    r = s / Le
    return np.array([
        1.0 - 3.0 * r**2 + 2.0 * r**3,
        s * (1.0 - r)**2,
        3.0 * r**2 - 2.0 * r**3,
        s * (r**2 - r),
    ])


def beam_element_stiffness(EI, Le):
    return (EI / Le**3) * np.array([
        [12.0, 6.0 * Le, -12.0, 6.0 * Le],
        [6.0 * Le, 4.0 * Le**2, -6.0 * Le, 2.0 * Le**2],
        [-12.0, -6.0 * Le, 12.0, -6.0 * Le],
        [6.0 * Le, 2.0 * Le**2, -6.0 * Le, 4.0 * Le**2],
    ])


def consistent_element_load(x_left, Le, q_func, gauss_order=8):
    xi, wi = np.polynomial.legendre.leggauss(gauss_order)
    fe = np.zeros(4, dtype=np.float64)
    for xi_i, wi_i in zip(xi, wi):
        s = 0.5 * Le * (xi_i + 1.0)
        x_g = x_left + s
        fe += wi_i * hermite_shape_functions(s, Le) * q_func(x_g) * (Le / 2.0)
    return fe


def solve_beam_hermite_on_grid(
    q_func, x_target, bc_type, L=1.0, EI=1.0,
    num_nodes_ref=1000, gauss_order=8, settlement=0.0, rotation=0.0,
):
    if num_nodes_ref < 3:
        raise ValueError("num_nodes_ref must be at least 3")
    n_elem = num_nodes_ref - 1
    total_dof = 2 * num_nodes_ref
    x_ref = np.linspace(0.0, L, num_nodes_ref)
    Le = L / n_elem
    K = np.zeros((total_dof, total_dof), dtype=np.float64)
    F = np.zeros(total_dof, dtype=np.float64)
    ke = beam_element_stiffness(EI, Le)
    element_loads = []

    for e in range(n_elem):
        x_left = x_ref[e]
        fe = consistent_element_load(x_left, Le, q_func, gauss_order=gauss_order)
        element_loads.append(fe)
        dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
        K[np.ix_(dof, dof)] += ke
        F[dof] += fe

    if bc_type == "simply-supported":
        fixed_dofs = np.array([0, total_dof - 2])
        fixed_vals = np.array([0.0, 0.0])
    elif bc_type == "cantilever":
        fixed_dofs = np.array([0, 1])
        fixed_vals = np.array([0.0, 0.0])
    elif bc_type == "clamped-clamped-settlement":
        fixed_dofs = np.array([0, 1, total_dof - 2, total_dof - 1])
        fixed_vals = np.array([settlement, 0.0, 0.0, 0.0])
    elif bc_type == "clamped-clamped-rotation":
        fixed_dofs = np.array([0, 1, total_dof - 2, total_dof - 1])
        fixed_vals = np.array([0.0, rotation, 0.0, 0.0])
    elif bc_type == "fixed-fixed":
        fixed_dofs = np.array([0, 1, total_dof - 2, total_dof - 1])
        fixed_vals = np.zeros(4)
    else:
        raise ValueError(f"Unsupported boundary type: {bc_type}")

    free_dofs = np.setdiff1d(np.arange(total_dof), fixed_dofs)
    U = np.zeros(total_dof, dtype=np.float64)
    U[fixed_dofs] = fixed_vals
    rhs = F[free_dofs] - K[np.ix_(free_dofs, fixed_dofs)] @ fixed_vals
    U[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], rhs)

    u_ref = U[0::2]
    phi_ref = U[1::2]
    M_sum = np.zeros(num_nodes_ref, dtype=np.float64)
    Q_sum = np.zeros(num_nodes_ref, dtype=np.float64)
    count = np.zeros(num_nodes_ref, dtype=np.float64)
    for e in range(n_elem):
        dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
        p_e = ke @ U[dof] - element_loads[e]
        Q_left, M_left = p_e[0], -p_e[1]
        Q_right, M_right = -p_e[2], p_e[3]
        Q_sum[e] += Q_left
        M_sum[e] += M_left
        count[e] += 1.0
        Q_sum[e + 1] += Q_right
        M_sum[e + 1] += M_right
        count[e + 1] += 1.0

    M_ref = M_sum / np.maximum(count, 1.0)
    Q_ref = Q_sum / np.maximum(count, 1.0)
    x_target = np.asarray(x_target, dtype=np.float64).reshape(-1)
    return (
        np.interp(x_target, x_ref, u_ref),
        np.interp(x_target, x_ref, phi_ref),
        np.interp(x_target, x_ref, M_ref),
        np.interp(x_target, x_ref, Q_ref),
    )


def build_case(case):
    x_sensors = np.linspace(0.0, L_val, m_sensors)
    x_batch = torch.linspace(0.0, L_val, 100, dtype=torch.float64).reshape(1, -1, 1)
    x_np = x_batch.squeeze().numpy()
    sigma = 0.04
    if LOAD_KIND == "concentrated":
        P, xp = case
        sensor_values = P * np.exp(-((x_sensors - xp)**2) / (2.0 * sigma**2))
        input_plot = P * np.exp(-((x_np - xp)**2) / (2.0 * sigma**2))
        q_func = lambda x, P=P, xp=xp: P * np.exp(-((x - xp)**2) / (2.0 * sigma**2))
        label = f"$P$={P}, $x_p$={xp}"
        fem_kwargs = {}
    elif LOAD_KIND == "linear":
        q0, qL = case
        sensor_values = q0 + (qL - q0) * x_sensors / L_val
        input_plot = q0 + (qL - q0) * x_np / L_val
        q_func = lambda x, q0=q0, qL=qL: q0 + (qL - q0) * x / L_val
        label = f"$q_0$={q0}, $q_L$={qL}"
        fem_kwargs = {}
    elif LOAD_KIND == "quadratic":
        q_end, dq = case
        c0 = q_end
        c2 = 4.0 * dq / L_val**2
        c1 = -4.0 * dq / L_val
        sensor_values = c2 * x_sensors**2 + c1 * x_sensors + c0
        input_plot = c2 * x_np**2 + c1 * x_np + c0
        q_func = lambda x, c0=c0, c1=c1, c2=c2: c2 * x**2 + c1 * x + c0
        label = f"$q_e$={q_end}, $\\Delta q$={dq}"
        fem_kwargs = {}
    elif LOAD_KIND == "settlement":
        sensor_values = np.repeat(case, m_sensors)
        input_plot = np.full_like(x_np, case, dtype=float)
        q_func = lambda x: 0.0
        label = f"$s$={case}"
        fem_kwargs = {"settlement": case}
    elif LOAD_KIND == "rotation":
        sensor_values = np.repeat(case, m_sensors)
        input_plot = np.full_like(x_np, case, dtype=float)
        q_func = lambda x: 0.0
        label = f"$\\theta$={case}"
        fem_kwargs = {"rotation": case}
    else:
        raise ValueError(f"Unsupported load kind: {LOAD_KIND}")

    branch_input = torch.cat([
        torch.tensor([sensor_values], dtype=torch.float64),
        torch.tensor([[EI_val, L_val]], dtype=torch.float64),
    ], dim=1)
    return branch_input, x_batch, x_np, np.asarray(input_plot), q_func, label, fem_kwargs


def test_cases():
    if LOAD_KIND == "concentrated":
        return [(-3.0, 0.3), (-2.0, 0.5), (-2.5, 0.7)]
    if LOAD_KIND == "linear":
        return [(-0.15, -0.05), (-0.05, -0.20), (-0.15, -0.15)]
    if LOAD_KIND == "quadratic":
        return [(-0.20, 0.05), (-0.05, 0.10), (-0.12, 0.08)]
    if LOAD_KIND == "settlement":
        return [-0.08, -0.05, -0.02]
    if LOAD_KIND == "rotation":
        return [-0.08, 0.04, 0.08]
    raise ValueError(f"Unsupported load kind: {LOAD_KIND}")

def plot_support_symbols(ax, color="black"):
    if BC_TYPE == "simply-supported":
        ax.scatter([0.0, L_val], [0.0, 0.0], marker="^", s=100, color=color, zorder=20, clip_on=False)
    elif BC_TYPE == "cantilever":
        ax.scatter([0.0], [0.0], marker="s", s=90, color=color, zorder=20, clip_on=False)
    else:
        ax.scatter([0.0, L_val], [0.0, 0.0], marker="s", s=80, color=color, zorder=20, clip_on=False)


def summary_metrics(pred, fem):
    abs_error = np.abs(pred - fem)
    mae = np.mean(abs_error)
    l2_rel = np.linalg.norm(pred - fem) / (np.linalg.norm(fem) + 1e-12)
    ss_res = np.sum((fem - pred)**2)
    ss_tot = np.sum((fem - np.mean(fem))**2) + 1e-12
    return mae, l2_rel, 1.0 - ss_res / ss_tot


def detail_metrics(name, pred, fem):
    abs_error = np.abs(pred - fem)
    peak_fem_abs = np.max(np.abs(fem))
    peak_pred_abs = np.max(np.abs(pred))
    idx_max_error = int(np.argmax(abs_error))
    max_abs_error = abs_error[idx_max_error]
    peak_diff = np.abs(peak_pred_abs - peak_fem_abs)
    return {
        "name": name,
        "max_err": max_abs_error,
        "pred_at_err": pred[idx_max_error],
        "fem_at_err": fem[idx_max_error],
        "rel_err": max_abs_error / peak_fem_abs * 100 if peak_fem_abs > 1e-12 else 0.0,
        "peak_pred": peak_pred_abs,
        "peak_fem": peak_fem_abs,
        "peak_diff": peak_diff,
        "peak_rel": peak_diff / peak_fem_abs * 100 if peak_fem_abs > 1e-12 else 0.0,
    }


def describe_case(case_index, case):
    if LOAD_KIND == "concentrated":
        P, xp = case
        return f"工况 {case_index}: local load peak P = {P}, center position xp = {xp}, sigma = 0.04"
    if LOAD_KIND == "linear":
        q0, qL = case
        return f"工况 {case_index}: linear load q0 = {q0}, qL = {qL}"
    if LOAD_KIND == "quadratic":
        qe, dq = case
        return f"工况 {case_index}: symmetric quadratic load qe = {qe}, dq = {dq}"
    if LOAD_KIND == "settlement":
        return f"工况 {case_index}: support settlement s = {case}"
    return f"工况 {case_index}: support rotation theta = {case}"


colors = ["#4583b6", "#B02425", "#218D42"]
field_names = ["Deflection (u)", "Rotation (phi)", "Moment (M)", "Shear Force (Q)"]
field_ylabels = ["Deflection $(u)$", "Rotation $(\\phi)$", "Moment $(M)$", "Shear Force $(Q)$"]
x_sensors = np.linspace(0.0, L_val, m_sensors)

fig_q, ax_q = plt.subplots(figsize=(10, 3))
fig_u, ax_u = plt.subplots(figsize=(10, 3))
fig_phi, ax_phi = plt.subplots(figsize=(10, 3))
fig_M, ax_M = plt.subplots(figsize=(10, 3))
fig_Q, ax_Q = plt.subplots(figsize=(10, 3))
response_axes = [(fig_u, ax_u), (fig_phi, ax_phi), (fig_M, ax_M), (fig_Q, ax_Q)]
fig_err, ax_err_matrix = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
ax_err = ax_err_matrix.flatten()
global_metrics_summary = {
    name: {"mae": [], "l2_rel": [], "r2": []} for name in field_names
}
case_cache = []

for i, case in enumerate(test_cases()):
    branch_input, test_x, x_np, input_plot, q_func, label, fem_kwargs = build_case(case)
    ax_q.plot(x_np, input_plot, color=colors[i], label=label, linewidth=3, alpha=0.8)
    ax_q.fill_between(x_np, input_plot, color=colors[i], alpha=0.1)
    with torch.no_grad():
        pred_tensors = model.predict_with_ansatz(branch_input, test_x, L=L_val, bc_type=BC_TYPE)
    pred_fields = [value.squeeze().detach().cpu().numpy() for value in pred_tensors]
    fem_fields = solve_beam_hermite_on_grid(
        q_func, x_np, BC_TYPE, L=L_val, EI=EI_val,
        num_nodes_ref=1000, gauss_order=8, **fem_kwargs,
    )

    for j, ((fig, ax_response), name) in enumerate(zip(response_axes, field_names)):
        ax_response.plot(x_np, fem_fields[j], color=colors[i], label=f"FEM ({label})", alpha=0.4, linewidth=4)
        ax_response.plot(x_np, pred_fields[j], color=colors[i], linestyle="--", label=f"Pred ({label})")
        error = np.abs(pred_fields[j] - fem_fields[j])
        ax_err[j].plot(x_np, error, color=colors[i], label=f"Error ({label})", zorder=5 - i)
        ax_err[j].fill_between(x_np, error, color=colors[i], alpha=0.1)
        mae, l2_rel, r2 = summary_metrics(pred_fields[j], fem_fields[j])
        global_metrics_summary[name]["mae"].append(mae)
        global_metrics_summary[name]["l2_rel"].append(l2_rel)
        global_metrics_summary[name]["r2"].append(r2)
    case_cache.append((case, pred_fields, fem_fields))


plot_configs = [
    (fig_q, ax_q, INPUT_YLABEL, LOAD_TITLE),
    (fig_u, ax_u, "Deflection $(u)$", f"Deflection Comparison: Pred vs FEM ({BOUNDARY_TITLE})"),
    (fig_phi, ax_phi, "Rotation $(\\phi)$", f"Rotation Comparison: Pred vs FEM ({BOUNDARY_TITLE})"),
    (fig_M, ax_M, "Moment $(M)$", f"Moment Comparison: Pred vs FEM ({BOUNDARY_TITLE})"),
    (fig_Q, ax_Q, "Shear Force $(Q)$", f"Shear Force Comparison: Pred vs FEM ({BOUNDARY_TITLE})"),
]

for fig, axis, ylabel, title in plot_configs:
    axis.hlines(0, 0, L_val, color="black", linewidth=3, zorder=5)
    plot_support_symbols(axis)
    axis.set_xlabel("Beam Position $(x)$", fontsize=20)
    axis.set_ylabel(ylabel, fontsize=20)
    axis.set_title(title, fontsize=20, pad=10)
    if axis is ax_M:
        axis.invert_yaxis()
    if axis is ax_q:
        axis.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
            fontsize=16, handlelength=2.2, labelspacing=0.45, borderaxespad=0.0,
        )
        fig.subplots_adjust(right=0.76)
    else:
        legend = axis.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    fig.show()


handles, labels = ax_u.get_legend_handles_labels()
fig_legend, ax_legend = plt.subplots(figsize=(10, 0.4), dpi=500)
ax_legend.axis("off")
fig_legend.legend(
    handles, labels, loc="center", ncol=4, frameon=False,
    fontsize=16, handlelength=2.5, columnspacing=1.5, labelspacing=0.5,
)
fig_legend.tight_layout()
fig_legend.show()


err_titles = [
    "Absolute Error: Deflection $(u)$",
    "Absolute Error: Rotation $(\\phi)$",
    "Absolute Error: Moment $(M)$",
    "Absolute Error: Shear Force $(Q)$",
]
err_legend_locs = ["upper center", "upper center", "upper right", "upper left"]
for j, axis in enumerate(ax_err):
    axis.set_title(err_titles[j], fontsize=16)
    axis.set_xlabel("Beam Position $(x)$", fontsize=16)
    axis.set_ylabel("Error Magnitude", fontsize=16)
    axis.hlines(0, 0, L_val, color="black", linewidth=3, zorder=5)
    plot_support_symbols(axis)
    axis.grid(True, linestyle="--", alpha=0.5)
    axis.legend(loc=err_legend_locs[j])
fig_err.tight_layout()
fig_err.show()


print("\n" + "=" * 85)
print("🚀 FNO vs high precision Hermite FEM Quantitative evaluation report on structural response depth")
print("=" * 85)

for i, (case, pred_fields, fem_fields) in enumerate(case_cache):
    metrics = [
        detail_metrics(name, pred, fem)
        for name, pred, fem in zip(field_names, pred_fields, fem_fields)
    ]
    print(f"\n▶ {describe_case(i + 1, case)}")
    print("=" * 85)
    print("【Table 1】Analysis of spatial maximum deviation location (find the point with the worst global fitting)")
    print(f"{'Physical Quantity':^9} | {'Maximum Absolute Error':^15} | {'Predicted Value':^15} | {'FEM Value':^15} | {'Relative Peak Error':^15}")
    print("-" * 85)
    for item in metrics:
        print(f"{item['name']:^10} | {item['max_err']:>15.4e} | {item['pred_at_err']:>15.4e} | {item['fem_at_err']:>15.4e} | {item['rel_err']:>12.4f} %")
    print("-" * 85)
    print("【Table 2】Global peak response evaluation ")
    print(f"{'Physical Quantity':^9} | {'Maximum Predicted Value (Absolute)':^13} | {'Maximum FEM Value (Absolute)':^13} | {'Peak Absolute Deviation':^15} | {'Peak Relative Deviation':^15}")
    print("-" * 85)
    for item in metrics:
        print(f"{item['name']:^10} | {item['peak_pred']:>15.4e} | {item['peak_fem']:>15.4e} | {item['peak_diff']:>15.4e} | {item['peak_rel']:>12.4f} %")
    print("=" * 85)


print("\n\n" + "★" * 85)
print("🏆 Summary of global comprehensive accuracy evaluation (average of all test conditions)")
print("★" * 85)
print(f"{'Physical Quantity':^9} | {'Mean MAE':^16} | {'Mean Relative L²':^20} | {'Mean R²':^15}")
print("-" * 85)
for name, values in global_metrics_summary.items():
    print(
        f"{name:^10} | {np.mean(values['mae']):>16.4e} | "
        f"{np.mean(values['l2_rel']):>20.4e} | {np.mean(values['r2']):>15.6f}"
    )
print("★" * 85 + "\n")


def get_last_value(logs, key):
    values = logs.get(key, [])
    if values is None or len(values) == 0:
        return np.nan
    return float(np.asarray(values, dtype=float)[-1])


adam_ultimate_loss = get_last_value(logs_data, "loss_history")
lbfgs_final_loss = get_last_value(logs_data, "lbfgs_loss_history")
print("\n" + "=" * 65)
print("📌 Final Training Loss Summary")
print("=" * 65)
print(f"{'Stage':^20} | {'Total Physics Loss':^28}")
print("-" * 65)
print(f"{'Adam ultimate':^20} | {adam_ultimate_loss:>28.6e}")
print(f"{'L-BFGS final':^20} | {lbfgs_final_loss:>28.6e}")
print("=" * 65)
