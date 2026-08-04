# ============================================================
# BP-PFNO Chapter 3.3: Unified Evaluation for Ablation and Parameter Tests
# File: BP-PFNO_3.3_Ablation_and_Param_EVA.py
# Function: load all trained models, evaluate by high-resolution Hermite FEM,
#           and save CSV metrics only; no training and no plotting.
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import qmc

torch.set_default_dtype(torch.float64)


RUN_ROOT = Path("./BP_PFNO_3p3_runs_no_dynamic_weights")
EVAL_ROOT = RUN_ROOT / "evaluation"
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

N_EVAL_GRID = 100
NUM_NODES_REF = 1000
GAUSS_ORDER = 8
LHS_TEST_NUM = 100
LHS_TEST_SEED = 2026
SUMMARY_CASE_SET = "lhs100"     


FEM_CACHE_PATH = EVAL_ROOT / f"fem_reference_grid{N_EVAL_GRID}_ref{NUM_NODES_REF}_lhs{LHS_TEST_NUM}.npz"

BASE_U_SCALE = 1e-3
BASE_PHI_SCALE = 1e-3
BASE_M_SCALE = 1e-2
BASE_Q_SCALE = 1e-2



class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cdouble)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1)//2 + 1,
            device=x.device,
            dtype=torch.cdouble
        )
        modes = min(self.modes1, x.size(-1)//2 + 1)
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :modes], self.weights1[:, :, :modes]
        )
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class BEP_PFNO_Combined(nn.Module):
    def __init__(self, m_sensors, modes=16, width=64, hidden_dim=128, scale_tuple=(1e-3, 1e-3, 1e-2, 1e-2)):
        super(BEP_PFNO_Combined, self).__init__()
        self.m_sensors = m_sensors
        self.fc0 = nn.Linear(4, width)

        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, hidden_dim)

        def build_decoder(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        self.decoder_Q = build_decoder(hidden_dim)
        self.decoder_M = build_decoder(hidden_dim)
        self.decoder_phi = build_decoder(hidden_dim)
        self.decoder_u = build_decoder(hidden_dim)

        self.U_scale = float(scale_tuple[0])
        self.Phi_scale = float(scale_tuple[1])
        self.M_scale = float(scale_tuple[2])
        self.Q_scale = float(scale_tuple[3])

    def forward(self, branch_input, x_batch):
        Batch_size = branch_input.shape[0]
        N_domain = x_batch.shape[1]

        q_sensors = branch_input[:, :self.m_sensors]
        params = branch_input[:, self.m_sensors:]

        q_x = torch.nn.functional.interpolate(
            q_sensors.unsqueeze(1), size=N_domain, mode="linear", align_corners=True
        ).transpose(1, 2)

        EI_x = params[:, 0:1].unsqueeze(2).expand(Batch_size, N_domain, 1)
        L_x = params[:, 1:2].unsqueeze(2).expand(Batch_size, N_domain, 1)
        fno_in = torch.cat([q_x, x_batch, EI_x, L_x], dim=-1)

        x = self.fc0(fno_in)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x); x2 = self.w0(x); x = torch.nn.functional.silu(x1 + x2)
        x1 = self.conv1(x); x2 = self.w1(x); x = torch.nn.functional.silu(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = torch.nn.functional.silu(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = torch.nn.functional.silu(x1 + x2)

        x = x.permute(0, 2, 1)
        fused_features = self.fc1(x)

        Q_raw = self.decoder_Q(fused_features)
        M_raw = self.decoder_M(fused_features)
        phi_raw = self.decoder_phi(fused_features)
        u_raw = self.decoder_u(fused_features)
        return u_raw, phi_raw, M_raw, Q_raw

    def predict_with_ansatz(
        self,
        branch_input,
        x_batch,
        L,
        bc_type='fixed-fixed',
        ansatz_mode='hard',
        use_load_scale=True
    ):
        u_raw, phi_raw, M_raw, Q_raw = self.forward(branch_input, x_batch)

        m_sensors = branch_input.shape[1] - 2
        q_sensors = branch_input[:, 0:m_sensors]
        q_val_mean = torch.mean(q_sensors, dim=1, keepdim=True).unsqueeze(1)
        q_scale = q_val_mean / 0.1
        if not use_load_scale:
            q_scale = torch.ones_like(q_scale)

        u_scaled = u_raw * self.U_scale * q_scale
        phi_scaled = phi_raw * self.Phi_scale * q_scale
        M_scaled = M_raw * self.M_scale * q_scale
        Q_scaled = Q_raw * self.Q_scale * q_scale

        if ansatz_mode == 'hard':
            if bc_type == 'fixed-fixed':
                u_pred = (x_batch**2) * ((L - x_batch)**2) * u_scaled
                phi_pred = x_batch * (L - x_batch) * phi_scaled
                M_pred = M_scaled
                Q_pred = Q_scaled
            elif bc_type == 'simply-supported':
                u_pred = x_batch * (L - x_batch) * u_scaled
                phi_pred = phi_scaled
                M_pred = x_batch * (L - x_batch) * M_scaled
                Q_pred = Q_scaled
            else:
                u_pred, phi_pred, M_pred, Q_pred = u_scaled, phi_scaled, M_scaled, Q_scaled
        else:
            u_pred, phi_pred, M_pred, Q_pred = u_scaled, phi_scaled, M_scaled, Q_scaled

        return u_pred, phi_pred, M_pred, Q_pred



def _gaussian_localized_load(x, P, xp, sigma):
    return P * np.exp(-((x - xp) ** 2) / (2 * sigma ** 2))


def _hermite_shape_functions(s, Le):
    r = s / Le
    return np.array([
        1 - 3 * r ** 2 + 2 * r ** 3,
        s * (1 - r) ** 2,
        3 * r ** 2 - 2 * r ** 3,
        s * (r ** 2 - r)
    ])


def _beam_element_stiffness(EI, Le):
    return (EI / Le ** 3) * np.array([
        [12,        6 * Le,    -12,       6 * Le],
        [6 * Le,    4 * Le**2, -6 * Le,   2 * Le**2],
        [-12,      -6 * Le,     12,      -6 * Le],
        [6 * Le,    2 * Le**2, -6 * Le,   4 * Le**2]
    ])


def _consistent_element_load_gaussian(x_left, Le, P, xp, sigma, gauss_order=8):
    xi, wi = np.polynomial.legendre.leggauss(gauss_order)
    fe = np.zeros(4, dtype=np.float64)
    for xi_i, wi_i in zip(xi, wi):
        s = 0.5 * Le * (xi_i + 1.0)
        x_g = x_left + s
        q_g = _gaussian_localized_load(x_g, P, xp, sigma)
        N = _hermite_shape_functions(s, Le)
        fe += wi_i * N * q_g * (Le / 2.0)
    return fe


def solve_beam_fem_fixed_gaussian(
    P,
    xp,
    sigma=0.04,
    L=1.0,
    EI=1.0,
    num_nodes=1000,
    gauss_order=8,
):
    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    n_elem = num_nodes - 1
    total_dof = num_nodes * 2
    x_ref = np.linspace(0.0, L, num_nodes)
    Le = L / n_elem

    K = np.zeros((total_dof, total_dof), dtype=np.float64)
    F = np.zeros(total_dof, dtype=np.float64)
    ke = _beam_element_stiffness(EI, Le)

    element_loads = []
    for e in range(n_elem):
        x_left = x_ref[e]
        fe = _consistent_element_load_gaussian(x_left, Le, P, xp, sigma, gauss_order=gauss_order)
        element_loads.append(fe)

        dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
        K[np.ix_(dof, dof)] += ke
        F[dof] += fe

    fixed_dofs = np.array([0, 1, total_dof - 2, total_dof - 1])
    free_dofs = np.setdiff1d(np.arange(total_dof), fixed_dofs)

    U = np.zeros(total_dof, dtype=np.float64)
    U[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])

    u_ref = U[0::2]
    phi_ref = U[1::2]

    M_sum = np.zeros(num_nodes, dtype=np.float64)
    Q_sum = np.zeros(num_nodes, dtype=np.float64)
    count = np.zeros(num_nodes, dtype=np.float64)

    for e in range(n_elem):
        dof = np.array([2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1])
        u_e = U[dof]
        fe = element_loads[e]
        p_e = ke @ u_e - fe

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

    return x_ref, u_ref, phi_ref, M_ref, Q_ref


def solve_beam_fem_fixed_gaussian_on_grid(
    P,
    xp,
    x_target,
    sigma=0.04,
    L=1.0,
    EI=1.0,
    num_nodes_ref=1000,
    gauss_order=8,
):
    x_target = np.asarray(x_target, dtype=np.float64).reshape(-1)
    x_ref, u_ref, phi_ref, M_ref, Q_ref = solve_beam_fem_fixed_gaussian(
        P=P,
        xp=xp,
        sigma=sigma,
        L=L,
        EI=EI,
        num_nodes=num_nodes_ref,
        gauss_order=gauss_order,
    )
    u_fem = np.interp(x_target, x_ref, u_ref)
    phi_fem = np.interp(x_target, x_ref, phi_ref)
    M_fem = np.interp(x_target, x_ref, M_ref)
    Q_fem = np.interp(x_target, x_ref, Q_ref)
    return u_fem, phi_fem, M_fem, Q_fem


def build_test_cases():
    cases = []
    canonical = [(-3.0, 0.3), (-2.0, 0.5), (-2.5, 0.7)]
    for i, (P, xp) in enumerate(canonical, start=1):
        cases.append({"case_set": "canonical", "case_id": f"C{i}", "P": float(P), "xp": float(xp)})

    sampler = qmc.LatinHypercube(d=2, seed=LHS_TEST_SEED)
    lhs = sampler.random(n=LHS_TEST_NUM)
    P_min, P_max = -5.0, -0.5
    xp_min, xp_max = 0.2, 0.8
    P_vals = P_min + lhs[:, 0] * (P_max - P_min)
    xp_vals = xp_min + lhs[:, 1] * (xp_max - xp_min)
    for i, (P, xp) in enumerate(zip(P_vals, xp_vals), start=1):
        cases.append({"case_set": "lhs100", "case_id": f"LHS{i:03d}", "P": float(P), "xp": float(xp)})
    return cases


def load_or_build_fem_cache(cases, x_np, sigma=0.04, L=1.0, EI=1.0):
    if FEM_CACHE_PATH.exists():
        cache = np.load(FEM_CACHE_PATH, allow_pickle=True)
        print(f"Read FEM cache: {FEM_CACHE_PATH}")
        return cache

    print("Start building FEM reference...")
    case_set_arr, case_id_arr, P_arr, xp_arr = [], [], [], []
    refs = []

    for idx, case in enumerate(cases, start=1):
        print(f"FEM cache {idx:4d}/{len(cases)} | {case['case_set']} | {case['case_id']} | P={case['P']:.4f}, xp={case['xp']:.4f}")
        u_fem, phi_fem, M_fem, Q_fem = solve_beam_fem_fixed_gaussian_on_grid(
            P=case["P"], xp=case["xp"], x_target=x_np,
            sigma=sigma, L=L, EI=EI,
            num_nodes_ref=NUM_NODES_REF, gauss_order=GAUSS_ORDER
        )
        refs.append(np.stack([u_fem, phi_fem, M_fem, Q_fem], axis=0))
        case_set_arr.append(case["case_set"])
        case_id_arr.append(case["case_id"])
        P_arr.append(case["P"])
        xp_arr.append(case["xp"])

    refs = np.asarray(refs, dtype=np.float64)
    np.savez_compressed(
        FEM_CACHE_PATH,
        case_set=np.asarray(case_set_arr),
        case_id=np.asarray(case_id_arr),
        P=np.asarray(P_arr, dtype=np.float64),
        xp=np.asarray(xp_arr, dtype=np.float64),
        refs=refs,
        x=x_np,
    )
    print(f"FEM reference saved: {FEM_CACHE_PATH}")
    return np.load(FEM_CACHE_PATH, allow_pickle=True)


def build_branch_input(P, xp, m_sensors, EI_val, L_val, sigma):
    x_sensors = np.linspace(0, L_val, m_sensors)
    q_sensor_data = P * np.exp(-((x_sensors - xp)**2) / (2 * sigma**2))
    test_branch_input = torch.cat([
        torch.tensor([q_sensor_data], dtype=torch.float64),
        torch.tensor([[EI_val, L_val]], dtype=torch.float64)
    ], dim=1)
    return test_branch_input, q_sensor_data


def evaluate_metrics(pred, fem):
    abs_err_arr = np.abs(pred - fem)
    mae_val = np.mean(abs_err_arr)
    l2_rel_val = np.linalg.norm(pred - fem) / (np.linalg.norm(fem) + 1e-12)
    ss_res = np.sum((fem - pred) ** 2)
    ss_tot = np.sum((fem - np.mean(fem)) ** 2) + 1e-12
    r2_val = 1 - (ss_res / ss_tot)
    max_abs_err = np.max(abs_err_arr)
    peak_fem_abs = np.max(np.abs(fem))
    peak_pred_abs = np.max(np.abs(pred))
    peak_rel_val = (np.abs(peak_pred_abs - peak_fem_abs) / (peak_fem_abs + 1e-12)) * 100
    return mae_val, l2_rel_val, r2_val, max_abs_err, peak_rel_val


def compute_gradients_fd_np(y, dx):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dx)
    dy[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dx)
    dy[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dx)
    return dy


def compute_physical_residual_metrics(pred_dict, q_domain, q_sensor, model, L):
    dx = L / (len(q_domain) - 1)
    u = pred_dict["u"]
    phi = pred_dict["phi"]
    M = pred_dict["M"]
    Q = pred_dict["Q"]

    du_dx = compute_gradients_fd_np(u, dx)
    dphi_dx = compute_gradients_fd_np(phi, dx)
    dM_dx = compute_gradients_fd_np(M, dx)
    dQ_dx = compute_gradients_fd_np(Q, dx)

    q_norm_abs = (np.mean(np.abs(q_sensor)) + 1e-4) / 0.1
    geo = np.mean(((du_dx - phi) / (BASE_PHI_SCALE * q_norm_abs + 1e-12))**2)
    const = np.mean(((M - dphi_dx) / (BASE_M_SCALE * q_norm_abs + 1e-12))**2)
    eq1 = np.mean(((Q - dM_dx) / (BASE_Q_SCALE * q_norm_abs + 1e-12))**2)
    eq2 = np.mean(((dQ_dx - q_domain) / (0.1 * q_norm_abs + 1e-12))**2)
    return geo, const, eq1, eq2



def load_config(run_dir):
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    payload = torch.load(run_dir / "model.pth", weights_only=False)
    return payload.get("config", {})


def load_model(run_dir):
    payload = torch.load(run_dir / "model.pth", weights_only=False)
    cfg = load_config(run_dir)

    scale_tuple = payload.get("scale_tuple", cfg.get("scale_tuple", (1e-3, 1e-3, 1e-2, 1e-2)))
    scale_tuple = tuple(scale_tuple)

    model = BEP_PFNO_Combined(
        m_sensors=int(payload.get("m_sensors", cfg.get("m_sensors", 51))),
        modes=int(payload.get("modes", cfg.get("modes", 16))),
        width=int(payload.get("width", cfg.get("width", 64))),
        hidden_dim=int(payload.get("hidden_dim", cfg.get("hidden_dim", 128))),
        scale_tuple=scale_tuple
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def get_last(logs, key):
    arr = logs.get(key, [])
    if arr is None or len(arr) == 0:
        return np.nan
    return float(np.asarray(arr, dtype=np.float64)[-1])


def get_final_dynamic_weights(logs, cfg):
    weights = logs.get(
        "final_dynamic_weights",
        cfg.get("final_dynamic_weights", [])
    )
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape == (4,):
        return arr

    history = logs.get("weight_history", [])
    hist_arr = np.asarray(history, dtype=np.float64)
    if hist_arr.ndim == 2 and hist_arr.shape[0] > 0 and hist_arr.shape[1] == 4:
        return hist_arr[-1]

    return np.full(4, np.nan, dtype=np.float64)


def evaluate_one_run(run_dir, fem_cache):
    model, cfg = load_model(run_dir)
    logs = torch.load(run_dir / "logs.pth", weights_only=False)

    L_val = float(cfg.get("L_val", 1.0))
    EI_val = float(cfg.get("EI_val", 1.0))
    sigma = float(cfg.get("sigma", 0.04))
    m_sensors = int(cfg.get("m_sensors", model.m_sensors))
    exp_id = cfg.get("exp_id", run_dir.parent.name)
    exp_group = cfg.get("exp_group", run_dir.parent.parent.name)
    seed = int(cfg.get("seed", run_dir.name.replace("seed_", "")))
    ansatz_mode = cfg.get("ansatz_mode", "hard")
    bc_type = cfg.get("bc_type", "fixed-fixed")
    use_load_scale = bool(cfg.get("use_load_scale", True))
    use_dynamic_weights = bool(
        logs.get(
            "use_dynamic_weights",
            cfg.get("use_dynamic_weights", False)
        )
    )
    loss_weighting = logs.get(
        "loss_weighting",
        cfg.get("loss_weighting", "equal_sum")
    )
    final_weights = get_final_dynamic_weights(logs, cfg)

    x_np = np.asarray(fem_cache["x"], dtype=np.float64)
    test_x = torch.tensor(x_np, dtype=torch.float64).reshape(1, -1, 1)
    case_set_arr = fem_cache["case_set"]
    case_id_arr = fem_cache["case_id"]
    P_arr = fem_cache["P"]
    xp_arr = fem_cache["xp"]
    refs = fem_cache["refs"]

    case_records = []
    canonical_pred_records = []
    quantity_names = ["u", "phi", "M", "Q"]

    for idx in range(len(P_arr)):
        P = float(P_arr[idx])
        xp = float(xp_arr[idx])
        case_set = str(case_set_arr[idx])
        case_id = str(case_id_arr[idx])

        branch_input, q_sensor = build_branch_input(P, xp, m_sensors, EI_val, L_val, sigma)
        q_domain = P * np.exp(-((x_np - xp)**2) / (2 * sigma**2))

        with torch.no_grad():
            u_pred, phi_pred, M_pred, Q_pred = model.predict_with_ansatz(
                branch_input, test_x, L=L_val,
                bc_type=bc_type,
                ansatz_mode=ansatz_mode,
                use_load_scale=use_load_scale
            )
            pred_arrays = [
                u_pred.squeeze().numpy(),
                phi_pred.squeeze().numpy(),
                M_pred.squeeze().numpy(),
                Q_pred.squeeze().numpy(),
            ]

        pred_dict = {q: arr for q, arr in zip(quantity_names, pred_arrays)}
        geo_res, const_res, eq1_res, eq2_res = compute_physical_residual_metrics(pred_dict, q_domain, q_sensor, model, L_val)

        if case_set == "canonical":
            canonical_pred_records.append({
                "case_id": case_id,
                "P": P,
                "xp": xp,
                "q_domain": q_domain.copy(),
                "u_pred": pred_arrays[0].copy(),
                "phi_pred": pred_arrays[1].copy(),
                "M_pred": pred_arrays[2].copy(),
                "Q_pred": pred_arrays[3].copy(),
                "u_fem": refs[idx, 0, :].copy(),
                "phi_fem": refs[idx, 1, :].copy(),
                "M_fem": refs[idx, 2, :].copy(),
                "Q_fem": refs[idx, 3, :].copy(),
            })

        for q_idx, quantity in enumerate(quantity_names):
            fem = refs[idx, q_idx, :]
            pred = pred_arrays[q_idx]
            mae, rel_l2, r2, max_abs_err, peak_rel = evaluate_metrics(pred, fem)
            case_records.append({
                "exp_group": exp_group,
                "exp_id": exp_id,
                "seed": seed,
                "case_set": case_set,
                "case_id": case_id,
                "P": P,
                "xp": xp,
                "quantity": quantity,
                "mae": mae,
                "rel_l2": rel_l2,
                "r2": r2,
                "max_abs_err": max_abs_err,
                "peak_rel_percent": peak_rel,
                "geo_res": geo_res,
                "const_res": const_res,
                "eq1_res": eq1_res,
                "eq2_res": eq2_res,
            })

    df_cases = pd.DataFrame(case_records)

    primary = df_cases[df_cases["case_set"] == SUMMARY_CASE_SET].copy()
    canonical = df_cases[df_cases["case_set"] == "canonical"].copy()

    def summarize_case_set(df, prefix=""):
        row = {}
        for quantity in quantity_names:
            sub = df[df["quantity"] == quantity]
            row[f"{prefix}{quantity}_mae"] = sub["mae"].mean()
            row[f"{prefix}{quantity}_rel_l2"] = sub["rel_l2"].mean()
            row[f"{prefix}{quantity}_r2"] = sub["r2"].mean()
            row[f"{prefix}{quantity}_max_abs_err"] = sub["max_abs_err"].mean()
            row[f"{prefix}{quantity}_peak_rel_percent"] = sub["peak_rel_percent"].mean()
        row[f"{prefix}avg_mae"] = np.mean([row[f"{prefix}{q}_mae"] for q in quantity_names])
        row[f"{prefix}avg_rel_l2"] = np.mean([row[f"{prefix}{q}_rel_l2"] for q in quantity_names])
        row[f"{prefix}avg_r2"] = np.mean([row[f"{prefix}{q}_r2"] for q in quantity_names])
        row[f"{prefix}avg_max_abs_err"] = np.mean([row[f"{prefix}{q}_max_abs_err"] for q in quantity_names])
        row[f"{prefix}avg_peak_rel_percent"] = np.mean([row[f"{prefix}{q}_peak_rel_percent"] for q in quantity_names])
        return row

    summary = {
        "exp_group": exp_group,
        "exp_id": exp_id,
        "seed": seed,
        "m_sensors": m_sensors,
        "adam_epochs": int(cfg.get("adam_epochs", np.nan)),
        "lbfgs_max_iter": int(cfg.get("lbfgs_max_iter", np.nan)),
        "ansatz_mode": ansatz_mode,
        "use_response_scale": bool(cfg.get("use_response_scale", True)),
        "use_load_scale": use_load_scale,
        "scale_tuple": str(cfg.get("scale_tuple", "")),
        "use_norm": bool(cfg.get("use_norm", True)),
        "use_dynamic_weights": use_dynamic_weights,
        "loss_weighting": loss_weighting,
        "final_w_geo": final_weights[0],
        "final_w_const": final_weights[1],
        "final_w_eq1": final_weights[2],
        "final_w_eq2": final_weights[3],
        "train_time_s": float(logs.get("train_time_s", np.nan)),
        "adam_final_loss": get_last(logs, "loss_history"),
        "adam_final_physics_loss": get_last(
            logs, "raw_total_loss_history"
        ),
        "adam_final_bc_loss": get_last(
            logs, "loss_bc_history"
        ),
        "lbfgs_final_loss": get_last(
            logs, "lbfgs_loss_history"
        ),
        "lbfgs_final_physics_loss": get_last(
            logs, "lbfgs_raw_total_loss_history"
        ),
        "lbfgs_final_bc_loss": get_last(
            logs, "lbfgs_loss_bc_history"
        ),
        "geo_res": primary["geo_res"].mean(),
        "const_res": primary["const_res"].mean(),
        "eq1_res": primary["eq1_res"].mean(),
        "eq2_res": primary["eq2_res"].mean(),
    }
    summary.update(summarize_case_set(primary, prefix=""))
    summary.update(summarize_case_set(canonical, prefix="canonical_"))

    df_summary = pd.DataFrame([summary])

    df_cases.to_csv(run_dir / "results_cases.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(run_dir / "results_summary.csv", index=False, encoding="utf-8-sig")

    if canonical_pred_records:
        np.savez_compressed(
            run_dir / "canonical_predictions.npz",
            x=x_np,
            records=np.asarray(canonical_pred_records, dtype=object),
        )

    return df_cases, df_summary


def find_run_dirs(root):
    model_paths = sorted(root.glob("**/model.pth"))
    run_dirs = []
    for p in model_paths:
        if p.parent.name.startswith("seed_"):
            run_dirs.append(p.parent)
    return run_dirs


def main():
    x_np = np.linspace(0.0, 1.0, N_EVAL_GRID)
    cases = build_test_cases()
    fem_cache = load_or_build_fem_cache(cases, x_np, sigma=0.04, L=1.0, EI=1.0)

    run_dirs = find_run_dirs(RUN_ROOT)
    if len(run_dirs) == 0:
        raise FileNotFoundError(f"No model.pth was found under {RUN_ROOT} , Please run the training script first.")

    print("\n" + "#" * 100)
    print(f"The unified evaluation was started, {len(run_dirs)} models were found。")
    print("#" * 100)

    all_cases = []
    all_summary = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        print("\n" + "=" * 100)
        print(f"[{idx}/{len(run_dirs)}] Evaluating: {run_dir}")
        print("=" * 100)
        try:
            df_cases, df_summary = evaluate_one_run(run_dir, fem_cache)
            all_cases.append(df_cases)
            all_summary.append(df_summary)
            print(f"Finish: {run_dir}")
        except Exception as e:
            print(f"Failed: {run_dir}")
            print(f"Error: {repr(e)}")

    if all_cases:
        df_all_cases = pd.concat(all_cases, ignore_index=True)
        df_all_cases.to_csv(RUN_ROOT / "ALL_results_cases.csv", index=False, encoding="utf-8-sig")
        print(f"Results of all working conditions have been saved: {RUN_ROOT / 'ALL_results_cases.csv'}")

    if all_summary:
        df_all_summary = pd.concat(all_summary, ignore_index=True)
        df_all_summary.to_csv(RUN_ROOT / "ALL_results_summary.csv", index=False, encoding="utf-8-sig")
        print(f"All results saved: {RUN_ROOT / 'ALL_results_summary.csv'}")

    print("\nFinish")


if __name__ == '__main__':
    main()
