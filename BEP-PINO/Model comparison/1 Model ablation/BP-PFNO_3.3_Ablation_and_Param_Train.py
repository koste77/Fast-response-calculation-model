# %%
# ============================================================
# BP-PFNO Chapter 3.3: Ablation and Parameter Batch Training
# File: BP-PFNO_3.3_Ablation_and_Param_Train.py
# Function: batch training and saving only; no evaluation and no plotting.
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import copy
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc

torch.set_default_dtype(torch.float64)


RUN_MODE = "all"
RUN_SEEDS = [1202]           #[1202, 1003, 9013]
SKIP_EXISTING = True           
SAVE_ROOT = "./BP_PFNO_3p3_runs_no_dynamic_weights"

SMOKE_ADAM_EPOCHS = 5
SMOKE_LBFGS_ITER = 3

# %%

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
        x = x.permute(0, 2, 1)  # [Batch, Channels, N_domain]

        x1 = self.conv0(x); x2 = self.w0(x); x = torch.nn.functional.silu(x1 + x2)
        x1 = self.conv1(x); x2 = self.w1(x); x = torch.nn.functional.silu(x1 + x2)
        x1 = self.conv2(x); x2 = self.w2(x); x = torch.nn.functional.silu(x1 + x2)
        x1 = self.conv3(x); x2 = self.w3(x); x = torch.nn.functional.silu(x1 + x2)

        x = x.permute(0, 2, 1)  # [Batch, N_domain, Channels]
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

def compute_gradients_fd(y, dx):

    dy_dx_interior = (y[:, 2:, :] - y[:, :-2, :]) / (2 * dx)
    dy_dx_left = (-3 * y[:, 0:1, :] + 4 * y[:, 1:2, :] - y[:, 2:3, :]) / (2 * dx)
    dy_dx_right = (3 * y[:, -1:, :] - 4 * y[:, -2:-1, :] + y[:, -3:-2, :]) / (2 * dx)
    dy_dx = torch.cat([dy_dx_left, dy_dx_interior, dy_dx_right], dim=1)
    return dy_dx


def compute_bc_penalty(u_pred, phi_pred, branch_input, model, cfg):
    if cfg.bc_type != 'fixed-fixed':
        return torch.tensor(0.0, dtype=torch.float64, device=u_pred.device)

    m_sensors = branch_input.shape[1] - 2
    q_sensors = branch_input[:, 0:m_sensors]
    q_norm_abs = (torch.mean(torch.abs(q_sensors), dim=1).view(-1, 1, 1) + 1e-4) / 0.1

    if cfg.use_norm:
        u_den = abs(model.U_scale) * q_norm_abs + 1e-12
        p_den = abs(model.Phi_scale) * q_norm_abs + 1e-12
        bc_loss = torch.mean((u_pred[:, 0:1, :] / u_den)**2)
        bc_loss = bc_loss + torch.mean((u_pred[:, -1:, :] / u_den)**2)
        bc_loss = bc_loss + torch.mean((phi_pred[:, 0:1, :] / p_den)**2)
        bc_loss = bc_loss + torch.mean((phi_pred[:, -1:, :] / p_den)**2)
    else:
        bc_loss = torch.mean(u_pred[:, 0:1, :]**2)
        bc_loss = bc_loss + torch.mean(u_pred[:, -1:, :]**2)
        bc_loss = bc_loss + torch.mean(phi_pred[:, 0:1, :]**2)
        bc_loss = bc_loss + torch.mean(phi_pred[:, -1:, :]**2)

    return bc_loss


def compute_physics_loss_combined(
    model,
    branch_input,
    x_batch,
    q_y,
    EI_z,
    L,
    cfg,
    weights=None
):

    u_pred, phi_pred, M_pred, Q_pred = model.predict_with_ansatz(
        branch_input, x_batch, L,
        bc_type=cfg.bc_type,
        ansatz_mode=cfg.ansatz_mode,
        use_load_scale=cfg.use_load_scale
    )

    dx = L / (x_batch.shape[1] - 1)

    du_dx = compute_gradients_fd(u_pred, dx)
    dphi_dx = compute_gradients_fd(phi_pred, dx)
    dM_dx = compute_gradients_fd(M_pred, dx)
    dQ_dx = compute_gradients_fd(Q_pred, dx)

    m_sensors = branch_input.shape[1] - 2
    q_sensors = branch_input[:, 0:m_sensors]
    q_norm_abs = (
        torch.mean(torch.abs(q_sensors), dim=1).view(-1, 1, 1) + 1e-4
    ) / 0.1

    if cfg.use_norm:
        loss_geo = torch.mean(
            ((du_dx - phi_pred) /
             (abs(model.Phi_scale) * q_norm_abs + 1e-12))**2
        )
        loss_const = torch.mean(
            ((M_pred - EI_z * dphi_dx) /
             (abs(model.M_scale) * q_norm_abs + 1e-12))**2
        )
        loss_eq1 = torch.mean(
            ((Q_pred - dM_dx) /
             (abs(model.Q_scale) * q_norm_abs + 1e-12))**2
        )
        loss_eq2 = torch.mean(
            ((dQ_dx - q_y) /
             (0.1 * q_norm_abs + 1e-12))**2
        )
    else:
        loss_geo = torch.mean((du_dx - phi_pred)**2)
        loss_const = torch.mean((M_pred - EI_z * dphi_dx)**2)
        loss_eq1 = torch.mean((Q_pred - dM_dx)**2)
        loss_eq2 = torch.mean((dQ_dx - q_y)**2)

    loss_physics = loss_geo + loss_const + loss_eq1 + loss_eq2

    if cfg.use_dynamic_weights:
        if weights is None:
            raise ValueError(
                "cfg.use_dynamic_weights=True requires physics weights."
            )
        weights = weights.to(
            dtype=loss_geo.dtype,
            device=loss_geo.device
        )
        loss_physics = (
            weights[0] * loss_geo +
            weights[1] * loss_const +
            weights[2] * loss_eq1 +
            weights[3] * loss_eq2
        )
    else:
        loss_physics = loss_geo + loss_const + loss_eq1 + loss_eq2

    loss_bc = torch.tensor(
        0.0, dtype=torch.float64, device=x_batch.device
    )
    if cfg.ansatz_mode == 'soft':
        loss_bc = compute_bc_penalty(
            u_pred, phi_pred, branch_input, model, cfg
        )
        loss_total = loss_physics + cfg.bc_penalty_coef * loss_bc
    else:
        loss_total = loss_physics

    return (
        loss_total,
        loss_geo,
        loss_const,
        loss_eq1,
        loss_eq2,
        loss_bc
    )


def update_dynamic_weights(cfg, dynamic_weights, l_geo, l_const, l_eq1, l_eq2):
    raw_geo = l_geo.item()
    raw_const = l_const.item()
    raw_eq1 = l_eq1.item()
    raw_eq2 = l_eq2.item()

    w_eq2_base = 1.0
    w_eq1_base = (
        torch.exp(
            torch.tensor(
                -raw_eq2 / cfg.dynamic_causal_tolerance
            )
        ).item() * w_eq2_base
    )
    w_const_base = (
        torch.exp(
            torch.tensor(
                -raw_eq1 / cfg.dynamic_causal_tolerance
            )
        ).item() * w_eq1_base
    )
    w_geo_base = (
        torch.exp(
            torch.tensor(
                -raw_const / cfg.dynamic_causal_tolerance
            )
        ).item() * w_const_base
    )

    current_losses = torch.tensor(
        [raw_geo, raw_const, raw_eq1, raw_eq2],
        dtype=torch.float64
    )
    mean_loss = torch.mean(current_losses) + 1e-12

    boost_factors = (
        current_losses / mean_loss
    ) ** cfg.dynamic_gamma

    target_weights = torch.tensor(
        [
            w_geo_base,
            w_const_base,
            w_eq1_base,
            w_eq2_base
        ],
        dtype=torch.float64
    )
    target_weights = target_weights * boost_factors

    return (
        cfg.dynamic_ema_alpha * dynamic_weights +
        (1.0 - cfg.dynamic_ema_alpha) * target_weights
    )


@dataclass
class ExpConfig:
    exp_id: str = "Base"
    exp_group: str = "base"
    seed: int = 1202

    Batch_size: int = 50
    m_sensors: int = 51
    N_domain: int = 100
    adam_epochs: int = 600
    lbfgs_max_iter: int = 500
    resample_freq: int = 10

    modes: int = 16
    width: int = 64
    hidden_dim: int = 128

    L_val: float = 1.0
    EI_val: float = 1.0
    sigma: float = 0.04
    P_min_val: float = -5.0
    P_max_val: float = -0.5
    xp_min_val: float = 0.2
    xp_max_val: float = 0.8

    bc_type: str = "fixed-fixed"
    ansatz_mode: str = "hard"             # hard / soft / none
    use_response_scale: bool = True
    use_load_scale: bool = True
    scale_tuple: tuple = (1e-3, 1e-3, 1e-2, 1e-2)
    use_norm: bool = True
    bc_penalty_coef: float = 10.0
    use_dynamic_weights: bool = False
    dynamic_weight_init: tuple = (1.0, 1.0, 1.0, 1.0)
    dynamic_ema_alpha: float = 0.95
    dynamic_causal_tolerance: float = 0.05
    dynamic_gamma: float = 0.5

    grad_clip_norm: float = 1.0

    save_root: str = SAVE_ROOT


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_scale_tuple(cfg):
    if cfg.use_response_scale:
        return tuple(cfg.scale_tuple)
    return (1.0, 1.0, 1.0, 1.0)


def make_batch(lhs_sampler, cfg, x_sensors):
    lhs_samples = lhs_sampler.random(n=cfg.Batch_size)
    P_np = cfg.P_min_val + lhs_samples[:, 0:1] * (cfg.P_max_val - cfg.P_min_val)
    xp_np = cfg.xp_min_val + lhs_samples[:, 1:2] * (cfg.xp_max_val - cfg.xp_min_val)

    q_sensor_data_np = P_np * np.exp(-((x_sensors - xp_np)**2) / (2 * cfg.sigma**2))
    q_sensor_data = torch.tensor(q_sensor_data_np, dtype=torch.float64)
    params_data = torch.tensor([[cfg.EI_val, cfg.L_val]], dtype=torch.float64).expand(cfg.Batch_size, -1)
    branch_input = torch.cat([q_sensor_data, params_data], dim=1)

    x_batch = torch.linspace(0, cfg.L_val, cfg.N_domain, dtype=torch.float64).view(1, cfg.N_domain, 1).expand(cfg.Batch_size, -1, -1)

    P_tensor = torch.tensor(P_np, dtype=torch.float64).unsqueeze(2)
    xp_tensor = torch.tensor(xp_np, dtype=torch.float64).unsqueeze(2)
    q_y_domain = P_tensor * torch.exp(-((x_batch - xp_tensor)**2) / (2 * cfg.sigma**2))

    return branch_input, x_batch, q_y_domain


def run_single_experiment(cfg):
    set_random_seed(cfg.seed)

    save_dir = (
        Path(cfg.save_root) /
        cfg.exp_group /
        cfg.exp_id /
        f"seed_{cfg.seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if (
        SKIP_EXISTING and
        (save_dir / "model.pth").exists() and
        (save_dir / "logs.pth").exists()
    ):
        print(
            f"[Skip] {cfg.exp_group}/{cfg.exp_id}/"
            f"seed_{cfg.seed} Already exists, skip"
        )
        return

    loss_weighting = (
        "dynamic_causal_ema"
        if cfg.use_dynamic_weights
        else "equal_sum"
    )

    print("\n" + "=" * 100)
    print(
        f"START: Group={cfg.exp_group} | "
        f"ExpID={cfg.exp_id} | Seed={cfg.seed}"
    )
    print(
        f"m_sensors={cfg.m_sensors}, "
        f"Adam epochs={cfg.adam_epochs}, "
        f"L-BFGS max_iter={cfg.lbfgs_max_iter}"
    )
    print(
        f"ansatz={cfg.ansatz_mode}, "
        f"response_scale={cfg.use_response_scale}, "
        f"load_scale={cfg.use_load_scale}, "
        f"norm={cfg.use_norm}, "
        f"loss_weighting={loss_weighting}"
    )
    print("=" * 100)

    model = BEP_PFNO_Combined(
        m_sensors=cfg.m_sensors,
        modes=cfg.modes,
        width=cfg.width,
        hidden_dim=cfg.hidden_dim,
        scale_tuple=get_scale_tuple(cfg)
    )

    dynamic_weights = None
    if cfg.use_dynamic_weights:
        dynamic_weights = torch.tensor(
            cfg.dynamic_weight_init,
            dtype=torch.float64
        )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.adam_epochs,
        eta_min=1e-5
    )

    lhs_sampler = qmc.LatinHypercube(d=2, seed=cfg.seed)
    x_sensors = np.linspace(
        0, cfg.L_val, cfg.m_sensors
    )

    loss_history = []
    raw_total_loss_history = []
    loss_geo_history = []
    loss_const_history = []
    loss_eq1_history = []
    loss_eq2_history = []
    loss_bc_history = []
    weight_history = []
    weighted_geo_history = []
    weighted_const_history = []
    weighted_eq1_history = []
    weighted_eq2_history = []

    if cfg.use_dynamic_weights:
        print(
            "================ Start Adam"
            "（dynamic causal EMA loss weights） ================"
        )
        print(
            f"Dynamic loss weights enabled | "
            f"initial_weights={dynamic_weights.numpy().round(6)}"
        )
    else:
        print(
            "================ Start Adam "
            "（No dynamic weight） ================"
        )
    start_time = time.time()

    branch_input, x_batch, q_y_domain = make_batch(
        lhs_sampler, cfg, x_sensors
    )

    for epoch in range(cfg.adam_epochs):
        model.train()
        optimizer.zero_grad()

        if epoch % cfg.resample_freq == 0 or epoch == 0:
            branch_input, x_batch, q_y_domain = make_batch(
                lhs_sampler, cfg, x_sensors
            )

        used_weights = None
        if cfg.use_dynamic_weights:
            used_weights = dynamic_weights.clone().detach()

        loss, l_geo, l_const, l_eq1, l_eq2, l_bc = (
            compute_physics_loss_combined(
                model,
                branch_input,
                x_batch,
                q_y_domain,
                cfg.EI_val,
                cfg.L_val,
                cfg,
                weights=used_weights
            )
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=cfg.grad_clip_norm
        )
        optimizer.step()
        scheduler.step()

        physics_total = (
            l_geo.item() +
            l_const.item() +
            l_eq1.item() +
            l_eq2.item()
        )

        if cfg.use_dynamic_weights:
            weighted_geo = used_weights[0].item() * l_geo.item()
            weighted_const = used_weights[1].item() * l_const.item()
            weighted_eq1 = used_weights[2].item() * l_eq1.item()
            weighted_eq2 = used_weights[3].item() * l_eq2.item()
            with torch.no_grad():
                dynamic_weights = update_dynamic_weights(
                    cfg,
                    dynamic_weights,
                    l_geo,
                    l_const,
                    l_eq1,
                    l_eq2
                )
            weight_history.append(dynamic_weights.numpy().copy())
            weighted_geo_history.append(weighted_geo)
            weighted_const_history.append(weighted_const)
            weighted_eq1_history.append(weighted_eq1)
            weighted_eq2_history.append(weighted_eq2)

        loss_history.append(loss.item())
        raw_total_loss_history.append(physics_total)
        loss_geo_history.append(l_geo.item())
        loss_const_history.append(l_const.item())
        loss_eq1_history.append(l_eq1.item())
        loss_eq2_history.append(l_eq2.item())
        loss_bc_history.append(l_bc.item())

        if epoch % 100 == 0 or epoch == cfg.adam_epochs - 1:
            elapsed_time = time.time() - start_time
            if cfg.use_dynamic_weights:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Weighted Loss: {loss.item():.4e} | "
                    f"Raw Physics Loss: {physics_total:.4e} | "
                    f"Dynamic Weights: {used_weights.numpy().round(4)} | "
                    f"BC Loss: {l_bc.item():.4e} | "
                    f"Elapsed Time: {elapsed_time:.2f}s"
                )
            else:
                print(
                    f"Epoch {epoch:4d} | "
                    f"Total Loss: {loss.item():.4e} | "
                    f"Physics Loss: {physics_total:.4e} | "
                    f"BC: {l_bc.item():.4e} | "
                f"Times: {elapsed_time:.2f}s"
            )

    if cfg.use_dynamic_weights:
        print(
            "Adam Finish, start L-BFGS"
            "（Fixed dynamic weight at the end of Adam）..."
        )
    else:
        print(
            "Adam Finish, start L-BFGS"
            "（No dynamic Weights）..."
        )

    fixed_weights = None
    if cfg.use_dynamic_weights:
        fixed_weights = dynamic_weights.clone().detach()
        print(
            "L-BFGS will use fixed dynamic weights from the end of Adam: "
            f"{fixed_weights.numpy().round(6)}"
        )

    lbfgs_optimizer = optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=cfg.lbfgs_max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9
    )

    lbfgs_branch, lbfgs_x_batch, lbfgs_q_domain = (
        make_batch(lhs_sampler, cfg, x_sensors)
    )

    lbfgs_loss_history = []
    lbfgs_raw_total_loss_history = []
    lbfgs_loss_geo_history = []
    lbfgs_loss_const_history = []
    lbfgs_loss_eq1_history = []
    lbfgs_loss_eq2_history = []
    lbfgs_loss_bc_history = []
    lbfgs_weighted_geo_history = []
    lbfgs_weighted_const_history = []
    lbfgs_weighted_eq1_history = []
    lbfgs_weighted_eq2_history = []
    lbfgs_eval_count = [0]

    def closure():
        lbfgs_optimizer.zero_grad()

        loss, l_geo, l_const, l_eq1, l_eq2, l_bc = (
            compute_physics_loss_combined(
                model,
                lbfgs_branch,
                lbfgs_x_batch,
                lbfgs_q_domain,
                cfg.EI_val,
                cfg.L_val,
                cfg,
                weights=fixed_weights
            )
        )
        loss.backward()

        raw_parts = [
            l_geo.item(),
            l_const.item(),
            l_eq1.item(),
            l_eq2.item()
        ]

        lbfgs_loss_history.append(loss.item())
        lbfgs_raw_total_loss_history.append(sum(raw_parts))
        lbfgs_loss_geo_history.append(raw_parts[0])
        lbfgs_loss_const_history.append(raw_parts[1])
        lbfgs_loss_eq1_history.append(raw_parts[2])
        lbfgs_loss_eq2_history.append(raw_parts[3])
        lbfgs_loss_bc_history.append(l_bc.item())
        if cfg.use_dynamic_weights:
            lbfgs_weighted_geo_history.append(
                fixed_weights[0].item() * raw_parts[0]
            )
            lbfgs_weighted_const_history.append(
                fixed_weights[1].item() * raw_parts[1]
            )
            lbfgs_weighted_eq1_history.append(
                fixed_weights[2].item() * raw_parts[2]
            )
            lbfgs_weighted_eq2_history.append(
                fixed_weights[3].item() * raw_parts[3]
            )

        if lbfgs_eval_count[0] % 50 == 0:
            print(
                f"L-BFGS Eval {lbfgs_eval_count[0]:4d} | "
                f"Total Loss: {loss.item():.4e} | "
                f"Physics Loss: {sum(raw_parts):.4e}"
            )
        lbfgs_eval_count[0] += 1
        return loss

    lbfgs_optimizer.step(closure)

    total_time = time.time() - start_time
    print("=" * 50)
    print(
        f"Finish：{cfg.exp_group}/"
        f"{cfg.exp_id}/seed_{cfg.seed}"
    )
    print(
        f"Total time: {total_time:.2f} s "
        f"({total_time/60:.2f} min)"
    )
    print("=" * 50)

    final_dynamic_weights = (
        fixed_weights.numpy().tolist()
        if cfg.use_dynamic_weights
        else []
    )
    config_payload = asdict(cfg)
    config_payload.update({
        'loss_weighting': loss_weighting,
        'final_dynamic_weights': final_dynamic_weights,
    })

    model_checkpoint = {
        'config': config_payload,
        'm_sensors': cfg.m_sensors,
        'modes': cfg.modes,
        'width': cfg.width,
        'hidden_dim': cfg.hidden_dim,
        'L_val': cfg.L_val,
        'EI_val': cfg.EI_val,
        'sigma': cfg.sigma,
        'scale_tuple': get_scale_tuple(cfg),
        'use_dynamic_weights': cfg.use_dynamic_weights,
        'dynamic_weight_init': cfg.dynamic_weight_init,
        'dynamic_ema_alpha': cfg.dynamic_ema_alpha,
        'dynamic_causal_tolerance': cfg.dynamic_causal_tolerance,
        'dynamic_gamma': cfg.dynamic_gamma,
        'loss_weighting': loss_weighting,
        'final_dynamic_weights': final_dynamic_weights,
        'model_state_dict': model.state_dict(),
    }
    torch.save(
        model_checkpoint,
        save_dir / "model.pth"
    )

    training_logs = {
        'config': config_payload,
        'use_dynamic_weights': cfg.use_dynamic_weights,
        'dynamic_weight_init': cfg.dynamic_weight_init,
        'dynamic_ema_alpha': cfg.dynamic_ema_alpha,
        'dynamic_causal_tolerance': cfg.dynamic_causal_tolerance,
        'dynamic_gamma': cfg.dynamic_gamma,
        'loss_weighting': loss_weighting,
        'final_dynamic_weights': final_dynamic_weights,
        'train_time_s': total_time,
        'loss_history': loss_history,
        'raw_total_loss_history': raw_total_loss_history,
        'loss_geo_history': loss_geo_history,
        'loss_const_history': loss_const_history,
        'loss_eq1_history': loss_eq1_history,
        'loss_eq2_history': loss_eq2_history,
        'loss_bc_history': loss_bc_history,
        'weight_history': weight_history,
        'weighted_geo_history': weighted_geo_history,
        'weighted_const_history': weighted_const_history,
        'weighted_eq1_history': weighted_eq1_history,
        'weighted_eq2_history': weighted_eq2_history,
        'lbfgs_loss_history': lbfgs_loss_history,
        'lbfgs_raw_total_loss_history':
            lbfgs_raw_total_loss_history,
        'lbfgs_loss_geo_history':
            lbfgs_loss_geo_history,
        'lbfgs_loss_const_history':
            lbfgs_loss_const_history,
        'lbfgs_loss_eq1_history':
            lbfgs_loss_eq1_history,
        'lbfgs_loss_eq2_history':
            lbfgs_loss_eq2_history,
        'lbfgs_loss_bc_history':
            lbfgs_loss_bc_history,
        'lbfgs_weighted_geo_history':
            lbfgs_weighted_geo_history,
        'lbfgs_weighted_const_history':
            lbfgs_weighted_const_history,
        'lbfgs_weighted_eq1_history':
            lbfgs_weighted_eq1_history,
        'lbfgs_weighted_eq2_history':
            lbfgs_weighted_eq2_history,
    }
    torch.save(
        training_logs,
        save_dir / "logs.pth"
    )

    with open(
        save_dir / "config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            config_payload,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Model saving: {save_dir / 'model.pth'}"
    )
    print(
        f"Logs saving: {save_dir / 'logs.pth'}"
    )
    print(
        f"Config saving: {save_dir / 'config.json'}"
    )



def build_ablation_plan():
    plan = []

    plan.append(
        ExpConfig(
            exp_id="Base",
            exp_group="ablation"
        )
    )
    plan.append(
        ExpConfig(
            exp_id="Abl-BC-Soft",
            exp_group="ablation",
            ansatz_mode="soft",
            bc_penalty_coef=10.0
        )
    )
    plan.append(
        ExpConfig(
            exp_id="Abl-BC-None",
            exp_group="ablation",
            ansatz_mode="none",
            bc_penalty_coef=0.0
        )
    )
    plan.append(
        ExpConfig(
            exp_id="Abl-Scale-None",
            exp_group="ablation",
            use_response_scale=False
        )
    )
    plan.append(
        ExpConfig(
            exp_id="Abl-Scale-Half",
            exp_group="ablation",
            scale_tuple=(
                5e-4, 5e-4, 5e-3, 5e-3
            )
        )
    )
    plan.append(
        ExpConfig(
            exp_id="Abl-Scale-Double",
            exp_group="ablation",
            scale_tuple=(
                2e-3, 2e-3, 2e-2, 2e-2
            )
        )
    )
    plan.append(
        ExpConfig(
            exp_id="Abl-LoadScale-Off",
            exp_group="ablation",
            use_load_scale=False
        )
    )

    plan.append(
        ExpConfig(
            exp_id="Abl-Precond-Off",
            exp_group="ablation",
            use_response_scale=False,
            use_load_scale=False,
            use_norm=False
        )
    )
    plan.append(
        ExpConfig(
            exp_id="Abl-Precond-Off-DW",
            exp_group="ablation",
            use_response_scale=False,
            use_load_scale=False,
            use_norm=False,
            use_dynamic_weights=True
        )
    )
    plan.append(
        ExpConfig(
            exp_id="Abl-Norm-Off",
            exp_group="ablation",
            use_norm=False
        )
    )

    return plan


def build_sensor_plan():
    plan = []
    for m in [11, 21, 31, 41, 51, 61, 81, 101]:
        plan.append(ExpConfig(exp_id=f"Sensor_{m}", exp_group="sensor", m_sensors=m))
    return plan


def build_epoch_plan():
    plan = []
    for ep in [200, 400, 600, 800, 1000]:
        plan.append(ExpConfig(exp_id=f"Epoch_{ep}", exp_group="epoch", adam_epochs=ep))
    return plan


def build_experiment_plan(run_mode):
    if run_mode == "smoke":
        cfg = ExpConfig(exp_id="Smoke_Base", exp_group="smoke")
        cfg.adam_epochs = SMOKE_ADAM_EPOCHS
        cfg.lbfgs_max_iter = SMOKE_LBFGS_ITER
        return [cfg]

    plan = []
    if run_mode in ["all", "ablation"]:
        plan.extend(build_ablation_plan())
    if run_mode in ["all", "sensor"]:
        plan.extend(build_sensor_plan())
    if run_mode in ["all", "epoch"]:
        plan.extend(build_epoch_plan())
    return plan


def main():
    plan = build_experiment_plan(RUN_MODE)
    if not plan:
        raise ValueError(f"RUN_MODE={RUN_MODE} NO training")

    print("\n" + "#" * 100)
    print(f"BP-PFNO Chapter 3.3 Batch Training | RUN_MODE={RUN_MODE} | Seeds={RUN_SEEDS}")
    print(f"Total experiments: {len(plan)} x {len(RUN_SEEDS)} seed(s)")
    print("#" * 100)

    for base_cfg in plan:
        for seed in RUN_SEEDS:
            cfg = copy.deepcopy(base_cfg)
            cfg.seed = int(seed)
            cfg.save_root = SAVE_ROOT
            run_single_experiment(cfg)

    print("\nFinish")


# %%

if __name__ == '__main__':
    main()
