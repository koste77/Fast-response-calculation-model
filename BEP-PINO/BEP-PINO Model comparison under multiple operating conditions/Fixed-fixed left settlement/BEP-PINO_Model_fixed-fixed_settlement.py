import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64)

RANDOM_SEED = 1202
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

TRAINING_SCRIPT = "BEP-PINO_Model_fixed-fixed_settlement.py"
TARGET_RELATIVE_DIR = Path("BEP-PINO Model comparison under multiple operating conditions/Fixed-fixed left settlement")
MODEL_SAVE_PATH = "BEP-PINO_Model_Settlement_fixed-fixed.pth"
LOGS_SAVE_PATH = "BEP-PINO_Logs_Settlement_fixed-fixed.pth"
BC_TYPE = "clamped-clamped-settlement"
LOAD_KIND = "settlement"


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
    return cwd


BASE_DIR = resolve_base_dir()


class SpectralConv1d(nn.Module):
    """1D Fourier spectral convolution."""

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cdouble)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cdouble,
        )
        modes = min(self.modes1, x.size(-1) // 2 + 1)
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :modes], self.weights1[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO_Combined(nn.Module):
    """FNO backbone with four parallel structural-response decoders."""

    def __init__(self, m_sensors, modes=16, width=64, hidden_dim=128):
        super(FNO_Combined, self).__init__()
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
                nn.Linear(hidden_dim // 2, 1),
            )

        self.decoder_Q = build_decoder(hidden_dim)
        self.decoder_M = build_decoder(hidden_dim)
        self.decoder_phi = build_decoder(hidden_dim)
        self.decoder_u = build_decoder(hidden_dim)

        self.U_scale = 0.1
        self.Phi_scale = 0.1
        self.M_scale = 1.0
        self.Q_scale = 1.0

    def forward(self, branch_input, x_batch):
        batch_size = branch_input.shape[0]
        n_domain = x_batch.shape[1]
        input_sensors = branch_input[:, :self.m_sensors]
        params = branch_input[:, self.m_sensors:]
        input_x = torch.nn.functional.interpolate(
            input_sensors.unsqueeze(1), size=n_domain, mode="linear", align_corners=True
        ).transpose(1, 2)
        EI_x = params[:, 0:1].unsqueeze(2).expand(batch_size, n_domain, 1)
        L_x = params[:, 1:2].unsqueeze(2).expand(batch_size, n_domain, 1)
        fno_in = torch.cat([input_x, x_batch, EI_x, L_x], dim=-1)

        x = self.fc0(fno_in).permute(0, 2, 1)
        x = torch.nn.functional.silu(self.conv0(x) + self.w0(x))
        x = torch.nn.functional.silu(self.conv1(x) + self.w1(x))
        x = torch.nn.functional.silu(self.conv2(x) + self.w2(x))
        x = torch.nn.functional.silu(self.conv3(x) + self.w3(x))
        fused_features = self.fc1(x.permute(0, 2, 1))
        return (
            self.decoder_u(fused_features),
            self.decoder_phi(fused_features),
            self.decoder_M(fused_features),
            self.decoder_Q(fused_features),
        )

    def predict_with_ansatz(self, branch_input, x_batch, L, bc_type=BC_TYPE):
        if bc_type != BC_TYPE:
            raise ValueError(f"Expected boundary type {BC_TYPE}, got {bc_type}")
        u_raw, phi_raw, M_raw, Q_raw = self.forward(branch_input, x_batch)
        case_value = branch_input[:, 0:1].unsqueeze(1)
        response_scale = case_value / 0.1
        u_scaled = u_raw * self.U_scale * response_scale
        phi_scaled = phi_raw * self.Phi_scale * response_scale
        M_scaled = M_raw * self.M_scale * response_scale
        Q_scaled = Q_raw * self.Q_scale * response_scale
        xi = x_batch / L
        lifting_u = case_value * (1.0 - 3.0 * xi**2 + 2.0 * xi**3)
        lifting_phi = case_value * (-6.0 * xi + 6.0 * xi**2) / L
        u_pred = lifting_u + (x_batch**2) * ((L - x_batch)**2) * u_scaled
        phi_pred = lifting_phi + x_batch * (L - x_batch) * phi_scaled
        M_pred = M_scaled
        Q_pred = Q_scaled
        return u_pred, phi_pred, M_pred, Q_pred


def compute_gradients_fd(y, dx):
    dy_dx_interior = (y[:, 2:, :] - y[:, :-2, :]) / (2 * dx)
    dy_dx_left = (-3 * y[:, 0:1, :] + 4 * y[:, 1:2, :] - y[:, 2:3, :]) / (2 * dx)
    dy_dx_right = (3 * y[:, -1:, :] - 4 * y[:, -2:-1, :] + y[:, -3:-2, :]) / (2 * dx)
    return torch.cat([dy_dx_left, dy_dx_interior, dy_dx_right], dim=1)


def compute_physics_loss_combined(model, branch_input, x_batch, q_y, EI_z, L, bc_type=BC_TYPE):
    u_pred, phi_pred, M_pred, Q_pred = model.predict_with_ansatz(branch_input, x_batch, L, bc_type)
    dx = L / (x_batch.shape[1] - 1)
    du_dx = compute_gradients_fd(u_pred, dx)
    dphi_dx = compute_gradients_fd(phi_pred, dx)
    dM_dx = compute_gradients_fd(M_pred, dx)
    dQ_dx = compute_gradients_fd(Q_pred, dx)
    case_value = branch_input[:, 0:1].unsqueeze(1)
    input_norm = (torch.abs(case_value) + 1e-4) / 0.1
    eq2_scale = model.Q_scale * input_norm
    loss_geo_norm = torch.mean(((du_dx - phi_pred) / (model.Phi_scale * input_norm))**2)
    loss_const_norm = torch.mean(((M_pred - EI_z * dphi_dx) / (model.M_scale * input_norm))**2)
    loss_eq1_norm = torch.mean(((Q_pred - dM_dx) / (model.Q_scale * input_norm))**2)
    loss_eq2_norm = torch.mean(((dQ_dx - q_y) / eq2_scale)**2)
    loss_physics = loss_geo_norm + loss_const_norm + loss_eq1_norm + loss_eq2_norm
    return loss_physics, loss_geo_norm, loss_const_norm, loss_eq1_norm, loss_eq2_norm


S_MIN_VAL, S_MAX_VAL = -0.1, 0.0
LHS_DIM = 1


def build_load_batch(lhs_samples, batch_size, m_sensors, n_domain, L_val, EI_val):
    settlement_np = S_MIN_VAL + lhs_samples[:, 0:1] * (S_MAX_VAL - S_MIN_VAL)
    sensor_np = np.repeat(settlement_np, m_sensors, axis=1)
    branch_input = torch.cat([
        torch.tensor(sensor_np, dtype=torch.float64),
        torch.tensor([[EI_val, L_val]], dtype=torch.float64).expand(batch_size, -1),
    ], dim=1)
    x_batch = torch.linspace(0.0, L_val, n_domain, dtype=torch.float64).view(1, n_domain, 1).expand(batch_size, -1, -1)
    q_domain = torch.zeros(batch_size, n_domain, 1, dtype=torch.float64)
    return branch_input, x_batch, q_domain


def load_metadata():
    return {"s_min_val": S_MIN_VAL, "s_max_val": S_MAX_VAL, "boundary_type": BC_TYPE}


if __name__ == "__main__":
    Batch_size, m_sensors, N_domain = 50, 1, 100
    modes, width, hidden_dim = 16, 64, 128
    L_val, EI_val = 1.0, 1.0

    model = FNO_Combined(m_sensors=m_sensors, modes=modes, width=width, hidden_dim=hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 800
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    loss_history, loss_geo_history, loss_const_history = [], [], []
    loss_eq1_history, loss_eq2_history, raw_total_loss_history = [], [], []
    lhs_sampler = qmc.LatinHypercube(d=LHS_DIM, seed=RANDOM_SEED)
    print("================ Adam Training: Support Settlement Input / Fixed-Fixed Settlement ================")
    start_time = time.time()
    resample_freq = 10

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        if epoch % resample_freq == 0:
            lhs_samples = lhs_sampler.random(n=Batch_size)
            branch_input, x_batch, q_y_domain = build_load_batch(
                lhs_samples, Batch_size, m_sensors, N_domain, L_val, EI_val
            )
        loss, l_geo, l_const, l_eq1, l_eq2 = compute_physics_loss_combined(
            model, branch_input, x_batch, q_y_domain, EI_val, L_val, BC_TYPE
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        raw_parts = [l_geo.item(), l_const.item(), l_eq1.item(), l_eq2.item()]
        loss_history.append(loss.item())
        loss_geo_history.append(raw_parts[0])
        loss_const_history.append(raw_parts[1])
        loss_eq1_history.append(raw_parts[2])
        loss_eq2_history.append(raw_parts[3])
        raw_total_loss_history.append(sum(raw_parts))

        if epoch % 100 == 0 or epoch == epochs - 1:
            elapsed_time = time.time() - start_time
            print(
                f"Epoch {epoch:4d} | Total Loss: {loss.item():.4e} | "
                f"Raw Loss: {sum(raw_parts):.4e} | Elapsed: {elapsed_time:.2f}s"
            )

    print("Adam training completed; starting L-BFGS.")
    lbfgs_optimizer = optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=500,
        tolerance_grad=1e-7, tolerance_change=1e-9,
    )
    lbfgs_sampler = qmc.LatinHypercube(d=LHS_DIM, seed=RANDOM_SEED + 1)
    lbfgs_lhs = lbfgs_sampler.random(n=Batch_size)
    lbfgs_branch, lbfgs_x_batch, lbfgs_q_domain = build_load_batch(
        lbfgs_lhs, Batch_size, m_sensors, N_domain, L_val, EI_val
    )
    lbfgs_loss_history, lbfgs_raw_total_loss_history = [], []
    lbfgs_loss_geo_history, lbfgs_loss_const_history = [], []
    lbfgs_loss_eq1_history, lbfgs_loss_eq2_history = [], []
    lbfgs_eval_count = [0]

    def closure():
        lbfgs_optimizer.zero_grad()
        loss, l_geo, l_const, l_eq1, l_eq2 = compute_physics_loss_combined(
            model, lbfgs_branch, lbfgs_x_batch, lbfgs_q_domain, EI_val, L_val, BC_TYPE
        )
        loss.backward()
        raw_parts = [l_geo.item(), l_const.item(), l_eq1.item(), l_eq2.item()]
        lbfgs_loss_history.append(loss.item())
        lbfgs_raw_total_loss_history.append(sum(raw_parts))
        lbfgs_loss_geo_history.append(raw_parts[0])
        lbfgs_loss_const_history.append(raw_parts[1])
        lbfgs_loss_eq1_history.append(raw_parts[2])
        lbfgs_loss_eq2_history.append(raw_parts[3])
        if lbfgs_eval_count[0] % 50 == 0:
            print(f"L-BFGS Eval {lbfgs_eval_count[0]:4d} | PDE Loss: {loss.item():.4e}")
        lbfgs_eval_count[0] += 1
        return loss

    lbfgs_optimizer.step(closure)
    total_time = time.time() - start_time
    print("=" * 70)
    print(f"Training completed in {total_time:.2f}s ({total_time / 60:.2f} min).")
    print("=" * 70)

    model_checkpoint = {
        "m_sensors": m_sensors,
        "modes": modes,
        "width": width,
        "hidden_dim": hidden_dim,
        "L_val": L_val,
        "EI_val": EI_val,
        "model_state_dict": model.state_dict(),
        **load_metadata(),
    }
    model_path = BASE_DIR / MODEL_SAVE_PATH
    torch.save(model_checkpoint, model_path)
    print(f"Model checkpoint saved to: {model_path}")

    training_logs = {
        "loss_history": loss_history,
        "loss_geo_history": loss_geo_history,
        "loss_const_history": loss_const_history,
        "loss_eq1_history": loss_eq1_history,
        "loss_eq2_history": loss_eq2_history,
        "raw_total_loss_history": raw_total_loss_history,
        "lbfgs_loss_history": lbfgs_loss_history,
        "lbfgs_raw_total_loss_history": lbfgs_raw_total_loss_history,
        "lbfgs_loss_geo_history": lbfgs_loss_geo_history,
        "lbfgs_loss_const_history": lbfgs_loss_const_history,
        "lbfgs_loss_eq1_history": lbfgs_loss_eq1_history,
        "lbfgs_loss_eq2_history": lbfgs_loss_eq2_history,
    }
    logs_path = BASE_DIR / LOGS_SAVE_PATH
    torch.save(training_logs, logs_path)
    print(f"Training logs saved to: {logs_path}")
