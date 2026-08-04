# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import time

RANDOM_SEED = 1202
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

TRAINING_SCRIPT = "BEP-PINO_Model_fixed-fixed_concentrated.py"
TARGET_RELATIVE_DIR = Path("BEP-PINO/Fixed-fixed/Fixed-fixed Concentrated load")
MODEL_SAVE_PATH = "BEP-PINO_Model_Concentrated_fixed-fixed.pth"
LOGS_SAVE_PATH = "BEP-PINO_Logs_Concentrated_fixed-fixed.pth"


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

# %%
class SpectralConv1d(nn.Module):
    """1D 傅里叶层 (Spectral Convolution)"""
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cdouble))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cdouble)
        modes = min(self.modes1, x.size(-1)//2 + 1)
        # 频域相乘
        out_ft[:, :, :modes] = torch.einsum("bix,iox->box", x_ft[:, :, :modes], self.weights1[:, :, :modes])
        # 逆变换回到物理域
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO_Combined(nn.Module):
    def __init__(self, m_sensors, modes=16, width=64, hidden_dim=128):
        super(FNO_Combined, self).__init__()
        self.m_sensors = m_sensors
        
        # FNO 输入通道: [q(x), x, EI, L] 共 4 个
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
            
        # 严格按照偏微分方程阶数从高到低构建
        self.decoder_Q = build_decoder(hidden_dim)           
        self.decoder_M = build_decoder(hidden_dim)       
        self.decoder_phi = build_decoder(hidden_dim)  
        self.decoder_u = build_decoder(hidden_dim)       
        
        self.U_scale = 1e-3
        self.Phi_scale = 1e-3
        self.M_scale = 1e-2
        self.Q_scale = 1e-2

    def forward(self, branch_input, x_batch):
        Batch_size = branch_input.shape[0]
        N_domain = x_batch.shape[1]
        
        # 解析输入特征
        q_sensors = branch_input[:, :self.m_sensors]
        params = branch_input[:, self.m_sensors:]
        
        # ================== 核心修复 1: 移除错误的端点线性连线 ==================
        # 使用 PyTorch 的 interpolate，将 m_sensors (如 51 点) 映射到 N_domain (如 100 点)
        # 从而在连续网格上完整保留高斯集中荷载的真实形状
        q_x = torch.nn.functional.interpolate(
            q_sensors.unsqueeze(1), size=N_domain, mode="linear", align_corners=True
        ).transpose(1, 2)
        # ========================================================================
        
        EI_x = params[:, 0:1].unsqueeze(2).expand(Batch_size, N_domain, 1)
        L_x = params[:, 1:2].unsqueeze(2).expand(Batch_size, N_domain, 1)
        
        fno_in = torch.cat([q_x, x_batch, EI_x, L_x], dim=-1)
        
        x = self.fc0(fno_in)
        x = x.permute(0, 2, 1)  # 转换为 [Batch, Channels, N_domain] 以适应 Conv1d
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = torch.nn.functional.silu(x1 + x2)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = torch.nn.functional.silu(x1 + x2)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = torch.nn.functional.silu(x1 + x2)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = torch.nn.functional.silu(x1 + x2)
        
        x = x.permute(0, 2, 1)  # 转换回 [Batch, N_domain, Channels]
        fused_features = self.fc1(x)
        
        # 前向传播
        Q_raw = self.decoder_Q(fused_features)
        M_raw = self.decoder_M(fused_features)
        phi_raw = self.decoder_phi(fused_features)
        u_raw = self.decoder_u(fused_features)

        
        return u_raw, phi_raw, M_raw, Q_raw

    def predict_with_ansatz(self, branch_input, x_batch, L, bc_type='fixed-fixed'):
        u_raw, phi_raw, M_raw, Q_raw = self.forward(branch_input, x_batch)
        
        # ================= 适应线性荷载的动态缩放 =================
        m_sensors = branch_input.shape[1] - 2 
        q_sensors = branch_input[:, 0:m_sensors]
        q_val_mean = torch.mean(q_sensors, dim=1, keepdim=True).unsqueeze(1)
        q_scale = q_val_mean / 0.1 
        
        u_scaled = u_raw * self.U_scale * q_scale
        phi_scaled = phi_raw * self.Phi_scale * q_scale
        M_scaled = M_raw * self.M_scale * q_scale
        Q_scaled = Q_raw * self.Q_scale * q_scale
        # =======================================================
        
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
        return u_pred, phi_pred, M_pred, Q_pred

# %%
def compute_gradients_fd(y, dx):
    """
    利用二阶有限差分计算梯度 (Finite Difference)
    - 内部点使用中心差分
    - 边界点使用二阶前向/后向非对称差分
    - 使用 torch.cat 拼接，防止 In-place 操作破坏 Autograd 的反向传播
    """
    # 内部点中心差分
    dy_dx_interior = (y[:, 2:, :] - y[:, :-2, :]) / (2 * dx)
    
    # 左边界前向差分
    dy_dx_left = (-3 * y[:, 0:1, :] + 4 * y[:, 1:2, :] - y[:, 2:3, :]) / (2 * dx)
    
    # 右边界后向差分
    dy_dx_right = (3 * y[:, -1:, :] - 4 * y[:, -2:-1, :] + y[:, -3:-2, :]) / (2 * dx)
    
    # 拼接保证计算图连续且形状不变
    dy_dx = torch.cat([dy_dx_left, dy_dx_interior, dy_dx_right], dim=1)
    return dy_dx

def compute_physics_loss_combined(model, branch_input, x_batch, q_y, EI_z, L, bc_type='fixed-fixed'):
    # 核心修改：不再需要 x_batch.requires_grad_(True)，极大节省显存和加速
    u_pred, phi_pred, M_pred, Q_pred = model.predict_with_ansatz(branch_input, x_batch, L, bc_type)
    
    # 计算均匀网格间距 dx
    dx = L / (x_batch.shape[1] - 1)
    
    # 使用数值有限差分 (FD) 计算空间导数
    du_dx = compute_gradients_fd(u_pred, dx)
    dphi_dx = compute_gradients_fd(phi_pred, dx)
    dM_dx = compute_gradients_fd(M_pred, dx)
    dQ_dx = compute_gradients_fd(Q_pred, dx)
    
    # ================= 损失归一化分母同步缩放 =================
    m_sensors = branch_input.shape[1] - 2
    q_sensors = branch_input[:, 0:m_sensors]
    q_norm_abs = (torch.mean(torch.abs(q_sensors), dim=1).view(-1, 1, 1) + 1e-4) / 0.1
    # =======================================================
    
    # 无量纲相对残差
    loss_geo_norm = torch.mean( ((du_dx - phi_pred) / (model.Phi_scale * q_norm_abs))**2 )
    loss_const_norm = torch.mean( ((M_pred - EI_z * dphi_dx) / (model.M_scale * q_norm_abs))**2 )
    loss_eq1_norm = torch.mean( ((Q_pred - dM_dx) / (model.Q_scale * q_norm_abs))**2 )
    loss_eq2_norm = torch.mean( ((dQ_dx - q_y) / (0.1 * q_norm_abs))**2 )  
    
    # 四项归一化物理损失采用等权求和，不使用动态权重
    loss_physics = loss_geo_norm + loss_const_norm + loss_eq1_norm + loss_eq2_norm
    return loss_physics, loss_geo_norm, loss_const_norm, loss_eq1_norm, loss_eq2_norm

# %%
if __name__ == '__main__':
    Batch_size, m_sensors, N_domain = 50, 51, 100 
    model = FNO_Combined(m_sensors=m_sensors, modes=16, width=64)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 800
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    L_val, EI_val = 1.0, 1.0       
    
    loss_history, loss_geo_history, loss_const_history, loss_eq1_history, loss_eq2_history, raw_total_loss_history = [], [], [], [], [], []
    
    # ================= 核心修改：集中荷载参数抽样 =================
    lhs_sampler = qmc.LatinHypercube(d=2)
    P_min_val, P_max_val = -5.0, -0.5    
    xp_min_val, xp_max_val = 0.2, 0.8    
    sigma = 0.04                         
    x_sensors = np.linspace(0, L_val, m_sensors)

    print("================ 开始 Adam 训练 (FNO - 集中荷载) ================")
    start_time = time.time()  
    
    # ================== 核心修复 2: 设置延迟重采样频率 ==================
    resample_freq = 10  
    # ====================================================================

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 仅在特定的 epoch 刷新数据集，其余时间使用缓存的批次进行深度收敛
        if epoch % resample_freq == 0 or epoch == 0:
            lhs_samples = lhs_sampler.random(n=Batch_size)
            P_np = P_min_val + lhs_samples[:, 0:1] * (P_max_val - P_min_val)
            xp_np = xp_min_val + lhs_samples[:, 1:2] * (xp_max_val - xp_min_val)
            
            # Branch 输入离散采样 (高斯集中荷载)
            q_sensor_data_np = P_np * np.exp(-((x_sensors - xp_np)**2) / (2 * sigma**2))
            q_sensor_data = torch.tensor(q_sensor_data_np, dtype=torch.float64)
            params_data = torch.tensor([[EI_val, L_val]]).expand(Batch_size, -1)
            branch_input = torch.cat([q_sensor_data, params_data], dim=1) 
            
            # Trunk 输入与域内荷载分布 (FNO 必须保持 linspace 均匀网格，不能 sort(rand))
            x_batch = torch.linspace(0, L_val, N_domain, dtype=torch.float64).view(1, N_domain, 1).expand(Batch_size, -1, -1)
            
            P_tensor = torch.tensor(P_np, dtype=torch.float64).unsqueeze(2) 
            xp_tensor = torch.tensor(xp_np, dtype=torch.float64).unsqueeze(2) 
            q_y_domain = P_tensor * torch.exp(-((x_batch - xp_tensor)**2) / (2 * sigma**2))

        # 此时的 branch_input 和 q_y_domain 在 epoch % 10 != 0 时，会沿用上一次生成的张量
        loss, l_geo, l_const, l_eq1, l_eq2 = compute_physics_loss_combined(
            model, branch_input, x_batch, q_y_domain, EI_val, L_val, 'fixed-fixed'
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        loss_geo_history.append(l_geo.item())
        loss_const_history.append(l_const.item())
        loss_eq1_history.append(l_eq1.item())
        loss_eq2_history.append(l_eq2.item())

        raw_geo, raw_const = l_geo.item(), l_const.item()
        raw_eq1, raw_eq2 = l_eq1.item(), l_eq2.item()
        raw_total_loss = raw_geo + raw_const + raw_eq1 + raw_eq2
        raw_total_loss_history.append(raw_total_loss)

        if epoch % 100 == 0 or epoch == epochs - 1:
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.4e} | 已用时: {elapsed_time:.2f}s")
    print("Adam 训练完成，启动 L-BFGS...")
    lbfgs_optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500, tolerance_grad=1e-7, tolerance_change=1e-9)
    
    # ================= L-BFGS 阶段同步集中荷载 =================
    lbfgs_lhs = lhs_sampler.random(n=Batch_size)
    lbfgs_P_np = P_min_val + lbfgs_lhs[:, 0:1] * (P_max_val - P_min_val)
    lbfgs_xp_np = xp_min_val + lbfgs_lhs[:, 1:2] * (xp_max_val - xp_min_val)
    
    lbfgs_q_sensor_np = lbfgs_P_np * np.exp(-((x_sensors - lbfgs_xp_np)**2) / (2 * sigma**2))
    lbfgs_branch = torch.cat([
        torch.tensor(lbfgs_q_sensor_np, dtype=torch.float64), 
        torch.tensor([[EI_val, L_val]]).expand(Batch_size, -1)
    ], dim=1) 
    
    lbfgs_x_batch = torch.linspace(0, L_val, N_domain, dtype=torch.float64).view(1, N_domain, 1).expand(Batch_size, -1, -1) 
    lbfgs_P_tensor = torch.tensor(lbfgs_P_np, dtype=torch.float64).unsqueeze(2)
    lbfgs_xp_tensor = torch.tensor(lbfgs_xp_np, dtype=torch.float64).unsqueeze(2)
    lbfgs_q_domain = lbfgs_P_tensor * torch.exp(-((lbfgs_x_batch - lbfgs_xp_tensor)**2) / (2 * sigma**2))

    lbfgs_loss_history, lbfgs_raw_total_loss_history = [], []
    lbfgs_loss_geo_history, lbfgs_loss_const_history, lbfgs_loss_eq1_history, lbfgs_loss_eq2_history = [], [], [], []
    lbfgs_eval_count = 0
    def closure():
        global lbfgs_eval_count
        lbfgs_optimizer.zero_grad()
        loss, l_geo, l_const, l_eq1, l_eq2 = compute_physics_loss_combined(
            model, lbfgs_branch, lbfgs_x_batch, lbfgs_q_domain, EI_val, L_val, 'fixed-fixed'
        )
        loss.backward()
        raw_parts = [l_geo.item(), l_const.item(), l_eq1.item(), l_eq2.item()]
        lbfgs_loss_history.append(loss.item())
        lbfgs_raw_total_loss_history.append(sum(raw_parts))
        lbfgs_loss_geo_history.append(raw_parts[0]); lbfgs_loss_const_history.append(raw_parts[1])
        lbfgs_loss_eq1_history.append(raw_parts[2]); lbfgs_loss_eq2_history.append(raw_parts[3])
    
        if lbfgs_eval_count % 50 == 0:
            print(f"L-BFGS Eval {lbfgs_eval_count:4d} | PDE Loss: {loss.item():.4e}")
        lbfgs_eval_count += 1
        return loss

    lbfgs_optimizer.step(closure)

    total_time = time.time() - start_time  
    print("=" * 50)
    print(f"🎉 训练全部完成！")
    print(f"⏱️ 总用时: {total_time:.2f} 秒 (约 {total_time/60:.2f} 分钟)")
    print("=" * 50)

# ========================== 1. 核心模型保存 ==========================
    model_checkpoint = {
        'm_sensors': m_sensors,
        'modes': 16,
        'width': 64,
        'hidden_dim': 128,
        'L_val': L_val,
        'EI_val': EI_val,
        'model_state_dict': model.state_dict()
    }
    model_save_path = BASE_DIR / MODEL_SAVE_PATH
    torch.save(model_checkpoint, model_save_path)
    print(f"✅ 核心模型参数已保存至: {model_save_path}")

# ========================== 2. 训练过程日志保存 ==========================
    training_logs = {
        'loss_history': loss_history,
        'loss_geo_history': loss_geo_history,
        'loss_const_history': loss_const_history,
        'loss_eq1_history': loss_eq1_history,
        'loss_eq2_history': loss_eq2_history,
        'raw_total_loss_history': raw_total_loss_history,
        'lbfgs_loss_history': lbfgs_loss_history, 'lbfgs_raw_total_loss_history': lbfgs_raw_total_loss_history,
        'lbfgs_loss_geo_history': lbfgs_loss_geo_history, 'lbfgs_loss_const_history': lbfgs_loss_const_history,
        'lbfgs_loss_eq1_history': lbfgs_loss_eq1_history, 'lbfgs_loss_eq2_history': lbfgs_loss_eq2_history
    }
    logs_save_path = BASE_DIR / LOGS_SAVE_PATH
    torch.save(training_logs, logs_save_path)
    print(f"✅ 训练过程日志已单独保存至: {logs_save_path}")
