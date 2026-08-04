import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

class SpectralConv1d(nn.Module):

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
        out_ft[:, :, :modes] = torch.einsum("bix,iox->box", x_ft[:, :, :modes], self.weights1[:, :, :modes])
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class HO_BEP_PINO_Combined(nn.Module):
    def __init__(self, m_sensors, modes=16, width=64, hidden_dim=128):
        super(HO_BEP_PINO_Combined, self).__init__()
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
        
        self.U_scale = 1e-3
        self.Phi_scale = 1e-3
        self.M_scale = 1e-2
        self.Q_scale = 1e-2

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
        
        x = x.permute(0, 2, 1)  
        fused_features = self.fc1(x)

        Q_raw = self.decoder_Q(fused_features)
        M_raw = self.decoder_M(fused_features)
        phi_raw = self.decoder_phi(fused_features)
        u_raw = self.decoder_u(fused_features)

        return u_raw, phi_raw, M_raw, Q_raw

    def predict_with_ansatz(self, branch_input, x_batch, L, bc_type='fixed-fixed'):
        u_raw, phi_raw, M_raw, Q_raw = self.forward(branch_input, x_batch)
        
        m_sensors = branch_input.shape[1] - 2 
        q_sensors = branch_input[:, 0:m_sensors]
        q_val_mean = torch.mean(q_sensors, dim=1, keepdim=True).unsqueeze(1)
        q_scale = q_val_mean / 0.1 
        
        u_scaled = u_raw * self.U_scale * q_scale
        phi_scaled = phi_raw * self.Phi_scale * q_scale
        M_scaled = M_raw * self.M_scale * q_scale
        Q_scaled = Q_raw * self.Q_scale * q_scale
        
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

model_path = "HO-BEP-PINO_Model_fixed-fixed_Concentrated.pth"
logs_path = "HO-BEP-PINO_Logs_fixed-fixed_Concentrated.pth"
print(f"Loading model data: {model_path} ...")

model_data = torch.load(model_path, weights_only=False)
L_val = model_data['L_val']
EI_val = model_data['EI_val']
m_sensors = model_data['m_sensors']

model = HO_BEP_PINO_Combined(
    m_sensors=m_sensors, 
    modes=model_data['modes'], 
    width=model_data['width']
)
model.load_state_dict(model_data['model_state_dict'])
model.eval()

logs_data = torch.load(logs_path, weights_only=False)
loss_history = logs_data['loss_history']
raw_total_loss_history = np.array(logs_data['raw_total_loss_history'])
lbfgs_loss_history = logs_data['lbfgs_loss_history']


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
        [12,       6 * Le,   -12,       6 * Le],
        [6 * Le,   4 * Le**2, -6 * Le,   2 * Le**2],
        [-12,     -6 * Le,    12,      -6 * Le],
        [6 * Le,   2 * Le**2, -6 * Le,   4 * Le**2]
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

def solve_beam_fem_fixed_gaussian(P, xp, sigma=0.04, L=1.0, EI=1.0, num_nodes=1000, gauss_order=8):
    if num_nodes < 3: raise ValueError("num_nodes must be at least 3.")
    if sigma <= 0: raise ValueError("sigma must be positive.")

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


def solve_beam_fem_fixed_gaussian_on_grid(P, xp, x_target, sigma=0.04, L=1.0, EI=1.0, num_nodes_ref=1000, gauss_order=8):
    x_target = np.asarray(x_target, dtype=np.float64).reshape(-1)
    x_ref, u_ref, phi_ref, M_ref, Q_ref = solve_beam_fem_fixed_gaussian(
        P=P, xp=xp, sigma=sigma, L=L, EI=EI, num_nodes=num_nodes_ref, gauss_order=gauss_order
    )

    u_fem = np.interp(x_target, x_ref, u_ref)
    phi_fem = np.interp(x_target, x_ref, phi_ref)
    M_fem = np.interp(x_target, x_ref, M_ref)
    Q_fem = np.interp(x_target, x_ref, Q_ref)

    return u_fem, phi_fem, M_fem, Q_fem

def plot_fixed_symbols(ax, color='black'):
    ax.plot(0, 0, 's', markersize=10, color=color, zorder=10) 
    ax.plot(L_val, 0, 's', markersize=10, color=color, zorder=10)
    if x_sensors is not None:
        ax.plot(x_sensors, np.zeros_like(x_sensors), 'o', markersize=4, color='green', zorder=9, alpha=0.6, 
        label='Sensors' if 'Sensors' not in ax.get_legend_handles_labels()[1] else "")

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
    'figure.dpi': 500, 'savefig.dpi': 500,
    'font.size': 16,           
    'xtick.labelsize': 16,   
    'ytick.labelsize': 16    
})

def plot_standardized_training_history(logs):
    components = [
        ('geo', 'Geometric'),
        ('const', 'Constitutive'),
        ('eq1', 'Equilibrium 1'),
        ('eq2', 'Equilibrium 2')
    ]

    total_adam_color = "#C0321A"      
    total_lbfgs_color = "#218D42"     

    component_colors = {
        'geo':   "#8074C8",   # Geometric
        'const': "#7895C1",   # Constitutive
        'eq1':   "#59A14F",   # Equilibrium 1
        'eq2':   "#EF8B67",   # Equilibrium 2
    }


    def stages(adam_key, lbfgs_key, label, adam_color=None, lbfgs_color=None):
        adam = np.asarray(logs.get(adam_key, []), dtype=float)
        lbfgs = np.asarray(logs.get(lbfgs_key, []), dtype=float)

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
                np.arange(adam.size, adam.size + lbfgs.size),
                lbfgs,
                color=lbfgs_color,
                linewidth=2.0,
                linestyle='--',
                label=f'L-BFGS {label}'
            )

    plt.figure(figsize=(10, 6))
    stages(
        'loss_history',
        'lbfgs_loss_history',
        'Total Physics Loss',
        adam_color=total_adam_color,
        lbfgs_color=total_lbfgs_color
    )
    plt.title('Total Physics Loss')
    plt.ylabel('Loss (Log Scale)', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for key, label in components:
        stages(
            f'loss_{key}_history',
            f'lbfgs_loss_{key}_history',
            label,
            adam_color=component_colors[key],
            lbfgs_color=component_colors[key]
        )
    plt.title('Physics Loss Components')
    plt.ylabel('Loss (Log Scale)', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
plot_standardized_training_history(logs_data)

plt.figure(figsize=(10, 6))
plt.semilogy(loss_history, color="#C0321A" , linewidth=1.5, label='Total Physics Loss')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss (Log Scale)', fontsize=16)
plt.title('Training Convergence Curve (HO FNO Model)', fontsize=16)    
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(logs_data['raw_total_loss_history'], color="#C0321A" , linewidth=1.5, label='raw_total_loss_history')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss (Log Scale)', fontsize=16)
plt.title('Training Convergence Curve (HO FNO Model)', fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
len_adam = len(loss_history)
len_lbfgs = len(lbfgs_loss_history)
x_adam = np.arange(len_adam)
x_lbfgs = np.arange(len_adam, len_adam + len_lbfgs)
plt.semilogy(x_adam, loss_history, color="#C0321A", linewidth=1.5, label='Adam Stage (Epochs)')
plt.semilogy(x_lbfgs, lbfgs_loss_history, color="#218D42", linewidth=2.0, label='L-BFGS Stage (Evaluations)')
plt.axvline(x=len_adam, color='gray', linestyle='--', linewidth=1.5, label='Optimizer Switch')
ax = plt.gca()
trans = ax.get_xaxis_transform()
plt.text(len_adam * 0.5, 0.1, 'Global Search\n(1st Order)', horizontalalignment='center', color="#C0321A", fontsize=14, alpha=0.8, transform=trans)   
plt.text(len_adam + len_lbfgs * 0.5, 0.1, 'Fine-tuning\n(2nd Order)', horizontalalignment='center', color="#218D42", fontsize=14, alpha=0.8, transform=trans)
plt.xlabel('Optimization Steps (Adam Epochs + L-BFGS Evaluations)', fontsize=16)
plt.ylabel('Total PDE Loss (Log Scale)', fontsize=16)
plt.title('Complete Training Convergence: Adam $\\rightarrow$ L-BFGS', fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper right', fontsize=14)
plt.tight_layout()
plt.show()


sigma_val = 0.04
test_loads = [(-3.0, 0.3), (-2.0, 0.5), (-2.5, 0.7)] 
colors = ['#4583b6', '#B02425', '#218D42'] 
test_x = torch.linspace(0, L_val, 100).reshape(1, -1, 1)
x_np = test_x.squeeze().numpy()
x_sensors = np.linspace(0, L_val, m_sensors)

fig_q, ax_q = plt.subplots(figsize=(10, 3))
fig_u, ax_u = plt.subplots(figsize=(5, 3))
fig_phi, ax_phi = plt.subplots(figsize=(5, 3))
fig_M, ax_M = plt.subplots(figsize=(5, 3))
fig_Q, ax_Q = plt.subplots(figsize=(5, 3))
fig_err, ax_err_matrix = plt.subplots(2, 2, figsize=(16, 12), dpi=500)
ax_err = ax_err_matrix.flatten() 

global_metrics_summary = {
    "Deflection (u)": {"mae": [], "l2_rel": [], "r2": []},
    "Rotation (phi)": {"mae": [], "l2_rel": [], "r2": []},
    "Moment (M)": {"mae": [], "l2_rel": [], "r2": []},
    "Shear Force (Q)": {"mae": [], "l2_rel": [], "r2": []}
}
    
for i, (P, xp) in enumerate(test_loads):
    q_sensor_data = P * np.exp(-((x_sensors - xp)**2) / (2 * sigma_val**2))
    test_branch_input = torch.cat([
        torch.tensor([q_sensor_data], dtype=torch.float64), 
        torch.tensor([[EI_val, L_val]], dtype=torch.float64)
    ], dim=1)

    q_plot_continuous = P * np.exp(-((x_np - xp)**2) / (2 * sigma_val**2))
    lbl = f'$P$={P}, $x_p$={xp}'
    ax_q.plot(x_np, q_plot_continuous, color=colors[i], label=f'$q(x)$ ({lbl})', linewidth=3, alpha=0.8)
    ax_q.fill_between(x_np, q_plot_continuous, color=colors[i], alpha=0.1)
    
    with torch.no_grad():
        u_pred, phi_pred, M_pred, Q_pred = model.predict_with_ansatz(test_branch_input, test_x, L=L_val, bc_type='fixed-fixed')
        u_pred_np, phi_pred_np = u_pred.squeeze().numpy(), phi_pred.squeeze().numpy() 
        M_pred_np, Q_pred_np = M_pred.squeeze().numpy(), Q_pred.squeeze().numpy()

    u_fem, phi_fem, M_fem, Q_fem = solve_beam_fem_fixed_gaussian_on_grid(
        P=P, xp=xp, x_target=x_np, sigma=sigma_val, L=L_val, EI=EI_val,
        num_nodes_ref=1000, gauss_order=8
    )

    error_u, error_phi = np.abs(u_pred_np - u_fem), np.abs(phi_pred_np - phi_fem) 
    error_M, error_Q = np.abs(M_pred_np - M_fem), np.abs(Q_pred_np - Q_fem)

    lbl = f'$P$={P}, $x_p$={xp}'
    ax_u.plot(x_np, u_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
    ax_u.plot(x_np, u_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')
    ax_phi.plot(x_np, phi_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
    ax_phi.plot(x_np, phi_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')
    ax_M.plot(x_np, M_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
    ax_M.plot(x_np, M_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')
    ax_Q.plot(x_np, Q_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
    ax_Q.plot(x_np, Q_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')

    error_data = [error_u, error_phi, error_M, error_Q]
    for j, err in enumerate(error_data):
        ax_err[j].plot(x_np, err, color=colors[i], label=f'Error ({lbl})', zorder=5-i)
        ax_err[j].fill_between(x_np, err, color=colors[i], alpha=0.1)

    def evaluate_metrics(name, pred, fem):
        abs_err_arr = np.abs(pred - fem)
        mae_val = np.mean(abs_err_arr)
        l2_rel_val = np.linalg.norm(pred - fem) / (np.linalg.norm(fem) + 1e-12)
        ss_res = np.sum((fem - pred) ** 2)
        ss_tot = np.sum((fem - np.mean(fem)) ** 2) + 1e-12
        r2_val = 1 - (ss_res / ss_tot)
        return {'name': name, 'mae': mae_val, 'l2_rel': l2_rel_val, 'r2': r2_val}

    metrics = [
        evaluate_metrics("Deflection (u)", u_pred_np, u_fem),
        evaluate_metrics("Rotation (phi)", phi_pred_np, phi_fem),
        evaluate_metrics("Moment (M)", M_pred_np, M_fem),
        evaluate_metrics("Shear Force (Q)", Q_pred_np, Q_fem)
    ]

    for m in metrics:
        global_metrics_summary[m['name']]["mae"].append(m['mae'])
        global_metrics_summary[m['name']]["l2_rel"].append(m['l2_rel'])
        global_metrics_summary[m['name']]["r2"].append(m['r2'])

plot_configs = [
    (fig_q, ax_q, 'Load Input $q(x)$', 'Gaussian Localized Load Distribution', {'loc': 'center left'}), 
    (fig_u, ax_u, 'Deflection $(u)$', 'Deflection Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'lower right'}),
    (fig_phi, ax_phi, 'Rotation $(\\phi)$', 'Rotation Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'lower right'}),
    (fig_M, ax_M, 'Moment $(M)$', 'Moment Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'upper center'}),
    (fig_Q, ax_Q, 'Shear Force $(Q)$', 'Shear Force Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'upper right'})
]

for fig, ax, ylabel, title, legend_kwargs in plot_configs:
    ax.hlines(0, 0, L_val, color='black', linewidth=3, zorder=5)
    plot_fixed_symbols(ax)
    ax.set_xlabel('Beam Position $(x)$', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20, pad=10)

    if ax == ax_M:
        ax.invert_yaxis()

    if ax == ax_q:
        ax.legend(
            loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, 
            fontsize=16, handlelength=2.2, labelspacing=0.45, borderaxespad=0.0
        )
        fig.subplots_adjust(right=0.76)
    else:
        leg = ax.get_legend()
        if leg is not None: leg.remove()

    fig.tight_layout()
    fig.show()


handles, labels = ax_u.get_legend_handles_labels()
fig_legend, ax_legend = plt.subplots(figsize=(10, 0.4), dpi=500)
ax_legend.axis('off')
fig_legend.legend(
    handles, labels, loc='center', ncol=4, frameon=False, 
    fontsize=16, handlelength=2.5, columnspacing=1.5, labelspacing=0.5
)
fig_legend.tight_layout()
fig_legend.show()

err_configs = [
    ('Absolute Error: Deflection $(u)$', {'loc': 'upper center'}), 
    ('Absolute Error: Rotation $(\\phi)$', {'loc': 'upper center'}), 
    ('Absolute Error: Moment $(M)$', {'loc': 'upper right'}), 
    ('Absolute Error: Shear Force $(Q)$', {'loc': 'upper left'})
]
for j, ax in enumerate(ax_err):
    title, legend_kwargs = err_configs[j]
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Beam Position $(x)$', fontsize=16)
    ax.set_ylabel('Error Magnitude', fontsize=16)
    ax.hlines(0, 0, L_val, color='black', linewidth=3, zorder=5)
    plot_fixed_symbols(ax)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(**legend_kwargs)
fig_err.tight_layout()
fig_err.show()


print("\n" + "="*85)
print(" HO-BEP-PINO vs high precision Hermite FEM Quantitative evaluation report on structural response depth (High-Order Formulation)")
print("="*85)

for i, (P, xp) in enumerate(test_loads):
    q_sensor_data = P * np.exp(-((x_sensors - xp)**2) / (2 * sigma_val**2))
    test_branch_input = torch.cat([
        torch.tensor([q_sensor_data], dtype=torch.float64), 
        torch.tensor([[EI_val, L_val]], dtype=torch.float64)
    ], dim=1)
    
    with torch.no_grad():
        u_pred, phi_pred, M_pred, Q_pred = model.predict_with_ansatz(test_branch_input, test_x, L=L_val, bc_type='fixed-fixed')
        u_pred_np, phi_pred_np = u_pred.squeeze().numpy(), phi_pred.squeeze().numpy()
        M_pred_np, Q_pred_np = M_pred.squeeze().numpy(), Q_pred.squeeze().numpy()

    u_fem, phi_fem, M_fem, Q_fem = solve_beam_fem_fixed_gaussian_on_grid(
        P=P, xp=xp, x_target=x_np, sigma=sigma_val, L=L_val, EI=EI_val,
        num_nodes_ref=1000, gauss_order=8
    )

    def evaluate_metrics(name, pred, fem):
        abs_err_arr = np.abs(pred - fem)
        peak_fem_abs, peak_pred_abs = np.max(np.abs(fem)), np.max(np.abs(pred))   
        
        idx_max_err = np.argmax(abs_err_arr)   
        max_abs_err = abs_err_arr[idx_max_err]
        rel_err_at_max = (max_abs_err / peak_fem_abs * 100) if peak_fem_abs > 1e-12 else 0.0
        abs_diff_peaks = np.abs(peak_pred_abs - peak_fem_abs) 
        rel_diff_peaks = (abs_diff_peaks / peak_fem_abs * 100) if peak_fem_abs > 1e-12 else 0.0
        
        return {
            'name': name, 'max_err': max_abs_err, 'pred_at_err': pred[idx_max_err], 'fem_at_err': fem[idx_max_err], 'rel_err': rel_err_at_max,
            'peak_pred': peak_pred_abs, 'peak_fem': peak_fem_abs, 'peak_diff': abs_diff_peaks, 'peak_rel': rel_diff_peaks
        }

    metrics = [
        evaluate_metrics("Deflection $(u)$", u_pred_np, u_fem),
        evaluate_metrics("Rotation $(\\phi)$", phi_pred_np, phi_fem),
        evaluate_metrics("Moment $(M)$", M_pred_np, M_fem),
        evaluate_metrics("Shear Force $(Q)$", Q_pred_np, Q_fem)
    ]

    print(f"\n Condition {i+1}: local load peak P = {P}, center position xp = {xp}, sigma = {sigma_val}")
    print("=" * 85)
    print(f"【Table 1】Analysis of spatial maximum deviation location (find the point with the worst global fitting)")
    print(f"{'Physical Quantity':^9} | {'Maximum Absolute Error':^15} | {'Predicted Value':^15} | {'FEM Value':^15} | {'Relative Peak Error':^15}")
    print("-" * 85)
    for m in metrics: print(f"{m['name']:^10} | {m['max_err']:>15.4e} | {m['pred_at_err']:>15.4e} | {m['fem_at_err']:>15.4e} | {m['rel_err']:>12.4f} %")
    
    print("-" * 85)
    print(f"【Table 2】Global peak response evaluation ")
    print(f"{'Physical Quantity':^9} | {'Maximum Predicted Value (Absolute)':^13} | {'Maximum FEM Value (Absolute)':^13} | {'Peak Absolute Deviation':^15} | {'Peak Relative Deviation':^15}")
    print("-" * 85)
    for m in metrics: print(f"{m['name']:^10} | {m['peak_pred']:>15.4e} | {m['peak_fem']:>15.4e} | {m['peak_diff']:>15.4e} | {m['peak_rel']:>12.4f} %")
    print("=" * 85)

print("\n\n" + "-"*85)
print(" Summary of global comprehensive accuracy evaluation (average of all test conditions)")
print("-"*85)
print(f"{'Physical Quantity':^9} | {'Mean MAE':^16} | {'Mean Relative L²':^20} | {'Mean R²':^15}")
print("-" * 85)
for name, data_dict in global_metrics_summary.items():
    mean_mae = np.mean(data_dict["mae"])
    mean_l2 = np.mean(data_dict["l2_rel"])
    mean_r2 = np.mean(data_dict["r2"])
    print(f"{name:^10} | {mean_mae:>16.4e} | {mean_l2:>20.4e} | {mean_r2:>15.6f}")
print("-" * 85 + "\n")


def get_last_value(logs, key):
    values = logs.get(key, [])
    if values is None or len(values) == 0:
        return np.nan
    return float(np.asarray(values, dtype=float)[-1])

adam_ultimate_loss = get_last_value(logs_data, "loss_history")
adam_ultimate_raw_loss = get_last_value(logs_data, "raw_total_loss_history")
lbfgs_final_loss = get_last_value(logs_data, "lbfgs_loss_history")
lbfgs_final_raw_loss = get_last_value(logs_data, "lbfgs_raw_total_loss_history")

print("\n" + "=" * 85)
print(" Final Training Loss Summary")
print("=" * 85)
print(f"{'Stage':^16} | {'Total Loss':^22} | {'Raw Total Loss':^22}")
print("-" * 85)
print(f"{'Adam ultimate':^16} | {adam_ultimate_loss:>22.6e} | {adam_ultimate_raw_loss:>22.6e}")
print(f"{'L-BFGS final':^16} | {lbfgs_final_loss:>22.6e} | {lbfgs_final_raw_loss:>22.6e}")
print("=" * 85)
