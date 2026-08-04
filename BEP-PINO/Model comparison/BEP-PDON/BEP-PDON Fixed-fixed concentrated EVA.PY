import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ResMLP, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())

        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            for _ in range(num_layers - 2)
        ])

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = x + layer(x)  
        return x


class BEP_PDON_Combined(nn.Module):
    def __init__(self, m_sensors, hidden_dim=128, num_layers=4):
        super(BEP_PDON_Combined, self).__init__()

        self.branch_net = ResMLP(m_sensors + 2, hidden_dim, num_layers)
        self.trunk_net = ResMLP(1, hidden_dim, num_layers)

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
        self.Q_scale = 1e-1

    def forward(self, branch_input, x_batch):
        b_features = self.branch_net(branch_input).unsqueeze(1)
        t_features = self.trunk_net(x_batch)
        fused_features = b_features * t_features

        Q_raw = self.decoder_Q(fused_features)
        M_input = torch.cat([fused_features], dim=-1)
        M_raw = self.decoder_M(M_input)
        phi_input = torch.cat([fused_features], dim=-1)
        phi_raw = self.decoder_phi(phi_input)
        u_input = torch.cat([fused_features], dim=-1)
        u_raw = self.decoder_u(u_input)

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



model_path = "BEP-PDON_Model_Fixed-fixed concentrated.pth"
logs_path = "BEP-PDON_Logs_Fixed-fixed concentrated.pth"
print(f" Loading model data: {model_path} ...")

model_data = torch.load(model_path, weights_only=False)
L_val, EI_val, m_sensors = model_data['L_val'], model_data['EI_val'], model_data['m_sensors']

state_dict = model_data['model_state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model = BEP_PDON_Combined(
    m_sensors=m_sensors,
    hidden_dim=model_data['hidden_dim'],
    num_layers=model_data['num_layers']
)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    raise RuntimeError(
        f"failed\nmissing keys: {missing}\nunexpected keys: {unexpected}"
    )
model.eval()

logs_data = torch.load(logs_path, weights_only=False)
loss_history = logs_data['loss_history']


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


def plot_standardized_training_history(logs):
    components = [
        ('geo', 'Geometric'),
        ('const', 'Constitutive'),
        ('eq1', 'Equilibrium 1'),
        ('eq2', 'Equilibrium 2')
    ]

    def plot_stages(adam_key, lbfgs_key, label):
        adam = np.asarray(logs.get(adam_key, []), dtype=float)
        lbfgs = np.asarray(logs.get(lbfgs_key, []), dtype=float)
        if adam.size:
            plt.semilogy(np.arange(adam.size), adam, label=f'Adam {label}')
        if lbfgs.size:
            plt.semilogy(
                np.arange(adam.size, adam.size + lbfgs.size),
                lbfgs,
                linestyle='--',
                label=f'L-BFGS {label}'
            )

    plt.figure(figsize=(10, 6))
    plot_stages('loss_history', 'lbfgs_loss_history', 'Total Physics Loss')
    plt.xlabel('Optimization Step', fontsize=16)
    plt.ylabel('Loss (Log Scale)', fontsize=16)
    plt.title('Total Physics Loss', fontsize=16)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for key, label in components:
        plot_stages(f'loss_{key}_history', f'lbfgs_loss_{key}_history', label)
    plt.xlabel('Optimization Step', fontsize=16)
    plt.ylabel('Loss (Log Scale)', fontsize=16)
    plt.title('Physics Loss Components', fontsize=16)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


plot_standardized_training_history(logs_data)


def solve_beam_fem_fixed_gaussian(P, xp, sigma=0.04, L=1.0, EI=1.0, num_nodes=100):
    n_elem = num_nodes - 1
    total_dof = num_nodes * 2
    x = np.linspace(0, L, num_nodes)
    Le = L / n_elem
    K, F = np.zeros((total_dof, total_dof)), np.zeros(total_dof)
    ke = (EI / Le**3) * np.array([
        [12, 6*Le, -12, 6*Le],
        [6*Le, 4*Le**2, -6*Le, 2*Le**2],
        [-12, -6*Le, 12, -6*Le],
        [6*Le, 2*Le**2, -6*Le, 4*Le**2]
    ])

    xi, w = np.polynomial.legendre.leggauss(5)

    for e in range(n_elem):
        x1 = x[e]
        fe = np.zeros(4)
        for i in range(len(xi)):
            s = (Le / 2) * (xi[i] + 1)
            x_g = x1 + s
            q_g = P * np.exp(-((x_g - xp)**2) / (2 * sigma**2))
            N = np.array([
                1 - 3*(s/Le)**2 + 2*(s/Le)**3,
                s * (1 - s/Le)**2,
                3*(s/Le)**2 - 2*(s/Le)**3,
                s * ((s/Le)**2 - s/Le)
            ])
            fe += w[i] * N * q_g * (Le / 2)

        dof = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for i in range(4):
            F[dof[i]] += fe[i]
            for j in range(4):
                K[dof[i], dof[j]] += ke[i, j]

    free_dofs = np.setdiff1d(np.arange(total_dof), [0, 1, total_dof-2, total_dof-1])
    U = np.zeros(total_dof)
    U[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], F[free_dofs])

    w_fem, phi_fem = U[0::2], U[1::2]
    M_fem, Q_fem = np.zeros(num_nodes), np.zeros(num_nodes)

    for e in range(n_elem):
        u_e = np.array([U[2*e], U[2*e+1], U[2*(e+1)], U[2*(e+1)+1]])
        x1 = x[e]
        fe = np.zeros(4)
        for i in range(len(xi)):
            s = (Le / 2) * (xi[i] + 1)
            x_g = x1 + s
            q_g = P * np.exp(-((x_g - xp)**2) / (2 * sigma**2))
            N = np.array([
                1 - 3*(s/Le)**2 + 2*(s/Le)**3,
                s * (1 - s/Le)**2,
                3*(s/Le)**2 - 2*(s/Le)**3,
                s * ((s/Le)**2 - s/Le)
            ])
            fe += w[i] * N * q_g * (Le / 2)

        P_e = np.dot(ke, u_e) - fe
        Q_fem[e], M_fem[e] = P_e[0], -P_e[1]
        if e == n_elem - 1:
            Q_fem[e+1], M_fem[e+1] = -P_e[2], P_e[3]

    return x, w_fem, phi_fem, M_fem, Q_fem


def plot_fixed_symbols(ax, x_sensors=None, color='black'):
    ax.plot(0, 0, 's', markersize=10, color=color, zorder=10)
    ax.plot(L_val, 0, 's', markersize=10, color=color, zorder=10)

    if x_sensors is None:
        x_sensors = globals().get('x_sensors', None)

    if x_sensors is not None:
        ax.plot(
            x_sensors,
            np.zeros_like(x_sensors),
            'o',
            markersize=4,
            color='green',
            zorder=9,
            alpha=0.6,
            label='Sensors' if 'Sensors' not in ax.get_legend_handles_labels()[1] else ""
        )


plt.figure(figsize=(10, 6))
if 'loss_geo_history' in logs_data:
    plt.semilogy(logs_data['loss_geo_history'], color="#8074C8", linewidth=1.5, label='Geo Loss ($du/dx = \\phi$)')
if 'loss_const_history' in logs_data:
    plt.semilogy(logs_data['loss_const_history'], color="#7895C1", linewidth=1.5, label='Const Loss ($M = EI \\cdot d\\phi/dx$)')
if 'loss_eq1_history' in logs_data:
    plt.semilogy(logs_data['loss_eq1_history'], color="#59E347", linewidth=1.5, label='Eq1 Loss ($Q = dM/dx$)')
if 'loss_eq2_history' in logs_data:
    plt.semilogy(logs_data['loss_eq2_history'], color="#EF8B67", linewidth=1.5, label='Eq2 Loss ($dQ/dx = q$)')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss (Log Scale)', fontsize=16)
plt.title('Individual Physics Component Losses (Adam Phase)', fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
adam_epochs = np.arange(len(logs_data.get('loss_geo_history', [])))
has_lbfgs = 'lbfgs_loss_geo_history' in logs_data and len(logs_data['lbfgs_loss_geo_history']) > 0
if has_lbfgs:
    lbfgs_iters = np.arange(len(adam_epochs), len(adam_epochs) + len(logs_data['lbfgs_loss_geo_history']))

def plot_combined_component(adam_key, lbfgs_key, color, label):
    if adam_key in logs_data:
        adam_data = logs_data[adam_key]
        plt.semilogy(adam_epochs, adam_data, color=color, linewidth=1.5, label=label)
        if has_lbfgs and lbfgs_key in logs_data:
            lbfgs_data = logs_data[lbfgs_key]
            plt.semilogy(lbfgs_iters, lbfgs_data, color=color, linewidth=1.5, linestyle='-')
            plt.plot(
                [adam_epochs[-1], lbfgs_iters[0]],
                [adam_data[-1], lbfgs_data[0]],
                color=color,
                linestyle='--',
                alpha=0.5
            )

plot_combined_component('loss_geo_history', 'lbfgs_loss_geo_history', "#8074C8", 'Geo Loss ($du/dx = \\phi$)')
plot_combined_component('loss_const_history', 'lbfgs_loss_const_history', "#7895C1", 'Const Loss ($M = EI \\cdot d\\phi/dx$)')
plot_combined_component('loss_eq1_history', 'lbfgs_loss_eq1_history', "#59E347", 'Eq1 Loss ($Q = dM/dx$)')
plot_combined_component('loss_eq2_history', 'lbfgs_loss_eq2_history', "#EF8B67", 'Eq2 Loss ($dQ/dx = q$)')
if has_lbfgs:
    plt.axvline(x=len(adam_epochs), color='gray', linestyle=':', linewidth=2, label='Optimizer Switch')
plt.xlabel('Optimization Steps (Epochs / L-BFGS Iters)', fontsize=16)
plt.ylabel('Loss (Log Scale)', fontsize=16)
plt.title('Individual Physics Component Losses (Adam + L-BFGS Phase)', fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
adam_epochs = np.arange(len(loss_history))
plt.semilogy(adam_epochs, loss_history, color="#C0321A", linewidth=1.5, label='Phase 1: Adam Loss')

if 'lbfgs_loss_history' in logs_data and len(logs_data['lbfgs_loss_history']) > 0:
    lbfgs_loss = logs_data['lbfgs_loss_history']
    lbfgs_iters = np.arange(len(loss_history), len(loss_history) + len(lbfgs_loss))
    plt.semilogy(lbfgs_iters, lbfgs_loss, color="#218D42", linewidth=1.5, label='Phase 2: L-BFGS Loss')
    plt.plot(
        [adam_epochs[-1], lbfgs_iters[0]],
        [loss_history[-1], lbfgs_loss[0]],
        color='gray',
        linestyle='--',
        alpha=0.8
    )
    plt.axvline(x=len(loss_history), color='gray', linestyle=':', linewidth=2, label='Optimizer Switch')

plt.xlabel('Optimization Steps (Epochs / L-BFGS Iters)', fontsize=16)
plt.ylabel('Loss (Log Scale)', fontsize=16)
plt.title('Complete Training Convergence Curve (Adam + L-BFGS)', fontsize=16)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.show()

if 'raw_total_loss_history' in logs_data:
    plt.figure(figsize=(10, 6))
    plt.semilogy(logs_data['raw_total_loss_history'], color="#C0321A", linewidth=1.5, label='Raw Total Physics Loss')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss (Log Scale)', fontsize=16)
    plt.title('Training Convergence Curve (Raw Physics Sum)', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='upper right', frameon=False)
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
        u_pred, phi_pred, M_pred, Q_pred = model.predict_with_ansatz(
            test_branch_input, test_x, L=L_val, bc_type='fixed-fixed'
        )
        u_pred_np, phi_pred_np = u_pred.squeeze().numpy(), phi_pred.squeeze().numpy()
        M_pred_np, Q_pred_np = M_pred.squeeze().numpy(), Q_pred.squeeze().numpy()

    _, u_fem, phi_fem, M_fem, Q_fem = solve_beam_fem_fixed_gaussian(P, xp, sigma_val, L_val, EI_val)

    error_u, error_phi = np.abs(u_pred_np - u_fem), np.abs(phi_pred_np - phi_fem)
    error_M, error_Q = np.abs(M_pred_np - M_fem), np.abs(Q_pred_np - Q_fem)

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

plot_configs = [
    (fig_q, ax_q, 'Load Input $q(x)$', 'Gaussian Localized Load Distribution', {'loc': 'center left'}),
    (fig_u, ax_u, 'Deflection $(u)$', 'Deflection Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'lower right'}),
    (fig_phi, ax_phi, 'Rotation $(\\phi)$', 'Rotation Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'lower right'}),
    (fig_M, ax_M, 'Moment $(M)$', 'Moment Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'upper center'}),
    (fig_Q, ax_Q, 'Shear Force $(Q)$', 'Shear Force Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'upper right'})
]

for fig, ax, ylabel, title, legend_kwargs in plot_configs:
    ax.hlines(0, 0, L_val, color='black', linewidth=3, zorder=5)
    plot_fixed_symbols(ax, x_sensors=x_sensors)
    ax.set_xlabel('Beam Position $(x)$', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20, pad=10)

    if ax == ax_M:
        ax.invert_yaxis()

    if ax == ax_q:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            fontsize=16,
            handlelength=2.2,
            labelspacing=0.45,
            borderaxespad=0.0
        )
        fig.subplots_adjust(right=0.76)

    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


    fig.tight_layout()
    fig.show()


handles, labels = ax_u.get_legend_handles_labels()

fig_legend, ax_legend = plt.subplots(figsize=(10, 0.4), dpi=500)
ax_legend.axis('off')
fig_legend.legend(
    handles,
    labels,
    loc='center',
    ncol=4,
    frameon=False,
    fontsize=16,
    handlelength=2.5,
    columnspacing=1.5,
    labelspacing=0.5
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
    plot_fixed_symbols(ax, x_sensors=x_sensors)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(**legend_kwargs, frameon=False)

fig_err.tight_layout()
fig_err.show()


print("\n" + "="*85)
print("🚀 FNO vs high precision Hermite FEM Quantitative evaluation report on structural response depth (High-Order Formulation)")
print("="*85)

global_metrics_summary = {
    "Deflection $(u)$": {"mae": [], "l2_rel": [], "r2": []},
    "Rotation $(\\phi)$": {"mae": [], "l2_rel": [], "r2": []},
    "Moment $(M)$": {"mae": [], "l2_rel": [], "r2": []},
    "Shear Force $(Q)$": {"mae": [], "l2_rel": [], "r2": []}
}

for i, (P, xp) in enumerate(test_loads):
    q_sensor_data = P * np.exp(-((x_sensors - xp)**2) / (2 * sigma_val**2))
    test_branch_input = torch.cat([
        torch.tensor([q_sensor_data], dtype=torch.float64),
        torch.tensor([[EI_val, L_val]], dtype=torch.float64)
    ], dim=1)

    with torch.no_grad():
        u_pred, phi_pred, M_pred, Q_pred = model.predict_with_ansatz(
            test_branch_input, test_x, L=L_val, bc_type='fixed-fixed'
        )
        u_pred_np, phi_pred_np = u_pred.squeeze().numpy(), phi_pred.squeeze().numpy()
        M_pred_np, Q_pred_np = M_pred.squeeze().numpy(), Q_pred.squeeze().numpy()

    _, u_fem, phi_fem, M_fem, Q_fem = solve_beam_fem_fixed_gaussian(P, xp, sigma_val, L_val, EI_val)

    def evaluate_metrics(name, pred, fem):
        abs_err_arr = np.abs(pred - fem)
        peak_fem_abs, peak_pred_abs = np.max(np.abs(fem)), np.max(np.abs(pred))
        idx_max_err = np.argmax(abs_err_arr)
        max_abs_err = abs_err_arr[idx_max_err]
        rel_err_at_max = (max_abs_err / peak_fem_abs * 100) if peak_fem_abs > 1e-12 else 0.0
        abs_diff_peaks = np.abs(peak_pred_abs - peak_fem_abs)
        rel_diff_peaks = (abs_diff_peaks / peak_fem_abs * 100) if peak_fem_abs > 1e-12 else 0.0

        mae_val = np.mean(abs_err_arr)
        l2_rel_val = np.linalg.norm(pred - fem) / (np.linalg.norm(fem) + 1e-12)
        ss_res = np.sum((fem - pred) ** 2)
        ss_tot = np.sum((fem - np.mean(fem)) ** 2) + 1e-12
        r2_val = 1 - (ss_res / ss_tot)

        return {
            'name': name,
            'max_err': max_abs_err,
            'pred_at_err': pred[idx_max_err],
            'fem_at_err': fem[idx_max_err],
            'rel_err': rel_err_at_max,
            'peak_pred': peak_pred_abs,
            'peak_fem': peak_fem_abs,
            'peak_diff': abs_diff_peaks,
            'peak_rel': rel_diff_peaks,
            'mae': mae_val,
            'l2_rel': l2_rel_val,
            'r2': r2_val
        }

    metrics = [
        evaluate_metrics("Deflection $(u)$", u_pred_np, u_fem),
        evaluate_metrics("Rotation $(\\phi)$", phi_pred_np, phi_fem),
        evaluate_metrics("Moment $(M)$", M_pred_np, M_fem),
        evaluate_metrics("Shear Force $(Q)$", Q_pred_np, Q_fem)
    ]

    for m in metrics:
        global_metrics_summary[m['name']]["mae"].append(m['mae'])
        global_metrics_summary[m['name']]["l2_rel"].append(m['l2_rel'])
        global_metrics_summary[m['name']]["r2"].append(m['r2'])

    print(f"\n Condition {i+1}: local load peak P = {P}, center position xp={xp}, sigma={sigma_val}")
    print("=" * 85)
    print(f"【Table 1】Spatial Maximum Deviation Location Analysis (Finding the Point with the Worst Global Fitting)")
    print(f"{'Physical Quantity':^9} | {'Maximum Absolute Error':^15} | {'Predicted Value':^15} | {'FEM Value':^15} | {'Relative Peak Error':^15}")
    print("-" * 85)
    for m in metrics:
        print(f"{m['name']:^10} | {m['max_err']:>15.4e} | {m['pred_at_err']:>15.4e} | {m['fem_at_err']:>15.4e} | {m['rel_err']:>12.4f} %")

    print("-" * 85)
    print(f"【Table 2】Global Peak Response Evaluation (Critical Envelope Values of Engineering Design)")
    print(f"{'Physical Quantity':^9} | {'Maximum Predicted Value (Absolute)':^13} | {'Maximum FEM Value (Absolute)':^13} | {'Peak Absolute Deviation':^15} | {'Peak Relative Deviation':^15}")
    print("-" * 85)
    for m in metrics:
        print(f"{m['name']:^10} | {m['peak_pred']:>15.4e} | {m['peak_fem']:>15.4e} | {m['peak_diff']:>15.4e} | {m['peak_rel']:>12.4f} %")

    print("-" * 85)
    print(f"【Table 3】Statistical Comprehensive Accuracy Metrics (MAE, L2 Relative Residual, R² Correlation Coefficient)")
    print(f"{'Physical Quantity':^9} | {'Mean Absolute Error (MAE)':^18} | {'L2 Relative Residual':^15} | {'R² Correlation Coefficient':^15}")
    print("-" * 85)
    for m in metrics:
        print(f"{m['name']:^10} | {m['mae']:>18.4e} | {m['l2_rel']:>15.4e} | {m['r2']:>15.6f}")
    print("=" * 85)

print("\n\n" + "-"*85)
print(" Summary of global comprehensive accuracy evaluation (average of all test conditions")
print("-"*85)
print(f"{'Physical Quantity':^9} | {'Mean MAE':^16} | {'Mean L2 Relative Residual':^20} | {'Mean R²':^15}")
print("-" * 85)
for name, data_dict in global_metrics_summary.items():
    mean_mae = np.mean(data_dict["mae"])
    mean_l2 = np.mean(data_dict["l2_rel"])
    mean_r2 = np.mean(data_dict["r2"])
    print(f"{name:^10} | {mean_mae:>16.4e} | {mean_l2:>20.4e} | {mean_r2:>15.6f}")
print("-" * 85 + "\n")


def get_last_value(logs, key):
    values = logs.get(key, [])
    if values is None or len(values) == 0: return np.nan
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
