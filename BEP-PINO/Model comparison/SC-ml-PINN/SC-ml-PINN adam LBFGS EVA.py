# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import logging
tf.get_logger().setLevel(logging.ERROR)


def gaussian_concentrated_load(x, P=-2.0, xp=0.5, sigma=0.04):
    return P * np.exp(-((x - xp) ** 2) / (2.0 * sigma ** 2))

def solve_beam_fem_fixed_gaussian_pinn_sign(P, xp, sigma=0.04, L=1.0, EI=1.0, num_nodes=1000, gauss_order=8):

    n_elem = num_nodes - 1
    total_dof = num_nodes * 2
    x_ref = np.linspace(0.0, L, num_nodes)
    Le = L / n_elem

    K = np.zeros((total_dof, total_dof), dtype=np.float64)
    F = np.zeros(total_dof, dtype=np.float64)
    
    ke = (EI / Le ** 3) * np.array([
        [12,       6 * Le,   -12,       6 * Le],
        [6 * Le,   4 * Le**2, -6 * Le,   2 * Le**2],
        [-12,     -6 * Le,    12,      -6 * Le],
        [6 * Le,   2 * Le**2, -6 * Le,   4 * Le**2]
    ])

    element_loads = []
    xi, wi = np.polynomial.legendre.leggauss(gauss_order)
    for e in range(n_elem):
        x_left = x_ref[e]
        fe = np.zeros(4, dtype=np.float64)
        for xi_i, wi_i in zip(xi, wi):
            s = 0.5 * Le * (xi_i + 1.0)
            x_g = x_left + s
            q_g = gaussian_concentrated_load(x_g, P=P, xp=xp, sigma=sigma)
            r = s / Le
            N = np.array([
                1 - 3 * r ** 2 + 2 * r ** 3,
                s * (1 - r) ** 2,
                3 * r ** 2 - 2 * r ** 3,
                s * (r ** 2 - r)
            ])
            fe += wi_i * N * q_g * (Le / 2.0)
        
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

    k_ref = M_ref / EI

    return x_ref, u_ref, phi_ref, k_ref, M_ref, Q_ref


def get_highres_fem_on_target_grid(P, xp, x_target, sigma=0.04, L=1.0, EI=1.0):
    x_target = np.asarray(x_target, dtype=np.float64).reshape(-1)
    x_ref, u_ref, phi_ref, k_ref, M_ref, Q_ref = solve_beam_fem_fixed_gaussian_pinn_sign(
        P=P, xp=xp, sigma=sigma, L=L, EI=EI, num_nodes=2000, gauss_order=8
    )

    u_fem = np.interp(x_target, x_ref, u_ref)
    phi_fem = np.interp(x_target, x_ref, phi_ref)
    k_fem = np.interp(x_target, x_ref, k_ref)
    M_fem = np.interp(x_target, x_ref, M_ref)
    Q_fem = np.interp(x_target, x_ref, Q_ref)

    return u_fem, phi_fem, k_fem, M_fem, Q_fem



class PINN_model:
    def __init__(self, layers, X, q, X_uc, u_c, X_ac, a_c, X_kc, k_c):
        self.max_X, self.min_X = X.max(0), X.min(0)
        self.layers_u = layers[0]
        self.layers_a = layers[1]
        self.layers_k = layers[2]
        self.layers_Q = layers[3]

        self.X = X.astype(np.float32)
        self.q = q.astype(np.float32)
        self.X_uc, self.u_c = X_uc.astype(np.float32), u_c.astype(np.float32)
        self.X_ac, self.a_c = X_ac.astype(np.float32), a_c.astype(np.float32)
        self.X_kc, self.k_c = X_kc.astype(np.float32), k_c.astype(np.float32)

        self.EI = np.float32(1.0)

        self.weights_u, self.biases_u = self.initialize_NN(self.layers_u)
        self.weights_a, self.biases_a = self.initialize_NN(self.layers_a)
        self.weights_k, self.biases_k = self.initialize_NN(self.layers_k)
        self.weights_Q, self.biases_Q = self.initialize_NN(self.layers_Q)

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.x_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        
        self.u_pred = self.net_u(self.x_tf)
        self.a_pred = self.net_a(self.x_tf)
        self.k_pred = self.net_k(self.x_tf)
        self.Q_pred = self.net_Q(self.x_tf)
        self.M_pred = self.EI * self.k_pred

        self.model_vars = []
        for var_list in [self.weights_u, self.biases_u, self.weights_a, self.biases_a,
                         self.weights_k, self.biases_k, self.weights_Q, self.biases_Q]:
            self.model_vars.extend(var_list)
        
        self.global_step = tf.Variable(0, trainable=False)
        self.model_vars.append(self.global_step)
        
        self.saver = tf.train.Saver(var_list=self.model_vars, max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())

    def xavier_init(self, size):
        in_dim, out_dim = size[0], size[1]
        xavier_stddev = 1.0 / np.sqrt((in_dim + out_dim) / 2.0)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)

    def initialize_NN(self, layers):
        weights, biases = [], []
        for l in range(0, len(layers) - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W); biases.append(b)
        return weights, biases

    def forward_pass(self, H, weights, biases, layers):
        H = 2.0 * (H - self.min_X) / (self.max_X - self.min_X) - 1.0
        for l in range(0, len(layers) - 2):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        H = tf.add(tf.matmul(H, weights[-1]), biases[-1])
        return H

    def net_u(self, x): return self.forward_pass(tf.concat([x], 1), self.weights_u, self.biases_u, self.layers_u)
    def net_a(self, x): return self.forward_pass(tf.concat([x], 1), self.weights_a, self.biases_a, self.layers_a)
    def net_k(self, x): return self.forward_pass(tf.concat([x], 1), self.weights_k, self.biases_k, self.layers_k)
    def net_Q(self, x): return self.forward_pass(tf.concat([x], 1), self.weights_Q, self.biases_Q, self.layers_Q)

    def predict_u(self, x): return self.sess.run(self.u_pred, {self.x_tf: x.astype(np.float32)})
    def predict_a(self, x): return self.sess.run(self.a_pred, {self.x_tf: x.astype(np.float32)})
    def predict_k(self, x): return self.sess.run(self.k_pred, {self.x_tf: x.astype(np.float32)})
    def predict_M(self, x): return self.sess.run(self.M_pred, {self.x_tf: x.astype(np.float32)})
    def predict_Q(self, x): return self.sess.run(self.Q_pred, {self.x_tf: x.astype(np.float32)})



if __name__ == '__main__':
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
        'figure.dpi': 500, 'savefig.dpi': 500,
        'font.size': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16    
    })

    L_val, EI_val = 1.0, 1.0
    sigma_val = 0.04
    test_loads = [(-3.0, 0.3), (-2.0, 0.5), (-2.5, 0.7)]
    layers = [[1] + 3 * [10] + [1] for _ in range(4)]
    colors = ['#4583b6', '#B02425', '#218D42'] 


    X_fake = np.linspace(0.0, L_val, 2).reshape(-1, 1).astype(np.float32)
    q_fake = np.zeros_like(X_fake)
    

    plt.figure(figsize=(10, 6))
    max_len_adam = 0
    max_len_lbfgs = 0
    
    for i, (P_val, xp_val) in enumerate(test_loads):
        logs = np.load(f'SC-ml-PINN_Combined_P{P_val}_xp{xp_val}_logs.npz')
        lbl = f'$P$={P_val}, $x_p$={xp_val}'
        
        adam_loss = logs['loss_log']
        lbfgs_loss = logs.get('loss_lbfgs_log', np.array([]))
        
        len_adam = len(adam_loss)
        len_lbfgs = len(lbfgs_loss)
        max_len_adam = max(max_len_adam, len_adam)
        max_len_lbfgs = max(max_len_lbfgs, len_lbfgs)
        
        x_adam = np.arange(len_adam)
        x_lbfgs = np.arange(len_adam, len_adam + len_lbfgs)
        
        plt.semilogy(x_adam, adam_loss, color=colors[i], linewidth=1.5, linestyle='-', label=f'Adam ({lbl})')
        if len_lbfgs > 0:
            plt.semilogy(x_lbfgs, lbfgs_loss, color=colors[i], linewidth=2.0, linestyle='--', label=f'L-BFGS ({lbl})')
    
    if max_len_lbfgs > 0:
        plt.axvline(x=max_len_adam, color='gray', linestyle=':', linewidth=1.5, label='Optimizer Switch')
        ax = plt.gca()
        trans = ax.get_xaxis_transform()
        plt.text(max_len_adam * 0.5, 0.1, 'Global Search\n(Adam)', horizontalalignment='center', color="black", fontsize=14, alpha=0.8, transform=trans)   
        plt.text(max_len_adam + max_len_lbfgs * 0.5, 0.1, 'Fine-tuning\n(L-BFGS-B)', horizontalalignment='center', color="black", fontsize=14, alpha=0.8, transform=trans)
    
    plt.xlabel('Optimization Steps (Adam Epochs + L-BFGS Evaluations)', fontsize=16)
    plt.ylabel('Total PINN Loss (Log Scale)', fontsize=16)
    plt.title('Complete Training Convergence: Adam $\\rightarrow$ L-BFGS-B', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='center right', fontsize=12)
    plt.tight_layout()
    plt.show()

    fig_q, ax_q = plt.subplots(figsize=(10, 3))
    fig_u, ax_u = plt.subplots(figsize=(5, 3))
    fig_phi, ax_phi = plt.subplots(figsize=(5, 3))
    fig_k, ax_k = plt.subplots(figsize=(5, 3))
    fig_M, ax_M = plt.subplots(figsize=(5, 3))
    fig_Q, ax_Q = plt.subplots(figsize=(5, 3))
    fig_err, ax_err_matrix = plt.subplots(3, 2, figsize=(16, 18), dpi=500)
    ax_err = ax_err_matrix.flatten() 
    
    global_metrics_summary = {
        "Deflection (u)": {"mae": [], "l2_rel": [], "r2": []},
        "Rotation (phi)": {"mae": [], "l2_rel": [], "r2": []},
        "Curvature (k)": {"mae": [], "l2_rel": [], "r2": []},
        "Moment (M)": {"mae": [], "l2_rel": [], "r2": []},
        "Shear Force (Q)": {"mae": [], "l2_rel": [], "r2": []}
    }

    x_np = np.linspace(0.0, L_val, 100)
    x_star = x_np.reshape(-1, 1).astype(np.float32)

    for i, (P_val, xp_val) in enumerate(test_loads):
        tf.reset_default_graph()
        save_prefix = f'SC-ml-PINN_Combined_P{P_val}_xp{xp_val}'
        
        logs = np.load(save_prefix + '_logs.npz')
        X_train_current = logs['X_train']

        q_plot_continuous = gaussian_concentrated_load(x_np, P_val, xp_val, sigma_val)
        lbl = f'$P$={P_val}, $x_p$={xp_val}'
        ax_q.plot(x_np, q_plot_continuous, color=colors[i], label=f'$q(x)$ ({lbl})', linewidth=3, alpha=0.8)
        ax_q.fill_between(x_np, q_plot_continuous, color=colors[i], alpha=0.1)
        
        model = PINN_model(layers, X_train_current, q_fake, X_fake, X_fake, X_fake, X_fake, X_fake, X_fake)
        model.saver.restore(model.sess, save_prefix + '.ckpt')
        
        u_pred_np = model.predict_u(x_star).reshape(-1)
        phi_pred_np = model.predict_a(x_star).reshape(-1)
        k_pred_np = model.predict_k(x_star).reshape(-1)
        M_pred_np = model.predict_M(x_star).reshape(-1)
        Q_pred_np = model.predict_Q(x_star).reshape(-1)

        u_fem, phi_fem, k_fem, M_fem, Q_fem = get_highres_fem_on_target_grid(
            P=P_val, xp=xp_val, x_target=x_np, sigma=sigma_val, L=L_val, EI=EI_val
        )
        
        error_u, error_phi = np.abs(u_pred_np - u_fem), np.abs(phi_pred_np - phi_fem) 
        error_k = np.abs(k_pred_np - k_fem)
        error_M, error_Q = np.abs(M_pred_np - M_fem), np.abs(Q_pred_np - Q_fem)

        ax_u.plot(x_np, u_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
        ax_u.plot(x_np, u_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')
        ax_phi.plot(x_np, phi_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
        ax_phi.plot(x_np, phi_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')
        ax_k.plot(x_np, k_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
        ax_k.plot(x_np, -k_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')
        ax_M.plot(x_np, M_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
        ax_M.plot(x_np, -M_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')
        ax_Q.plot(x_np, Q_fem, color=colors[i], label=f'FEM ({lbl})', alpha=0.4, linewidth=4)
        ax_Q.plot(x_np, -Q_pred_np, color=colors[i], linestyle='--', label=f'Pred ({lbl})')

        error_data = [error_u, error_phi, error_k, error_M, error_Q]
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
            evaluate_metrics("Curvature (k)", k_pred_np, k_fem),
            evaluate_metrics("Moment (M)", M_pred_np, M_fem),
            evaluate_metrics("Shear Force (Q)", Q_pred_np, Q_fem)
        ]

        for m in metrics:
            global_metrics_summary[m['name']]["mae"].append(m['mae'])
            global_metrics_summary[m['name']]["l2_rel"].append(m['l2_rel'])
            global_metrics_summary[m['name']]["r2"].append(m['r2'])

    def plot_fixed_symbols(ax, color='black'):
        ax.plot(0, 0, 's', markersize=10, color=color, zorder=10) 
        ax.plot(L_val, 0, 's', markersize=10, color=color, zorder=10)

    plot_configs = [
        (fig_q, ax_q, 'Load Input $q(x)$', 'Gaussian Localized Load Distribution', {'loc': 'center left'}), 
        (fig_u, ax_u, 'Deflection $(u)$', 'Deflection Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'lower right'}),
        (fig_phi, ax_phi, 'Rotation $(\\phi)$', 'Rotation Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'lower right'}),
        (fig_k, ax_k, 'Curvature $(k)$', 'Curvature Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'lower center'}),
        (fig_M, ax_M, 'Moment $(M)$', 'Moment Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'upper center'}),
        (fig_Q, ax_Q, 'Shear Force $(Q)$', 'Shear Force Comparison: Pred vs FEM (Fixed-Fixed)', {'loc': 'upper right'})
    ]

    for fig, ax, ylabel, title, legend_kwargs in plot_configs:
        ax.hlines(0, 0, L_val, color='black', linewidth=3, zorder=5)
        plot_fixed_symbols(ax)
        ax.set_xlabel('Beam Position $(x)$', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=20, pad=10)

        if ax == ax_M or ax == ax_k:
            ax.invert_yaxis()

        if ax == ax_q:
            ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False, 
                      fontsize=16, handlelength=2.2, labelspacing=0.45, borderaxespad=0.0)
            fig.subplots_adjust(right=0.76)
        else:
            leg = ax.get_legend()
            if leg is not None: leg.remove()

        fig.tight_layout()
        fig.show()

    handles, labels = ax_u.get_legend_handles_labels()
    fig_legend, ax_legend = plt.subplots(figsize=(10, 0.4), dpi=500)
    ax_legend.axis('off')
    fig_legend.legend(handles, labels, loc='center', ncol=4, frameon=False, 
                      fontsize=16, handlelength=2.5, columnspacing=1.5, labelspacing=0.5)
    fig_legend.tight_layout()
    fig_legend.show()

    err_configs = [
        ('Absolute Error: Deflection $(u)$', {'loc': 'upper center'}), 
        ('Absolute Error: Rotation $(\\phi)$', {'loc': 'upper center'}), 
        ('Absolute Error: Curvature $(k)$', {'loc': 'upper center'}),
        ('Absolute Error: Moment $(M)$', {'loc': 'upper right'}), 
        ('Absolute Error: Shear Force $(Q)$', {'loc': 'upper left'})
    ]
    for j, ax in enumerate(ax_err[:5]):
        title, legend_kwargs = err_configs[j]
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Beam Position $(x)$', fontsize=16)
        ax.set_ylabel('Error Magnitude', fontsize=16)
        ax.hlines(0, 0, L_val, color='black', linewidth=3, zorder=5)
        plot_fixed_symbols(ax)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(**legend_kwargs)
    
    ax_err[-1].axis('off')  
    fig_err.tight_layout()
    fig_err.show()

    print("\n" + "="*85)
    print(" SC-ml-PINN vs High-Res Hermite FEM Quantitative evaluation report on structural response depth")
    print("="*85)

    for i, (P_val, xp_val) in enumerate(test_loads):
        tf.reset_default_graph()
        save_prefix = f'SC-ml-PINN_Combined_P{P_val}_xp{xp_val}'
        
        logs = np.load(save_prefix + '_logs.npz')
        X_train_current = logs['X_train']
        
        model = PINN_model(layers, X_train_current, q_fake, X_fake, X_fake, X_fake, X_fake, X_fake, X_fake)
        model.saver.restore(model.sess, save_prefix + '.ckpt')

        u_pred_np = model.predict_u(x_star).reshape(-1)
        phi_pred_np = model.predict_a(x_star).reshape(-1)
        k_pred_np = model.predict_k(x_star).reshape(-1)
        M_pred_np = model.predict_M(x_star).reshape(-1)
        Q_pred_np = model.predict_Q(x_star).reshape(-1)

        u_fem, phi_fem, k_fem, M_fem, Q_fem = get_highres_fem_on_target_grid(
            P=P_val, xp=xp_val, x_target=x_np, sigma=sigma_val, L=L_val, EI=EI_val
        )

        def evaluate_metrics_details(name, pred, fem):
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

        metrics_detailed = [
            evaluate_metrics_details("Deflection (u)", u_pred_np, u_fem),
            evaluate_metrics_details("Rotation (phi)", phi_pred_np, phi_fem),
            evaluate_metrics_details("Curvature (k)", k_pred_np, k_fem),
            evaluate_metrics_details("Moment (M)", M_pred_np, M_fem),
            evaluate_metrics_details("Shear Force (Q)", Q_pred_np, Q_fem)
        ]

        print(f"\ncondition {i+1}: local load peak P = {P_val}, center position xp = {xp_val}, sigma = {sigma_val}")
        print("=" * 85)
        print(f"【Table 1】Analysis of spatial maximum deviation location (find the point with the worst global fitting)")
        print(f"{'Physical Quantity':^9} | {'Maximum Absolute Error':^15} | {'Predicted Value':^15} | {'FEM Value':^15} | {'Relative Peak Error':^15}")
        print("-" * 85)
        for m in metrics_detailed: print(f"{m['name']:^15} | {m['max_err']:>15.4e} | {m['pred_at_err']:>15.4e} | {m['fem_at_err']:>15.4e} | {m['rel_err']:>12.4f} %")
        
        print("-" * 85)
        print(f"【Table 2】Global peak response evaluation ")
        print(f"{'Physical Quantity':^9} | {'Maximum Predicted Value (Absolute)':^13} | {'Maximum FEM Value (Absolute)':^13} | {'Peak Absolute Deviation':^15} | {'Peak Relative Deviation':^15}")
        print("-" * 85)
        for m in metrics_detailed: print(f"{m['name']:^15} | {m['peak_pred']:>15.4e} | {m['peak_fem']:>15.4e} | {m['peak_diff']:>15.4e} | {m['peak_rel']:>12.4f} %")
        print("=" * 85)

    print("\n\n" + "-"*85)
    print(" Summary of global comprehensive accuracy evaluation (average of all test conditions)")
    print("-"*85)
    print(f"{'Physical Quantity':^15} | {'Mean MAE':^16} | {'Mean Relative L²':^20} | {'Mean R²':^15}")
    print("-" * 85)
    for name, data_dict in global_metrics_summary.items():
        mean_mae = np.mean(data_dict["mae"])
        mean_l2 = np.mean(data_dict["l2_rel"])
        mean_r2 = np.mean(data_dict["r2"])
        print(f"{name:^15} | {mean_mae:>16.4e} | {mean_l2:>20.4e} | {mean_r2:>15.6f}")
    print("-" * 85 + "\n")


    print("\n" + "=" * 85)
    print(" Final Training Loss Summary (PINN Adam + L-BFGS-B)")
    print("=" * 85)
    print(f"{'Condition':^22} | {'Adam Final Loss':^18} | {'L-BFGS Final Loss':^18}")
    print("-" * 85)
    for i, (P_val, xp_val) in enumerate(test_loads):
        logs = np.load(f'SC-ml-PINN_Combined_P{P_val}_xp{xp_val}_logs.npz')
        adam_final_loss = logs['loss_log'][-1]
        
        lbfgs_log = logs.get('loss_lbfgs_log', np.array([]))
        lbfgs_final_loss = lbfgs_log[-1] if len(lbfgs_log) > 0 else np.nan
        
        cond_str = f"P={P_val}, xp={xp_val}"
        print(f"{cond_str:^22} | {adam_final_loss:>18.6e} | {lbfgs_final_loss:>18.6e}")
    print("=" * 85)