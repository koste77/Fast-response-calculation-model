
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.optimize
try:
    from scipy.stats import qmc
except Exception:
    qmc = None


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import logging
tf.get_logger().setLevel(logging.ERROR)


class CustomScipyOptimizer:
    def __init__(self, loss, method='L-BFGS-B', options=None):
        self.loss = loss
        self.method = method
        self.options = options
        
        self.var_list = tf.trainable_variables()
        self.grads = tf.gradients(self.loss, self.var_list)
        
        self.assign_ops = []
        self.pl_ops = []
        for v in self.var_list:
            pl = tf.placeholder(dtype=v.dtype.base_dtype, shape=v.shape)
            self.pl_ops.append(pl)
            self.assign_ops.append(tf.assign(v, pl))

    def minimize(self, sess, feed_dict, fetches=None, loss_callback=None):
        self.sess = sess
        self.feed_dict = feed_dict
        self.loss_callback = loss_callback
        
        init_weights = self.get_weights()
        
        scipy.optimize.minimize(
            fun=self.loss_grad, 
            x0=init_weights, 
            method=self.method, 
            jac=True, 
            options=self.options,
            callback=self.callback_wrapper
        )

    def get_weights(self):
        weights = self.sess.run(self.var_list)
        return np.concatenate([w.flatten() for w in weights])

    def set_weights(self, flat_weights):
        idx = 0
        feed_dict = {}
        for v, pl in zip(self.var_list, self.pl_ops):
            shape = v.shape.as_list()
            v_size = np.prod(shape)
            feed_dict[pl] = flat_weights[idx:idx+v_size].reshape(shape)
            idx += v_size
        self.sess.run(self.assign_ops, feed_dict=feed_dict)

    def loss_grad(self, w):
        self.set_weights(w)
        loss_val, grad_vals = self.sess.run([self.loss, self.grads], feed_dict=self.feed_dict)
        flat_grad = np.concatenate([g.flatten() for g in grad_vals])
        return loss_val.astype(np.float64), flat_grad.astype(np.float64)

    def callback_wrapper(self, w):
        if self.loss_callback is not None:
            loss_val = self.sess.run(self.loss, feed_dict=self.feed_dict)
            self.loss_callback(loss_val)


def gaussian_concentrated_load(x, P, xp, sigma=0.04):
    x = np.asarray(x, dtype=np.float32)
    return P * np.exp(-((x - xp) ** 2) / (2.0 * sigma ** 2))


def lhs_load_cases(n_cases, P_min=-5.0, P_max=-0.5, xp_min=0.2, xp_max=0.8, seed=1234):
    if qmc is not None:
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        samples = sampler.random(n=n_cases)
    else:
        rng = np.random.RandomState(seed)
        samples = np.zeros((n_cases, 2), dtype=np.float32)
        for j in range(2):
            perm = rng.permutation(n_cases)
            samples[:, j] = (perm + rng.rand(n_cases)) / n_cases

    P = P_min + samples[:, 0:1] * (P_max - P_min)
    xp = xp_min + samples[:, 1:2] * (xp_max - xp_min)
    return P.astype(np.float32), xp.astype(np.float32)


def build_parametric_collocation(P_cases, xp_cases, L=1.0, sigma=0.04,
                                 n_x_global=121, n_x_local=41, local_width=5.0):
    X_all, q_all = [], []
    x_global = np.linspace(0.0, L, n_x_global).reshape(-1, 1).astype(np.float32)

    for P_i, xp_i in zip(P_cases.reshape(-1), xp_cases.reshape(-1)):
        left = max(0.0, float(xp_i) - local_width * sigma)
        right = min(L, float(xp_i) + local_width * sigma)
        x_local = np.linspace(left, right, n_x_local).reshape(-1, 1).astype(np.float32)
        x_extra = np.array([[0.0], [L], [float(xp_i)]], dtype=np.float32)
        x_i = np.unique(np.vstack([x_global, x_local, x_extra]).reshape(-1)).reshape(-1, 1).astype(np.float32)
        x_i = np.sort(x_i, axis=0)

        P_col = np.full_like(x_i, P_i, dtype=np.float32)
        xp_col = np.full_like(x_i, xp_i, dtype=np.float32)
        X_i = np.hstack([x_i, P_col, xp_col]).astype(np.float32)
        q_i = gaussian_concentrated_load(x_i, P=P_i, xp=xp_i, sigma=sigma).astype(np.float32)

        X_all.append(X_i)
        q_all.append(q_i)

    return np.vstack(X_all).astype(np.float32), np.vstack(q_all).astype(np.float32)


def build_parametric_boundary(P_cases, xp_cases, L=1.0):
    X_b = []
    for P_i, xp_i in zip(P_cases.reshape(-1), xp_cases.reshape(-1)):
        X_b.append([0.0, float(P_i), float(xp_i)])
        X_b.append([L, float(P_i), float(xp_i)])
    X_b = np.array(X_b, dtype=np.float32)
    y_zero = np.zeros((X_b.shape[0], 1), dtype=np.float32)
    return X_b, y_zero



class PINN_model:
    def __init__(self, layers, X, q, X_uc, u_c, X_ac, a_c, X_kc, k_c,
                 EI=1.0, L=1.0,
                 P_min=-5.0, P_max=-0.5, xp_min=0.2, xp_max=0.8):

        self.max_X = np.array([[L, P_max, xp_max]], dtype=np.float32)
        self.min_X = np.array([[0.0, P_min, xp_min]], dtype=np.float32)

        self.layers_u = layers[0]
        self.layers_a = layers[1]
        self.layers_k = layers[2]
        self.layers_Q = layers[3]

        self.X = X.astype(np.float32)
        self.q = q.astype(np.float32)
        self.X_uc, self.u_c = X_uc.astype(np.float32), u_c.astype(np.float32)
        self.X_ac, self.a_c = X_ac.astype(np.float32), a_c.astype(np.float32)
        self.X_kc, self.k_c = X_kc.astype(np.float32), k_c.astype(np.float32)

        self.EI = np.float32(EI)
        self.L = np.float32(L)

        self.weights_u, self.biases_u = self.initialize_NN(self.layers_u, name_prefix='u')
        self.weights_a, self.biases_a = self.initialize_NN(self.layers_a, name_prefix='a')
        self.weights_k, self.biases_k = self.initialize_NN(self.layers_k, name_prefix='k')
        self.weights_Q, self.biases_Q = self.initialize_NN(self.layers_Q, name_prefix='Q')

        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

        self.X_tf = tf.placeholder(dtype=tf.float32, shape=[None, 3])
        self.q_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.X_u_tf = tf.placeholder(dtype=tf.float32, shape=[None, 3])
        self.X_a_tf = tf.placeholder(dtype=tf.float32, shape=[None, 3])
        self.X_k_tf = tf.placeholder(dtype=tf.float32, shape=[None, 3])

        self.u_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.a_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.k_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.u_pred = self.net_u(self.X_tf)
        self.a_pred = self.net_a(self.X_tf)
        self.k_pred = self.net_k(self.X_tf)
        self.Q_pred = self.net_Q(self.X_tf)
        self.M_pred = self.EI * self.k_pred

        self.u_c_pred = self.net_u(self.X_u_tf)
        self.a_c_pred = self.net_a(self.X_a_tf)
        self.k_c_pred = self.net_k(self.X_k_tf)

        self.f_Q_q_pred, self.f_M_Q_pred, self.f_a_k_pred, self.f_u_a_pred = self.net_f(self.X_tf)

        self.loss_c = tf.reduce_mean(tf.square(self.u_c_pred - self.u_c_tf)) \
            + tf.reduce_mean(tf.square(self.a_c_pred - self.a_c_tf))

        self.loss_Q_q = tf.reduce_mean(tf.square(self.f_Q_q_pred))
        self.loss_M_Q = tf.reduce_mean(tf.square(self.f_M_Q_pred))
        self.loss_a_k = tf.reduce_mean(tf.square(self.f_a_k_pred))
        self.loss_u_a = tf.reduce_mean(tf.square(self.f_u_a_pred))
        self.loss_f = self.loss_Q_q + self.loss_M_Q + self.loss_a_k + self.loss_u_a
        self.loss = self.loss_c + self.loss_f

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        starter_learning_rate = 1e-3
        self.learning_rate = tf.train.exponential_decay(
            starter_learning_rate, self.global_step, 1000, 0.9, staircase=False
        )
        
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss, global_step=self.global_step
        )
        
        self.optimizer_lbfgs = CustomScipyOptimizer(
            self.loss,
            method='L-BFGS-B',
            options={'maxiter': 50000,
                     'maxfun': 50000,
                     'maxcor': 50,
                     'maxls': 50,
                     'ftol': 1.0 * np.finfo(float).eps,
                     'gtol': 1e-12}  

        self.loss_log = []
        self.loss_lbfgs_log = [] 
        self.loss_c_log = []
        self.loss_f_log = []
        self.loss_Q_q_log = []
        self.loss_M_Q_log = []
        self.loss_a_k_log = []
        self.loss_u_a_log = []
        self.lr_log = []

        self.model_vars = []
        for var_list in [self.weights_u, self.biases_u,
                         self.weights_a, self.biases_a,
                         self.weights_k, self.biases_k,
                         self.weights_Q, self.biases_Q]:
            self.model_vars.extend(var_list)
        self.model_vars.append(self.global_step)
        self.saver = tf.train.Saver(var_list=self.model_vars, max_to_keep=1)

        self.sess.run(tf.global_variables_initializer())

    def xavier_init(self, size, name=None):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1.0 / np.sqrt((in_dim + out_dim) / 2.0)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32, name=name)

    def initialize_NN(self, layers, name_prefix='net'):
        weights = []
        biases = []
        for l in range(0, len(layers) - 1):
            W = self.xavier_init([layers[l], layers[l + 1]], name=f'{name_prefix}_W_{l}')
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32),
                            dtype=tf.float32, name=f'{name_prefix}_b_{l}')
            weights.append(W)
            biases.append(b)
        return weights, biases

    def forward_pass(self, H, weights, biases, layers):
        H = 2.0 * (H - self.min_X) / (self.max_X - self.min_X) - 1.0
        for l in range(0, len(layers) - 2):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        H = tf.add(tf.matmul(H, weights[-1]), biases[-1])
        return H

    def net_u(self, X):
        return self.forward_pass(X, self.weights_u, self.biases_u, self.layers_u)

    def net_a(self, X):
        return self.forward_pass(X, self.weights_a, self.biases_a, self.layers_a)

    def net_k(self, X):
        return self.forward_pass(X, self.weights_k, self.biases_k, self.layers_k)

    def net_Q(self, X):
        return self.forward_pass(X, self.weights_Q, self.biases_Q, self.layers_Q)

    def net_f(self, X):
        Q = self.net_Q(X)
        k = self.net_k(X)
        a = self.net_a(X)
        u = self.net_u(X)
        M = self.EI * k

        grad_Q = tf.gradients(Q, X)[0]
        grad_M = tf.gradients(M, X)[0]
        grad_a = tf.gradients(a, X)[0]
        grad_u = tf.gradients(u, X)[0]

        Q_x = grad_Q[:, 0:1]
        M_x = grad_M[:, 0:1]
        a_x = grad_a[:, 0:1]
        u_x = grad_u[:, 0:1]

        f_Q_q = Q_x + self.q_tf
        f_M_Q = M_x - Q
        f_a_k = a_x + k
        f_u_a = u_x - a
        return f_Q_q, f_M_Q, f_a_k, f_u_a

    def train(self, nIter=20000):
        start_time = timeit.default_timer()
        tf_dict = {
            self.X_u_tf: self.X_uc, self.u_c_tf: self.u_c,
            self.X_a_tf: self.X_ac, self.a_c_tf: self.a_c,
            self.X_k_tf: self.X_kc, self.k_c_tf: self.k_c,
            self.X_tf: self.X,
            self.q_tf: self.q
        }

        for it in range(nIter):
            self.sess.run(self.train_op, tf_dict)

            loss_value, loss_c_value, loss_f_value, loss_Q_q_value, loss_M_Q_value, \
                loss_a_k_value, loss_u_a_value, lr_value = self.sess.run(
                    [self.loss, self.loss_c, self.loss_f,
                     self.loss_Q_q, self.loss_M_Q, self.loss_a_k, self.loss_u_a,
                     self.learning_rate], tf_dict
                )

            self.loss_log.append(loss_value)
            self.loss_c_log.append(loss_c_value)
            self.loss_f_log.append(loss_f_value)
            self.loss_Q_q_log.append(loss_Q_q_value)
            self.loss_M_Q_log.append(loss_M_Q_value)
            self.loss_a_k_log.append(loss_a_k_value)
            self.loss_u_a_log.append(loss_u_a_value)
            self.lr_log.append(lr_value)

            if it % 1000 == 0:
                elapsed = timeit.default_timer() - start_time
                print('Adam It: %d, Loss: %.3e, Loss_c: %.3e, Loss_f: %.3e, Time: %.2f' %
                      (it, loss_value, loss_c_value, loss_f_value, elapsed))
                start_time = timeit.default_timer()

    def callback(self, loss):
        self.loss_lbfgs_log.append(loss)
        if len(self.loss_lbfgs_log) % 100 == 0:
            print('L-BFGS Eval %d, Loss: %e' % (len(self.loss_lbfgs_log), loss))

    def train_p(self):

        print(" L-BFGS-B ...")
        tf_dict = {
            self.X_u_tf: self.X_uc, self.u_c_tf: self.u_c,
            self.X_a_tf: self.X_ac, self.a_c_tf: self.a_c,
            self.X_k_tf: self.X_kc, self.k_c_tf: self.k_c,
            self.X_tf: self.X,
            self.q_tf: self.q
        }
        
        init_loss = self.sess.run(self.loss, feed_dict=tf_dict)
        self.loss_lbfgs_log.append(init_loss)
        
        self.optimizer_lbfgs.minimize(
            self.sess,
            feed_dict=tf_dict,
            fetches=[self.loss],
            loss_callback=self.callback
        )
        print(f"L-BFGS-B  {len(self.loss_lbfgs_log)-1} ")

    def make_input(self, x, P, xp):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        P_col = np.full_like(x, float(P), dtype=np.float32)
        xp_col = np.full_like(x, float(xp), dtype=np.float32)
        return np.hstack([x, P_col, xp_col]).astype(np.float32)

    def predict_u(self, x, P, xp): return self.sess.run(self.u_pred, {self.X_tf: self.make_input(x, P, xp)})
    def predict_a(self, x, P, xp): return self.sess.run(self.a_pred, {self.X_tf: self.make_input(x, P, xp)})
    def predict_k(self, x, P, xp): return self.sess.run(self.k_pred, {self.X_tf: self.make_input(x, P, xp)})
    def predict_Q(self, x, P, xp): return self.sess.run(self.Q_pred, {self.X_tf: self.make_input(x, P, xp)})
    def predict_M(self, x, P, xp): return self.sess.run(self.M_pred, {self.X_tf: self.make_input(x, P, xp)})

# %%
if __name__ == '__main__':
    tf.reset_default_graph()
    np.random.seed(1202)
    tf.set_random_seed(1202)


    L_val = 1.0
    EI_val = 1.0
    sigma_val = 0.04
    bc_type = 'fixed-fixed'

    P_min_val, P_max_val = -5.0, -0.5
    xp_min_val, xp_max_val = 0.2, 0.8

    N_load_cases = 500
    P_cases, xp_cases = lhs_load_cases(
        N_load_cases, P_min_val, P_max_val, xp_min_val, xp_max_val, seed=1202
    )

    layers = [
        [3] + 3 * [10] + [1],  # u
        [3] + 3 * [10] + [1],  # phi / a
        [3] + 3 * [10] + [1],  # k
        [3] + 3 * [10] + [1],  # Q
    ]

    X_train, q_train = build_parametric_collocation(
        P_cases, xp_cases, L=L_val, sigma=sigma_val,
        n_x_global=121, n_x_local=41, local_width=5.0
    )

    X_uc, u_c = build_parametric_boundary(P_cases, xp_cases, L=L_val)
    X_ac, a_c = build_parametric_boundary(P_cases, xp_cases, L=L_val)
    X_kc, k_c = build_parametric_boundary(P_cases, xp_cases, L=L_val)


    plt.figure(figsize=(8, 5), dpi=160)
    plt.scatter(xp_cases.reshape(-1), P_cases.reshape(-1), s=35, alpha=0.8)
    plt.xlabel('Load position xp')
    plt.ylabel('Load amplitude P')
    plt.title('LHS load cases used by parametric PINN')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    x_plot = np.linspace(0, L_val, 1001).reshape(-1, 1).astype(np.float32)
    plt.figure(figsize=(10, 4), dpi=160)
    for i in range(min(8, N_load_cases)):
        label = f'P={P_cases[i, 0]:.2f}, xp={xp_cases[i, 0]:.2f}' if i < 4 else None
        plt.plot(x_plot,
                 gaussian_concentrated_load(x_plot, P_cases[i, 0], xp_cases[i, 0], sigma_val),
                 linewidth=1.4, alpha=0.75, label=label)
    plt.xlabel('Location x')
    plt.ylabel('q(x)')
    plt.title('Examples of Gaussian concentrated loads from LHS range')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()

    print('================  Adam + L-BFGS-B ================')
    print(f'LHS: P=[{P_min_val}, {P_max_val}], xp=[{xp_min_val}, {xp_max_val}], sigma={sigma_val}')
    print(f'N_load_cases={N_load_cases}, N_domain_total={X_train.shape[0]}, N_boundary={X_uc.shape[0]}')

    model = PINN_model(layers, X_train, q_train, X_uc, u_c, X_ac, a_c, X_kc, k_c,
                       EI=EI_val, L=L_val,
                       P_min=P_min_val, P_max=P_max_val,
                       xp_min=xp_min_val, xp_max=xp_max_val)
    
    global_start_time = timeit.default_timer()

    model.train(20000)
    model.train_p()

    global_end_time = timeit.default_timer()
    global_total_time = global_end_time - global_start_time

    save_prefix = 'P-ml-PINN adam LBFGS'
    model.saver.save(model.sess, save_prefix + '.ckpt')

    np.savez(save_prefix + '_logs.npz',
             loss_log=np.array(model.loss_log),
             loss_lbfgs_log=np.array(model.loss_lbfgs_log), 
             loss_c_log=np.array(model.loss_c_log),
             loss_f_log=np.array(model.loss_f_log),
             loss_Q_q_log=np.array(model.loss_Q_q_log),
             loss_M_Q_log=np.array(model.loss_M_Q_log),
             loss_a_k_log=np.array(model.loss_a_k_log),
             loss_u_a_log=np.array(model.loss_u_a_log),
             lr_log=np.array(model.lr_log),
             X_train=X_train,
             q_train=q_train,
             X_uc=X_uc,
             X_ac=X_ac,
             X_kc=X_kc,
             P_cases=P_cases,
             xp_cases=xp_cases,
             P_min_val=P_min_val,
             P_max_val=P_max_val,
             xp_min_val=xp_min_val,
             xp_max_val=xp_max_val,
             sigma_val=sigma_val,
             L_val=L_val,
             EI_val=EI_val,
             N_load_cases=N_load_cases,
             bc_type=bc_type,
             layers=np.array(layers, dtype=object))

    print('=' * 80)
    print('Finish：')
    print(save_prefix + '.ckpt')
    print(save_prefix + '_logs.npz')
    print('=' * 80)
    

    print(f"  {N_load_cases} :")
    print(f"   - Adam Epochs: {len(model.loss_log)}")
    print(f"   - L-BFGS Epochs: {len(model.loss_lbfgs_log) - 1}")
    print(f"   - Times: {global_total_time:.2f} s")