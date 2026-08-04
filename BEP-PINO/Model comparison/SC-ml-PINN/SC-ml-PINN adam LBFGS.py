# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import timeit
import scipy.optimize

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


def gaussian_concentrated_load(x, P=-2.0, xp=0.5, sigma=0.04):
    return P * np.exp(-((x - xp) ** 2) / (2.0 * sigma ** 2))

# ============================================================
# 2. PINN 独立四网络模型 (集成 Adam + 自定义 L-BFGS-B)
# ============================================================
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
        self.q_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_u_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_a_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.x_k_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.u_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.a_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.k_c_tf = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.u_pred = self.net_u(self.x_tf)
        self.a_pred = self.net_a(self.x_tf)
        self.k_pred = self.net_k(self.x_tf)
        self.Q_pred = self.net_Q(self.x_tf)
        self.M_pred = self.EI * self.k_pred

        self.u_c_pred = self.net_u(self.x_u_tf)
        self.a_c_pred = self.net_a(self.x_a_tf)
        self.k_c_pred = self.net_k(self.x_k_tf)

        self.f_Q_q_pred, self.f_M_Q_pred, self.f_a_k_pred, self.f_u_a_pred = self.net_f(self.x_tf)

        self.loss_c = tf.reduce_mean(tf.square(self.u_c_pred - self.u_c_tf)) \
            + tf.reduce_mean(tf.square(self.a_c_pred - self.a_c_tf))

        self.loss_Q_q = tf.reduce_mean(tf.square(self.f_Q_q_pred))
        self.loss_M_Q = tf.reduce_mean(tf.square(self.f_M_Q_pred))
        self.loss_a_k = tf.reduce_mean(tf.square(self.f_a_k_pred))
        self.loss_u_a = tf.reduce_mean(tf.square(self.f_u_a_pred))
        self.loss_f = self.loss_Q_q + self.loss_M_Q + self.loss_a_k + self.loss_u_a

        self.loss = self.loss_c + self.loss_f

        self.global_step = tf.Variable(0, trainable=False)
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
        )

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
        for var_list in [self.weights_u, self.biases_u, self.weights_a, self.biases_a,
                         self.weights_k, self.biases_k, self.weights_Q, self.biases_Q]:
            self.model_vars.extend(var_list)
        self.model_vars.append(self.global_step)
        self.saver = tf.train.Saver(var_list=self.model_vars, max_to_keep=1)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1.0 / np.sqrt((in_dim + out_dim) / 2.0)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        for l in range(0, len(layers) - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def forward_pass(self, H, weights, biases, layers):
        H = 2.0 * (H - self.min_X) / (self.max_X - self.min_X) - 1.0
        for l in range(0, len(layers) - 2):
            H = tf.tanh(tf.add(tf.matmul(H, weights[l]), biases[l]))
        H = tf.add(tf.matmul(H, weights[-1]), biases[-1])
        return H

    def net_u(self, x):
        return self.forward_pass(tf.concat([x], 1), self.weights_u, self.biases_u, self.layers_u)

    def net_a(self, x):
        return self.forward_pass(tf.concat([x], 1), self.weights_a, self.biases_a, self.layers_a)

    def net_k(self, x):
        return self.forward_pass(tf.concat([x], 1), self.weights_k, self.biases_k, self.layers_k)

    def net_Q(self, x):
        return self.forward_pass(tf.concat([x], 1), self.weights_Q, self.biases_Q, self.layers_Q)

    def net_f(self, x):
        Q = self.net_Q(x)
        k = self.net_k(x)
        a = self.net_a(x)
        u = self.net_u(x)

        M = self.EI * k
        Q_x = tf.gradients(Q, x)[0]
        M_x = tf.gradients(M, x)[0]
        a_x = tf.gradients(a, x)[0]
        u_x = tf.gradients(u, x)[0]

        f_Q_q = Q_x + self.q_tf
        f_M_Q = M_x - Q
        f_a_k = a_x + k
        f_u_a = u_x - a
        return f_Q_q, f_M_Q, f_a_k, f_u_a

    def train(self, nIter=30000):
        start_time = timeit.default_timer()
        tf_dict = {
            self.x_u_tf: self.X_uc, self.u_c_tf: self.u_c,
            self.x_a_tf: self.X_ac, self.a_c_tf: self.a_c,
            self.x_k_tf: self.X_kc, self.k_c_tf: self.k_c,
            self.x_tf: self.X, self.q_tf: self.q
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
            
            if it % 5000 == 0:
                elapsed = timeit.default_timer() - start_time
                print(f'Adam It: {it}, Loss: {loss_value:.3e}, lr: {lr_value:.2e}, Time: {elapsed:.2f}s')
                start_time = timeit.default_timer()

    def callback(self, loss):
        self.loss_lbfgs_log.append(loss)
        if len(self.loss_lbfgs_log) % 1000 == 0:
            print(f'L-BFGS Eval {len(self.loss_lbfgs_log)}, Loss: {loss:.3e}')

    def train_p(self):
        print(" L-BFGS-B ...")
        tf_dict = {
            self.x_u_tf: self.X_uc, self.u_c_tf: self.u_c,
            self.x_a_tf: self.X_ac, self.a_c_tf: self.a_c,
            self.x_k_tf: self.X_kc, self.k_c_tf: self.k_c,
            self.x_tf: self.X, self.q_tf: self.q
        }
        self.optimizer_lbfgs.minimize(
            self.sess,
            feed_dict=tf_dict,
            fetches=[self.loss],
            loss_callback=self.callback
        )
        print("L-BFGS-B FINISH")


if __name__ == '__main__':
    np.random.seed(1202)
    tf.set_random_seed(1202)
    
    L_val, EI_val = 1.0, 1.0
    sigma_val = 0.04
    test_loads = [(-3.0, 0.3), (-2.0, 0.5), (-2.5, 0.7)]
    
    layers = [[1] + 3 * [10] + [1] for _ in range(4)]
    
    print("\n" + "="*85)
    print(" START PINN three condition train (Adam + L-BFGS-B)")
    print("="*85)
    
    for i, (P_val, xp_val) in enumerate(test_loads):
        tf.reset_default_graph()
        
        X_star = np.linspace(0.0, L_val, 1001).reshape(-1, 1).astype(np.float32)
        idx = np.random.choice(X_star.shape[0], 300, replace=False)
        X_global = X_star[idx, :]
        x_left = max(0.0, xp_val - 5.0 * sigma_val)
        x_right = min(L_val, xp_val + 5.0 * sigma_val)
        X_local = np.linspace(x_left, x_right, 201).reshape(-1, 1).astype(np.float32)
        X_extra = np.array([[0.0], [L_val], [xp_val]], dtype=np.float32)
        X_train = np.unique(np.vstack([X_global, X_local, X_extra]).reshape(-1))
        X_train = np.sort(X_train).reshape(-1, 1).astype(np.float32)

        q_train = gaussian_concentrated_load(X_train, P=P_val, xp=xp_val, sigma=sigma_val).astype(np.float32)
        
        X_uc = np.array([[0.0], [L_val]], dtype=np.float32)
        u_c = np.array([[0.0], [0.0]], dtype=np.float32)
        X_ac = np.array([[0.0], [L_val]], dtype=np.float32)
        a_c = np.array([[0.0], [0.0]], dtype=np.float32)
        X_kc = np.array([[0.0], [L_val]], dtype=np.float32)
        k_c = np.array([[0.0], [0.0]], dtype=np.float32)

        print(f"\n condition {i+1}/3: P={P_val}, xp={xp_val}")
        model = PINN_model(layers, X_train, q_train, X_uc, u_c, X_ac, a_c, X_kc, k_c)
 
        case_start_time = timeit.default_timer()

        model.train(nIter=20000)
        

        model.train_p()

        case_end_time = timeit.default_timer()
        case_total_time = case_end_time - case_start_time
        # ------------------------------------

        save_prefix = f'SC-ml-PINN_Combined_P{P_val}_xp{xp_val}'
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
                 q_train=q_train)
        print(f"model save: {save_prefix}")
        
        print(f"condition {i+1} :")
        print(f"   - Adam Epochs : {len(model.loss_log)}")
        print(f"   - L-BFGS Epochs : {len(model.loss_lbfgs_log)}")
        print(f"   - Times: {case_total_time:.2f} s")
        # ------------------------------------