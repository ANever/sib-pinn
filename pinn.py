import os
import numpy as np
import tensorflow as tf

class PINN(tf.keras.Sequential):
    def __init__(
        self,
        f_in, f_out, f_hid, depth, 
        in_lb, in_ub, in_mean, 
        w_init="Glorot", b_init="zeros", act="tanh", lr=1e-3, seed=42,
    ):
        super().__init__()
        self.f_in   = f_in
        self.f_out  = f_out
        self.f_hid  = f_hid
        self.depth  = depth
        self.lb     = in_lb      # lower bound of input
        self.ub     = in_ub      # upper bound of input
        self.mean   = in_mean    # mean of input
        self.w_init = w_init     # weight initialization
        self.b_init = b_init     # bias initialization
        self.act    = act        # activation
        self.lr     = lr
        self.seed   = seed
        self.f_scl  = "minmax"   # "linear" / "minmax" / "mean"
        self.d_type = tf.float32
        
        self.act_func = self.init_act_func(self.act)
        
        # seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # build a network
        self._layers = [self.f_in] + (self.depth - 1) * [self.f_hid] + [self.f_out]
        self._weights, self._biases, self._alphas, self._params \
            = self.dnn_initializer(self._layers)

        # optimizer (overwrite the learning rate if necessary)
        # self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=1e-3, decay_steps=1000, decay_rate=.9
        # )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # system params
        
        self.bet = tf.constant(.5 , dtype=self.d_type)
        self.gamma = tf.constant(.2 , dtype=self.d_type)
        self.epsilon = tf.constant(.1 , dtype=self.d_type)
        
        self.custom_vars = {}
        
    def init_custom_vars(self, dict_):
        self.custom_vars = dict_

    def init_act_func(self, act):
        if act == "tanh":
            return lambda u: tf.math.tanh(u)
        elif act == "softplus":
            return lambda u: tf.math.softplus(u)
        elif act == "silu" or act == "swish":
            return lambda u: tf.multiply(u, tf.math.sigmoid(u))
        elif act == "gelu":
            return lambda u: tf.multiply(u, tf.math.sigmoid(1.702 * u))
        elif act == "mish":
            return lambda u: tf.multiply(u, tf.math.tanh(tf.math.softplus(u)))
        else:
            raise NotImplementedEror(">>>>> forward_pass (act)")
    
    def dnn_initializer(self, layers):
        weights = []
        biases  = []
        alphas  = []
        params  = []
        for l in range(0, self.depth):
            w = self.weight_initializer(shape=[layers[l], layers[l+1]], depth=l)
            b = self.bias_initializer  (shape=[       1,  layers[l+1]], depth=l)
            weights.append(w)
            biases.append(b)
            params.append(w)
            params.append(b)
            if l < self.depth - 1:
                a = tf.constant(1., dtype=self.d_type, name="a"+str(l))
                alphas.append(a)
        return weights, biases, alphas, params

    def weight_initializer(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.w_init == "Glorot":
            std = np.sqrt(2 / (in_dim + out_dim))
        elif self.w_init == "He":
            std = np.sqrt(2 / in_dim)
        elif self.w_init == "LeCun":
            std = np.sqrt(1 / in_dim)
        else:
            raise NotImplementedEror(">>>>> weight_initializer")
        weight = tf.Variable(
            tf.random.truncated_normal(shape = [in_dim, out_dim], \
            mean = 0., stddev = std, dtype = self.d_type), \
            dtype = self.d_type, name = "w" + str(depth)
            )
        return weight

    def bias_initializer(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.b_init == "zeros":
            bias = tf.Variable(
                tf.zeros(shape = [in_dim, out_dim], dtype = self.d_type), \
                dtype = self.d_type, name = "b" + str(depth)
                )
        elif self.b_init == "ones":
            bias = tf.Variable(
                tf.ones(shape = [in_dim, out_dim], dtype = self.d_type), \
                dtype = self.d_type, name = "b" + str(depth)
                )
        else:
            raise NotImplementedEror(">>>>> bias_initializer")
        return bias

    #@tf.function
    def forward_pass(self, x):
        # feature scaling
        if self.f_scl == None or self.f_scl == "linear":
            z = tf.identity(x)
        elif self.f_scl == "minmax":
            z = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        elif self.f_scl == "mean":
            z = (x - self.mean) / (self.ub - self.lb)
        else:
            raise NotImplementedEror(">>>>> forward_pass (f_scl)")

        # forward pass        
        for l in range(0, self.depth - 1):
            w = self._weights[l]
            b = self._biases [l]
            a = self._alphas [l]
            u = tf.math.add(tf.linalg.matmul(z, w), b)
            u = tf.multiply(a, u)
            z = self.act_func(u)
        w = self._weights[-1]
        b = self._biases [-1]
        u = tf.math.add(tf.linalg.matmul(z, w), b)
        z = tf.identity(u)   # identity
        y = tf.identity(z)
        return y

    #TODO here x and y may be converted into one variable then derivatives may be taken from jacobian in additional index, that is removed now with [:,:,0]
    @tf.function
    def compute_pde(self, y, x, eq_string, compute_grads=False):
        if compute_grads:
            with tf.GradientTape(persistent=True) as tp1:
                tp1.watch(y)
                tp1.watch(x)
                with tf.GradientTape(persistent=True) as tp2:
                    tp2.watch(y)
                    tp2.watch(x)
                    u_ = self.forward_pass(tf.concat([x, y], axis=1))
                u_x = tp2.batch_jacobian(u_, x)[:,:,0]
                u_y = tp2.batch_jacobian(u_, y)[:,:,0]
                del tp2
            u_xx = tp1.batch_jacobian(u_x, x)[:,:,0]
            del tp1
            g = eval(eq_string, locals()|self.custom_vars)
        else:
            u_ = self.forward_pass(tf.concat([x, y], axis=1))
            g = eval(eq_string, locals()|self.custom_vars)
        return u_, g

    @tf.function
    def sum_square_errors(self, error_func, vals, exact_vals):
      vals_ = tf.convert_to_tensor(vals)
      exact_vals_ = tf.convert_to_tensor(exact_vals, dtype=tf.float32)
      n = len(vals)
      i, result = tf.constant(0), tf.constant(0.)
      c = lambda i, _: tf.less(i, n)
      b = lambda i, result: (i + 1, result + error_func(vals_[i], exact_vals_[i]))
      return tf.while_loop(c, b, [i, result])[1]
    
    @tf.function
    def std_error(self, vals, exact_vals):
        return self.sum_square_errors(lambda x, y: tf.reduce_mean(tf.square(x - y)), vals, exact_vals)
    
    @tf.function    
    def loss_(self, t, x, exact_vals,
        eq_string,#=default_eq_string,
        compute_grads=False
    ):
        u_, g_ = self.compute_pde(t, x, eq_string, compute_grads)
        g_ = tf.stack(g_)
        loss = self.std_error(g_, exact_vals)
        return loss
    
    def infer(self, t, x):
        u_, g_ = self.compute_pde(t, x, compute_grads=False)
        return u_, g_
    
    @tf.function
    def train(self, conditions, conds_string):
        with tf.GradientTape(persistent=False) as tp:
            losses = eval(conds_string)
            loss_glb = tf.math.reduce_sum(losses)
        grad = tp.gradient(loss_glb, self._params)
        del tp
        
        self.optimizer.apply_gradients(zip(grad, self._params))
        return loss_glb, losses
