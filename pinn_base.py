import os
import numpy as np
import tensorflow as tf

import tensorflow.keras as keras

from utils import eval_dict, replace_words


class PINN(tf.keras.Sequential):
    def __init__(
        self,
        # f_in,
        # f_out,
        f_hid,
        depth,
        in_lb,
        in_ub,
        var_names,
        func_names,
        w_init="Glorot",
        b_init="zeros",
        act="tanh",
        lr=1e-3,
        seed=42,
    ):
        super().__init__()
        self.var_names = var_names
        self.f_in = int(len(var_names))  # f_in)
        self.f_out = int(len(func_names))  # f_out)
        self.f_hid = int(f_hid)
        self.depth = int(depth)
        self.lb = in_lb  # lower bound of input
        self.ub = in_ub  # upper bound of input
        self.mean = (in_lb + in_ub) / 2
        self.w_init = w_init  # weight initialization
        self.b_init = b_init  # bias initialization
        self.act = act  # activation
        self.lr = lr  # learning rate
        self.seed = int(seed)
        self.f_scl = "minmax"  # "linear" / "minmax" / "mean"
        self.d_type = tf.float32

        self.act_func = self.init_act_func(self.act)

        self.dynamic_normalisation = True
        # self.gammas = None
        # seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # build a network
        self.add(keras.layers.InputLayer((self.f_in,)))
        for _ in range(self.depth):
            self.add(keras.layers.Dense(self.f_hid, activation=self.act_func))
        self.add(keras.layers.Dense(self.f_out))

        # optimizer (overwrite the learning rate if necessary)
        self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr, decay_steps=1000, decay_rate=0.9
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # system params
        self.custom_vars = {}

    def _inner_lambda(self, _dict_func, var_names: list, other_dicts={}):  # _variables
        inner_vars_dict = {}
        for name in var_names:
            inner_vars_dict[name] = eval(name)
        return eval(_dict_func, other_dicts | inner_vars_dict)

    def init_custom_vars(
        self, dict_consts: dict, dict_funcs: dict = {}, var_names: list = []
    ):
        def make_lambda(string):
            return lambda _variables: eval(
                string, self.custom_vars | {"tf": tf, "_variables": _variables}
            )

        self.custom_vars = eval_dict(dict_consts, {"tf": tf})
        for key in self.custom_vars.keys():
            self.custom_vars[key] = tf.constant(self.custom_vars[key])

        replecement_dict = {}
        for i in range(len(var_names)):
            replecement_dict[var_names[i]] = "_variables[:," + str(i) + "]"
        for key in dict_funcs.keys():
            self.custom_vars.update({
                key: make_lambda(replace_words(dict_funcs[key], replecement_dict))
            })

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
            raise NotImplementedError(">>>>> forward_pass (act)")

    @tf.function#(jit_compile=True)
    def compute_pde(self, vars, eq_string, compute_grads=False):
        if compute_grads:
            with tf.GradientTape(
                persistent=False, watch_accessed_variables=True
            ) as tp1:
                #tp1.watch(vars)
                with tf.GradientTape(
                    persistent=True, watch_accessed_variables=True
                ) as tp2:
                    #tp2.watch(vars)
                    u_ = self(vars, training=True)
                u_x = tp2.batch_jacobian(u_, vars)
                del tp2
            u_xx = tp1.batch_jacobian(u_x, vars)
            del tp1
            # try:
            g = eval(eq_string, locals() | self.custom_vars | {"tf": tf})
            # except:
            #    raise ValueError(eq_string)
            g = tf.transpose(tf.convert_to_tensor(g, dtype=tf.float32))
        else:
            u_ = self(vars, training=True)
            self.var_names
            try:
                g = eval(eq_string, locals() | self.custom_vars | {"tf": tf})
            except:  # SyntaxError:
                raise SyntaxError("The line has an error\n" + eq_string)
            g = tf.transpose(tf.convert_to_tensor(g, dtype=tf.float32))
        return u_, g

    @tf.function
    def std_error(self, vals, exact_vals):
        return tf.reduce_mean(tf.square(vals - exact_vals))

    @tf.function
    def loss_(self, x, exact_vals, eq_string, compute_grads):
        _, g_ = self.compute_pde(x, eq_string, compute_grads)
        loss = self.std_error(g_, exact_vals)
        return loss

    # def infer(self, x):
    #    u_, g_ = self.compute_pde(x, compute_grads=False)
    #    return u_, g_

    @tf.function
    def normalize(self, input_vector):
        vector0 = tf.identity(input_vector)
        vector = tf.identity(vector0)
        return input_vector * tf.math.reduce_max(vector) / vector

    def normalize_losses(self, vec):
        return vec * self.gammas

    def init_dynamical_normalisastion(self, num_of_losses):
        self.gammas = tf.Variable(tf.ones(num_of_losses), tf.float32)

    @tf.function
    def update_gammas(self, grads):
        def full_reduct(v):
            v = tf.math.abs(v)
            # v = v*v
            while True:
                try:
                    v = tf.reduce_sum(v, axis=1)
                except:
                    break
            return v

        grd = tf.cast([full_reduct(v) for v in grads], tf.float32)
        grd = tf.reduce_sum(tf.math.abs(grd), axis=0)
        # new_gammas = tf.math.reduce_min(grd) * tf.math.divide(tf.ones_like(grd), grd)
        # self.gammas.assign(tf.math.abs(new_gammas)/2)

    @tf.function
    def train(self, conditions, conds_string):
        with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tp:
            #tp.watch(self.trainable_weights)
            losses = tf.cast(eval(conds_string), tf.float32)
            losses_normed = self.normalize_losses(losses)
            grads = tp.jacobian(losses_normed, self.trainable_weights)
        del tp
        self.update_gammas(grads)
        loss_glb = tf.math.reduce_sum(losses_normed)
        #del tp
        grad = [tf.reduce_sum(v, axis=0) for v in grads]
        """
        with tf.GradientTape(persistent=False) as tp1:
            losses2 = tf.cast(eval(conds_string),tf.float32)
            loss_glb = tf.math.reduce_sum(losses2)
            grad = tp1.gradient(loss_glb, self.trainable_weights)
            del tp1
        """
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))
        return loss_glb, losses
