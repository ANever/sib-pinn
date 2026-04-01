import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras

from tqdm import tqdm

from ..utils import (
    make_logger,
    replace_words,
    write_logger,
    eval_dict,
    plot_loss_curve,
    plot_comparison1d,
    to_gif,
    from_file
)

class AddConstantOuts(tf.keras.layers.Layer):
    def __init__(self, n_outs, **kwargs):
        super().__init__(**kwargs)
        self.n_const_outs = n_outs
        
    def build(self, input_shape):
        self.const_outs = self.add_weight(name="bias",
                                    shape=(self.n_const_outs,),
                                    initializer='zeros',
                                    trainable=True)
                                    
    def call(self, inputs):
        const_outs = keras.ops.outer(tf.ones(inputs.shape[0]), self.const_outs)
        return tf.keras.ops.concatenate((inputs, const_outs), axis=-1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] += self.n_const_outs
        return output_shape


class PINN_BASE(tf.keras.Sequential):
    def __init__(
        self,
        in_lb,
        in_ub,
        var_names,
        func_names,
        const_outs_names=[],
        act = 'tanh',
        lr=1e-3,
        dyn_norm=None,
        beta=0.2,
        seed=42,
    ):
        super().__init__()
        self.var_names = var_names
        self.func_names = func_names
        self.const_outs_names = const_outs_names
        self.f_in = int(len(var_names))  # f_in)
        self.f_out = int(len(func_names))  # f_out)
        self.lb = in_lb  # lower bound of input
        self.ub = in_ub  # upper bound of input
        self.mean = (in_lb + in_ub) / 2
        self.act = act  # activation
        self.lr = lr  # learning rate
        self.seed = int(seed)
        self.f_scl = "minmax"  # "linear" / "minmax" / "mean"
        self.d_type = tf.float32
        #self.model_name = "pinn"
        self.act_func = self.init_act_func(self.act)
        
        self.func_names += self.const_outs_names
        
        print(self.func_names)
        
        # Note that it assumes that first element of var_names belongs to pde
        self.dynamic_normalisation = dyn_norm
        if 0 <= beta and beta <= 1: 
            self.beta = beta
        else:
            raise ValueError("parameter beta must be between 0 and 1")

        # seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        # optimizer (overwrite the learning rate if necessary)
        #self.lr = tf.keras.optimizers.schedules.ExponentialDecay(
        #    initial_learning_rate=self.lr, decay_steps=3000, decay_rate=0.7
        #)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr) #
        self.custom_vars = {}

    #def _inner_lambda(self, _dict_func, var_names: list, other_dicts={}):  # _variables
    #    inner_vars_dict = {}
    #    for name in var_names:
    #        inner_vars_dict[name] = eval(name)
    #    return eval(_dict_func, other_dicts | inner_vars_dict)

    def init_custom_vars(self, dict_consts: dict, dict_funcs: dict = {}):
        def make_lambda(string):
            string = compile(string, "<string>", "eval",optimize=1)
            return tf.function(lambda vars_, u_: eval(
                string, self.custom_vars | {"tf": tf, "tfp":tfp, "vars_": vars_, "u_": u_}
            ))

        self.custom_vars = eval_dict(dict_consts, {"tf": tf})
        for key in self.custom_vars.keys():
            self.custom_vars[key] = tf.constant(self.custom_vars[key])

        replecement_dict = {}
        for i, name in enumerate(self.var_names):
            replecement_dict[name] = "vars_[:," + str(i) + "]"
        for i, name in enumerate(self.func_names):
            replecement_dict[name] = "u_[:," + str(i) + "]"
        for key in dict_funcs.keys():
            self.custom_vars.update({
                key: make_lambda(replace_words(dict_funcs[key], replecement_dict))
            })

    
    def postinit(self):
        n = len(self.const_outs_names)
        print(n)
        if n > 0:
            self.add(AddConstantOuts(n))
            
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
                persistent=True, watch_accessed_variables=True
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
            g = eval(eq_string, locals() | self.custom_vars | {"tf": tf})
        else:
            u_ = self(vars, training=True)
            g = eval(eq_string, locals() | self.custom_vars | {"tf": tf})
        g = tf.convert_to_tensor(g, dtype=tf.float32)
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

    def init_dynamical_normalisation(self, num_of_losses):
        self.gammas = tf.Variable(tf.ones(num_of_losses), dtype=tf.float32, trainable=False)

    @tf.autograph.experimental.do_not_convert
    def update_gammas(self, grads):
        # TODO: реализовать проверку совпадения первых размерностей всех градиентов
        # dims = set([tf.shape(v)[0] for v in grads])
        # if len(dims) > 1:
        #     raise ValueError(f"All tensors must have common first dimension, got: {dims}")
        
        if self.dynamic_normalisation:
            # Приведем градиенты в векторный вид, 
            # т.е. теперь все производные выстроены в одну строчку длины,
            # равной суммарному кол-ву обучаемых параметров модели
        
            grd = tf.concat([tf.reshape(v, [tf.shape(v)[0], -1]) for v in grads], axis=1)
            
            # TODO We assume here that the first grad is from PDE loss
            # Problem: might be several pdes
            match self.dynamic_normalisation:
                case "max_avg":
                    grd_mean_abs = tf.reduce_mean(tf.abs(grd), axis=1)
                    gammas_cup = tf.reduce_max(tf.abs(grd[0])) * tf.divide(tf.ones_like(grd_mean_abs), self.gammas * grd_mean_abs)

                case "inv_dir":
                    grd = tf.math.reduce_std(grd, axis=1)
                    gammas_cup = tf.reduce_max(grd) * tf.divide(tf.ones_like(grd), grd)
        
                case "dyn_norm": 
                    grd = tf.norm(grd, axis=1)
                    gammas_cup = tf.reduce_max(grd) * tf.divide(tf.ones_like(grd), grd)
                
                case None:
                    gammas_cup = self.gammas

                case _:
                    raise NotImplementedError(f"update_gammas has no dynamical normalisation option '{self.dynamic_normalisation}'")
            self.gammas.assign(self.beta * gammas_cup + (1 - self.beta) * self.gammas)

    @tf.function
    def train(self):
        conditions = self.conditions
        conds_string = self.conds_string
        with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tp:
            losses = tf.cast(eval(conds_string), tf.float32)
            #losses_normed = self.normalize_losses(losses)
            losses_normed = losses
            grads = tp.jacobian(losses_normed, self.trainable_weights)
        del tp
        self.update_gammas(grads)
        loss_glb = tf.math.reduce_sum(losses_normed)
        grad = [tf.reduce_sum(v, axis=0) for v in grads]
        self.optimizer.apply_gradients(zip(grad, self.trainable_weights))
        return loss_glb, losses
    
    @tf.function
    def eval_loss(self, conditions, conds_string):
        losses = tf.cast(eval(conds_string), tf.float32)
        losses_normed = tf.reduce_sum(self.normalize_losses(losses))
        return losses_normed
        
    #@tf.function
    def train_lbfgs(self, conditions, conds_string):
        loss = lambda: self.eval_loss(conditions, conds_string)
        res = lbfgs_minimize(self.trainable_weights, loss)
        return res
    
    def run_training(self, output_dir=''):
        logger_path = make_logger("seed: in model", output_dir=output_dir)
        losses_logs = np.empty((len(self.conds.keys()), 1))

        # training
        wait = 0
        loss_best = tf.constant(1e20)
        loss_save = tf.constant(1e20)
        t0 = time.perf_counter()

        args = eval_dict(self.settings["ARGS"])
        N = int(args["epochs"])
        pbar = tqdm(range(N), total=N, desc="N")
        #tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = 'logdir',
        #                                             histogram_freq = 1,)
                                                     
        for epoch in pbar:
            #tf.profiler.experimental.start('logdir')
            loss_glb, losses = self.train()#(model.conditions, model.conds_string)
            losses_logs = np.append(losses_logs, np.expand_dims(losses, axis=0).T, axis=1)
            elps = time.perf_counter() - t0
            pbar.set_postfix_str(f"Loss={loss_glb:.6f}", refresh=False)
            losses = dict(zip(self.conds.keys(), losses))
            logger_data = [key + f": {losses[key]:.3e}, " for key in losses.keys()]
            logger_data = f"epoch: {epoch:d}, loss_total: {loss_glb:.3e}, " + ", ".join(
                logger_data
            )
            write_logger(logger_path, logger_data)

            # early stopping
            lr_down_flag = False
            if loss_glb < loss_best:
                loss_best = loss_glb
                wait = 0
                if lr_down_flag:
                    self.lr *= 0.9
            else:
                if wait >= args["patience"]:
                    print(">>>>> early stopping")
                    break
                wait += 1
                if loss_glb > loss_best * 10:
                    lr_down_flag = True
            # monitor
            if epoch % 1000 == 0:
                
                var_names = self.settings["IN_VAR_NAMES"]
                func_names = self.func_names
        
                file_extension = "jpg"
                u_ = self(self.x_ref)
                u_n = u_.numpy().transpose()
                self.generate_plot_commons(epoch)
                #plot_commons = {
                #    "epoch": epoch,
                #    "x": self.x_ref[:, 0],
                #    "y": None, #x_ref[:, 1],
                #    "xlabel": var_names[0],
                #    "ylabel": None, #var_names[1],
                #}
                #for func, title in zip(u_n, func_names):
                #    plot_comparison1d(u_inf=func, 
                #                      title=title, 
                #                      file_extension=file_extension, 
                #                      output_dir=output_dir,
                #                      **plot_commons)
                
                plot_loss_curve(epoch, 
                                losses_logs[:, 1:], 
                                labels=list(self.conds.keys()), 
                                file_extension=file_extension,
                                output_dir=output_dir,)
    def generate_plot_commons(self, epoch=''):
        plot_commons = {
                    "epoch": epoch,
                    "x": self.x_ref[:, 0],
                    "xlabel": var_names[0],
                    }
        if len(self.settings["IN_VAR_NAMES"])>1:
            plot_commons["y"] = x_ref[:, 1]
            plot_commons["ylabel"] = var_names[1]


class PINN(PINN_BASE):
    def __init__(
        self,
        f_hid,
        depth,
        w_init="Glorot",
        b_init="zeros",
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.f_hid = int(f_hid)
        self.depth = int(depth)
        self.w_init = w_init  # weight initialization
        self.b_init = b_init  # bias initialization
        self.model_name = "pinn"
        
        
        # build a network
        self.add(keras.layers.InputLayer((self.f_in,)))
        for _ in range(self.depth):
            self.add(keras.layers.Dense(self.f_hid, activation=self.act_func))
        self.add(keras.layers.Dense(self.f_out))
