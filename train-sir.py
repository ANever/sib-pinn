"""
********************************************************************************
training
********************************************************************************
"""

import os
import time
import yaml
import numpy as np
import tensorflow as tf

from config_gpu import config_gpu
from pinn_base import PINN
from utils import (
    make_logger,
    write_logger,
    eval_dict,
    gen_condition,
    plot_comparison,
    plot_loss_curve,
)

filename = "./settings-sir.yaml"


def main():
    # read settings
    with open(filename, mode="r") as file:
        settings = yaml.safe_load(file)

    # run hyperparameters args
    logger_path = make_logger("seed: in model")
    args = eval_dict(settings["ARGS"])

    # ======model=======

    model_args = eval_dict(settings["MODEL"], {"tf": tf, "np": np})

    for key in model_args.keys():
        if isinstance(model_args[key], list):
            model_args[key] = tf.constant(model_args[key], tf.float32)

    var_names = settings["IN_VAR_NAMES"]
    func_names = settings["OUT_VAR_NAMES"]
    model = PINN(var_names=var_names, func_names=func_names, **model_args)
    model.init_custom_vars(
        dict_consts=settings["CUSTOM_CONSTS"],
        dict_funcs=settings["CUSTOM_FUNCS"],
        var_names=var_names,
    )

    print("INIT DONE")

    # tmin, xmin = in_lb = model_args['in_lb']
    # tmax, xmax = in_ub = model_args['in_ub']

    tmin, x1min, x2min = in_lb = model_args["in_lb"]
    tmax, x1max, x2max = in_ub = model_args["in_ub"]

    # ======conditions=======

    conds = eval_dict(settings["CONDS"], locals() | {"tf": tf} | model.custom_vars, 1)
    conditions = []
    for key in list(conds.keys()):
        cond_ = gen_condition(
            conds, key, func_names=func_names, var_names=var_names, **model.custom_vars
        )
        conditions.append(cond_)
    cond_string = [
        "self.loss_(*conditions[" + str(i) + "])," for i in range(len(conditions))
    ]
    cond_string = "(" + "".join(cond_string) + ")"

    model.init_dynamical_normalisastion(len(conditions))

    # ======outputs=======

    ns = eval_dict(settings["NS"])
    x = [0] * len(var_names)
    for i in range(len(var_names)):
        x[i] = np.linspace(in_lb[i], in_ub[i], ns["nx"][i])
    x = np.meshgrid(*x)
    for i in range(len(var_names)):
        x[i] = tf.cast(x[i].reshape(-1, 1), dtype=tf.float32)
    x_ref = tf.transpose(tf.cast(x, dtype=tf.float32))[0]
    u_ref = tf.cast(np.zeros(ns["nx"]).reshape(-1, 1), dtype=tf.float32)
    exact = tf.cast(model.custom_vars["exact"](x_ref), dtype=tf.float32)

    # log
    losses_logs = np.empty((len(conds.keys()), 1))

    # training
    wait = 0
    loss_best = tf.constant(1e20)
    loss_save = tf.constant(1e20)
    t0 = time.perf_counter()

    cond_string_here = [
        "model.loss_(*conditions[" + str(i) + "])," for i in range(len(conditions))
    ]
    cond_string_here = "(" + "".join(cond_string_here) + ")"

    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tp:
        tp.watch(model.trainable_weights)
        losses = tf.cast(eval(cond_string_here), tf.float32)
        losses_normed = model.normalize_losses(losses)
        grads = tp.jacobian(losses_normed, model.trainable_weights)
        model.update_gammas(grads)
        loss_glb = tf.math.reduce_sum(losses_normed)
    del tp
    grad = [tf.reduce_sum(v, axis=0) for v in grads]
    print(grad)
    # print(tf.math.reduce_mean(grad))

    print("START TRAINING")
    for epoch in range(1, int(args["epochs"]) + 1):
        # gradient descent
        loss_glb, losses = model.train(conditions, cond_string)
        # log
        t1 = time.perf_counter()
        elps = t1 - t0

        losses = dict(zip(conds.keys(), losses))
        logger_data = [key + f": {losses[key]:.3e}, " for key in losses.keys()]
        logger_data = f"epoch: {epoch:d}, loss_total: {loss_glb:.3e}, " + ", ".join(
            logger_data
        )
        write_logger(logger_path, logger_data)
        if epoch % 1000 == 0:
            print(logger_data)
            #    print(model.gammas)
            print(elps)
        if epoch % 250 == 0:
            print(">>>>> saving")
            model.save_weights("./saved_weights/weights_ep" + str(epoch))
            if loss_glb < loss_save:
                model.save_weights("./best_weights/best_weights")
                loss_save = loss_glb

        # early stopping
        if loss_glb < loss_best:
            loss_best = loss_glb
            wait = 0
        else:
            if wait >= args["patience"]:
                print(">>>>> early stopping")
                break
            wait += 1

        # monitor
        if epoch % 1000 == 0:
            u_ = model(x_ref)
            u_n = u_.numpy().transpose()
            print("ERROR ", np.max(np.abs(exact - u_[:, 0])))
            plot_commons = {
                "epoch": epoch,
                "x": x_ref[:, 0],
                "y": x_ref[:, 1],
                "umin": -0.1,
                "umax": 3.0,
                "vmin": 0.0,
                "vmax": 0.05,
                "xmin": tmin,
                "xmax": tmax,
                "xlabel": var_names[0],
                "ymin": x1min,
                "ymax": x1max,
                "ylabel": var_names[1],
            }
            # for func, title in zip(u_n,func_names):
            #    plot_comparison(u_inf=func,title=title, **plot_commons)
            #    plot_comparison(u_inf=exact,title=title+'exact', **plot_commons)
            #    plot_comparison(u_inf=(np.abs(exact-func)), title=title+'diff', **plot_commons)
            plot_loss_curve(epoch, losses_logs[:, 1:], labels=list(conds.keys()))

    def phi_s(t, S, I):
        x = tf.cast(np.array([[t, S, I]], dtype=float), dtype=tf.float32)
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tp1:
            tp1.watch(x)
            phi = model(x)
            _phi_s = tp1.batch_jacobian(phi, x)
        return _phi_s[0, 0, 1]

    import scipy as sp
    from scipy.integrate import RK45
    from copy import deepcopy
    import matplotlib.pyplot as plt

    from scipy.special import jv

    plt.style.use("grayscale")

    control_log = []

    def _right_side(t, Y, w1, w2, w3, alpha, beta, mu, save_control=False, **kwargs):
        X = deepcopy(Y)
        control = (-w2 + mu * phi_s(t, *Y[:2])) * Y[0] / 2 / w3
        # control = 1.
        if save_control:
            control_log.append(np.array(control))
        X[0] = -alpha * Y[0] * Y[1] - mu * control * Y[0]
        X[1] = alpha * Y[0] * Y[1] - beta * Y[1]
        X[2] = alpha * Y[0] * Y[1]

        X[3] = -alpha * Y[3] * Y[4]
        X[4] = alpha * Y[3] * Y[4] - beta * Y[4]
        X[5] = alpha * Y[3] * Y[4]
        return np.array(X)

    print("START INTEGRATION")

    def right_side(t, Y, save_control=False):
        return _right_side(
            t, Y, save_control=save_control, **(settings["CUSTOM_CONSTS"])
        )

    def RK45my(f, t, x0, h, n):
        x = np.zeros((n, len(x0)))
        x[0] = x0
        for i in range(1, n):
            t = h * i
            k1 = f(t, x[i - 1], save_control=True)
            k2 = f(t, x[i - 1] + h * k1 / 2)
            k3 = f(t, x[i - 1] + h * k2 / 2)
            k4 = f(t, x[i - 1] + h * k3)
            x[i] = x[i - 1] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    n = 50
    h = 0.02
    y_values = RK45my(right_side, 0, np.array([0.9, 0.1, 0.0, 0.9, 0.1, 0.0]), h, n)
    t_values = np.linspace(0, n * h, n)
    """
    #solution = RK45(right_side, 0 , np.array([0.9, 0.1]) , 1 ,1, 0.001, 3**-6)
#RK45(right_side, t0=0, t_bound=1, y0=np.array([0.9, 0.1]), first_step=0.05, max_step=0.05, vectorized=True)
    
    # collect data
    t_values = []
    y_values = []
    for i in range(100):
        # get solution step state
        solution.step()
        t_values.append(solution.t)
        y_values.append(solution.y)
        print(solution.y)
        # break loop after modeling is finished
        if solution.status == 'finished':
            break
    """
    print(y_values)

    y_values = np.array(y_values)

    w1 = np.array(settings["CUSTOM_CONSTS"]["w1"])
    w2 = np.array(settings["CUSTOM_CONSTS"]["w2"])
    w3 = np.array(settings["CUSTOM_CONSTS"]["w3"])

    control_log.append(0.0)
    control_log = np.array(control_log)
    y_values = np.array(y_values)
    print(y_values)
    cost_controlled = (
        np.sum(
            w1 * y_values[:, 1] ** 2
            + w2 * y_values[:, 2] * control_log
            + w3 * control_log**2
        )
        * h
    )
    cost_uncontrolled = np.sum(w1 * y_values[:, 4] ** 2) * h
    print(
        "COSTS:\n with control: ",
        cost_controlled,
        "\n without control: ",
        cost_uncontrolled,
    )
    width = 3
    plt.plot(
        t_values,
        y_values[:, 0],
        linestyle="-.",
        linewidth=width,
        label="S с управлением",
    )
    plt.plot(
        t_values,
        y_values[:, 1],
        linestyle=":",
        linewidth=width,
        label="I с управлением",
    )

    plt.plot(t_values, y_values[:, 3], linewidth=width, label="S без управления")
    plt.plot(
        t_values,
        y_values[:, 4],
        linewidth=width,
        linestyle="--",
        label="I без управления",
    )
    plt.xlabel("t")
    plt.legend()
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tick_params(axis="both", which="minor", labelsize=10)
    plt.savefig("si-comparison.png", dpi=300)
    plt.savefig("si-comparison.eps")
    plt.show()
    plt.clf()

    plt.plot(
        t_values,
        y_values[:, 2],
        linewidth=width,
        linestyle="-.",
        label="Итого I с управлением",
    )
    plt.plot(t_values, y_values[:, 5], linewidth=width, label="Итого I без управления")
    plt.xlabel("t")
    plt.legend()
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tick_params(axis="both", which="minor", labelsize=10)

    plt.savefig("i.png", dpi=300)
    plt.savefig("i.eps")
    plt.show()
    plt.clf()

    plt.plot(
        t_values,
        control_log,
        linewidth=width,
    )
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tick_params(axis="both", which="minor", labelsize=10)

    plt.xlabel("t")
    plt.savefig("control.png", dpi=300)
    plt.savefig("control.eps")
    plt.show()


if __name__ == "__main__":
    config_gpu(flag=0, verbose=True)
    main()

    print("SUCCESS")
