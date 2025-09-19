"""
********************************************************************************
training
********************************************************************************
"""
import time
import yaml
import numpy as np 
import tensorflow as tf
import pickle


from config_gpu import config_gpu
from pinn_base import PINN
from utils import (
    make_logger,
    write_logger,
    eval_dict,
    gen_condition,
    plot_loss_curve,
    plot_comparison,
)

from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()

filename = "./settings/sir-controlled (Copy).yaml"

def main():
    # read settings
    with open(filename, mode="r") as file:
        settings = yaml.safe_load(file)

    # run hyperparameters args
    logger_path = make_logger("seed: in model")
    args = eval_dict(settings["ARGS"])

    # ======model=======

    model_args = eval_dict(settings["MODEL"], {"tf": tf, "": np})

    for key in model_args.keys():
        if isinstance(model_args[key], list):
            model_args[key] = tf.constant(model_args[key], tf.float32)

    var_names = settings["IN_VAR_NAMES"]
    func_names = settings["OUT_VAR_NAMES"]
    model = PINN(var_names=var_names, func_names=func_names, **(model_args))
    model.init_custom_vars(
        dict_consts=settings["CUSTOM_CONSTS"],
        dict_funcs=settings["CUSTOM_FUNCS"],
        var_names=var_names,
    )
    print(model.custom_vars)

    print("INIT DONE")

    in_lb = model_args["in_lb"]
    tmin = in_lb[0]
    xmin = in_lb[1]
    in_ub = model_args["in_ub"]
    tmax = in_ub[0]
    xmax = in_ub[1]

    # tmin, x1min, x2min = in_lb = model_args["in_lb"]
    # tmax, x1max, x2max = in_ub = model_args["in_ub"]

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
    _x = [0] * len(var_names)
    for i in range(len(var_names)):
        _x[i] = tf.linspace(in_lb[i], in_ub[i], ns["nx"][i])
    print(len(_x))
    _x = (tf.meshgrid(*_x))

    #print(x)
    
    #x = tf.cast(np.empty((len(_x), int(np.prod(_x.shape)))), dtype=tf.float32)
    x = [0]*len(_x)
    for i in range(len(var_names)):
        x[i] = tf.reshape(_x[i],(-1,1))#tf.cast(_x[i].reshape(-1, 1), dtype=tf.float32)
    #x = tf.reshape(_x,()
    x_ref = tf.transpose(tf.cast(x, dtype=tf.float32))[0]
    u_ref = tf.cast(np.zeros(ns['nx']).reshape(-1, 1), dtype=tf.float32)
    exact = tf.cast(model.custom_vars["exact"](x_ref), dtype=tf.float32)
    
    print(x)

    print(x_ref)
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

    print("START TRAINING")
    for epoch in range(1, int(args["epochs"]) + 1):
        # gradient descent
        loss_glb, losses = model.train(conditions, cond_string)
        # log
        t1 = time.perf_counter()
        elps = t1 - t0
        #print(loss_glb)
        losses = dict(zip(conds.keys(), losses))
        logger_data = [key + f": {losses[key]:.3e}, " for key in losses.keys()]
        logger_data = f"epoch: {epoch:d}, loss_total: {loss_glb:.3e}, " + ", ".join(
            logger_data
        )
        write_logger(logger_path, logger_data)
        if epoch % 200 == 0:
            print(logger_data)
            print(elps)
        if epoch % 250 == 0:
            print(">>>>> saving")
            w_ending = '.weights.h5'
            model.save_weights("./saved_weights/weights_ep" + str(epoch)+w_ending)
            if loss_glb < loss_save:
                model.save_weights("./best_weights/best_weights"+w_ending)
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
            print("Estimation error ", np.max(np.abs(exact - u_[:, 0])))
            plot_commons = {
                "epoch": epoch,
                "x": x_ref[:, 0],
                "y": x_ref[:, 1],
                "xlabel": var_names[0],
                "ylabel": var_names[1],
            }
            for func, title in zip(u_n, func_names):
                plot_comparison(u_inf=func, title=title, **plot_commons)
                #    plot_comparison(u_inf=exact,title=title+'exact', **plot_commons)
                #    plot_comparison(u_inf=(.abs(exact-func)), title=title+'diff', **plot_commons)
                #    with open(title + str(epoch) + ".pickle", "wb") as handle:
                #    pickle.dump(u_n, handle, protocol=pickle.HIGHEST_PROTOCOL)
            plot_loss_curve(epoch, losses_logs[:, 1:], labels=list(conds.keys()))
if __name__ == "__main__":
    config_gpu(flag=0, verbose=True)
    main()
