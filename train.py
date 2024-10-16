import os
import time
import yaml
import numpy as np
import tensorflow as tf

from config_gpu import config_gpu
from pinn import PINN
from utils import *

filename = "./settings.yaml"

def main():
    # read settings
    with open(filename, mode="r") as f:
        settings = yaml.safe_load(f)

    # seed and run hyperparameters args
    seed = settings["SEED"]["seed"]
    logger_path = make_logger("seed: " + str(seed))
    args = eval_dict(settings['ARGS'])
    
    #======model=======
    
    model_args = eval_dict(settings['MODEL'],{'tf':tf, 'np':np})
    
    for key in model_args.keys():
        if isinstance(model_args[key], list):
            model_args[key] = tf.constant(model_args[key], tf.float32)
    
    model = PINN(**(model_args))
    model.init_custom_vars(eval_dict(settings['CUSTOM_VARS']))
    
    tmin, xmin = in_lb = model_args['in_lb']
    tmax, xmax = in_ub = model_args['in_ub']
    
    #======conditions=======
    
    conds = eval_dict(settings['CONDS'], locals()|{'tf':tf}, 1)
    var_names = settings['VAR_NAMES']
    conditions = []
    for key in list(conds.keys()):
        cond_ = gen_condition(conds, key, func_names=var_names)
        conditions.append(cond_)

    cond_string = ['self.loss_(*conditions['+str(i)+']),' for i in range(len(conditions))]
    cond_string = '(' + ''.join(cond_string) + ')'
    #======outputs=======
    
    ns = eval_dict(settings['NS'])
    nt, nx = ns['nx']
    
    # bounds
    t = np.linspace(tmin, tmax, nt)
    x = np.linspace(xmin, xmax, nx)
    
    # reference
    t_ref, x_ref = np.meshgrid(t, x)
    t_ref = t_ref.reshape(-1, 1)
    x_ref = x_ref.reshape(-1, 1)
    u_ref = np.zeros((nt,nx)).reshape(-1, 1)
    t_ref = tf.cast(t_ref, dtype=tf.float32)
    x_ref = tf.cast(x_ref, dtype=tf.float32)
    u_ref = tf.cast(u_ref, dtype=tf.float32)
    
    # log
    log_names = ['epoch', 'glb', 'pde', 'ic', 'bc', 'u_l2']
    losses_logs = np.empty((6,1))
    
    # training
    wait = 0
    loss_best = tf.constant(9999.)
    loss_save = tf.constant(9999.)
    t0 = time.perf_counter()  
    
    
    for epoch in range(1, args['epochs']+1):
        # gradient descent
        loss_glb, losses = model.train(
                conditions, 
                cond_string
            )
        loss_pde, loss_ic, loss_bc1, loss_bc2 = losses
        loss_bc = loss_bc1 + loss_bc2
        
        # log
        losses_logs = np.append(losses_logs, np.array([[epoch, loss_glb, loss_pde, loss_ic, loss_bc, 0]]).T, axis=1)
        t1 = time.perf_counter()
        elps = t1 - t0
        
        logger_data = \
            f"epoch: {epoch:d}, " \
            f"loss_glb: {loss_glb:.3e}, " \
            f"loss_pde: {loss_pde:.3e}, " \
            f"loss_ic: {loss_ic:.3e}, " \
            f"loss_bc: {loss_bc:.3e}, " \
            f"loss_best: {loss_best:.3e}, " \
            f"wait: {wait:d}, " \
            f"elps: {elps:.3f}, " \
            f" "
            
        print(logger_data)
        write_logger(logger_path, logger_data)
        if epoch %1000 == 0:
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
            if wait >= args['patience']:
                print(">>>>> early stopping")
                break
            wait += 1
        
        # monitor
        if epoch % 1000 == 0:
            u_ = model.forward_pass(tf.concat([x_ref, t_ref], axis=1))
            u_n = u_.numpy()
            comps = u_n.transpose()
            comps_names = var_names
            for comp, title in zip(comps,comps_names):
                plot_comparison(
                        epoch, 
                        x=t_ref, y=x_ref, u_inf=comp,
                        umin=-0.1, umax=3.,
                        vmin= 0., vmax=.05,
                        xmin=tmin, xmax=tmax, xlabel="t",
                        ymin=xmin, ymax=xmax, ylabel="x", title=title
                    )
            plot_loss_curve(
                epoch,
                losses_logs[:,1:],
                labels = log_names
            )
            
if __name__ == "__main__":
    config_gpu(flag=0, verbose=True)
    main()
