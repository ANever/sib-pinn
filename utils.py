import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import re

def gen_points(num, bounds, n_vars=2):
        points = [0]*n_vars
        for i in range(n_vars):
            points[i] = tf.random.uniform((num, 1), bounds[i][0], bounds[i][1], dtype=tf.float32)
        return points
    
def gen_condition(conds, cond_name, **parser_kwargs):
    cond_dict = conds[cond_name]
    t, x = gen_points(cond_dict['N'], cond_dict['point_area'])
    c = tf.transpose(cond_dict['right_side'](t,x))
    eq_string = line_parser(cond_dict['eq_string'], **parser_kwargs)
    compute_grads = cond_dict['compute_grads']
    #plt.scatter(t,x)
    #plt.show()
    return (t, x, c, eq_string, compute_grads)
        
def line_parser(eq_string, func_names, variable_list = ['y', 'x']):
        splited = eq_string.split(' ')

        ops_stack = []

        def is_der_operator(string: str):
            if re.findall('\(d\/d..?\)', string):
                return True
            else:
                return False
            
        def apply_ops(ops_stack: list, func: str, variable_list: list):
            dif_powers = [0]*len(variable_list)
            for op in ops_stack:
                op = op.replace('(d/d', '')
                op = op.replace(')', '')
                op = op.split('^')

                var_index = variable_list.index(op[0])
                try:
                    power = op[1]
                except:
                    power = 1
                dif_powers[var_index] = int(power)
            previous = ''
            f_name = 'u_'
            dif_string = ''
            for i in range(len(variable_list)):
                dif_string += variable_list[i]*dif_powers[i]
            func_index = func_names.index(func)
            return (f_name + dif_string +'[:,'+str(func_index) +']')
            
        def is_func(string:str, variable_list):
            if string in func_names:
                return (True, 'basis')
            else:
                return (False, None)
        res = ''
        for i in range(len(splited)):
            if is_der_operator(splited[i]):
                ops_stack.append(splited[i])
            elif is_func(splited[i], variable_list)[0]:
                res += apply_ops(ops_stack, splited[i], variable_list)
                ops_stack = []
            else:
                res += splited[i]
        return res

def eval_dict(d, kwargs={},recursion=0):
    if recursion == 0:
        for key in d.keys():
            if key not in ['eq_string', 'act']:
                d[key] = eval(str(d[key]), kwargs)
        return d
    else:
        for key in d.keys():
            d[key] = eval_dict(d[key], kwargs, recursion-1)
        return d

def make_logger(add_data=None):
    now = datetime.datetime.now()
    now = now.strftime("%Y_%m_%d_%H_%M_%S")

    f_path = "./results/"
    f_name = now + ".txt"
    path = os.path.join(f_path, f_name)

    with open(path, mode="a") as f:
        print(add_data, file=f)
    return path

def write_logger(path, log):
    with open(path, mode="a") as f:
        print(log, file=f)



def plotting():
    fig, ax = plt.subplots(figsize=(8, 4))
    
    
    plt.clf()
    plt.close()

def plot_comparison(
                epoch, 
                x, y, u_inf, 
                umin, umax,
                vmin, vmax,
                xmin, xmax, xlabel, 
                ymin, ymax, ylabel, title = ''):
                
    fig, ax = plt.subplots(figsize=(8, 4))
    #ax.set_xticks
    xticks = (np.max(x) - np.min(x)) / 4.
    yticks = (np.max(y) - np.min(y)) / 4.
    
    plt.scatter(x, y, c=u_inf, cmap="turbo")#, vmin=umin, vmax=umax)
    plt.colorbar(ticks=np.arange(np.min(u_inf)+1e-6,  np.max(u_inf), (np.max(u_inf) - np.min(u_inf))/5))
    plt.xticks(np.arange(np.min(x), np.max(x)+1e-6, xticks))
    plt.yticks(np.arange(np.min(y), np.max(y)+1e-6, yticks))
    plt.xlim(np.min(x), np.max(x))
    plt.ylim(np.min(y), np.max(y))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("inference")
    
    plt.savefig("./results/comparison_" +title+"_"+ str(epoch) + ".png", dpi=300)
    plt.clf()
    plt.close()


def plot_loss_curve(
    epoch,
    logs, 
    labels
):
    epoch_log = logs[0]
    plt.figure(figsize=(8, 4))
    for log, label in zip(logs[1:],labels[1:]):
        plt.plot(epoch_log, log, ls="-",  alpha=.7, label=label)#, c="k")
    #plt.plot(epoch_log, loss_pde_log, ls="--", alpha=.3, label="loss_pde", c="tab:blue")
    #plt.plot(epoch_log, loss_ic_log,  ls="--", alpha=.3, label="loss_ic",  c="tab:orange")
    #plt.plot(epoch_log, loss_bc_log,  ls="--", alpha=.3, label="loss_bc",  c="tab:green")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xscale("linear")
    plt.yscale("log")
    plt.grid(alpha=.5)
    plt.tight_layout()
    plt.savefig("./results/loss_curve_" + str(epoch) + ".png", dpi=300)
    plt.clf()
    plt.close()
