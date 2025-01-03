Implementation of PINN, based on Tensorflow GradientTape.

Based on Shota Deguchi code https://github.com/ShotaDeguchi/DN_PINN.

Introduces TeX-like input of differential equations and arbitrary names of functions (see settings.yaml example).


## Usage
Please make the following directories

* `saved_weights`
* `best_weights`
* `results`

in each working directory, as required by `train.py`, `utils.py`.

To train a model, run
```
python train.py
```

To define a model fill `settings.yaml` file the following way:

set parameters of the training
```
ARGS:
  epochs: 5000     # max number of epochs
  batch_size: -1   # batch size for one epoch
  patience: 1000   # number of epochs without lowering loss before early stop
```
set model parameters
```
MODEL:
  f_in : 2          # number of input variables/nodes in output layers
  f_out: 3          # number of output variables/nodes in output layer
  f_hid: 20           # width of hidden layers
  depth: 3            # number of hidden layers
  in_lb: "[0.,-1.]"   # lower bounds of input variables
  in_ub: "[1.,1.]"    # upper bounds of input variables
  act: tanh         # activation function
  seed: 0             # random seed
```
set number of points generated for plotting grid
```
NS:
  nx: [100, 100]
```
```
IN_VAR_NAMES:
 [t, x]      # names of input functions/variables to be used in loss definition

OUT_VAR_NAMES:
 [s, i, r]   # names of output functions/variables to be used in loss definition
```
```
CUSTOM_CONSTS:  # custom defined variables to be used in loss definition
  epsilon: 0.1
  gamma: 0.3
  beta: 0.5

CUSTOM_FUNCS:   # custom defined variables to be used in loss definition, should depend on input variables and may depend on other custom vars and consts
  f: 1.+ x*0
  exact: x
```
set collocation, border etc. conditions to be used in loss
```
CONDS:
  pde:                                                             # arbitrary name for condition
     eq_string:  (d/dt) s - epsilon* (d/dx)^2 s ,                  # left side, dependant on PINN output and its derivatives
                 (d/dt) i + gamma* i - epsilon* (d/dx)^2 i , 
                 (d/dt) r - gamma* i - epsilon* (d/dx)^2 r
     right_side: x*[0., 0., 0.]                                    # right side, non-dependant of PINN outputs
     point_area:  [[tmin, tmax], [xmin, xmax]]                     # limits for input variables, used to generate collocation points
     N: int(1e4)                                                   # number of collocation points generated
```

## Device (CPU / GPU)
<code>train.py</code> assumes the use of a GPU. To train a model on the CPU, simply change the flag parameter in <code>config_gpu(flag)</code> from <code>flag=0</code> to <code>flag=-1</code>. A short description can be found in <code>config_gpu.py</code>. 
