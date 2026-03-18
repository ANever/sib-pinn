import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
from pinn_base import PINN_BASE
from pinn_wave import WaveBasis

class Separator(tf.keras.layers.Layer):
    def __init__(self, multiplier, **kwargs):
        super().__init__(**kwargs)
        self.multiplier = multiplier
        
    def build(self, input_shape):
        self.f_out = input_shape[-1]*self.multiplier

    def call(self, inputs):
        return tf.reshape(tf.keras.ops.repeat(tf.inputs, self.multiplier) (len(tf.inputs), self.multiplier))
    
    def compute_output_shape(self, input_shape):
        return (*input_shape, self.multiplier)
        
class Combinator(tf.keras.layers.Layer):
    def __init__(self, outputs_lens, **kwargs):
        super().__init__(**kwargs)
        self.outputs_lens = outputs_lens
        n = tf.math.reduce_max(outputs_lens, axis=0)
        self.num_outputs = tf.math.reduce_sum(outputs_lens, axis=0)
        v = []
        for i, _len in enumerate(outputs_lens):
            v = v + list((np.array(range(_len)) + n*i))
        self.selection_matrix = tf.one_hot(v, n*len(outputs_lens))
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                  shape=(*input_shape, self.num_outputs), # here input_shape[-1] are active variables
                                  initializer='glorot_uniform',
                                  trainable=True)
        
    def call(self, inputs):
        return tf.matmul(keras.ops.ravel(tf.matmul(inputs, self.kernel)), self.selection_matrix)
    
    def compute_output_shape(self, input_shape):
        return (self.num_outputs,)
        


class DenseSeparated(tf.keras.layers.Layer):
    def __init__(self, num_outputs, activation=None, **kwargs):  #activation=None
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                  shape=(*input_shape, self.num_outputs),
                                  initializer='glorot_uniform',
                                  trainable=True)
                                         
        self.bias = self.add_weight(name="bias", shape=(*input_shape[:-1], self.num_outputs),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        return self.act(tf.matmul(inputs, self.kernel) + self.bias)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.num_outputs
        return output_shape

class PINN_SEP(PINN_BASE):
    def __init__(  
        self,
        f_hid,
        depth,
        w_init="Glorot",
        b_init="zeros",
        #dynamic_normalisation=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.f_hid = int(f_hid)
        self.depth = int(depth)
        self.w_init = w_init  # weight initialization
        self.b_init = b_init  # bias initialization
        self.model_name = "pinn_sep"
        
        outs = [2,3,1]
        multiplier = len(outs)
        self.add(Separator(multiplier))
        for _ in range(self.depth):
            self.add(DenseSeparated(self.f_hid, activation=self.act_func))
        self.add(Combinator(outs))
