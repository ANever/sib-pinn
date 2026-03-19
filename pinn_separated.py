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

from tensorflow.keras.layers import RepeatVector



'''
class Separator(tf.keras.layers.Layer):
    def __init__(self, multiplier, **kwargs):
        super().__init__(**kwargs)
        self.multiplier = multiplier
        
    def call(self, inputs):
        repeated_inputs = tf.keras.ops.repeat(inputs, self.multiplier, axis=-1)
        return tf.reshape(repeated_inputs, (*inputs.shape, self.multiplier))
    
    def compute_output_shape(self, input_shape):
        return (*input_shape, self.multiplier)
'''

class Combinator(tf.keras.layers.Layer):
    def __init__(self, selection_matrix, **kwargs):
        super().__init__(**kwargs)
        self.selection_matrix = selection_matrix
        self.num_outputs = selection_matrix.shape[1]
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                  shape=(self.num_outputs, *input_shape[1:]),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
    def call(self, inputs):
        return keras.ops.matmul(tf.tensordot(inputs, self.kernel, axes=[[1,2], [1,2]]), tf.transpose(self.selection_matrix))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)

class DenseSeparated(tf.keras.layers.Layer):
    def __init__(self, num_outputs, activation=None, **kwargs):  #activation=None
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                    shape = (self.num_outputs, input_shape[1], input_shape[2]),
                                    initializer='glorot_uniform',
                                    trainable=True)

        self.bias = self.add_weight(name="bias",
                                    shape=(self.num_outputs, input_shape[2]),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        #return self.activation(tf.matmul(inputs, self.kernel))# + self.bias)
        return self.activation(keras.ops.add(tf.tensordot(self.kernel, inputs), self.bias))
        #rewrite it to be convolution of inputs of [1,2] 
        #return self.activation(keras.ops.add(keras.ops.matmul(self.kernel, inputs), self.bias))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[1] = self.num_outputs
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
        
        #outs = [2,2,1]
        outs = [5]
        multiplier = len(outs)
        
        n = np.max(outs, axis=0)
        num_outputs = np.sum(outs, axis=0)
        v = []
        for i, _len in enumerate(outs):
            v = v + list((np.array(range(_len)) + n*i))
        selection_matrix = tf.one_hot(v, n*len(outs))
        
        self.add(WaveBasis())
        #self.add(Separator(multiplier))
        self.add(RepeatVector(multiplier))
        #self.add(keras.layers.Permute((2, 1))) #temporary TODO get rid of this (rework DenseSeparated)
        for _ in range(self.depth):
            self.add(DenseSeparated(self.f_hid, activation=self.act_func))
        self.add(Combinator(selection_matrix))
