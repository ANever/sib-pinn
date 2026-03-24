import os
import warnings
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import RepeatVector, Reshape, Flatten

from .base import PINN_BASE
from .wave import WaveBasis

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
        self.num_outputs = selection_matrix.shape[-1]
        self.selection_matrix = (selection_matrix)
        
    def build(self, input_shape):
        self.out_shape = (input_shape[0], input_shape[-1]*input_shape[-2])
        self.kernel = self.add_weight(name="kernel",
                                  shape=(self.out_shape[-1], self.num_outputs),
                                  initializer='glorot_uniform',
                                  trainable=True)
        
    def call(self, inputs):
        #return keras.ops.matmul(tf.tensordot(inputs, self.kernel, axes=[[1,2], [1,2]]), (self.selection_matrix))
        mat = keras.ops.matmul(tf.keras.ops.reshape(inputs, self.out_shape), self.kernel)
        return keras.ops.matmul(mat, self.selection_matrix)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)

class Choice(tf.keras.layers.Layer):
    def __init__(self, selection_matrix, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = selection_matrix.shape[-1]
        self.selection_matrix = selection_matrix
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                  shape=(input_shape[-1], self.num_outputs),
                                  initializer='glorot_uniform',
                                  trainable=True)

    def call(self, inputs):
        mat = keras.ops.matmul(inputs, self.kernel)
        return keras.ops.matmul(mat, self.selection_matrix)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)
        
class DenseSeparated(tf.keras.layers.Layer):
    def __init__(self, num_outputs, activation=None, **kwargs):  #activation=None
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                    shape = (input_shape[-2], input_shape[-1], self.num_outputs),
                                    initializer='glorot_uniform',
                                    trainable=True)

        self.bias = self.add_weight(name="bias",
                                    shape=(input_shape[-2], self.num_outputs),
                                    initializer='zeros',
                                    trainable=True)

    def call(self, inputs):
        return self.activation(keras.ops.add(keras.ops.matmul(inputs, self.kernel), self.bias))

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        #output_shape[1] = self.num_outputs
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
        self.add(RepeatVector(multiplier))
        for _ in range(self.depth):
            self.add(DenseSeparated(self.f_hid, activation=self.act_func))
        #self.add(Reshape((self.f_hid*multiplier,)))
        self.add(Flatten())
        self.add(Choice(selection_matrix))
        
