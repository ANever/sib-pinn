import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from pinn_base import PINN_BASE

class WaveBasis(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_of_funcs = 3
            
    def call(self, inputs):
        return tf.keras.layers.Concatenate(axis=1)([inputs, tf.math.pow(inputs,2), tf.math.cos(inputs)])

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] *= self.num_of_funcs
        return output_shape

class PINN_WAVE(PINN_BASE):
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
        self.model_name = "SI_pinn"
        
        self.add(WaveBasis())
        for _ in range(self.depth):
            self.add(keras.layers.Dense(self.f_hid, activation=self.act_func))
        self.add(keras.layers.Dense(self.f_out))
        