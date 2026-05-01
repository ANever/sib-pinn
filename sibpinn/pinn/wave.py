import os
import warnings
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from .base import PINN_BASE

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class WaveBasis(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.expand = lambda inputs: [inputs, tf.math.pow(inputs,2), tf.math.pow(inputs,3), tf.math.cos(inputs), tf.math.cos(2*inputs), tf.math.cos(3*inputs)]
        self.num_of_funcs = len(self.expand(tf.constant(1.)))
        
    def call(self, inputs):
        return tf.keras.layers.concatenate(self.expand(inputs), axis=1)
#, tf.math.cos(3*inputs), tf.math.cos(4*inputs) , tf.math.pow(inputs,3), tf.math.pow(inputs,4)
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
        
        self.postinit()
