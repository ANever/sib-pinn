"""
********************************************************************************
training
********************************************************************************
"""

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from sibpinn.config_gpu import config_gpu
from sibpinn.pinn.base import PINN
from sibpinn.pinn.separated import PINN_SEP
from sibpinn.pinn.wave import PINN_WAVE
from sibpinn.utils import from_file

    
def train1d(filename, model_class, **kwargs):
    # read settings
    model = from_file(filename, model_class)
    # run hyperparameters args
    model.run_training(**kwargs)

            
if __name__ == "__main__":
    config_gpu(flag=0, verbose=True)

    settings_file = 'sir-controlled.yaml'
    train1d(filename="./settings/"+settings_file, model_class=PINN_WAVE, output_dir="/simplest-sir")

