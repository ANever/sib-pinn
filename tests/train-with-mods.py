"""
********************************************************************************
training
********************************************************************************
"""

#from sibpinn import PINN_WAVE, PINN, PINN_SEP
import sibpinn as sp
import yaml
import numpy as np
problem_name = 'simplest-sir-mfg-inverse'#'sir-controlled' #'simplest-sir-mfg'

if __name__ == "__main__":
    sp.config_gpu(flag=0, verbose=True)
    filename = "./settings/"+problem_name+".yaml"
    for i in 45*np.array([2**i for i in range(10)]):
        with open(filename, mode="r") as file:
            settings = yaml.safe_load(file)
        settings['CONDS']['data']['filename'] =  'raw_data/colloc_solution_I_'+str(i)+'.pkl'
        model = sp.from_settings(settings, 
                            model_class=sp.PINN_WAVE)
        model.run_training(output_dir="/"+problem_name+str(i))

