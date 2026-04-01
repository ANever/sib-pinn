"""
********************************************************************************
training
********************************************************************************
"""

#from sibpinn import PINN_WAVE, PINN, PINN_SEP
import sibpinn as sp

problem_name = 'simplest-sir-mfg'            

if __name__ == "__main__":
    sp.config_gpu(flag=0, verbose=True)

    model = sp.from_file(filename="./settings/"+problem_name+".yaml", 
                        model_class=sp.PINN_WAVE)
    model.run_training(output_dir="/"+problem_name)

