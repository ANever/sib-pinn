"""
********************************************************************************
training
********************************************************************************
"""

#from sibpinn import PINN_WAVE, PINN, PINN_SEP
import sibpinn as sp
import yaml
import numpy as np
import pickle as pkl
problem_name = 'simplest-sir-mfg-inverse'#'sir-controlled' #'simplest-sir-mfg'

noise = [0,0.01, 0.05, 0.1, 0.2]
n = len(noise)

if __name__ == "__main__":
    sp.config_gpu(flag=0, verbose=True)
    filename = "./settings/"+problem_name+".yaml"
    for i in 45*np.array([2**i for i in range(10)]):
        noiseless_filename = 'raw_data/colloc_solution_I_'+str(i)+'.pkl'
        with open(noiseless_filename, mode="rb") as datafile:
            data = pkl.load(datafile)
        x = np.array(data['points'])
        for j in range(n):
            with open(filename, mode="r") as file:
                settings = yaml.safe_load(file)
            data_filename = 'raw_data/colloc_solution_I_'+str(i)+'_'+str(j)+'.pkl'
            settings['CONDS']['data']['filename'] = data_filename 
            c = np.array(data['data']) 
            c *= (1 + np.random.normal(0,noise[j],c.shape))
            out_dict = {'points':x, 'data':c}
            print(c.shape)
            with open(data_filename, mode="wb") as file:
                pkl.dump(out_dict, file)
            
            
            model = sp.from_settings(settings, 
                                model_class=sp.PINN_WAVE)
            model.run_training(output_dir="/"+problem_name+str(i)+'_'+str(j))

