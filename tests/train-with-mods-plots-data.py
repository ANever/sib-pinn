"""
********************************************************************************
training
********************************************************************************
"""

#from sibpinn import PINN_WAVE, PINN, PINN_SEP
#import sibpinn as sp
from clspde.prepare import from_file
import yaml
import numpy as np
import pickle as pkl

problem_name = 'simplest-sir-mfg-inverse'#'sir-controlled' #'simplest-sir-mfg'

noise = [0,0.01, 0.05, 0.1, 0.2]
n = len(noise)

import pandas as pd

if __name__ == "__main__":
    #out_dict = {'coefs':sol.cells_coefs}
    settings_filename = "settings/cls_simplest_mfg.yaml"
    settings, sol, iteration_dict = from_file(settings_filename)

    with open('colloc_solution_coefs.pkl', 'rb') as out_file:
        coefs_dict = pkl.load(out_file)
        sol.cells_coefs = coefs_dict['coefs']
    #sp.config_gpu(flag=0, verbose=True)
    #filename = "./settings/"+problem_name+".yaml"
    names = ['S', 'I', 'uS', 'uI', 'beta_max']
    error_tab = np.zeros((len(names), 10, n))
    for ii, i in enumerate(45*np.array([2**i for i in range(10)])):
        for j in range(n):
            print('\n')
            output_dir="results/"+problem_name+str(i)+'_'+str(j)
            for k, var in enumerate(names):
                with open(output_dir+'/data_'+var+'__best.txt', 'rb') as data_loc:
                    data_var = pkl.load(data_loc)
                error = 0
                for l in range(len(data_var[1])):
                    try:
                        real_dat = sol.eval(data_var[0][l], np.zeros(1), k)
                    except:
                        real_dat = 20
                    error += np.abs(data_var[1][l] - real_dat)/(real_dat+1e-7)
                error/= len(data_var[1])
                print(error)
                error_tab[k,ii,j] = error
            var = 'beta_max'
            #with open(output_dir+'/data_'+var+'__best.txt', 'rb') as data_loc:
            #    data_var = pkl.load(data_loc)
            #    error = abs(data_var[1][0] - 20)/20
            #    #error to output
            #print(error)
            #error_tab[-1,i,j] = error
    for i, name in enumerate(names):
        df = pd.DataFrame(error_tab[i])
        df['x'] = 50*np.array([2**i for i in range(10)])
        df.to_csv('err_'+name+'.csv')
