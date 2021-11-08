"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt

"""

import pyvinecopulib as pv
import numpy as np
import pandas as pd
from experiments_utils import make_random_vinecopula, emp_cdf
from models.igc import ImplicitGenerativeCopula
from models.mv_copulas import GaussianCopula
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def run_vinecop_experiment(vine_structure, vc_from_json=False, write_log=False, n_train=5000, n_test=50000, tll_vcop=True, param_vcop=True, gaussian_cop=True, igc_cop=True):
    
    """ Experiments for estimating bivariate parametric copulas with non-parametric methods """
    
    ### set up vine ###
    #cop = pv.Vinecop(vine_structure, get_pvtrees(trees))
    if vc_from_json:
        cop = pv.Vinecop(vine_structure)
    else:
        cop = pv.Vinecop(**vine_structure)
    
    if write_log:
        dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
        cop.to_json("logs/vine_cops/vine_copula"+dt_string+".json")
    
    print("\nTrue copula:\n")    
    print(cop)
    
    ### simulate training and test data ###
    u_train = cop.simulate(n_train)
    u_test = cop.simulate(n_test)
    cdf_true = cop.cdf(u_test)
    
    L2_scores = {}
    L1_scores = {}


    #### vine copula ###
    if param_vcop:
        cop_param = pv.Vinecop(data=u_train)
        print("\n\nParametric vine fit:\n")
        print(cop_param)
                
        cdf_param =  cop_param.cdf(u_test, N=100000)
        L1_scores["param"] = np.sum(np.abs(cdf_true-cdf_param))
        L2_scores["param"] = np.sum(np.square(cdf_true-cdf_param))
    

    #### non-parametric tll2 vine copula ###
    if tll_vcop:
        controls_tll = pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll])
        cop_tll = pv.Vinecop(data=u_train, controls=controls_tll)
        print("\n\nTLL vine fit:\n")
        print(cop_tll)
                
        cdf_tll =  cop_tll.cdf(u_test, N=100000)
        L1_scores["tll"] = np.sum(np.abs(cdf_true-cdf_tll))
        L2_scores["tll"] = np.sum(np.square(cdf_true-cdf_tll))


    #### Gaussian copula ###
    if gaussian_cop:
        cop_gauss = GaussianCopula()
        cop_gauss.fit(u_train)
        u_cop_gauss = cop_gauss.simulate(100000)
        cdf_gauss = emp_cdf(v=u_test, u=u_cop_gauss)
        L1_scores["gauss"] = np.sum(np.abs(cdf_true-cdf_gauss))
        L2_scores["gauss"] = np.sum(np.square(cdf_true-cdf_gauss))


    ### fit igc copula ###
    if igc_cop:
        cop_igc = ImplicitGenerativeCopula(dim_latent=u_train.shape[1]*3, dim_out=u_train.shape[1], n_samples_train=200, n_layers=2, n_neurons=100)                     
        hist = cop_igc.fit(u_train, batch_size=100, epochs=500)
        #hist.plot()
        
        cdf_igc = cop_igc.cdf(v=u_test, n=100000)
        L1_scores["igc"] = np.sum(np.abs(cdf_true-cdf_igc))
        L2_scores["igc"] = np.sum(np.square(cdf_true-cdf_igc))


    return L1_scores, L2_scores



### run experiments for synthetic data from vine copulas ###
N_ROUNDS = 25
DIM = 5
N_TRAIN = 5000
N_TEST = 10000

L1_SCORES = []
L2_SCORES = []

from_logs = False # if True load vine copula from log file, if False generate random vine 

path = "logs/vine_cops/"
vc_list = os.listdir(path)

for i in range(N_ROUNDS):  
    
    
    if from_logs:
        vc = path+vc_list[i]
    else:
        vc = make_random_vinecopula(dim=DIM)
    
    l1, l2 = run_vinecop_experiment(vc, 
                                    vc_from_json=from_logs, 
                                    n_train=N_TRAIN, 
                                    n_test=N_TEST, 
                                    write_log=False, 
                                    tll_vcop=True, 
                                    param_vcop=True, 
                                    gaussian_cop=True, 
                                    igc_cop=True)
    
    L1_SCORES.append(pd.Series(l1))
    L2_SCORES.append(pd.Series(l2))

    L1 = pd.DataFrame(L1_SCORES)
    #L1.to_csv("results/vines/L1.csv")
    
    L2 = pd.DataFrame(L2_SCORES)
    #L2.to_csv("results/vines/L2.csv")


# fig,ax=plt.subplots()
# sns.boxplot(data=L1, ax=ax)
# fig.suptitle("L1 Goodness of Fit")

# fig,ax=plt.subplots()
# sns.boxplot(data=L2, ax=ax)
# fig.suptitle("L2 Goodness of Fit")
