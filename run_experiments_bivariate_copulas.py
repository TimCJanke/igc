import pyvinecopulib as pv
import numpy as np
import pandas as pd
from experiments_utils import random_bicop, get_pvcopfamily, beta_copula_cdf, emp_cdf, gaussian_mixture_copula
from models.igc import ImplicitGenerativeCopula
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

### set options ###
N_TRAIN = 1000
N_TEST = 5000
N_ROUNDS = 25

FAMILIES = ["student", "clayton", "gumbel", "gaussian_mixture"]
L2_SCORES={}

### run experiments ###
LOG={}
for fam in FAMILIES:
    print(fam)
    l2=[]
    logs_fam = {"true": [], "param": [], "param_select": []}
    for i in range(N_ROUNDS):
        cdf_vals={}
        
        if fam != "gaussian_mixture":
        # true copula
            params = random_bicop(fam)
            logs_fam["true"].append(params)
            params["family"] = get_pvcopfamily(fam)
            copula = pv.Bicop(**params)
            u_train = copula.simulate(N_TRAIN)
            U_TEST = copula.simulate(N_TEST)
            cdf_vals_true = np.reshape(copula.cdf(U_TEST), (-1,1))

        else:
            u_train, U_TEST, cdf_vals_true, params = gaussian_mixture_copula(n_train=N_TRAIN, n_test=N_TEST, n_sim=200000)
            cdf_vals_true = np.reshape(cdf_vals_true, (-1,1))
            logs_fam["true"].append(params)
            
            
        if fam != "gaussian_mixture":
            
            # fit parametric copula        
            controls_param = pv.FitControlsBicop(family_set=[get_pvcopfamily(fam)], selection_criterion="bic")
            cop_param = pv.Bicop()
            cop_param.select(u_train, controls=controls_param)
            
            logs_fam["param"].append(cop_param.str())
            cdf_vals["param"] = cop_param.cdf(U_TEST)
        
        
            # fit parametric copula without knowing the correct family
            controls_param_select = pv.FitControlsBicop(family_set=[pv.BicopFamily.indep, 
                                                                    pv.BicopFamily.gaussian, 
                                                                    pv.BicopFamily.student, 
                                                                    pv.BicopFamily.clayton, 
                                                                    pv.BicopFamily.gumbel, 
                                                                    pv.BicopFamily.frank, 
                                                                    pv.BicopFamily.joe, 
                                                                    pv.BicopFamily.bb1, 
                                                                    pv.BicopFamily.bb6, 
                                                                    pv.BicopFamily.bb7, 
                                                                    pv.BicopFamily.bb8],
                                                                    selection_criterion="bic")
            cop_param_select = pv.Bicop()
            cop_param_select.select(u_train, controls=controls_param_select)
            logs_fam["param_select"].append(cop_param_select.str())
            cdf_vals["param_select"] = cop_param_select.cdf(U_TEST)


        # fit non-parametric beta copula
        cdf_vals["beta"] = beta_copula_cdf(u_train=u_train, u_test=U_TEST)

        # fit non-parametric copula
        controls_tll1 = pv.FitControlsBicop(nonparametric_method="linear", family_set=[pv.BicopFamily.tll])
        cop_tll1 = pv.Bicop()
        cop_tll1.select(u_train, controls=controls_tll1)
        cdf_vals["tll1"] = emp_cdf(U_TEST, cop_tll1.simulate(200000))

        # fit non-parametric copula
        controls_tll2 = pv.FitControlsBicop(nonparametric_method="quadratic", family_set=[pv.BicopFamily.tll])
        cop_tll2 = pv.Bicop()
        cop_tll2.select(u_train, controls=controls_tll2)
        cdf_vals["tll2"] = emp_cdf(U_TEST, cop_tll2.simulate(200000))

        
        # # fit IGC copula
        cop_igc = ImplicitGenerativeCopula(dim_latent=6, dim_out=2, n_samples_train=200, n_layers=2, n_neurons=100)           
        hist=cop_igc.fit(u_train, batch_size=100, epochs=500)
        cdf_vals["igc"] = cop_igc.cdf(v=U_TEST, n=200000)
        
        errors = cdf_vals_true-pd.DataFrame(cdf_vals)
        l2.append(errors.pow(2).sum(axis=0))
    
    L2_SCORES[fam] = pd.DataFrame(l2)
    LOG[fam] = logs_fam


for fam,scores in L2_SCORES.items():
    fix,ax = plt.subplots()
    ax = sns.boxplot(data=scores)
    ax.set_title(f"L2 score for {fam}")
    #scores.to_csv("results/bivariate_L2_"+fam+".csv")

# def save_obj(obj, name ):
#     with open('logs/'+ name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
# for fam, log in LOG.items():
#     save_obj(log, "logs_"+fam+"_"+dt_string)
