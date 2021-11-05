"""
@author: Tim_Janke
"""

import numpy as np
import pandas as pd
import os
import subprocess
import pyvinecopulib as pv
import copy
from statsmodels.distributions.empirical_distribution import ECDF as fit_ecdf


###### functions for generating random vines ####

def random_bicop(family):
    """ sample a copula with random parameters for given family """
    if family == "gaussian":
        return {"family": family, "rotation": 0, "parameters": np.random.choice((-1,1))*np.random.uniform(0.5,0.95,1)}
    
    elif family == "student":
        return {"family": family, "rotation": 0, "parameters": np.array([np.random.choice((-1,1))*np.random.uniform(0.5,0.95), np.random.uniform(2.0, 10.0)])}

    elif any(x==family for x in ["clayton", "gumbel", "frank", "joe"]):
        if family == "frank":
            return {"family": family, "rotation": 0, "parameters": np.random.uniform(10.0, 25.0, 1)}
        else:  
            return {"family": family, "rotation": np.random.choice((0, 90, 180, 270)), "parameters": np.random.uniform(2.0,10.0,1)}    
    elif family == "bb1":
        return {"family": family, "rotation": np.random.choice((0, 90, 180, 270)), "parameters": np.array([np.random.uniform(1.0,5.0), np.random.uniform(1.0, 5.0)])}
    elif family=="bb7":
        return {"family": family, "rotation": np.random.choice((0, 90, 180, 270)), "parameters": np.array([np.random.uniform(1.0,6.0), np.random.uniform(2.0, 20.0)])}
                
    elif family == "indep":
        return {"family": family}
    
    else:
        raise ValueError("Unknown copula family.")


def random_tree(dim, families):
    """ create a tree with random families and random parameters"""
    trees = []
    for d_i in range(dim-1):
        tree_i=[]
        for j in range(dim-d_i-1):
            tree_i.append(random_bicop(np.random.choice(families)))
        trees.append(tree_i)
    return trees


def make_random_vinecopula(dim=3, families=["gaussian", "student", "clayton", "gumbel", "frank", "joe", "bb1", "bb7", "indep"]):
    """ creates a dictionary with info for a random vine copula """
    vine_cop = {}
    #vine_cop["vine_class"] = "rvine"
    vine_cop["structure"] = pv.RVineStructure.simulate(dim)
    vine_cop["pair_copulas"] = get_pvtrees(random_tree(dim, families))
    #vine_cop["d"] = dim
    return vine_cop



def get_pvcopfamily(family):
    """ maps strings to pyvinecopulib bivaraite families """
    if family == "gaussian":
        return pv.BicopFamily.gaussian
    elif family == "student":
        return pv.BicopFamily.student
    elif family == "clayton":
        return pv.BicopFamily.clayton
    elif family == "gumbel":
        return pv.BicopFamily.gumbel
    elif family == "frank":
        return pv.BicopFamily.frank
    elif family == "joe":
        return pv.BicopFamily.joe
    elif family == "bb1":
        return pv.BicopFamily.bb1
    elif family == "bb7":
        return pv.BicopFamily.bb7
    elif family == "indep":
        return pv.BicopFamily.indep
    else:
        raise ValueError("Unknown copula family.")

def get_pvtrees(trees):
    """ creates pyvinecopulib tree list """
    _trees = copy.deepcopy(trees)
    tree_list = []
    for tree_i in _trees:
        cops = []
        for cop_j in tree_i:
            cop_j["family"] = get_pvcopfamily(cop_j["family"])
            cops.append(pv.Bicop(**cop_j))
        tree_list.append(cops)
    return tree_list



def emp_cdf(v, u):
    """ evaluate the empirical copula at points v using the samples u"""
    # cdf is evaluated at points v, v has to be a MxD vector in [0,1]^D, cdf is evaluated at these points
    # u are samples from model NxD vector in [0,1]^D, u should be very large
    # larger u will lead to better estimation of the empirical copula but slows down computation
    cdf_vals = np.empty(shape=(len(v)))
    for i in range(v.shape[0]):
        cdf_vals[i] = np.sum(np.all(u<=v[[i],:], axis=1))
    return cdf_vals/len(u)


def beta_copula_cdf(u_train, u_test, rscript_path):

    # rscript_path is ususally something like "C:/Users/USERNAME/R/R-4.0.4/bin/Rscript.exe"

    # write csv files
    pd.DataFrame(u_train).to_csv("R/_u_train_bicop.csv", header=False, index=False)    
    pd.DataFrame(u_test).to_csv("R/_u_test_bicop.csv", header=False, index=False)
    # run R script
    subprocess.run([rscript_path, "R/_beta_copula_cdf.R"]) # TODO: assumes that R script is in current working directory
    # read results from R script
    cdf_beta = pd.read_csv("R/_cdf_beta.csv", header=None, index_col=False) # TODO: assumes that csv is in current working directory
    
    # remove csv files
    os.remove("R/_u_train_bicop.csv")
    os.remove("R/_u_test_bicop.csv")
    os.remove("R/_cdf_beta.csv")
    
    return np.squeeze(cdf_beta.values)


def gaussian_copula_cdf(u_train, u_test, rscript_path):

    # rscript_path is ususally something like "C:/Users/USERNAME/R/R-4.0.4/bin/Rscript.exe"

    # write csv files
    pd.DataFrame(u_train).to_csv("R/_u_train_gausscop.csv", header=False, index=False)    
    pd.DataFrame(u_test).to_csv("R/_u_test_gausscop.csv", header=False, index=False)
    # run R script
    subprocess.run([rscript_path, "R/_gaussian_copula_cdf.R"]) # TODO: assumes that R script is in current working directory
    # read results from R script
    cdf_gauss = pd.read_csv("R/_cdf_gausscop.csv", header=None, index_col=False) # TODO: assumes that csv is in current working directory
    pdf_gauss = pd.read_csv("R/_pdf_gausscop.csv", header=None, index_col=False) # TODO: assumes that csv is in current working directory


    # remove csv files
    os.remove("R/_u_train_gausscop.csv")
    os.remove("R/_u_test_gausscop.csv")
    os.remove("R/_cdf_gausscop.csv")
    os.remove("R/_pdf_gausscop.csv")

    
    return np.squeeze(cdf_gauss.values), np.squeeze(pdf_gauss.values)



def gaussian_mixture_copula(n_train, n_test, n_sim=100000):
    mu_1 = np.random.uniform(-5,5,2)
    sigma_1 = (np.eye(2)+np.eye(2)[[1,0],:]*np.random.uniform(-0.95,0.95,1))*np.random.uniform(0.8,1.2)
    
    mu_2 = np.random.uniform(-5,5,2)
    sigma_2 = (np.eye(2)+np.eye(2)[[1,0],:]*np.random.uniform(-0.95,0.95,1))*np.random.uniform(0.8,1.2)

    x1 = np.random.multivariate_normal(mean=mu_1, cov=sigma_1, size=n_sim)
    x2 = np.random.multivariate_normal(mean=mu_2, cov=sigma_2, size=n_sim)

    X = np.vstack([x1,x2])
    np.random.shuffle(X)
    cdf_1 = fit_ecdf(X[:,0])
    cdf_2 = fit_ecdf(X[:,1])
    U = np.column_stack((cdf_1(X[:,0]), cdf_2(X[:,1])))
    
    X_train = np.vstack([np.random.multivariate_normal(mean=mu_1, cov=sigma_1, size=n_train),
                         np.random.multivariate_normal(mean=mu_2, cov=sigma_2, size=n_train)])
    np.random.shuffle(X_train)
    U_train = np.column_stack([cdf_1(X_train[:,0]), cdf_2(X_train[:,1])])

    X_test = np.vstack([np.random.multivariate_normal(mean=mu_1, cov=sigma_1, size=n_test),
                        np.random.multivariate_normal(mean=mu_2, cov=sigma_2, size=n_test)])
    np.random.shuffle(X_test)
    U_test = np.column_stack([cdf_1(X_test[:,0]), cdf_2(X_test[:,1])])
    
    cdf_test = emp_cdf(v=U_test, u=U)
    
    params = {"mu_1": mu_1, "sigma_1": sigma_1, "mu_2": mu_2, "sigma_2": sigma_2}
    
    return U_train, U_test, cdf_test, params
