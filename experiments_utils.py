"""
@author: Tim_Janke
"""

import numpy as np
import pandas as pd
#from helpers import emp_cdf
from models.igc import ImplicitGenerativeCopula, GMMNCopula, GenerativeMomentMatchingNetwork, GenerativeAdversarialNetwork
from models.mv_copulas import GaussianCopula
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from models.utils import cdf_interpolator
import os
import subprocess
import pyvinecopulib as pv
import copy
from statsmodels.distributions.empirical_distribution import ECDF as fit_ecdf


def run_experiment(data_train,
                    data_test,
                    margins_model="ECDF", # margins model for copula based methods
                    evaluate_cdf=False,   # evaluate copula by ISE and IAE
                    evaluate_ed_cop=False, # evaluate copula distribution via energy distance
                    evaluate_ed_data=False, # evaluate data distribution via energy distance
                    evaluate_likelihood = False, # evaluate data distribution via KDE LogLik
                    n_eval = int(1e5),  # number of points to generate for evaluation
                    IndepCop=False, GaussCop=False, VineCop=False, GMMNCop=False, GMMNFull=False, IGC=True, GAN=False, # models to use (GMMNFull and GAN operate on data distribution)
                    options_nn={"n_neurons": 100, "n_layers": 2, "n_samples_train": 200}, # options for NN architecture (for GMMN and IGC)
                    options_nn_training={"epochs": 500, "batch_size": 100}, # options for NN architecture (for GMMN and IGC)
                    options_gan={"n_neurons": 100, "n_layers": 2}, # options for GAN architecture
                    options_gan_training={"epochs": 500, "batch_size": 100}, # options for GAN architecture
                    bw_kde = None, # bandwidth for KDE
                    fit_margins_on_train_and_test=False # if True margins will be fit an compete data set, otherwise only traing data
                    ):

    """ Evaluate models for given test and training data set."""

    print("Training models ...")
    
    
    if fit_margins_on_train_and_test:
        data_test_margins= data_test
    else:
        data_test_margins = None

    models_joint, models_margins = fit_models(data_train=data_train,
                                                data_test=data_test_margins, # only used for fitting marginal models if not None
                                                margins_model=margins_model, 
                                                IndepCop=IndepCop, 
                                                GaussCop=GaussCop, 
                                                VineCop=VineCop, 
                                                GMMNCop=GMMNCop, 
                                                GMMNFull=GMMNFull,
                                                GAN=GAN,
                                                IGC=IGC,
                                                options_nn=options_nn,
                                                options_nn_training=options_nn_training,
                                                options_gan=options_gan,
                                                options_gan_training=options_gan_training)
    print("Done.\n")

    print("Sampling data for evalutation...")
    data_models_v, data_models_y = make_data_eval(models_joint, models_margins, n=n_eval)
    print("Done.\n")

    print("Computing evaluation metrics...")
    all_scores = pd.DataFrame(index=models_joint.keys())

    if evaluate_cdf:
        print("ISE and IAE...")
        pobs_test = []
        for i in range(data_test.shape[1]):
            pobs_test.append(models_margins[i].cdf(data_test[:,i]))
        pobs_test = np.column_stack(pobs_test)

        cdf_vals_test = emp_cdf(pobs_test,pobs_test)
        cdf_vals_models = {}
        for key_i, v_i in data_models_v.items():
            cdf_vals_models[key_i] = emp_cdf(v=pobs_test, u=v_i)

        ise, iae = eval_copula_cdf(cdf_vals_models, cdf_vals_test)
        all_scores["ise"] = pd.Series(ise)
        all_scores["iae"] = pd.Series(iae)


    if evaluate_ed_cop:
        print("ED unit space...")
        ed = eval_energy_distance(data_models_v, data_test, standardize=False)
        all_scores["ED_unitspace"] = pd.Series(ed)

    if evaluate_ed_data:
        print("ED data space...")
        ed = eval_energy_distance(data_models_y, data_test, standardize=True)
        all_scores["ED_dataspace"] = pd.Series(ed)

    if evaluate_likelihood:
        print("LogLikelihood data space...")
        ll = eval_likelihood(data_models_y, data_test, bw=bw_kde)
        all_scores["NLL_dataspace"] = pd.Series(ll)        

    print("All done.\n")

    return all_scores, data_models_v, data_models_y, models_joint, models_margins


class IdentityMargins(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    def cdf(self, x):
        return x

    def icdf(self, tau):
        return self.ppf(tau)

    def ppf(self, tau):
        return tau



def fit_models(data_train, data_test=None, margins_model="ECDF", IndepCop=False, GaussCop=False, VineCop=True, GMMNCop=True, GMMNFull=False, IGC=True, GAN=False,
                    options_nn={"n_neurons": 100, "n_layers": 2, "n_samples_train": 200},
                    options_nn_training={"epochs": 500, "batch_size": 100},
                    options_gan={"n_neurons": 50, "n_layers": 1},
                    options_gan_training={"epochs": 100, "batch_size": 100},
                    ):
    
    """ Trains models for given data set"""
    
    models_margins=[]
    pobs_train=[]
    for i in range(data_train.shape[1]):
        
        if data_test is not None:
            data_train_i = np.concatenate((data_train[:,i], data_test[:,i]), axis=0)
        else:
            data_train_i = data_train[:,i]

        if margins_model == "identity":
            # if margins are assumed to be uniform already
            mdl_i = IdentityMargins()

        elif margins_model == "ECDF":
            #linear interpolation of empirical marginal CDFs
            mdl_i = cdf_interpolator(data_train_i, 
                                        kind="linear", 
                                        x_min=np.min(data_train_i)-np.diff(np.sort(data_train_i)[0:2])[0], 
                                        x_max=np.max(data_train_i)+np.diff(np.sort(data_train_i)[-2:])[0])
        
        # elif margins_model == "KDE":
        #     mdl_i = sm.nonparametric.KDEUnivariate(data_train_i)
        #     mdl_i.fit()

        else:
            raise ValueError("Unknown model type for margins.")

        models_margins.append(mdl_i)
        pobs_train.append(mdl_i.cdf(data_train[:,i]))
    
    pobs_train = np.column_stack(pobs_train)

    models_joint = {}

    # Independence Copula
    if IndepCop:
        models_joint["indep"] = None

    # Gaussian Copula
    if GaussCop:
        cop_gauss = GaussianCopula()
        cop_gauss.fit(pobs_train)
        models_joint["gauss"] = cop_gauss


    # Vine Copula (TLL2)
    if VineCop:
        controls_tll = pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll])
        cop_tll = pv.Vinecop(data=pobs_train, controls=controls_tll)
        models_joint["vine_tll2"] = cop_tll

    pobs_train_nn = pobs_train[0:int(np.floor(len(pobs_train)/options_nn_training["batch_size"])*options_nn_training["batch_size"])] #make training data dividebale by 100
    
    # IGC model
    if IGC:
        cop_igc = ImplicitGenerativeCopula(dim_latent=pobs_train.shape[1]*3, dim_out=pobs_train.shape[1], **options_nn)
        cop_igc.fit(pobs_train_nn, **options_nn_training)
        models_joint["igc"] = cop_igc

    # GMMN copula
    if GMMNCop:
        cop_gmmn = GMMNCopula(dim_latent=pobs_train.shape[1]*3, dim_out=pobs_train.shape[1], **options_nn)           
        cop_gmmn.fit(pobs_train_nn, **options_nn_training)
        models_joint["gmmn_cop"] = cop_gmmn


    # GMMN with ED loss (models both margins and joint at once)
    if GMMNFull:
        gmmn = GenerativeMomentMatchingNetwork(dim_latent=pobs_train.shape[1]*3, dim_out=pobs_train.shape[1], **options_nn)           
        gmmn.fit(data_train, **options_nn_training)
        models_joint["gmmn_full"] = gmmn

    # GAN (models both margins and joint at once)
    if GAN:
        gan = GenerativeAdversarialNetwork(dim_latent=pobs_train.shape[1]*3, dim_out=pobs_train.shape[1], **options_gan)
        gan.fit(data_train, **options_gan_training)
        models_joint["gan"] = gan

    return models_joint, models_margins



def make_data_eval(models_joint, models_margins, n=int(1e5)):
    data_v = {}
    data_y = {}

    # generate samples in unit space
    for key_i, mdl_i in models_joint.items(): 
        if key_i == "gmmn_cop":
            data_v[key_i] = mdl_i.simulate(n) 
        elif key_i == "gmmn_full":
            data_v[key_i] = None
        elif key_i == "gan":
            data_v[key_i] = None
        elif key_i == "indep":
            data_v[key_i] = np.random.uniform(0.0,1.0,size=(n, len(models_margins)))
        else:
            data_v[key_i] = mdl_i.simulate(n)

    # obtain samples in data space by transforming samples componentwise via the inverse cdf
    for key_i, v_i in data_v.items():
        
        if key_i == "gmmn_full":
            data_y[key_i] = models_joint["gmmn_full"].simulate(n)
            data_v[key_i] = models_joint["gmmn_full"]._to_pobs(data_y[key_i])
        elif key_i == "gan":
            data_y[key_i] = models_joint["gan"].simulate(n)
            data_v[key_i] = models_joint["gan"]._to_pobs(data_y[key_i])        
        else:
            y=[]
            for j, mdl_j in enumerate(models_margins):
                y.append(mdl_j.icdf(v_i[:,j]))
            data_y[key_i] = np.column_stack(y)

    return data_v, data_y
    

def eval_copula_cdf(cdf_vals_models, cdf_vals_test):
    # compute ISE and IAE from cdf values
    ise = {}
    iae = {}
    for key_i, cdf_i in cdf_vals_models.items():
            eps = cdf_vals_test-cdf_i
            iae[key_i] = np.sum(np.abs(eps))
            ise[key_i] = np.sum(np.square(eps))
    return ise, iae




def eval_likelihood(data_models, data_test, bw, n_eval=10000):
    # evaluate likelihood of test data under KDE based likelihood from trained models
    nll={}
    if bw is None:
        grid_cv = GridSearchCV(KernelDensity(), param_grid={"bandwidth": np.logspace(-1.0,1.0,10)}) # use CV to find best bandwidth on the test data
        grid_cv.fit(data_test)
        bw_opt = grid_cv.best_params_["bandwidth"]
        print(bw_opt)

    elif isinstance(bw, (list, tuple, np.ndarray)):
        grid_cv = GridSearchCV(KernelDensity(), param_grid={"bandwidth": bw}) # use CV to find best bandwidth on the test data
        grid_cv.fit(data_test)
        bw_opt = grid_cv.best_params_["bandwidth"]
        print(bw_opt)        

    elif isinstance(bw, float):
        bw_opt = bw

    for key_i, y_i in data_models.items():
        kde_model = KernelDensity(bandwidth=bw_opt).fit(y_i)
        nll[key_i] = -np.mean(kde_model.score_samples(data_test[0:n_eval])) # compute likelihood of test data under KDE
    return nll



def eval_energy_distance(data_models, data_test, standardize=False, n_eval=int(5e3)):

    if standardize:
        means = np.expand_dims(np.mean(data_test, axis=0),0)
        stds = np.expand_dims(np.std(data_test, axis=0),0)
    else:
        means = np.zeros((1, data_test.shape[1]))   
        stds = np.ones((1, data_test.shape[1]))

    ed_df={}
    for key_i, y_i in data_models.items():
        ed_df[key_i] = energy_distance(X=(data_test[0:n_eval,:]-means)/stds, Y=(y_i[0:n_eval,:]-means)/stds)

    return ed_df


def energy_distance(X,Y):
    # X,Y are shape NxD with N samples and D dimensions
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    X = np.expand_dims(X.T,0) # (N_x,D) --> (1,D,N_x)
    Y = np.expand_dims(Y.T,0) # (N_y,D) --> (1,D,N_y)

    ed_xx = np.sum(np.sqrt(np.sum(np.square(X - np.repeat(np.transpose(X, axes=(2,1,0)), repeats=n_x, axis=2)), axis=1)))
    ed_yy = np.sum(np.sqrt(np.sum(np.square(Y - np.repeat(np.transpose(Y, axes=(2,1,0)), repeats=n_y, axis=2)), axis=1)))
    ed_xy = np.sum(np.sqrt(np.sum(np.square(Y - np.repeat(np.transpose(X, axes=(2,1,0)), repeats=n_y, axis=2)), axis=1)))
    
    return 2*ed_xy/(n_x*n_y) -  ed_yy/(n_y*(n_y-1)) - ed_xx/(n_y*(n_y-1))



###### functions for generating random copulas and vines ####

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


### more helper functions ###

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
