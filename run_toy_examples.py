import numpy as np
import pandas as pd
from experiments_utils import run_experiment
from sklearn.datasets import make_swiss_roll
from scipy.stats import multivariate_normal
import os


def run_toy_example(dataset, n_train, n_test):
    
    N = n_train+n_test
    
    if dataset == "swiss_roll":
        data,_= make_swiss_roll(N, noise=0.2)
        data = data[:,[0,2]]
    
    elif dataset == "ring_of_gaussians":
        r=10.0
        data = []
        data.append(multivariate_normal([r, 0], [[1.0, 0.0], [0.0, 1.0]]).rvs(N))
        data.append(multivariate_normal([0, r], [[1.0, 0.0], [0.0, 1.0]]).rvs(N))
        data.append(multivariate_normal([-r, 0], [[1.0, 0.0], [0.0, 1.0]]).rvs(N))
        data.append(multivariate_normal([0, -r], [[1.0, 0.0], [0.0, 1.0]]).rvs(N))
        
        data.append(multivariate_normal([r/np.sqrt(2), r/np.sqrt(2)], [[1.0, 0.0], [0.0, 1.0]]).rvs(N))
        data.append(multivariate_normal([-r/np.sqrt(2), -r/np.sqrt(2)], [[1.0, 0.0], [0.0, 1.0]]).rvs(N))
        data.append(multivariate_normal([r/np.sqrt(2), -r/np.sqrt(2)], [[1.0, 0.0], [0.0, 1.0]]).rvs(N))
        data.append(multivariate_normal([-r/np.sqrt(2), r/np.sqrt(2)], [[1.0, 0.0], [0.0, 1.0]]).rvs(N))
        data = np.vstack(data)
    
    elif dataset == "grid":
        data=[]
        n_blobs = 5
        for i in range(n_blobs):
            for j in range(n_blobs):
                data.append(multivariate_normal([-10+i*20/(n_blobs-1), -10+j*20/(n_blobs-1)], [[1.0/n_blobs, 0], [0, 1.0/n_blobs]]).rvs(N))
        data = np.vstack(data)
    
    else:
        raise ValueError("Unknown data set.")
    
    # prep data
    np.random.shuffle(data)
    data_train = data[0:n_train]
    data_test = data[n_train:n_test+n_train]
    
    # run experiment
    all_scores, _, data_models_v, data_models_y, _, _ = run_experiment(data_train,
                                                                        data_test,
                                                                        evaluate_likelihood=True,
                                                                        GaussCop=True, 
                                                                        VineCop=True, 
                                                                        GMMNCop=True, 
                                                                        GMMNFull=True,
                                                                        GAN = True,
                                                                        IGC=True,
                                                                        options_nn={"n_neurons": 100, "n_layers": 2, "n_samples_train": 200},
                                                                        options_nn_training={"epochs": 500, "batch_size": 100},
                                                                        options_gan={"n_neurons": 100, "n_layers": 2},
                                                                        options_gan_training={"epochs": 500, "batch_size": 100},                                                                                          
                                                                        bw_kde = 0.15)

    return all_scores["NLL_dataspace"], data_models_y, data_models_v, data_train, data_test

N_ROUNDS = 3
N_TRAINS = [5000]
N_TEST = 5000
DATASETS = ["swiss_roll","ring_of_gaussians", "grid"]
PATH = "results\\toy_examples\\"

results = {}
for d, dataset in enumerate(DATASETS):
    for j, n_train in enumerate(N_TRAINS):
        res_i=[]
        for i in range(N_ROUNDS):
            nll, data_y, data_v, _, _ = run_toy_example(dataset=dataset, n_train=n_train, n_test=N_TEST)
            folder = os.path.join(os.getcwd(),PATH+dataset+"\\"+str(n_train)+"\\"+str(i))  
            os.makedirs(folder)
            nll.to_csv(folder+"/nll.csv")
            res_i.append(nll)
            for key_i, d_i in data_y.items():
                np.save(folder+"/"+key_i, d_i)
        results[dataset+str(n_train)] = pd.DataFrame(res_i)
       
results_agg_mean={}
results_agg_median={}
results_agg_std={}

for key, df in results.items():
    df.to_csv("results/toy_examples/"+key+".csv")
    results_agg_mean[key] = np.mean(df,axis=0)
    results_agg_median[key] = np.median(df,axis=0)
    results_agg_std[key] = np.std(df,axis=0)


pd.DataFrame(results_agg_mean).to_csv("results/toy_examples/mean_nll.csv")
pd.DataFrame(results_agg_median).to_csv("results/toy_examples/median_nll.csv")
pd.DataFrame(results_agg_std).to_csv("results/toy_examples/std_nll.csv")