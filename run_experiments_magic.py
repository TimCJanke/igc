import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import KFold
from experiments_utils import gaussian_copula_cdf, emp_cdf
from models.igc import ImplicitGenerativeCopula
import pyvinecopulib as pv
from models.mv_copulas import GaussianCopula


data = pd.read_csv("data/magic.csv", index_col=False)
data = data.loc[:,['fLength', 'Width', 'fConc', 'fM3Long','fM3Trans', 'class']]
data = data.dropna(axis=0, how="any")
data = data[data["class"]=="g"]
data = data.drop(("class"), axis=1)

pobs = {}
for i in range(data.shape[1]):
    ticker = data.columns[i]
    series = data.iloc[:,i].values
    pobs[ticker] = rankdata(data.iloc[:,i].values)/(len(data.iloc[:,i].values)+1)
pobs = pd.DataFrame(pobs)


L2 = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(pobs.values):
    
    L2_scores={}
    pobs_train = pobs.values[test_index,:]
    pobs_test = pobs.values[train_index,:]
    cdf_test = emp_cdf(pobs_test, pobs_test)


    # Gaussian Copula
    cop_gauss = GaussianCopula()
    cop_gauss.fit(pobs_train)
    u_cop_gauss = cop_gauss.simulate(100000)
    cdf_vals_gauss = emp_cdf(v=pobs_test, u=u_cop_gauss)
    L2_scores["gauss_2"] = np.sum(np.square(cdf_test-cdf_vals_gauss))
    

    # Vine Copula (TLL2)
    controls_tll = pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll])
    cop_tll = pv.Vinecop(data=pobs_train, controls=controls_tll)
    print("\n\nTLL vine fit:\n")
    print(cop_tll)   
    cdf_tll =  cop_tll.cdf(pobs_test, N=100000)
    L2_scores["tll"] = np.sum(np.square(cdf_test-cdf_tll))


    # Vine Copula
    cop_param = pv.Vinecop(data=pobs_train)
    print("\n\nParam vine fit:\n")
    print(cop_param)   
    cdf_param =  cop_param.cdf(pobs_test, N=100000)
    L2_scores["tll_param"] = np.sum(np.square(cdf_test-cdf_param))


    # IGC model
    pobs_train_igc = pobs_train[0:int(np.floor(len(pobs_train)/100)*100)] #make training data dividebale by 100
    cop_igc = ImplicitGenerativeCopula(dim_latent=pobs_train.shape[1]*3, dim_out=pobs_train.shape[1], n_samples_train=200, n_layers=2, n_neurons=100)           
    hist=cop_igc.fit(pobs_train_igc, batch_size=100, epochs=500)
    #hist.plot()
    cdf_igc = cop_igc.cdf(v=pobs_test, n=100000)
    L2_scores["igc"] = np.sum(np.square(cdf_test-cdf_igc))

    L2.append(pd.Series(L2_scores))


L2 = pd.DataFrame(L2)
# L2.to_csv("results/L2_magic.csv")


