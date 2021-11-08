from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from models import cae
import pandas as pd
import time
from datetime import datetime
from mmd_teststatistics import get_mmdtest_matrix

dataset = "fashion_mnist"
auto_encoder_model_path = "models/autoencoder/autoencoder_"+dataset

N_TRAIN = 60000
N_TEST = 10000
EPOCHS = 100

SAVE_IMAGES = False
N_IMAGES_SAVE = 100

IGC = True
GMMN = True
VINE = True
GAUSS = True
INDEP = True
VAE = True

MMD_RBF = True
SIGMA_IMG = 8.0
SIGMA_U = 1.5
SIGMA_Z = 350.0

#%% Load data

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
x_train = x_train[..., np.newaxis]
x_train = x_train.astype('float32') / 255.
np.random.shuffle(x_train)
x_train = x_train[0:N_TRAIN]

x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')
x_test = x_test[..., np.newaxis]
x_test = x_test.astype('float32') / 255.
np.random.shuffle(x_test)
x_test = x_test[0:N_TEST]


#%% set up models
times={}
models={}

if IGC:
    print("Fitting igc copula model...")
    model_igc = cae.IGCAutoEncoder(x_train, ae_model=auto_encoder_model_path)
    tic = time.time()
    hist = model_igc.fit(epochs=EPOCHS, batch_size=100, n_samples_train=200)
    times["igc"] = time.time()-tic
    models["igc"] = model_igc
    print("Done.\n")


if GMMN:
    print("Fitting gmmn copula model...")
    model_gmmn = cae.GMMNCopulaAutoEncoder(x_train, ae_model=auto_encoder_model_path)
    tic = time.time()
    hist = model_gmmn.fit(epochs=EPOCHS, batch_size=100, n_samples_train=200)
    times["gmmn"] = time.time()-tic
    models["gmmn"] = model_gmmn
    print("Done.\n")


if VINE:
    print("Fitting vine copula model...")
    model_vine = cae.VineCopulaAutoEncoder(x_train, ae_model=auto_encoder_model_path)
    tic = time.time()    
    model_vine.fit()
    times["vine"] = time.time()-tic
    models["vine"] = model_vine
    print("Done.\n")


if GAUSS:
    print("Fitting Gaussian copula model...")
    model_gauss = cae.GaussianCopulaAutoEncoder(x_train, ae_model=auto_encoder_model_path)
    model_gauss.fit()
    models["gauss"] = model_gauss
    print("Done.\n")


if INDEP:
    print("Fitting independence copula model...")
    model_indep = cae.IndependenceCopulaCopulaAutoEncoder(x_train, ae_model=auto_encoder_model_path)
    models["indep"] = model_indep
    print("Done.\n")
    
if VAE:
    print("Loading VAE model...")
    model_vae = cae.VariationalAutoEncoder()
    models["vae"] = model_vae
    print("Done.\n")



#%% generate images
if SAVE_IMAGES:
    print("Saving generated images...")
    for key in models:
        np.save("results/images/"+dataset+"_"+key+".npy", np.squeeze(models[key].sample_images(N_IMAGES_SAVE)))
    print("Done.\n")

#%% sample from models

# obtain samples in code space for test set
mdl_void = cae.IndependenceCopulaCopulaAutoEncoder(x_test, ae_model=auto_encoder_model_path)
u_test = mdl_void.u[0:N_TEST]
z_test = mdl_void.z[0:N_TEST]
del mdl_void


samples_u = {}
samples_z = {}
samples_img = {}

for key in models:
    
    # Copula based models
    if key != "vae":
        # sample from copula in code sapce
        u_tmp = models[key]._sample_u(n_samples=N_TEST)
        samples_u[key] = u_tmp
    
        # sample from data distribution in code space by inverse marginals CDFs
        z_tmp = models[key]._sample_z(u=u_tmp)
        samples_z[key] = z_tmp
    
        # decode samples from code sapce to images
        images_tmp = np.reshape(np.squeeze(models[key].sample_images(z=z_tmp)), (-1,32*32))
        samples_img[key] = images_tmp

        del u_tmp
        del z_tmp
    
    # VAE
    else:
        images_tmp = np.reshape(np.squeeze(models[key].sample_images(N_TEST)), (-1,32*32))
        samples_img[key] = images_tmp
    
    del images_tmp



#%% compute scores

# if bandwidths are not specified they can be automatically set
def get_mmd_bandwidth(X):
    mxx = []
    for i in range(X.shape[0]):
        mxx.append(np.sum(np.square(X[[i],:] - X), axis=1))
    return np.sqrt(np.median(np.concatenate(mxx))/2)

if SIGMA_IMG is None:
    SIGMA_IMG = get_mmd_bandwidth(np.reshape(np.squeeze(x_test[0:5000]), (-1,32*32)))
 
if SIGMA_U is None:
    SIGMA_U = get_mmd_bandwidth(u_test[0:10000])
   
if SIGMA_Z is None:
    SIGMA_Z = get_mmd_bandwidth(z_test[0:10000])


print("Computing scores in image sapce...\n")
pvals_img, mmd_image = get_mmdtest_matrix(samples_true=np.reshape(np.squeeze(x_test[0:N_TEST]), (-1,32*32)), samples_models=samples_img, sigma=SIGMA_IMG, computeMMDs=True)
print("Done.\n")


print("Computing scores in unit code sapce...\n")
pvals_u, mmd_u = get_mmdtest_matrix(samples_true=u_test[0:N_TEST], samples_models=samples_u, sigma=SIGMA_U, computeMMDs=True)
print("Done.\n")


print("Computing scores in data code sapce...\n")
pvals_z, mmd_z = get_mmdtest_matrix(samples_true=z_test[0:N_TEST], samples_models=samples_z, sigma=SIGMA_Z, computeMMDs=True)
print("Done.\n")


#%% save results

print("Writing scores to disk...")
all_scores = pd.DataFrame((mmd_image, mmd_z, mmd_u), index=["mmd_image", "mmd_z", "mmd_u",])
all_scores = all_scores.transpose()

dt_string = datetime.now().strftime("%Y%m%d%H%M%S")
all_scores.to_csv("results/scores_autoencoder_experiments_"+dt_string+".csv")

pd.Series(times, name="time").to_csv("results/fittingtimes_autoencoder_experiments_"+dt_string+".csv")

pvals_img.to_csv("results/pvals_img_autoencoder_experiments_"+dt_string+".csv")
pvals_u.to_csv("results/pvals_cop_autoencoder_experiments_"+dt_string+".csv")
pvals_z.to_csv("results/pvals_latent_autoencoder_experiments_"+dt_string+".csv")

print("All done.")