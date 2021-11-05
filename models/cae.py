"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt
"""

import numpy as np
import tensorflow as tf
from models.igc import ImplicitGenerativeCopula, GMMNCopula
from models.utils import cdf_interpolator
import pyvinecopulib as pv
from models import mv_copulas
import matplotlib.pyplot as plt


class CopulaAutoEncoder(object):
    def __init__(self, x, ae_model):
        if isinstance(ae_model, str):
            ae_model = tf.keras.models.load_model(ae_model)
        self.encoder_model = ae_model.encoder
        self.decoder_model = ae_model.decoder
        self.z = self._encode(x)
        self.margins = self._fit_margins(self.z)
        self.u = self._cdf(self.z)

        # if x_test is not None:
        #     self.u_test = self._cdf((self._encode(x_test)))
        # else:
        #     self.u_test = None
        
    def _encode(self, x):
        # encode images to latent space
        return self.encoder_model(x).numpy()


    def _decode(self, z):
        # decode latent space samples to images
        return self.decoder_model(z).numpy()
    
    
    def _cdf(self, z):
        # get pseudo obs
        u = np.zeros_like(z)
        for i in range(u.shape[1]):
            u[:,i] = self.margins[i].cdf(z[:,i])
        return u
        
    
    def _ppf(self, u):
        # inverse marginal cdf
        z = np.zeros_like(u)
        for i in range(z.shape[1]):
            z[:,i] = self.margins[i].ppf(u[:,i])
        return z
    
    
    def _fit_margins(self, z):
        # get the marginal distributions via ecdf interpolation
        margins = []
        for i in range(z.shape[1]):
            margins.append(cdf_interpolator(z[:,i], 
                                            kind="linear", 
                                            x_min=np.min(z[:,i])-np.diff(np.sort(z[:,i])[0:2])[0], 
                                            x_max=np.max(z[:,i])+np.diff(np.sort(z[:,i])[-2:])[0]))
        return margins      


    def _sample_u(self, n_samples):
        # sample from copula
        return self.copula.simulate(n_samples)


    def _sample_z(self, n_samples):
        # sample from latent space
        return self._ppf(self._sample_u(n_samples))
        
        
    def sample_images(self, n_samples):
        # sample an image
        return self._decode(self._sample_z(n_samples))

    def show_images(self, n=5, imgs=None, cmap="gray", title=None):
        if imgs is None:
            imgs = self.sample_images(n)
        
        plt.figure(figsize=(16, 3))
        for i in range(n):
            ax = plt.subplot(1, n, i+1)
            plt.imshow(np.squeeze(imgs[i]*255), vmin=0, vmax=255, cmap=cmap)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.suptitle(title)
        plt.tight_layout()



class IGCAutoEncoder(CopulaAutoEncoder):
    """ Copula Auto Encoder with Implicit Generative Copula """

    def fit(self, epochs=100, batch_size=100, n_samples_train=200, regen_noise=1000000, validation_split=0.0, validation_data=None): 
        if validation_data is not None:
            u_test = self._cdf((self._encode(validation_data)))
        else:
            u_test = None

        #self.copula = ImplicitGenerativeCopula(dim_out = self.z.shape[1], n_samples_train=n_samples_train, dim_latent=self.z.shape[1]*2)
        self.copula = ImplicitGenerativeCopula(dim_out = self.z.shape[1], n_samples_train=n_samples_train, dim_latent=self.z.shape[1]*2, n_layers=3, n_neurons=200)

        hist = self.copula.fit(self.u, epochs=epochs, batch_size=batch_size, validation_data=u_test, regen_noise=regen_noise, validation_split=0.0)
        return hist


    def save_copula_model(self, path):
        self.copula.save_model(path)


    def load_copula_model(self, path, n_samples_train=200):
        self.copula = ImplicitGenerativeCopula(dim_out = self.z.shape[1], n_samples_train=n_samples_train, dim_latent=self.z.shape[1]*2)
        self.copula.load_model(path)
        print("Loaded saved copula model.")



class GMMNCopulaAutoEncoder(CopulaAutoEncoder):
    """ Copula Auto Encoder with GMMN Copula """

    def fit(self, epochs=100, batch_size=100, n_samples_train=200, regen_noise=10000000, validation_split=0.0, validation_data=None): 
        if validation_data is not None:
            u_test = self._cdf((self._encode(validation_data)))
        else:
            u_test = None

        #self.copula = GMMNCopula(dim_out = self.z.shape[1], n_samples_train=n_samples_train, dim_latent=self.z.shape[1]*2)
        self.copula = GMMNCopula(dim_out = self.z.shape[1], n_samples_train=n_samples_train, dim_latent=self.z.shape[1]*2, n_layers=3, n_neurons=200)

        hist = self.copula.fit(self.u, epochs=epochs, batch_size=batch_size, validation_data=u_test, regen_noise=regen_noise, validation_split=0.0)
        return hist


    def _sample_u(self, n_samples, normalize_marginals=False):
        # sample from copula
        return self.copula.simulate(n_samples, normalize_marginals=normalize_marginals)


    def _sample_z(self, n_samples, normalize_marginals=False):
        # sample from latent space
        return self._ppf(self._sample_u(n_samples, normalize_marginals=normalize_marginals))
        

    def sample_images(self, n_samples, normalize_marginals=False):
        # sample an image
        return self._decode(self._sample_z(n_samples, normalize_marginals=normalize_marginals))


    def show_images(self, n=5, imgs=None, cmap="gray", title=None, normalize_marginals=False):
        if imgs is None:
            imgs = self.sample_images(n, normalize_marginals=normalize_marginals)
        
        plt.figure(figsize=(16, 3))
        for i in range(n):
            ax = plt.subplot(1, n, i+1)
            plt.imshow(np.squeeze(imgs[i]*255), vmin=0, vmax=255, cmap=cmap)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.suptitle(title)
        plt.tight_layout()


    def save_copula_model(self, path):
        self.copula.save_model(path)


    def load_copula_model(self, path, n_samples_train=200):
        self.copula = GMMNCopula(dim_out = self.z.shape[1], n_samples_train=n_samples_train, dim_latent=self.z.shape[1]*2)
        self.copula.load_model(path)
        print("Loaded saved copula model.")



class VineCopulaAutoEncoder(CopulaAutoEncoder):
    """ Copula Auto Encoder with Vine Copula """

    def fit(self, families="nonparametric", show_trace=False, trunc_lvl=18446744073709551615):
        if families == "nonparametric":
            controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll], trunc_lvl=trunc_lvl, show_trace=show_trace)
        elif families == "parametric":
            controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.indep, 
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
                                                         trunc_lvl=trunc_lvl,
                                                         show_trace=show_trace)
        else:
            controls = pv.FitControlsVinecop(trunc_lvl=trunc_lvl, show_trace=show_trace)

        self.copula = pv.Vinecop(data=self.u, controls=controls)

    def save_model(self, path):
        self.copula.to_json(path)
        print(f"Saved vine copula model to {path}.")

    def load_model(self, path):
        self.copula = pv.Vinecop(filename=path)
        print("Loaded vine copula model.")




class GaussianCopulaAutoEncoder(CopulaAutoEncoder):
    """ Copula Auto Encoder with Gaussian Copula """
    def fit(self):
        self.copula = mv_copulas.GaussianCopula()
        self.copula.fit(self.u)



class IndependenceCopulaCopulaAutoEncoder(CopulaAutoEncoder):
    """ Copula Auto Encoder with Independence Copula """
    def fit(self):
        pass

    def _sample_u(self, n_samples):
        return np.random.uniform(0.0, 1.0, size=(n_samples, self.u.shape[1]))


class VariationalAutoEncoder(object):
    def __init__(self, decoder_model="models/autoencoder/VAE_decoder_fashion_mnist_100epochs", latent_dim=25):
        if isinstance(decoder_model, str):
            self.decoder_model = tf.keras.models.load_model(decoder_model) 
        else:       
            self.decoder_model = decoder_model
        self.decoder_model.compile()
        self.latent_dim = 25

    def _sample_z(self, n_samples):
        # sample from latent space
        return np.random.normal(loc=0.0, scale=1.0, size=(n_samples, self.latent_dim))
    
    def _decode(self,z):
        return self.decoder_model.predict(z)

    def fit(self):
        pass

    def sample_images(self, n_samples):
        # sample an image
        return self._decode(self._sample_z(n_samples))

    def show_images(self, n=5, imgs=None, cmap="gray", title=None):
        if imgs is None:
            imgs = self.sample_images(n)
        
        plt.figure(figsize=(16, 3))
        for i in range(n):
            ax = plt.subplot(1, n, i+1)
            plt.imshow(np.squeeze(imgs[i]*255), vmin=0, vmax=255, cmap=cmap)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.suptitle(title)
        plt.tight_layout()