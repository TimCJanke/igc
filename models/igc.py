"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.utils import shuffle
from statsmodels.distributions.empirical_distribution import ECDF as fit_ecdf
from scipy.interpolate import RegularGridInterpolator
from sklearn.model_selection import train_test_split
from models.utils import mmd_score
import pandas as pd
from tensorflow.keras import regularizers




############### define loss ################
def ed(y_data, y_model):
    """Compute Energy distance

    Args:
        y_data (tf.tensor, shape 1xDxN): Samples from true distribution.
        y_model (tf. tensor, shape 1xDxM): Samples from model.

    Returns:
        tf.float: Energy distance for batch
    """
    n_samples_model = tf.cast(tf.shape(y_model)[2], dtype=tf.float32)
    n_samples_data = tf.cast(tf.shape(y_data)[2], dtype=tf.float32)
    
    N = y_model.shape[-1]


    mmd_12 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y_model - tf.repeat(tf.transpose(y_data, perm=(2,1,0)), repeats=N, axis=2)), axis=1)+K.epsilon()))
    mmd_22 = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(y_model - tf.repeat(tf.transpose(y_model, perm=(2,1,0)), repeats=N, axis=2)), axis=1)+K.epsilon()))

    loss = 2*mmd_12/(n_samples_model*n_samples_data) -  mmd_22/(n_samples_model*(n_samples_model-1))
    return loss


# subclass Keras loss
class MaxMeanDiscrepancy(Loss):
    def __init__(self, beta=1.0, name="MaxMeanDiscrepancy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.beta = beta

    def call(self, y_data, y_model):
        return ed(y_data, y_model)

    def get_config(self):
        cfg = super().get_config()
        cfg['beta'] = self.beta
        return cfg



################ define soft rank layer ################
class SoftRank(layers.Layer):
    """Differentiable ranking layer"""
    def __init__(self, alpha=1000.0):
        super(SoftRank, self).__init__()
        self.alpha = alpha # constant for scaling the sigmoid to approximate sign function, larger values ensure better ranking, overflow is handled properly by tensorflow

    def call(self, inputs, training=None):
        # input is a ?xSxD tensor, we wish to rank the S samples in each dimension per each batch
        # output is  ?xSxD tensor where for each dimension the entries are (rank-0.5)/N_rank
        if training:
            x = tf.expand_dims(inputs, axis=-1) #(?,S,D) -> (?,S,D,1)
            x_2 = tf.tile(x, (1,1,1,tf.shape(x)[1])) # (?,S,D,1) -> (?,S,D,S) (samples are repeated along axis 3, i.e. the last axis)
            x_1 = tf.transpose(x_2, (0,3,2,1)) #  (?,S,D,S) -> (?,S,D,S) (samples are repeated along axis 1)
            return tf.transpose(tf.reduce_sum(tf.sigmoid(self.alpha*(x_1-x_2)), axis=1), perm=(0,2,1))/(tf.cast(tf.shape(x)[1], dtype=tf.float32))
        return inputs
    
    def get_config(self):
        return {"alpha": self.alpha}



################ IGC model ################
class ImplicitGenerativeCopula(object):
    def __init__(self, dim_latent=10, dim_out=2, n_samples_train=100, n_layers=3, n_neurons=200, activation="relu", alpha=1000.0, mu=0.0, sigma=1.0, sigmoid_layer=False, sigmoid_slope=1.0, optimizer="Adam"):
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.n_samples_train = n_samples_train
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma
        self.sigmoid_layer = sigmoid_layer
        self.sigmoid_slope = sigmoid_slope
        self.optimizer = optimizer


        self.model = self._build_model()
        #keras.utils.plot_model(self.model, show_shapes=True)
        #self.model.summary()
        
    
    def _build_model(self):
        z_in = tf.keras.Input(shape=(self.dim_latent, self.n_samples_train), name="z_in")
        
        z = layers.Permute((2,1))(z_in)
        for i in range(self.n_layers):
            z = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1, padding="valid", data_format="channels_last", activation=self.activation)(z)
        z = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, padding="valid", data_format="channels_last", activation="linear")(z)
        
        if self.sigmoid_layer:
            z = layers.Lambda(lambda arg: tf.math.scalar_mul(self.sigmoid_slope, arg))(z)
            z = layers.Activation("sigmoid")(z)
        
        z = SoftRank(alpha=self.alpha)(z)
        z = layers.Permute((2,1))(z)

        mdl = Model(inputs=z_in, outputs=z)
        mdl.compile(loss=MaxMeanDiscrepancy(), optimizer=self.optimizer)

        return mdl

    def save_model(self, path):
        tf.keras.models.save_model(self.model, path) 
        print(f"Model saved to {path}.")


    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'MaxMeanDiscrepancy': MaxMeanDiscrepancy})
        self._fit_marginals()
        print("Loaded saved model.")

    
    def fit(self, y_train, regen_noise=1e12, batch_size=10, epochs=10, validation_split=0.0, validation_data=None):
        """ fit the model """


        if validation_split>0.0:
            training_data, validation_data = train_test_split(y_train, train_size=int((1.0-validation_split)*len(y_train)))
        else:
            training_data = y_train

        self.loss_dict = {"train": []}
        if validation_data is not None:
            self.loss_dict["val"] = []


        y_train_ = np.transpose(np.reshape(training_data,(-1, batch_size, training_data.shape[1]), order="C"), axes=(0,2,1))
        
        for i in tqdm(range(epochs)):
            if i%regen_noise == 0:
                z_train = self._make_some_noise(y_train_.shape[0])

            
            y_train_, z_train_ = shuffle(y_train_, z_train)
            loss_batch=[]
            for j in range(y_train_.shape[0]):
                loss_batch.append(self.model.train_on_batch(x=z_train_[[j],:,:], y=y_train_[[j],:,:]))
            self.loss_dict["train"].append(np.mean(loss_batch))

            if validation_data is not None:
                self._fit_marginals()
                self.loss_dict["val"].append(self._eval(validation_data, self.simulate(np.minimum(2000, len(validation_data)))))

        self._fit_marginals()
        return pd.DataFrame(self.loss_dict)


    def _eval(self, x, y):       
        return mmd_score(x,y, drop_xx=True)


    def _fit_marginals(self, n_samples=int(1e6)):
        """ fit marginal distributions via ECDF on large set of samples """
        samples_latent = self._get_latent_samples(n_samples)
        samples_latent = np.concatenate((samples_latent,
                                        np.ones((1, samples_latent.shape[1])),
                                        np.zeros((1, samples_latent.shape[1]))), axis=0)
        self.cdfs = []
        for i in range(samples_latent.shape[1]):
            self.cdfs.append(fit_ecdf(samples_latent[:,i]))
    

    def _get_latent_samples(self, n_samples):
        """ Draw samples from the latent distribution """
        return np.reshape(np.transpose(self.model.predict(self._make_some_noise(np.ceil(n_samples/self.n_samples_train).astype(int))), (0,2,1)), (-1,self.dim_out))[0:n_samples,:]


    def simulate(self, n_samples=100, return_latent_samples=False):
        """ draw n_samples randomly from distribution  """
        samples_latent = self._get_latent_samples(n_samples)
        samples = []
        for i in range(self.dim_out):
            samples.append(self.cdfs[i](samples_latent[:,i]))
        if return_latent_samples:
            return np.column_stack(samples), samples_latent
        else:
            return np.column_stack(samples)


    def cdf(self, v, u=None, n=10000):
        """ evaluate the empirical copula at points u using n samples"""
        # cdf is evaluated at points v, v has to be a MxD vector in [0,1]^D, cdf is evaluated at these points
        # u are samples from model NxD vector in [0,1]^D, if None n points will be sampled
        # larger n will lead to better estimation of the empirical copula but slows down computation

        if u is None:
            u = self.simulate(n)
        cdf_vals = np.empty(shape=(len(v)))
        for i in range(v.shape[0]):
            cdf_vals[i] = np.sum(np.all(u<=v[[i],:], axis=1))
        return cdf_vals/len(u)
    
    
    def get_cdf(self, u=None, n=10000, grid=np.arange(0.0, 1.1, 0.1)):
        """ Obtain a linearly interpolated cdf on specified grid. Very slow for large n or fine grid in d>3 """
        if u is None:
            u = self.simulate(n) # draw samples
        mgrid = np.meshgrid(*[grid for i in range(self.dim_out)], indexing="ij") # prepare grid in d dimensions
        mgrid = np.column_stack([np.ravel(i) for i in mgrid]) # reshape
        c_vals = np.empty(len(mgrid))
        for i in range(len(mgrid)):
            c_vals[i] = np.sum(np.all(u<=mgrid[[i],:], axis=1)) # compute empirical cdf for each grid point
        C_rs = np.reshape(c_vals/len(u),[len(grid) for i in range(self.dim_out)], order="C")
        cdf_fun = RegularGridInterpolator([grid for i in range(self.dim_out)], C_rs) # obtain linear interpolator function for grid
        return cdf_fun
    
    
    def _make_some_noise(self, n):
        """ returns normally distributed noise of dimension (N,DIM_LATENT,N_SAMPLES_TRAIN) """
        return np.random.normal(loc=self.mu, scale=self.sigma, size=(n, self.dim_latent, self.n_samples_train))




################ GMMN copula model ################
class GMMNCopula(ImplicitGenerativeCopula):
    """ A GMMN based generative copula with sigmoid output layer """
    
    def _build_model(self):
        z_in = tf.keras.Input(shape=(self.dim_latent, self.n_samples_train), name="z_in")
        
        z = layers.Permute((2,1))(z_in)
        for i in range(self.n_layers):
            z = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1, padding="valid", data_format="channels_last", activation=self.activation)(z)
        z = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, padding="valid", data_format="channels_last", activation="linear")(z)

        z = layers.Activation("sigmoid")(z)
        z = layers.Permute((2,1))(z)

        mdl = Model(inputs=z_in, outputs=z)
        mdl.compile(loss=MaxMeanDiscrepancy(), optimizer=self.optimizer)

        return mdl

    def fit(self, y_train, regen_noise=1e10, batch_size=10, epochs=10, validation_split=0.0, validation_data=None):
        """ fit the model """


        if validation_split>0.0:
            training_data, validation_data = train_test_split(y_train, train_size=int((1.0-validation_split)*len(y_train)))
        else:
            training_data = y_train

        self.loss_dict = {"train": []}
        if validation_data is not None:
            self.loss_dict["val"] = []


        y_train_ = np.transpose(np.reshape(training_data,(-1, batch_size, training_data.shape[1]), order="C"), axes=(0,2,1))
        
        for i in tqdm(range(epochs)):
            if i%regen_noise == 0:
                z_train = self._make_some_noise(y_train_.shape[0])

            y_train_, z_train_ = shuffle(y_train_, z_train)
            loss_batch=[]
            for j in range(y_train_.shape[0]):
                loss_batch.append(self.model.train_on_batch(x=z_train_[[j],:,:], y=y_train_[[j],:,:]))
            self.loss_dict["train"].append(np.mean(loss_batch))

            if validation_data is not None:
                self._fit_marginals()
                self.loss_dict["val"].append(self._eval(validation_data, self.simulate(np.minimum(2000, len(validation_data)))))

        self._fit_marginals()

        return pd.DataFrame(self.loss_dict)


    def simulate(self, n_samples=100, return_latent_samples=False, normalize_marginals=False):
        """ draw n_samples randomly from distribution  """
        samples_latent = self._get_latent_samples(n_samples)
        samples = []
        for i in range(self.dim_out):
            if normalize_marginals:
                samples.append(self.cdfs[i](samples_latent[:,i]))
            else:
                samples.append(samples_latent[:,i])

        if return_latent_samples:
            return np.column_stack(samples), samples_latent
        else:
            return np.column_stack(samples)


    def cdf(self, v, u=None, n=10000, normalize_marginals=False):
        """ evaluate the empirical copula at points u using n samples"""
        # cdf is evaluated at points v, v has to be a MxD vector in [0,1]^D, cdf is evaluated at these points
        # u are samples from model NxD vector in [0,1]^D, if None n points will be sampled
        # larger n will lead to better estimation of the empirical copula but slows down computation

        if u is None:
            u = self.simulate(n, normalize_marginals=normalize_marginals)
        cdf_vals = np.empty(shape=(len(v)))
        for i in range(v.shape[0]):
            cdf_vals[i] = np.sum(np.all(u<=v[[i],:], axis=1))
        return cdf_vals/len(u)
    
    def get_cdf(self, u=None, n=10000, grid=np.arange(0.0, 1.1, 0.1), normalize_marginals=False):
        """ Obtain a linearly interpolated cdf on specified grid. Very slow for large n or fine grid in d>3 """
        if u is None:
            u = self.simulate(n, normalize_marginals=normalize_marginals) # draw samples
        mgrid = np.meshgrid(*[grid for i in range(self.dim_out)], indexing="ij") # prepare grid in d dimensions
        mgrid = np.column_stack([np.ravel(i) for i in mgrid]) # reshape
        c_vals = np.empty(len(mgrid))
        for i in range(len(mgrid)):
            c_vals[i] = np.sum(np.all(u<=mgrid[[i],:], axis=1)) # compute empirical cdf for each grid point
        C_rs = np.reshape(c_vals/len(u),[len(grid) for i in range(self.dim_out)], order="C")
        cdf_fun = RegularGridInterpolator([grid for i in range(self.dim_out)], C_rs) # obtain linear interpolator function for grid
        return cdf_fun




################ GMMN model ################
class GenerativeMomentMatchingNetwork(object):
    """ A GMMN type model that operates directly on the data"""
    def __init__(self, dim_latent=10, dim_out=2, n_samples_train=100, n_layers=3, n_neurons=200, activation="relu", alpha=1000.0, mu=0.0, sigma=1.0, optimizer="Adam"):
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.n_samples_train = n_samples_train
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.mu = mu
        self.sigma = sigma
        self.optimizer = optimizer


        self.model = self._build_model()
        #keras.utils.plot_model(self.model, show_shapes=True)
        #self.model.summary()
        
    
    def _build_model(self):
        z_in = tf.keras.Input(shape=(self.dim_latent, self.n_samples_train), name="z_in")
        
        z = layers.Permute((2,1))(z_in)
        for i in range(self.n_layers):
            z = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1, padding="valid", data_format="channels_last", activation=self.activation)(z)
        z = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, padding="valid", data_format="channels_last", activation="linear")(z)
        
        z = layers.Permute((2,1))(z)

        mdl = Model(inputs=z_in, outputs=z)
        mdl.compile(loss=MaxMeanDiscrepancy(), optimizer=self.optimizer)

        return mdl


    def save_model(self, path):
        tf.keras.models.save_model(self.model, path) 
        print(f"Model saved to {path}.")



    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'MaxMeanDiscrepancy': MaxMeanDiscrepancy})
        self._fit_marginals()
        print("Loaded saved model.")

    

    def fit(self, y_train, regen_noise=1e12, batch_size=10, epochs=10, validation_split=0.0, validation_data=None):
        """ fit the model """
        
        self.scaler = MinMaxScaler()
        self.scaler.fit(y_train)
        y_train = self.scaler.transform(y_train)

        if validation_split>0.0:
            training_data, validation_data = train_test_split(y_train, train_size=int((1.0-validation_split)*len(y_train)))
        else:
            training_data = y_train

        self.loss_dict = {"train": []}
        if validation_data is not None:
            self.loss_dict["val"] = []


        y_train_ = np.transpose(np.reshape(training_data,(-1, batch_size, training_data.shape[1]), order="C"), axes=(0,2,1))
        
        for i in tqdm(range(epochs)):
            if i%regen_noise == 0:
                z_train = self._make_some_noise(y_train_.shape[0])

            y_train_, z_train_ = shuffle(y_train_, z_train)
            loss_batch=[]
            for j in range(y_train_.shape[0]):
                loss_batch.append(self.model.train_on_batch(x=z_train_[[j],:,:], y=y_train_[[j],:,:]))
            self.loss_dict["train"].append(np.mean(loss_batch))

            if validation_data is not None:
                self.loss_dict["val"].append(self._eval(validation_data, self.simulate(np.minimum(2000, len(validation_data)))))

        self._fit_marginals()

        return pd.DataFrame(self.loss_dict)


    def _fit_marginals(self, n_samples=int(1e6)):
        """ fit marginal distributions via ECDF on large set of samples """
        x = self.simulate(n_samples)
        self.cdfs = []
        for i in range(x.shape[1]):
            self.cdfs.append(fit_ecdf(x[:,i])) 


    def _eval(self, x, y):
        return mmd_score(x,y, drop_xx=True)


    def _to_pobs(self, x):
        u = []
        for i in range(x.shape[1]):
            u.append(self.cdfs[i](x[:,i]))
        return np.column_stack(u)        


    def simulate_copula(self, n_samples):
        """ Draw samples from the learned distribution """
        return self._to_pobs(self.simulate(n_samples))


    def simulate(self, n_samples):
        """ Draw samples from the learned distribution """
        return self.scaler.inverse_transform(np.reshape(np.transpose(self.model.predict(self._make_some_noise(np.ceil(n_samples/self.n_samples_train).astype(int))), (0,2,1)), (-1,self.dim_out))[0:n_samples,:])


    def _make_some_noise(self, n):
        """ returns normally distributed noise of dimension (N,DIM_LATENT,N_SAMPLES_TRAIN) """
        return np.random.normal(loc=self.mu, scale=self.sigma, size=(n, self.dim_latent, self.n_samples_train))



################ GAN model ################
class GenerativeAdversarialNetwork(object):
    """ A vanilla GAN that operates directly on the data """
    def __init__(self, dim_latent=10, dim_out=2, n_samples_train=100, n_layers=3, n_neurons=200, activation="relu", alpha=1000.0, mu=0.0, sigma=1.0, optimizer="Adam"):
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.n_samples_train = n_samples_train
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.mu = mu
        self.sigma = sigma
        self.optimizer = optimizer


        self.model = self._build_model()
        #keras.utils.plot_model(self.model, show_shapes=True)
        #self.model.summary()


    def _build_model(self):
        model = buildGAN(discriminator= self._build_discriminator(), 
                                generator=self._build_generator(), 
                                latent_dim=self.dim_latent)

        model.compile(d_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      g_optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss_fn=keras.losses.BinaryCrossentropy())
        
        return model


    def _build_generator(self):
        z_in = tf.keras.Input(shape=(self.dim_latent,), name="z_in")
        z = z_in
        for i in range(self.n_layers):
            z = layers.Dense(self.n_neurons, activation=self.activation)(z)
        y = layers.Dense(self.dim_out, activation="linear")(z)

        return Model(inputs=z_in, outputs=y, name="generator")


    def _build_discriminator(self):
        x_in = tf.keras.Input(shape=(self.dim_out,), name="x_in")
        x = x_in
        for i in range(self.n_layers):
            x = layers.Dense(self.n_neurons, activation=self.activation)(x)
        y = layers.Dense(1, activation="sigmoid")(x)

        return Model(inputs=x_in, outputs=y, name="discriminator")


    def save_model(self, path):
        tf.keras.models.save_model(self.model, path) 
        print(f"Model saved to {path}.")


    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print("Loaded saved model.")


    def fit(self, y_train, batch_size=32, epochs=10, validation_split=0.0, validation_data=None):
        """ fit the model """
        
        self.scaler = MinMaxScaler()
        self.scaler.fit(y_train)
        y_train = tf.convert_to_tensor(self.scaler.transform(y_train), dtype=tf.float32)
        
        self.loss_dict = self.model.fit(y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        self._fit_marginals()

        return pd.DataFrame(self.loss_dict.history)


    def _fit_marginals(self, n_samples=int(1e6)):
        """ fit marginal distributions via ECDF on large set of samples """
        x = self.simulate(n_samples)
        self.cdfs = []
        for i in range(x.shape[1]):
            self.cdfs.append(fit_ecdf(x[:,i])) 


    def _eval(self, x, y):
        return mmd_score(x,y, drop_xx=True)


    def _to_pobs(self, x):
        u = []
        for i in range(x.shape[1]):
            u.append(self.cdfs[i](x[:,i]))
        return np.column_stack(u)


    def simulate_copula(self, n_samples):
        """ Draw samples from the learned distribution """
        return self._to_pobs(self.simulate(n_samples))


    def simulate(self, n_samples):
        """ Draw samples from the learned distribution """
        return self.scaler.inverse_transform(self.model.generator(self._make_some_noise(n_samples)))

    def _make_some_noise(self, n):
        """ returns normally distributed noise of dimension (N, DIM_LATENT) """
        return np.random.normal(loc=self.mu, scale=self.sigma, size=(n, self.dim_latent))




class buildGAN(keras.Model):
    """ class for building a vanilla GAN model """
    def __init__(self, discriminator, generator, latent_dim):
        super(buildGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(buildGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, data_true):
        batch_size = data_true.shape[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        data_fake = self.generator(random_latent_vectors)
        combined_images = tf.concat([data_fake, data_true], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
