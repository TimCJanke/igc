import numpy as np
from scipy.stats import multivariate_normal, norm
import sys

class GaussianCopula(object):
    """Class for a multivariate distribution that uses the Gaussian copula."""
    
    def __init__(self):
        self.epsilon = 1e-8

    def _get_sigma(self, y):
        # compute covariance matrix
        sigma = np.corrcoef(y, rowvar=False)
        sigma = np.nan_to_num(sigma, nan=0.0)
        # If singular, add some noise to the diagonal
        if np.linalg.cond(sigma) > 1.0 / sys.float_info.epsilon:
            sigma = sigma + np.identity(sigma.shape[0]) * self.epsilon
            print("singular covariance, added noise")
        return sigma

    def fit(self, u):
        # distribution
        self.sigma = self._get_sigma(norm.ppf(u))
        self.mvn = multivariate_normal(mean=None, cov=self.sigma)


    def simulate(self, n=1):
        # simulate n samples
        return norm.cdf(self.mvn.rvs(n))

class IndependenceCopula(object):
    """Class for a multivariate distribution that uses the Gaussian copula."""
    
    def __init__(self, dim):
        self.dim = dim

    def fit(self, u):
        pass

    def cdf(self, u):
        # evaluate cdf at u
        return np.prod(u, axis=1)

    def simulate(self, n=1):
        # simulate n samples
        return np.random.uniform(0.0,1.0,size=(n, self.dim))
