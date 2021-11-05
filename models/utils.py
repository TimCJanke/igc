"""
@author: Tim Janke, Energy Information Networks & Systems @ TU Darmstadt

"""

import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.stats import rankdata

class cdf_interpolator(object):
    """
    Interpolation for CDf and inverse CDF for non-parametric 1-D distribution p(x) represented by samples.
    Uses scipy's PchipInterpolator(cubic) or interp1d(linear) to define CDF and PPF from given quantiles.

     Parameters
     ----------
     x : array, shape (N,)
         A 1-D array samples.
     x_min : float
         The smallest possible real value.
     x_max : float
         The largest possible real value.
    kind: str (optional)
        type of interpolation ("linear", "cubic", "stepwise"), defaults to "linear"

     Methods
     -------
     ppf(tau):
         Interpolated PPF.
     cdf(x):
        Interpolated CDF. 
    sample(size):
        Sample from distribution.
    """    
    def __init__(self, x, x_min, x_max, kind="linear", eps=1e-6):
        self.q = np.sort(x)
        self.taus = rankdata(self.q)/(len(self.q)+1)
        self.x_min = x_min
        self.x_max = x_max
        self.kind = kind # "cubic" or "linear"
        self.eps = eps
        
        self.q_ = self._get_q_()
        self.taus_ = np.concatenate(([0.0], self.taus, [1.0]))
        
        self.ppf_fun = self._interpolate_ppf()
        self.cdf_fun = self._interpolate_cdf()
        
            
        
    def _get_q_(self):
        # enure strictly monotone function by recursively adding epsilon until all values in q_ are increasing
        q_ = np.concatenate(([self.x_min], self.q, [self.x_max]))
        #mask_non_increasing = np.diff(q_, prepend=self.x_min-self.eps)<=0
        # while any(mask_non_increasing):
        #     q_[mask_non_increasing] = q_[mask_non_increasing] + self.eps
        #     mask_non_increasing = np.diff(q_, prepend=self.x_min-self.eps)<=0
        return q_

    def _interpolate_ppf(self):
        if self.kind == "cubic":
            return PchipInterpolator(x=self.taus_,
                                     y=self.q_, 
                                     extrapolate=False)
        elif self.kind == "linear":
            return interp1d(x=self.taus_,
                            y=self.q_, 
                            bounds_error=False,
                            fill_value=(self.x_min, self.x_max))
        else:
            raise ValueError("Unknown interpolation kind.")
        
    def _interpolate_cdf(self):
        tau_grid = np.linspace(0,1.0,len(self.taus)*2+2,endpoint=True)
        q_grid =  self.ppf_fun(tau_grid)
        q_grid[-1] = q_grid[-1]+self.eps
        if self.kind == "cubic":
            return PchipInterpolator(x=q_grid, 
                                     y=tau_grid, 
                                     extrapolate=False)
        elif self.kind == "linear":
            return interp1d(x=q_grid,
                            y=tau_grid,
                            bounds_error=False,
                            fill_value=(0.0,1.0))
        else:
            raise ValueError("Unknown interpolation kind.")
            
    
    def ppf(self, tau):
        """
        Interpolated PPF aka inverse CDF aka Quantile Function

        Parameters
        ----------
        tau : array of floats bewtween (0,1)
            Query point for PPF.

        Returns
        -------
        array of same shape as tau
            Values of the PPF at points in q.

        """
        return self.ppf_fun(tau)
    
    def cdf(self, x):
        """
        Interpolated CDF.

        Parameters
        ----------
        x : array of floats
            Query point for CDF.

        Returns
        -------
        array of same shape as x
            Values of the CDF at points in x.

        """
        return self.cdf_fun(np.clip(x, a_min=self.x_min, a_max=self.x_max))
    
    def icdf(self, tau):
        """ Alias for ppf """
        return self.ppf(tau)

    def sample(self, size=1):
        """
        Draw random samples from distribution.

        Parameters
        ----------
        size : int or tuple of ints
            Defines number and shape of returned samples.

        Returns
        -------
        float or array
            Random samples from sitribution.
        
        """
        return self.ppf(np.random.uniform(low=0.0, high=1.0, size=size))


#Maximum Mean Discrepancy (MMD)
def mmd_score(x,y, kernel="pow", sigma=1.0, beta=1.0, sigmas_mixture=[0.001, 0.01, 0.15, 0.25, 0.50, 0.75], drop_xx=False, drop_yy=False):
    """
    Compute Maximum Mean Discrepancy (MMD) between two multivariate distributions based on samples.
    For kernel = "pow" and beta=1.0 the MMD is equivalent to the energy distance.

    Args:
        x (numpy array): Samples from distribution p
        y (numpy array): Samples from distribution q
        kernel (str, optional): Type of kernel function. Defaults to "rbf".
        sigma (float, optional): bandwidth parameter for RBF kernel. Defaults to 1.0.
        beta (float, optional): Power for power kernel. Defaults to 1.0. This kernel recovers the energy distance.
        sigmas_mixture (list, optional): Bandwidths for mixture of RBF kernels. Defaults to [0.01, 0.1, 1.0, 10.0].

    Returns:
        float: MMD score
    """

    beta=1.0
    N = x.shape[0]
    M = y.shape[0]
    

    if kernel == "rbf":
        def kern(a):
            return np.exp(-np.power(a,2)/(2*sigma**2))

    elif kernel == "pow":
        def kern(a):
            return -np.power(a,beta)

    elif kernel == "rbf_mixture":
        def kern(a):
            res = 0.0
            for sig in sigmas_mixture:
                res = res + np.exp(-np.power(a,2)/(2*sig**2))
            return res

    def get_score(x,y):
        res = 0.0
        for i in range(x.shape[0]):
            res = res + np.sum(kern(np.sqrt(np.sum(np.square(x[[i],:] - y), axis=1))))
        return res

    if (drop_xx is False) and (drop_yy is False):
        return get_score(x,x)/(N*(N-1)) - 2*get_score(x,y)/(N*M)  + get_score(y,y)/(M*(M-1))

    else:
        if drop_xx:
            return get_score(y,y)/(M*(M-1)) - 2*get_score(x,y)/(N*M)
        elif drop_yy:
            return get_score(x,x)/(N*(N-1)) - 2*get_score(x,y)/(N*M)


