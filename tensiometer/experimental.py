"""
Experimental features.

For test purposes:

import os, sys
import time
import gc
from numba import jit
import numpy as np
import getdist.chains as gchains
gchains.print_load_details = False
from getdist import MCSamples, WeightedSamples
import scipy
from scipy.linalg import sqrtm
from scipy.integrate import simps
from scipy.spatial import cKDTree

# imports for parallel calculations:
import multiprocessing
import joblib
# number of threads available:
if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = multiprocessing.cpu_count()

from getdist import loadMCSamples, MCSamples, WeightedSamples

# add path for correct version of tensiometer:
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import tensiometer.mcmc_tension as tmt
import tensiometer.utilities as utils

chain_1 = loadMCSamples('./test_chains/DES')
chain_2 = loadMCSamples('./test_chains/Planck18TTTEEE')
chain_12 = loadMCSamples('./test_chains/Planck18TTTEEE_DES')
chain_prior = loadMCSamples('./test_chains/prior')

import matplotlib.pyplot as plt

diff_chain = tmt.parameter_diff_chain(chain_1, chain_2, boost=1)
num_params, num_samples = diff_chain.samples.T.shape

param_names = None
scale = None
method = 'brute_force'
feedback=2
n_threads = 1
"""

import os
import time
import gc
from numba import jit, njit
import numpy as np
import getdist.chains as gchains
gchains.print_load_details = False
from getdist import MCSamples, WeightedSamples
import scipy
from scipy.linalg import sqrtm
from scipy.integrate import simps
from scipy.spatial import cKDTree

from . import mcmc_tension as tmt

"""
# initialize param names:
if param_names is None:
    param_names = diff_chain.getParamNames().getRunningNames()
else:
    chain_params = diff_chain.getParamNames().list()
    if not np.all([name in chain_params for name in param_names]):
        raise ValueError('Input parameter is not in the diff chain.\n',
                         'Input parameters ', param_names, '\n'
                         'Possible parameters', chain_params)
# indexes:
ind = [diff_chain.index[name] for name in param_names]
# some initial calculations:
_num_samples = np.sum(diff_chain.weights)
_num_params = len(ind)
# number of effective samples:
_num_samples_eff = np.sum(diff_chain.weights)**2 / \
    np.sum(diff_chain.weights**2)
# whighten samples:
white_samples = utils.whiten_samples(diff_chain.samples[:, ind],
                                      diff_chain.weights)
weights = diff_chain.weights
white_samples = white_samples

"""

################# minimize scale:

def UCV_SP_bandwidth(white_samples, weights, near=1, near_max=200):
    """

    near = 10
    near_max = 200

    """
    # digest input:
    n, d = white_samples.shape
    fac = 2**(-d/2.)
    # prepare the Tree with the samples:
    data_tree = cKDTree(white_samples, balanced_tree=True)
    # compute the weights vectors:
    wtot = np.sum(weights)
    weights2 = weights**2
    neff = wtot**2 / np.sum(weights2)
    alpha = wtot / (wtot - weights)
    # query the Tree for the maximum number of nearest neighbours:
    dist, idx = data_tree.query(white_samples, np.arange(2, near_max+1), n_jobs=-1)
    r2 = np.square(dist)
    # do all sort of precomputations:
    R = dist[:, near]
    R2 = r2[:, near]
    R2s = R2[:, None] + R2[idx]
    term_1 = fac*np.sum(weights2/R**d)
    weight_term = weights[:, None]*weights[idx]
    R2sd = R2s**(-d/2)
    Rd = R[:, None]**d
    R21 = r2/R2s
    R22 = r2/R2[:, None]
    alpha_temp = alpha[:, None]

    # define helper for minimization:
    @njit
    def _helper(gamma):
        # compute the i != j sum:
        temp = weight_term*(R2sd*gamma**(-d/2)*np.exp(-0.5*R21/gamma) - 2.*alpha_temp/Rd/gamma**d*np.exp(-0.5*R22/gamma))
        # sum:
        _ucv = term_1/gamma**d + np.sum(temp)
        _ucv = _ucv / wtot**2
        #
        return _ucv

    # initial guess:
    x0 = tmt.AMISE_bandwidth(d, neff)[0, 0]
    # call optimizer:
    res = scipy.optimize.minimize(lambda x: _helper(np.exp(x)), x0=np.log(x0), method='Nelder-Mead')
    res.x = np.exp(res.x)
    #
    return res






pass
