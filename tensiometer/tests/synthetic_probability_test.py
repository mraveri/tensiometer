###############################################################################
# initial imports:

import unittest
unittest.TestLoader.sortTestMethodsUsing = None

import tensiometer.synthetic_probability.synthetic_probability as sp
from tensiometer.synthetic_probability import flow_profiler

import tensiometer.utilities.stats_utilities as stutilities

import numpy as np

from getdist import MCSamples

# tensorflow imports:
import tensorflow as tf
import tensorflow_probability as tfp

# getdist settings to ensure consistency of plots:
getdist_settings = {'ignore_rows': 0.0, 
                    'smooth_scale_2D': 0.3,
                    'smooth_scale_1D': 0.3,
                    }   

###############################################################################

class test_synthetic_probability(unittest.TestCase):

    def setUp(self):

        # define the parameters of the problem:
        dim = 6
        num_gaussians = 3
        num_samples = 10000

        # we seed the random number generator to get reproducible results:
        seed = 100
        np.random.seed(seed)
        # we define the range for the means and covariances:
        mean_range = (-0.5, 0.5)
        cov_scale = 0.4**2
        # means and covs:
        means = np.random.uniform(mean_range[0], mean_range[1], num_gaussians*dim).reshape(num_gaussians, dim)
        weights = np.random.rand(num_gaussians)
        weights = weights / np.sum(weights)
        covs = [cov_scale*stutilities.vector_to_PDM(np.random.rand(int(dim*(dim+1)/2))) for _ in range(num_gaussians)]

        # cast to required precision:
        means = means.astype(np.float32)
        weights = weights.astype(np.float32)
        covs = [cov.astype(np.float32) for cov in covs]

        # initialize distribution:
        distribution = tfp.distributions.Mixture(
            cat=tfp.distributions.Categorical(probs=weights),
            components=[
                tfp.distributions.MultivariateNormalTriL(loc=_m, scale_tril=tf.linalg.cholesky(_c))
                for _m, _c in zip(means, covs)
            ], name='Mixture')

        # sample the distribution:
        samples = distribution.sample(num_samples).numpy()
        # calculate log posteriors:
        logP = distribution.log_prob(samples).numpy()

        # create MCSamples from the samples:
        self.chain = MCSamples(samples=samples, 
                               settings=getdist_settings,
                               loglikes=-logP,
                               name_tag='Mixture',
                               )
        
        # profiler options:
        self.profiler_options = {
            'num_gd_interactions_1D': 100,  # number of gradient descent interactions for the 1D profile
            'num_gd_interactions_2D': 100,  # number of gradient descent interactions for the 2D profile
            'scipy_options': {  # options for the scipy polishing minimizer
                        'ftol': 1.e-06,
                        'gtol': 0.0,
                        'maxls': 40,
                    },
            'scipy_use_jac': True,  # use the jacobian in the minimizer
            'num_points_1D': 64, # number of points for the 1D profile
            'num_points_2D': 32, # number of points per dimension for the 2D profile
            'smooth_scale_1D': 0.2, # smoothing scale for the 1D profile
            'smooth_scale_2D': 0.2, # smoothing scale for the 2D profile
            }


    def test_flow_from_chain(self):
        
        # train single flow, selecting from a population of two:
        kwargs = {
          'feedback': 2,
          'plot_every': 0,
          'pop_size': 2,
          'epochs': 5,}
        flow = sp.flow_from_chain(self.chain, **kwargs)
        # call profiler to test the flow:
        flow_profile = flow_profiler.posterior_profile_plotter(flow, initialize_cache=False, feedback=2)
        flow_profile.update_cache(params=None, update_MAP=True, update_1D=False, update_2D=False, 
                                  **self.profiler_options)

        
    def test_average_flow_from_chain(self):
        
        # train average flow:
        kwargs = {
          'feedback': 2,
          'plot_every': 0,
          'pop_size': 1,
          'num_flows': 3,
          'epochs': 5,
        }
        average_flow = sp.average_flow_from_chain(self.chain, **kwargs)
        # call profiler to test the flow:
        flow_profile = flow_profiler.posterior_profile_plotter(average_flow, initialize_cache=False, feedback=2)
        flow_profile.update_cache(params=None, update_MAP=True, update_1D=False, update_2D=False, 
                                  **self.profiler_options)
