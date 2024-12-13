###############################################################################
# initial imports:

import unittest

import tensiometer.mcmc_tension.param_diff as pd
import tensiometer.mcmc_tension.kde as mt
import tensiometer.utilities.stats_utilities as stut
from getdist import loadMCSamples

import os
import numpy as np

###############################################################################


class test_mcmc_shift(unittest.TestCase):

    def setUp(self):
        # get path:
        self.here = os.path.dirname(os.path.abspath(__file__))
        # get chains:
        self.chain_1 = loadMCSamples(self.here+'/../../test_chains/DES')
        self.chain_2 = loadMCSamples(self.here+'/../../test_chains/Planck18TTTEEE')
        self.chain_12 = loadMCSamples(self.here+'/../../test_chains/Planck18TTTEEE_DES')
        self.chain_prior = loadMCSamples(self.here+'/../../test_chains/prior')
        # thin the chain:
        self.chain_1.getConvergeTests()
        self.chain_2.getConvergeTests()
        self.chain_12.getConvergeTests()
        self.chain_prior.getConvergeTests()
        self.chain_1.weighted_thin(int(self.chain_1.indep_thin))
        self.chain_2.weighted_thin(int(self.chain_2.indep_thin))
        self.chain_12.weighted_thin(int(self.chain_12.indep_thin))
        self.chain_prior.weighted_thin(int(self.chain_prior.indep_thin))
        # get difference chain:
        self.diff_chain = pd.parameter_diff_chain(self.chain_1,
                                                  self.chain_2,
                                                  boost=1)

    # test that different exact methods give the same result:
    def test_kde_shift(self):
        # get brute force resuls:
        res_1 = mt.kde_parameter_shift(self.diff_chain,
                                       method='brute_force',
                                       scale=0.5)
        # get nearest elimination results:
        res_2 = mt.kde_parameter_shift(self.diff_chain,
                                       method='neighbor_elimination',
                                       scale=0.5)
        print(res_1, res_2)
        assert np.allclose(res_1, res_2)
        # now with high feedback:
        res_3 = mt.kde_parameter_shift(self.diff_chain,
                                       method='brute_force',
                                       feedback=2,
                                       scale=0.5)
        assert np.allclose(res_1, res_3)
        res_4 = mt.kde_parameter_shift(self.diff_chain,
                                       method='neighbor_elimination',
                                       feedback=2,
                                       scale=0.5)
        print(res_3, res_4)
        assert np.allclose(res_2, res_4)
        assert np.allclose(res_3, res_4)
        # now with given parameter names:
        param_names = ['delta_omegam', 'delta_sigma8']
        res_1 = mt.kde_parameter_shift(self.diff_chain,
                                       param_names=param_names,
                                       method='brute_force',
                                       scale=0.5)
        # get nearest elimination results:
        res_2 = mt.kde_parameter_shift(self.diff_chain,
                                       param_names=param_names,
                                       method='neighbor_elimination',
                                       scale=0.5)
        print(res_1, res_2)
        assert np.allclose(res_1, res_2)

    # test band selectors:
    def test_band(self):
        # prepare:
        n, d = self.diff_chain.samples.shape
        weights = self.diff_chain.weights
        wtot = np.sum(weights)
        neff = wtot**2 / np.sum(weights**2)
        # compute bands:
        mt.Scotts_bandwidth(d, neff)
        mt.AMISE_bandwidth(d, neff)
        mt.MAX_bandwidth(d, neff)
        mt.MISE_bandwidth_1d(d, neff)
        mt.MISE_bandwidth(d, neff)
        # whiten samples:
        white_samples = stut.whiten_samples(self.diff_chain.samples, weights)
        mt.UCV_bandwidth(weights, white_samples, mode='1d', feedback=1)
        mt.UCV_SP_bandwidth(white_samples, weights, near=1, near_max=20, feedback=1)

    # test FFT methods in 1 and 2d:
    def test_fft_shift(self):
        # test FFT in 1d:
        param_names = ['delta_sigma8']
        mt.kde_parameter_shift_1D_fft(self.diff_chain, param_names=param_names, feedback=2)
        # test FFT in 2d:
        param_names = ['delta_omegam', 'delta_sigma8']
        mt.kde_parameter_shift_2D_fft(self.diff_chain, param_names=param_names, feedback=2)

    # test ball and ellipse estimators:
    def test_ball_kde(self):
        res_1 = mt.kde_parameter_shift(self.diff_chain,
                                       method='brute_force',
                                       feedback=2,
                                       scale='BALL')
        res_2 = mt.kde_parameter_shift(self.diff_chain,
                                       method='neighbor_elimination',
                                       feedback=2,
                                       scale='BALL')
        assert np.allclose(res_1, res_2)

        res_1 = mt.kde_parameter_shift(self.diff_chain,
                                       method='brute_force',
                                       feedback=2,
                                       scale='ELL')
        res_2 = mt.kde_parameter_shift(self.diff_chain,
                                       method='neighbor_elimination',
                                       feedback=2,
                                       scale='ELL')
        assert np.allclose(res_1, res_2)


###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
