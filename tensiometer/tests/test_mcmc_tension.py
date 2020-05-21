###############################################################################
# initial imports:

import unittest

import tensiometer.mcmc_tension as mt
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
        self.diff_chain = mt.parameter_diff_chain(self.chain_1,
                                                  self.chain_2,
                                                  boost=1)

    # test that different exact methods give the same result:
    def test_from_confidence_to_sigma_result(self):
        # get brute force resuls:
        res_1 = mt.exact_parameter_shift(self.diff_chain,
                                         method='brute_force',
                                         scale=0.5)
        # get nearest elimination results:
        res_2 = mt.exact_parameter_shift(self.diff_chain,
                                         method='nearest_elimination',
                                         scale=0.5)
        print(res_1, res_2)
        assert np.allclose(res_1, res_2)
        # now with high feedback:
        res_3 = mt.exact_parameter_shift(self.diff_chain,
                                         method='brute_force',
                                         feedback=2,
                                         scale=0.5)
        assert np.allclose(res_1, res_3)
        res_4 = mt.exact_parameter_shift(self.diff_chain,
                                         method='nearest_elimination',
                                         feedback=2,
                                         scale=0.5)
        print(res_3, res_4)
        assert np.allclose(res_2, res_4)
        assert np.allclose(res_3, res_4)
        # now with given parameter names:
        param_names = ['delta_omegam', 'delta_sigma8']
        res_1 = mt.exact_parameter_shift(self.diff_chain,
                                         param_names=param_names,
                                         method='brute_force',
                                         scale=0.5)
        # get nearest elimination results:
        res_2 = mt.exact_parameter_shift(self.diff_chain,
                                         param_names=param_names,
                                         method='nearest_elimination',
                                         scale=0.5)
        print(res_1, res_2)
        assert np.allclose(res_1, res_2)

###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
