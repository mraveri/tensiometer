###############################################################################
# initial imports:

import unittest

import tensiometer.chains_convergence as conv
import tensiometer.utilities.stats_utilities as stut
from getdist import loadMCSamples

import os
import numpy as np

###############################################################################


class test_convergence(unittest.TestCase):

    def setUp(self):
        # get path:
        self.here = os.path.dirname(os.path.abspath(__file__))
        # get chains:
        self.chain = loadMCSamples(self.here+'/../../test_chains/DES')

    # test standard Gelman Rubin for multiple chains:
    def test_GR_test(self):
        res1 = conv.GR_test(self.chain)
        res2 = conv.GR_test(stut.get_separate_mcsamples(self.chain))
        assert np.allclose(res1[0], res2[0]) and np.allclose(res1[1], res2[1])
        res3 = conv.GR_test(stut.get_separate_mcsamples(self.chain),
                            param_names=self.chain.getParamNames().getRunningNames())
        assert np.allclose(res1[0], res3[0]) and np.allclose(res1[1], res3[1])

    # test standard Gelman Rubin for two chains:
    def test_GR_test_two_chains(self):
        res2 = conv.GR_test(stut.get_separate_mcsamples(self.chain)[:2])
        res3 = conv.GR_test(stut.get_separate_mcsamples(self.chain)[:2],
                            param_names=self.chain.getParamNames().getRunningNames())
        assert np.allclose(res2[0], res3[0]) and np.allclose(res2[1], res3[1])

    # test higher moments test:
    def test_GRn_test(self):
        kwargs = {}
        print(conv.GRn_test(self.chain, n=2, param_names=None, feedback=2,
                            optimizer='ParticleSwarm', **kwargs))
        print(conv.GRn_test(self.chain, n=3, param_names=None, feedback=2,
                            optimizer='ParticleSwarm', **kwargs))
        print(conv.GRn_test(self.chain, n=2, param_names=None, feedback=2,
                            optimizer='TrustRegions', **kwargs))

    # test higher moments test with two chains:
    def test_GRn_test_two_chains(self):
        kwargs = {}
        print(conv.GRn_test(stut.get_separate_mcsamples(self.chain)[:2], n=2, param_names=None, feedback=0,
                            optimizer='ParticleSwarm', **kwargs))

    def test_errors(self):
        #self.assertRaises(TypeError, conv.GR_test('test'))
        #self.assertRaises(TypeError, conv.GR_test(['test']))
        #self.assertRaises(ValueError, conv.GR_test([]))
        pass

###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
