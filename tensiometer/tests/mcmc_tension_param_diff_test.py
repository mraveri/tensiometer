###############################################################################
# initial imports:

import unittest

import tensiometer.mcmc_tension.param_diff as pd
from getdist import MCSamples, loadMCSamples

import os

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
        
    # test that the difference between two chains, where one of them has a fixed parameter, is computed correctly:
    def test_param_diff(self):
        # create modified chain 1 with a missing parameter:
        names = [p.name for p in self.chain_1.getParamNames().names if p.name != 'tau']
        indices = [self.chain_1.index[n] for n in names]
        labels = [self.chain_1.getParamNames().parWithName(n).label for n in names]
        samples = self.chain_1.samples[:, indices]
        self.chain_1_mod = MCSamples(samples=samples,
                                     names=names,
                                     labels=labels,
                                     weights=self.chain_1.weights,
                                     loglikes=self.chain_1.loglikes,
                                     ranges=self.chain_1.ranges,
                                     ignore_rows=self.chain_1.ignore_rows,
                                     sampler=self.chain_1.sampler)
        self.chain_1_mod.updateBaseStatistics()

        # get difference chain:
        self.diff_chain = pd.parameter_diff_chain(self.chain_1_mod,
                                                  self.chain_2,
                                                  boost=1,
                                                  fixed_params={'tau': 0.05})

###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
