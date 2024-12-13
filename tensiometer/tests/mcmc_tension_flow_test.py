###############################################################################
# initial imports:

import unittest

import tensiometer.mcmc_tension.param_diff as pd
from getdist import loadMCSamples

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
        # get difference chain:
        self.diff_chain = pd.parameter_diff_chain(self.chain_1,
                                                  self.chain_2,
                                                  boost=1)

    # test that the MAF can be initialized and trained for few epochs:
    def test_flow_runs(self):
        pass
        #diff_flow_callback = mt.FlowCallback(self.diff_chain, feedback=0)
        ## Train model
        #diff_flow_callback.train(epochs=5)
        ## Compute tension
        #diff_flow_callback.estimate_shift(tol=1.0, max_iter=10)


###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
