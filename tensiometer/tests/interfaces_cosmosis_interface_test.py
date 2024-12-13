###############################################################################
# initial imports:

import unittest

import tensiometer.interfaces.cosmosis_interface as ci

import os

###############################################################################


class test_cosmosis_interface(unittest.TestCase):

    def setUp(self):
        # get path:
        self.here = os.path.dirname(os.path.abspath(__file__))
        # chain dir:
        self.chain_dir = self.here+'/../../test_chains/'

    def test_MCSamplesFromCosmosis(self):
        # import the chain:
        chain_name = self.chain_dir+'DES_multinest_cosmosis'
        chain = ci.MCSamplesFromCosmosis(chain_name)


###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
