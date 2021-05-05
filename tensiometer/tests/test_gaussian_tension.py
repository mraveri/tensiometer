###############################################################################
# initial imports:

import unittest

import tensiometer.gaussian_tension as gt

import os
import numpy as np
from getdist.gaussian_mixtures import GaussianND
from getdist import loadMCSamples

###############################################################################
# initial setup common for all tests:


def setup_test(type):
    # define two Gaussian distributions:
    type.n1 = 2
    type.n2 = 3
    type.mean1 = 1.*np.ones(type.n1)
    type.mean2 = 2.*np.ones(type.n2)
    type.cov1 = 1.*np.diag(np.ones(type.n1))
    type.cov2 = 2.*np.diag(np.ones(type.n2))
    type.param_names1 = ['p'+str(i) for i in range(type.n1)]
    type.param_names2 = ['p'+str(i) for i in range(type.n2)]
    type.Gaussian1 = GaussianND(type.mean1, type.cov1,
                                names=type.param_names1)
    type.Gaussian2 = GaussianND(type.mean2, type.cov2,
                                names=type.param_names2)
    type.chain_1 = type.Gaussian1.MCSamples(1000)
    type.chain_2 = type.Gaussian2.MCSamples(1000)
    # define the prior:
    type.GaussianPrior = GaussianND(type.mean2, 10.*type.cov2,
                                    names=type.param_names2)
    type.prior_chain = type.GaussianPrior.MCSamples(1000)

###############################################################################


class test_helpers(unittest.TestCase):

    def setUp(self):
        setup_test(self)

    def test_helpers(self):
        assert self.chain_1.getParamNames().getRunningNames() == \
            gt._check_param_names(self.chain_1, param_names=None)
        assert gt._check_param_names(self.chain_1, param_names=['p1']) \
            == ['p1']
        gt._check_chain_type(self.chain_1)

    def test_errors(self):
        with self.assertRaises(ValueError):
            gt._check_param_names(self.chain_1, param_names=['test'])
        with self.assertRaises(TypeError):
            gt._check_chain_type(self.Gaussian1)

###############################################################################


class test_utilities(unittest.TestCase):

    def setUp(self):
        setup_test(self)

    def test_get_prior_covariance(self):
        self.chain_1.setRanges({'p0': [0., 1.0],
                                'p1': [0., 1.0]})
        gt.get_prior_covariance(self.chain_1)
        gt.get_prior_covariance(self.chain_2)

    def test_get_Neff(self):
        assert np.allclose(gt.get_Neff(self.chain_1), 2.0)
        gt.get_Neff(self.chain_1, prior_chain=self.prior_chain)
        assert np.allclose(gt.get_Neff(self.chain_1, param_names=['p1']), 1.0)
        assert np.allclose(gt.get_Neff(self.chain_1, prior_factor=1.0), 2.0)

    def test_gaussian_approximation(self):
        gt.gaussian_approximation(self.chain_1)
        gt.gaussian_approximation(self.chain_1, param_names=['p1'])
        self.chain_1.label = 'chain_1'
        temp = gt.gaussian_approximation(self.chain_1)
        assert temp.label == 'Gaussian '+self.chain_1.label
        self.chain_1.label = None
        self.chain_1.name_tag = 'chain_1'
        temp = gt.gaussian_approximation(self.chain_1)
        assert temp.label == 'Gaussian_'+self.chain_1.name_tag

###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
