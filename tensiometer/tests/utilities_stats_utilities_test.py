###############################################################################
# initial imports:

import unittest

import tensiometer.utilities.stats_utilities as stut

import numpy as np
from scipy.stats import multivariate_normal

from getdist import MCSamples

###############################################################################

def helper_MCsamples(ndim=4, nsamp=10000):

    random_state = np.random.default_rng(0)
    A = random_state.random((ndim,ndim))
    cov = np.dot(A, A.T)

    distribution = multivariate_normal([0]*ndim, cov)
    samps = distribution.rvs(nsamp)
    loglikes = -distribution.logpdf(samps)

    names = ["x%s"%i for i in range(ndim)]
    labels =  ["x_%s"%i for i in range(ndim)]

    samples = MCSamples(samples=samps, names=names, labels=labels, loglikes=loglikes)
    #
    return samples

###############################################################################

class test_confidence_to_sigma(unittest.TestCase):

    def setUp(self):
        pass

    # test against known output:
    def test_from_confidence_to_sigma_result(self):
        result = stut.from_confidence_to_sigma(np.array([0.68, 0.95, 0.997]))
        known_result = np.array([0.99445788, 1.95996398, 2.96773793])
        assert np.allclose(result, known_result)

    def test_from_sigma_to_confidence_result(self):
        result = stut.from_sigma_to_confidence(np.array([1., 2., 3.]))
        known_result = np.array([0.68268949, 0.95449974, 0.9973002])
        assert np.allclose(result, known_result)

    # test that one function is the inverse of the other:
    def test_sigma_confidence_inverse(self):
        test_numbers = np.arange(1, 6)
        test_confidence = stut.from_sigma_to_confidence(test_numbers)
        test_sigma = stut.from_confidence_to_sigma(test_confidence)
        assert np.allclose(test_numbers, test_sigma)

    # test raises:
    def test_errors(self):
        with self.assertRaises(ValueError):
            stut.from_confidence_to_sigma(-1.)
        with self.assertRaises(ValueError):
            stut.from_confidence_to_sigma(2.)
        with self.assertRaises(ValueError):
            stut.from_sigma_to_confidence(-1.)

###############################################################################


class test_chi2_to_sigma(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        assert np.allclose(stut.from_chi2_to_sigma(1., 1.), 1.0)
        assert np.allclose(stut.from_chi2_to_sigma(5.0, 2.),
                           stut.from_chi2_to_sigma(5.0, 2., 1000))
        assert np.allclose(stut.from_chi2_to_sigma(20.0, 2.),
                           stut.from_chi2_to_sigma(20.0, 2., 9))

    # test raises:
    def test_errors(self):
        with self.assertRaises(ValueError):
            stut.from_chi2_to_sigma(-2., 2.)
        with self.assertRaises(ValueError):
            stut.from_chi2_to_sigma(2., -2.)

###############################################################################


class test_KL_decomposition(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        # generate two random positive matrices:
        self.mat_1 = np.random.rand(10)
        self.mat_2 = np.random.rand(10)
        self.mat_1 = stut.vector_to_PDM(self.mat_1)
        self.mat_2 = stut.vector_to_PDM(self.mat_2)

    # test values:
    def test_values(self):
        # test with random matrices:
        stut.KL_decomposition(self.mat_1, self.mat_2)
        # test that, if the second matrix is the identity then this is equal to eigenvalues:
        kl_eig, kl_eigv = stut.KL_decomposition(self.mat_1, np.identity(self.mat_2.shape[0]))
        eig, eigv = np.linalg.eigh(self.mat_1)
        assert np.allclose(eig, kl_eig)
        assert np.allclose(eigv, kl_eigv)

    # test raises:
    def test_errors(self):
        d = 10
        wrong_mat = np.random.rand(d, d)
        right_mat = stut.vector_to_PDM(np.random.rand(d*(d+1)//2))
        with self.assertRaises(ValueError):
            stut.KL_decomposition(right_mat, wrong_mat)


###############################################################################


class test_QR_inverse(unittest.TestCase):

    def setUp(self):
        d = 10
        self.mat = stut.vector_to_PDM(np.random.rand(d*(d+1)//2))

    # test values:
    def test_values(self):
        assert np.allclose(np.linalg.inv(self.mat), stut.QR_inverse(self.mat))

###############################################################################


class test_clopper_pearson_binomial_trial(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        low, high = stut.clopper_pearson_binomial_trial(1., 2.)

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_PDM_vectorization(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        # generate a random vector between -1, 1 (seeded so reproducible):
        np.random.seed(0)
        # sweep dimensions from low to medium
        for d in range(2, 20):
            num = d*(d+1)//2
            # get some random matrices:
            for i in range(10):
                vec = 2.*np.random.rand(num) -1.
                # get the corresponding PDM matrix:
                mat = stut.vector_to_PDM(vec)
                # check that it is positive definite:
                assert np.all(np.linalg.eig(mat)[0] > 0)
                # transform back. This can be different from the previous one
                # because of many discrete symmetries in defining the
                # eigenvectors
                vec2 = stut.PDM_to_vector(mat)
                # transform again. This has to be equal, discrete symmetries for
                # eigenvectors do not matter once they are paired with eigenvalues:
                mat2 = stut.vector_to_PDM(vec2)
                assert np.allclose(mat, mat2)

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_make_list(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        _input = ['1', '2']
        _list = stut.make_list(_input)
        _list = stut.make_list('1')

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_bernoulli_thin(unittest.TestCase):

    def setUp(self):
        self.chain = helper_MCsamples()

    # test values:
    def test_values(self):
        stut.bernoulli_thin(self.chain)

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_random_samples_reshuffle(unittest.TestCase):

    def setUp(self):
        self.chain = helper_MCsamples()

    # test values:
    def test_values(self):
        stut.random_samples_reshuffle(self.chain)
        self.chain.loglikes = None
        stut.random_samples_reshuffle(self.chain)

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_whiten_samples(unittest.TestCase):

    def setUp(self):
        self.chain = helper_MCsamples()

    # test values:
    def test_values(self):
        stut.whiten_samples(self.chain.samples, self.chain.weights)

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_filter_kwargs(unittest.TestCase):

    def setUp(self):
        pass

    # test values:
    def test_values(self):
        stut.filter_kwargs({}, stut.filter_kwargs)

    # test raises:
    def test_errors(self):
        pass

###############################################################################


class test_is_outlier(unittest.TestCase):

    def setUp(self):
        self.chain = helper_MCsamples(ndim=1)

    # test values:
    def test_values(self):
        stut.is_outlier(self.chain.samples)
        stut.is_outlier(self.chain.samples[0])

    # test raises:
    def test_errors(self):
        pass

###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
