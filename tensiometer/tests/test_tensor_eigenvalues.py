###############################################################################
# initial imports:

import unittest

import tensiometer.tensor_eigenvalues as te

import numpy as np

###############################################################################


class test_utilities(unittest.TestCase):

    def setUp(self):
        pass

    def test_random_symm_tensor(self):
        assert te.random_symm_tensor(d=2, m=2).shape == (2, 2)
        assert te.random_symm_tensor(d=2, m=4).shape == (2, 2, 2, 2)
        assert te.random_symm_tensor(d=4, m=2).shape == (4, 4)
        assert te.random_symm_tensor(d=4, m=4).shape == (4, 4, 4, 4)
        temp = te.random_symm_tensor(d=8, m=2)
        assert np.allclose(temp, temp.T)
        temp = te.random_symm_tensor(d=8, m=4)
        assert np.allclose(temp, temp.T)

    def test_random_symm_positive_tensor(self):
        assert te.random_symm_positive_tensor(d=2, m=2).shape == (2, 2)
        assert te.random_symm_positive_tensor(d=2, m=4).shape == (2, 2, 2, 2)
        assert te.random_symm_positive_tensor(d=4, m=2).shape == (4, 4)
        assert te.random_symm_positive_tensor(d=4, m=4).shape == (4, 4, 4, 4)
        temp = te.random_symm_positive_tensor(d=8, m=2)
        assert np.allclose(temp, temp.T)
        temp = te.random_symm_positive_tensor(d=8, m=4)
        assert np.allclose(temp, temp.T)

    def test_identity_tensor(self):
        assert te.identity_tensor(d=2, m=2).shape == (2, 2)

###############################################################################


if __name__ == '__main__':
    unittest.main(verbosity=2)
