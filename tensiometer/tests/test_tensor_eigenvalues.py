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

class TestRandomSymmPositiveTensor(unittest.TestCase):

    def test_random_symm_positive_tensor(self):
        tensor = te.random_symm_positive_tensor(d=2, m=2)
        self.assertEqual(tensor.shape, (2, 2))
        
        tensor = te.random_symm_positive_tensor(d=2, m=4)
        self.assertEqual(tensor.shape, (2, 2, 2, 2))
        
        tensor = te.random_symm_positive_tensor(d=4, m=2)
        self.assertEqual(tensor.shape, (4, 4))
        
        tensor = te.random_symm_positive_tensor(d=4, m=4)
        self.assertEqual(tensor.shape, (4, 4, 4, 4))
        
        temp = te.random_symm_positive_tensor(d=8, m=2)
        self.assertTrue(np.allclose(temp, temp.T))
        
        temp = te.random_symm_positive_tensor(d=8, m=4)
        self.assertTrue(np.allclose(temp, temp.T))

###############################################################################

class TestIdentityTensor(unittest.TestCase):

    def test_identity_tensor(self):
        tensor = te.identity_tensor(d=2, m=2)
        self.assertEqual(tensor.shape, (2, 2))

        tensor = te.identity_tensor(d=3, m=3)
        self.assertEqual(tensor.shape, (3, 3, 3))

        tensor = te.identity_tensor(d=4, m=2)
        self.assertEqual(tensor.shape, (4, 4))

        tensor = te.identity_tensor(d=5, m=4)
        self.assertEqual(tensor.shape, (5, 5, 5, 5))

        tensor = te.identity_tensor(d=6, m=3)
        self.assertEqual(tensor.shape, (6, 6, 6))

        tensor = te.identity_tensor(d=7, m=2)
        self.assertEqual(tensor.shape, (7, 7))

        tensor = te.identity_tensor(d=8, m=4)
        self.assertEqual(tensor.shape, (8, 8, 8, 8))

        tensor = te.identity_tensor(d=9, m=3)
        self.assertEqual(tensor.shape, (9, 9, 9))

        tensor = te.identity_tensor(d=10, m=2)
        self.assertEqual(tensor.shape, (10, 10))

###############################################################################
class TestNumberEigenvalues(unittest.TestCase):

    def test_number_eigenvalues(self):
        self.assertEqual(te.number_eigenvalues(d=2, m=2), 2)
        self.assertEqual(te.number_eigenvalues(d=2, m=4), 6)
        self.assertEqual(te.number_eigenvalues(d=4, m=2), 4)
        self.assertEqual(te.number_eigenvalues(d=4, m=4), 108)
        self.assertEqual(te.number_eigenvalues(d=8, m=2), 8)
        self.assertEqual(te.number_eigenvalues(d=8, m=4), 17496)

###############################################################################

if __name__ == '__main__':
    unittest.main(verbosity=2)