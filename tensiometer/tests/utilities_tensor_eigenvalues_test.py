###############################################################################
# initial imports:

import unittest

import tensiometer.utilities.tensor_eigenvalues as te

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

class TestTensorDeflation(unittest.TestCase):

    def setUp(self):
        """ Set up test data. """
        self.A_2D = np.array([[1, 2], [3, 4]])  # A simple 2x2 matrix
        self.A_3D = np.random.rand(3, 3, 3)    # A random 3x3x3 tensor
        self.x_2D = np.array([1, 1])           # A vector of ones for 2D
        self.x_3D = np.array([1, 0, -1])       # A sample vector for 3D
        self.l = 2                             # Scalar to deflate by

    def test_tensor_deflation_2D(self):
        """ Test deflation on a 2D matrix with a simple vector. """
        result = te.tensor_deflation(self.A_2D, self.l, self.x_2D)
        
        # Manually calculating expected deflation
        outer_product = np.outer(self.x_2D, self.x_2D)
        expected = self.A_2D - self.l * outer_product
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_deflation_3D(self):
        """ Test deflation on a 3D tensor with a vector. """
        result = te.tensor_deflation(self.A_3D, self.l, self.x_3D)
        
        # Construct the outer product manually for 3D
        outer_product = np.multiply.outer(self.x_3D, np.outer(self.x_3D, self.x_3D))
        expected = self.A_3D - self.l * outer_product
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_deflation_zero_scalar(self):
        """ Test deflation when scalar is zero (should return the original tensor). """
        result = te.tensor_deflation(self.A_2D, 0, self.x_2D)
        
        # If scalar is zero, no deflation should occur
        expected = self.A_2D
        
        # Asserting equality
        np.testing.assert_array_equal(result, expected)

    def test_tensor_deflation_zero_vector(self):
        """ Test deflation with a zero vector (should return the original tensor). """
        zero_vector = np.zeros_like(self.x_2D)
        result = te.tensor_deflation(self.A_2D, self.l, zero_vector)
        
        # With a zero vector, no deflation should occur
        expected = self.A_2D
        
        # Asserting equality
        np.testing.assert_array_equal(result, expected)


###############################################################################

class TestTensorContraction(unittest.TestCase):

    def setUp(self):
        """ Set up test data. """
        # A simple 2x2 matrix (rank 2 tensor)
        self.A_2D = np.array([[1, 2], [3, 4]])
        # A random 3x3x3 tensor (rank 3 tensor)
        self.A_3D = np.random.rand(3, 3, 3)
        # Sample vectors for contraction
        self.x_2D = np.array([1, 1])      # For contracting a 2D tensor
        self.x_3D = np.array([1, 0, -1])  # For contracting a 3D tensor

    def test_tensor_contraction_2D_n1(self):
        """ Test contraction of a 2D tensor with a vector, contracted once (n=1). """
        result = te.tensor_contraction_brute_1(self.A_2D, self.x_2D, n=1)
        
        # Expected result: dot product of A and x_2D
        expected = np.dot(self.A_2D, self.x_2D)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_2D_n2(self):
        """ Test full contraction of a 2D tensor (contracted twice, n=2). """
        result = te.tensor_contraction_brute_1(self.A_2D, self.x_2D, n=2)
        
        # Expected result: dot product of (A * x_2D) and x_2D
        intermediate = np.dot(self.A_2D, self.x_2D)
        expected = np.dot(intermediate, self.x_2D)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_3D_n1(self):
        """ Test contraction of a 3D tensor with a vector, contracted once (n=1). """
        result = te.tensor_contraction_brute_1(self.A_3D, self.x_3D, n=1)
        
        # Expected result: contraction along the first axis (dot product with x_3D)
        expected = np.tensordot(self.A_3D, self.x_3D, axes=1)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_3D_n2(self):
        """ Test contraction of a 3D tensor with a vector, contracted twice (n=2). """
        result = te.tensor_contraction_brute_1(self.A_3D, self.x_3D, n=2)
        
        # Expected result: contract the tensor twice
        intermediate = np.tensordot(self.A_3D, self.x_3D, axes=1)
        expected = np.tensordot(intermediate, self.x_3D, axes=1)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_3D_n3(self):
        """ Test full contraction of a 3D tensor (contracted three times, n=3). """
        result = te.tensor_contraction_brute_1(self.A_3D, self.x_3D, n=3)
        
        # Expected result: fully contracted into a scalar
        intermediate_1 = np.tensordot(self.A_3D, self.x_3D, axes=1)
        intermediate_2 = np.tensordot(intermediate_1, self.x_3D, axes=1)
        expected = np.dot(intermediate_2, self.x_3D)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_zero_contracts(self):
        """ Test contraction with n=0 (should return the original tensor). """
        result = te.tensor_contraction_brute_1(self.A_2D, self.x_2D, n=0)
        
        # No contraction, so the result should be the original tensor
        expected = self.A_2D
        
        # Asserting equality
        np.testing.assert_array_equal(result, expected)

###############################################################################

class TestTensorContractionBrute2(unittest.TestCase):

    def setUp(self):
        """ Set up test data. """
        # A simple 2x2 matrix (rank 2 tensor)
        self.A_2D = np.array([[1, 2], [3, 4]])
        # A random 3x3x3 tensor (rank 3 tensor)
        self.A_3D = np.random.rand(3, 3, 3)
        # Sample vectors for contraction
        self.x_2D = np.array([1, 1])      # For contracting a 2D tensor
        self.x_3D = np.array([1, 0, -1])  # For contracting a 3D tensor

    def test_tensor_contraction_2D_n1(self):
        """ Test contraction of a 2D tensor with a vector, contracted once (n=1). """
        result = te.tensor_contraction_brute_2(self.A_2D, self.x_2D, n=1)
        
        # Expected result: dot product of A and x_2D
        expected = np.dot(self.A_2D, self.x_2D)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_2D_n2(self):
        """ Test full contraction of a 2D tensor (contracted twice, n=2). """
        result = te.tensor_contraction_brute_2(self.A_2D, self.x_2D, n=2)
        
        # Expected result: dot product of (A * x_2D) and x_2D
        intermediate = np.dot(self.A_2D, self.x_2D)
        expected = np.dot(intermediate, self.x_2D)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_3D_n1(self):
        """ Test contraction of a 3D tensor with a vector, contracted once (n=1). """
        result = te.tensor_contraction_brute_2(self.A_3D, self.x_3D, n=1)
        
        # Expected result: contraction along the first axis (dot product with x_3D)
        expected = np.tensordot(self.A_3D, self.x_3D, axes=1)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_3D_n2(self):
        """ Test contraction of a 3D tensor with a vector, contracted twice (n=2). """
        result = te.tensor_contraction_brute_2(self.A_3D, self.x_3D, n=2)
        
        # Expected result: contract the tensor twice
        intermediate = np.tensordot(self.A_3D, self.x_3D, axes=1)
        expected = np.tensordot(intermediate, self.x_3D, axes=1)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_3D_n3(self):
        """ Test full contraction of a 3D tensor (contracted three times, n=3). """
        result = te.tensor_contraction_brute_2(self.A_3D, self.x_3D, n=3)
        
        # Expected result: fully contracted into a scalar
        intermediate_1 = np.tensordot(self.A_3D, self.x_3D, axes=1)
        intermediate_2 = np.tensordot(intermediate_1, self.x_3D, axes=1)
        expected = np.dot(intermediate_2, self.x_3D)
        
        # Asserting equality
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensor_contraction_zero_contracts(self):
        """ Test contraction with n=0 (should return the original tensor). """
        result = te.tensor_contraction_brute_2(self.A_2D, self.x_2D, n=0)
        
        # No contraction, so the result should be the original tensor
        expected = self.A_2D
        
        # Asserting equality
        np.testing.assert_array_equal(result, expected)

###############################################################################

class TestSphereOptimization(unittest.TestCase):

    def setUp(self):
        """ Set up test data. """
        # A simple 3D unit vector on the sphere
        self.x = np.array([1.0, 0.0, 0.0])
        # Random 3D vector to simulate a Euclidean gradient
        self.egrad = np.array([0.5, 0.1, -0.2])
        # Random 3x3 Hessian matrix
        self.ehess = np.array([[0.2, 0.1, 0.05],
                               [0.1, 0.3, -0.1],
                               [0.05, -0.1, 0.4]])
        # Tangent vector for the sphere
        self.u = np.array([0.0, 1.0, 0.0])

    def test_eu_to_sphere_grad(self):
        """ Test Euclidean gradient conversion to spherical gradient. """
        result = te.eu_to_sphere_grad(self.x, self.egrad)
        
        # Expected gradient projection: egrad - <x, egrad> * x
        dot_product = np.dot(self.x, self.egrad)
        expected = self.egrad - dot_product * self.x
        
        # Asserting the spherical gradient is computed correctly
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_eu_to_sphere_grad_orthogonality(self):
        """ Test if the spherical gradient is orthogonal to the vector x (tangent space). """
        result = te.eu_to_sphere_grad(self.x, self.egrad)
        
        # The result should be orthogonal to x, i.e., dot(result, x) == 0
        dot_product = np.dot(result, self.x)
        
        # Assert that the spherical gradient is orthogonal to the vector on the sphere
        self.assertAlmostEqual(dot_product, 0.0, places=5)

    def test_eu_to_sphere_grad_zero_gradient(self):
        """ Test if the spherical gradient is zero when egrad is zero. """
        zero_egrad = np.zeros_like(self.egrad)
        result = te.eu_to_sphere_grad(self.x, zero_egrad)
        
        # The gradient should also be zero
        expected = zero_egrad
        
        # Asserting the result is zero
        np.testing.assert_array_equal(result, expected)

    def test_eu_to_sphere_hess(self):
        """ Test the Euclidean Hessian conversion to spherical Hessian. """
        result = te.eu_to_sphere_hess(self.x, self.egrad, self.ehess, self.u)
        
        # Manually compute the expected spherical Hessian
        ehess_dot_u = np.dot(self.ehess, self.u)
        expected = ehess_dot_u - np.dot(self.x, ehess_dot_u) * self.x \
                   - np.dot(self.x, self.egrad) * self.u
        
        # Asserting the spherical Hessian is computed correctly
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_eu_to_sphere_hess_orthogonality(self):
        """ Test if the spherical Hessian is orthogonal to the vector x (tangent space). """
        result = te.eu_to_sphere_hess(self.x, self.egrad, self.ehess, self.u)
        
        # The result should be orthogonal to x, i.e., dot(result, x) == 0
        dot_product = np.dot(result, self.x)
        
        # Assert that the spherical Hessian is orthogonal to the vector on the sphere
        self.assertAlmostEqual(dot_product, 0.0, places=5)
        
###############################################################################

if __name__ == '__main__':
    unittest.main(verbosity=2)