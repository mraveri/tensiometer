"""Tests for tensor eigenvalue utilities."""

#########################################################################################################
# Imports

import unittest
from unittest.mock import patch

import numpy as np

import tensiometer.utilities.tensor_eigenvalues as te

#########################################################################################################
# Tensor Eigenvalue Utilities tests


class TestTensorEigenvalueUtilities(unittest.TestCase):

    """Tensor eigenvalue utility test suite."""
    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_random_symm_tensor(self):
        """Test Random symm tensor."""
        assert te.random_symm_tensor(d=2, m=2).shape == (2, 2)
        assert te.random_symm_tensor(d=2, m=4).shape == (2, 2, 2, 2)
        assert te.random_symm_tensor(d=4, m=2).shape == (4, 4)
        assert te.random_symm_tensor(d=4, m=4).shape == (4, 4, 4, 4)
        temp = te.random_symm_tensor(d=8, m=2)
        assert np.allclose(temp, temp.T)
        temp = te.random_symm_tensor(d=8, m=4)
        assert np.allclose(temp, temp.T)

    def test_random_symm_positive_tensor(self):
        """Test Random symm positive tensor."""
        assert te.random_symm_positive_tensor(d=2, m=2).shape == (2, 2)
        assert te.random_symm_positive_tensor(d=2, m=4).shape == (2, 2, 2, 2)
        assert te.random_symm_positive_tensor(d=4, m=2).shape == (4, 4)
        assert te.random_symm_positive_tensor(d=4, m=4).shape == (4, 4, 4, 4)
        temp = te.random_symm_positive_tensor(d=8, m=2)
        assert np.allclose(temp, temp.T)
        temp = te.random_symm_positive_tensor(d=8, m=4)
        assert np.allclose(temp, temp.T)

    def test_identity_tensor(self):
        """Test Identity tensor."""
        assert te.identity_tensor(d=2, m=2).shape == (2, 2)

#########################################################################################################
# Random Symm Positive Tensor tests

class TestRandomSymmPositiveTensor(unittest.TestCase):

    """Test Random Symm Positive Tensor test suite."""
    def test_random_symm_positive_tensor(self):
        """Test Random symm positive tensor."""
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

#########################################################################################################
# Identity Tensor tests

class TestIdentityTensor(unittest.TestCase):

    """Test Identity Tensor test suite."""
    def test_identity_tensor(self):
        """Test Identity tensor."""
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

#########################################################################################################
# Number Eigenvalues tests

class TestNumberEigenvalues(unittest.TestCase):

    """Test Number Eigenvalues test suite."""
    def test_number_eigenvalues(self):
        """Test Number eigenvalues."""
        self.assertEqual(te.number_eigenvalues(d=2, m=2), 2)
        self.assertEqual(te.number_eigenvalues(d=2, m=4), 6)
        self.assertEqual(te.number_eigenvalues(d=4, m=2), 4)
        self.assertEqual(te.number_eigenvalues(d=4, m=4), 108)
        self.assertEqual(te.number_eigenvalues(d=8, m=2), 8)
        self.assertEqual(te.number_eigenvalues(d=8, m=4), 17496)

#########################################################################################################
# Tensor Deflation tests

class TestTensorDeflation(unittest.TestCase):

    """Test Tensor Deflation test suite."""
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


#########################################################################################################
# Tensor Contraction tests

class TestTensorContraction(unittest.TestCase):

    """Test Tensor Contraction test suite."""
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

#########################################################################################################
# Tensor Contraction Brute2 tests

class TestTensorContractionBrute2(unittest.TestCase):

    """Test Tensor Contraction Brute2 test suite."""
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

#########################################################################################################
# Sphere Optimization tests

class TestSphereOptimization(unittest.TestCase):

    """Test Sphere Optimization test suite."""
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
        
#########################################################################################################
# Tensor Eigenvalues Extra tests


class TestTensorEigenvaluesExtra(unittest.TestCase):
    """Tensor eigenvalues extra test suite."""
    def test_random_symmetry_helpers(self):
        """Test Random symmetry helpers."""
        tens = te.random_symm_tensor(3, 3, vmin=0.0, vmax=0.1)
        self.assertTrue(np.allclose(tens, np.transpose(tens, axes=(1, 0, 2))))
        pos_tens = te.random_symm_positive_tensor(2, 4, vmin=0.0, vmax=0.1)
        self.assertTrue(np.all(pos_tens >= 0))
        ident = te.identity_tensor(2, 3)
        self.assertEqual(ident.shape, (2, 2, 2))
        self.assertEqual(te.number_eigenvalues(2, 3), 4)

    def test_tensor_contractions_and_deflation(self):
        """Test Tensor contractions and deflation."""
        A = np.ones((2, 2, 2))
        x = np.array([1.0, 2.0])
        res1 = te.tensor_contraction_brute_1(A, x, n=2)
        res2 = te.tensor_contraction_brute_2(A, x, n=2)
        self.assertTrue(np.allclose(res1, res2))
        deflated = te.tensor_deflation(A, l=1.5, x=np.array([1.0, 0.0]))
        self.assertEqual(deflated.shape, A.shape)

    def test_sphere_transforms(self):
        """Test Sphere transforms."""
        x = np.array([1.0, 0.0])
        egrad = np.array([2.0, 3.0])
        ehess = np.array([[1.0, 0.0], [0.0, 1.0]])
        u = np.array([0.0, 1.0])
        grad = te.eu_to_sphere_grad(x, egrad)
        self.assertAlmostEqual(np.dot(grad, x), 0.0)
        hess = te.eu_to_sphere_hess(x, egrad, ehess, u)
        self.assertEqual(hess.shape, (2,))

    def test_rayleigh_helpers(self):
        """Test Rayleigh helpers."""
        A = np.eye(2)
        x = np.array([1.0, 0.0])
        self.assertAlmostEqual(te.tRq(x, A), 1.0)
        res = te.tRq_nder(x, A, 1)
        self.assertTrue(np.allclose(res, np.array([2.0, 0.0])))

    def test_brute_autograd_and_max(self):
        """Test Brute autograd and max."""
        A = np.eye(2)
        # monkeypatch solvers to avoid heavy optimization
        original_ps = te.pymanopt.solvers.ParticleSwarm
        original_tr = te.pymanopt.solvers.TrustRegions

        class DummySolver:
            """Dummy Solver test suite."""
            def __init__(self, *args, **kwargs):
                """Init."""
                pass

            def solve(self, problem):
                """Solve."""
                return np.array([1.0, 0.0])

        te.pymanopt.solvers.ParticleSwarm = DummySolver
        te.pymanopt.solvers.TrustRegions = DummySolver
        try:
            val, vec = te.max_tRq_brute(A, optimizer="ParticleSwarm")
            self.assertAlmostEqual(val, 1.0)
            self.assertTrue(np.allclose(vec, np.array([1.0, 0.0])))
            val2, vec2 = te.max_GtRq_brute(A, A, optimizer="TrustRegions")
            self.assertTrue(np.isfinite(val2))
            self.assertEqual(vec2.shape[0], 2)
        finally:
            te.pymanopt.solvers.ParticleSwarm = original_ps
            te.pymanopt.solvers.TrustRegions = original_tr

    def test_brute_2d(self):
        """Test Brute 2d."""
        A = np.eye(2)
        eig, eigv = te.tRq_brute_2D(A, num_points=10)
        self.assertEqual(eig.shape[0], eigv.shape[0])
        eig2, eigv2 = te.GtRq_brute_2D(A, A, num_points=10)
        self.assertEqual(eig2.shape[0], eigv2.shape[0])

    def test_power_method(self):
        """Test Power method."""
        A = np.eye(2)
        val, vec = te.max_tRq_power(A, maxiter=5, tol=1e-6, x0=np.array([1.0, 0.0]), history=False)
        self.assertAlmostEqual(val, 1.0)
        self.assertTrue(np.allclose(vec, np.array([1.0, 0.0])))


class TestTensorEigenvaluesBranch(unittest.TestCase):
    """Tensor eigenvalues branch test suite."""
    def test_max_tRq_brute_trustregions_branch(self):
        """Test Max t Rq brute trustregions branch."""
        class DummySolver:
            """Dummy Solver test suite."""
            def __init__(self, *args, **kwargs):
                """Init."""
                pass

            def solve(self, problem):
                """Solve."""
                return np.array([1.0, 0.0])

        with patch.object(te.pymanopt.solvers, "TrustRegions", DummySolver):
            val, vec = te.max_tRq_brute(np.eye(2), optimizer="TrustRegions")

        self.assertAlmostEqual(val, 1.0)
        self.assertTrue(np.allclose(vec, np.array([1.0, 0.0])))

    def test_max_tRq_power_warning_and_history(self):
        """Test Max t Rq power warning and history."""
        np.random.seed(0)
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        val, vec, history = te.max_tRq_power(A, maxiter=1, tol=0.0, x0=None, history=True)
        self.assertEqual(len(history), 1)
        self.assertEqual(vec.shape, (2,))
        self.assertTrue(np.isfinite(val))

    def test_max_tRq_shift_power_warning_and_history(self):
        """Test Max t Rq shift power warning and history."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        val, vec, history = te.max_tRq_shift_power(A, alpha=0.1, maxiter=1, tol=0.0, x0=None, history=True)
        self.assertEqual(len(history), 1)
        self.assertEqual(vec.shape, (2,))
        self.assertTrue(np.isfinite(val))

    def test_max_tRq_geap_warning_and_history(self):
        """Test Max t Rq geap warning and history."""
        A = np.eye(2)
        val, vec, history = te.max_tRq_geap(A, maxiter=1, tol=0.0, x0=None, history=True)
        self.assertEqual(history.shape[0], 1)
        self.assertEqual(vec.shape, (2,))
        self.assertTrue(np.isfinite(val))

    def test_max_tRq_shift_power_break_and_no_history(self):
        """Test Max t Rq shift power break and no history."""
        A = np.eye(2)
        val, vec = te.max_tRq_shift_power(A, alpha=0.0, maxiter=3, tol=1.0, x0=np.array([1.0, 0.0]), history=False)
        self.assertAlmostEqual(val, 1.0)
        self.assertEqual(vec.shape, (2,))

    def test_max_tRq_geap_break_no_history(self):
        """Test Max t Rq geap break no history."""
        A = np.eye(2)
        val, vec = te.max_tRq_geap(A, maxiter=3, tol=1.0, x0=np.array([1.0, 0.0]), history=False)
        self.assertAlmostEqual(val, 1.0)
        self.assertEqual(vec.shape, (2,))

    def test_tRq_dyn_sys_brute_basic(self):
        """Test T Rq dyn sys brute basic."""
        out = te.tRq_dyn_sys_brute(0.0, np.array([1.0, 0.0]), np.eye(2), d=2, m=2)
        np.testing.assert_array_almost_equal(out, np.zeros(2))

    def test_max_tRq_dynsys_warning_path(self):
        """Test Max t Rq dynsys warning path."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])

        def fake_odeint(func, y0, t, h0=None, full_output=False, **kwargs):
            """Fake odeint."""
            return np.stack([y0, y0 * 0.5]), {"hu": [h0 or 0.5]}

        with patch("tensiometer.utilities.tensor_eigenvalues.scipy.integrate.odeint", side_effect=fake_odeint):
            val, vec = te.max_tRq_dynsys(A, maxiter=1, tol=0.0, x0=None, h0=0.5, history=False)

        self.assertEqual(vec.shape, (2,))
        self.assertTrue(np.isfinite(val))

    def test_max_tRq_dynsys_break_by_h0_and_history(self):
        """Test Max t Rq dynsys break by h0 and history."""
        A = np.eye(2)

        def fake_odeint(func, y0, t, h0=None, full_output=False, **kwargs):
            """Fake odeint."""
            return np.stack([y0, y0]), {"hu": [h0 or 1.0]}

        with patch("tensiometer.utilities.tensor_eigenvalues.scipy.integrate.odeint", side_effect=fake_odeint):
            val, vec, hist = te.max_tRq_dynsys(
                A, maxiter=1, tol=1.0, x0=np.array([1.0, 0.0]), h0=1.0, history=True
            )

        self.assertEqual(vec.shape, (2,))
        self.assertEqual(hist.shape[0], 1)
        self.assertAlmostEqual(val, 1.0)

    def test_max_tRq_dynsys_h0_limit_break(self):
        """Test Max t Rq dynsys h0 limit break."""
        A = np.eye(2)

        def fake_odeint(func, y0, t, h0=None, full_output=False, **kwargs):
            """Fake odeint."""
            return np.stack([y0, y0]), {"hu": [h0]}

        with patch("tensiometer.utilities.tensor_eigenvalues.scipy.integrate.odeint", side_effect=fake_odeint):
            val, vec = te.max_tRq_dynsys(
                A, maxiter=2, tol=0.0, x0=np.array([0.5, 0.5]), h0=2e6, history=False
            )

        self.assertEqual(vec.shape, (2,))
        self.assertTrue(np.isfinite(val))

    def test_GtRq_Jac_brute(self):
        """Test Gt Rq Jac brute."""
        res = te.GtRq_Jac_brute(
            x=np.array([1.0, 0.0]),
            m=2,
            Axm=2.0,
            Bxm=1.0,
            Axmm1=np.array([2.0, 0.0]),
            Bxmm1=np.array([1.0, 0.0]),
        )
        np.testing.assert_array_almost_equal(res, np.array([4.0, 0.0]))

    def test_max_GtRq_brute_trustregions_branch(self):
        """Test Max Gt Rq brute trustregions branch."""
        class DummySolver:
            """Dummy Solver test suite."""
            def __init__(self, *args, **kwargs):
                """Init."""
                pass

            def solve(self, problem):
                """Solve."""
                return np.array([1.0, 0.0])

        with patch.object(te.pymanopt.solvers, "TrustRegions", DummySolver):
            val, vec = te.max_GtRq_brute(np.eye(2), np.eye(2), optimizer="TrustRegions")

        self.assertTrue(np.isfinite(val))
        self.assertEqual(vec.shape, (2,))

    def test_max_GtRq_brute_particleswarm_branch(self):
        """Test Max Gt Rq brute particleswarm branch."""
        class DummySolver:
            """Dummy Solver test suite."""
            def __init__(self, *args, **kwargs):
                """Init."""
                pass

            def solve(self, problem):
                """Solve."""
                return np.array([1.0, 0.0])

        with patch.object(te.pymanopt.solvers, "ParticleSwarm", DummySolver):
            val, vec = te.max_GtRq_brute(np.eye(2), np.eye(2), optimizer="ParticleSwarm")

        self.assertTrue(np.isfinite(val))
        self.assertEqual(vec.shape, (2,))

    def test_max_GtRq_geap_power_history_and_warning(self):
        """Test Max Gt Rq geap power history and warning."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        # maxiter None triggers sys.maxsize fallback and history return
        lam, vec, hist = te.max_GtRq_geap_power(A, np.eye(2), maxiter=None, tol=1.0, history=True)
        self.assertTrue(np.isfinite(lam))
        self.assertEqual(vec.shape, (2,))
        self.assertIsInstance(hist, np.ndarray)

        # small maxiter enforces warning path
        lam2, vec2 = te.max_GtRq_geap_power(
            A, np.eye(2), maxiter=1, tol=0.0, x0=np.array([1.0, 1.0]), history=False
        )
        self.assertTrue(np.isfinite(lam2))
        self.assertEqual(vec2.shape, (2,))

    def test_max_GtRq_geap_power_warning_banner(self):
        """Test Max Gt Rq geap power warning banner."""
        A = np.array([[2.0, 0.0], [0.0, 1.0]])
        lam, vec = te.max_GtRq_geap_power(
            A, np.eye(2), maxiter=10002, tol=-1.0, x0=np.array([0.6, 0.8]), history=False
        )
        self.assertTrue(np.isfinite(lam))
        self.assertEqual(vec.shape, (2,))

    def test_invalid_optimizer_paths_raise(self):
        """Test Invalid optimizer paths raise."""
        with self.assertRaises(NameError):
            te.max_tRq_brute(np.eye(1), optimizer="Unknown")

        with self.assertRaises(NameError):
            te.max_GtRq_brute(np.eye(1), np.eye(1), optimizer="Unknown")


#########################################################################################################
# Script entry point

if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
