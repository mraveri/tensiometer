"""Behaviour tests for the PyTorch bijectors, distributions and autodiff helpers."""

#########################################################################################################
# Imports

import io
import math
import pickle
import unittest

import numpy as np
import torch
from scipy import stats

from tensiometer.synthetic_probability import autodiff
from tensiometer.synthetic_probability import bijectors as bj
from tensiometer.synthetic_probability import distributions as dist
from tensiometer.synthetic_probability import fixed_bijectors as fb
from tensiometer.synthetic_probability import tensor_utilities as tu
from tensiometer.synthetic_probability import trainable_bijectors as tb

#########################################################################################################
# Helper functions


def _tolerance():
    """Absolute and relative tolerance for the active precision.

    :returns: 1e-10 in float64, 1e-4 in float32.
    """
    if tu.get_precision() == torch.float64:
        return 1e-10
    return 1e-4


def _assert_close(actual, expected, tolerance=None):
    """Compare two tensors or arrays on the host.

    :param actual: computed value.
    :param expected: reference value.
    :param tolerance: absolute and relative tolerance, defaults to :func:`_tolerance`.
    """
    if tolerance is None:
        tolerance = _tolerance()
    np.testing.assert_allclose(tu.to_numpy(actual), tu.to_numpy(expected), rtol=tolerance, atol=tolerance)


def _randomize_parameters(module, scale=0.1):
    """Overwrite all the parameters of a module with small Gaussian values.

    :param module: ``torch.nn.Module``.
    :param scale: standard deviation of the values.
    """
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(scale * torch.randn_like(parameter))


def _random_tril(dimension, random_state):
    """Random lower triangular matrix with positive diagonal.

    :param dimension: matrix size ``D``.
    :param random_state: numpy random generator.
    :returns: array of shape ``(D, D)``.
    """
    matrix = np.tril(0.3 * random_state.standard_normal((dimension, dimension)), -1)
    return matrix + np.diag(random_state.uniform(0.5, 2.0, dimension))


def _random_spline_parameters(dimension, knots, range_min, range_max, random_state):
    """Random bins and slopes of a rational quadratic spline for each dimension.

    :param dimension: number of dimensions ``D``.
    :param knots: number of bins ``K``.
    :param range_min: lower end of the domain.
    :param range_max: upper end of the domain.
    :param random_state: numpy random generator.
    :returns: tuple ``(widths, heights, slopes)`` of shapes ``(D, K)``, ``(D, K)``, ``(D, K - 1)``.
    """
    width = range_max - range_min
    widths = random_state.uniform(0.5, 1.5, (dimension, knots))
    widths = width * widths / widths.sum(axis=-1, keepdims=True)
    heights = random_state.uniform(0.5, 1.5, (dimension, knots))
    heights = width * heights / heights.sum(axis=-1, keepdims=True)
    slopes = random_state.uniform(0.3, 2.5, (dimension, knots - 1))
    return widths, heights, slopes


def _sinh_log_det(x):
    """Log derivative of ``sinh``."""
    return torch.log(torch.cosh(x))


def _vector_forward(x):
    """Nonlinear triangular map ``(sinh(x0), x_rest * exp(x0))``."""
    return torch.cat([torch.sinh(x[..., :1]), x[..., 1:] * torch.exp(x[..., :1])], dim=-1)


def _vector_inverse(y):
    """Inverse of :func:`_vector_forward`."""
    first = torch.asinh(y[..., :1])
    return torch.cat([first, y[..., 1:] * torch.exp(-first)], dim=-1)


def _vector_log_det(x):
    """Log determinant of :func:`_vector_forward`."""
    return torch.log(torch.cosh(x[..., 0])) + (x.shape[-1] - 1) * x[..., 0]


def _make_maf(dimension, transformer, hidden_units=None):
    """Masked autoregressive flow with small random weights and biases.

    :param dimension: number of dimensions ``D``.
    :param transformer: affine or spline transformer.
    :param hidden_units: hidden layer sizes (default ``[8, 8]``).
    :returns: :class:`MaskedAutoregressiveFlow`.
    """
    if hidden_units is None:
        hidden_units = [8, 8]
    network = tb.MaskedAutoregressiveNetwork(dimension, transformer.num_params, hidden_units)
    _randomize_parameters(network, scale=0.1)
    return tb.MaskedAutoregressiveFlow(network, transformer)


def _bijector_catalog(dimension=4, seed=0):
    """Build one instance of every bijector together with inputs in its domain.

    :param dimension: number of dimensions ``D``.
    :param seed: random seed.
    :returns: dictionary ``name -> (bijector, x)`` with ``x`` of shape ``(N, D)``.
    """
    torch.manual_seed(seed)
    random_state = np.random.default_rng(seed)
    num_samples = 16
    shape = (num_samples, dimension)
    x_wide = random_state.uniform(-2.5, 2.5, shape)
    x_narrow = random_state.uniform(-0.9, 0.9, shape)
    widths, heights, slopes = _random_spline_parameters(dimension, 5, -2., 2., random_state)
    boundary = random_state.uniform(0.5, 2.0, dimension)
    catalog = {
        'identity': (bj.Identity(), x_wide),
        'shift': (bj.Shift(random_state.standard_normal(dimension)), x_wide),
        'scale': (bj.Scale(np.array([2., 0.5, -1.5, 3.])[:dimension]), x_wide),
        'log_scale': (bj.Scale(log_scale=random_state.standard_normal(dimension)), x_wide),
        'tanh': (bj.Tanh(), x_wide),
        'normal_cdf': (bj.NormalCDF(), 0.8 * x_wide),
        'permute': (bj.Permute(np.roll(np.arange(dimension), 1)), x_wide),
        'affine_tril': (bj.AffineTriL(random_state.standard_normal(dimension),
                                      _random_tril(dimension, random_state)), x_wide),
        'invert_tanh': (bj.Invert(bj.Tanh()), x_narrow),
        'chain': (bj.Chain([bj.Shift(1.), bj.Scale(2.), bj.Tanh()]), x_wide),
        'blockwise': (bj.Blockwise([bj.Scale(2.),
                                    bj.AffineTriL(np.ones(2), _random_tril(2, random_state)),
                                    bj.Tanh()],
                                   block_sizes=[1, 2, dimension - 3]), x_wide),
        'inline_elementwise': (bj.Inline(torch.sinh, torch.asinh,
                                         forward_log_det_jacobian_fn=_sinh_log_det,
                                         forward_min_event_ndims=0), x_wide),
        'inline_autodiff': (bj.Inline(_vector_forward, _vector_inverse), x_wide),
        'spline': (tb.RationalQuadraticSpline(widths, heights, slopes, range_min=-2.), x_wide),
        'circular_spline': (tb.RationalQuadraticSpline(widths, heights, slopes, range_min=-2.,
                                                       boundary_knot_slope=boundary), x_wide),
        'mod1d': (fb.Mod1D(-1., 2.), random_state.uniform(-1., 2., shape)),
        'uniform_prior': (fb.uniform_prior(-1., 3.), x_wide),
        'scale_roto_shift': (tb.ScaleRotoShift(dimension), x_wide),
        'affine_maf': (_make_maf(dimension, tb.AffineTransformer()), x_wide),
        'spline_maf': (_make_maf(dimension, tb.SplineTransformer(spline_knots=4, range_max=3.)), x_wide),
        'circular_maf': (_make_maf(dimension, tb.SplineTransformer(spline_knots=4, range_max=3., circular=True)),
                         x_wide),
        'flex_spline_maf': (tb.MaskedAutoregressiveFlow(
            tb.FlexAutoregressiveNetwork(dimension, tb.SplineTransformer(spline_knots=4, range_max=3.).num_params,
                                         [6]),
            tb.SplineTransformer(spline_knots=4, range_max=3.)), x_wide),
        'autoregressive_flow': (tb.AutoregressiveFlow(
            dimension, transformation_type=['affine', 'spline'], autoregressive_type=['masked', 'flex'],
            n_transformations=2, hidden_units=[8], range_max=3., scale_roto_shift=True,
            permutations=[np.roll(np.arange(dimension), 1), np.arange(dimension)[::-1]]).bijector, x_wide),
    }
    _randomize_parameters(catalog['flex_spline_maf'][0], scale=0.1)
    return catalog


def _forward_jacobian(bijector, x):
    """Batch Jacobian of the forward map.

    :param bijector: bijector.
    :param x: inputs of shape ``(N, D)``.
    :returns: tensor of shape ``(N, D, D)``.
    """
    x = autodiff.prepare_input(x)
    return autodiff.batch_jacobian(bijector.forward, x)


def _inverse_jacobian(bijector, y):
    """Batch Jacobian of the inverse map.

    :param bijector: bijector.
    :param y: inputs of shape ``(N, D)``.
    :returns: tensor of shape ``(N, D, D)``.
    """
    y = autodiff.prepare_input(tu.to_tensor(tu.to_numpy(y)))
    return autodiff.batch_jacobian(bijector.inverse, y)

#########################################################################################################
# Generic bijector properties


class TestBijectorCatalog(unittest.TestCase):
    """Properties shared by every bijector."""

    @classmethod
    def setUpClass(cls):
        """Build the bijector catalog once."""
        cls.catalog = _bijector_catalog()

    def test_round_trip(self):
        """Test that inverse(forward(x)) recovers x."""
        for name, (bijector, x) in self.catalog.items():
            with self.subTest(bijector=name):
                y = bijector.forward(x)
                self.assertEqual(y.dtype, tu.get_precision())
                self.assertEqual(tuple(y.shape), x.shape)
                _assert_close(bijector.inverse(y), x)

    def test_call_matches_forward(self):
        """Test that calling a bijector applies the forward map."""
        for name, (bijector, x) in self.catalog.items():
            with self.subTest(bijector=name):
                _assert_close(bijector(x), bijector.forward(x), tolerance=0.)

    def test_forward_log_det_is_minus_inverse_log_det(self):
        """Test that fldj(x) == -ildj(forward(x))."""
        for name, (bijector, x) in self.catalog.items():
            with self.subTest(bijector=name):
                y = bijector.forward(x)
                fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
                ildj = bijector.inverse_log_det_jacobian(y, event_ndims=1)
                self.assertEqual(tuple(fldj.shape), (x.shape[0],))
                self.assertEqual(fldj.dtype, tu.get_precision())
                _assert_close(fldj, -ildj)

    def test_log_det_matches_slogdet_of_jacobian(self):
        """Test fldj and ildj against slogdet of the autodiff Jacobian."""
        for name, (bijector, x) in self.catalog.items():
            with self.subTest(bijector=name):
                jacobian = _forward_jacobian(bijector, x)
                self.assertEqual(tuple(jacobian.shape), (x.shape[0], x.shape[1], x.shape[1]))
                slogdet = torch.linalg.slogdet(jacobian)[1]
                _assert_close(bijector.forward_log_det_jacobian(x, event_ndims=1), slogdet)
                y = bijector.forward(x)
                inverse_slogdet = torch.linalg.slogdet(_inverse_jacobian(bijector, y))[1]
                _assert_close(bijector.inverse_log_det_jacobian(y, event_ndims=1), inverse_slogdet)

    def test_joint_methods_match_separate_calls(self):
        """Test forward_and_log_det_jacobian and inverse_and_log_det_jacobian."""
        for name, (bijector, x) in self.catalog.items():
            with self.subTest(bijector=name):
                y, fldj = bijector.forward_and_log_det_jacobian(x, event_ndims=1)
                _assert_close(y, bijector.forward(x))
                _assert_close(fldj, bijector.forward_log_det_jacobian(x, event_ndims=1))
                x_back, ildj = bijector.inverse_and_log_det_jacobian(y, event_ndims=1)
                _assert_close(x_back, bijector.inverse(y))
                _assert_close(ildj, bijector.inverse_log_det_jacobian(y, event_ndims=1))

    def test_torch_save_and_pickle(self):
        """Test that every bijector survives torch.save/torch.load and pickle."""
        for name, (bijector, x) in self.catalog.items():
            with self.subTest(bijector=name):
                y = bijector.forward(x)
                fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
                buffer = io.BytesIO()
                torch.save(bijector, buffer)
                buffer.seek(0)
                restored_torch = torch.load(buffer, weights_only=False)
                restored_pickle = pickle.loads(pickle.dumps(bijector))
                for restored in (restored_torch, restored_pickle):
                    self.assertEqual(restored.name, bijector.name)
                    np.testing.assert_array_equal(tu.to_numpy(restored.forward(x)), tu.to_numpy(y))
                    np.testing.assert_array_equal(tu.to_numpy(restored.inverse(y)), tu.to_numpy(bijector.inverse(y)))
                    np.testing.assert_array_equal(
                        tu.to_numpy(restored.forward_log_det_jacobian(x, event_ndims=1)), tu.to_numpy(fldj))

    def test_numpy_and_list_inputs(self):
        """Test that array-like inputs are converted to the active precision."""
        for name, (bijector, x) in self.catalog.items():
            with self.subTest(bijector=name):
                from_numpy = bijector.forward(np.asarray(x, dtype=np.float64))
                from_list = bijector.forward(x.tolist())
                self.assertTrue(torch.is_tensor(from_numpy))
                self.assertEqual(from_numpy.dtype, tu.get_precision())
                _assert_close(from_numpy, from_list, tolerance=0.)

#########################################################################################################
# Base class and event dimensions


class _NoLogDet(bj.Bijector):
    """Bijector without log determinant hooks."""

    def _forward(self, x):
        """Double the input."""
        return 2. * x

    def _inverse(self, y):
        """Halve the input."""
        return y / 2.


class _InverseLogDetOnly(_NoLogDet):
    """Doubling bijector that only implements a constant, scalar inverse log determinant."""

    def _inverse_log_det_jacobian(self, y):
        """Constant ``-log 2`` per element, as a scalar."""
        return torch.tensor(-math.log(2.), dtype=y.dtype)


class TestBijectorBase(unittest.TestCase):
    """Bijector base class behaviour."""

    def test_default_name(self):
        """Test the default lowercase class name."""
        self.assertEqual(bj.Tanh().name, 'tanh')
        self.assertEqual(bj.Shift(1., name='custom').name, 'custom')
        self.assertEqual(bj.Invert(bj.Tanh()).name, 'invert_tanh')

    def test_missing_log_det_raises(self):
        """Test that a bijector without log determinant hooks raises."""
        bijector = _NoLogDet()
        x = torch.ones(3, 2, dtype=tu.get_precision())
        _assert_close(bijector.inverse(bijector.forward(x)), x)
        with self.assertRaises(NotImplementedError):
            bijector.forward_log_det_jacobian(x)
        with self.assertRaises(NotImplementedError):
            bijector.inverse_log_det_jacobian(x)

    def test_base_class_hooks_not_implemented(self):
        """Test that the bare base class implements neither map."""
        bijector = bj.Bijector()
        x = torch.ones(2, 3, dtype=tu.get_precision())
        with self.assertRaises(NotImplementedError):
            bijector.forward(x)
        with self.assertRaises(NotImplementedError):
            bijector.inverse(x)

    def test_forward_log_det_from_inverse(self):
        """Test that the forward log determinant is derived from an inverse-only hook."""
        bijector = _InverseLogDetOnly()
        x = torch.linspace(-1., 1., 6, dtype=tu.get_precision()).reshape(3, 2)
        expected = np.full(3, 2. * math.log(2.))
        _assert_close(bijector.forward_log_det_jacobian(x, event_ndims=1), expected)
        _assert_close(bijector.inverse_log_det_jacobian(bijector.forward(x), event_ndims=1), -expected)

    def test_scalar_log_det_is_broadcast(self):
        """Test that a constant log determinant is broadcast to the batch shape before the reduction."""
        bijector = _InverseLogDetOnly()
        y = torch.zeros(4, 3, dtype=tu.get_precision())
        elementwise = bijector.inverse_log_det_jacobian(y, event_ndims=0)
        self.assertEqual(tuple(elementwise.shape), (4, 3))
        _assert_close(elementwise, np.full((4, 3), -math.log(2.)))
        _assert_close(bijector.inverse_log_det_jacobian(y, event_ndims=2), -12. * math.log(2.))

    def test_extra_repr(self):
        """Test that the module representation shows the bijector name."""
        self.assertEqual(bj.Shift(1., name='my_shift').extra_repr(), "name='my_shift'")
        self.assertIn("name='my_shift'", repr(bj.Shift(1., name='my_shift')))

    def test_inputs_moved_to_bijector_device(self):
        """Test that inputs on another device are moved to the device of the bijector parameters."""
        bijector = bj.Shift(np.array([1., 2.])).to('meta')
        y = bijector.forward(torch.zeros(3, 2, dtype=tu.get_precision()))
        self.assertEqual(y.device.type, 'meta')
        self.assertEqual(tuple(y.shape), (3, 2))
        self.assertEqual(y.dtype, tu.get_precision())

    def test_event_ndims_validation(self):
        """Test that event_ndims below min_event_ndims raises ValueError."""
        x = torch.zeros(5, 3, dtype=tu.get_precision())
        vector_bijectors = [bj.Permute([1, 2, 0]), bj.AffineTriL(np.zeros(3), np.eye(3)),
                            bj.Chain([bj.Tanh(), bj.Permute([1, 2, 0])]), bj.Blockwise([bj.Tanh()], [3]),
                            tb.ScaleRotoShift(3)]
        for bijector in vector_bijectors:
            with self.subTest(bijector=bijector.name):
                self.assertEqual(bijector.min_event_ndims, 1)
                with self.assertRaises(ValueError):
                    bijector.forward_log_det_jacobian(x, event_ndims=0)
                with self.assertRaises(ValueError):
                    bijector.inverse_log_det_jacobian(x, event_ndims=0)
                with self.assertRaises(ValueError):
                    bijector.forward_and_log_det_jacobian(x, event_ndims=0)

    def test_event_ndims_reduction(self):
        """Test the shapes and sums of log determinants for different event_ndims."""
        x = torch.linspace(-1., 1., 24, dtype=tu.get_precision()).reshape(2, 4, 3)
        tanh = bj.Tanh()
        elementwise = tanh.forward_log_det_jacobian(x, event_ndims=0)
        self.assertEqual(tuple(elementwise.shape), (2, 4, 3))
        _assert_close(tanh.forward_log_det_jacobian(x, event_ndims=1), elementwise.sum(-1))
        _assert_close(tanh.forward_log_det_jacobian(x, event_ndims=2), elementwise.sum((-1, -2)))
        permute = bj.Permute([2, 0, 1])
        self.assertEqual(tuple(permute.forward_log_det_jacobian(x, event_ndims=1).shape), (2, 4))
        self.assertEqual(tuple(permute.forward_log_det_jacobian(x, event_ndims=2).shape), (2,))
        scale = bj.Scale(3.)
        _assert_close(scale.forward_log_det_jacobian(x, event_ndims=2), np.full(2, 12. * math.log(3.)))

    def test_extra_batch_dimensions(self):
        """Test that vector bijectors act on inputs with extra batch axes."""
        catalog = _bijector_catalog(dimension=4, seed=1)
        for name in ['affine_tril', 'scale_roto_shift', 'spline_maf', 'blockwise']:
            bijector, x = catalog[name]
            with self.subTest(bijector=name):
                x = tu.to_tensor(np.stack([x, 0.5 * x], axis=1))
                y = bijector.forward(x)
                self.assertEqual(tuple(y.shape), tuple(x.shape))
                _assert_close(bijector.inverse(y), x)
                fldj = bijector.forward_log_det_jacobian(x, event_ndims=1)
                self.assertEqual(tuple(fldj.shape), tuple(x.shape[:2]))
                _assert_close(fldj[:, 1], bijector.forward_log_det_jacobian(x[:, 1], event_ndims=1))

#########################################################################################################
# Fixed bijectors


class TestFixedBijectors(unittest.TestCase):
    """Specific checks of the fixed bijectors."""

    def test_chain_order(self):
        """Test that Chain applies its bijectors last to first."""
        chain = bj.Chain([bj.Shift(1.), bj.Scale(2.)])
        x = torch.linspace(-2., 2., 7, dtype=tu.get_precision()).reshape(7, 1)
        _assert_close(chain(x), 2. * x + 1.)
        _assert_close(chain.inverse(2. * x + 1.), x)
        _assert_close(chain.forward_log_det_jacobian(x), np.full(7, math.log(2.)))
        self.assertEqual(len(chain.bijectors), 2)
        self.assertIsInstance(chain.bijectors, torch.nn.ModuleList)

    def test_empty_chain_is_identity(self):
        """Test that an empty Chain is the identity."""
        chain = bj.Chain()
        x = np.array([[1., 2.]])
        _assert_close(chain.forward(x), x, tolerance=0.)
        _assert_close(chain.inverse(x), x, tolerance=0.)
        _assert_close(chain.forward_log_det_jacobian(x), np.zeros(1), tolerance=0.)

    def test_blockwise_matches_manual(self):
        """Test Blockwise with uneven block sizes against a manual computation."""
        random_state = np.random.default_rng(3)
        tril = _random_tril(2, random_state)
        blocks = [bj.Scale(2.), bj.AffineTriL(np.array([1., -1.]), tril), bj.Tanh()]
        blockwise = bj.Blockwise(blocks, block_sizes=[1, 2, 1])
        x = tu.to_tensor(random_state.standard_normal((6, 4)))
        expected = torch.cat([blocks[0].forward(x[:, :1]), blocks[1].forward(x[:, 1:3]),
                              blocks[2].forward(x[:, 3:])], dim=-1)
        _assert_close(blockwise.forward(x), expected)
        expected_log_det = (math.log(2.) + np.sum(np.log(np.diag(tril)))
                            + tu.to_numpy(blocks[2].forward_log_det_jacobian(x[:, 3:], event_ndims=1)))
        _assert_close(blockwise.forward_log_det_jacobian(x), expected_log_det)
        _assert_close(blockwise.inverse(expected), x)

    def test_blockwise_errors(self):
        """Test Blockwise input validation."""
        with self.assertRaises(ValueError):
            bj.Blockwise([bj.Tanh(), bj.Tanh()], block_sizes=[1])
        blockwise = bj.Blockwise([bj.Tanh(), bj.Identity()])
        self.assertEqual(blockwise.block_sizes, [1, 1])
        with self.assertRaises(ValueError):
            blockwise.forward(torch.zeros(2, 3, dtype=tu.get_precision()))

    def test_permute(self):
        """Test Permute forward and inverse."""
        permutation = [2, 0, 3, 1]
        permute = bj.Permute(permutation)
        x = tu.to_tensor(np.arange(8.).reshape(2, 4))
        y = permute.forward(x)
        _assert_close(y, tu.to_numpy(x)[:, permutation], tolerance=0.)
        _assert_close(permute.inverse(y), x, tolerance=0.)
        _assert_close(permute.forward_log_det_jacobian(x), np.zeros(2), tolerance=0.)
        with self.assertRaises(ValueError):
            bj.Permute([0, 0, 1])
        with self.assertRaises(ValueError):
            bj.Permute([[0, 1], [1, 0]])

    def test_scale_arguments(self):
        """Test the Scale constructor options."""
        x = torch.ones(2, 1, dtype=tu.get_precision())
        _assert_close(bj.Scale(log_scale=math.log(3.)).forward(x), 3. * x)
        _assert_close(bj.Scale(-2.).forward_log_det_jacobian(x), np.full(2, math.log(2.)))
        with self.assertRaises(ValueError):
            bj.Scale()
        with self.assertRaises(ValueError):
            bj.Scale(scale=1., log_scale=0.)

    def test_invert(self):
        """Test that Invert swaps the maps and log determinants."""
        tanh = bj.Tanh()
        inverted = bj.Invert(tanh)
        y = torch.linspace(-0.9, 0.9, 5, dtype=tu.get_precision()).reshape(5, 1)
        _assert_close(inverted.forward(y), torch.atanh(y))
        _assert_close(inverted.inverse(torch.atanh(y)), y)
        _assert_close(inverted.forward_log_det_jacobian(y), tanh.inverse_log_det_jacobian(y))
        self.assertEqual(inverted.min_event_ndims, tanh.min_event_ndims)

    def test_affine_tril_unbatched_input(self):
        """Test AffineTriL on a single vector."""
        tril = np.array([[2., 0.], [0.5, 1.5]])
        affine = bj.AffineTriL(np.array([1., 2.]), tril)
        x = np.array([0.3, -0.7])
        y = affine.forward(x)
        _assert_close(y, tril @ x + np.array([1., 2.]))
        _assert_close(affine.inverse(y), x)
        _assert_close(affine.forward_log_det_jacobian(x), np.log(3.))

    def test_normal_cdf_matches_scipy(self):
        """Test NormalCDF against scipy.stats.norm."""
        normal_cdf = bj.NormalCDF()
        x = np.linspace(-5., 5., 41)
        _assert_close(normal_cdf.forward(x), stats.norm.cdf(x), tolerance=1e-6)
        p = np.linspace(0.01, 0.99, 41)
        _assert_close(normal_cdf.inverse(p), stats.norm.ppf(p), tolerance=max(_tolerance(), 1e-5))
        _assert_close(normal_cdf.forward_log_det_jacobian(x, event_ndims=0), stats.norm.logpdf(x))
        edges = normal_cdf.inverse(np.array([0., 1.]))
        self.assertTrue(torch.all(torch.isfinite(edges)))
        self.assertLess(float(edges[0]), -5.)
        self.assertGreater(float(edges[1]), 5.)

    def test_normal_cdf_edges_are_symmetric(self):
        """Both edges of [0, 1] get the same bounded inverse and log determinant; outside is NaN."""
        normal_cdf = bj.NormalCDF()
        edges = normal_cdf.inverse(np.array([0., 1.]))
        self.assertAlmostEqual(float(edges[0]), -float(edges[1]), places=4)
        log_dets = normal_cdf.inverse_log_det_jacobian(np.array([0., 1.]), event_ndims=0)
        self.assertAlmostEqual(float(log_dets[0]), float(log_dets[1]), places=3)
        self.assertLess(float(log_dets[0]), 40.)
        outside = normal_cdf.inverse(np.array([-0.1, 1.1]))
        self.assertTrue(torch.all(torch.isnan(outside)))

    def test_tanh_log_det_is_stable(self):
        """Test the Tanh log determinant for large inputs."""
        tanh = bj.Tanh()
        x = np.array([-30., -3., 0., 3., 30.])
        log_det = tanh.forward_log_det_jacobian(x, event_ndims=0)
        self.assertTrue(torch.all(torch.isfinite(log_det)))
        _assert_close(log_det[1:4], np.log(1. - np.tanh(x[1:4])**2))

#########################################################################################################
# Inline bijector


class TestInline(unittest.TestCase):
    """Inline bijector with and without log determinant functions."""

    def setUp(self):
        """Common inputs."""
        self.x = tu.to_tensor(np.random.default_rng(4).uniform(-1.5, 1.5, (7, 3)))

    def test_elementwise_with_log_det(self):
        """Test an elementwise Inline with an explicit log determinant."""
        inline = bj.Inline(torch.sinh, torch.asinh, forward_log_det_jacobian_fn=_sinh_log_det,
                           forward_min_event_ndims=0, name='sinh')
        self.assertEqual(inline.name, 'sinh')
        self.assertEqual(inline.min_event_ndims, 0)
        _assert_close(inline.forward(self.x), torch.sinh(self.x))
        _assert_close(inline.forward_log_det_jacobian(self.x, event_ndims=0), _sinh_log_det(self.x))
        y = inline.forward(self.x)
        _assert_close(inline.inverse_log_det_jacobian(y, event_ndims=0), -_sinh_log_det(self.x))

    def test_elementwise_autodiff_fallback(self):
        """Test the autodiff log determinant of an elementwise Inline."""
        inline = bj.Inline(torch.sinh, torch.asinh, forward_min_event_ndims=0)
        _assert_close(inline.forward_log_det_jacobian(self.x, event_ndims=0), _sinh_log_det(self.x))
        y = inline.forward(self.x)
        _assert_close(inline.inverse_log_det_jacobian(y, event_ndims=0), -_sinh_log_det(self.x))

    def test_vector_autodiff_fallback(self):
        """Test the autodiff log determinant of a vector Inline."""
        inline = bj.Inline(_vector_forward, _vector_inverse)
        self.assertEqual(inline.min_event_ndims, 1)
        _assert_close(inline.forward_log_det_jacobian(self.x), _vector_log_det(self.x))
        y = inline.forward(self.x)
        _assert_close(inline.inverse_log_det_jacobian(y), -_vector_log_det(self.x))
        detached = inline.forward_log_det_jacobian(self.x)
        self.assertFalse(detached.requires_grad)

    def test_autodiff_fallback_is_differentiable(self):
        """Test that the autodiff log determinant keeps the graph of an input requiring gradients."""
        inline = bj.Inline(_vector_forward, _vector_inverse)
        x = autodiff.prepare_input(self.x)
        log_det = inline.forward_log_det_jacobian(x)
        (gradient,) = torch.autograd.grad(log_det.sum(), x)
        expected = torch.zeros_like(self.x)
        expected[:, 0] = torch.tanh(self.x[:, 0]) + (self.x.shape[1] - 1)
        _assert_close(gradient, expected)

    def test_single_log_det_function(self):
        """Test Inline with only one of the two log determinant functions."""
        only_forward = bj.Inline(_vector_forward, _vector_inverse, forward_log_det_jacobian_fn=_vector_log_det)
        only_inverse = bj.Inline(_vector_forward, _vector_inverse,
                                 inverse_log_det_jacobian_fn=lambda y: -_vector_log_det(_vector_inverse(y)))
        y = only_forward.forward(self.x)
        for inline in (only_forward, only_inverse):
            _assert_close(inline.forward_log_det_jacobian(self.x), _vector_log_det(self.x))
            _assert_close(inline.inverse_log_det_jacobian(y), -_vector_log_det(self.x))

    def test_errors(self):
        """Test Inline error handling."""
        with self.assertRaises(ValueError):
            bj.Inline(torch.sinh, torch.asinh, forward_min_event_ndims=2)
        with self.assertRaises(NotImplementedError):
            bj.Inline(inverse_fn=torch.asinh).forward(self.x)
        with self.assertRaises(NotImplementedError):
            bj.Inline(forward_fn=torch.sinh).inverse(self.x)

#########################################################################################################
# Rational quadratic splines


class TestSplines(unittest.TestCase):
    """Rational quadratic spline properties."""

    def setUp(self):
        """Random spline parameters for two dimensions."""
        random_state = np.random.default_rng(5)
        self.range_min = -2.
        self.range_max = 3.
        self.widths, self.heights, self.slopes = _random_spline_parameters(
            2, 6, self.range_min, self.range_max, random_state)
        self.boundary = np.array([0.7, 1.8])
        self.spline = tb.RationalQuadraticSpline(self.widths, self.heights, self.slopes, range_min=self.range_min)
        self.circular = tb.RationalQuadraticSpline(self.widths, self.heights, self.slopes, range_min=self.range_min,
                                                   boundary_knot_slope=self.boundary)

    def test_monotonic(self):
        """Test that sorted inputs give strictly sorted outputs."""
        x = np.repeat(np.linspace(-4., 5., 400)[:, None], 2, axis=1)
        for spline in (self.spline, self.circular):
            y = tu.to_numpy(spline.forward(x))
            self.assertTrue(np.all(np.diff(y, axis=0) > 0.))

    def test_continuity_at_knots(self):
        """Test values and slopes at the knots, including the domain bounds."""
        knots_x = self.range_min + np.concatenate([np.zeros((2, 1)), np.cumsum(self.widths, axis=-1)], axis=-1)
        knots_y = self.range_min + np.concatenate([np.zeros((2, 1)), np.cumsum(self.heights, axis=-1)], axis=-1)
        x = knots_x.T
        _assert_close(self.spline.forward(x), knots_y.T, tolerance=max(_tolerance(), 1e-5))
        _assert_close(self.spline.inverse(knots_y.T), x, tolerance=max(_tolerance(), 1e-5))
        epsilon = 1e-3
        below = tu.to_numpy(self.spline.forward(x - epsilon))
        above = tu.to_numpy(self.spline.forward(x + epsilon))
        self.assertTrue(np.all(np.abs(above - below) < 10. * epsilon))
        interior_log_slope = tu.to_numpy(self.spline.forward_log_det_jacobian(x[1:-1], event_ndims=0))
        _assert_close(interior_log_slope, np.log(self.slopes.T), tolerance=max(_tolerance(), 1e-5))

    def test_standard_identity_tails(self):
        """Test that the standard spline is the identity outside the domain."""
        x = np.array([[-10., -2.5], [3.5, 12.]])
        _assert_close(self.spline.forward(x), x, tolerance=0.)
        _assert_close(self.spline.inverse(x), x, tolerance=0.)
        _assert_close(self.spline.forward_log_det_jacobian(x, event_ndims=0), np.zeros_like(x), tolerance=0.)

    def test_circular_linear_tails(self):
        """Test the linear tails of the circular spline."""
        x = np.array([[-10., -2.5], [3.5, 12.]])
        expected = np.where(x > self.range_max,
                            self.boundary * (x - self.range_max) + self.range_max,
                            self.boundary * (x - self.range_min) + self.range_min)
        _assert_close(self.circular.forward(x), expected)
        _assert_close(self.circular.inverse(expected), x)
        _assert_close(self.circular.forward_log_det_jacobian(x, event_ndims=0),
                      np.log(self.boundary) * np.ones_like(x))

    def test_positive_slopes_with_slope_min(self):
        """Test that the transformer slopes stay above slope_min for extreme parameters."""
        slope_min = 1e-3
        for circular in (False, True):
            transformer = tb.SplineTransformer(spline_knots=4, range_max=2., slope_min=slope_min, circular=circular)
            params = torch.cat([torch.zeros(3, 2, 2 * 4, dtype=tu.get_precision()),
                                -1e3 * torch.ones(3, 2, transformer.num_params - 8, dtype=tu.get_precision())], -1)
            _, _, slopes, boundary = transformer.spline_parameters(params)
            self.assertTrue(torch.all(slopes >= slope_min * (1. - 1e-6)))
            if circular:
                self.assertTrue(torch.all(boundary >= slope_min * (1. - 1e-6)))
            y = transformer.forward(torch.zeros(3, 2, dtype=tu.get_precision()), params)
            self.assertTrue(torch.all(torch.isfinite(y)))

#########################################################################################################
# Autoregressive networks and flows


class TestAutoregressive(unittest.TestCase):
    """MADE and masked autoregressive flow structure."""

    def test_made_output_depends_on_previous_inputs(self):
        """Test that MADE output d depends only on inputs < d."""
        torch.manual_seed(0)
        dimension, params = 4, 3
        for hidden_units in ([6, 6], []):
            network = tb.MaskedAutoregressiveNetwork(dimension, params, hidden_units)
            _randomize_parameters(network, scale=0.5)
            x = autodiff.prepare_input(np.random.default_rng(0).standard_normal((5, dimension)))
            output = network(x)
            self.assertEqual(tuple(output.shape), (5, dimension, params))
            jacobian = tu.to_numpy(autodiff.batch_jacobian(network, x))
            self.assertEqual(jacobian.shape, (5, dimension, params, dimension))
            for d in range(dimension):
                with self.subTest(hidden_units=hidden_units, output=d):
                    np.testing.assert_array_equal(jacobian[:, d, :, d:], 0.)
                    if d > 0:
                        self.assertTrue(np.all(np.any(jacobian[:, d, :, :d] != 0., axis=(1, 2))))

    def test_made_one_dimension_is_bias_only(self):
        """Test that a one dimensional MADE has a constant output."""
        network = tb.MaskedAutoregressiveNetwork(1, 2, [4])
        _randomize_parameters(network, scale=0.5)
        output = network(tu.to_tensor(np.array([[-1.], [0.], [2.]])))
        self.assertEqual(tuple(output.shape), (3, 1, 2))
        _assert_close(output[0], output[1])
        _assert_close(output[0], output[2])
        x = autodiff.prepare_input(np.array([[-1.], [0.], [2.]]))
        np.testing.assert_array_equal(tu.to_numpy(autodiff.batch_jacobian(network, x)), 0.)

    def test_maf_jacobians_are_lower_triangular(self):
        """Test that MAF Jacobians are lower triangular with the identity permutation."""
        torch.manual_seed(1)
        dimension = 4
        x = np.random.default_rng(1).uniform(-2., 2., (6, dimension))
        for transformer in (tb.AffineTransformer(), tb.SplineTransformer(spline_knots=4, range_max=3.)):
            maf = _make_maf(dimension, transformer)
            y = maf.forward(x)
            for jacobian in (_forward_jacobian(maf, x), _inverse_jacobian(maf, y)):
                jacobian = tu.to_numpy(jacobian)
                upper = np.triu(np.ones((dimension, dimension), dtype=bool), 1)
                np.testing.assert_array_equal(jacobian[:, upper], 0.)
                self.assertTrue(np.any(jacobian[:, np.tril(np.ones((dimension, dimension), dtype=bool), -1)] != 0.))
            inverse_diagonal = np.diagonal(tu.to_numpy(_inverse_jacobian(maf, y)), axis1=1, axis2=2)
            _assert_close(maf.inverse_log_det_jacobian(y), np.sum(np.log(np.abs(inverse_diagonal)), axis=-1))

    def test_third_derivatives_through_spline_maf(self):
        """Test that third derivatives through a spline MAF are finite."""
        torch.manual_seed(2)
        maf = _make_maf(3, tb.SplineTransformer(spline_knots=4, range_max=3.))
        y = autodiff.prepare_input(np.random.default_rng(2).uniform(-2., 2., (5, 3)))
        value = maf.inverse_log_det_jacobian(y) + maf.inverse(y).pow(2).sum(-1)
        first = torch.autograd.grad(value.sum(), y, create_graph=True)[0]
        second = torch.autograd.grad(first[:, 0].sum(), y, create_graph=True)[0]
        third = torch.autograd.grad(second[:, 0].sum(), y, create_graph=True)[0]
        for derivative in (first, second, third):
            self.assertTrue(torch.all(torch.isfinite(derivative)))
        self.assertTrue(torch.any(third != 0.))

    def test_variance_scaling_matches_keras_formula(self):
        """Test the sample standard deviation of the VarianceScaling initializer."""
        torch.manual_seed(3)
        scale, fan_out, fan_in = 2., 300, 500
        weight = torch.empty(fan_out, fan_in, dtype=tu.get_precision())
        tb.VarianceScaling(scale)(weight)
        expected_std = math.sqrt(scale / ((fan_in + fan_out) / 2.))
        truncation = 2. * expected_std / 0.87962566103423978
        self.assertAlmostEqual(float(weight.std()) / expected_std, 1., delta=0.02)
        self.assertLess(abs(float(weight.mean())), 0.05 * expected_std)
        self.assertLessEqual(float(weight.abs().max()), truncation * (1. + 1e-6))

    def test_manual_seed_determinism(self):
        """Test that torch.manual_seed makes the initialization reproducible."""
        states = []
        for seed in (7, 7, 8):
            torch.manual_seed(seed)
            flow = tb.AutoregressiveFlow(3, transformation_type='spline', n_transformations=2,
                                         hidden_units=[4], permutations=False, scale_roto_shift=True)
            states.append(flow.bijector.state_dict())
        for key in states[0].keys():
            self.assertTrue(torch.equal(states[0][key], states[1][key]))
        self.assertFalse(all(torch.equal(states[0][key], states[2][key]) for key in states[0].keys()))

#########################################################################################################
# Distributions


class TestDistributions(unittest.TestCase):
    """Transformed and mixture distributions."""

    def setUp(self):
        """A correlated Gaussian as a transformed standard normal."""
        self.mean = np.array([1., -2., 0.5])
        self.covariance = np.array([[2., 0.3, 0.1], [0.3, 1., -0.2], [0.1, -0.2, 0.5]])
        self.bijector = fb.multivariate_normal(self.mean, self.covariance)
        self.base = dist.standard_normal(3)
        self.distribution = dist.TransformedDistribution(self.base, self.bijector)
        self.x = np.random.default_rng(6).standard_normal((10, 3))

    def test_standard_normal(self):
        """Test the base distribution dtype and density."""
        log_prob = self.base.log_prob(tu.to_tensor(self.x))
        self.assertEqual(log_prob.dtype, tu.get_precision())
        _assert_close(log_prob, stats.multivariate_normal(np.zeros(3), np.eye(3)).logpdf(self.x))

    def test_transformed_log_prob(self):
        """Test TransformedDistribution.log_prob against base log_prob plus ildj and scipy."""
        log_prob = self.distribution.log_prob(tu.to_tensor(self.x))
        expected = (self.base.log_prob(self.bijector.inverse(self.x))
                    + self.bijector.inverse_log_det_jacobian(self.x, event_ndims=1))
        _assert_close(log_prob, expected)
        _assert_close(log_prob, stats.multivariate_normal(self.mean, self.covariance).logpdf(self.x))

    def test_transformed_log_prob_extra_batch_axis(self):
        """Test TransformedDistribution.log_prob on ``(N, M, D)`` inputs."""
        x = tu.to_tensor(self.x.reshape(5, 2, 3))
        log_prob = self.distribution.log_prob(x)
        self.assertEqual(tuple(log_prob.shape), (5, 2))
        _assert_close(log_prob.reshape(-1), self.distribution.log_prob(tu.to_tensor(self.x)))

    def test_transformed_sample(self):
        """Test TransformedDistribution.sample shape, moments and seeding."""
        torch.manual_seed(0)
        samples = self.distribution.sample(20000)
        self.assertEqual(tuple(samples.shape), (20000, 3))
        self.assertEqual(samples.dtype, tu.get_precision())
        np.testing.assert_allclose(tu.to_numpy(samples).mean(axis=0), self.mean, atol=0.05)
        np.testing.assert_allclose(np.cov(tu.to_numpy(samples).T), self.covariance, atol=0.06)
        torch.manual_seed(1)
        first = self.distribution.sample(5)
        torch.manual_seed(1)
        second = self.distribution.sample(5)
        self.assertTrue(torch.equal(first, second))

    def test_mixture_log_prob(self):
        """Test Mixture.log_prob against a logsumexp of the components."""
        other = dist.TransformedDistribution(self.base, bj.Shift(np.array([2., 0., -1.])))
        mixture = dist.Mixture([1., 3.], [self.distribution, other])
        _assert_close(mixture.weights, np.array([0.25, 0.75]))
        x = tu.to_tensor(self.x)
        log_probs = np.stack([tu.to_numpy(self.distribution.log_prob(x)), tu.to_numpy(other.log_prob(x))], axis=-1)
        expected = np.logaddexp(log_probs[:, 0] + np.log(0.25), log_probs[:, 1] + np.log(0.75))
        _assert_close(mixture.log_prob(x), expected)

    def test_mixture_sample_and_errors(self):
        """Test Mixture.sample and weight validation."""
        other = dist.TransformedDistribution(self.base, bj.Shift(np.array([50., 50., 50.])))
        mixture = dist.Mixture([0.2, 0.8], [self.distribution, other])
        torch.manual_seed(0)
        samples = tu.to_numpy(mixture.sample(4000))
        self.assertEqual(samples.shape, (4000, 3))
        fraction_far = np.mean(samples[:, 0] > 25.)
        self.assertAlmostEqual(fraction_far, 0.8, delta=0.03)
        self.assertGreater(np.mean(samples[:2000, 0] > 25.), 0.7)
        self.assertGreater(np.mean(samples[2000:, 0] > 25.), 0.7)
        with self.assertRaises(ValueError):
            dist.Mixture([0.5, 0.5], [self.distribution])

    def test_mixture_zero_samples(self):
        """Test that drawing zero samples from a mixture gives an empty ``(0, D)`` tensor."""
        other = dist.TransformedDistribution(self.base, bj.Shift(np.array([2., 0., -1.])))
        mixture = dist.Mixture([0.5, 0.5], [self.distribution, other])
        samples = mixture.sample(0)
        self.assertEqual(tuple(samples.shape), (0, 3))
        self.assertEqual(samples.dtype, tu.get_precision())

    def test_transformed_base_without_mean(self):
        """Test a transformed distribution whose base has no ``mean`` (a mixture) and hence no device."""
        other = dist.TransformedDistribution(self.base, bj.Shift(np.array([2., 0., -1.])))
        mixture = dist.Mixture([0.3, 0.7], [self.distribution, other])
        self.assertIsNone(dist._distribution_device(mixture))
        shift = bj.Shift(np.array([1., 1., 1.]))
        transformed = dist.TransformedDistribution(mixture, shift)
        x = tu.to_tensor(self.x)
        _assert_close(transformed.log_prob(x), mixture.log_prob(x - 1.))

    def test_transformed_base_on_other_device(self):
        """Test that base space points are moved to the device of the base distribution."""

        class MetaBase(object):
            """Base distribution living on the meta device that records its inputs."""

            def __init__(self):
                """Init."""
                self.mean = torch.zeros(3, device='meta')
                self.inputs = []

            def log_prob(self, z):
                """Zero log density on the device of ``z``."""
                self.inputs.append(z)
                return torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)

        base = MetaBase()
        transformed = dist.TransformedDistribution(base, bj.Shift(np.array([1., 2., 3.])))
        log_prob = transformed.log_prob(tu.to_tensor(self.x))
        self.assertEqual(base.inputs[0].device.type, 'meta')
        self.assertEqual(log_prob.device.type, 'meta')
        self.assertEqual(tuple(log_prob.shape), (10,))

#########################################################################################################
# Autodiff helpers


class TestAutodiff(unittest.TestCase):
    """Edge cases of the autodiff helpers."""

    def test_prepare_input_converts_graph_tensors(self):
        """Test that tensors requiring gradients are moved to the requested device and precision."""
        other = torch.float64 if tu.get_precision() == torch.float32 else torch.float32
        x = torch.ones(2, 3, dtype=other, requires_grad=True)
        prepared = autodiff.prepare_input(x)
        self.assertEqual(prepared.dtype, tu.get_precision())
        self.assertTrue(prepared.requires_grad)
        moved = autodiff.prepare_input(x, device='meta')
        self.assertEqual(moved.device.type, 'meta')
        self.assertEqual(moved.dtype, tu.get_precision())
        self.assertTrue(moved.requires_grad)
        same = torch.ones(2, 3, dtype=tu.get_precision(), requires_grad=True)
        self.assertIs(autodiff.prepare_input(same, device='cpu'), same)

    def test_gradient_of_input_independent_function(self):
        """Test that a graph-attached output not depending on the input has zero gradient."""
        weight = torch.ones(1, dtype=tu.get_precision(), requires_grad=True)
        x = autodiff.prepare_input(np.ones((4, 2)))
        gradient = autodiff.gradient(lambda z: weight.expand(z.shape[0]) * 2., x)
        self.assertEqual(tuple(gradient.shape), (4, 2))
        _assert_close(gradient, np.zeros((4, 2)), tolerance=0.)

    def test_batch_jacobian_of_input_independent_function(self):
        """Test that a graph-attached output not depending on the input has zero Jacobian."""
        weight = torch.ones(3, dtype=tu.get_precision(), requires_grad=True)
        x = autodiff.prepare_input(np.ones((4, 2)))
        jacobian = autodiff.batch_jacobian(lambda z: weight.expand(z.shape[0], 3) * 2., x)
        self.assertEqual(tuple(jacobian.shape), (4, 3, 2))
        _assert_close(jacobian, np.zeros((4, 3, 2)), tolerance=0.)

    def test_batch_jacobian_of_empty_output(self):
        """Test the Jacobian of a function with no output components."""
        x = autodiff.prepare_input(np.ones((4, 2)))
        jacobian = autodiff.batch_jacobian(lambda z: z[:, :0] ** 2, x)
        self.assertEqual(tuple(jacobian.shape), (4, 0, 2))
        self.assertEqual(jacobian.dtype, tu.get_precision())

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
