"""Tests for the trainable bijectors: networks, splines, ScaleRotoShift and AutoregressiveFlow."""

#########################################################################################################
# Imports

import contextlib
import io
import math
import unittest

import numpy as np
import torch

from tensiometer.synthetic_probability import autodiff
from tensiometer.synthetic_probability import bijectors as bj
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


def _constant_half(weight):
    """Custom initializer setting every entry to 0.5."""
    torch.nn.init.constant_(weight, 0.5)


def _randomize_parameters(module, scale=0.1):
    """Overwrite all the parameters of a module with small Gaussian values.

    :param module: ``torch.nn.Module``.
    :param scale: standard deviation of the values.
    """
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(scale * torch.randn_like(parameter))


def _mafs(flow):
    """Masked autoregressive flows of an :class:`AutoregressiveFlow` in chain order.

    :param flow: :class:`AutoregressiveFlow`.
    :returns: list of :class:`MaskedAutoregressiveFlow`.
    """
    return [_b for _b in flow.bijector.bijectors if isinstance(_b, tb.MaskedAutoregressiveFlow)]


def _count(flow, bijector_type):
    """Number of bijectors of a given type in the chain of a flow.

    :param flow: :class:`AutoregressiveFlow`.
    :param bijector_type: bijector class.
    :returns: integer count.
    """
    return sum(1 for _b in flow.bijector.bijectors if isinstance(_b, bijector_type))


def _small_flow(num_params=2, **kwargs):
    """Small :class:`AutoregressiveFlow` for structure tests.

    :param num_params: number of parameters.
    :param kwargs: options overriding the defaults (one affine layer, no permutations).
    :returns: :class:`AutoregressiveFlow`.
    """
    options = {'transformation_type': 'affine', 'n_transformations': 1, 'hidden_units': [4], 'permutations': False}
    options.update(kwargs)
    return tb.AutoregressiveFlow(num_params, **options)

#########################################################################################################
# Trainable transformation base class


class TestTrainableTransformation(unittest.TestCase):
    """TrainableTransformation interface."""

    def test_base_class_raises(self):
        """Test that the base class without bijector raises NotImplementedError."""
        base = tb.TrainableTransformation()
        with self.assertRaises(NotImplementedError):
            base.parameters()
        with self.assertRaises(NotImplementedError):
            base.reset_parameters()

    def test_subclass_with_bijector(self):
        """Test the parameters passthrough of a subclass."""
        class _Single(tb.TrainableTransformation):
            """Transformation holding one ScaleRotoShift."""

            def __init__(self):
                """Build the bijector."""
                self.bijector = tb.ScaleRotoShift(3)

        transformation = _Single()
        self.assertEqual(len(list(transformation.parameters())), 3)
        before = [_p.detach().clone() for _p in transformation.parameters()]
        torch.manual_seed(0)
        transformation.reset_parameters()
        after = list(transformation.parameters())
        self.assertFalse(all(torch.equal(_b, _a) for _b, _a in zip(before, after)))

#########################################################################################################
# ScaleRotoShift


class TestScaleRotoShift(unittest.TestCase):
    """Trainable affine bijector."""

    def setUp(self):
        """Seed and inputs."""
        torch.manual_seed(0)
        self.x = tu.to_tensor(np.random.default_rng(0).standard_normal((6, 3)))

    def test_identity_with_zeros_initializer(self):
        """Test that the zeros initializer gives the identity map."""
        bijector = tb.ScaleRotoShift(3, initializer='zeros')
        self.assertEqual(bijector.name, 'Affine')
        self.assertEqual(bijector.min_event_ndims, 1)
        y = bijector.forward(self.x)
        self.assertEqual(y.dtype, tu.get_precision())
        _assert_close(y, self.x)
        _assert_close(bijector.inverse(y), self.x)
        fldj = bijector.forward_log_det_jacobian(self.x, event_ndims=1)
        self.assertEqual(tuple(fldj.shape), (6,))
        _assert_close(fldj, np.zeros(6))
        _assert_close(bijector.inverse_log_det_jacobian(y, event_ndims=1), np.zeros(6))

    def test_parameters(self):
        """Test the parameter shapes and dtypes."""
        bijector = tb.ScaleRotoShift(4)
        shapes = {name: tuple(parameter.shape) for name, parameter in bijector.named_parameters()}
        self.assertEqual(shapes, {'shift': (4,), 'log_scale': (4,), 'rotation': (6,)})
        for parameter in bijector.parameters():
            self.assertEqual(parameter.dtype, tu.get_precision())

    def test_random_parameters_invertible(self):
        """Test round trip and log determinants with random (glorot) parameters."""
        bijector = tb.ScaleRotoShift(3)
        self.assertFalse(torch.all(bijector.log_scale == 0.))
        self.assertFalse(torch.all(bijector.rotation == 0.))
        y = bijector.forward(self.x)
        _assert_close(bijector.inverse(y), self.x)
        fldj = bijector.forward_log_det_jacobian(self.x)
        _assert_close(fldj, float(bijector.log_scale.detach().sum()) * np.ones(6))
        _assert_close(bijector.inverse_log_det_jacobian(y), -fldj)
        jacobian = autodiff.batch_jacobian(bijector.forward, autodiff.prepare_input(self.x))
        _assert_close(torch.linalg.slogdet(jacobian)[1], fldj)

    def test_matrix_is_symmetric_with_scale_eigenvalues(self):
        """Test that the affine matrix is symmetric with eigenvalues exp(log_scale)."""
        bijector = tb.ScaleRotoShift(3)
        jacobian = autodiff.batch_jacobian(bijector.forward, autodiff.prepare_input(self.x))[0]
        matrix = tu.to_numpy(jacobian).astype(np.float64)
        _assert_close(matrix, matrix.T)
        _assert_close(np.sort(np.linalg.eigvalsh(matrix)), np.sort(np.exp(tu.to_numpy(bijector.log_scale))))

    def test_disabled_components(self):
        """Test that disabled components are zero buffers."""
        bijector = tb.ScaleRotoShift(3, scale=False, roto=False, shift=False)
        self.assertEqual(len(list(bijector.parameters())), 0)
        _assert_close(bijector.forward(self.x), self.x)
        _assert_close(bijector.forward_log_det_jacobian(self.x), np.zeros(6))
        only_shift = tb.ScaleRotoShift(3, scale=False, roto=False)
        self.assertEqual([name for name, _ in only_shift.named_parameters()], ['shift'])
        _assert_close(only_shift.forward(self.x), self.x + only_shift.shift)

    def test_callable_initializer(self):
        """Test a custom callable initializer."""
        bijector = tb.ScaleRotoShift(2, initializer=_constant_half)
        for parameter in bijector.parameters():
            _assert_close(parameter, 0.5 * np.ones(parameter.shape), tolerance=0.)
        with self.assertRaises(ValueError):
            tb.ScaleRotoShift(2, initializer='not_an_initializer')

    def test_reset_parameters(self):
        """Test that reset_parameters draws new values."""
        bijector = tb.ScaleRotoShift(3)
        before = [_p.detach().clone() for _p in bijector.parameters()]
        bijector.reset_parameters()
        for old, new in zip(before, bijector.parameters()):
            self.assertFalse(torch.equal(old, new))

    def test_one_dimension(self):
        """Test the one dimensional case without rotation."""
        bijector = tb.ScaleRotoShift(1)
        self.assertEqual(bijector.rotation.numel(), 0)
        x = self.x[:, :1]
        _assert_close(bijector.forward(x), x * torch.exp(bijector.log_scale) + bijector.shift)

#########################################################################################################
# Circular rational quadratic spline


class TestCircularSpline(unittest.TestCase):
    """Rational quadratic spline with linear tails."""

    def setUp(self):
        """Two bins on [-1, 1] with boundary slope 1.5."""
        self.boundary = 1.5
        self.spline = tb.RationalQuadraticSpline(
            bin_widths=[0.8, 1.2], bin_heights=[1.0, 1.0], knot_slopes=[1.0],
            range_min=-1., boundary_knot_slope=self.boundary)

    def test_round_trip(self):
        """Test inverse(forward(x)) inside and outside the domain."""
        x = np.linspace(-3., 3., 25)
        y = self.spline.forward(x)
        _assert_close(self.spline.inverse(y), x, tolerance=max(_tolerance(), 1e-5))

    def test_tails(self):
        """Test the linear tails and the tail log determinant."""
        x = np.array([-2.5, -1.2, 1.5, 3.])
        expected = np.where(x > 1., self.boundary * (x - 1.) + 1., self.boundary * (x + 1.) - 1.)
        _assert_close(self.spline.forward(x), expected)
        _assert_close(self.spline.forward_log_det_jacobian(x, event_ndims=0), np.full(4, math.log(self.boundary)))
        _assert_close(self.spline.inverse(np.array([2.])), np.array([1. + 1. / self.boundary]))
        _assert_close(self.spline.inverse_log_det_jacobian(np.array([2.]), event_ndims=0),
                      np.array([-math.log(self.boundary)]))

    def test_boundary_values(self):
        """Test that the domain bounds are fixed points."""
        _assert_close(self.spline.forward(np.array([-1., 1.])), np.array([-1., 1.]))
        _assert_close(self.spline.forward(np.array([0.])), self.spline.forward(np.array([0.])), tolerance=0.)

#########################################################################################################
# Spline transformer options


class TestSplineTransformer(unittest.TestCase):
    """Spline parametrization options and errors."""

    def _random_params(self, transformer, scale=1.):
        """Random unconstrained parameters of shape ``(5, 2, num_params)``."""
        torch.manual_seed(1)
        return scale * torch.randn(5, 2, transformer.num_params, dtype=tu.get_precision())

    def test_number_of_parameters(self):
        """Test the number of parameters for every option."""
        knots = 3
        self.assertEqual(tb.SplineTransformer(spline_knots=knots).num_params, 3 * knots - 1)
        self.assertEqual(tb.SplineTransformer(spline_knots=knots, circular=True).num_params, 3 * knots)
        self.assertEqual(tb.SplineTransformer(spline_knots=knots, equispaced_x_knots=True).num_params, 2 * knots - 1)
        self.assertEqual(tb.SplineTransformer(spline_knots=knots, equispaced_y_knots=True).num_params, 2 * knots - 1)
        self.assertEqual(tb.SplineTransformer(spline_knots=knots, equispaced_x_knots=True, circular=True).num_params,
                         2 * knots)
        self.assertEqual(tb.AffineTransformer.num_params, 2)

    def test_range(self):
        """Test the default and explicit domain."""
        transformer = tb.SplineTransformer(range_max=2.)
        self.assertEqual(transformer.range_min, -2.)
        self.assertEqual(transformer.interval_width, 4.)
        transformer = tb.SplineTransformer(range_min=-0.5, range_max=1.5)
        self.assertEqual(transformer.range_min, -0.5)
        widths, heights, _, _ = transformer.spline_parameters(self._random_params(transformer))
        _assert_close(widths.sum(-1), 2. * np.ones((5, 2)))
        _assert_close(heights.sum(-1), 2. * np.ones((5, 2)))

    def test_equispaced_knots(self):
        """Test fixed bin widths or heights."""
        for option in ('equispaced_x_knots', 'equispaced_y_knots'):
            with self.subTest(option=option):
                transformer = tb.SplineTransformer(spline_knots=4, range_max=2., **{option: True})
                widths, heights, slopes, boundary = transformer.spline_parameters(self._random_params(transformer))
                fixed = widths if option == 'equispaced_x_knots' else heights
                free = heights if option == 'equispaced_x_knots' else widths
                _assert_close(fixed, np.ones((5, 2, 4)))
                self.assertFalse(torch.allclose(free, torch.ones_like(free)))
                self.assertEqual(tuple(slopes.shape), (5, 2, 3))
                self.assertIsNone(boundary)

    def test_slope_std(self):
        """Test slopes built from the average bin slope."""
        transformer = tb.SplineTransformer(spline_knots=4, range_max=2., equispaced_x_knots=True, slope_std=0.25)
        zeros = torch.zeros(5, 2, transformer.num_params, dtype=tu.get_precision())
        _, _, slopes, _ = transformer.spline_parameters(zeros)
        expected = math.log1p(math.exp(10.)) / 10.
        _assert_close(slopes, expected * np.ones((5, 2, 3)), tolerance=max(_tolerance(), 1e-6))
        params = self._random_params(transformer, scale=3.)
        widths, heights, slopes, _ = transformer.spline_parameters(params)
        average = (heights[..., 1:] + heights[..., :-1]) / (widths[..., 1:] + widths[..., :-1])
        self.assertTrue(torch.all(slopes > 0.))
        self.assertTrue(torch.all(slopes >= average - 0.25 - 1e-6))
        self.assertTrue(torch.all(slopes <= average + 0.25 + 0.1))

    def test_min_bin_width_and_height(self):
        """Test the minimum bin sizes for extreme parameters."""
        transformer = tb.SplineTransformer(spline_knots=4, range_max=2., min_bin_width=0.05, min_bin_height=0.1)
        params = self._random_params(transformer, scale=50.)
        widths, heights, _, _ = transformer.spline_parameters(params)
        self.assertTrue(torch.all(widths >= 0.05 * 4. * (1. - 1e-5)))
        self.assertTrue(torch.all(heights >= 0.1 * 4. * (1. - 1e-5)))
        _assert_close(widths.sum(-1), 4. * np.ones((5, 2)), tolerance=max(_tolerance(), 1e-5))
        unconstrained = tb.SplineTransformer(spline_knots=4, range_max=2.)
        widths, _, _, _ = unconstrained.spline_parameters(params)
        self.assertLess(float(widths.min()), 0.05 * 4.)

    def test_errors(self):
        """Test the invalid option combinations."""
        with self.assertRaises(ValueError):
            tb.SplineTransformer(equispaced_x_knots=True, equispaced_y_knots=True)
        with self.assertRaises(ValueError):
            tb.SplineTransformer(range_max=-1.)
        with self.assertRaises(ValueError):
            tb.SplineTransformer(spline_knots=4, min_bin_width=0.25)
        with self.assertRaises(ValueError):
            tb.SplineTransformer(spline_knots=4, min_bin_height=0.3)
        with self.assertRaises(NotImplementedError):
            tb.SplineTransformer(circular=True, slope_std=0.5)

    def test_transformer_round_trip_and_log_det(self):
        """Test the transformer round trip and elementwise log derivative."""
        for circular in (False, True):
            with self.subTest(circular=circular):
                transformer = tb.SplineTransformer(spline_knots=5, range_max=2., circular=circular)
                params = self._random_params(transformer)
                x = tu.to_tensor(np.random.default_rng(2).uniform(-3., 3., (5, 2)))
                y = transformer.forward(x, params)
                _assert_close(transformer.inverse(y, params), x)
                derivative = autodiff.gradient(lambda _x: transformer.forward(_x, params), autodiff.prepare_input(x))
                _assert_close(transformer.inverse_log_det_jacobian(y, params), -torch.log(derivative))

    def test_affine_transformer(self):
        """Test the affine transformer convention (shift, log_scale)."""
        transformer = tb.AffineTransformer()
        x = tu.to_tensor(np.array([[1., -2.]]))
        params = tu.to_tensor(np.array([[[0.5, math.log(2.)], [-1., 0.]]]))
        y = transformer.forward(x, params)
        _assert_close(y, np.array([[2.5, -3.]]))
        _assert_close(transformer.inverse(y, params), x)
        _assert_close(transformer.inverse_log_det_jacobian(y, params), np.array([[-math.log(2.), 0.]]))

#########################################################################################################
# Autoregressive networks


class TestNetworks(unittest.TestCase):
    """MADE and flex autoregressive networks."""

    def test_flex_network_shapes_and_identity_dims(self):
        """Test the flex network output shape, identity dimensions and bias-only first dimension."""
        torch.manual_seed(0)
        network = tb.FlexAutoregressiveNetwork(3, 2, hidden_units=[4, 2], identity_dims=[1])
        _randomize_parameters(network, scale=0.5)
        x = tu.to_tensor(np.random.default_rng(0).standard_normal((5, 3)))
        output = network(x)
        self.assertEqual(tuple(output.shape), (5, 3, 2))
        self.assertEqual(output.dtype, tu.get_precision())
        _assert_close(output[:, 1, :], np.zeros((5, 2)), tolerance=0.)
        _assert_close(output[:, 0, :], output[:1, 0, :].expand(5, 2), tolerance=0.)
        self.assertIsInstance(network.networks[0], tb.BiasOnly)
        self.assertEqual(len(network.networks), 2)

    def test_flex_network_hidden_sizes(self):
        """Test the scaling of the hidden sizes with the dimension."""
        scaled = tb.FlexAutoregressiveNetwork(3, 2, hidden_units=[6, 3])
        fixed = tb.FlexAutoregressiveNetwork(3, 2, hidden_units=[6, 3], scale_with_dim=False)
        scaled_sizes = [[_l.out_features for _l in scaled.networks[i].layers] for i in (1, 2)]
        fixed_sizes = [[_l.out_features for _l in fixed.networks[i].layers] for i in (1, 2)]
        self.assertEqual(scaled_sizes, [[4, 2, 2], [6, 3, 2]])
        self.assertEqual(fixed_sizes, [[6, 3, 2], [6, 3, 2]])
        self.assertEqual([_l.in_features for _l in fixed.networks[2].layers], [2, 6, 3])

    def test_flex_network_is_autoregressive(self):
        """Test that flex output d depends only on inputs < d."""
        torch.manual_seed(1)
        network = tb.FlexAutoregressiveNetwork(4, 3, hidden_units=[5])
        _randomize_parameters(network, scale=0.5)
        x = autodiff.prepare_input(np.random.default_rng(1).standard_normal((4, 4)))
        jacobian = tu.to_numpy(autodiff.batch_jacobian(network, x))
        for d in range(4):
            np.testing.assert_array_equal(jacobian[:, d, :, d:], 0.)
            if d > 0:
                self.assertTrue(np.any(jacobian[:, d, :, :d] != 0.))

    def test_made_shapes_and_initializers(self):
        """Test MADE shapes, activations and initializer options."""
        x = tu.to_tensor(np.random.default_rng(2).standard_normal((3, 4)))
        network = tb.MaskedAutoregressiveNetwork(4, 5, [8, 8], activation='tanh')
        self.assertEqual(tuple(network(x).shape), (3, 4, 5))
        self.assertEqual(len(network.layers), 3)
        zeros = tb.MaskedAutoregressiveNetwork(4, 2, [8], kernel_initializer='zeros')
        _assert_close(zeros(x), np.zeros((3, 4, 2)), tolerance=0.)
        with self.assertRaises(ValueError):
            tb.MaskedAutoregressiveNetwork(4, 2, [8], kernel_initializer='bad')
        with self.assertRaises(ValueError):
            tb.MaskedAutoregressiveNetwork(4, 2, [8], activation='bad')

    def test_initializer_and_activation_specifications(self):
        """Test the conversion of initializer and activation specifications."""
        self.assertIsInstance(tb.get_initializer(None), tb.GlorotUniform)
        self.assertIsInstance(tb.get_initializer('zeros'), tb.Zeros)
        self.assertIs(tb.get_initializer(_constant_half), _constant_half)
        with self.assertRaises(ValueError):
            tb.get_initializer(3.)
        self.assertIsNone(tb.get_activation(None))
        self.assertIsNone(tb.get_activation('linear'))
        self.assertIs(tb.get_activation('tanh'), torch.tanh)
        self.assertIs(tb.get_activation(torch.relu), torch.relu)
        with self.assertRaises(ValueError):
            tb.get_activation(3)

    def test_masked_linear_mask_and_repr(self):
        """Test the mask shape validation and the representation of MaskedLinear."""
        with self.assertRaises(ValueError):
            tb.MaskedLinear(3, 2, mask=np.ones((3, 2)))
        masked = tb.MaskedLinear(3, 2, mask=np.tril(np.ones((2, 3))))
        plain = tb.MaskedLinear(3, 2)
        self.assertEqual(masked.extra_repr(), 'in_features=3, out_features=2, masked=True')
        self.assertIn('masked=False', repr(plain))

    def test_networks_without_hidden_layers(self):
        """Test that networks built without hidden units are single linear layers."""
        x = tu.to_tensor(np.random.default_rng(3).standard_normal((5, 3)))
        made = tb.MaskedAutoregressiveNetwork(3, 2)
        self.assertEqual(len(made.layers), 1)
        self.assertEqual(tuple(made(x).shape), (5, 3, 2))
        feed_forward = tb.FeedForward(3, 4, kernel_initializer='zeros')
        self.assertEqual(len(feed_forward.layers), 1)
        # without hidden layers the kernel initializer is ignored and the default one is used:
        self.assertIsInstance(feed_forward.layers[0].kernel_initializer, tb.GlorotUniform)
        self.assertTrue(bool((feed_forward.layers[0].weight != 0.).any()))
        _assert_close(feed_forward(x), x @ feed_forward.layers[0].weight.T)
        flex = tb.FlexAutoregressiveNetwork(3, 2)
        self.assertEqual([len(_n.layers) for _n in flex.networks[1:]], [1, 1])
        self.assertEqual(tuple(flex(x).shape), (5, 3, 2))

    def test_min_var_permutations(self):
        """Test min_var_permutations returns non identity permutations."""
        np.random.seed(0)
        permutations = tb.min_var_permutations(d=4, n=3, min_number=50)
        self.assertEqual(len(permutations), 3)
        for permutation in permutations:
            np.testing.assert_array_equal(np.sort(permutation), np.arange(4))
            self.assertFalse(np.all(permutation == np.arange(4)))

#########################################################################################################
# AutoregressiveFlow construction


class TestAutoregressiveFlow(unittest.TestCase):
    """AutoregressiveFlow constructor semantics."""

    def setUp(self):
        """Seed the random generators."""
        torch.manual_seed(0)
        np.random.seed(0)

    def test_defaults(self):
        """Test the default number of layers and hidden units."""
        flow = tb.AutoregressiveFlow(4, permutations=False)
        self.assertEqual(len(_mafs(flow)), 6)
        self.assertEqual([_l.out_features for _l in _mafs(flow)[0].conditioner.layers], [12, 12, 8])
        self.assertIsNone(flow.range_max)
        self.assertIsInstance(flow.bijector, bj.Chain)
        self.assertEqual(flow.device, tu.check_device(None))
        for parameter in flow.parameters():
            self.assertEqual(parameter.device.type, flow.device.type)
            self.assertEqual(parameter.dtype, tu.get_precision())
        initializer = _mafs(flow)[0].conditioner.layers[0].kernel_initializer
        self.assertIsInstance(initializer, tb.VarianceScaling)
        self.assertAlmostEqual(initializer.scale, 1. / 6.)

    def test_round_trip(self):
        """Test the round trip of a default spline flow."""
        flow = tb.AutoregressiveFlow(3, transformation_type='spline', n_transformations=2, hidden_units=[8])
        x = tu.to_tensor(np.random.default_rng(0).uniform(-3., 3., (10, 3)))
        y = flow.bijector.forward(x)
        _assert_close(flow.bijector.inverse(y), x)
        _assert_close(flow.bijector.forward_log_det_jacobian(x), -flow.bijector.inverse_log_det_jacobian(y))

    def test_range_max_adjustment(self):
        """Test the range_max adjustment for splines."""
        flow = _small_flow(transformation_type='spline', parameters_min=np.array([-5., -6.]),
                           parameters_max=np.array([9., 6.]))
        self.assertIsInstance(flow.range_max, float)
        self.assertEqual(flow.range_max, 10.)
        self.assertEqual(_mafs(flow)[0].transformer.range_max, 10.)
        self.assertEqual(_mafs(flow)[0].transformer.range_min, -10.)
        flow = _small_flow(1, transformation_type='spline', parameters_min=np.array([-20.]),
                           parameters_max=np.array([3.]), range_max=1.)
        self.assertEqual(flow.range_max, 21.)
        flow = _small_flow(1, transformation_type='spline', parameters_min=np.array([-1.]),
                           parameters_max=np.array([1.]), range_max=5.)
        self.assertEqual(flow.range_max, 5.)
        flow = _small_flow(1, parameters_min=np.array([-20.]), parameters_max=np.array([20.]))
        self.assertIsNone(flow.range_max)

    def test_range_max_adjustment_feedback(self):
        """Test the feedback printed by the range adjustment and the construction."""
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _small_flow(1, transformation_type='spline', parameters_min=np.array([-50.]),
                        parameters_max=np.array([60.]), range_max=1., feedback=2)
        text = output.getvalue()
        self.assertIn('range_max', text)
        self.assertIn('new range_max: 61.0', text)
        self.assertIn('Building Autoregressive Flow', text)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _small_flow(1, transformation_type='spline', parameters_min=np.array([-50.]),
                        parameters_max=np.array([60.]), range_max=1., feedback=0)
        self.assertEqual(output.getvalue(), '')

    def test_map_to_unitcube(self):
        """Test the unit cube option and its spline requirement."""
        with self.assertRaises(ValueError):
            _small_flow(1, map_to_unitcube=True)
        flow = _small_flow(2, transformation_type='spline', map_to_unitcube=True)
        layers = list(flow.bijector.bijectors)
        self.assertEqual(len(layers), 3)
        self.assertIsInstance(layers[0], bj.Invert)
        self.assertIsInstance(layers[0].bijector, bj.NormalCDF)
        self.assertIsInstance(layers[2], bj.NormalCDF)
        transformer = layers[1].transformer
        self.assertEqual((transformer.range_min, transformer.range_max), (0., 1.))
        x = tu.to_tensor(np.random.default_rng(1).uniform(-2., 2., (6, 2)))
        _assert_close(flow.bijector.inverse(flow.bijector.forward(x)), x, tolerance=max(_tolerance(), 1e-4))

    def test_periodic_preprocessing(self):
        """Test the periodic preprocessing layer and circular splines."""
        flow = _small_flow(2, transformation_type='spline', n_transformations=2, permutations=True,
                           periodic_params=np.array([True, False]), range_max=2.)
        layers = list(flow.bijector.bijectors)
        self.assertIsInstance(layers[0], bj.Blockwise)
        self.assertEqual(layers[0].name, 'PeriodicPreprocessing')
        self.assertIsInstance(layers[0].bijectors[0], bj.Scale)
        _assert_close(layers[0].bijectors[0].scale, 0.5, tolerance=0.)
        self.assertIsInstance(layers[0].bijectors[1], bj.Identity)
        self.assertEqual(_count(flow, bj.Permute), 1)
        self.assertNotIsInstance(layers[1], bj.Permute)
        for maf in _mafs(flow):
            self.assertTrue(maf.transformer.circular)
        x = tu.to_tensor(np.random.default_rng(2).uniform(-1.5, 1.5, (6, 2)))
        y = flow.bijector.forward(x)
        self.assertEqual(tuple(y.shape), (6, 2))
        _assert_close(flow.bijector.inverse(y), x)

    def test_periodic_options(self):
        """Test all-false periodic parameters and the spline requirement."""
        flow = _small_flow(2, transformation_type='spline', periodic_params=[False, False])
        self.assertEqual(_count(flow, bj.Blockwise), 0)
        self.assertFalse(_mafs(flow)[0].transformer.circular)
        with self.assertRaises(ValueError):
            _small_flow(2, periodic_params=[True, False])

    def test_mixed_transformation_types(self):
        """Test per-layer transformation and autoregressive types."""
        flow = _small_flow(2, transformation_type=['affine', 'spline'], autoregressive_type=['masked', 'flex'],
                           n_transformations=2, equispaced_x_knots=True, scale_roto_shift=True)
        mafs = _mafs(flow)
        self.assertIsInstance(mafs[0].transformer, tb.AffineTransformer)
        self.assertIsInstance(mafs[0].conditioner, tb.MaskedAutoregressiveNetwork)
        self.assertIsInstance(mafs[1].transformer, tb.SplineTransformer)
        self.assertTrue(mafs[1].transformer.equispaced_x_knots)
        self.assertIsInstance(mafs[1].conditioner, tb.FlexAutoregressiveNetwork)
        self.assertEqual([_m.name for _m in mafs], ['affine_maf_0', 'spline_maf_1'])
        affine_names = [_b.name for _b in flow.bijector.bijectors if isinstance(_b, tb.ScaleRotoShift)]
        self.assertEqual(affine_names, ['affine_0', 'affine_1'])
        self.assertIsNotNone(flow.range_max)
        x = tu.to_tensor(np.array([[0., 0.1], [1., -1.]]))
        _assert_close(flow.bijector.inverse(flow.bijector.forward(x)), x)
        with self.assertRaises(ValueError):
            _small_flow(2, transformation_type=['affine'], n_transformations=2)
        with self.assertRaises(ValueError):
            _small_flow(2, autoregressive_type=['masked', 'flex', 'flex'], n_transformations=2)

    def test_permutations_variants(self):
        """Test the permutations argument."""
        flow = _small_flow(3, n_transformations=2, permutations=True)
        self.assertIsInstance(flow.permutations, list)
        self.assertEqual(len(flow.permutations), 2)
        self.assertEqual(_count(flow, bj.Permute), 2)
        for value in (False, None):
            flow = _small_flow(3, n_transformations=2, permutations=value)
            self.assertFalse(flow.permutations)
            self.assertEqual(_count(flow, bj.Permute), 0)
        permutations = [np.array([2, 0, 1]), np.array([1, 2, 0])]
        flow = _small_flow(3, n_transformations=2, permutations=permutations)
        layers = [_b for _b in flow.bijector.bijectors if isinstance(_b, bj.Permute)]
        for layer, permutation in zip(layers, permutations):
            np.testing.assert_array_equal(tu.to_numpy(layer.permutation), permutation)
        for invalid in (np.bool_(True), 'abc', 3, [np.array([0, 1, 2])]):
            with self.subTest(permutations=invalid):
                with self.assertRaises(ValueError):
                    _small_flow(3, n_transformations=2, permutations=invalid)

    def test_invalid_types(self):
        """Test invalid transformation and autoregressive types."""
        with self.assertRaises(ValueError):
            _small_flow(1, transformation_type='invalid')
        with self.assertRaises(ValueError):
            _small_flow(1, autoregressive_type='bad')

    def test_kernel_initializer(self):
        """Test custom, string and invalid kernel initializers."""
        flow = _small_flow(2, kernel_initializer=_constant_half)
        for layer in _mafs(flow)[0].conditioner.layers:
            _assert_close(layer.weight, 0.5 * tu.to_numpy(layer.mask), tolerance=0.)
        flow = _small_flow(2, kernel_initializer='zeros')
        x = tu.to_tensor(np.array([[0.3, -0.7]]))
        _assert_close(flow.bijector.forward(x), x, tolerance=0.)
        with self.assertRaises(ValueError):
            _small_flow(2, kernel_initializer='bad')

    def test_unsupported_keras_kwargs(self):
        """Test that Keras only options raise and unknown options are ignored."""
        for key in ('kernel_regularizer', 'bias_constraint', 'use_bias', 'input_order', 'validate_args', 'dtype'):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    _small_flow(2, **{key: None})
        flow = _small_flow(2, some_unrelated_option=1)
        self.assertEqual(len(_mafs(flow)), 1)

    def test_forwarded_kwargs(self):
        """Test that spline and ScaleRotoShift options are forwarded."""
        flow = _small_flow(2, transformation_type='spline', slope_min=1e-2, min_bin_width=0.01,
                           scale_roto_shift=True, initializer='zeros', roto=False)
        transformer = _mafs(flow)[0].transformer
        self.assertEqual(transformer.slope_min, 1e-2)
        self.assertEqual(transformer.min_bin_width, 0.01)
        affine = flow.bijector.bijectors[-1]
        self.assertIsInstance(affine, tb.ScaleRotoShift)
        self.assertEqual([name for name, _ in affine.named_parameters()], ['shift', 'log_scale'])
        _assert_close(affine.log_scale, np.zeros(2), tolerance=0.)

    def test_parameters_and_reset(self):
        """Test the parameters passthrough and reset_parameters."""
        flow = _small_flow(2, transformation_type='spline', autoregressive_type='flex', scale_roto_shift=True)
        parameters = list(flow.parameters())
        self.assertEqual(len(parameters), len(list(flow.bijector.parameters())))
        with torch.no_grad():
            for parameter in parameters:
                parameter.fill_(0.3)
        flow.reset_parameters()
        changed = [not torch.all(_p == 0.3) for _p in flow.parameters()]
        self.assertTrue(all(changed))

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
