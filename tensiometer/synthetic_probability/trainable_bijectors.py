"""
This file contains the definition of the trainable bijectors that are needed to define normalizing flows.

The main building blocks are:

- masked autoregressive networks (MADE) and per-dimension ("flex") autoregressive networks
  that produce the parameters of the transformation of each dimension;
- affine and rational quadratic spline transformers that use those parameters;
- :class:`MaskedAutoregressiveFlow`, the autoregressive bijector combining the two;
- :class:`ScaleRotoShift`, a trainable affine map;
- :class:`AutoregressiveFlow`, which chains several of these layers.

References: Papamakarios et al. 2017 (arXiv:1705.07057), Germain et al. 2015 (MADE,
arXiv:1502.03509), Durkan et al. 2019 (Neural Spline Flows, arXiv:1906.04032).
"""

###############################################################################
# initial imports and set-up:

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ..utilities import stats_utilities as stutils
from . import bijectors as bj
from . import tensor_utilities as tu

###############################################################################
# utility function to generate random permutations with minimum stack variance:


def min_var_permutations(d, n, min_number=10000):
    """
    Find a random permutation sequence that has the minimum (sample) variance between components.
    This is useful for MAFs, in which we want to concatenate several triangular transformations.

    :param d: dimension of the problem
    :param n: number of dimensions
    :param min_number: number of random trials
    :return: the permutation that has less variance
    """
    permutation = None
    identity = np.arange(d)
    perm_var = np.inf
    for _ in range(max(min_number, 2 * d * n)):
        # draw the permutation ensemble:
        _n_perm = 0
        _temp_perm = []
        while _n_perm < n:
            _temp = np.random.permutation(d)
            if not np.all(_temp == identity):
                _temp_perm.append(_temp)
                _n_perm += 1
        # calculate variance:
        _temp_var = np.var(np.sum(_temp_perm, axis=0))
        # save minimum:
        if _temp_var < perm_var:
            perm_var = _temp_var
            permutation = _temp_perm
    #
    return permutation


###############################################################################
# generic class:


class TrainableTransformation(object):
    """
    Base interface for trainable transformations used in normalizing flows.

    Subclasses store the trainable bijector in ``self.bijector``. Persistence is handled by
    the flow snapshot (:meth:`~tensiometer.synthetic_probability.synthetic_probability.FlowCallback.save`).
    """

    bijector = None

    def parameters(self):
        """
        Trainable parameters of the transformation.

        :returns: iterator over the parameters of ``self.bijector``.
        :raises NotImplementedError: if the transformation has no bijector.
        """
        if self.bijector is None:
            raise NotImplementedError('TrainableTransformation subclasses must define self.bijector')
        return self.bijector.parameters()

    def reset_parameters(self):
        """
        Re-initialize all the trainable parameters.

        :raises NotImplementedError: if the transformation has no bijector.
        """
        if self.bijector is None:
            raise NotImplementedError('TrainableTransformation subclasses must define self.bijector')
        reset_module_parameters(self.bijector)


def reset_module_parameters(module):
    """
    Call ``reset_parameters()`` on every submodule that defines it.

    :param module: a ``torch.nn.Module``.
    """
    for submodule in module.modules():
        reset = getattr(submodule, 'reset_parameters', None)
        if callable(reset):
            reset()

###############################################################################
# weight initializers:


class VarianceScaling(object):
    """
    Keras ``VarianceScaling(scale, mode='fan_avg', distribution='truncated_normal')`` initializer.

    :param scale: scaling factor of the variance.
    """

    def __init__(self, scale=1.0):
        self.scale = float(scale)

    def __call__(self, weight):
        """
        Initialize a ``(out, in)`` weight in place.

        :param weight: weight tensor.
        """
        fan_out, fan_in = weight.shape[0], weight.shape[1]
        fan_avg = max(1., (fan_in + fan_out) / 2.)
        std = math.sqrt(self.scale / fan_avg) / 0.87962566103423978
        with torch.no_grad():
            torch.nn.init.trunc_normal_(weight, mean=0., std=std, a=-2. * std, b=2. * std)


class GlorotUniform(object):
    """Glorot (Xavier) uniform initializer, the Keras default for dense layers."""

    def __call__(self, weight):
        """
        Initialize a weight in place.

        :param weight: weight tensor.
        """
        with torch.no_grad():
            if weight.dim() == 1:
                if weight.numel() > 0:
                    limit = math.sqrt(3. / weight.numel())
                    torch.nn.init.uniform_(weight, -limit, limit)
            elif weight.numel() > 0:
                torch.nn.init.xavier_uniform_(weight)


class Zeros(object):
    """Initializer that sets all the entries to zero."""

    def __call__(self, weight):
        """
        Initialize a weight in place.

        :param weight: weight tensor.
        """
        with torch.no_grad():
            torch.nn.init.zeros_(weight)


def get_initializer(initializer, default=None):
    """
    Convert an initializer specification to a callable.

    :param initializer: None (use ``default``), ``'glorot_uniform'``, ``'zeros'`` or a
        callable acting in place on a weight tensor.
    :param default: initializer used when ``initializer`` is None.
    :returns: callable initializer.
    :raises ValueError: if the specification is not recognized.
    """
    if initializer is None:
        initializer = default
    if initializer is None:
        return GlorotUniform()
    if isinstance(initializer, str):
        if initializer == 'glorot_uniform':
            return GlorotUniform()
        if initializer == 'zeros':
            return Zeros()
        raise ValueError('Unknown initializer ' + repr(initializer) + '. Use glorot_uniform, zeros or a callable.')
    if callable(initializer):
        return initializer
    raise ValueError('Unknown initializer ' + repr(initializer) + '. Use glorot_uniform, zeros or a callable.')


_ACTIVATIONS = {
    'asinh': torch.asinh,
    'tanh': torch.tanh,
    'relu': torch.relu,
    'sigmoid': torch.sigmoid,
    'softplus': F.softplus,
    'elu': F.elu,
    'gelu': F.gelu,
    'silu': F.silu,
}


def get_activation(activation):
    """
    Convert an activation specification to a torch callable.

    :param activation: a callable, one of the names in ``_ACTIVATIONS``, or None / ``'linear'``
        for no activation.
    :returns: callable or None.
    :raises ValueError: if the name is not recognized.
    """
    if activation is None:
        return None
    if isinstance(activation, str):
        if activation == 'linear':
            return None
        if activation not in _ACTIVATIONS:
            raise ValueError('Unknown activation ' + repr(activation) + '. Use a torch callable or one of '
                             + ', '.join(sorted(_ACTIVATIONS.keys())))
        return _ACTIVATIONS[activation]
    if callable(activation):
        return activation
    raise ValueError('Unknown activation ' + repr(activation))

###############################################################################
# dense and masked layers:


class MaskedLinear(torch.nn.Module):
    """
    Dense layer ``y = x W^T + b`` with an optional binary mask on the weights.

    :param in_features: number of inputs.
    :param out_features: number of outputs.
    :param mask: optional ``(out_features, in_features)`` array of zeros and ones.
    :param kernel_initializer: callable initializing the weight in place (called under ``torch.no_grad()``).
    """

    def __init__(self, in_features, out_features, mask=None, kernel_initializer=None):
        super().__init__()
        tu.lock_precision()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.weight = torch.nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=tu.prec))
        self.bias = torch.nn.Parameter(torch.zeros(self.out_features, dtype=tu.prec))
        if mask is not None:
            mask = torch.as_tensor(np.asarray(mask), dtype=tu.prec)
            if tuple(mask.shape) != (self.out_features, self.in_features):
                raise ValueError('mask must have shape (out_features, in_features)')
            self.register_buffer('mask', mask)
        else:
            self.mask = None
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the weight with the kernel initializer and the bias with zeros."""
        with torch.no_grad():
            if self.in_features > 0:
                self.kernel_initializer(self.weight)
                if self.mask is not None:
                    self.weight.mul_(self.mask)
            self.bias.zero_()

    def forward(self, x):
        """
        Apply the layer.

        :param x: input of shape ``(..., in_features)``.
        :returns: output of shape ``(..., out_features)``.
        """
        weight = self.weight if self.mask is None else self.weight * self.mask
        return F.linear(x, weight, self.bias)

    def extra_repr(self):
        """Layer sizes."""
        return 'in_features={}, out_features={}, masked={}'.format(
            self.in_features, self.out_features, self.mask is not None)


def _made_degrees(num_params, hidden_units):
    """
    Degrees of the MADE units with left-to-right input order and equal hidden degrees
    (TensorFlow Probability ``AutoregressiveNetwork`` defaults).

    :param num_params: number of inputs ``D``.
    :param hidden_units: list of hidden layer sizes.
    :returns: list of integer arrays, one per layer (inputs first).
    """
    degrees = [np.arange(1, num_params + 1)]
    for units in hidden_units:
        min_degree = min(np.min(degrees[-1]), num_params - 1)
        degrees.append(np.maximum(
            min_degree,
            np.ceil(np.arange(1, units + 1) * (num_params - 1) / float(units + 1)).astype(np.int64)))
    return degrees


def _made_masks(degrees, params):
    """
    MADE masks in torch ``(out, in)`` layout.

    :param degrees: output of :func:`_made_degrees`.
    :param params: number of parameters per dimension.
    :returns: list of mask arrays.
    """
    masks = []
    for inp, out in zip(degrees[:-1], degrees[1:]):
        masks.append((inp[:, np.newaxis] <= out[np.newaxis, :]).T)
    output_mask = degrees[-1][:, np.newaxis] < degrees[0][np.newaxis, :]
    output_mask = np.repeat(output_mask, params, axis=1)
    masks.append(output_mask.T)
    return [_m.astype(np.float64) for _m in masks]

###############################################################################
# autoregressive networks:


class MaskedAutoregressiveNetwork(torch.nn.Module):
    """
    Masked autoencoder for distribution estimation (MADE).

    Output ``[..., d, :]`` depends only on inputs ``[..., :d]``. Follows TensorFlow
    Probability ``AutoregressiveNetwork`` with ``input_order='left-to-right'`` and
    ``hidden_degrees='equal'``.

    :param num_params: number of dimensions ``D``.
    :param params: number of parameters produced per dimension.
    :param hidden_units: list of hidden layer sizes.
    :param activation: activation of the hidden layers (default ``torch.asinh``).
    :param kernel_initializer: weight initializer, defaults to ``VarianceScaling(1.)``.
    """

    def __init__(self, num_params, params, hidden_units=None, activation=torch.asinh, kernel_initializer=None):
        super().__init__()
        if hidden_units is None:
            hidden_units = []
        self.num_params = int(num_params)
        self.params = int(params)
        self.hidden_units = [int(_h) for _h in hidden_units]
        self.activation = get_activation(activation)
        kernel_initializer = get_initializer(kernel_initializer, default=VarianceScaling(1.))
        masks = _made_masks(_made_degrees(self.num_params, self.hidden_units), self.params)
        sizes = [self.num_params] + self.hidden_units + [self.num_params * self.params]
        self.layers = torch.nn.ModuleList([
            MaskedLinear(sizes[i], sizes[i + 1], mask=masks[i], kernel_initializer=kernel_initializer)
            for i in range(len(sizes) - 1)])

    def forward(self, x):
        """
        Evaluate the network.

        :param x: input of shape ``(..., D)``.
        :returns: parameters of shape ``(..., D, params)``.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        x = self.layers[-1](x)
        return x.reshape(x.shape[:-1] + (self.num_params, self.params))


class BiasOnly(torch.nn.Module):
    """
    Constant (trainable) output, used for the first dimension of a flex network.

    :param params: number of outputs.
    """

    def __init__(self, params):
        super().__init__()
        tu.lock_precision()
        self.params = int(params)
        self.bias = torch.nn.Parameter(torch.zeros(self.params, dtype=tu.prec))

    def reset_parameters(self):
        """Reset the bias to zero."""
        with torch.no_grad():
            self.bias.zero_()

    def forward(self, x):
        """
        Broadcast the bias to the batch shape.

        :param x: input of shape ``(..., d)``.
        :returns: tensor of shape ``(..., params)``.
        """
        return self.bias.expand(x.shape[:-1] + (self.params,))


class FeedForward(torch.nn.Module):
    """
    Plain feed forward network.

    :param dim_in: number of inputs.
    :param dim_out: number of outputs.
    :param hidden_units: list of hidden layer sizes.
    :param activation: activation of the hidden layers.
    :param kernel_initializer: weight initializer of the hidden and output layers.
    """

    def __init__(self, dim_in, dim_out, hidden_units=None, activation=torch.asinh, kernel_initializer=None):
        super().__init__()
        if hidden_units is None:
            hidden_units = []
        self.activation = get_activation(activation)
        if len(hidden_units) == 0:
            kernel_initializer = None
        sizes = [int(dim_in)] + [int(_h) for _h in hidden_units] + [int(dim_out)]
        self.layers = torch.nn.ModuleList([
            MaskedLinear(sizes[i], sizes[i + 1], kernel_initializer=kernel_initializer)
            for i in range(len(sizes) - 1)])

    def forward(self, x):
        """
        Evaluate the network.

        :param x: input of shape ``(..., dim_in)``.
        :returns: output of shape ``(..., dim_out)``.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        return self.layers[-1](x)


class FlexAutoregressiveNetwork(torch.nn.Module):
    """
    Autoregressive network made of one feed forward network per dimension; the network of
    dimension ``d`` sees the inputs ``[..., :d]``.

    :param num_params: number of dimensions ``D``.
    :param params: number of parameters produced per dimension.
    :param hidden_units: list of hidden layer sizes.
    :param activation: activation of the hidden layers.
    :param kernel_initializer: weight initializer.
    :param scale_with_dim: scale the hidden sizes of dimension ``d`` by ``(d + 1) / D``.
    :param identity_dims: optional iterable of dimensions whose parameters are always zero.
    """

    def __init__(self, num_params, params, hidden_units=None, activation=torch.asinh,
                 kernel_initializer=None, scale_with_dim=True, identity_dims=None):
        super().__init__()
        if hidden_units is None:
            hidden_units = []
        self.num_params = int(num_params)
        self.params = int(params)
        if identity_dims is None:
            identity_dims = []
        self.identity_dims = [int(_d) for _d in identity_dims]
        networks = []
        for dim in range(self.num_params):
            if dim in self.identity_dims:
                networks.append(None)
            elif dim == 0:
                networks.append(BiasOnly(self.params))
            else:
                if scale_with_dim:
                    _hidden = [int(np.ceil(_h * (dim + 1) / self.num_params)) for _h in hidden_units]
                else:
                    _hidden = list(hidden_units)
                networks.append(FeedForward(dim, self.params, hidden_units=_hidden,
                                            activation=activation, kernel_initializer=kernel_initializer))
        self.networks = torch.nn.ModuleList([_n for _n in networks if _n is not None])
        self.network_index = [None if _n is None else sum(1 for _m in networks[:i] if _m is not None)
                              for i, _n in enumerate(networks)]

    def forward(self, x):
        """
        Evaluate the network.

        :param x: input of shape ``(..., D)``.
        :returns: parameters of shape ``(..., D, params)``.
        """
        outputs = []
        for dim in range(self.num_params):
            index = self.network_index[dim]
            if index is None:
                outputs.append(x.new_zeros(x.shape[:-1] + (self.params,)))
            else:
                outputs.append(self.networks[index](x[..., :dim]))
        return torch.stack(outputs, dim=-2)

###############################################################################
# rational quadratic splines:


def _knot_positions(bin_sizes, range_min):
    """Cumulative knot positions starting at ``range_min``, shape ``(..., K + 1)``."""
    start = torch.full(bin_sizes.shape[:-1] + (1,), float(range_min), dtype=bin_sizes.dtype, device=bin_sizes.device)
    return torch.cat([start, range_min + torch.cumsum(bin_sizes, dim=-1)], dim=-1)


def _gather_last(values, indices):
    """Gather along the last axis and drop it."""
    return torch.gather(values, -1, indices)[..., 0]


def _rqs_shared(inputs, bin_widths, bin_heights, knot_slopes, range_min, boundary_slope, inverse):
    """
    Quantities shared by the rational quadratic spline forward, inverse and log determinant.

    :param inputs: points of shape ``(...)``.
    :param bin_widths: bin widths of shape ``(..., K)``.
    :param bin_heights: bin heights of shape ``(..., K)``.
    :param knot_slopes: interior knot slopes of shape ``(..., K - 1)``.
    :param range_min: lower end of the spline domain.
    :param boundary_slope: None for identity tails (TFP), else slope of the linear tails ``(...)``.
    :param inverse: True if ``inputs`` are in the output space.
    :returns: dictionary of broadcast tensors.
    """
    kx = _knot_positions(bin_widths, range_min)
    ky = _knot_positions(bin_heights, range_min)
    if boundary_slope is None:
        pad = torch.ones_like(knot_slopes[..., :1])
        kd = torch.cat([pad, knot_slopes, pad], dim=-1)
    else:
        pad = boundary_slope.unsqueeze(-1)
        kd = torch.cat([pad, knot_slopes, pad], dim=-1)
    shape = torch.broadcast_shapes(inputs.shape + (1,), kx.shape, ky.shape, kd.shape)
    kx = torch.broadcast_to(kx, shape)
    ky = torch.broadcast_to(ky, shape)
    kd = torch.broadcast_to(kd, shape)
    inputs = torch.broadcast_to(inputs, shape[:-1])
    knots = ky if inverse else kx
    low = knots[..., 0]
    high = knots[..., -1]
    out_dn = inputs <= low
    out_up = inputs >= high
    out = out_dn | out_up
    safe = torch.where(out, low, inputs)
    num_bins = knots.shape[-1] - 1
    indices = torch.searchsorted(knots[..., :-1].contiguous(), safe.unsqueeze(-1).contiguous(), right=True) - 1
    indices = torch.clamp(indices, min=0, max=num_bins - 1)
    x_k = _gather_last(kx, indices)
    x_kp1 = _gather_last(kx, indices + 1)
    y_k = _gather_last(ky, indices)
    y_kp1 = _gather_last(ky, indices + 1)
    d_k = _gather_last(kd, indices)
    d_kp1 = _gather_last(kd, indices + 1)
    h_k = y_kp1 - y_k
    w_k = x_kp1 - x_k
    return {
        'inputs': inputs, 'safe': safe, 'out': out, 'out_dn': out_dn, 'out_up': out_up,
        'range_min': kx[..., 0], 'range_max': kx[..., -1],
        'x_k': x_k, 'y_k': y_k, 'd_k': d_k, 'd_kp1': d_kp1, 'h_k': h_k, 'w_k': w_k, 's_k': h_k / w_k,
    }


def rqs_forward(x, bin_widths, bin_heights, knot_slopes, range_min=-1., boundary_slope=None):
    """
    Forward rational quadratic spline (Durkan et al. 2019, appendix A.1).

    Outside the domain the map is the identity (``boundary_slope`` None, as in TensorFlow
    Probability) or linear with slope ``boundary_slope`` (circular splines).

    :param x: points.
    :param bin_widths: bin widths ``(..., K)``.
    :param bin_heights: bin heights ``(..., K)``.
    :param knot_slopes: interior knot slopes ``(..., K - 1)``.
    :param range_min: lower end of the domain.
    :param boundary_slope: optional slope at the domain ends and of the linear tails.
    :returns: transformed points.
    """
    d = _rqs_shared(x, bin_widths, bin_heights, knot_slopes, range_min, boundary_slope, inverse=False)
    relx = (d['safe'] - d['x_k']) / d['w_k']
    spline_val = d['y_k'] + ((d['h_k'] * (d['s_k'] * relx**2 + d['d_k'] * relx * (1. - relx)))
                             / (d['s_k'] + (d['d_kp1'] + d['d_k'] - 2. * d['s_k']) * relx * (1. - relx)))
    x = d['inputs']
    if boundary_slope is None:
        return torch.where(d['out'], x, spline_val)
    slope = torch.broadcast_to(boundary_slope, x.shape)
    result = torch.where(d['out_up'], slope * (x - d['range_max']) + d['range_max'], spline_val)
    return torch.where(d['out_dn'], slope * (x - d['range_min']) + d['range_min'], result)


def rqs_inverse(y, bin_widths, bin_heights, knot_slopes, range_min=-1., boundary_slope=None):
    """
    Inverse rational quadratic spline (Durkan et al. 2019, appendix A.3).

    :param y: points.
    :param bin_widths: bin widths ``(..., K)``.
    :param bin_heights: bin heights ``(..., K)``.
    :param knot_slopes: interior knot slopes ``(..., K - 1)``.
    :param range_min: lower end of the domain.
    :param boundary_slope: optional slope at the domain ends and of the linear tails.
    :returns: transformed points.
    """
    d = _rqs_shared(y, bin_widths, bin_heights, knot_slopes, range_min, boundary_slope, inverse=True)
    y = d['inputs']
    rely = torch.where(d['out'], torch.zeros_like(y), d['safe'] - d['y_k'])
    term2 = rely * (d['d_kp1'] + d['d_k'] - 2. * d['s_k'])
    a = d['h_k'] * (d['s_k'] - d['d_k']) + term2
    b = d['h_k'] * d['d_k'] - term2
    c = -d['s_k'] * rely
    discriminant = torch.clamp(b**2 - 4. * a * c, min=0.)
    denominator = -b - torch.sqrt(discriminant)
    zero_rely = rely == 0.
    denominator = torch.where(zero_rely, torch.ones_like(denominator), denominator)
    relx = torch.where(zero_rely, torch.zeros_like(rely), (2. * c) / denominator)
    spline_val = relx * d['w_k'] + d['x_k']
    if boundary_slope is None:
        return torch.where(d['out'], y, spline_val)
    slope = torch.broadcast_to(boundary_slope, y.shape)
    result = torch.where(d['out_up'], (y - d['range_max']) / slope + d['range_max'], spline_val)
    return torch.where(d['out_dn'], (y - d['range_min']) / slope + d['range_min'], result)


def rqs_forward_log_det_jacobian(x, bin_widths, bin_heights, knot_slopes, range_min=-1., boundary_slope=None):
    """
    Elementwise log derivative of the forward rational quadratic spline (appendix A.2).

    :param x: points.
    :param bin_widths: bin widths ``(..., K)``.
    :param bin_heights: bin heights ``(..., K)``.
    :param knot_slopes: interior knot slopes ``(..., K - 1)``.
    :param range_min: lower end of the domain.
    :param boundary_slope: optional slope at the domain ends and of the linear tails.
    :returns: log derivative with the shape of ``x``.
    """
    d = _rqs_shared(x, bin_widths, bin_heights, knot_slopes, range_min, boundary_slope, inverse=False)
    relx = torch.where(d['out'], torch.full_like(d['safe'], 0.5), (d['safe'] - d['x_k']) / d['w_k'])
    grad = (2. * torch.log(d['s_k'])
            + torch.log(d['d_kp1'] * relx**2 + 2. * d['s_k'] * relx * (1. - relx) + d['d_k'] * (1. - relx)**2)
            - 2. * torch.log((d['d_kp1'] + d['d_k'] - 2. * d['s_k']) * relx * (1. - relx) + d['s_k']))
    if boundary_slope is None:
        return torch.where(d['out'], torch.zeros_like(grad), grad)
    return torch.where(d['out'], torch.log(torch.broadcast_to(boundary_slope, grad.shape)), grad)


class RationalQuadraticSpline(bj.Bijector):
    """
    Elementwise rational quadratic spline bijector with fixed parameters.

    :param bin_widths: bin widths ``(..., K)`` summing to the domain width.
    :param bin_heights: bin heights ``(..., K)`` summing to the domain height.
    :param knot_slopes: interior knot slopes ``(..., K - 1)``.
    :param range_min: lower end of the domain (default -1).
    :param boundary_knot_slope: None for identity tails, else the slope at the domain ends
        and of the linear tails (circular spline).
    :param name: name of the bijector.
    """

    min_event_ndims = 0

    def __init__(self, bin_widths, bin_heights, knot_slopes, range_min=-1., boundary_knot_slope=None, name=None):
        super().__init__(name=name)
        self.range_min = float(range_min)
        self.register_buffer('bin_widths', tu.to_tensor(tu.to_numpy(bin_widths), device='cpu'))
        self.register_buffer('bin_heights', tu.to_tensor(tu.to_numpy(bin_heights), device='cpu'))
        self.register_buffer('knot_slopes', tu.to_tensor(tu.to_numpy(knot_slopes), device='cpu'))
        if boundary_knot_slope is None:
            self.boundary_knot_slope = None
        else:
            self.register_buffer('boundary_knot_slope', tu.to_tensor(tu.to_numpy(boundary_knot_slope), device='cpu'))

    def _arguments(self):
        """Spline parameters in the order of the ``rqs_*`` functions."""
        return self.bin_widths, self.bin_heights, self.knot_slopes, self.range_min, self.boundary_knot_slope

    def _forward(self, x):
        """Forward spline."""
        return rqs_forward(x, *self._arguments())

    def _inverse(self, y):
        """Inverse spline."""
        return rqs_inverse(y, *self._arguments())

    def _forward_log_det_jacobian(self, x):
        """Log derivative of the forward spline."""
        return rqs_forward_log_det_jacobian(x, *self._arguments())

###############################################################################
# transformers:


class AffineTransformer(object):
    """
    Elementwise affine transformation ``y = x * exp(log_scale) + shift`` with parameters
    ``params[..., 0] = shift`` and ``params[..., 1] = log_scale`` (TensorFlow Probability
    masked autoregressive flow convention).
    """

    num_params = 2

    def forward(self, x, params):
        """
        Apply the transformation.

        :param x: points ``(..., D)``.
        :param params: parameters ``(..., D, 2)``.
        :returns: transformed points.
        """
        return x * torch.exp(params[..., 1]) + params[..., 0]

    def inverse(self, y, params):
        """
        Apply the inverse transformation.

        :param y: points ``(..., D)``.
        :param params: parameters ``(..., D, 2)``.
        :returns: transformed points.
        """
        return (y - params[..., 0]) * torch.exp(-params[..., 1])

    def inverse_log_det_jacobian(self, y, params):
        """
        Elementwise inverse log derivative.

        :param y: points ``(..., D)``.
        :param params: parameters ``(..., D, 2)``.
        :returns: tensor ``(..., D)``.
        """
        return -params[..., 1] + torch.zeros_like(y)


class SplineTransformer(object):
    """
    Elementwise rational quadratic spline whose bins and slopes are given by unconstrained
    parameters (softmax for the bins, sigmoid or softplus for the slopes).

    :param spline_knots: number of bins ``K``.
    :param range_max: upper end of the spline domain.
    :param range_min: lower end of the domain, defaults to ``-range_max``.
    :param equispaced_x_knots: use fixed, evenly spaced bin widths.
    :param equispaced_y_knots: use fixed, evenly spaced bin heights.
    :param slope_min: minimum knot slope.
    :param min_bin_width: minimum bin width, as a fraction of the domain width.
    :param min_bin_height: minimum bin height, as a fraction of the domain height.
    :param slope_std: if given, slopes are finite differences of the bins plus
        ``slope_std * tanh(.)``, made positive with a softplus.
    :param softplus_alpha: sharpness of that softplus.
    :param circular: use linear tails with a trainable boundary slope (periodic parameters).
    :raises ValueError: for inconsistent options.
    :raises NotImplementedError: for ``slope_std`` with ``circular``.
    """

    def __init__(self, spline_knots=8, range_max=5., range_min=None, equispaced_x_knots=False,
                 equispaced_y_knots=False, slope_min=1e-4, min_bin_width=0., min_bin_height=0.,
                 slope_std=None, softplus_alpha=10., circular=False):
        range_max = float(tu.to_numpy(range_max))
        if range_min is None:
            if not range_max > 0.:
                raise ValueError('range_max must be positive when range_min is not given.')
            range_min = -range_max
        range_min = float(tu.to_numpy(range_min))
        if equispaced_x_knots and equispaced_y_knots:
            raise ValueError('Cannot have both x and y knots equispaced.')
        if spline_knots * min_bin_width >= 1. or spline_knots * min_bin_height >= 1.:
            raise ValueError('spline_knots * min_bin_width and spline_knots * min_bin_height must be smaller than 1.')
        if circular and slope_std is not None:
            raise NotImplementedError('slope_std is not implemented for circular splines.')
        self.spline_knots = int(spline_knots)
        self.range_max = range_max
        self.range_min = range_min
        self.interval_width = range_max - range_min
        self.equispaced_x_knots = bool(equispaced_x_knots)
        self.equispaced_y_knots = bool(equispaced_y_knots)
        self.slope_min = float(slope_min)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.slope_std = slope_std
        self.softplus_alpha = float(softplus_alpha)
        self.circular = bool(circular)
        num_params = 3 * self.spline_knots if self.circular else 3 * self.spline_knots - 1
        if self.equispaced_x_knots:
            num_params -= self.spline_knots
        if self.equispaced_y_knots:
            num_params -= self.spline_knots
        self.num_params = num_params

    def _bins(self, params, minimum):
        """Softmax bins scaled to the domain."""
        return self.interval_width * (minimum + (1. - self.spline_knots * minimum) * torch.softmax(params, dim=-1))

    def spline_parameters(self, params):
        """
        Map unconstrained parameters to spline bins and slopes.

        :param params: tensor ``(..., D, num_params)``.
        :returns: tuple ``(bin_widths, bin_heights, knot_slopes, boundary_slope)``; the
            boundary slope is None for standard splines.
        """
        knots = self.spline_knots
        delta = self.interval_width / knots
        start = 0
        if self.equispaced_x_knots:
            bin_widths = delta * torch.ones_like(params[..., :knots])
        else:
            bin_widths = self._bins(params[..., start:start + knots], self.min_bin_width)
            start += knots
        if self.equispaced_y_knots:
            bin_heights = delta * torch.ones_like(params[..., :knots])
        else:
            bin_heights = self._bins(params[..., start:start + knots], self.min_bin_height)
            start += knots
        knot_slopes = params[..., start:]
        if self.slope_std is None:
            knot_slopes = self.slope_min + (2. - self.slope_min) * 2. * torch.sigmoid(knot_slopes)
        else:
            average_slope = (bin_heights[..., 1:] + bin_heights[..., :-1]) / (bin_widths[..., 1:] + bin_widths[..., :-1])
            knot_slopes = average_slope + self.slope_std * torch.tanh(knot_slopes)
            knot_slopes = F.softplus(knot_slopes * self.softplus_alpha) / self.softplus_alpha
        if self.circular:
            return bin_widths, bin_heights, knot_slopes[..., :-1], knot_slopes[..., -1]
        return bin_widths, bin_heights, knot_slopes, None

    def forward(self, x, params):
        """
        Apply the spline.

        :param x: points ``(..., D)``.
        :param params: parameters ``(..., D, num_params)``.
        :returns: transformed points.
        """
        widths, heights, slopes, boundary = self.spline_parameters(params)
        return rqs_forward(x, widths, heights, slopes, self.range_min, boundary)

    def inverse(self, y, params):
        """
        Apply the inverse spline.

        :param y: points ``(..., D)``.
        :param params: parameters ``(..., D, num_params)``.
        :returns: transformed points.
        """
        widths, heights, slopes, boundary = self.spline_parameters(params)
        return rqs_inverse(y, widths, heights, slopes, self.range_min, boundary)

    def inverse_log_det_jacobian(self, y, params):
        """
        Elementwise inverse log derivative.

        :param y: points ``(..., D)``.
        :param params: parameters ``(..., D, num_params)``.
        :returns: tensor ``(..., D)``.
        """
        widths, heights, slopes, boundary = self.spline_parameters(params)
        x = rqs_inverse(y, widths, heights, slopes, self.range_min, boundary)
        return -rqs_forward_log_det_jacobian(x, widths, heights, slopes, self.range_min, boundary)

###############################################################################
# masked autoregressive flow:


class MaskedAutoregressiveFlow(bj.Bijector):
    """
    Autoregressive bijector: dimension ``d`` of the output is an elementwise transformation of
    input ``d`` whose parameters depend on the outputs ``[..., :d]``.

    The inverse (density evaluation) takes one network pass, the forward (sampling) takes
    ``D`` passes, as in TensorFlow Probability.

    :param conditioner: network mapping ``(..., D)`` to ``(..., D, transformer.num_params)``.
    :param transformer: :class:`AffineTransformer` or :class:`SplineTransformer`.
    :param name: name of the bijector.
    """

    min_event_ndims = 1

    def __init__(self, conditioner, transformer, name=None):
        super().__init__(name=name)
        self.conditioner = conditioner
        self.transformer = transformer

    def _forward(self, x):
        """Sampling direction, ``D`` autoregressive passes."""
        y = torch.zeros_like(x)
        for _ in range(x.shape[-1]):
            y = self.transformer.forward(x, self.conditioner(y))
        return y

    def _inverse(self, y):
        """Density direction, one pass."""
        return self.transformer.inverse(y, self.conditioner(y))

    def _inverse_log_det_jacobian(self, y):
        """Inverse log determinant."""
        return self.transformer.inverse_log_det_jacobian(y, self.conditioner(y)).sum(-1)

    def _forward_log_det_jacobian(self, x):
        """Forward log determinant, ``-ildj(forward(x))``."""
        return -self._inverse_log_det_jacobian(self._forward(x))

    def forward_and_log_det_jacobian(self, x, event_ndims=1):
        """
        Forward map and log determinant with a single sampling loop.

        :param x: input tensor of shape ``(..., D)``.
        :param event_ndims: number of trailing axes forming one event (default 1, at least 1).
        :returns: tuple ``(y, log_det)``, ``log_det`` with the batch shape of ``x``.
        :raises ValueError: if ``event_ndims`` is smaller than 1.
        """
        x = self._convert(x)
        y = self._forward(x)
        return y, self._reduce(-self._inverse_log_det_jacobian(y), x, event_ndims)

    def inverse_and_log_det_jacobian(self, y, event_ndims=1):
        """
        Inverse map and log determinant with a single network pass.

        :param y: input tensor of shape ``(..., D)``.
        :param event_ndims: number of trailing axes forming one event (default 1, at least 1).
        :returns: tuple ``(x, log_det)``, ``log_det`` with the batch shape of ``y``.
        :raises ValueError: if ``event_ndims`` is smaller than 1.
        """
        y = self._convert(y)
        params = self.conditioner(y)
        x = self.transformer.inverse(y, params)
        log_det = self.transformer.inverse_log_det_jacobian(y, params).sum(-1)
        return x, self._reduce(log_det, y, event_ndims)

###############################################################################
# class to build a scaling, rotation and shift bijector:


def _qr_orthogonal_factor(matrix):
    """Orthogonal factor of the QR decomposition."""
    return torch.linalg.qr(matrix)[0]


class ScaleRotoShift(bj.Bijector):
    """
    Trainable affine bijector ``y = x A + shift`` with ``A = Q^T diag(exp(log_scale)) Q``,
    where ``Q`` is the orthogonal factor of the QR decomposition of a unit lower triangular
    matrix built from the rotation parameters. Positive definite and invertible for any
    parameter value (see arXiv:1906.00587).

    :param dimension: number of dimensions.
    :param scale: include the scaling.
    :param roto: include the rotation.
    :param shift: include the shift.
    :param initializer: ``'glorot_uniform'`` (default, as the TensorFlow version), ``'zeros'``
        (identity map) or a callable acting in place on a parameter tensor.
    :param name: name of the bijector.
    """

    min_event_ndims = 1

    def __init__(self, dimension, scale=True, roto=True, shift=True, initializer='glorot_uniform', name='Affine'):
        super().__init__(name=name)
        self.dimension = int(dimension)
        self.initializer = get_initializer(initializer)
        num_rotations = self.dimension * (self.dimension - 1) // 2
        self._register('shift', self.dimension, shift)
        self._register('log_scale', self.dimension, scale)
        self._register('rotation', num_rotations, roto)
        rows, cols = np.tril_indices(self.dimension, -1)
        self.register_buffer('tril_rows', torch.as_tensor(rows, dtype=torch.long))
        self.register_buffer('tril_cols', torch.as_tensor(cols, dtype=torch.long))
        self.reset_parameters()

    def _register(self, name, size, trainable):
        """Register a parameter, or a zero buffer when the component is disabled."""
        value = torch.zeros(size, dtype=tu.prec)
        if trainable:
            setattr(self, name, torch.nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def reset_parameters(self):
        """Initialize the trainable components with the initializer."""
        with torch.no_grad():
            for tensor in (self.shift, self.log_scale, self.rotation):
                if isinstance(tensor, torch.nn.Parameter) and tensor.numel() > 0:
                    self.initializer(tensor)

    def _orthogonal(self):
        """Orthogonal factor of the rotation parametrization."""
        eye = torch.eye(self.dimension, dtype=self.log_scale.dtype, device=self.log_scale.device)
        lower = torch.zeros_like(eye).index_put((self.tril_rows, self.tril_cols), self.rotation)
        return tu.call_with_cpu_fallback('linalg_qr', _qr_orthogonal_factor, eye + lower)

    def _matrices(self):
        """Affine matrix and its inverse."""
        q = self._orthogonal()
        affine = q.transpose(-1, -2) @ (torch.exp(self.log_scale)[:, None] * q)
        inverse_affine = q.transpose(-1, -2) @ (torch.exp(-self.log_scale)[:, None] * q)
        return affine, inverse_affine

    def _forward(self, x):
        """Apply the affine map."""
        affine, _ = self._matrices()
        return x @ affine + self.shift

    def _inverse(self, y):
        """Apply the inverse affine map."""
        _, inverse_affine = self._matrices()
        return (y - self.shift) @ inverse_affine

    def _forward_log_det_jacobian(self, x):
        """Log determinant, the sum of the log scales."""
        return self.log_scale.sum() * x.new_ones(x.shape[:-1])

###############################################################################
# helper class to build a spline-autoregressive flow:

_UNSUPPORTED_KERAS_KWARGS = [
    'kernel_regularizer', 'bias_regularizer', 'activity_regularizer', 'kernel_constraint',
    'bias_constraint', 'use_bias', 'bias_initializer', 'input_order', 'hidden_degrees',
    'conditional', 'conditional_event_shape', 'conditional_input_layers', 'unroll_loop',
    'validate_args', 'is_constant_jacobian', 'event_ndims', 'dtype',
]


class AutoregressiveFlow(TrainableTransformation):
    """
    Trainable transformation made of a chain of autoregressive layers.

    Each layer is (optionally) a permutation, an autoregressive transformation (affine or
    rational quadratic spline) parametrized by a masked or flex autoregressive network, and
    (optionally) a :class:`ScaleRotoShift` layer.

    :param num_params: number of parameters of the distribution.
    :param transformation_type: ``'affine'`` or ``'spline'``, or a list with one entry per layer.
    :param autoregressive_type: ``'masked'`` or ``'flex'``, or a list with one entry per layer.
    :param n_transformations: number of layers, defaults to ``ceil(2 log2 D) + 2``.
    :param hidden_units: list of hidden layer sizes of the networks.
    :param periodic_params: list of booleans, True for periodic parameters (spline only).
    :param activation: activation of the networks (torch callable or name), default ``torch.asinh``.
    :param kernel_initializer: network weight initializer, defaults to Keras-like variance
        scaling with scale ``1 / n_transformations``.
    :param permutations: True (random permutations with minimum variance), False/None (none)
        or a list of ``n_transformations`` permutations.
    :param scale_roto_shift: append a :class:`ScaleRotoShift` to every layer.
    :param parameters_min: minimum of the training samples (used to adjust ``range_max``).
    :param parameters_max: maximum of the training samples (used to adjust ``range_max``).
    :param map_to_unitcube: map to the unit cube before each spline (spline only).
    :param spline_knots: number of spline bins.
    :param range_max: spline domain half width.
    :param equispaced_x_knots: fixed, evenly spaced bin widths.
    :param equispaced_y_knots: fixed, evenly spaced bin heights.
    :param autoregressive_scale_with_dim: scale the flex network sizes with the dimension.
    :param autoregressive_identity_dims: dimensions left unchanged by the flex network.
    :param device: device of the bijector, None for the default device.
    :param feedback: feedback level.
    :param kwargs: options of :class:`SplineTransformer` and :class:`ScaleRotoShift`; other
        keys are ignored.
    :raises ValueError: for inconsistent or unsupported options.
    """

    def __init__(
            self,
            num_params,
            transformation_type='affine',  # 'affine' or 'spline'
            autoregressive_type='masked',  # 'masked' or 'flex'
            n_transformations=None,
            hidden_units=None,
            periodic_params=None,
            activation=torch.asinh,
            kernel_initializer=None,
            permutations=True,
            scale_roto_shift=False,
            parameters_min=None,
            parameters_max=None,
            # spline parameters:
            map_to_unitcube=False,
            spline_knots=8,
            range_max=5.,
            equispaced_x_knots=False,
            equispaced_y_knots=False,
            # other parameters:
            autoregressive_scale_with_dim=True,
            autoregressive_identity_dims=None,
            device=None,
            feedback=0,
            **kwargs):

        # check for Keras options that have no equivalent:
        _unsupported = [key for key in _UNSUPPORTED_KERAS_KWARGS if key in kwargs.keys()]
        if len(_unsupported) > 0:
            raise ValueError('Unsupported options for AutoregressiveFlow: ' + ', '.join(_unsupported))

        if n_transformations is None:
            n_transformations = int(np.ceil(2 * np.log2(num_params)) + 2)

        if hidden_units is None:
            num_hidden = int(np.ceil(2 * np.log2(num_params)) + 2)
            hidden_units = [num_hidden * 2] * 2

        if isinstance(transformation_type, str):
            _transformation_types = [transformation_type] * n_transformations
        else:
            _transformation_types = list(transformation_type)
            if len(_transformation_types) != n_transformations:
                raise ValueError('transformation_type needs one entry per transformation.')
        if isinstance(autoregressive_type, str):
            _autoregressive_types = [autoregressive_type] * n_transformations
        else:
            _autoregressive_types = list(autoregressive_type)
            if len(_autoregressive_types) != n_transformations:
                raise ValueError('autoregressive_type needs one entry per transformation.')
        _has_spline = 'spline' in _transformation_types

        # initialize permutations:
        if permutations is None:
            _permutations = False
        elif isinstance(permutations, bool):
            if permutations:
                _permutations = min_var_permutations(d=num_params, n=n_transformations)
            else:
                _permutations = False
        elif isinstance(permutations, Iterable) and not isinstance(permutations, str):
            _permutations = [np.asarray(_p).astype(np.int64) for _p in permutations]
            if len(_permutations) != n_transformations:
                raise ValueError('permutations needs one permutation per transformation.')
        else:
            raise ValueError('permutations must be True, False, None or a list of permutations, got '
                             + repr(permutations))
        self.permutations = _permutations

        # check type of architecture:
        if map_to_unitcube and not _has_spline:
            raise ValueError('map_to_unitcube requires spline transformations.')

        # check periodic parameters:
        if periodic_params is not None:
            # if all parameters are not periodic then set to None:
            if np.all(np.logical_not(periodic_params)):
                periodic_params = None
        if periodic_params is not None and not _has_spline:
            raise ValueError('periodic parameters require spline transformations.')

        # check ranges for non-periodic parameters:
        range_max = float(range_max)
        if parameters_min is not None and parameters_max is not None:
            if _has_spline:
                # the spline range better enclose all the samples:
                _temp_range_max = float(max(np.abs(np.amin(parameters_min)), np.abs(np.amax(parameters_max))))
                if range_max < _temp_range_max:
                    if feedback > 0:
                        print('WARNING: range_max should be larger than the maximum range of the data and is beeing adjusted.')
                        print('    range_max:', range_max)
                        print('    max range:', _temp_range_max)
                    range_max = float(tu.np_prec(_temp_range_max + 1.))
                    if feedback > 0:
                        print('    new range_max:', range_max)
        # save range max:
        self.range_max = range_max if _has_spline else None

        # initialize kernel initializer:
        if kernel_initializer is None:
            kernel_initializer = VarianceScaling(scale=1. / n_transformations)

        # Build transformed distribution
        bijectors = []

        # handle first bijectors for periodic parameters:
        if periodic_params is not None:
            # scale periodic parameters to the spline range:
            temp_bijectors = []
            for i in range(num_params):
                if periodic_params[i]:
                    temp_bijectors.append(bj.Scale(1. / range_max))
                else:
                    temp_bijectors.append(bj.Identity())
            bijectors.append(bj.Blockwise(temp_bijectors, name='PeriodicPreprocessing'))

        for i in range(n_transformations):

            # add permutations:
            if _permutations:
                if periodic_params is None or i > 0:
                    bijectors.append(bj.Permute(_permutations[i]))

            # add map to unit cube
            if map_to_unitcube:
                bijectors.append(bj.Invert(bj.NormalCDF()))

            # add main transformation
            _transformation_type = _transformation_types[i]
            _autoregressive_type = _autoregressive_types[i]

            if _transformation_type == 'affine':
                transformer = AffineTransformer()
            elif _transformation_type == 'spline':
                _spline_kwargs = stutils.filter_kwargs(kwargs, SplineTransformer)
                _spline_kwargs['spline_knots'] = spline_knots
                _spline_kwargs['equispaced_x_knots'] = equispaced_x_knots
                _spline_kwargs['equispaced_y_knots'] = equispaced_y_knots
                if map_to_unitcube:
                    _spline_kwargs['range_min'] = 0.
                    _spline_kwargs['range_max'] = 1.
                else:
                    _spline_kwargs['range_max'] = range_max
                    _spline_kwargs['circular'] = periodic_params is not None
                transformer = SplineTransformer(**_spline_kwargs)
            else:
                raise ValueError('Unknown transformation_type ' + repr(_transformation_type))
            transf_params = transformer.num_params

            ## first, get networks that parametrize transformation
            if _autoregressive_type == 'flex':
                network = FlexAutoregressiveNetwork(
                    num_params,
                    transf_params,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    scale_with_dim=autoregressive_scale_with_dim,
                    identity_dims=autoregressive_identity_dims)
            elif _autoregressive_type == 'masked':
                network = MaskedAutoregressiveNetwork(
                    num_params,
                    transf_params,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer)
            else:
                raise ValueError('Unknown autoregressive_type ' + repr(_autoregressive_type))

            bijectors.append(MaskedAutoregressiveFlow(network, transformer,
                                                      name=_transformation_type + '_maf_' + str(i)))
            if map_to_unitcube:
                bijectors.append(bj.NormalCDF())

            # add affine layer:
            if scale_roto_shift:
                _affine_kwargs = stutils.filter_kwargs(kwargs, ScaleRotoShift)
                _affine_kwargs.pop('dimension', None)
                _affine_kwargs['name'] = 'affine_' + str(i)
                bijectors.append(ScaleRotoShift(num_params, **_affine_kwargs))

        self.bijector = bj.Chain(bijectors)
        self.device = tu.check_device(device)
        self.bijector.to(self.device)

        if feedback > 1:
            print("    Building Autoregressive Flow")
            print("    - # parameters          :", num_params)
            print("    - periodic parameters   :", periodic_params)
            print("    - # transformations     :", n_transformations)
            print("    - hidden_units          :", hidden_units)
            print("    - transformation_type   :", transformation_type)
            print("    - autoregressive_type   :", autoregressive_type)
            print("    - permutations          :", permutations)
            print("    - scale_roto_shift      :", scale_roto_shift)
            print("    - activation            :", activation)
            print("    - device                :", self.device)
