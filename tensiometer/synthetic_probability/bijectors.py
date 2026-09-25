"""
Bijector base class and fixed (non trainable) bijectors implemented in PyTorch.

A bijector is an invertible, differentiable map between spaces of the same dimension.
The conventions follow TensorFlow Probability:

- ``forward(x)`` maps the base (abstract) space to the data space and ``inverse(y)`` goes back;
- ``forward_log_det_jacobian(x, event_ndims=1)`` returns ``log |det dy/dx|`` summed over the
  trailing ``event_ndims`` axes, ``inverse_log_det_jacobian(y, event_ndims=1)`` the inverse one;
- :class:`Chain` applies its bijectors last to first in the forward direction.

Bijectors are ``torch.nn.Module`` objects, so their parameters and constants move with
``.to(device)``, are saved with the flow and can be trained. No results are cached, which keeps
nested (higher order) automatic differentiation exact.
"""

###############################################################################
# initial imports and set-up:

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import tensor_utilities as tu

###############################################################################
# base class:


class Bijector(torch.nn.Module):
    """
    Base class for all bijectors.

    Subclasses implement ``_forward`` and ``_inverse`` and at least one of
    ``_forward_log_det_jacobian`` and ``_inverse_log_det_jacobian``. The log determinant hooks
    return values for events of ``min_event_ndims`` dimensions (elementwise for 0, per vector
    for 1); the public methods broadcast and sum them to the requested ``event_ndims``.

    :param name: name of the bijector, defaults to the lowercase class name.
    """

    min_event_ndims = 0

    def __init__(self, name=None):
        super().__init__()
        tu.lock_precision()
        if name is None:
            name = type(self).__name__.lower()
        self.name = name

    ###########################################################################
    # input handling:

    def _device(self):
        """Return the device of the bijector tensors, or None if it holds none."""
        return tu.module_device(self)

    def _convert(self, x):
        """
        Convert an input to a tensor with the active precision on the bijector device.

        :param x: array-like or tensor.
        :returns: tensor.
        """
        device = self._device()
        if torch.is_tensor(x):
            if x.dtype != tu.prec:
                x = x.to(tu.prec)
            if device is not None and x.device != device:
                x = x.to(device)
            return x
        return tu.to_tensor(x, device=device)

    def _check_event_ndims(self, event_ndims):
        """
        Validate the requested event dimensionality.

        :param event_ndims: number of trailing axes that form one event.
        :raises ValueError: if ``event_ndims`` is smaller than ``min_event_ndims``.
        """
        if event_ndims < self.min_event_ndims:
            raise ValueError(
                'event_ndims (' + str(event_ndims) + ') must be at least min_event_ndims ('
                + str(self.min_event_ndims) + ') for bijector ' + self.name)

    def _reduce(self, log_det, x, event_ndims):
        """
        Broadcast a log determinant to the batch shape and sum it over the event axes.

        :param log_det: log determinant from a hook.
        :param x: input of the hook, used for the batch shape.
        :param event_ndims: number of trailing axes that form one event.
        :returns: tensor of shape ``x.shape[:x.dim() - event_ndims]``.
        :raises ValueError: if ``event_ndims`` is smaller than ``min_event_ndims``.
        """
        self._check_event_ndims(event_ndims)
        target_shape = x.shape[:x.dim() - self.min_event_ndims]
        log_det = torch.as_tensor(log_det, dtype=x.dtype, device=x.device)
        if log_det.shape != target_shape:
            log_det = torch.broadcast_to(log_det, target_shape)
        reduce_ndims = event_ndims - self.min_event_ndims
        if reduce_ndims > 0:
            log_det = log_det.sum(dim=tuple(range(-reduce_ndims, 0)))
        return log_det

    ###########################################################################
    # public interface:

    def forward(self, x):
        """
        Apply the forward map.

        :param x: input tensor of shape ``(..., D)``.
        :returns: transformed tensor.
        """
        return self._forward(self._convert(x))

    def inverse(self, y):
        """
        Apply the inverse map.

        :param y: input tensor of shape ``(..., D)``.
        :returns: transformed tensor.
        """
        return self._inverse(self._convert(y))

    def forward_log_det_jacobian(self, x, event_ndims=1):
        """
        Log absolute determinant of the Jacobian of the forward map.

        :param x: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tensor with the batch shape of ``x``.
        """
        x = self._convert(x)
        return self._reduce(self._forward_log_det_jacobian(x), x, event_ndims)

    def inverse_log_det_jacobian(self, y, event_ndims=1):
        """
        Log absolute determinant of the Jacobian of the inverse map.

        :param y: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tensor with the batch shape of ``y``.
        """
        y = self._convert(y)
        return self._reduce(self._inverse_log_det_jacobian(y), y, event_ndims)

    def forward_and_log_det_jacobian(self, x, event_ndims=1):
        """
        Forward map and its log determinant, sharing work where possible.

        :param x: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tuple ``(y, log_det)``.
        """
        x = self._convert(x)
        return self._forward(x), self._reduce(self._forward_log_det_jacobian(x), x, event_ndims)

    def inverse_and_log_det_jacobian(self, y, event_ndims=1):
        """
        Inverse map and its log determinant, sharing work where possible.

        :param y: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tuple ``(x, log_det)``.
        """
        y = self._convert(y)
        return self._inverse(y), self._reduce(self._inverse_log_det_jacobian(y), y, event_ndims)

    ###########################################################################
    # hooks:

    def _forward(self, x):
        """Forward map on a converted tensor."""
        raise NotImplementedError

    def _inverse(self, y):
        """Inverse map on a converted tensor."""
        raise NotImplementedError

    def _forward_log_det_jacobian(self, x):
        """Forward log determinant, derived from the inverse one by default."""
        if type(self)._inverse_log_det_jacobian is Bijector._inverse_log_det_jacobian:
            raise NotImplementedError('Bijector ' + self.name + ' implements no log determinant.')
        return -self._inverse_log_det_jacobian(self._forward(x))

    def _inverse_log_det_jacobian(self, y):
        """Inverse log determinant, derived from the forward one by default."""
        if type(self)._forward_log_det_jacobian is Bijector._forward_log_det_jacobian:
            raise NotImplementedError('Bijector ' + self.name + ' implements no log determinant.')
        return -self._forward_log_det_jacobian(self._inverse(y))

    def extra_repr(self):
        """Show the bijector name in the module representation."""
        return 'name=' + repr(self.name)

###############################################################################
# elementwise bijectors:


class Identity(Bijector):
    """Identity map."""

    min_event_ndims = 0

    def _forward(self, x):
        """Return the input."""
        return x

    def _inverse(self, y):
        """Return the input."""
        return y

    def _forward_log_det_jacobian(self, x):
        """Zero log determinant."""
        return torch.zeros_like(x)

    def _inverse_log_det_jacobian(self, y):
        """Zero log determinant."""
        return torch.zeros_like(y)


class Shift(Bijector):
    """
    Shift ``y = x + shift``.

    :param shift: shift, a scalar or a tensor broadcastable to the input.
    :param name: name of the bijector.
    """

    min_event_ndims = 0

    def __init__(self, shift, name=None):
        super().__init__(name=name)
        self.register_buffer('shift', torch.as_tensor(tu.to_numpy(shift), dtype=tu.prec))

    def _forward(self, x):
        """Add the shift."""
        return x + self.shift

    def _inverse(self, y):
        """Subtract the shift."""
        return y - self.shift

    def _forward_log_det_jacobian(self, x):
        """Zero log determinant."""
        return torch.zeros_like(x)

    def _inverse_log_det_jacobian(self, y):
        """Zero log determinant."""
        return torch.zeros_like(y)


class Scale(Bijector):
    """
    Scaling ``y = scale * x``.

    :param scale: scale factor (scalar or broadcastable tensor).
    :param log_scale: logarithm of the scale, alternative to ``scale``.
    :param name: name of the bijector.
    :raises ValueError: if not exactly one of ``scale`` and ``log_scale`` is given.
    """

    min_event_ndims = 0

    def __init__(self, scale=None, log_scale=None, name=None):
        super().__init__(name=name)
        if (scale is None) == (log_scale is None):
            raise ValueError('Exactly one of scale and log_scale must be given.')
        if scale is None:
            scale = np.exp(tu.to_numpy(log_scale))
        self.register_buffer('scale', torch.as_tensor(tu.to_numpy(scale), dtype=tu.prec))

    def _forward(self, x):
        """Multiply by the scale."""
        return x * self.scale

    def _inverse(self, y):
        """Divide by the scale."""
        return y / self.scale

    def _forward_log_det_jacobian(self, x):
        """Log of the absolute scale."""
        return torch.log(torch.abs(self.scale)) * torch.ones_like(x)


class Tanh(Bijector):
    """Hyperbolic tangent ``y = tanh(x)``."""

    min_event_ndims = 0

    def _forward(self, x):
        """Apply tanh."""
        return torch.tanh(x)

    def _inverse(self, y):
        """Apply atanh."""
        return torch.atanh(y)

    def _forward_log_det_jacobian(self, x):
        """Stable ``log(1 - tanh(x)^2)``."""
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class NormalCDF(Bijector):
    """
    Standard normal cumulative distribution function ``y = Phi(x)``.

    The inverse clamps inputs in ``[0, 1]`` to ``[eps, 1 - eps]`` of the active precision so
    that it stays finite where float32 saturates (about +-5.2 in float32, +-8.1 in float64);
    the clamp is symmetric so that points on either edge of a prior get the same bounded
    Jacobian. Inputs outside ``[0, 1]`` give NaN, as in TensorFlow Probability. On devices without an ``ndtri`` kernel (MPS) the inverse runs on
    the host.
    """

    min_event_ndims = 0

    def _forward(self, x):
        """Apply the normal CDF."""
        return torch.special.ndtr(x)

    def _inverse(self, y):
        """Apply the normal quantile function."""
        finfo = torch.finfo(y.dtype)
        outside = (y < 0.) | (y > 1.)
        y_safe = torch.clamp(y, min=finfo.eps, max=1. - finfo.eps)
        x = tu.call_with_cpu_fallback('ndtri', torch.special.ndtri, y_safe)
        return torch.where(outside, torch.full_like(x, float('nan')), x)

    def _forward_log_det_jacobian(self, x):
        """Log of the normal density."""
        return -0.5 * x**2 - 0.5 * math.log(2. * math.pi)

###############################################################################
# vector bijectors:


class Permute(Bijector):
    """
    Permutation of the last axis, ``y = x[..., permutation]``.

    :param permutation: sequence of integers, a permutation of ``range(D)``.
    :param name: name of the bijector.
    :raises ValueError: if ``permutation`` is not a permutation.
    """

    min_event_ndims = 1

    def __init__(self, permutation, name=None):
        super().__init__(name=name)
        permutation = np.asarray(tu.to_numpy(permutation)).astype(np.int64)
        if permutation.ndim != 1 or not np.array_equal(np.sort(permutation), np.arange(len(permutation))):
            raise ValueError('permutation must be a permutation of range(D), got ' + str(permutation))
        self.register_buffer('permutation', torch.as_tensor(permutation))
        self.register_buffer('inverse_permutation', torch.as_tensor(np.argsort(permutation)))

    def _forward(self, x):
        """Permute the last axis."""
        return x[..., self.permutation]

    def _inverse(self, y):
        """Undo the permutation."""
        return y[..., self.inverse_permutation]

    def _forward_log_det_jacobian(self, x):
        """Zero log determinant."""
        return x.new_zeros(x.shape[:-1])

    def _inverse_log_det_jacobian(self, y):
        """Zero log determinant."""
        return y.new_zeros(y.shape[:-1])


class AffineTriL(Bijector):
    """
    Affine map with a lower triangular matrix, ``y = L x + shift``.

    ``AffineTriL(mean, cholesky(cov))`` maps standard normal samples to samples with the
    given mean and covariance.

    :param shift: shift vector of shape ``(D,)``.
    :param scale_tril: lower triangular matrix of shape ``(D, D)`` with non-zero diagonal.
    :param name: name of the bijector.
    """

    min_event_ndims = 1

    def __init__(self, shift, scale_tril, name=None):
        super().__init__(name=name)
        scale_tril = torch.tril(torch.as_tensor(tu.to_numpy(scale_tril), dtype=tu.prec))
        self.register_buffer('shift', torch.as_tensor(tu.to_numpy(shift), dtype=tu.prec))
        self.register_buffer('scale_tril', scale_tril)

    def _forward(self, x):
        """Apply ``L x + shift`` to row vectors."""
        return x @ self.scale_tril.transpose(-1, -2) + self.shift

    def _inverse(self, y):
        """Solve ``L x = y - shift`` for row vectors."""
        rhs = y - self.shift
        squeeze = rhs.dim() == 1
        if squeeze:
            rhs = rhs.unsqueeze(0)
        result = torch.linalg.solve_triangular(self.scale_tril.transpose(-1, -2), rhs, upper=True, left=False)
        if squeeze:
            result = result.squeeze(0)
        return result

    def _forward_log_det_jacobian(self, x):
        """Sum of the log absolute diagonal of ``L``."""
        return torch.log(torch.abs(torch.diagonal(self.scale_tril))).sum() * x.new_ones(x.shape[:-1])

###############################################################################
# composite bijectors:


class Invert(Bijector):
    """
    Swap the forward and inverse maps of a bijector.

    :param bijector: bijector to invert.
    :param name: name of the bijector, defaults to ``'invert_' + bijector.name``.
    """

    def __init__(self, bijector, name=None):
        if name is None:
            name = 'invert_' + bijector.name
        super().__init__(name=name)
        self.bijector = bijector
        self.min_event_ndims = bijector.min_event_ndims

    def forward(self, x):
        """
        Inverse map of the wrapped bijector.

        :param x: input tensor of shape ``(..., D)``.
        :returns: transformed tensor.
        """
        return self.bijector.inverse(x)

    def inverse(self, y):
        """
        Forward map of the wrapped bijector.

        :param y: input tensor of shape ``(..., D)``.
        :returns: transformed tensor.
        """
        return self.bijector.forward(y)

    def forward_log_det_jacobian(self, x, event_ndims=1):
        """
        Inverse log determinant of the wrapped bijector.

        :param x: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tensor with the batch shape of ``x``.
        :raises ValueError: if ``event_ndims`` is smaller than ``min_event_ndims``.
        """
        return self.bijector.inverse_log_det_jacobian(x, event_ndims=event_ndims)

    def inverse_log_det_jacobian(self, y, event_ndims=1):
        """
        Forward log determinant of the wrapped bijector.

        :param y: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tensor with the batch shape of ``y``.
        :raises ValueError: if ``event_ndims`` is smaller than ``min_event_ndims``.
        """
        return self.bijector.forward_log_det_jacobian(y, event_ndims=event_ndims)

    def forward_and_log_det_jacobian(self, x, event_ndims=1):
        """
        Inverse map and log determinant of the wrapped bijector.

        :param x: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tuple ``(y, log_det)``.
        :raises ValueError: if ``event_ndims`` is smaller than ``min_event_ndims``.
        """
        return self.bijector.inverse_and_log_det_jacobian(x, event_ndims=event_ndims)

    def inverse_and_log_det_jacobian(self, y, event_ndims=1):
        """
        Forward map and log determinant of the wrapped bijector.

        :param y: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tuple ``(x, log_det)``.
        :raises ValueError: if ``event_ndims`` is smaller than ``min_event_ndims``.
        """
        return self.bijector.forward_and_log_det_jacobian(y, event_ndims=event_ndims)


class Chain(Bijector):
    """
    Composition of bijectors. As in TensorFlow Probability the forward map applies the
    bijectors from the last to the first: ``Chain([f, g]).forward(x) == f(g(x))``.

    :param bijectors: list of bijectors.
    :param name: name of the bijector.
    """

    def __init__(self, bijectors=None, name=None):
        super().__init__(name=name)
        if bijectors is None:
            bijectors = []
        self.bijectors = torch.nn.ModuleList(list(bijectors))
        self.min_event_ndims = max([_b.min_event_ndims for _b in self.bijectors], default=0)

    def forward(self, x):
        """
        Apply the bijectors last to first.

        :param x: input tensor of shape ``(..., D)``.
        :returns: transformed tensor (the converted input for an empty chain).
        """
        for bijector in reversed(self.bijectors):
            x = bijector.forward(x)
        if len(self.bijectors) == 0:
            x = self._convert(x)
        return x

    def inverse(self, y):
        """
        Apply the inverses first to last.

        :param y: input tensor of shape ``(..., D)``.
        :returns: transformed tensor (the converted input for an empty chain).
        """
        for bijector in self.bijectors:
            y = bijector.inverse(y)
        if len(self.bijectors) == 0:
            y = self._convert(y)
        return y

    def forward_and_log_det_jacobian(self, x, event_ndims=1):
        """
        Forward map and summed log determinants.

        :param x: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1), passed to
            every bijector of the chain.
        :returns: tuple ``(y, log_det)``, ``log_det`` with the batch shape of ``x``.
        :raises ValueError: if ``event_ndims`` is smaller than the largest ``min_event_ndims``
            of the chain.
        """
        x = self._convert(x)
        self._check_event_ndims(event_ndims)
        log_det = x.new_zeros(x.shape[:x.dim() - event_ndims])
        for bijector in reversed(self.bijectors):
            x, _log_det = bijector.forward_and_log_det_jacobian(x, event_ndims=event_ndims)
            log_det = log_det + _log_det
        return x, log_det

    def inverse_and_log_det_jacobian(self, y, event_ndims=1):
        """
        Inverse map and summed log determinants.

        :param y: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1), passed to
            every bijector of the chain.
        :returns: tuple ``(x, log_det)``, ``log_det`` with the batch shape of ``y``.
        :raises ValueError: if ``event_ndims`` is smaller than the largest ``min_event_ndims``
            of the chain.
        """
        y = self._convert(y)
        self._check_event_ndims(event_ndims)
        log_det = y.new_zeros(y.shape[:y.dim() - event_ndims])
        for bijector in self.bijectors:
            y, _log_det = bijector.inverse_and_log_det_jacobian(y, event_ndims=event_ndims)
            log_det = log_det + _log_det
        return y, log_det

    def forward_log_det_jacobian(self, x, event_ndims=1):
        """
        Summed forward log determinants.

        :param x: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tensor with the batch shape of ``x``.
        :raises ValueError: if ``event_ndims`` is smaller than the largest ``min_event_ndims``
            of the chain.
        """
        return self.forward_and_log_det_jacobian(x, event_ndims=event_ndims)[1]

    def inverse_log_det_jacobian(self, y, event_ndims=1):
        """
        Summed inverse log determinants.

        :param y: input tensor.
        :param event_ndims: number of trailing axes forming one event (default 1).
        :returns: tensor with the batch shape of ``y``.
        :raises ValueError: if ``event_ndims`` is smaller than the largest ``min_event_ndims``
            of the chain.
        """
        return self.inverse_and_log_det_jacobian(y, event_ndims=event_ndims)[1]


class Blockwise(Bijector):
    """
    Apply bijector ``i`` to the contiguous block ``i`` of the last axis.

    Replaces the TensorFlow Probability ``Chain([Invert(Split), JointMap(list), Split])``
    pattern.

    :param bijectors: list of bijectors, one per block.
    :param block_sizes: sizes of the blocks, defaults to one dimension per bijector.
    :param name: name of the bijector.
    :raises ValueError: if the number of block sizes differs from the number of bijectors.
    """

    min_event_ndims = 1

    def __init__(self, bijectors, block_sizes=None, name=None):
        super().__init__(name=name)
        self.bijectors = torch.nn.ModuleList(list(bijectors))
        if block_sizes is None:
            block_sizes = [1] * len(self.bijectors)
        block_sizes = [int(_s) for _s in block_sizes]
        if len(block_sizes) != len(self.bijectors):
            raise ValueError('Blockwise needs one block size per bijector.')
        self.block_sizes = block_sizes

    def _split(self, x):
        """Split the last axis into blocks."""
        if x.shape[-1] != sum(self.block_sizes):
            raise ValueError(
                'Blockwise ' + self.name + ' expects ' + str(sum(self.block_sizes))
                + ' dimensions, got ' + str(x.shape[-1]))
        return torch.split(x, self.block_sizes, dim=-1)

    def _forward(self, x):
        """Apply each forward map to its block."""
        return torch.cat([_b.forward(_x) for _b, _x in zip(self.bijectors, self._split(x))], dim=-1)

    def _inverse(self, y):
        """Apply each inverse map to its block."""
        return torch.cat([_b.inverse(_y) for _b, _y in zip(self.bijectors, self._split(y))], dim=-1)

    def _forward_log_det_jacobian(self, x):
        """Sum of the block log determinants."""
        log_det = x.new_zeros(x.shape[:-1])
        for bijector, block in zip(self.bijectors, self._split(x)):
            log_det = log_det + bijector.forward_log_det_jacobian(block, event_ndims=1)
        return log_det

    def _inverse_log_det_jacobian(self, y):
        """Sum of the block log determinants."""
        log_det = y.new_zeros(y.shape[:-1])
        for bijector, block in zip(self.bijectors, self._split(y)):
            log_det = log_det + bijector.inverse_log_det_jacobian(block, event_ndims=1)
        return log_det


def _log_abs_det(matrix):
    """Log absolute determinant of a batch of matrices."""
    return torch.linalg.slogdet(matrix)[1]


class Inline(Bijector):
    """
    Bijector defined by user functions.

    The functions must be torch functions (so that derivatives work) and, for the flow to be
    saved, module level functions (lambdas and closures cannot be pickled). If no log
    determinant function is given it is computed with automatic differentiation.

    :param forward_fn: forward map.
    :param inverse_fn: inverse map.
    :param forward_log_det_jacobian_fn: optional forward log determinant per event.
    :param inverse_log_det_jacobian_fn: optional inverse log determinant per event.
    :param forward_min_event_ndims: event dimensionality of the functions (0 or 1, default 1).
    :param name: name of the bijector.
    :raises ValueError: if ``forward_min_event_ndims`` is not 0 or 1.
    """

    def __init__(self,
                 forward_fn=None,
                 inverse_fn=None,
                 forward_log_det_jacobian_fn=None,
                 inverse_log_det_jacobian_fn=None,
                 forward_min_event_ndims=1,
                 name='inline'):
        super().__init__(name=name)
        if forward_min_event_ndims not in (0, 1):
            raise ValueError('forward_min_event_ndims must be 0 or 1.')
        self.min_event_ndims = int(forward_min_event_ndims)
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn
        self.forward_log_det_jacobian_fn = forward_log_det_jacobian_fn
        self.inverse_log_det_jacobian_fn = inverse_log_det_jacobian_fn

    def _forward(self, x):
        """Call the user forward function."""
        if self.forward_fn is None:
            raise NotImplementedError('Inline bijector ' + self.name + ' has no forward function.')
        return self.forward_fn(x)

    def _inverse(self, y):
        """Call the user inverse function."""
        if self.inverse_fn is None:
            raise NotImplementedError('Inline bijector ' + self.name + ' has no inverse function.')
        return self.inverse_fn(y)

    def _autodiff_log_det(self, fn, x):
        """Log absolute determinant of the Jacobian of ``fn`` computed with autograd."""
        from . import autodiff
        _x = x if x.requires_grad else x.detach().requires_grad_(True)
        if self.min_event_ndims == 0:
            derivative = autodiff.gradient(fn, _x, create_graph=x.requires_grad)
            return torch.log(torch.abs(derivative))
        jacobian = autodiff.batch_jacobian(fn, _x, create_graph=x.requires_grad)
        return tu.call_with_cpu_fallback('linalg_slogdet', _log_abs_det, jacobian)

    def _forward_log_det_jacobian(self, x):
        """User or autodiff forward log determinant."""
        if self.forward_log_det_jacobian_fn is not None:
            return self.forward_log_det_jacobian_fn(x)
        if self.inverse_log_det_jacobian_fn is not None:
            return -self.inverse_log_det_jacobian_fn(self._forward(x))
        return self._autodiff_log_det(self._forward, x)

    def _inverse_log_det_jacobian(self, y):
        """User or autodiff inverse log determinant."""
        if self.inverse_log_det_jacobian_fn is not None:
            return self.inverse_log_det_jacobian_fn(y)
        if self.forward_log_det_jacobian_fn is not None:
            return -self.forward_log_det_jacobian_fn(self._inverse(y))
        return self._autodiff_log_det(self._inverse, y)
