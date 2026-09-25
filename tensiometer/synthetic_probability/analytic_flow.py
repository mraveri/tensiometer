"""
File containing methods to define analytic flows, from known distributions from numpy or other non-differentialble functions.

This has still many things to do to be a fully functioning version of the main flow and should be used with care.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import scipy.optimize
from getdist import MCSamples

import torch

# relative imports:
from . import tensor_utilities as tu
from ..utilities import stats_utilities as stutils


###############################################################################
# numerical derivatives:


def _central_hessian(function, x, f0, steps):
    """
    Central finite-difference Hessian for one set of steps.

    :param function: function of a ``(D,)`` float64 array.
    :param x: point, shape ``(D,)``.
    :param f0: ``function(x)`` as a float64 array.
    :param steps: step of each coordinate, shape ``(D,)``.
    :returns: array of shape ``f0.shape + (D, D)``.
    """
    num_params = x.size
    hessian = np.empty(f0.shape + (num_params, num_params))

    def _value(point):
        return np.asarray(tu.to_numpy(function(point)), dtype=np.float64)

    for i in range(num_params):
        step_i = np.zeros(num_params)
        step_i[i] = steps[i]
        hessian[..., i, i] = (_value(x + step_i) - 2. * f0 + _value(x - step_i)) / steps[i]**2
        for j in range(i + 1, num_params):
            step_j = np.zeros(num_params)
            step_j[j] = steps[j]
            value = (_value(x + step_i + step_j) - _value(x + step_i - step_j)
                     - _value(x - step_i + step_j) + _value(x - step_i - step_j)) / (4. * steps[i] * steps[j])
            hessian[..., i, j] = value
            hessian[..., j, i] = value
    return hessian


def _finite_difference_hessian(function, x, num_steps=20):
    """
    Adaptive finite-difference Hessian.

    Central second differences are computed for a sequence of steps halving from
    ``2 max(|x_i|, 1)``, consecutive estimates are combined with one Richardson
    extrapolation step, and each entry takes the estimate with the smallest error estimate:
    its change with respect to the previous step plus the round-off error of the function
    values (from the precision of the values the function returns). Choosing the step
    adaptively keeps the result accurate both for float64 densities and for densities
    evaluated in float32.

    :param function: function of a ``(D,)`` array, returning a scalar or an array.
    :param x: point, shape ``(D,)``.
    :param num_steps: number of steps in the sequence.
    :returns: array of shape ``np.shape(function(x)) + (D, D)``.
    """
    x = np.asarray(tu.to_numpy(x), dtype=np.float64).reshape(-1)
    raw = np.asarray(tu.to_numpy(function(x)))
    if np.issubdtype(raw.dtype, np.floating):
        eps = np.finfo(raw.dtype).eps
    else:
        eps = np.finfo(np.float64).eps
    f0 = raw.astype(np.float64)
    noise = 10. * eps * np.maximum(np.abs(f0), 1.)
    scale = np.maximum(np.abs(x), 1.)
    steps = [2. * 2.**-k * scale for k in range(num_steps)]
    estimates = [_central_hessian(function, x, f0, _steps) for _steps in steps]
    extrapolated = [(4. * estimates[k + 1] - estimates[k]) / 3. for k in range(num_steps - 1)]
    errors = []
    for k in range(1, len(extrapolated)):
        _steps = steps[k + 1]
        roundoff = noise[..., np.newaxis, np.newaxis] / np.outer(_steps, _steps)
        errors.append(np.abs(extrapolated[k] - extrapolated[k - 1]) + roundoff)
    best = np.argmin(np.stack(errors), axis=0)
    candidates = np.stack(extrapolated[1:])
    hessian = np.take_along_axis(candidates, best[np.newaxis], axis=0)[0]
    return 0.5 * (hessian + np.swapaxes(hessian, -1, -2))


###############################################################################
# wrapper for torch distributions:


class torch_prob_wrapper():
    """Thin wrapper around ``torch.distributions`` distributions for flow usage."""
    def __init__(self, dist, prec=None, name=None, device=None):
        """
        Initialize the wrapper.

        :param dist: ``torch.distributions.Distribution`` instance with event shape ``(D,)``.
        :param prec: dtype used for casting inputs, defaults to the active precision.
        :param name: label of the distribution, defaults to the class name of ``dist``.
        :param device: device where the inputs are moved, defaults to the device of the
            distribution tensors (CPU if it cannot be found).
        """
        self.dist = dist
        self.label = name if name is not None else type(dist).__name__
        num_params = int(dist.event_shape[0])
        self.num_params = num_params
        self.names = ['p'+str(i) for i in range(num_params)]
        self.labels = list(self.names)
        self.lims = None
        self.prec = prec if prec is not None else tu.get_precision()
        if device is None:
            device = getattr(getattr(dist, 'mean', None), 'device', 'cpu')
        self.device = tu.resolve_device(device)

    def log_pdf(self, coord):
        """
        Evaluate the log density.

        :param coord: coordinates (array-like or tensor).
        :returns: CPU tensor with the log probability of ``coord``.
        """
        _coord = tu.to_tensor(coord, device=self.device, dtype=self.prec)
        with torch.no_grad():
            return self.dist.log_prob(_coord).detach().cpu()

    def pdf(self, coord):
        """
        Evaluate the density.

        :param coord: coordinates (array-like or tensor).
        :returns: numpy array with the density of ``coord``.
        """
        return np.exp(tu.to_numpy(self.log_pdf(coord)).astype(np.float64))

    def sim(self, N):
        """
        Draw samples from the wrapped distribution.

        :param N: number of samples.
        :returns: CPU tensor of sampled points.
        """
        with torch.no_grad():
            return self.dist.sample((int(N),)).detach().cpu()
    
###############################################################################
# analytic flow class:


class analytic_flow():
    """Analytic flow backed by a user-provided distribution with sampling and log-density methods."""
    
    def __init__(self, dist, name_tag=None, param_names=None, param_labels=None, lims=None):
        """
        Analytic Flow model. This is build with numpy functions in mind and can be easily adapted to other cases.

        dist needs to implement sampling with a method called dist.sim
        and a pdf value, with a method dist.pdf

        :param dist: distribution object with a ``sim(N)`` method returning ``(N, D)`` samples
            and either a ``log_pdf`` or a ``pdf`` method (``log_pdf`` is preferred when present),
            e.g. a :class:`torch_prob_wrapper`. Its ``label``, ``names``, ``labels`` and ``lims``
            attributes provide the defaults below.
        :param name_tag: name of the flow, defaults to ``dist.label``.
        :param param_names: list of parameter names, defaults to ``dist.names``.
        :param param_labels: list of parameter LaTeX labels, defaults to ``dist.labels``.
        :param lims: dictionary of parameter ranges ``{name: [min, max]}``, defaults to ``dist.lims``.
        :raises ValueError: if ``dist`` has no ``sim`` method, or neither a ``log_pdf`` nor a ``pdf`` method,
            or if ``param_names`` does not match the number of parameters of ``dist``.
        """

        # check if the input distribution has the correct methods:
        keys = ['sim']
        for _k in keys:
            if not (hasattr(dist, _k) and callable(getattr(dist, _k))):
                raise ValueError('Input distribution does not have the '+_k+' method.')
        
        # check pdf method:
        if not (hasattr(dist, 'log_pdf') and callable(getattr(dist, 'log_pdf'))):
            self.has_log_pdf = False
            if not (hasattr(dist, 'pdf') and callable(getattr(dist, 'pdf'))):
                raise ValueError('Input distribution does not have the pdf method nor the log_pdf method.')
        else:
            self.has_log_pdf = True
        
        # copy in distribution:
        self._dist = dist
        
        # initialize tedious informations, from distribution or not:

        # name tag:
        if name_tag is not None:
            self.name_tag = name_tag
        else:
            self.name_tag = self._dist.label

        # param names:
        if param_names is not None:
            self.param_names = param_names
        else:
            self.param_names = self._dist.names

        # param labels:
        if param_labels is not None:
            self.param_labels = param_labels
        else:
            self.param_labels = self._dist.labels

        # param ranges:
        if lims is not None:
            self.parameter_ranges = lims
        else:
            self.parameter_ranges = self._dist.lims

        # initialize other things:
        self.num_params = len(self.param_names)
        if hasattr(self._dist, 'names') and len(self._dist.names) != self.num_params:
            raise ValueError('param_names has ' + str(self.num_params) + ' entries while the input distribution has '
                             + str(len(self._dist.names)) + ' parameters.')

        #
        return None

    def cast(self, v):
        """
        Cast values to a CPU tensor with the flow precision.

        :param v: array-like or tensor to cast.
        :returns: CPU tensor with the active precision.
        """
        return tu.to_tensor(tu.to_numpy(v), device='cpu')
    
    def sample(self, num_samples):
        """
        Draw samples from the analytic distribution.

        :param num_samples: number of samples to generate.
        :returns: sampled points as a CPU tensor.
        """
        return self.cast(self._dist.sim(num_samples))

    def MCSamples(self, size, logLikes=True, **kwargs):
        """
        Return MCSamples object from the syntetic probability.

        :param size: number of samples
        :param logLikes: logical, whether to include log-likelihoods or not.
            When ``True`` the getdist ``loglikes`` are minus the log probability of the samples.
        :param kwargs: additional arguments forwarded to the getdist ``MCSamples`` constructor,
            filtered to its named arguments (e.g. ``settings``, ``weights``).
        :returns: getdist ``MCSamples`` object with the samples, names, labels, ranges and name tag of the flow.
        """
        samples = self.sample(size)
        if logLikes:
            loglikes = -self.log_probability(samples)
        else:
            loglikes = None
        mc_samples = MCSamples(
            samples=tu.to_numpy(samples),
            loglikes=None if loglikes is None else tu.to_numpy(loglikes),
            names=self.param_names,
            labels=self.param_labels,
            ranges=self.parameter_ranges,
            name_tag=self.name_tag,
            **stutils.filter_kwargs(kwargs, MCSamples)
            )
        #
        return mc_samples
    
    def log_probability(self, coord):
        """
        Evaluate the log probability of coordinates under the analytic distribution.

        :param coord: coordinates to evaluate.
        :returns: log probability tensor.
        """
        # digest input:
        _coord = tu.to_numpy(coord)
        # return:
        if self.has_log_pdf:
            return self.cast(self._dist.log_pdf(_coord))
        else:
            return self.cast(np.log(tu.to_numpy(self._dist.pdf(_coord))))
    
    def log_probability_jacobian(self, coord):
        """
        Compute the Jacobian of the log probability at given coordinates.

        :param coord: coordinates to evaluate.
        :returns: Jacobian of the log density.
        """
        # digest input:
        _coord = tu.to_numpy(coord)
        # cheaply vectorize and branch:
        if self.has_log_pdf:
            if len(_coord.shape) > 1:
                return self.cast(np.array([scipy.optimize.approx_fprime(_c, lambda x: np.squeeze(tu.to_numpy(self._dist.log_pdf(x)))) for _c in _coord]))
            else:
                return self.cast(scipy.optimize.approx_fprime(_coord, lambda x: np.squeeze(tu.to_numpy(self._dist.log_pdf(x)))))
        else:
            if len(_coord.shape) > 1:
                return self.cast(np.array([scipy.optimize.approx_fprime(_c, lambda x: np.log(tu.to_numpy(self._dist.pdf(x)))) for _c in _coord]))
            else:
                return self.cast(scipy.optimize.approx_fprime(_coord, lambda x: np.log(tu.to_numpy(self._dist.pdf(x)))))
        
    def _log_density(self, x):
        """Log density of one point, from ``log_pdf`` when available and ``log(pdf)`` otherwise."""
        if self.has_log_pdf:
            return tu.to_numpy(self._dist.log_pdf(x))
        return np.log(tu.to_numpy(self._dist.pdf(x)))

    def _point_hessian(self, x):
        """Hessian of the log density at one point, shape ``(D, D)``."""
        hessian = _finite_difference_hessian(self._log_density, x)
        if hessian.ndim > 2 and np.prod(hessian.shape[:-2]) == 1:
            hessian = hessian.reshape(hessian.shape[-2:])
        return hessian

    def log_probability_hessian(self, coord):
        """
        Compute the Hessian of the log probability at given coordinates, with adaptive
        finite differences (``_finite_difference_hessian``).

        :param coord: coordinates to evaluate, shape ``(D,)`` or ``(N, D)``.
        :returns: Hessian of the log density, shape ``(D, D)`` or ``(N, D, D)``.
        """
        # digest input:
        _coord = tu.to_numpy(coord)
        # cheaply vectorize:
        if len(_coord.shape) > 1:
            return self.cast(np.array([self._point_hessian(_c) for _c in _coord]))
        else:
            return self.cast(self._point_hessian(_coord))

