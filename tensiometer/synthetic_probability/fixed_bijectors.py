"""
This file contains analytic fixed bijectors that are not trainable.
These are used to pre-Gaussianize the prior distribution and to perform
other operations on the distribution.

These bijectors are meant to be fixed and not trainable.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import torch

from . import bijectors as bj
from . import tensor_utilities as tu

###############################################################################
# Definitions of analytic bijectors for Gaussianizing the prior:


def uniform_prior(a, b):
    """
    Return a bijector that Gaussianizes the 1D uniform distribution on
    ``[a, b]``.

    :param a: lower bound of the uniform distribution.
    :param b: upper bound of the uniform distribution.
    :returns: :class:`~tensiometer.synthetic_probability.bijectors.Chain` mapping a standard
        normal to the uniform distribution.
    """
    a = float(tu.to_numpy(a))
    b = float(tu.to_numpy(b))
    return bj.Chain([bj.Shift((a + b) / 2.), bj.Scale(b - a), bj.Shift(-0.5), bj.NormalCDF()])


def normal(mean, sigma):
    """
    Return a bijector that normalizes a 1D normal distribution.

    The bijector shifts and rescales the distribution even though it is
    already Gaussian to ensure consistent downstream handling.

    :param mean: mean of the normal distribution.
    :param sigma: standard deviation of the normal distribution.
    :returns: :class:`~tensiometer.synthetic_probability.bijectors.Chain` normalizing the 1D
        normal distribution.
    """
    return bj.Chain([bj.Shift(float(tu.to_numpy(mean))), bj.Scale(float(tu.to_numpy(sigma)))])


def multivariate_normal(mean, covariance):
    """
    Return a bijector that normalizes an ``N``-D normal distribution.

    :param mean: mean of the normal distribution, shape ``(N,)``.
    :param covariance: covariance matrix of the distribution, shape ``(N, N)``.
    :returns: :class:`~tensiometer.synthetic_probability.bijectors.AffineTriL` mapping a
        standard normal to the multivariate normal.
    """
    covariance = np.asarray(tu.to_numpy(covariance), dtype=np.float64)
    return bj.AffineTriL(np.asarray(tu.to_numpy(mean)), np.linalg.cholesky(covariance))


###############################################################################
# helper function to generate analytic prior bijectors


def prior_bijector_helper(prior_dict_list=None, name=None, loc=None, cov=None, **kwargs):
    """
    Build a composite bijector from a list of simple prior definitions.

    Each element of ``prior_dict_list`` describes a one-dimensional prior with
    ``mode`` set to ``'uniform'`` or ``'gaussian'`` and the corresponding
    bounds or moments. Any ``None`` entry falls back to the identity bijector.

    :param prior_dict_list: list of dictionaries describing the prior for each
        parameter.
    :param name: optional name assigned to the resulting bijector.
    :param loc: mean of a multivariate Gaussian prior, used when ``prior_dict_list`` is None.
    :param cov: covariance of a multivariate Gaussian prior, used with ``loc``.
    :param kwargs: additional keyword arguments, accepted for compatibility and ignored.
    :returns: :class:`~tensiometer.synthetic_probability.bijectors.Blockwise` mapping the
        Gaussian base space to the concatenated priors.
    :raises ValueError: if a prior mode is unknown or no prior is given.
    """
    if prior_dict_list is not None:  # Mix of uniform and gaussian one-dimensional priors

        # Build one-dimensional bijectors
        temp_bijectors = []
        for prior_dict in prior_dict_list:
            if prior_dict is not None:
                if 'mode' not in prior_dict.keys():
                    raise ValueError('Prior dictionaries need a mode key.')
                if prior_dict['mode'] == 'uniform':
                    temp_bijectors.append(uniform_prior(prior_dict['lower'], prior_dict['upper']))
                elif prior_dict['mode'] == 'gaussian':
                    temp_bijectors.append(normal(prior_dict['mean'], prior_dict['scale']))
                else:
                    raise ValueError('Unknown prior mode ' + str(prior_dict['mode']))
            else:
                temp_bijectors.append(bj.Identity())

        return bj.Blockwise(temp_bijectors, name=name)

    elif loc is not None:  # Multivariate Gaussian prior
        if cov is None:
            raise ValueError('A multivariate Gaussian prior needs both loc and cov.')
        return multivariate_normal(loc, cov)
    else:
        raise ValueError('prior_bijector_helper needs either prior_dict_list or loc and cov.')

###############################################################################
# definition of the fixed modulus bijector


class Mod1D(bj.Bijector):
    """
    A bijector that performs modulus operation on a 1D input.

    This bijector maps an input ``x`` to
    ``x - floor((x - minval) / delta) * delta`` where ``delta`` is the distance
    between ``maxval`` and ``minval``.

    :param minval: lower bound of the modulus.
    :param maxval: upper bound of the modulus.
    :param name: name of the bijector.
    """

    min_event_ndims = 0

    def __init__(self, minval=0.0, maxval=1.0, name='mod'):
        super().__init__(name=name)
        self.minval = float(minval)
        self.maxval = float(maxval)
        self.delta = self.maxval - self.minval

    def _forward(self, x):
        """Apply the modulus operation."""
        return x - torch.floor((x - self.minval) / self.delta) * self.delta

    def _inverse(self, y):
        """Inverse of the modulus operation (same as forward)."""
        return y - torch.floor((y - self.minval) / self.delta) * self.delta

    def _forward_log_det_jacobian(self, x):
        """Jacobian log determinant for the forward transform."""
        return torch.zeros_like(x)

    def _inverse_log_det_jacobian(self, y):
        """Jacobian log determinant for the inverse transform."""
        return torch.zeros_like(y)
