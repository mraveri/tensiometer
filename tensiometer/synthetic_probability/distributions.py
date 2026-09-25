"""
Probability distributions used by the flows: the standard normal base distribution, the
distribution of a bijector applied to a base distribution, and mixtures of distributions.
"""

###############################################################################
# initial imports and set-up:

import torch

from . import tensor_utilities as tu

###############################################################################
# base distribution:


def standard_normal(num_params, device=None):
    """
    Multivariate standard normal distribution with independent components.

    Argument validation is disabled so that non-finite inputs give non-finite densities
    instead of errors, as in TensorFlow Probability.

    :param num_params: number of dimensions ``D``.
    :param device: device of the distribution tensors, None for the CPU.
    :returns: ``torch.distributions.Independent`` with event shape ``(D,)``.
    """
    device = torch.device('cpu') if device is None else tu.resolve_device(device)
    loc = torch.zeros(num_params, dtype=tu.prec, device=device)
    scale = torch.ones(num_params, dtype=tu.prec, device=device)
    return torch.distributions.Independent(
        torch.distributions.Normal(loc, scale, validate_args=False), 1, validate_args=False)

###############################################################################
# transformed distribution:


def _distribution_device(distribution):
    """Device of the tensors of a distribution, or None if it cannot be found."""
    try:
        mean = distribution.mean
    except (AttributeError, NotImplementedError):
        return None
    return mean.device if torch.is_tensor(mean) else None


class TransformedDistribution(object):
    """
    Distribution of ``bijector.forward(z)`` with ``z`` drawn from ``distribution``.

    If the bijector and the base distribution live on different devices, the base space
    points are moved to the device of the base distribution for the density evaluation.

    :param distribution: base distribution with ``log_prob`` and ``sample``.
    :param bijector: bijector mapping the base space to the data space.
    """

    def __init__(self, distribution, bijector):
        self.distribution = distribution
        self.bijector = bijector

    def log_prob(self, x):
        """
        Log density of the transformed distribution.

        :param x: points of shape ``(..., D)``.
        :returns: log density of shape ``x.shape[:-1]``.
        """
        z, log_det = self.bijector.inverse_and_log_det_jacobian(x, event_ndims=1)
        device = _distribution_device(self.distribution)
        if device is not None and z.device != device:
            z = z.to(device)
            log_det = log_det.to(device)
        return self.distribution.log_prob(z) + log_det

    def sample(self, num_samples):
        """
        Draw samples.

        :param num_samples: number of samples ``N``.
        :returns: tensor of shape ``(N, D)``.
        """
        z = self.distribution.sample((int(num_samples),))
        return self.bijector.forward(z)

###############################################################################
# mixture distribution:


class Mixture(object):
    """
    Mixture of distributions with fixed weights.

    :param weights: mixture weights of shape ``(K,)``; they are normalized to sum to one.
    :param components: list of ``K`` distributions with ``log_prob`` and ``sample``, sharing one device.
    :raises ValueError: if the number of weights differs from the number of components.
    """

    def __init__(self, weights, components):
        weights = tu.to_tensor(weights)
        if weights.dim() != 1 or weights.shape[0] != len(components):
            raise ValueError('Mixture needs one weight per component.')
        self.weights = weights / weights.sum()
        self.components = list(components)

    def log_prob(self, x):
        """
        Log density of the mixture.

        :param x: points of shape ``(..., D)``.
        :returns: log density of shape ``x.shape[:-1]``.
        """
        log_probs = torch.stack([_c.log_prob(x) for _c in self.components], dim=-1)
        log_weights = torch.log(self.weights.to(log_probs.device))
        return torch.logsumexp(log_probs + log_weights, dim=-1)

    def sample(self, num_samples):
        """
        Draw samples: the number of samples of each component is multinomial and the
        concatenated samples are shuffled.

        :param num_samples: number of samples ``N``.
        :returns: tensor of shape ``(N, D)``.
        """
        num_samples = int(num_samples)
        # torch.multinomial cannot draw zero samples:
        if num_samples == 0:
            return self.components[0].sample(0)
        probs = self.weights.detach().cpu().to(torch.float64)
        counts = torch.distributions.Multinomial(num_samples, probs=probs).sample().to(torch.long)
        samples = [_c.sample(int(_n)) for _c, _n in zip(self.components, counts) if int(_n) > 0]
        samples = torch.cat(samples, dim=0)
        return samples[torch.randperm(samples.shape[0], device=samples.device)]
