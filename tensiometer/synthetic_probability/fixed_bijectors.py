"""
This file contains analytic fixed bijectors that are not trainable.
These are used to pre-Gaussianize the prior distribution and to perform
other operations on the distribution.

These bijectors are meant to be fixed and not trainable.
"""

###############################################################################
# initial imports and set-up:

import numpy as np

# tensorflow imports:
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

# default precision
np_prec = np.float32

###############################################################################
# Definitions of analytic bijectors for Gaussianizing the prior:

def uniform_prior(a, b, prec=np_prec):
    """
    Return a bijector that Gaussianizes the 1D uniform distribution on
    ``[a, b]``.

    :param a: lower bound of the uniform distribution.
    :param b: upper bound of the uniform distribution.
    :param prec: precision type used for the bijector parameters.
    :returns: ``tfb.Chain`` mapping the uniform distribution to a Gaussian.
    """
    return tfb.Chain([tfb.Shift(prec((a+b)/2)), tfb.Scale(prec(b-a)), tfb.Shift(-0.5), tfb.NormalCDF()])


def normal(mean, sigma, prec=np_prec):
    """
    Return a bijector that normalizes a 1D normal distribution.

    The bijector shifts and rescales the distribution even though it is
    already Gaussian to ensure consistent downstream handling.

    :param mean: mean of the normal distribution.
    :param sigma: standard deviation of the normal distribution.
    :param prec: precision parameter used for TensorFlow constants.
    :returns: ``tfb.Chain`` normalizing the 1D normal distribution.
    """
    return tfb.Chain([tfb.Shift(prec(mean)), tfb.Scale(prec(sigma))])


def multivariate_normal(mean, covariance, prec=np_prec):
    """
    Return a bijector that normalizes an ``N``-D normal distribution.

    :param mean: mean of the normal distribution.
    :param covariance: covariance matrix of the distribution.
    :param prec: precision type used for TensorFlow tensors.
    :returns: ``tfb.Bijector`` normalizing the multivariate normal.
    """
    return tfd.MultivariateNormalTriL(mean=mean.astype(prec), scale_tril=tf.linalg.cholesky(covariance.astype(prec))).bijector


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
    :param loc: unused placeholder maintained for backward compatibility.
    :param cov: unused placeholder maintained for backward compatibility.
    :returns: ``tfb.Chain`` mapping the concatenated priors to a Gaussian
        base space.
    """
    if prior_dict_list is not None:  # Mix of uniform and gaussian one-dimensional priors

        # Build one-dimensional bijectors
        n = len(prior_dict_list)
        temp_bijectors = []
        for i in range(n):
            
            if prior_dict_list[i] is not None:
                if 'mode' in prior_dict_list[i].keys():
                    if prior_dict_list[i]['mode'] == 'uniform':
                        temp_bijectors.append(uniform_prior(prior_dict_list[i]['lower'], prior_dict_list[i]['upper']))
                    elif prior_dict_list[i]['mode'] == 'gaussian':
                        temp_bijectors.append(normal(prior_dict_list[i]['mean'], prior_dict_list[i]['scale']))
                    else:
                        raise ValueError
            else:
                temp_bijectors.append(tfb.Identity())       
            
        # Need Split() to split/merge inputs
        split = tfb.Split(n, axis=-1)

        # Chain all
        return tfb.Chain([tfb.Invert(split), tfb.JointMap(temp_bijectors), split], name=name)

    elif loc is not None:  # Multivariate Gaussian prior
        assert cov is not None
        return multivariate_normal(loc, cov)
    else:
        raise ValueError

###############################################################################
# definition of the fixed modulus bijector


class Mod1D(tfb.Bijector):
    """
    A bijector that performs modulus operation on a 1D input.

    This bijector maps an input ``x`` to
    ``x - floor((x - minval) / delta) * delta`` where ``delta`` is the distance
    between ``maxval`` and ``minval``.

    :param minval: lower bound of the modulus.
    :param maxval: upper bound of the modulus.
    :param validate_args: whether to validate input arguments.
    :param name: name of the bijector.
    :param dtype: data type of the input.

    """

    def __init__(
            self,
            minval=0.0,
            maxval=1.0,
            validate_args=False,
            name='mod',
            dtype=np_prec):
        """
        Initializes the Mod1D bijector.

        :param minval: lower bound of the modulus.
        :param maxval: upper bound of the modulus.
        :param validate_args: whether to validate input arguments.
        :param name: name of the bijector.
        :param dtype: data type of the input.
        """

        parameters = dict(locals())
        
        self.delta = maxval - minval
        self.minval = minval
        self.maxval = maxval

        with tf.name_scope(name) as name:

            super(Mod1D, self).__init__(
                forward_min_event_ndims=0,
                is_constant_jacobian=True,
                validate_args=validate_args,
                parameters=parameters,
                dtype=dtype,
                name=name)

    @classmethod
    def _is_increasing(cls):
        """Signal that the bijector is monotonic."""
        return True

    def _forward(self, x):
        """Apply the modulus operation."""
        return x - tf.math.floor((x-self.minval)/self.delta)*self.delta

    def _inverse(self, y):
        """Inverse of the modulus operation (same as forward)."""
        return y - tf.math.floor((y-self.minval)/self.delta)*self.delta
    
    def _forward_log_det_jacobian(self, x):
        """Jacobian log determinant for the forward transform."""
        return tf.zeros(tf.shape(x))

    def _inverse_log_det_jacobian(self, y):
        """Jacobian log determinant for the inverse transform."""
        return tf.zeros(tf.shape(y))
    
