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
    Returns bijector that Gaussianize the 1D uniform distribution in [a, b].

    Parameters:
    a (float): The lower bound of the uniform distribution.
    b (float): The upper bound of the uniform distribution.
    prec (function, optional): The precision function used for numerical stability. Defaults to np_prec.

    Returns:
    tfb.Chain: The bijector Gaussianizing the uniform distribution.
    """
    return tfb.Chain([tfb.Shift(prec((a+b)/2)), tfb.Scale(prec(b-a)), tfb.Shift(-0.5), tfb.NormalCDF()])


def normal(mean, sigma, prec=np_prec):
    """
    Returns bijector that Gaussianize the 1D normal distribution with mean and variance sigma.
    This ammounts to a shift and rescaling since the normal distribution is already Gaussian.

    Parameters:
    mean (float): The mean of the normal distribution.
    sigma (float): The standard deviation of the normal distribution.
    prec (float, optional): The precision parameter. Defaults to np_prec.

    Returns:
    tfb.Chain: The bijector for the normal distribution.
    """
    return tfb.Chain([tfb.Shift(prec(mean)), tfb.Scale(prec(sigma))])


def multivariate_normal(mean, covariance, prec=np_prec):
    """
    Returns bijector to Gaussianize the ND normal distribution with mean mu and given covariance.
    This ammounts to a shift and rescaling since the normal distribution is already Gaussian.

    Parameters:
    mean (array-like): Mean of the normal distribution.
    covariance (array-like): Covariance matrix of the normal distribution.
    prec (dtype, optional): Precision type for numerical computations. Defaults to np_prec.

    Returns:
    tfb.Bijector: Bijector for the multivariate normal distribution.
    """
    return tfd.MultivariateNormalTriL(mean=mean.astype(prec), scale_tril=tf.linalg.cholesky(covariance.astype(prec))).bijector


###############################################################################
# helper function to generate analytic prior bijectors


def prior_bijector_helper(prior_dict_list=None, name=None, loc=None, cov=None, **kwargs):
    """
    Example usage

    # prior_dict_list should contain a mode keyword. If mode is 'uniform', then
    # the prior is uniform on the interval [lower, upper]. If mode is 'gaussian',
    # then the prior is gaussian with mean and scale.
    
    # uniform on x
    a = -1
    b = 3

    # gaussian on y
    mu = 0.5
    sig = 3.

    prior = prior_bijector_helper([{'lower':a, 'upper':b}, {'mean':mu, 'scale':sig}])
    diff = FlowCallback(chain, trainable_bijector=prior, Y2X_is_identity=True)

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

    This bijector maps an input `x` to `x - floor((x - minval) / delta) * delta`,
    where `minval` is the lower bound of the modulus and `delta` is the difference
    between the upper and lower bounds.

    Args:
        minval (float): The lower bound of the modulus. Default is 0.0.
        maxval (float): The upper bound of the modulus. Default is 1.0.
        validate_args (bool): Whether to validate input arguments. Default is False.
        name (str): The name of the bijector. Default is 'mod'.
        dtype (tf.DType): The data type of the input. Default is np_prec.

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

        Args:
            minval (float): The lower bound of the modulus.
            maxval (float): The upper bound of the modulus.
            validate_args (bool): Whether to validate input arguments.
            name (str): The name of the bijector.
            dtype (tf.DType): The data type of the input.
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
        return True

    def _forward(self, x):
        return x - tf.math.floor((x-self.minval)/self.delta)*self.delta

    def _inverse(self, y):
        return y - tf.math.floor((y-self.minval)/self.delta)*self.delta
    
    def _forward_log_det_jacobian(self, x):
        return tf.zeros(tf.shape(x))

    def _inverse_log_det_jacobian(self, y):
        return tf.zeros(tf.shape(y))
    