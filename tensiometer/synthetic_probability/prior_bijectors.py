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
# auxiliary definitions of analytic bijectors for the prior:


def uniform_prior(a, b, prec=np_prec):
    """
    Returns bijector for 1D uniform distribution in [a, b].
    """
    return tfb.Chain([tfb.Shift(prec((a+b)/2)), tfb.Scale(prec(b-a)), tfb.Shift(-0.5), tfb.NormalCDF()])


def normal(mean, sigma, prec=np_prec):
    """
    Returns bijector for 1D normal distribution with mean and variance sigma.
    """
    return tfb.Chain([tfb.Shift(prec(mean)), tfb.Scale(prec(sigma))])


def multivariate_normal(mean, covariance, prec=np_prec):
    """
    Returns bijector for ND normal distribution with mean mu and covariance.
    """
    return tfd.MultivariateNormalTriL(mean=mean.astype(prec), scale_tril=tf.linalg.cholesky(covariance.astype(prec))).bijector


###############################################################################
# helper function to generate analytic prior bijectors


def prior_bijector_helper(prior_dict_list=None, name=None, loc=None, cov=None, **kwargs):
    """
    Example usage

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
            if 'lower' in prior_dict_list[i].keys():
                temp_bijectors.append(uniform_prior(prior_dict_list[i]['lower'], prior_dict_list[i]['upper']))
            elif 'mean' in prior_dict_list[i].keys():
                temp_bijectors.append(normal(prior_dict_list[i]['mean'], prior_dict_list[i]['scale']))
            else:
                raise ValueError

        # Need Split() to split/merge inputs
        split = tfb.Split(n, axis=-1)

        # Chain all
        return tfb.Chain([tfb.Invert(split), tfb.JointMap(temp_bijectors), split], name=name)

    elif loc is not None:  # Multivariate Gaussian prior
        assert cov is not None
        return multivariate_normal(loc, cov)
    else:
        raise ValueError
