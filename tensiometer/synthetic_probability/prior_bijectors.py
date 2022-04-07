###############################################################################
# initial imports and set-up:

import numpy as np

# tensorflow imports:
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from tensorflow.keras.callbacks import Callback
prec = tf.float32
np_prec = np.float32


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
    def uniform(a, b):
        return tfb.Chain([tfb.Shift(np_prec((a+b)/2)), tfb.Scale(np_prec(b-a)), tfb.Shift(-0.5), tfb.NormalCDF()])

    def normal(mu, sig):
        return tfb.Chain([tfb.Shift(np_prec(mu)), tfb.Scale(np_prec(sig))])

    def multivariate_normal(loc, cov):
        return tfd.MultivariateNormalTriL(loc=loc.astype(np_prec), scale_tril=tf.linalg.cholesky(cov.astype(np_prec))).bijector

    if prior_dict_list is not None:  # Mix of uniform and gaussian one-dimensional priors

        # Build one-dimensional bijectors
        n = len(prior_dict_list)
        temp_bijectors = []
        for i in range(n):
            if 'lower' in prior_dict_list[i].keys():
                temp_bijectors.append(uniform(prior_dict_list[i]['lower'], prior_dict_list[i]['upper']))
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
