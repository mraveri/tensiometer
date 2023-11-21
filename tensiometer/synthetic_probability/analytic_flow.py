"""
File containing methods to define analytic flows, from known distributions from numpy or other non-differentialble functions.

This has still many things to do to be a fully functioning version of the main flow and should be used with care.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import scipy.optimize

import numdifftools as numdiff
from getdist import MCSamples
import tensorflow as tf

# relative imports:
from . import synthetic_probability as sp
from .. import utilities as utils

###############################################################################
# analytic flow class:


class analytic_flow():
    
    def __init__(self, dist, name_tag=None, param_names=None, param_labels=None, lims=None):
        """
        Analytic Flow model. This is build with numpy functions in mind and can be easily adapted to other cases.

        dist needs to implement sampling with a method called dist.sim
        and a pdf value, with a method dist.pdf
        """

        # check if the input distribution has the correct methods:
        keys = ['sim', 'pdf']
        for _k in keys:
            if not (hasattr(dist, _k) and callable(getattr(dist, _k))):
                raise ValueError('Input distribution does not have the '+_k+' method.')
            
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
            self.param_labels = [name.label for name in self._dist.paramNames.parsWithNames(self.param_names)]

        # param ranges:
        if lims is not None:
            self.parameter_ranges = lims
        else:
            self.parameter_ranges = self._dist.lims

        # initialize other things:
        self.num_params = len(self._dist.names)

        # initialize derivatives:
        self._jacobian_logP = numdiff.Jacobian(lambda x: np.log(self._dist.pdf(x)))
        self._hessian_logP = numdiff.Hessian(lambda x: np.log(self._dist.pdf(x)))

        #
        return None

    def cast(self, v):
        return tf.cast(v, dtype=sp.prec)
    
    def sample(self, num_samples):
        return self.cast(self._dist.sim(num_samples))

    def MCSamples(self, size, logLikes=True, **kwargs):
        """
        Return MCSamples object from the syntetic probability.

        :param size: number of samples
        :param logLikes: logical, whether to include log-likelihoods or not.
        """
        samples = self.sample(size)
        if logLikes:
            loglikes = -self.log_probability(samples)
        else:
            loglikes = None
        mc_samples = MCSamples(
            samples=samples.numpy(),
            loglikes=loglikes.numpy(),
            names=self.param_names,
            labels=self.param_labels,
            ranges=self.parameter_ranges,
            name_tag=self.name_tag,
            **utils.filter_kwargs(kwargs, MCSamples)
            )
        #
        return mc_samples
    
    def log_probability(self, coord):
        # digest input:
        if tf.is_tensor(coord):
            _coord = coord.numpy()
        else:
            _coord = coord
        #
        return self.cast(np.log(self._dist.pdf(_coord)))
    
    def log_probability_jacobian(self, coord):
        # digest input:
        if tf.is_tensor(coord):
            _coord = coord.numpy()
        else:
            _coord = coord
        # cheaply vectorize:
        #if len(_coord.shape) > 1:
        #    return self.cast(np.array([self._jacobian_logP(_c)[0] for _c in _coord]))
        #else:
        #    return self.cast(self._jacobian_logP(_coord))
        if len(_coord.shape) > 1:
            return self.cast(np.array([scipy.optimize.approx_fprime(_c, lambda x: np.log(self._dist.pdf(x))) for _c in _coord]))
        else:
            return self.cast(scipy.optimize.approx_fprime(_coord, lambda x: np.log(self._dist.pdf(x))))
        
    def log_probability_hessian(self, coord):
        # digest input:
        if tf.is_tensor(coord):
            _coord = coord.numpy()
        else:
            _coord = coord
        # cheaply vectorize:
        if len(_coord.shape) > 1:
            return self.cast(np.array([self._hessian_logP(_c)[0] for _c in _coord]))
        else:
            return self.cast(self._hessian_logP(_coord))
