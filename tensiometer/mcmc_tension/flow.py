"""
This module contains functions to estimate the probability of a parameter shift given a 
synthetic probability model of the parameter difference distribution.

For further details we refer to `arxiv 2105.03324 <https://arxiv.org/pdf/2105.03324.pdf>`_.
"""

###############################################################################
# initial imports and set-up:

import numpy as np

from ..utilities import stats_utilities as stutils

###############################################################################
# function to estimate the flow probability of zero shift:


def estimate_shift(flow, prior_flow=None, tol=0.05, max_iter=1000, step=100000):
    """
    Compute the normalizing flow estimate of the probability of a parameter shift 
    given the input parameter difference chain. 
    This is done with a Monte Carlo estimate by comparing the probability density 
    at the zero-shift point to that at samples drawn from the normalizing flow 
    approximation of the distribution.

    :param flow: the input flow for a parameter difference distribution.
    :param prior_flow: the input flow for the prior distribution, defaults to None.
    :param tol: absolute tolerance on the shift significance, defaults to 0.05.
    :type tol: float, optional
    :param max_iter: maximum number of sampling steps, defaults to 1000.
    :type max_iter: int, optional
    :param step: number of samples per step, defaults to 100000.
    :type step: int, optional
    :return: probability value and error estimate.
    """
    err = np.inf
    counter = max_iter

    # define threshold for tension calculation:
    _thres = flow.log_probability(flow.cast([np.zeros(flow.num_params)]))[0]
    if prior_flow is not None:
        _thres = _thres - prior_flow.log_probability(prior_flow.cast([np.zeros(prior_flow.num_params)]))[0]

    _num_filtered = 0
    _num_samples = 0
    while err > tol and counter >= 0:
        counter -= 1
        # sample from the flow:
        _s = flow.sample(step)
        # compute probability values:
        _s_prob = flow.log_probability(_s)
        if prior_flow is not None:
            _s_prob = _s_prob - prior_flow.log_probability(prior_flow.cast(_s))
        # test:
        _t = np.array(_s_prob > _thres)
        # update counters:
        _num_filtered += np.sum(_t)
        _num_samples += step
        _P = float(_num_filtered)/float(_num_samples)
        _low, _upper = stutils.clopper_pearson_binomial_trial(float(_num_filtered),
                                                            float(_num_samples),
                                                            alpha=0.32)

        err = np.abs(stutils.from_confidence_to_sigma(_upper)-stutils.from_confidence_to_sigma(_low))

    return _P, _low, _upper


def estimate_shift_from_samples(flow, prior_flow=None):
    """
    Compute the normalizing flow estimate of the probability of a parameter shift 
    given the input parameter difference chain. 
    This is done with a Monte Carlo estimate by comparing the probability density 
    at the zero-shift point to that at samples drawn from the normalizing flow 
    approximation of the distribution.
    This function does not use the flow to sample but rather uses the input samples.
    As a downside there is no control on the end accuracy of the estimate.
    
    :param flow: the input flow for a parameter difference distribution.
    :param prior_flow: the input flow for the prior distribution, defaults to None.
    :param tol: absolute tolerance on the shift significance, defaults to 0.05.
    :type tol: float, optional
    :param max_iter: maximum number of sampling steps, defaults to 1000.
    :type max_iter: int, optional
    :param step: number of samples per step, defaults to 100000.
    :type step: int, optional
    :return: probability value and error estimate.
    """

    # define threshold for tension calculation:
    _thres = flow.log_probability(flow.cast([np.zeros(flow.num_params)]))[0]
    if prior_flow is not None:
        _thres = _thres - prior_flow.log_probability(prior_flow.cast([np.zeros(prior_flow.num_params)]))[0]

    # calculate probability on the samples:
    _s_prob = flow.log_probability(flow.cast(flow.chain_samples))
    if prior_flow is not None:
        _s_prob = _s_prob - prior_flow.log_probability(prior_flow.cast(flow.chain_samples))

    # calculate probability of shift:
    _t = np.array(_s_prob > _thres)
    _num_filtered = np.sum(_t)
    _num_samples = len(_t)
    _P = float(_num_filtered)/float(_num_samples)
    _low, _upper = stutils.clopper_pearson_binomial_trial(float(_num_filtered),
                                                        float(_num_samples),
                                                        alpha=0.32)

    return _P, _low, _upper

###############################################################################
# helper function to compute tension with default MAF:


def flow_parameter_shift(diff_chain, cache_dir=None, root_name='sprob', tol=0.05, max_iter=1000, step=100000, **kwargs):
    """
    Wrapper function to compute a normalizing flow estimate of the probability of a parameter shift given the input 
    parameter difference chain. 
    The function accepts as kwargs all the ones that are relevant for the function flow_from_chain.

    :param diff_chain: input parameter difference chain.
    :type diff_chain: :class:`~getdist.mcsamples.MCSamples`
    :param cache_dir: name of the directory to save training cache files. If none (default) does not cache.
    :param root_name: root name for the cache files.
    :param tol: absolute tolerance on the shift significance, defaults to 0.05.
    :type tol: float, optional
    :param max_iter: maximum number of sampling steps, defaults to 1000.
    :type max_iter: int, optional
    :param step: number of samples per step, defaults to 100000.
    :type step: int, optional
    :return: probability value and error estimate, then the parameter difference flow
    """
    # since this function uses the synthetic_probability module, we need to import it here:
    from ..synthetic_probability import synthetic_probability as sp
    # initialize and train parameter difference flow:
    diff_flow = sp.flow_from_chain(diff_chain, cache_dir=cache_dir, root_name=root_name, **kwargs)
    # Compute tension:
    result = estimate_shift(diff_flow, tol=tol, max_iter=max_iter, step=step)
    #
    return result, diff_flow
