"""
This file contains some functions to study convergence of the chains and
to compare the two posteriors.
"""

###############################################################################
# initial imports:

import copy
import time
import numpy as np
from getdist import MCSamples

from . import gaussian_tension as gtens
from .utilities import stats_utilities as stutils
from .utilities import tensor_eigenvalues as teig

###############################################################################
# Helpers for input tests:


def _helper_chains_to_chainlist(chains):
    if isinstance(chains, list):
        for ch in chains:
            if not isinstance(ch, MCSamples):
                raise TypeError('Input list does not contain MCSamples')
        chainlist = chains
    elif isinstance(chains, MCSamples):
        chainlist = stutils.get_separate_mcsamples(chains)
    else:
        raise TypeError('Input chains is not of MCSamples type nor a \
                         list of chains.')
    # check:
    if len(chainlist) < 2:
        raise ValueError('List of chains has less than two elements.')
    #
    return chainlist

###############################################################################
# Gelman Rubin for the means:


def GR_test(chains, param_names=None):
    """
    Function performing the Gelman Rubin (GR) test
    (described in
    `Gelman and Rubin 92 <http://www.stat.columbia.edu/~gelman/research/published/itsim.pdf>`_
    and
    `Brooks and Gelman 98 <http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf>`_)
    on a list of :class:`~getdist.mcsamples.MCSamples` or on a single
    :class:`~getdist.mcsamples.MCSamples` with different sub-chains.
    This test compares the variation of the mean across a pool of chains
    with the expected variation of the mean under the pdf that is being
    sampled.
    If we define the covariance of the mean as:

    .. math::
        C_{ij} \\equiv {\\rm Cov}_c({\\rm Mean}_s(\\theta))_{ij}

    and the mean covariance as:

    .. math::
        M_{ij} \\equiv {\\rm Mean}_c[{\\rm Cov}_s(\\theta)_{ij}]

    then we seek to maximize:

    .. math::
        R-1 = {\\rm max_{\\theta}}\\frac{C_{ij} \\theta^i \\theta^j}
              {M_{ij}\\theta^i \\theta^j}

    where the subscript :math:`c` means that the statistics is computed across
    chains while the subscrit :math:`s` indicates that it is computed across
    samples.
    In this case the maximization is solved by finding the maximum eigenvalue
    of :math:`C M^{-1}`.

    :param chains: single or list of :class:`~getdist.mcsamples.MCSamples`
    :param param_names: names of the parameters involved in the test.
        By default uses all non-derived parameters.
    :returns: value of the GR test and corresponding parameter combination
    """
    # digest chain or chains:
    chainlist = _helper_chains_to_chainlist(chains)
    # digest parameter names:
    for ch in chainlist:
        param_names = gtens._check_param_names(ch, param_names)
    # get samples and weights:
    idx = [ch.index[name] for name in param_names]
    samples = [ch.samples[:, idx] for ch in chainlist]
    weights = [ch.weights for ch in chainlist]
    #
    return GR_test_from_samples(samples, weights)


def GR_test_from_samples(samples, weights):
    """
    Lower level function to perform the Gelman Rubin (GR) test.
    This works on a list of samples from different chains and corresponding
    weights.
    Refer to :meth:`tensiometer.chains_convergence.GR_test` for
    more details of what this function is doing.

    :param samples: list of samples from different chains
    :param weights: weights of the samples for each chain
    :returns: value of the GR test and corresponding parameter combination
    """
    # initialization:
    num_chains = len(samples)
    # sum of weights:
    tot_weights = np.array([np.sum(wh) for wh in weights])
    # means and covariances:
    means = [np.dot(weights[ind], samples[ind])/tot_weights[ind]
             for ind in range(num_chains)]
    covs = [np.cov(samples[ind].T, aweights=weights[ind], ddof=0)
            for ind in range(num_chains)]
    # compute R-1:
    VM = np.cov(np.array(means).T)
    MV = np.mean(covs, axis=0)
    #
    if VM.ndim == 0:
        res, mode = VM/MV, np.array([1])
    else:
        eig, eigv = np.linalg.eig(np.dot(VM, stutils.QR_inverse(MV)))
        ind = np.argmax(eig)
        res, mode = np.abs(eig[ind]), np.abs(eigv[:, ind])
    #
    return res, mode

###############################################################################
# Gelman Rubin like test for higher moments and in 1D:


def GRn_test_1D(chains, n, param_name, theta0=None):
    """
    One dimensional higher moments test. Compares the variation of a given
    moment among the population of chains with the expected variation
    of that quantity from the samples pdf.

    This test is defined by:

    .. math::
        R_n(\\theta_0)-1 = \\frac{{\\rm Var}_c
        ({\\rm Mean}_s(\\theta-\\theta_0)^n)}{{\\rm Mean}_c
        ({\\rm Var}_s(\\theta-\\theta_0)^n) }

    where the subscript :math:`c` means that the statistics is computed across
    chains while the subscrit :math:`s` indicates that it is computed across
    samples.

    :param chains: single or list of :class:`~getdist.mcsamples.MCSamples`
    :param n: order of the moment
    :param param_name: names of the parameter involved in the test.
    :param theta0: center of the moments. By default equal to the mean.
    :returns: value of the GR moment test and corresponding parameter
        combination (an array with one since this works in 1D)
    """
    # digest chain or chains:
    chainlist = _helper_chains_to_chainlist(chains)
    # digest parameter names:
    param_name = stutils.make_list(param_name)
    for ch in chainlist:
        param_name = gtens._check_param_names(ch, param_name)
    if len(param_name) != 1:
        raise ValueError('GRn_test_1D works for one parameter only.')
    # get the weights:
    weights = [ch.weights for ch in chainlist]
    # get the samples:
    samples = [ch.samples[:, ch.index[param_name[0]]] for ch in chainlist]
    #
    return GRn_test_1D_samples(samples, weights, n, theta0)


def GRn_test_1D_samples(samples, weights, n, theta0=None):
    """
    Lower level function to compute the one dimensional higher moments
    test.
    This works on a list of samples from different chains and corresponding
    weights.
    Refer to :meth:`tensiometer.chains_convergence.GRn_test_1D` for
    more details of what this function is doing.

    :param samples: list of samples from different chains
    :param weights: weights of the samples for each chain
    :param n: order of the moment
    :param theta0: center of the moments. By default equal to the mean.
    :returns: value of the GR moment test and corresponding parameter
        combination (an array with one since this works in 1D)
    """
    # initialize:
    num_chains = len(samples)
    # get the weights:
    tot_weights = np.array([np.sum(wh) for wh in weights])
    # get the central samples:
    if theta0 is None:
        means = [np.dot(weights[ind], samples[ind])/tot_weights[ind]
                 for ind in range(num_chains)]
        central_samples = [samples[ind] - means[ind]
                           for ind in range(num_chains)]
    else:
        central_samples = [samples[ind] - theta0 for ind in range(num_chains)]
    # compute moments:
    moments = np.array([np.dot(weights[ind], central_samples[ind]**n)
                        / tot_weights[ind] for ind in range(num_chains)])
    moments2 = np.array([np.dot(weights[ind], central_samples[ind]**(2*n))
                         / tot_weights[ind] for ind in range(num_chains)])
    #
    return np.var(moments)/(np.mean(moments2-moments**2))

###############################################################################
# Gelman Rubin like test for higher moments:


def _helper_1(wh, samps, n, temp_EQ):
    for w, s in zip(wh, samps):
        res = s
        for rk in range(n-1):
            res = np.multiply.outer(res, s)
        temp_EQ += w*res
    return temp_EQ/np.sum(wh)


def _helper_2(wh, samps, n, temp_VQ, temp_EQ):
    for w, s in zip(wh, samps):
        res = s
        for rk in range(n-1):
            res = np.multiply.outer(res, s)
        temp_VQ += w*np.multiply.outer(res-temp_EQ, res-temp_EQ)
    return temp_VQ/np.sum(wh)


def GRn_test(chains, n, theta0=None, param_names=None, feedback=0,
             optimizer='ParticleSwarm', **kwargs):
    """
    Multi dimensional higher order moments convergence test.
    Compares the variation of a given
    moment among the population of chains with the expected variation
    of that quantity from the samples pdf.


    We first build the :math:`k` order tensor of parameter differences around a
    point :math:`\\tilde{\\theta}`:

    .. math::
        Q^{(k)} \\equiv Q_{i_1, \\dots, i_k} \\equiv (\\theta_{i_1}
        -\\tilde{\\theta}_{i_1}) \\cdots (\\theta_{i_k}
        -\\tilde{\\theta}_{i_k})

    then we build the tensor encoding its covariance across chains

    .. math::
        V_M = {\\rm Var}_c (E_s [Q^{(k)}])

    which is a :math:`2k`rank tensor of dimension :math:`n` and then
    build the second tensor encoding the mean in chain moment:

    .. math::
        M_V = {\\rm Mean}_c (E_s[(Q^{(k)}-E_s[Q^{(k)}])
        \\otimes(Q^{(k)}-E_s[Q^{(k)}])])

    where we have suppressed all indexes to not crowd the notation.

    Then we maximize over parameters:

    .. math::
        R_n -1 \\equiv {\\rm max}_\\theta
        \\frac{V_M \\theta^{2k}}{M_V \\theta^{2k}}

    where :math:`\\theta^{2k}` is the tensor product of :math:`\\theta` for
    :math:`2k` times.

    Differently from the 2D case this problem has no solution in terms of
    eigenvalues of tensors so far and the solution is obtained by numerical
    minimization with the pymanopt library.

    :param chains: single or list of :class:`~getdist.mcsamples.MCSamples`
    :param n: order of the moment
    :param theta0: center of the moments. By default equal to the mean
    :param param_names: names of the parameters involved in the test.
        By default uses all non-derived parameters.
    :param feedback: level of feedback. 0=no feedback, >0 increasingly chatty
    :param optimizer: choice of optimization algorithm for pymanopt.
        Default is ParticleSwarm, other possibility is TrustRegions.
    :param kwargs: keyword arguments for the optimizer.
    :returns: value of the GR moment test and corresponding parameter
        combination
    """
    # if n=1 we return the standard GR test:
    if n == 1:
        return GR_test(chains, param_names=param_names)
    # digest chain or chains:
    chainlist = _helper_chains_to_chainlist(chains)
    # digest parameter names:
    for ch in chainlist:
        param_names = gtens._check_param_names(ch, param_names)
    # if there is only one parameter call the specific function:
    if len(param_names) == 1:
        return GRn_test_1D(chainlist, n, param_name=param_names), np.array([1])
    # get the weights:
    weights = [ch.weights for ch in chainlist]
    # get the samples:
    samples = [ch.samples[:, [ch.index[name] for name in param_names]]
               for ch in chainlist]
    # call the samples function:
    return GRn_test_from_samples(samples, weights, n, theta0=theta0,
                                 feedback=feedback, optimizer=optimizer,
                                 **kwargs)


def GRn_test_from_samples(samples, weights, n, theta0=None, feedback=0,
                          optimizer='ParticleSwarm', **kwargs):
    """
    Lower level function to compute the multi dimensional higher moments
    test.
    This works on a list of samples from different chains and corresponding
    weights.
    Refer to :meth:`tensiometer.chains_convergence.GRn_test` for
    more details of what this function is doing.

    :param samples: list of samples from different chains
    :param weights: weights of the samples for each chain
    :param n: order of the moment
    :param theta0: center of the moments. By default equal to the mean
    :param feedback: level of feedback. 0=no feedback, >0 increasingly chatty
    :param optimizer: choice of optimization algorithm for pymanopt.
        Default is ParticleSwarm, other possibility is TrustRegions.
    :param kwargs: keyword arguments for the optimizer.
    :returns: value of the GR moment test and corresponding parameter
        combination
    """
    # initialization:
    initial_time = time.time()
    num_chains = len(samples)
    num_params = samples[0].shape[1]
    tot_weights = np.array([np.sum(wh) for wh in weights])
    # get the central samples:
    if theta0 is None:
        means = [np.dot(weights[ind], samples[ind])/tot_weights[ind]
                 for ind in range(num_chains)]
        central_samples = [samples[ind] - means[ind]
                           for ind in range(num_chains)]
    else:
        central_samples = [samples[ind] - theta0 for ind in range(num_chains)]
    # loop over the chains:
    EQ, VQ = [], []
    if feedback > 0:
        print('Started tensor calculations')
    for ind in range(num_chains):
        t0 = time.time()
        samps = central_samples[ind]
        wh = weights[ind]
        # compute expectation of Q:
        temp_EQ = np.zeros(tuple([num_params for i in range(n)]))
        temp_EQ = _helper_1(wh, samps, n, temp_EQ)
        # compute the covariance:
        temp_VQ = np.zeros(tuple([num_params for i in range(2*n)]))
        temp_VQ = _helper_2(wh, samps, n, temp_VQ, temp_EQ)
        # save results:
        EQ.append(copy.deepcopy(temp_EQ))
        VQ.append(copy.deepcopy(temp_VQ))
        # feedback:
        t1 = time.time()
        if feedback > 0:
            print('Chain '+str(ind+1)+') time', round(t1-t0, 1), '(s)')
    # compute statistics over chains:
    MV = np.mean(VQ, axis=0)
    VM = np.zeros(tuple([num_params for i in range(2*n)]))
    temp = np.mean(EQ, axis=0)
    for temp_EQ in EQ:
        VM += np.multiply.outer(temp_EQ-temp, temp_EQ-temp)
    VM = VM/float(len(EQ))
    # do the tensor optimization:
    if optimizer == 'GEAP':
        results = teig.max_GtRq_geap_power(VM, MV, **kwargs)
    else:
        results = teig.max_GtRq_brute(VM, MV, feedback=0,
                                      optimizer=optimizer, **kwargs)
    # finalize:
    final_time = time.time()
    if feedback > 0:
        print('Total time ', round(final_time-initial_time, 1), '(s)')
    #
    return results
