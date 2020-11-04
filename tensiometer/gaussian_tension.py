"""
This file contains the functions and utilities to compute agreement and
disagreement between two different chains using a Gaussian approximation
for the posterior.

For more details on the method implemented see
`arxiv 1806.04649 <https://arxiv.org/pdf/1806.04649.pdf>`_
and `arxiv 1912.04880 <https://arxiv.org/pdf/1912.04880.pdf>`_.
"""

"""
For testing purposes:

from getdist import loadMCSamples, MCSamples, WeightedSamples
chain_1 = loadMCSamples('./test_chains/DES')
chain_2 = loadMCSamples('./test_chains/Planck18TTTEEE')
chain_12 = loadMCSamples('./test_chains/Planck18TTTEEE_DES')
chain_prior = loadMCSamples('./test_chains/prior')

chain = chain_1

import tensiometer.utilities as utils
import matplotlib.pyplot as plt
"""

###############################################################################
# initial imports:

import scipy
import numpy as np
from getdist import MCSamples
from getdist.gaussian_mixtures import GaussianND

from . import utilities as utils

###############################################################################
# series of helpers to check input of functions:


def _check_param_names(chain, param_names):
    """
    Utility to check input param names.
    """
    if param_names is None:
        param_names = chain.getParamNames().getRunningNames()
    else:
        param_list = chain.getParamNames().list()
        if not np.all([name in param_list for name in param_names]):
            raise ValueError('Input parameter is not in the chain',
                             chain.name_tag, '\n'
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', param_list)
    #
    return param_names


def _check_chain_type(chain):
    """
    Check if an object is a GetDist chain.
    """
    # test the type of the chain:
    if not isinstance(chain, MCSamples):
        raise TypeError('Input chain is not of MCSamples type.')

###############################################################################


def get_prior_covariance(chain, param_names=None):
    """
    Utility to estimate the prior covariance from the ranges of a chain.
    The flat range prior covariance
    (`link <https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)>`_)
    is given by:

    .. math:: C_{ij} = \\delta_{ij} \\frac{( max(p_i) - min(p_i) )^2}{12}

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: optional choice of parameter names to
        restrict the calculation.
    :return: the estimated covariance of the prior.
    """
    # get the parameter names to use:
    param_names = _check_param_names(chain, param_names)
    # get the ranges:
    _prior_min = []
    _prior_max = []
    for name in param_names:
        # lower bound:
        if name in chain.ranges.lower.keys():
            _prior_min.append(chain.ranges.lower[name])
        else:
            _prior_min.append(-1.e30)
            # upper bound:
        if name in chain.ranges.upper.keys():
            _prior_max.append(chain.ranges.upper[name])
        else:
            _prior_max.append(1.e30)
    _prior_min = np.array(_prior_min)
    _prior_max = np.array(_prior_max)
    #
    return np.diag((_prior_max-_prior_min)**2/12.)

###############################################################################


def get_Neff(chain, prior_chain=None, param_names=None, prior_factor=1.0):
    """
    Function to compute the number of effective parameters constrained by a
    chain over the prior.
    The number of effective parameters is defined as in Eq. (29) of
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) as:

    .. math:: N_{\\rm eff} \\equiv
        N -{\\rm tr}[ \\mathcal{C}_\\Pi^{-1}\\mathcal{C}_p ]

    where :math:`N` is the total number of nominal parameters of the chain,
    :math:`\\mathcal{C}_\\Pi` is the covariance of the prior and
    :math:`\\mathcal{C}_p` is the posterior covariance.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param prior_chain: (optional) the prior chain.
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :param param_names: (optional) parameter names to restrict the
        calculation of :math:`N_{\\rm eff}`.
        If none is given the default assumes that all running parameters
        should be used.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :return: the number of effective parameters.
    """
    # initialize param names:
    param_names = _check_param_names(chain, param_names)
    # initialize prior covariance:
    if prior_chain is not None:
        # check parameter names:
        param_names = _check_param_names(prior_chain, param_names)
        # get the prior covariance:
        C_Pi = prior_chain.cov(pars=param_names)
    else:
        C_Pi = get_prior_covariance(chain, param_names=param_names)
    # multiply by prior factor:
    C_Pi = prior_factor*C_Pi
    # get the posterior covariance:
    C_p = chain.cov(pars=param_names)
    # compute the number of effective parameters
    _temp = np.dot(np.linalg.inv(C_Pi), C_p)
    # compute Neff from the regularized spectrum of the eigenvalues:
    _eigv, _eigvec = np.linalg.eig(_temp)
    _eigv[_eigv > 1.] = 1.
    _eigv[_eigv < 0.] = 0.
    #
    _Ntot = len(_eigv)
    _Neff = _Ntot - np.sum(_eigv)
    #
    return _Neff

###############################################################################


def gaussian_approximation(chain, param_names=None):
    """
    Function that computes the Gaussian approximation of a given chain.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param param_names: (optional) parameter names to restrict the
        Gaussian approximation.
        If none is given the default assumes that all parameters
        should be used.
    :return: :class:`~getdist.gaussian_mixtures.GaussianND` object with the
        Gaussian approximation of the chain.
    """
    # initial checks:
    _check_chain_type(chain)
    if param_names is None:
        param_names = chain.getParamNames().list()
    param_names = _check_param_names(chain, param_names)
    # get the mean:
    mean = chain.getMeans(pars=[chain.index[name]
                          for name in param_names])
    # get the covariance:
    cov = chain.cov(pars=param_names)
    # get the labels:
    param_labels = [_n.label for _n
                    in chain.getParamNames().parsWithNames(param_names)]
    # get label:
    if chain.label is not None:
        label = 'Gaussian_'+chain.label
    elif chain.name_tag is not None:
        label = 'Gaussian_'+chain.name_tag
    else:
        label = None
    # initialize the Gaussian distribution:
    gaussian_approx = GaussianND(mean, cov,
                                 names=param_names,
                                 labels=param_labels,
                                 label=label)
    #
    return gaussian_approx

###############################################################################


def Q_DM(chain_1, chain_2, prior_chain=None, param_names=None,
         cutoff=0.05, prior_factor=1.0):
    """
    Compute the value and degrees of freedom of the quadratic form giving the
    probability of a difference between the means of the two input chains,
    in the Gaussian approximation.

    This is defined as in
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) to be:

    .. math:: Q_{\\rm DM} \\equiv (\\theta_1-\\theta_2)
        (\\mathcal{C}_1+\\mathcal{C}_2
        -\\mathcal{C}_1\\mathcal{C}_\\Pi^{-1}\\mathcal{C}_2
        -\\mathcal{C}_2\\mathcal{C}_\\Pi^{-1}\\mathcal{C}_1)^{-1}
        (\\theta_1-\\theta_2)^T

    where :math:`\\theta_i` is the parameter mean of the i-th posterior,
    :math:`\\mathcal{C}` the posterior covariance and :math:`\\mathcal{C}_\\Pi`
    the prior covariance.
    :math:`Q_{\\rm DM}` is :math:`\\chi^2` distributed with number of degrees
    of freedom equal to the rank of the shift covariance.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        the second input chain.
    :param prior_chain: (optional) the prior only chain.
        If the prior is not well approximated by a ranged prior and is
        informative it is better to explicitly use a prior only chain.
        If this is not given the algorithm will assume ranged priors
        with the ranges computed from the input chains.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param cutoff: (optional) the algorithms needs to detect prior
        constrained directions (that do not contribute to the test)
        from data constrained directions.
        This is achieved through a Karhunen–Loeve decomposition to avoid issues
        with physical dimensions of parameters and cutoff sets the minimum
        improvement with respect to the prior that is used.
        Default is five percent.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :return: :math:`Q_{\\rm DM}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm DM}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using the cdf method of :py:data:`scipy.stats.chi2` or
        :meth:`tensiometer.utilities.from_chi2_to_sigma`.
    """
    # initial checks:
    if cutoff < 0.0:
        raise ValueError('The KL cutoff has to be greater than zero.\n',
                         'Input value ', cutoff)
    # initialize param names:
    param_names_1 = _check_param_names(chain_1, param_names)
    param_names_2 = _check_param_names(chain_2, param_names)
    # get common names:
    param_names = [name for name in param_names_1 if name in param_names_2]
    if len(param_names) == 0:
        raise ValueError('Chains do not have shared parameters.\n',
                         'Parameters for chain_1 ', param_names_1, '\n',
                         'Parameters for chain_2 ', param_names_2, '\n')
    # initialize prior covariance:
    if prior_chain is not None:
        param_names = _check_param_names(prior_chain, param_names)
        # get the prior covariance:
        C_Pi = prior_chain.cov(pars=param_names)
    else:
        C_Pi1 = get_prior_covariance(chain_1, param_names=param_names)
        C_Pi2 = get_prior_covariance(chain_2, param_names=param_names)
        if not np.allclose(C_Pi1, C_Pi2):
            raise ValueError('The chains have different priors.')
        else:
            C_Pi = C_Pi1
    # scale prior covariance:
    C_Pi = prior_factor*C_Pi
    # get the posterior covariances:
    C_p1, C_p2 = chain_1.cov(pars=param_names), chain_2.cov(pars=param_names)
    # get the means:
    theta_1 = chain_1.getMeans(pars=[chain_1.index[name]
                               for name in param_names])
    theta_2 = chain_2.getMeans(pars=[chain_2.index[name]
                               for name in param_names])
    param_diff = theta_1-theta_2
    # do the calculation of Q:
    C_Pi_inv = utils.QR_inverse(C_Pi)
    temp = np.dot(np.dot(C_p1, C_Pi_inv), C_p2)
    diff_covariance = C_p1 + C_p2 - temp - temp.T
    # take the directions that are best constrained over the prior:
    eig_1, eigv_1 = utils.KL_decomposition(C_p1, C_Pi)
    eig_2, eigv_2 = utils.KL_decomposition(C_p2, C_Pi)
    # get the smallest spectrum, if same use first:
    if np.sum(1./eig_1-1. > cutoff) <= np.sum(1./eig_2-1. > cutoff):
        eig, eigv = eig_1, eigv_1
    else:
        eig, eigv = eig_2, eigv_2
    # get projection matrix:
    proj_matrix = eigv[1./eig-1. > cutoff]
    # get dofs of Q:
    dofs = np.sum(1./eig-1. > cutoff)
    # project parameter difference:
    param_diff = np.dot(proj_matrix, param_diff)
    # project covariance:
    temp_cov = np.dot(np.dot(proj_matrix, diff_covariance), proj_matrix.T)
    # compute Q:
    Q_DM = np.dot(np.dot(param_diff, utils.QR_inverse(temp_cov)), param_diff)
    #
    return Q_DM, dofs

###############################################################################


def Q_UDM_KL_components(chain_1, chain_12, param_names=None):
    """
    Function that computes the Karhunen–Loeve (KL) decomposition of the
    covariance of a chain with the covariance of that chain joint with another
    one.
    This function is used for the parameter shift algorithm in
    update form.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :return: the KL eigenvalues, the KL eigenvectors and the parameter names
        that are used, sorted in decreasing order.
    """
    # initialize param names:
    param_names_1 = _check_param_names(chain_1, param_names)
    param_names_12 = _check_param_names(chain_12, param_names)
    # get common names:
    param_names = [name for name in param_names_1
                   if name in param_names_12]
    if len(param_names) == 0:
        raise ValueError('Chains do not have shared parameters.\n',
                         'Parameters for chain_1 ', param_names_1, '\n',
                         'Parameters for chain_12 ', param_names_12, '\n')
    # get the posterior covariances:
    C_p1, C_p12 = chain_1.cov(pars=param_names), chain_12.cov(pars=param_names)
    # perform the KL decomposition:
    KL_eig, KL_eigv = utils.KL_decomposition(C_p1, C_p12)
    # sort:
    idx = np.argsort(KL_eig)[::-1]
    KL_eig = KL_eig[idx]
    KL_eigv = KL_eigv[:, idx]
    #
    return KL_eig, KL_eigv, param_names

###############################################################################


def Q_UDM_get_cutoff(chain_1, chain_2, chain_12,
                     prior_chain=None, param_names=None, prior_factor=1.0):
    """
    Function to estimate the cutoff for the spectrum of parameter
    differences in update form to match Delta Neff.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        the second chain that joined with the first one (modulo the prior)
        should give the joint chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param prior_chain: :class:`~getdist.mcsamples.MCSamples` (optional)
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :return: the optimal KL cutoff, KL eigenvalues, KL eigenvectors and the
        parameter names that are used.
    """
    # get all shared parameters:
    param_names_1 = _check_param_names(chain_1, param_names)
    param_names_2 = _check_param_names(chain_2, param_names)
    # get common names:
    param_names = [name for name in param_names_1 if name in param_names_2]
    if len(param_names) == 0:
        raise ValueError('Chains do not have shared parameters.\n',
                         'Parameters for chain_1 ', param_names_1, '\n',
                         'Parameters for chain_2 ', param_names_2, '\n')
    # get the KL decomposition:
    KL_eig, KL_eigv, param_names = Q_UDM_KL_components(chain_1,
                                                       chain_12,
                                                       param_names=param_names)
    # get the cutoff that matches the dofs of Q_DMAP:
    N_1 = get_Neff(chain_1,
                   prior_chain=prior_chain,
                   param_names=param_names,
                   prior_factor=prior_factor)
    N_2 = get_Neff(chain_2,
                   prior_chain=prior_chain,
                   param_names=param_names,
                   prior_factor=prior_factor)
    N_12 = get_Neff(chain_12,
                    prior_chain=prior_chain,
                    param_names=param_names,
                    prior_factor=prior_factor)
    target_dofs = round(N_1 + N_2 - N_12)
    # compute the cutoff:

    def _helper(_c):
        return np.sum(KL_eig[KL_eig > 1.] > _c)-target_dofs
    # define the extrema:
    _a = 1.0
    _b = np.amax(KL_eig)
    # check bracketing:
    if _helper(_a)*_helper(_b) > 0:
        raise ValueError('Cannot find optimal cutoff.\n',
                         'This might be a problem with the prior.\n',
                         'You may try providing a prior chain.\n',
                         'KL spectrum:', KL_eig,
                         'Target dofs:', target_dofs)
    else:
        KL_cutoff = scipy.optimize.bisect(_helper, _a, _b)
    #
    return KL_cutoff, KL_eig, KL_eigv, param_names

###############################################################################


def Q_UDM_fisher_components(chain_1, chain_12, param_names=None, which='1'):
    """
    Compute the decomposition of the Fisher matrix in terms of KL modes.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param which: (optional) which decomposition to report. Possibilities are
        '1' for the chain 1 Fisher matrix, '2' for the chain 2 Fisher matrix
        and '12' for the joint Fisher matrix.
    :return: parameter names used in the calculation, values of improvement
        and the Fisher matrix.
    """
    KL_eig, KL_eigv, param_names = Q_UDM_KL_components(chain_1,
                                                       chain_12,
                                                       param_names=param_names)
    # compute Fisher and fractional fisher matrix:
    if which == '1':
        fisher = np.sum(KL_eigv*KL_eigv/KL_eig, axis=1)
        fractional_fisher = ((KL_eigv*KL_eigv/KL_eig).T/fisher).T
    elif which == '2':
        fisher = np.sum(KL_eigv*KL_eigv*(KL_eig-1.)/(KL_eig), axis=1)
        fractional_fisher = ((KL_eigv*KL_eigv*(KL_eig-1.)/(KL_eig)).T/fisher).T
    elif which == '12':
        fisher = np.sum(KL_eigv*KL_eigv, axis=1)
        fractional_fisher = ((KL_eigv*KL_eigv).T/fisher).T
    else:
        raise ValueError('Input parameter which can only be: 1, 2, 12.')
    #
    return param_names, KL_eig, fractional_fisher

###############################################################################


def Q_UDM(chain_1, chain_12, lower_cutoff=1.05, upper_cutoff=100.,
          param_names=None):
    """
    Compute the value and degrees of freedom of the quadratic form giving the
    probability of a difference between the means of the two input chains,
    in update form with the Gaussian approximation.

    This is defined as in
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) to be:

    .. math:: Q_{\\rm UDM} \\equiv (\\theta_1-\\theta_{12})
        (\\mathcal{C}_1-\\mathcal{C}_{12})^{-1}
        (\\theta_1-\\theta_{12})^T

    where :math:`\\theta_1` is the parameter mean of the first posterior,
    :math:`\\theta_{12}` is the parameter mean of the joint posterior,
    :math:`\\mathcal{C}` the posterior covariance and :math:`\\mathcal{C}_\\Pi`
    the prior covariance.
    :math:`Q_{\\rm UDM}` is :math:`\\chi^2` distributed with number of degrees
    of freedom equal to the rank of the shift covariance.

    In case of uninformative priors the statistical significance of
    :math:`Q_{\\rm UDM}` is the same as the one reported by
    :math:`Q_{\\rm DM}` but offers likely mitigation against non-Gaussianities
    of the posterior distribution.
    In the case where both chains are Gaussian :math:`Q_{\\rm UDM}` is
    symmetric if the first input chain is swapped :math:`1\\leftrightarrow 2`.
    If the input distributions are not Gaussian it is better to use the most
    constraining chain as the base for the parameter update.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param lower_cutoff: (optional) the algorithms needs to detect prior
        constrained directions (that do not contribute to the test)
        from data constrained directions.
        This is achieved through a Karhunen–Loeve decomposition to avoid issues
        with physical dimensions of parameters and cutoff sets the minimum
        improvement with respect to the prior that is used.
        Default is five percent.
    :param upper_cutoff: (optional) upper cutoff for the selection of KL modes.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :return: :math:`Q_{\\rm UDM}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm UDM}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using the cdf method of :py:data:`scipy.stats.chi2` or
        :meth:`tensiometer.utilities.from_chi2_to_sigma`.
    """
    # get the cutoff and perform the KL decomposition:
    _temp = Q_UDM_KL_components(chain_1, chain_12, param_names=param_names)
    KL_eig, KL_eigv, param_names = _temp
    # get the parameter means:
    theta_1 = chain_1.getMeans(pars=[chain_1.index[name]
                               for name in param_names])
    theta_12 = chain_12.getMeans(pars=[chain_12.index[name]
                                 for name in param_names])
    shift = theta_1 - theta_12
    # do the Q_UDM calculation:
    _filter = np.logical_and(KL_eig > lower_cutoff, KL_eig < upper_cutoff)
    Q_UDM = np.sum(np.dot(KL_eigv.T, shift)[_filter]**2./(KL_eig[_filter]-1.))
    dofs = np.sum(_filter)
    #
    return Q_UDM, dofs

###############################################################################
# Likelihood based estimators:


def get_MAP_loglike(chain, feedback=True):
    """
    Utility function to obtain the data part of the maximum posterior for
    a given chain.
    The best possibility is that a separate file with the posterior
    explicit MAP is given. If this is not the case then the function will try
    to get the likelihood at MAP from the samples. This possibility is far more
    noisy in general.

    :param chain: :class:`~getdist.mcsamples.MCSamples`
        the input chain.
    :param feedback: logical flag to set whether the function should print
        a warning every time the explicit MAP file is not found.
        By default this is true.
    :return: the data log likelihood at maximum posterior.
    """
    # we first try to get the best fit from explicit maximization:
    try:
        # get the best fit from the explicit MAP:
        best_fit = chain.getBestFit(max_posterior=True)
        if len(best_fit.chiSquareds) == 0:
            _best_fit_data_like = best_fit.logLike
            if 'prior' in best_fit.getParamDict().keys():
                _best_fit_data_like -= best_fit.getParamDict()['prior']
        else:
            # get the total data likelihood:
            _best_fit_data_like = 0.0
            for _dat in best_fit.chiSquareds:
                _best_fit_data_like += _dat[1].chisq
    except Exception as ex:
        # we use the best fit from the chains.
        # This is noisy so we print a warning:
        if feedback:
            print(ex)
            print('WARNING: using MAP from samples. This can be noisy.')
        _best_fit_data_like = 0.0
        # get chi2 list:
        chi_list = [name for name in chain.getLikeStats().list()
                    if 'chi2_' in name]
        # assume that we have chi2_data and the chi_2 prior:
        if 'chi2_prior' in chi_list:
            chi_list = chi_list[:chi_list.index('chi2_prior')]
        # if empty we have to guess:
        if len(chi_list) == 0:
            _best_fit_data_like = chain.getLikeStats().logLike_sample
        else:
            for name in chi_list:
                _best_fit_data_like += \
                    chain.getLikeStats().parWithName(name).bestfit_sample
    # normalize:
    _best_fit_data_like = -0.5*_best_fit_data_like
    #
    return _best_fit_data_like

###############################################################################


def Q_MAP(chain, num_data, prior_chain=None,
          normalization_factor=0.0, prior_factor=1.0, feedback=True):
    """
    Compute the value and degrees of freedom of the quadratic form giving
    the goodness of fit measure at maximum posterior (MAP), in
    Gaussian approximation.

    This is defined as in
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) to be:

    .. math:: Q_{\\rm MAP} \\equiv -2\\ln \\mathcal{L}(\\theta_{\\rm MAP})

    where :math:`\\mathcal{L}(\\theta_{\\rm MAP})` is the data likelihood
    evaluated at MAP.
    In Gaussian approximation this is distributed as:

    .. math:: Q_{\\rm MAP} \\sim \\chi^2(d-N_{\\rm eff})

    where :math:`d` is the number of data points and :math:`N_{\\rm eff}`
    is the number of effective parameters, as computed by the function
    :func:`tensiometer.gaussian_tension.get_Neff`.

    :param chain: :class:`~getdist.mcsamples.MCSamples`
        the input chain.
    :param num_data: number of data points.
    :param prior_chain: (optional) the prior chain.
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :param normalization_factor: (optional) likelihood normalization factor.
        This should make the likelihood a chi square.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :param feedback: logical flag to set whether the function should print
        a warning every time the explicit MAP file is not found.
        By default this is true.
    :return: :math:`Q_{\\rm MAP}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm MAP}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using the cdf method of :py:data:`scipy.stats.chi2` or
        :meth:`tensiometer.utilities.from_chi2_to_sigma`.
    """
    # get the best fit:
    best_fit_data_like = get_MAP_loglike(chain, feedback=feedback)
    # get the number of effective parameters:
    Neff = get_Neff(chain, prior_chain=prior_chain, prior_factor=prior_factor)
    # compute Q_MAP:
    Q_MAP = -2.*best_fit_data_like + normalization_factor
    # compute the number of degrees of freedom:
    dofs = float(num_data) - Neff
    #
    return Q_MAP, dofs

###############################################################################


def Q_DMAP(chain_1, chain_2, chain_12, prior_chain=None,
           param_names=None, prior_factor=1.0, feedback=True):
    """
    Compute the value and degrees of freedom of the quadratic form giving
    the goodness of fit loss measure, in Gaussian approximation.

    This is defined as in
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_) to be:

    .. math:: Q_{\\rm DMAP} \\equiv Q_{\\rm MAP}^{12} -Q_{\\rm MAP}^{1}
        -Q_{\\rm MAP}^{2}

    where :math:`Q_{\\rm MAP}^{12}` is the joint likelihood at maximum
    posterior (MAP) and :math:`Q_{\\rm MAP}^{i}` is the likelihood at MAP
    for the two single data sets.
    In Gaussian approximation this is distributed as:

    .. math:: Q_{\\rm DMAP} \\sim \\chi^2(N_{\\rm eff}^1 + N_{\\rm eff}^2 -
        N_{\\rm eff}^{12})

    where :math:`N_{\\rm eff}` is the number of effective parameters,
    as computed by the function :func:`tensiometer.gaussian_tension.get_Neff`
    for the joint and the two single data sets.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        the second input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param prior_chain: (optional) the prior chain.
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param prior_factor: (optional) factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :param feedback: logical flag to set whether the function should print
        a warning every time the explicit MAP file is not found.
        By default this is true.
    :return: :math:`Q_{\\rm DMAP}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm DMAP}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using the cdf method of :py:data:`scipy.stats.chi2` or
        :meth:`tensiometer.utilities.from_chi2_to_sigma`.
    """
    # check that all chains have the same running parameters:

    # get the data best fit for the chains:
    best_fit_data_like_1 = get_MAP_loglike(chain_1, feedback=feedback)
    best_fit_data_like_2 = get_MAP_loglike(chain_2, feedback=feedback)
    best_fit_data_like_12 = get_MAP_loglike(chain_12, feedback=feedback)
    # get the number of effective parameters:
    Neff_1 = get_Neff(chain_1,
                      prior_chain=prior_chain,
                      param_names=param_names,
                      prior_factor=prior_factor)
    Neff_2 = get_Neff(chain_2,
                      prior_chain=prior_chain,
                      param_names=param_names,
                      prior_factor=prior_factor)
    Neff_12 = get_Neff(chain_12,
                       prior_chain=prior_chain,
                       param_names=param_names,
                       prior_factor=prior_factor)
    # compute delta Neff:
    dofs = Neff_1 + Neff_2 - Neff_12
    # compute Q_DMAP:
    Q_DMAP = -2.*best_fit_data_like_12 \
        + 2.*best_fit_data_like_1 \
        + 2.*best_fit_data_like_2
    #
    return Q_DMAP, dofs
