"""
This file contains the functions and utilities to compute agreement and
disagreement between two different chains using a Gaussian approximation
for the posterior.

For more details on the method implemented see
`arxiv 1806.04649 <https://arxiv.org/pdf/1806.04649.pdf>`_
and `arxiv 1912.04880 <https://arxiv.org/pdf/1912.04880.pdf>`_.
"""

###############################################################################
# initial imports:

import scipy
import numpy as np
import copy
from getdist import MCSamples
from getdist.gaussian_mixtures import GaussianND
import matplotlib.pyplot as plt

from .utilities import stats_utilities as stutils

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


def _check_common_names(param_names_1, param_names_2):
    """
    Utility to get the common param names between two chains.
    """
    param_names = [name for name in param_names_1 if name in param_names_2]
    if len(param_names) == 0:
        raise ValueError('Chains do not have shared parameters.\n',
                         'Parameters for chain_1 ', param_names_1, '\n',
                         'Parameters for chain_2 ', param_names_2, '\n')
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

    .. math:: C_{ij} = \\delta_{ij} \\frac{( \\mathrm{max}(p_i) - \\mathrm{min}(p_i) )^2}{12}

    :param chain: the input chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: choice of parameter names to
        restrict the calculation.
    :type param_names: optional
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


def get_localized_covariance(chain_1, chain_2, param_names,
                             localize_params=None, scale=10.):
    """
    This function calculates the localized covariance matrix of `chain_1` using `chain_2` as a reference.
    The localization is performed by adjusting the weights of the samples in `chain_1` based on their
    likelihoods with respect to `chain_2`. The resulting covariance matrix is then computed by removing
    the localization covariance from the full covariance matrix of `chain_1`.

    :param chain_1: the first chain to be localized.
    :type chain_1: :class:`~getdist.mcsamples.MCSamples`
    :param chain_2: the second chain used for localization.
    :type chain_2: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: the names of the parameters.
    :param localize_params: the parameters to be localized. If not provided, all parameters will be localized.
    :type localize_params: optional
    :param scale: the scaling factor for the covariance matrix. Default is 10.
    :type scale: optional
    :return: The localized covariance matrix.
    """
    # initialize param names:
    param_names_1 = _check_param_names(chain_1, param_names)
    param_names_2 = _check_param_names(chain_2, param_names)
    param_names = _check_common_names(param_names_1, param_names_2)
    # check localized parameters:
    if localize_params is None:
        localize_params = param_names
    else:
        if not np.all([name in param_names for name in localize_params]):
            raise ValueError('Input localize_params is not in param_names')
    # get mean and covariance of the chain that we use for localization:
    mean = chain_2.getMeans(pars=[chain_2.index[name]
                                  for name in localize_params])
    cov = chain_2.cov(pars=localize_params)
    inv_cov = np.linalg.inv(scale**2*cov)
    sqrt_inv_cov = scipy.linalg.sqrtm(inv_cov)
    # get the Gaussian chi2:
    idx = [chain_1.index[name] for name in localize_params]
    X = np.dot(sqrt_inv_cov, (chain_1.samples[:, idx] - mean).T).T
    logLikes = (X*X).sum(axis=1)
    max_logLikes = np.amin(logLikes)
    # compute weights:
    new_weights = chain_1.weights * np.exp(-(logLikes - max_logLikes))
    # check that weights are reasonable:
    old_neff_samples = np.sum(chain_1.weights)**2 / np.sum(chain_1.weights**2)
    new_neff_samples = np.sum(new_weights)**2 / np.sum(new_weights**2)
    if old_neff_samples / new_neff_samples > 10.:
        print('WARNING: localization of covariance is resulting in too many '
              + 'samples being under-weighted.\n'
              + 'Neff original = ', round(old_neff_samples, 3), '\n'
              + 'Neff new      = ', round(new_neff_samples, 3), '\n'
              + 'this can result in large errors and can be improved with '
              + 'more samples in chain_1.')
    # compute covariance with all parameters:
    idx_full = [chain_1.index[name] for name in param_names]
    # compute covariance:
    cov2 = np.cov(chain_1.samples[:, idx_full].T, aweights=new_weights)
    # remove localization covariance:
    idx_rel = [param_names.index(name) for name in localize_params]
    inv_cov2 = np.linalg.inv(cov2)
    inv_cov2[np.ix_(idx_rel, idx_rel)] = inv_cov2[np.ix_(idx_rel, idx_rel)] \
        - inv_cov
    cov2 = np.linalg.inv(inv_cov2)
    #
    return cov2

###############################################################################


def get_Neff(chain, prior_chain=None, param_names=None,
             prior_factor=1.0, localize=False, **kwargs):
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

    :param chain: the input chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param prior_chain: the prior chain.
        If the prior is not well approximated by
        a ranged prior and is informative it is better to explicitly
        use a prior only chain.
        If this is not given the algorithm will assume ranged priors with the
        ranges computed from the input chain.
    :type prior_chain: :class:`~getdist.mcsamples.MCSamples` - optional
    :param param_names: parameter names to restrict the
        calculation of :math:`N_{\\rm eff}`.
        If none is given the default assumes that all running parameters
        should be used.
    :type param_names: optional
    :param prior_factor: factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :type prior_factor: optional
    :param localize: whether to localize the covariance.
    :type localize: optional
    :return: the number of effective parameters.
    """
    # initialize param names:
    param_names = _check_param_names(chain, param_names)
    # initialize prior covariance:
    if prior_chain is not None:
        # check parameter names:
        param_names = _check_param_names(prior_chain, param_names)
        # get the prior covariance:
        if localize:
            C_Pi = get_localized_covariance(prior_chain, chain,
                                            param_names, **kwargs)
        else:
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
    _Neff = _Ntot - np.real(np.sum(_eigv))
    #
    return _Neff

###############################################################################


def gaussian_approximation(chain, param_names=None, **kwargs):
    """
    Function that computes the Gaussian approximation of a given chain.

    :param chain: the input chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: parameter names to restrict the
        Gaussian approximation.
        If none is given the default assumes that all parameters
        should be used.
    :type param_names: optional
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
        label = 'Gaussian '+chain.label
    elif chain.name_tag is not None:
        label = 'Gaussian_'+chain.name_tag
    else:
        label = None
    # initialize the Gaussian distribution:
    gaussian_approx = GaussianND(mean, cov,
                                 names=param_names,
                                 labels=param_labels,
                                 label=label,
                                 **kwargs)
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
    :param prior_chain: the prior only chain.
        If the prior is not well approximated by a ranged prior and is
        informative it is better to explicitly use a prior only chain.
        If this is not given the algorithm will assume ranged priors
        with the ranges computed from the input chains.
    :type prior_chain: :class:`~getdist.mcsamples.MCSamples` - optional
    :param param_names: parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :type param_names: optional
    :param cutoff: the algorithms needs to detect prior
        constrained directions (that do not contribute to the test)
        from data constrained directions.
        This is achieved through a Karhunen–Loeve decomposition to avoid issues
        with physical dimensions of parameters and cutoff sets the minimum
        improvement with respect to the prior that is used.
        Default is five percent.
    :type cutoff: optional
    :param prior_factor: factor to scale the prior covariance.
        In case of strongly non-Gaussian posteriors it might be useful to
        artificially tighten the prior to have less noise in telling apart
        parameter space directions that are constrained by data and prior.
        Default is no scaling, prior_factor=1.
    :type prior_factor: optional
    :return: :math:`Q_{\\rm DM}` value and number of degrees of freedom.
        Since :math:`Q_{\\rm DM}` is :math:`\\chi^2` distributed the
        probability to exceed the test can be computed
        using the cdf method of :py:data:`scipy.stats.chi2` or
        :meth:`tensiometer.utilities.stats_utilities.from_chi2_to_sigma`.
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
    C_Pi_inv = stutils.QR_inverse(C_Pi)
    temp = np.dot(np.dot(C_p1, C_Pi_inv), C_p2)
    diff_covariance = C_p1 + C_p2 - temp - temp.T
    # take the directions that are best constrained over the prior:
    eig_1, eigv_1 = stutils.KL_decomposition(C_p1, C_Pi)
    eig_2, eigv_2 = stutils.KL_decomposition(C_p2, C_Pi)
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
    Q_DM = np.dot(np.dot(param_diff, stutils.QR_inverse(temp_cov)), param_diff)
    #
    return Q_DM, dofs

###############################################################################


def linear_CPCA(fisher_1, fisher_12, param_names,
                conditional_params=[], marginalized_parameters=[],
                normparam=None, dimensional_reduce=True,
                dimensional_threshold=0.1):
    """
    Performs the CPCA analysis of two Fisher matrices.
    As discussed in (`Dacunha et al. 22 <https://arxiv.org/pdf/1806.04649.pdf>`_)
    this quantifies the modes that the joint chain improves over the single one.
    Note this is a lower-level function that does not require GetDist chains.

    :param fisher_1: first input Fisher matrix.
    :type fisher_1: numpy.ndarray
    :param fisher_12: second input Fisher matrix.
    :type fisher_12: numpy.ndarray
    :param param_names: list of names of the parameters to use
    :type param_names: list[str]
    :param conditional_params: (optional) list of parameters to treat as fixed,
        i.e. for KL_PCA conditional on fixed values of these parameters
    :type conditional_params: list[str]
    :param param_map: (optional) a transformation to apply to parameter values;
        A list or string containing either N (no transformation)
        or L (for log transform) or M (for minus log transform of negative
        parameters) for each parameter.
        By default uses log if no parameter values cross zero.
        The transformed parameters are added to the joint chain.
    :type param_map: Union[str, List[str]]
    :param normparam: (optional) name of parameter to normalize result
        (i.e. this parameter will have unit power)
        By default scales to the parameter that has the largest impactr on the KL mode variance.
    :type normparam: str
    :param dimensional_reduce: (optional) perform dimensional reduction of the KL modes considered
        keeping only parameters with a large impact on KL mode variances.
        Default is True.
    :type dimensional_reduce: bool
    :param dimensional_threshold: (optional) threshold for dimensional reducetion.
        Default is 10% so that parameters with a contribution less than 10% of KL mode
        variance are neglected from a specific KL mode.
    :type dimensional_threshold: float
    :param verbose: (optional) chatty output. Default True.
    :type verbose: bool
    :return: dictionary containing the results of the CPCA analysis
    :rtype: dict
    """
    # initialize param names:
    num_params = len(param_names)
    if num_params != fisher_1.shape[0]:
        raise ValueError('Input fisher matrix 1 has size', fisher_1.shape[0], '\n',
                         ' while param_names has length', num_params)
    if num_params != fisher_12.shape[0]:
        raise ValueError('Input fisher matrix 12 has size', fisher_12.shape[0], '\n',
                         ' while param_names has length', num_params)
    # test validity of conditional parameters:
    if len(conditional_params) > 0:
        if not np.all([name in param_names for name in conditional_params]):
            raise ValueError('Input conditional_params:', conditional_params, '\n',
                             'are not all in param_names:', param_names)
    # test validity of marginalized parameters:
    if len(marginalized_parameters) > 0:
        if not np.all([name in param_names for name in marginalized_parameters]):
            raise ValueError('Input marginalized_parameters:', marginalized_parameters, '\n',
                             'are not all in param_names:', param_names)
    # fix parameters in Fisher matrices:
    keep_indexes = [param_names.index(name) for name in param_names if name not in conditional_params]
    param_names_to_use = [param_names[ind] for ind in keep_indexes]
    F_p1 = fisher_1[:, keep_indexes][keep_indexes, :]
    F_p12 = fisher_12[:, keep_indexes][keep_indexes, :]
    # marginalize Fisher over parameters:
    if len(marginalized_parameters) > 0:
        keep_indexes = [param_names_to_use.index(name) for name in param_names_to_use if name not in marginalized_parameters]
        param_names_to_use = [param_names_to_use[ind] for ind in keep_indexes]
        F_p1 = np.linalg.inv(np.linalg.inv(F_p1)[:, keep_indexes][keep_indexes, :])
        F_p12 = np.linalg.inv(np.linalg.inv(F_p12)[:, keep_indexes][keep_indexes, :])
    sqrt_F_p12 = scipy.linalg.sqrtm(F_p12)
    # recompute number of parameters:
    num_params = len(param_names_to_use)
    # initialize internal variables:
    if normparam is not None:
        normparam = param_names.index(normparam)
    # perform the CPCA decomposition:
    CPC_eig, CPC_eigv = stutils.KL_decomposition(F_p12, F_p1)
    # sort in decreasing order (best mode first):
    idx = np.argsort(CPC_eig)[::-1]
    CPC_eig, CPC_eigv = CPC_eig[idx], CPC_eigv[:, idx]
    # compute Neff from KL decomposition:
    _temp = CPC_eig.copy()
    _temp[_temp < 1.] = 1.
    Neff_spectrum = 1.-1./_temp
    Neff = np.sum(Neff_spectrum)
    # compute KL divergence:
    KL_spectrum = 0.5 / np.log(2.) * (np.log(_temp) + 2./_temp - 1.)
    KL_divergence = np.sum(KL_spectrum)
    # compute contributions:
    temp = np.dot(sqrt_F_p12, CPC_eigv)
    contributions = temp * temp / CPC_eig
    # compute the dimensional reduction matrix:
    if dimensional_reduce:
        reduction_filter = contributions > dimensional_threshold
    else:
        reduction_filter = np.ones((num_params, num_params), dtype=bool)
    if normparam is not None:
        reduction_filter[normparam, :] = True
    # compute projector:
    projector = np.linalg.inv(CPC_eigv).copy()
    projector[np.logical_not(reduction_filter.T)] = 0
    # prepare return of the function:
    results_dict = {}
    # mode results:
    results_dict['CPCA_eig'] = CPC_eig.copy()
    results_dict['CPCA_eigv'] = CPC_eigv.copy()
    results_dict['CPCA_var_contributions'] = contributions.copy()
    results_dict['CPCA_var_filter'] = reduction_filter.copy()
    results_dict['CPCA_projector'] = projector
    # Neff results:
    results_dict['Neff'] = Neff
    results_dict['Neff_spectrum'] = Neff_spectrum
    # KL divergence:
    results_dict['KL_divergence'] = KL_divergence
    results_dict['KL_spectrum'] = KL_spectrum
    # parameter names:
    results_dict['param_names'] = param_names_to_use
    results_dict['conditional_params'] = conditional_params
    results_dict['marginalized_parameters'] = marginalized_parameters
    results_dict['normparam'] = normparam
    #
    return results_dict


def linear_CPCA_chains(chain_1, chain_12, param_names, **kwargs):
    """
    Performs the CPCA analysis of two chains.
    As discussed in (`Dacunha et al. 22 <https://arxiv.org/pdf/1806.04649.pdf>`_)
    this quantifies the modes that the joint chain improves over the single one.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples` the reference input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples` the joint input chain.
    :param param_names: parameter names to use in the calculation. Defaults to all
        running parameters.
    """
    # test if chains:
    _check_chain_type(chain_1)
    _check_chain_type(chain_12)
    # initialize param names:
    param_names_1 = _check_param_names(chain_1, param_names)
    param_names_12 = _check_param_names(chain_12, param_names)
    param_names = _check_common_names(param_names_1, param_names_12)
    # get Fisher from the chains:
    fisher_1 = np.linalg.inv(chain_1.cov(param_names))
    fisher_12 = np.linalg.inv(chain_12.cov(param_names))
    # do CPCA with Fishers:
    CPCA_results = linear_CPCA(fisher_1, fisher_12, param_names,
                               **stutils.filter_kwargs(kwargs, linear_CPCA)
                               )
    param_names = CPCA_results['param_names']
    # add mean and parameter labels to results:
    mean = chain_12.getMeans(pars=[chain_12.index[name] for name in param_names])
    CPCA_results['reference_point'] = mean.copy()
    labels = [chain_12.parLabel(chain_12.index[name]) for name in param_names]
    CPCA_results['param_labels'] = copy.deepcopy(labels)
    # compute correlation of CPCA modes with parameters:
    idx_to_use = [chain_12.index[name] for name in param_names]
    proj_samples = np.dot(CPCA_results['CPCA_projector'], (chain_12.samples[:, idx_to_use]-mean).T)
    proj_cov = np.cov(np.vstack((proj_samples, chain_12.samples.T)),
                      aweights=chain_12.weights)
    temp = np.diag(1./np.sqrt(np.diag(proj_cov)))
    proj_corr = np.dot(np.dot(temp, proj_cov), temp)[:len(param_names), :][:, len(param_names):]
    # save correlation in results:
    CPCA_results['correlation_mode_parameter'] = proj_corr.copy()
    CPCA_results['correlation_parameter_names'] = chain_12.getParamNames().list().copy()
    #
    return CPCA_results


def print_CPCA_results(CPCA_results, verbose=True, num_modes=None):
    """
    Print the results of the CPCA analysis.

    :param CPCA_results: The results of the CPCA analysis.
    :type CPCA_results: dict
    :param verbose: Whether to print additional information. Defaults to True.
    :type verbose: bool, optional
    :param num_modes: The number of modes to consider. Defaults to None.
    :type num_modes: int, optional
    :return: The formatted text containing the CPCA results.
    :rtype: str
    """
    PCAtext = ''
    # initialize parameters:
    param_names = CPCA_results['param_names']
    num_params = len(param_names)
    if num_modes is None:
        num_modes = num_params
    PCAtext += 'CPCA for '+str(num_params)+' parameters:\n'
    if verbose:
        PCAtext += '\n'
    # parameter names and labels:
    if 'param_labels' in CPCA_results.keys():
        labels = CPCA_results['param_labels']
    else:
        labels = CPCA_results['param_names']
    if verbose:
        for i in range(num_params):
            PCAtext += "%4s : %s\n" % (str(i + 1), labels[param_names.index(param_names[i])])
        PCAtext += '\n'
    # conditional parameters:
    if len(CPCA_results['conditional_params']) > 0:
        PCAtext += '  with '+str(len(CPCA_results['conditional_params']))+' fixed parameters:\n'
        if verbose:
            PCAtext += '\n'
            for i in range(len(CPCA_results['conditional_params'])):
                PCAtext += "%4s : %s\n" % (str(i + 1), CPCA_results['conditional_params'][i])
            PCAtext += '\n'
    # marginalized parameters:
    if len(CPCA_results['marginalized_parameters']) > 0:
        PCAtext += '  with '+str(len(CPCA_results['marginalized_parameters']))+' marginalized parameters:\n\n'
        if verbose:
            for i in range(len(CPCA_results['marginalized_parameters'])):
                PCAtext += "%4s : %s\n" % (str(i + 1), CPCA_results['marginalized_parameters'][i])
            PCAtext += '\n'
    # write Neff analysis:
    PCAtext += 'Neff = %5.2f\n' % (CPCA_results['Neff'])
    PCAtext += 'Neff mode = ['
    for temp in CPCA_results['Neff_spectrum']:
        PCAtext += '%5.2f' % (temp)
    PCAtext += ']\n\n'
    # write KL divergence:
    PCAtext += '<KL> divergence = %5.2f\n' % (CPCA_results['KL_divergence'])
    PCAtext += '<KL> mode = ['
    for temp in CPCA_results['KL_spectrum']:
        PCAtext += '%5.2f' % (temp)
    PCAtext += ']\n\n'
    # write out eigenvalues:
    PCAtext += 'CPC amplitudes - 1 (variance improvement per mode):\n'
    for i in range(num_modes):
        PCAtext += '%4s : %8.4f' % (str(i + 1), CPCA_results['CPCA_eig'][i]-1.)
        if CPCA_results['CPCA_eig'][i]-1. > 0.:
            PCAtext += ' (%8.1f %%)' % (np.sqrt(CPCA_results['CPCA_eig'][i]-1.)*100.)
        else:
            PCAtext += ' (     noisy)'
        PCAtext += '\n'
    # write out CPC eigenvectors:
    if verbose:
        PCAtext += '\n'
        PCAtext += 'CPC modes:\n'
        for i in range(num_modes):
            if CPCA_results['CPCA_eig'][i]-1. > 0.:
                PCAtext += '%4s :' % str(i + 1)
                for j in range(num_params):
                    PCAtext += '%8.3f' % (CPCA_results['CPCA_eigv'][j, i])
                PCAtext += '\n'
    # write out parameter contributions to KL mode variance:
    PCAtext += '\nParameter contribution to CPC-mode variance:\n'
    max_length = np.amax([len(name) for name in param_names + ['mode number']])
    PCAtext += f" {'mode number':<{max_length}} :"
    for i in range(num_modes):
        if CPCA_results['CPCA_eig'][i]-1. > 0.:
            PCAtext += '%8i' % (i+1)
    PCAtext += '\n'
    for i in range(num_params):  # loop over parameters
        PCAtext += f" {param_names[i]:<{max_length}} :"
        for j in range(num_modes):  # loop over modes
            if CPCA_results['CPCA_eig'][j]-1. > 0.:
                PCAtext += '%8.3f' % (CPCA_results['CPCA_var_contributions'][i, j])
        PCAtext += '\n'
    # write out CPC components constraints:
    PCAtext += '\nCPC parameter combinations:\n'
    for i in range(num_modes):  # loop over modes
        if CPCA_results['CPCA_eig'][i]-1. > 0.:
            # summary of mode improvement:
            summary = '%4s : %8.3f' % (str(i + 1), CPCA_results['CPCA_eig'][i]-1.)
            if CPCA_results['CPCA_eig'][i]-1. > 0.:
                summary += ' (%8.1f %%)' % (np.sqrt(CPCA_results['CPCA_eig'][i]-1.)*100.)
            summary += '\n'
            # compute normalization of mode:
            if CPCA_results['normparam'] is not None:
                norm = CPCA_results['CPCA_projector'][i, CPCA_results['normparam']]
            else:
                norm = CPCA_results['CPCA_projector'][i, np.argmax(CPCA_results['CPCA_var_contributions'][:, i])]
            # write the mode:
            string = ''
            for j in range(num_params):  # loop over parameters
                if CPCA_results['CPCA_var_filter'][j, i]:
                    # get normalized coefficient:
                    _norm_eigv = CPCA_results['CPCA_projector'][i, j] / norm
                    # format it to string and save it:
                    _temp = '{0:+.2f}'.format(np.round(_norm_eigv, 2))
                    if 'reference_point' in CPCA_results.keys():
                        _mean = CPCA_results['reference_point'][j]
                        _mean = '{0:+}'.format(np.round(-_mean, 2))
                        string += _temp+'*('+labels[j]+' '+_mean+') '
                    else:
                        string += _temp+'*('+labels[j]+') '
            summary += '      '+string+'= 0 +- '+'%.2g (post) / %.2g (prior)' % (np.sqrt(1./CPCA_results['CPCA_eig'][i]) / np.abs(norm), 1. / np.abs(norm))
            summary += '\n'
            PCAtext += summary
    # write out CPC correlation with parameters (if present):
    if 'correlation_mode_parameter' in CPCA_results.keys():
        # compute maximum name length:
        max_length = np.amax([len(name) for name in CPCA_results['correlation_parameter_names']])
        # write first line:
        PCAtext += '\nCPC modes parameters correlations:\n'
        PCAtext += f" {'mode number':<{max_length}} :"
        for i in range(num_modes):
            if CPCA_results['CPCA_eig'][i]-1. > 0.:
                PCAtext += '%8i' % (i+1)
        PCAtext += '\n'
        # then write out parameter per parameter:
        for i, name in enumerate(CPCA_results['correlation_parameter_names']):
            PCAtext += f" {name:<{max_length}} :"
            for j in range(num_modes):  # loop over modes
                if CPCA_results['CPCA_eig'][j]-1. > 0.:
                    PCAtext += '%8.3f' % (CPCA_results['correlation_mode_parameter'][j, i])
            PCAtext += '\n'
    #
    return PCAtext

###############################################################################


def Q_UDM_KL_components(chain_1, chain_12, param_names=None):
    """
    Function that computes the Karhunen–Loeve (KL) decomposition of the
    covariance of a chain with the covariance of that chain joint with another
    one.
    This function is used for the parameter shift algorithm in
    update form.

    :param chain_1: the first input chain.
    :type chain_1: :class:`~getdist.mcsamples.MCSamples`
    :param chain_12: the joint input chain.
    :type chain_12: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :type param_names: list, optional
    :return: the KL eigenvalues, the KL eigenvectors and the parameter names
        that are used, sorted in decreasing order.
    :rtype: tuple
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
    KL_eig, KL_eigv = stutils.KL_decomposition(C_p1, C_p12)
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
        and fractional Fisher matrix.
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
    return param_names, KL_eig, fractional_fisher, fisher


def Q_UDM_covariance_components(chain_1, chain_12, param_names=None,
                                which='1'):
    """
    Compute the decomposition of the covariance matrix in terms of KL modes.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        the first input chain.
    :param chain_12: :class:`~getdist.mcsamples.MCSamples`
        the joint input chain.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param which: (optional) which decomposition to report. Possibilities are
        '1' for the chain 1 covariance matrix, '2' for the chain 2 covariance
        matrix and '12' for the joint covariance matrix.
    :return: parameter names used in the calculation, values of improvement,
        fractional covariance matrix and covariance matrix
        (inverse covariance).
    """
    KL_eig, KL_eigv, param_names = Q_UDM_KL_components(chain_1,
                                                       chain_12,
                                                       param_names=param_names)
    # inverse KL components:
    KL_eigv = stutils.QR_inverse(KL_eigv)
    # compute covariance and fractional covariance matrix:
    if which == '1':
        diag_cov = np.sum(KL_eigv*KL_eigv*KL_eig, axis=1)
        fractional_cov = ((KL_eigv*KL_eigv*KL_eig).T/diag_cov).T
    elif which == '2':
        diag_cov = np.sum(KL_eigv*KL_eigv*KL_eig/(KL_eig-1.), axis=1)
        fractional_cov = ((KL_eigv*KL_eigv*KL_eig/(KL_eig-1.)).T/diag_cov).T
    elif which == '12':
        diag_cov = np.sum(KL_eigv*KL_eigv, axis=1)
        fractional_cov = ((KL_eigv*KL_eigv).T/diag_cov).T
    else:
        raise ValueError('Input parameter which can only be: 1, 2, 12.')
    #
    return param_names, KL_eig, fractional_cov

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
        :meth:`tensiometer.utilities.stats_utilities.from_chi2_to_sigma`.
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
        :meth:`tensiometer.utilities.stats_utilities.from_chi2_to_sigma`.
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
        :meth:`tensiometer.utilities.stats_utilities.from_chi2_to_sigma`.
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
