"""
This file contains the functions and utilities to compute agreement and
disagreement between two different chains with Monte Carlo methods.

For more details on the method implemented see
`arxiv 1806.04649 <https://arxiv.org/pdf/1806.04649.pdf>`_
and `arxiv 1912.04880 <https://arxiv.org/pdf/1912.04880.pdf>`_.
"""

"""
For test purposes:

from getdist import loadMCSamples, MCSamples, WeightedSamples
chain_1 = loadMCSamples('./test_chains/DES')
chain_2 = loadMCSamples('./test_chains/Planck18TTTEEE')
chain_12 = loadMCSamples('./test_chains/Planck18TTTEEE_DES')
chain_prior = loadMCSamples('./test_chains/prior')

import tensiometer.utilities as utils
import matplotlib.pyplot as plt

diff_chain = parameter_diff_chain(chain_1, chain_2, boost=1)
param_names = None
scale = None
method = 'brute_force'
feedback=2
n_threads = 1
"""

###############################################################################
# initial imports:

import os
import time
import gc
import numpy as np
import getdist.chains as gchains
gchains.print_load_details = False
from getdist import MCSamples, WeightedSamples
from scipy.linalg import sqrtm
from scipy.integrate import simps

from . import utilities as utils

# imports for parallel calculations:
import multiprocessing
import joblib
# number of threads available:
if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = multiprocessing.cpu_count()

###############################################################################
# Parameter difference chain:


def parameter_diff_weighted_samples(samples_1, samples_2, boost=1,
                                    indexes_1=None, indexes_2=None):
    """
    Compute the parameter differences of two input weighted samples.
    The parameters of the difference samples are related to the
    parameters of the input samples, :math:`\\theta_1` and
    :math:`\\theta_2` by:

    .. math:: \\Delta \\theta \\equiv \\theta_1 - \\theta_2

    This function does not assume Gaussianity of the chain.
    This functions does assume that the parameter determinations from the two
    chains (i.e. the underlying data sets) are uncorrelated.
    Do not use this function for chains that are correlated.

    :param samples_1: :class:`~getdist.chains.WeightedSamples`
        first input weighted samples with :math:`n_1` samples.
    :param samples_2: :class:`~getdist.chains.WeightedSamples`
        second input weighted samples with :math:`n_2` samples.
    :param boost: (optional) boost the number of samples in the
        difference. By default the length of the difference samples
        will be the length of the longest one.
        Given two samples the full difference samples can contain
        :math:`n_1\\times n_2` samples but this is usually prohibitive
        for realistic chains.
        The boost parameters wil increase the number of samples to be
        :math:`{\\rm boost}\\times {\\rm max}(n_1,n_2)`.
        Default boost parameter is one.
        If boost is None the full difference chain is going to be computed
        (and will likely require a lot of memory and time).
    :param indexes_1: (optional) array with the indexes of the parameters to
        use for the first samples. By default this tries to use all
        parameters.
    :param indexes_2: (optional) array with the indexes of the parameters to
        use for the second samples. By default this tries to use all
        parameters.
    :return: :class:`~getdist.chains.WeightedSamples` the instance with the
        parameter difference samples.
    """
    # test for type, this function assumes that we are working with MCSamples:
    if not isinstance(samples_1, WeightedSamples):
        raise TypeError('Input samples_1 is not of WeightedSamples type.')
    if not isinstance(samples_2, WeightedSamples):
        raise TypeError('Input samples_2 is not of WeightedSamples type.')
    # get indexes:
    if indexes_1 is None:
        indexes_1 = np.arange(samples_1.samples.shape[1])
    if indexes_2 is None:
        indexes_2 = np.arange(samples_2.samples.shape[1])
    # check:
    if not len(indexes_1) == len(indexes_2):
        raise ValueError('The samples do not containt the same number',
                         'of parameters.')
    num_params = len(indexes_1)
    # order the chains so that the second chain is always with less points:
    if (len(samples_1.weights) >= len(samples_2.weights)):
        ch1, ch2 = samples_1, samples_2
        sign = +1.
        ind1, ind2 = indexes_1, indexes_2
    else:
        ch1, ch2 = samples_2, samples_1
        sign = -1.
        ind1, ind2 = indexes_2, indexes_1
    # get number of samples:
    num_samps_1 = len(ch1.weights)
    num_samps_2 = len(ch2.weights)
    if boost is None:
        sample_boost = num_samps_2
    else:
        sample_boost = min(boost, num_samps_2)
    # create the arrays (these might be big depending on boost level...):
    weights = np.empty((num_samps_1*sample_boost))
    difference_samples = np.empty((num_samps_1*sample_boost, num_params))
    if ch1.loglikes is not None and ch2.loglikes is not None:
        loglikes = np.empty((num_samps_1*sample_boost))
    else:
        loglikes = None
    # compute the samples:
    for ind in range(sample_boost):
        base_ind = int(float(ind)/float(sample_boost)*num_samps_2)
        _indexes = range(base_ind, base_ind+num_samps_1)
        # compute weights (as the product of the weights):
        weights[ind*num_samps_1:(ind+1)*num_samps_1] = \
            ch1.weights*np.take(ch2.weights, _indexes, mode='wrap')
        # compute the likelihood difference:
        if ch1.loglikes is not None and ch2.loglikes is not None:
            loglikes[ind*num_samps_1:(ind+1)*num_samps_1] = \
                ch1.loglikes-np.take(ch2.loglikes, _indexes, mode='wrap')
        # compute the difference samples:
        difference_samples[ind*num_samps_1:(ind+1)*num_samps_1, :] = \
            ch1.samples[:, ind1] \
            - np.take(ch2.samples[:, ind2], _indexes, axis=0, mode='wrap')
    # get additional informations:
    if samples_1.name_tag is not None and samples_2.name_tag is not None:
        name_tag = samples_1.name_tag+'_diff_'+samples_2.name_tag
    else:
        name_tag = None
    if samples_1.label is not None and samples_2.label is not None:
        label = samples_1.label+' diff '+samples_2.label
    else:
        label = None
    if samples_1.min_weight_ratio is not None and \
       samples_2.min_weight_ratio is not None:
        min_weight_ratio = min(samples_1.min_weight_ratio,
                               samples_2.min_weight_ratio)
    # initialize the weighted samples:
    diff_samples = WeightedSamples(ignore_rows=0,
                                   samples=sign*difference_samples,
                                   weights=weights, loglikes=loglikes,
                                   name_tag=name_tag, label=label,
                                   min_weight_ratio=min_weight_ratio)
    #
    return diff_samples

###############################################################################


def parameter_diff_chain(chain_1, chain_2, boost=1):
    """
    Compute the chain of the parameter differences between the two input
    chains. The parameters of the difference chain are related to the
    parameters of the input chains, :math:`\\theta_1` and :math:`\\theta_2` by:

    .. math:: \\Delta \\theta \\equiv \\theta_1 - \\theta_2

    This function only returns the differences for the parameters that are
    common to both chains.
    This function preserves the chain separation (if any) so that the
    convergence of the difference chain can be tested.
    This function does not assume Gaussianity of the chain.
    This functions does assume that the parameter determinations from the two
    chains (i.e. the underlying data sets) are uncorrelated.
    Do not use this function for chains that are correlated.

    :param chain_1: :class:`~getdist.mcsamples.MCSamples`
        first input chain with :math:`n_1` samples
    :param chain_2: :class:`~getdist.mcsamples.MCSamples`
        second input chain with :math:`n_2` samples
    :param boost: (optional) boost the number of samples in the
        difference chain. By default the length of the difference chain
        will be the length of the longest chain.
        Given two chains the full difference chain can contain
        :math:`n_1\\times n_2` samples but this is usually prohibitive
        for realistic chains.
        The boost parameters wil increase the number of samples to be
        :math:`{\\rm boost}\\times {\\rm max}(n_1,n_2)`.
        Default boost parameter is one.
        If boost is None the full difference chain is going to be computed
        (and will likely require a lot of memory and time).
    :return: :class:`~getdist.mcsamples.MCSamples` the instance with the
        parameter difference chain.
    """
    # check input:
    if boost is not None:
        if boost < 1:
            raise ValueError('Minimum boost is 1\n Input value is ', boost)
    # test for type, this function assumes that we are working with MCSamples:
    if not isinstance(chain_1, MCSamples):
        raise TypeError('Input chain_1 is not of MCSamples type.')
    if not isinstance(chain_2, MCSamples):
        raise TypeError('Input chain_2 is not of MCSamples type.')
    # get the parameter names:
    param_names_1 = chain_1.getParamNames().list()
    param_names_2 = chain_2.getParamNames().list()
    # get the common names:
    param_names = [_p for _p in param_names_1 if _p in param_names_2]
    num_params = len(param_names)
    if num_params == 0:
        raise ValueError('There are no shared parameters to difference')
    # get the names and labels:
    diff_param_names = ['delta_'+name for name in param_names]
    diff_param_labels = ['\\Delta '+name.label for name in
                         chain_1.getParamNames().parsWithNames(param_names)]
    # get parameter indexes:
    indexes_1 = [chain_1.index[name] for name in param_names]
    indexes_2 = [chain_2.index[name] for name in param_names]
    # get separate chains:
    if not hasattr(chain_1, 'chain_offsets'):
        _chains_1 = [chain_1]
    else:
        _chains_1 = chain_1.getSeparateChains()
    if not hasattr(chain_2, 'chain_offsets'):
        _chains_2 = [chain_2]
    else:
        _chains_2 = chain_2.getSeparateChains()
    # set the boost:
    if chain_1.sampler == 'nested' \
       or chain_2.sampler == 'nested' or boost is None:
        chain_boost = max(len(_chains_1), len(_chains_2))
        sample_boost = None
    else:
        chain_boost = min(boost, max(len(_chains_1), len(_chains_2)))
        sample_boost = boost
    # get the combinations:
    if len(_chains_1) > len(_chains_2):
        temp_ind = np.indices((len(_chains_2), len(_chains_1)))
    else:
        temp_ind = np.indices((len(_chains_1), len(_chains_2)))
    ind1 = np.concatenate([np.diagonal(temp_ind, offset=i, axis1=1, axis2=2)[0]
                           for i in range(chain_boost)])
    ind2 = np.concatenate([np.diagonal(temp_ind, offset=i, axis1=1, axis2=2)[1]
                           for i in range(chain_boost)])
    chains_combinations = [[_chains_1[i], _chains_2[j]]
                           for i, j in zip(ind1, ind2)]
    # compute the parameter difference samples:
    diff_chain_samples = [parameter_diff_weighted_samples(samp1,
                          samp2, boost=sample_boost, indexes_1=indexes_1,
                          indexes_2=indexes_2) for samp1, samp2
                          in chains_combinations]
    # create the samples:
    diff_samples = MCSamples(names=diff_param_names, labels=diff_param_labels)
    diff_samples.chains = diff_chain_samples
    diff_samples.makeSingle()
    # get the ranges:
    _ranges = {}
    for name, _min, _max in zip(diff_param_names,
                                np.amin(diff_samples.samples, axis=0),
                                np.amax(diff_samples.samples, axis=0)):
        _ranges[name] = [_min, _max]
    diff_samples.setRanges(_ranges)
    # initialize other things:
    if chain_1.name_tag is not None and chain_2.name_tag is not None:
        diff_samples.name_tag = chain_1.name_tag+'_diff_'+chain_2.name_tag
    # set distinction between base and derived parameters:
    _temp = diff_samples.getParamNames().list()
    _temp_paramnames = chain_1.getParamNames()
    for _nam in diff_samples.getParamNames().parsWithNames(_temp):
        _temp_name = _nam.name.replace('delta_', '')
        _nam.isDerived = _temp_paramnames.parWithName(_temp_name).isDerived
    # update and compute everything:
    diff_samples.updateBaseStatistics()
    diff_samples.deleteFixedParams()
    #
    return diff_samples

###############################################################################
# KDE bandwidth selection:


def Scotts_RT(num_params, num_samples):
    """
    Compute Scott's rule of thumb bandwidth covariance scaling.
    This is the default scaling that is used to compute the KDE estimate of
    parameter shifts.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :return: Scott's scaling.
    """
    return num_samples**(-2./(num_params+4.))


def Silvermans_RT(num_params, num_samples):
    """
    Compute Silverman's rule of thumb bandwidth covariance scaling.
    This is the default scaling that is used to compute the KDE estimate of
    parameter shifts.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :return: Silverman's scaling.
    """
    return (num_samples * (num_params + 2.) / 4.)**(-2. / (num_params + 4.))


def OptimizeBandwidth_1D(diff_chain, param_names=None, num_bins=1000):
    """
    Compute an estimate of an optimal bandwidth for covariance scaling as in
    GetDist. This is performed on whitened samples (with identity covariance),
    in 1D, and then scaled up with a dimensionality correction.

    :param diff_chain:
    :param param_names:
    :param num_bins: number of bins used for the 1D estimate
    :return: scaling vector for the whitened parameters
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # some initial calculations:
    _samples_cov = diff_chain.cov(pars=param_names)
    _num_params = len(ind)
    # whiten the samples:
    _temp = sqrtm(utils.QR_inverse(_samples_cov))
    white_samples = diff_chain.samples[:, ind].dot(_temp)
    # make these samples so that we can use GetDist band optization:
    temp_samples = MCSamples(samples=white_samples,
                             weights=diff_chain.weights,
                             ignore_rows=0, sampler=diff_chain.sampler)
    # now get optimal band for each parameter:
    bands = []
    for i in range(_num_params):
        # get the parameter:
        par = temp_samples._initParamRanges(i, paramConfid=None)
        # get the bins:
        temp_result = temp_samples._binSamples(temp_samples.samples[:, i],
                                               par, num_bins)
        bin_indices, bin_width, binmin, binmax = temp_result
        bins = np.bincount(bin_indices, weights=temp_samples.weights,
                           minlength=num_bins)
        # get the optimal smoothing scale:
        N_eff = temp_samples._get1DNeff(par, i)
        band = temp_samples.getAutoBandwidth1D(bins, par, i, N_eff=N_eff,
                                               mult_bias_correction_order=0,
                                               kernel_order=0) \
            * (binmax - binmin)
        # correction for dimensionality:
        dim_factor = Silvermans_RT(_num_params, N_eff)/Silvermans_RT(1., N_eff)
        dim_factor = Scotts_RT(_num_params, N_eff)/Scotts_RT(1., N_eff)
        #
        bands.append(band**2.*dim_factor)
    #
    return np.array(bands)

###############################################################################
# Parameter difference integrals:


def _gauss_kde_pdf(x, samples, weights):
    """
    Utility function to compute the Gaussian log KDE probability at x from
    already whitened samples, possibly with weights.
    Normalization constants are ignored.
    """
    X = x-samples
    return np.log(np.dot(weights, np.exp(-0.5*(X*X).sum(axis=1))))


def _epa_kde_pdf(x, samples, weights):
    """
    Utility function to compute the Epanechnikov log KDE probability at x from
    already whitened samples, possibly with weights.
    Normalization constants are ignored.
    """
    X = x-samples
    X2 = (X*X).sum(axis=1)
    return np.log(np.dot(weights[X2 < 1.], 1.-X2[X2 < 1.]))


def _brute_force_parameter_shift(white_samples, weights, zero_prob,
                                 num_samples, feedback, kernel='Gaussian'):
    """
    Brute force parallelized algorithm for parameter shift.
    """
    # get feedback:
    if feedback > 1:
        from tqdm import tqdm
        def feedback_helper(x): return tqdm(x, ascii=True)
    else:
        def feedback_helper(x): return x
    # select kernel:
    if kernel is None or kernel == 'Gaussian':
        log_kernel = _gauss_kde_pdf
    elif kernel == 'Epa':
        log_kernel = _epa_kde_pdf
    else:
        raise ValueError('Unknown kernel given')
    # run:
    with joblib.Parallel(n_jobs=n_threads) as parallel:
        _kde_eval_pdf = parallel(joblib.delayed(log_kernel)
                                 (samp, white_samples, weights)
                                 for samp in feedback_helper(white_samples))
    # filter for probability calculation:
    _filter = _kde_eval_pdf > zero_prob
    # compute number of filtered elements:
    _num_filtered = np.sum(weights[_filter])
    #
    return _num_filtered


def _nearest_parameter_shift(white_samples, weights, zero_prob, num_samples,
                             feedback, **kwargs):
    # import specific for this function:
    from scipy.spatial import cKDTree
    if feedback > 1:
        from tqdm import tqdm
        def feedback_helper(x): return tqdm(x, ascii=True)
    else:
        def feedback_helper(x): return x
    # get options:
    stable_cycle = kwargs.get('stable_cycle', 4)
    chunk_size = kwargs.get('chunk_size', 40)
    smallest_improvement = kwargs.get('smallest_improvement', 1.e-4)
    # the tree elimination has to work with probabilities to go incremental:
    _zero_prob = np.exp(zero_prob)
    # build tree:
    if feedback > 0:
        print('Building KD-Tree')
    data_tree = cKDTree(white_samples, leafsize=chunk_size,
                        balanced_tree=True)
    # make sure that the weights are floats:
    _weights = weights.astype(np.float)
    # initialize the calculation to zero:
    _num_elements = len(_weights)
    _kde_eval_pdf = np.zeros(_num_elements)
    _filter = np.ones(_num_elements, dtype=bool)
    _last_n = 0
    _stable_cycle = 0
    # loop over the neighbours:
    if feedback > 0:
        print('Neighbours elimination')
    for i in range(_num_elements//chunk_size):
        ind_min = chunk_size*i
        ind_max = chunk_size*i+chunk_size
        _dist, _ind = data_tree.query(white_samples[_filter],
                                      ind_max, n_jobs=-1)
        _kde_eval_pdf[_filter] += np.sum(
            _weights[_ind[:, ind_min:ind_max]]
            * np.exp(-0.5*np.square(_dist[:, ind_min:ind_max])), axis=1)
        _filter[_filter] = _kde_eval_pdf[_filter] < _zero_prob
        _num_filtered = np.sum(_filter)
        if feedback > 1:
            print('nearest_elimination: chunk', i+1)
            print('    surviving elements', _num_filtered,
                  'of', _num_elements)
        # check if calculation has converged:
        _term_check = float(np.abs(_num_filtered-_last_n)) \
            / float(_num_elements) < smallest_improvement
        if _term_check and _num_filtered < _num_elements:
            _stable_cycle += 1
            if _stable_cycle >= stable_cycle:
                break
        elif not _term_check and _stable_cycle > 0:
            _stable_cycle = 0
        elif _num_filtered == 0:
            break
        else:
            _last_n = _num_filtered
    # clean up memory:
    del(data_tree)
    # brute force the leftovers:
    if feedback > 0:
        print('nearest_elimination: polishing')
    with joblib.Parallel(n_jobs=n_threads) as parallel:
        _kde_eval_pdf[_filter] = parallel(joblib.delayed(_gauss_kde_pdf)
                                          (samp, white_samples, weights)
                                          for samp in feedback_helper(white_samples[_filter]))
        _filter[_filter] = _kde_eval_pdf[_filter] < np.log(_zero_prob)
    if feedback > 0:
        print('    surviving elements', np.sum(_filter),
              'of', _num_elements)
    # compute number of filtered elements:
    _num_filtered = num_samples - np.sum(weights[_filter])
    #
    return _num_filtered


def exact_parameter_shift_2D_fft(diff_chain, param_names=None,
                                 scale=None, nbins=1024, feedback=1,
                                 boundary_correction_order=1,
                                 mult_bias_correction_order=1,
                                 **kwarks):
    """
    Compute the MCMC estimate of the probability of a parameter shift given
    an input parameter difference chain in 2 dimensions and by using FFT.
    This function uses GetDist 2D fft and optimal bandwidth estimates to
    perform the MCMC parameter shift integral discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        If none is provided the algorithm uses GetDist optimized bandwidth.
    :param nbins: (optional) number of 2D bins for the fft.
        Default is 1024.
    :param mult_bias_correction_order: (optional) multiplicative bias
        correction passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param boundary_correction_order: (optional) boundary correction
        passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :return: probability value and error estimate.
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # check that we only have two parameters:
    if len(param_names) != 2:
        raise ValueError('Calling 2D algorithm with more than 2 parameters')
    # initialize scale:
    if scale is None or isinstance(scale, str):
        scale = -1
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # compute the density with GetDist:
    t0 = time.time()
    density = diff_chain.get2DDensity(x=ind[0], y=ind[1], normalized=True,
                                      fine_bins_2D=nbins,
                                      smooth_scale_2D=scale,
                                      boundary_correction_order=boundary_correction_order,
                                      mult_bias_correction_order=mult_bias_correction_order)
    # initialize the spline:
    density._initSpline()
    # get density of zero:
    prob_zero = density.spl([0.], [0.])[0][0]
    # do the MC integral:
    probs = density.spl.ev(diff_chain.samples[:, ind[0]],
                           diff_chain.samples[:, ind[1]])
    # filter:
    _filter = probs > prob_zero
    # if there are samples above zero then use MC:
    if np.sum(_filter) > 0:
        _num_filtered = float(np.sum(diff_chain.weights[_filter]))
        _num_samples = float(np.sum(diff_chain.weights))
        _P = float(_num_filtered)/float(_num_samples)
        _low, _upper = utils.clopper_pearson_binomial_trial(_num_filtered,
                                                            _num_samples,
                                                            alpha=0.32)
    # if there are no samples try to do the integral:
    else:
        norm = simps(simps(density.P, density.y), density.x)
        _second_filter = density.P < prob_zero
        density.P[_second_filter] = 0
        _P = simps(simps(density.P, density.y), density.x)/norm
        _low, _upper = None, None
    #
    t1 = time.time()
    if feedback > 0:
        print('Time taken for FFT-KDE calculation:', round(t1-t0, 1), '(s)')
    #
    return _P, _low, _upper


def exact_parameter_shift(diff_chain, param_names=None,
                          scale=None, method='brute_force',
                          feedback=1, **kwargs):
    """
    Compute the MCMC estimate of the probability of a parameter shift given
    an input parameter difference chain.
    This function uses a Kernel Density Estimate (KDE) algorithm discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).
    If the difference chain contains :math:`n_{\\rm samples}` this algorithm
    scales as :math:`O(n_{\\rm samples}^2)` and might require long run times.
    For this reason the algorithm is parallelized with the
    joblib library.
    To compute the KDE density estimate several methods are implemented.

    In the 2D case this defaults to
    :meth:`~tensiometer.mcmc_tension.exact_parameter_shift_2D_fft`
    unless the kwarg use_fft is False.

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        If none is provided the algorithm uses Silverman's
        rule of thumb scaling.
    :param method: (optional) a string containing the indication for the method
        to use in the KDE calculation. This can be very intensive so different
        techniques are provided.

        * method = brute_force is a parallelized brute force method. This
          method scales as :math:`O(n_{\\rm samples}^2)` and can be afforded
          only for small tensions. When suspecting a difference that is
          larger than 95% other methods are better.
        * method = nearest_elimination is a KD Tree based elimination method.
          For large tensions this scales as
          :math:`O(n_{\\rm samples}\\log(n_{\\rm samples}))`
          and in worse case scenarions, with small tensions, this can scale
          as :math:`O(n_{\\rm samples}^2)` but with significant overheads
          with respect to the brute force method.
          When expecting a statistically significant difference in parameters
          this is the recomended algorithm.

        Suggestion is to go with brute force for small problems, nearest
        elimination for big problems with signifcant tensions.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :param kwargs: extra options to pass to the KDE algorithm.
        The nearest_elimination algorithm accepts the following optional
        arguments:

        * stable_cycle: (default 2) number of elimination cycles that show
          no improvement in the result.
        * chunk_size: (default 40) chunk size for elimination cycles.
          For best perfornamces this parameter should be tuned to result
          in the greatest elimination rates.
        * smallest_improvement: (default 1.e-4) minimum percentage improvement
          rate before switching to brute force.
        * use_fft: whether to use fft when possible.
    :return: probability value and error estimate.
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # in the 2D case use FFT:
    use_fft = kwargs.get('use_fft', True)
    if len(ind) == 2 and use_fft:
        res = exact_parameter_shift_2D_fft(diff_chain,
                                           param_names=param_names,
                                           scale=scale,
                                           feedback=feedback, **kwargs)
        _P, _low, _upper = res
        return _P, _low, _upper
    # some initial calculations:
    _samples_cov = diff_chain.cov(pars=param_names)
    _num_samples = np.sum(diff_chain.weights)
    _num_params = len(ind)
    # number of effective samples:
    _num_samples_eff = np.sum(diff_chain.weights)**2 / \
        np.sum(diff_chain.weights**2)
    # scale for the kde:
    if scale is None:
        num_bins = kwargs.get('num_bins', 1000)
        scale = OptimizeBandwidth_1D(diff_chain, param_names=param_names,
                                     num_bins=num_bins)
    elif scale == 'Silverman':
        scale = Silvermans_RT(_num_params, _num_samples_eff)
        scale = scale*np.ones(_num_params)
    elif scale == 'Scott':
        scale = Scotts_RT(_num_params, _num_samples_eff)
        scale = scale*np.ones(_num_params)
    elif isinstance(scale, int) or isinstance(scale, float):
        scale = scale*np.ones(_num_params)
    # feedback:
    if feedback > 0:
        with np.printoptions(precision=3):
            print(f'Neff samples    : {_num_samples_eff:.2f}')
            print(f'Smoothing scale :', scale)
    # whiten the samples:
    _temp = np.diag(np.sqrt(scale))
    _temp = np.dot(np.dot(_temp, _samples_cov), _temp)
    _kernel_cov = np.linalg.inv(_temp)
    # whighten the samples:
    _temp = sqrtm(_kernel_cov)
    _white_samples = diff_chain.samples[:, ind].dot(_temp)
    # probability of zero:
    _kde_prob_zero = _gauss_kde_pdf(np.zeros(_num_params),
                                    _white_samples,
                                    diff_chain.weights)
    # compute the KDE:
    t0 = time.time()
    if method == 'brute_force':
        _num_filtered = _brute_force_parameter_shift(_white_samples,
                                                     diff_chain.weights,
                                                     _kde_prob_zero,
                                                     _num_samples,
                                                     feedback)
    elif method == 'nearest_elimination':
        _num_filtered = _nearest_parameter_shift(_white_samples,
                                                 diff_chain.weights,
                                                 _kde_prob_zero,
                                                 _num_samples,
                                                 feedback, **kwargs)
    else:
        raise ValueError('Unknown method provided:', method)
    t1 = time.time()
    # clean up:
    gc.collect()
    # feedback:
    if feedback > 0:
        print('KDE method:', method)
        print('Time taken for KDE calculation:', round(t1-t0, 1), '(s)')
    # probability and binomial error estimate:
    _P = float(_num_filtered)/float(_num_samples)
    _low, _upper = utils.clopper_pearson_binomial_trial(float(_num_filtered),
                                                        float(_num_samples),
                                                        alpha=0.32)
    #
    return _P, _low, _upper
