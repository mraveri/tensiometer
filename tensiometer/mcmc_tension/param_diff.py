"""
This module contains functions to obtain the parameter difference posterior
given posterior samples from two independent chains.

For further details we refer to `arxiv 2105.03324 <https://arxiv.org/pdf/2105.03324.pdf>`_.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import getdist.chains as gchains
gchains.print_load_details = False
from getdist import MCSamples, WeightedSamples
from ..utilities import stats_utilities as stutils

###############################################################################
# Parameter difference chain:


def parameter_diff_weighted_samples(samples_1, samples_2, boost=1,
                                    indexes_1=None, indexes_2=None,
                                    periodic_indexes=None):
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
    :param periodic_indexes: (optional) dictionary with the indexes of the
        parameters that are periodic. The keys are the indexes and the values
        are the ranges of the parameters.
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
        # compute the likelihood:
        if ch1.loglikes is not None and ch2.loglikes is not None:
            loglikes[ind*num_samps_1:(ind+1)*num_samps_1] = \
                ch1.loglikes+np.take(ch2.loglikes, _indexes, mode='wrap')
        # compute the difference samples:
        difference_samples[ind*num_samps_1:(ind+1)*num_samps_1, :] = \
            ch1.samples[:, ind1] \
            - np.take(ch2.samples[:, ind2], _indexes, axis=0, mode='wrap')
    # reapply sign:
    difference_samples = sign*difference_samples
    # apply periodicity:
    if periodic_indexes is not None:
        for _ind, _range in periodic_indexes.items():
            # compute period from range:
            _period = _range[1] - _range[0]
            # compute positive side:
            _filter = np.logical_and(difference_samples[:, _ind] > 0,
                                     difference_samples[:, _ind] > _period / 2.0)
            difference_samples[_filter, _ind] = difference_samples[_filter, _ind] - _period
            # compute negative side:
            _filter = np.logical_and(difference_samples[:, _ind] < 0,
                                     difference_samples[:, _ind] < -_period / 2.0)
            difference_samples[_filter, _ind] = difference_samples[_filter, _ind] + _period
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
                                   samples=difference_samples,
                                   weights=weights, loglikes=loglikes,
                                   name_tag=name_tag, label=label,
                                   min_weight_ratio=min_weight_ratio)
    #
    return diff_samples

###############################################################################


def parameter_diff_chain(chain_1, chain_2, boost=1, param_names=None, periodic_params=None, **kwargs):
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
    :param param_names: (optional) list with the names of the parameters to
        use for the difference chain. By default this tries to use all
        parameters.
    :param periodic_params: (optional) dictionary with the names of the
        parameters that are periodic. The keys are the names and the values
        are the ranges of the parameters.
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
    _param_names = [_p for _p in param_names_1 if _p in param_names_2]
    num_params = len(_param_names)
    if num_params == 0:
        raise ValueError('There are no shared parameters to difference')
    if param_names is None:
        param_names = _param_names
    else:
        # check that the input names are in the common names:
        if not all([_p in _param_names for _p in param_names]):
            raise ValueError('Not all input parameters are shared between chains')    
        # get the parameter names:   
        param_names = [_p for _p in param_names if _p in _param_names]
    # get the names and labels:
    diff_param_names = ['delta_'+name for name in param_names]
    diff_param_labels = ['\\Delta '+name.label for name in
                         chain_1.getParamNames().parsWithNames(param_names)]
    # get parameter indexes:
    indexes_1 = [chain_1.index[name] for name in param_names]
    indexes_2 = [chain_2.index[name] for name in param_names]
    # process periodic parameters:
    _periodic_params = None
    if periodic_params is not None:
        _periodic_params = {}
        for name, _range in periodic_params.items():
            if name in param_names:
                _ind = param_names.index(name)
                _periodic_params[_ind] = _range
    # get separate chains:
    if not hasattr(chain_1, 'chain_offsets'):
        _chains_1 = [chain_1]
    else:
        if chain_1.chain_offsets is None:
            _chains_1 = [chain_1]
        else:
            _chains_1 = chain_1.getSeparateChains()
    if not hasattr(chain_2, 'chain_offsets'):
        _chains_2 = [chain_2]
    else:
        if chain_2.chain_offsets is None:
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
    diff_chain_samples = [
        parameter_diff_weighted_samples(samp1,
                                        samp2,
                                        boost=sample_boost, 
                                        indexes_1=indexes_1,
                                        indexes_2=indexes_2,
                                        periodic_indexes=_periodic_params) 
        for samp1, samp2 in chains_combinations]
    # create the samples:
    diff_samples = MCSamples(names=diff_param_names, labels=diff_param_labels,
                             **stutils.filter_kwargs(kwargs, MCSamples))
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
        _temp_name = _nam.name.replace('delta_', '', 1)
        _nam.isDerived = _temp_paramnames.parWithName(_temp_name).isDerived
    # update and compute everything:
    diff_samples.updateBaseStatistics()
    diff_samples.deleteFixedParams()
    #
    return diff_samples
