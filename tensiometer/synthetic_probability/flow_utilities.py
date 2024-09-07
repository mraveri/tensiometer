"""
This file contains a set of utilities operating on flows.
"""

###############################################################################
# initial imports and set-up:

import numpy as np

from getdist import MCSamples

###############################################################################
# get samples at each intermediate space:

def get_samples_bijectors(flow, feedback=False, extra_samples=None):
    
    # initialize the flow:
    _flow = flow
    # initialize lists to store samples:
    training_samples_spaces = []
    validation_samples_spaces = []
    if extra_samples is not None:
        extra_samples_spaces = []
    # initialize MCSamples with original space samples:
    training_samples_spaces.append(MCSamples(samples=_flow.chain_samples[_flow.training_idx, :],
                                            weights=_flow.chain_weights[_flow.training_idx],
                                            name_tag='original_space',
                                            ))
    validation_samples_spaces.append(MCSamples(samples=_flow.chain_samples[_flow.test_idx, :],
                                            weights=_flow.chain_weights[_flow.test_idx],
                                            name_tag='original_space',
                                            ))
    if extra_samples is not None:
        extra_samples_spaces.append(extra_samples)
    # append MCSamples with training samples:
    training_samples_spaces.append(MCSamples(samples=_flow.training_samples,
                                            weights=_flow.training_weights,
                                            name_tag='training_space',
                                            ))
    validation_samples_spaces.append(MCSamples(samples=_flow.test_samples,
                                            weights=_flow.test_weights,
                                            name_tag='validation_space',
                                            ))
    if extra_samples is not None:
        extra_samples_spaces.append(flow.fixed_bijector.inverse(extra_samples).numpy())
    # loop over bijectors:
    _temp_train_samples = _flow.training_samples
    _temp_test_samples = _flow.test_samples
    if extra_samples is not None:
        _temp_extra_samples = flow.fixed_bijector.inverse(extra_samples).numpy()
    for ind, bijector in enumerate(_flow.trainable_bijector._bijectors):
        # feedback:
        if feedback:
            print(ind, '- bijector name: ', bijector.name)
        # process samples trough the bijector:
        _temp_train_samples = bijector.inverse(_temp_train_samples)
        _temp_test_samples = bijector.inverse(_temp_test_samples)
        if extra_samples is not None:
            _temp_extra_samples = bijector.inverse(_temp_extra_samples)
            extra_samples_spaces.append(_temp_extra_samples.numpy())
        # get the training samples:
        training_samples_spaces.append(MCSamples(samples=_temp_train_samples.numpy(),
                                                weights=_flow.training_weights,
                                                name_tag=str(ind)+'_after_'+bijector.name,
                                                ))
        # get the validation samples:
        validation_samples_spaces.append(MCSamples(samples=_temp_test_samples.numpy(),
                                                weights=_flow.test_weights,
                                                name_tag=str(ind)+'_after_'+bijector.name,
                                                ))
    # return the samples:
    if extra_samples is not None:
        return training_samples_spaces, validation_samples_spaces, extra_samples_spaces
    else:
        return training_samples_spaces, validation_samples_spaces

###############################################################################
# flow KL divergence:

def KL_divergence(flow_1, flow_2, num_samples=1000, num_batches=100):
    """
    Calculates the Kullback-Leibler (KL) divergence between two flows:
    
    D_KL(flow_1 || flow_2) = E_{x ~ flow_1} [log(flow_1(x)) - log(flow_2(x))]

    Parameters:
    flow_1 (Flow): The flow for the sampling distribution.
    flow_2 (Flow): The second flow distribution.
    num_samples (int): The number of samples to draw from the flows, per batch. Default is 1000.
    num_batches (int): The number of batches to run. Default is 100.

    Returns:
    mean (float): The mean KL divergence.
    std (float): The standard deviation of the KL divergence.
    """
    _log_prob_diff = []
    for _ in range(num_batches):
        # sample from the first flow:
        _temp_samples = flow_1.sample(num_samples)
        # calculate the log probability difference:
        _temp_diff = flow_1.log_probability(_temp_samples) - flow_2.log_probability(_temp_samples)
        # filter out not finite values:
        _temp_diff = _temp_diff[np.isfinite(_temp_diff)]
        # average and append:        
        _log_prob_diff.append(np.mean(_temp_diff))
    # convert to numpy array:
    _log_prob_diff = np.array(_log_prob_diff)
    # filter out not finite values:
    _log_prob_diff = _log_prob_diff[np.isfinite(_log_prob_diff)]
    #
    return np.mean(_log_prob_diff), np.std(_log_prob_diff)