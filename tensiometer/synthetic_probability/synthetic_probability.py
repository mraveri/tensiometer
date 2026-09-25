"""
Main file containing the synthetic probability class and methods.

The flows are implemented in PyTorch. Precision (float32 by default, float64 on request)
and the default device (cpu, cuda, mps) are process-wide settings, see
:mod:`~tensiometer.synthetic_probability.tensor_utilities`. Public methods accept numpy
arrays or tensors on any device and return detached CPU tensors (``.numpy()`` works on the
results); when the input is a tensor that requires gradients the result stays attached to
the graph on the flow device, so that the methods can be nested and differentiated.
"""

###############################################################################
# initial imports and set-up:

import os
import copy
import hashlib
import inspect
import json
import pickle
import time
import warnings
import numpy as np
import getdist.chains as gchains
from getdist import MCSamples
import scipy
import scipy.integrate
from scipy.spatial import cKDTree
import scipy.stats
from collections.abc import Iterable
import torch

# plotting:
import matplotlib
from matplotlib import pyplot as plt

# local imports:
from . import lr_schedulers as lr
from . import loss_functions as loss
from . import trainable_bijectors as tb
from . import fixed_bijectors as pb
from . import bijectors as bj
from . import distributions as ds
from . import autodiff
from . import training
from . import tensor_utilities as tu

from ..utilities import stats_utilities as stutils
from .. import gaussian_tension
from .. import __version__ as _tensiometer_version

gchains.print_load_details = False

# version of the snapshot file format written by FlowCallback.save:
SNAPSHOT_FORMAT_VERSION = 1

# save mode of the snapshot being written (None outside of save):
_save_mode = None


def __getattr__(name):
    """
    Forward ``prec`` and ``np_prec`` to the live values of
    :mod:`~tensiometer.synthetic_probability.tensor_utilities`.

    :param name: attribute name.
    :returns: the torch or numpy dtype of the active precision.
    :raises AttributeError: for any other name.
    """
    if name == 'prec':
        return tu.prec
    if name == 'np_prec':
        return tu.np_prec
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def _normalize_weights(weights):
    """
    Normalize weights to sum to the sample count.

    :param weights: array of weights.
    :returns: normalized weights array.
    """
    weights = np.array(weights, dtype=float, copy=True)
    if len(weights) == 0:
        return weights
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        weights = np.ones_like(weights, dtype=float)
        total = np.sum(weights)
    weights *= len(weights) / total
    return weights


def _precision_name(dtype):
    """Name of a torch floating dtype, for example ``'float32'``."""
    return str(dtype).replace('torch.', '')


# plotting global settings:
matplotlib_backend = matplotlib.get_backend()
try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    pass
ipython_plotting = 'inline' in matplotlib_backend
cluster_plotting = 'agg' in matplotlib_backend
if not ipython_plotting and not cluster_plotting:
    plt.ion()

# options for all plots:
plot_options = {
    # lines:
    'lines.linewidth': 1.0,  # line width in points
    # axes:
    'axes.linewidth': 0.8,  # edge line width
    'axes.titlelocation': 'left',  # alignment of the title: {left, right, center}
    'axes.titlesize': 10,  # font size of the axes title
    'axes.labelsize': 8,  # font size of the x and y labels
    # ticks:
    'xtick.labelsize': 8,  # font size of the tick labels
    'ytick.labelsize': 8,  # font size of the tick labels
    # legend:
    'legend.loc': 'best',
    'legend.frameon': False,  # if True, draw the legend on a background patch
    'legend.fontsize': 8,
}

###############################################################################
# main class to compute NF-based probability distributions:


class FlowCallback(training.Callback):
    """
    A class to compute the normalizing flow interpolation of a probability density given the samples.

    A normalizing flow is trained to approximate the distribution and then used to numerically evaluate the probablity of a parameter shift (see REF). To do so, it defines a bijective mapping that is optimized to gaussianize the difference chain samples. This mapping is performed in two steps, using the gaussian approximation as pre-whitening. The notations used in the code are:

    * `X` designates samples in the original parameter difference space;
    * `Y` designates samples in the gaussian approximation space, `Y` is obtained by shifting and scaling `X` by its mean and covariance (like a PCA);
    * `Z` designates samples in the gaussianized space, connected to `Y` with a normalizing flow denoted `trainable_bijector`.

    The user may provide the `trainable_bijector` as a :class:`~tensiometer.synthetic_probability.bijectors.Bijector` object or make use of the utility class :class:`~tensiometer.synthetic_probability.trainable_bijectors.AutoregressiveFlow` to instantiate a Masked Autoregressive Flow (with `trainable_bijector='AutoregressiveFlow'`).

    This class derives from :class:`~tensiometer.synthetic_probability.training.Callback`, which allows for visualization during training. The normalizing flows (X->Y->Z) are implemented as :class:`~tensiometer.synthetic_probability.bijectors.Bijector` objects (PyTorch modules) and trained with a :class:`~tensiometer.synthetic_probability.training.Trainer`.

    Here is an example:

    .. code-block:: python

        # Initialize the flow and model
        diff_flow_callback = FlowCallback(chain, trainable_bijector='AutoregressiveFlow')
        # Train the model
        diff_flow_callback.train()
        # Save and reload, without the chain:
        diff_flow_callback.save('diff_flow.pt')
        diff_flow_callback = FlowCallback.load('diff_flow.pt')

    :param chain: input parameter difference chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :type param_names: list, optional
    :param param_ranges: dictionary with the ranges of all the parameters, by default from the chain.
    :param periodic_params: list of names of periodic parameters.
    :param feedback: feedback level, defaults to 1. Zero is no feedback (including training plotting). One is a little feedback. Two and higher is a lot of feedback (useful for debug).
    :type feedback: int, optional
    :param plot_every: how much to plot during training. This quantifies how many epochs should pass before plotting.
    :type plot_every: int, optional
    :param initialize_model: build the trainer at initialization (otherwise at the first training).
    :param prior_bijector: ``'ranges'`` (uniform priors on the parameter ranges), a bijector, or None.
    :param apply_rescaling: whiten the samples with their Gaussian approximation (True),
        rescale them independently (``'independent'``) or not (False).
    :param trainable_bijector: ``'AutoregressiveFlow'``, a
        :class:`~tensiometer.synthetic_probability.trainable_bijectors.TrainableTransformation`,
        a bijector, or None. Defaults to ``'AutoregressiveFlow'``.
    :param validation_split: fraction of samples to use for the validation sample, defaults to 0.1
    :type validation_split: float, optional
    :param device: compute device (``'cpu'``, ``'cuda'``, ``'cuda:N'``, ``'mps'``), defaults to
        the device set with :func:`~tensiometer.synthetic_probability.tensor_utilities.set_device`
        or ``TENSIOMETER_DEVICE`` (cpu unless changed).
    :param kwargs: options of the trainable transformation, of the loss function and of the
        training split (``rng``, ``validation_training_idx``, ``learning_rate``, ...).
    :reference: George Papamakarios, Theo Pavlakou, Iain Murray (2017). Masked Autoregressive Flow for Density Estimation. `arXiv:1705.07057 <https://arxiv.org/abs/1705.07057>`_
    """

    # defaults for objects built without the constructor:
    is_light = False
    _trainer_initialized = False

    # attributes that are not stored in light snapshots (see save):
    _full_only_attributes = (
        'chain_samples', 'chain_weights', 'chain_loglikes', 'chain_nearest_index',
        'training_samples', 'test_samples', 'training_weights', 'test_weights',
        'training_logP_preabs', 'test_logP_preabs', 'training_idx', 'test_idx',
        'training_dataset', 'validation_dataset', 'chi2Y', 'chi2Z', 'loss', 'trainer',
    )

    def __init__(
            self,
            chain,
            param_names=None,
            param_ranges=None,
            periodic_params=None,
            feedback=1,
            plot_every=10,
            initialize_model=True,
            prior_bijector='ranges',
            apply_rescaling=True,
            trainable_bijector='AutoregressiveFlow',
            validation_split=0.1,
            device=None,
            **kwargs):

        # check input:
        if feedback == None:
            feedback = 0
        if feedback < 0 or not isinstance(feedback, int):
            raise ValueError('feedback needs to be a positive integer')
        if plot_every < 0 or not isinstance(plot_every, int):
            raise ValueError('plot_every needs to be a positive integer')
        if 'trainable_bijector_path' in kwargs:
            raise ValueError('trainable_bijector_path has been removed. Save the whole flow with '
                             'flow.save(path) and restore it with FlowCallback.load(path).')

        # read in varaiables:
        self.feedback = feedback
        self.plot_every = plot_every
        self.is_light = False
        # precision and device:
        self.device = tu.check_device(device)
        tu.lock_precision()
        self.prec = tu.get_precision()
        self.np_prec = tu.np_prec

        # initialize internal samples from chain:
        self._init_chain(chain,
                         param_names=param_names,
                         param_ranges=param_ranges,
                         periodic_params=periodic_params,
                         **kwargs)
        # initialize fixed bijector:
        self._init_fixed_bijector(prior_bijector=prior_bijector, apply_rescaling=apply_rescaling)
        # initialize trainable bijector:
        self._init_trainable_bijector(trainable_bijector=trainable_bijector, **kwargs)
        # initialize training dataset:
        self._init_training_dataset(
            validation_split=validation_split, **stutils.filter_kwargs(kwargs, self._init_training_dataset))
        # initialize distribution:
        self._init_distribution()
        # initialize loss function:
        self._init_loss_function(**kwargs)
        # initialize trainer:
        self._trainer_initialized = False
        if initialize_model:
            self._init_trainer()
        # initialize training metrics and plotting:
        self._init_training_monitoring()

        # initialize internal variables:
        self.is_trained = False
        self.MAP_coord = None
        self.MAP_logP = None

    def _init_chain(self, chain=None, param_names=None, param_ranges=None, periodic_params=None, init_nearest=False, **kwargs):
        """
        Read in MCMC sample chain and save internal quantities.
        """
        # return if we have no chain:
        if chain is None:
            return None

        # feedback:
        if self.feedback > 0:
            print('* Initializing samples')
        _time = time.time()

        # save name of the flow:
        if chain.name_tag is not None:
            self.name_tag = chain.name_tag + '_flow'
        else:
            self.name_tag = 'flow'
        # feedback:
        if self.feedback > 1:
            print('    - flow name:', self.name_tag)
            print('    - precision:', self.prec)
            print('    - device   :', self.device)

        # initialize param names:
        if param_names is None:
            param_names = chain.getParamNames().getRunningNames()
        else:
            chain_params = chain.getParamNames().list()
            if not np.all([name in chain_params for name in param_names]):
                raise ValueError(
                    'Input parameter is not in the chain.\n', 'Input parameters ', param_names, '\n'
                    'Possible parameters', chain_params)
        # save param names:
        self.param_names = param_names
        # save param labels:
        self.param_labels = [name.label for name in chain.getParamNames().parsWithNames(param_names)]

        # check periodic parameters:
        if periodic_params is not None:
            if not isinstance(periodic_params, Iterable) or isinstance(periodic_params, str):
                periodic_params = [periodic_params]
            for name in periodic_params:
                if name not in param_names:
                    raise ValueError('Periodic parameter ', name, ' is not in the chain.')
        else:
            periodic_params = []
        self.periodic_params = periodic_params
        self.trainable_periodic_params = []

        # initialize ranges:
        self.parameter_ranges = {}
        for name in param_names:
            # get ranges from user:
            if param_ranges is not None:
                if name not in param_ranges.keys():
                    raise ValueError(
                        'Range for parameter ', name, ' is not specified.\n',
                        'When passing ranges explicitly all parameters have to be included.')
                else:
                    self.parameter_ranges[name] = copy.deepcopy(param_ranges[name])
            # get ranges from MCSamples:
            else:
                temp_range = []
                # lower:
                if name in chain.ranges.lower.keys():
                    temp_range.append(chain.ranges.lower[name])
                else:
                    temp_range.append(np.amin(chain.samples[:, chain.index[name]]))
                # upper:
                if name in chain.ranges.upper.keys():
                    temp_range.append(chain.ranges.upper[name])
                else:
                    temp_range.append(np.amax(chain.samples[:, chain.index[name]]))
                # save:
                self.parameter_ranges[name] = copy.deepcopy(temp_range)
        # check that all samples are within ranges:
        for name in param_names:
            if np.any(chain.samples[:, chain.index[name]] < self.parameter_ranges[name][0]) or \
                    np.any(chain.samples[:, chain.index[name]] > self.parameter_ranges[name][1]):
                raise ValueError('Samples for parameter ', name,
                                 ' are outside the specified range: ', self.parameter_ranges[name],
                                 ' min/max values are: ', np.amin(chain.samples[:, chain.index[name]]),
                                 np.amax(chain.samples[:, chain.index[name]]))
        # feedback:
        if self.feedback > 1:
            print('    - flow parameters and ranges:')
            for name in param_names:
                print('      ' + name + ' : [{0:.6g}, {1:.6g}]'.format(*self.parameter_ranges[name]))
            if self.periodic_params is not None:
                print('    - periodic parameters:', self.periodic_params)

        # initialize sample MAP:
        if chain.loglikes is not None:
            temp = chain.samples[np.argmin(chain.loglikes), :]
        else:
            temp = chain.samples[np.argmax(chain.weights), :]
        self.sample_MAP = np.array([temp[chain.index[name]] for name in param_names])
        # try to get real best fit:
        try:
            self.chain_MAP = np.array([name.best_fit for name in chain.getBestFit().parsWithNames(param_names)])
        except:
            self.chain_MAP = None

        # initialize the samples:
        ind = [chain.index[name] for name in param_names]
        self.num_params = len(ind)
        self.chain_samples = chain.samples[:, ind].astype(self.np_prec)
        self.chain_weights = chain.weights.astype(self.np_prec)

        # initialize loglikes:
        self.has_loglikes = chain.loglikes is not None
        if not self.has_loglikes:
            self.chain_loglikes = None
        else:
            self.chain_loglikes = chain.loglikes.astype(self.np_prec)

        # initialize nearest neighbours:
        if init_nearest:
            self._init_nearest_samples()

        # print feedback:
        if self.feedback > 0:
            print(f'    - time taken: {time.time() - _time:.4f} seconds')
        #
        return None

    def _init_nearest_samples(self):
        """
        Initializes samples in a tree for nearest neighbour searches. Takes a while in high dimensions...
        """
        # cache nearest neighbours indexes, in whitened coordinates:
        if self.chain_samples.shape[0] < 2:
            temp_cov = np.eye(self.num_params)
        else:
            _weight_sum = np.sum(self.chain_weights)
            if not np.isfinite(_weight_sum) or _weight_sum <= 0:
                temp_cov = np.cov(self.chain_samples.T)
            else:
                temp_cov = np.cov(self.chain_samples.T, aweights=self.chain_weights)
            if not np.all(np.isfinite(temp_cov)):
                temp_cov = np.eye(self.num_params)
        white_samples = np.dot(scipy.linalg.sqrtm(np.linalg.inv(temp_cov)), self.chain_samples.T).T
        data_tree = cKDTree(white_samples, balanced_tree=True)
        r2, idx = data_tree.query(white_samples, (2), workers=-1)
        self.chain_nearest_index = idx.copy()
        #
        return None

    def _init_fixed_bijector(self, prior_bijector='ranges', apply_rescaling=True):
        """
        Intitialize prior and whitening bijector.
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing fixed bijector')
        _time = time.time()

        # Prior bijector setup:
        if isinstance(prior_bijector, str) and prior_bijector == 'ranges':
            # extend slightly the ranges to avoid overflows:
            temp_ranges = []
            for name in self.param_names:
                if name not in self.periodic_params:
                    temp_range = self.parameter_ranges[name]
                    center = 0.5 * (temp_range[0] + temp_range[1])
                    length = temp_range[1] - temp_range[0]
                    eps = 0.01
                    temp_ranges.append({
                        'lower': self.cast(center - 0.5 * length * (1. + eps)),
                        'upper': self.cast(center + 0.5 * length * (1. + eps)),
                        'mode': 'uniform'
                    })
                else:
                    temp_ranges.append(None)
            # define bijector:
            self.prior_bijector = pb.prior_bijector_helper(temp_ranges)
        elif isinstance(prior_bijector, bj.Bijector):
            self.prior_bijector = prior_bijector
        elif prior_bijector is None or prior_bijector is False:
            self.prior_bijector = bj.Identity()
        else:
            raise ValueError('prior_bijector must be "ranges", a bijector or None, got ' + repr(prior_bijector))
        self.bijectors = [self.prior_bijector]

        # feedback:
        if self.feedback > 1:
            print('    - using prior bijector:', prior_bijector)

        # Whitening bijector:
        if apply_rescaling:
            # calculate gaussian approximation, leaving out periodic parameters:
            with torch.no_grad():
                temp_X = tu.to_numpy(self.prior_bijector.inverse(self.chain_samples))
            temp_chain = MCSamples(samples=temp_X, weights=self.chain_weights, names=self.param_names)
            temp_gaussian_approx = gaussian_tension.gaussian_approximation(temp_chain, param_names=self.param_names)
            # calculate mean and covariance:
            _mean = np.array(temp_gaussian_approx.means[0], dtype=np.float64)
            _cov = np.array(temp_gaussian_approx.covs[0], dtype=np.float64)
            # periodic parameters are handled differently, rescaling to unit box:
            if len(self.periodic_params) > 0:
                for name in self.periodic_params:
                    _index = self.param_names.index(name)
                    a, b = self.parameter_ranges[name]
                    _mean[_index] = 0.5*(a+b)
                    _cov[_index, :] = 0.0
                    _cov[:, _index] = 0.0
                    _cov[_index, _index] = (0.5*(b-a))**2
            if apply_rescaling == 'independent':
                _scale_tril = np.diag(np.sqrt(np.diagonal(_cov)))
            else:
                _scale_tril = np.linalg.cholesky(_cov)
            self.bijectors.append(bj.AffineTriL(_mean, _scale_tril))

        # feedback:
        if self.feedback > 1:
            if apply_rescaling:
                print('    - rescaling samples')
            else:
                print('    - not rescaling samples')

        # if we have periodic coordinates we need to add a modulus bijector:
        if len(self.periodic_params) > 0:
            # check if we are doing rescalings:
            if not apply_rescaling:
                raise ValueError('Cannot use periodic parameters without rescaling')
            # chain the bijectors and evaluate them:
            _temp_bijectors = bj.Chain(self.bijectors)
            with torch.no_grad():
                _temp_samples = tu.to_numpy(_temp_bijectors.inverse(self.chain_samples))
            # build modulus bijector:
            temp_bijectors = []
            for name in self.param_names:
                if name in self.periodic_params:
                    # get index of periodic parameter:
                    _index = self.param_names.index(name)
                    # compute circular mean:
                    _avg_sin = np.average(np.sin(np.pi*_temp_samples[:,_index]), weights=self.chain_weights)
                    _avg_cos = np.average(np.cos(np.pi*_temp_samples[:,_index]), weights=self.chain_weights)
                    _circ_mean = float(self.np_prec(np.arctan2(_avg_sin, _avg_cos) / np.pi))
                    # shift and mod the samples to calculate variance:
                    with torch.no_grad():
                        _tmp = tu.to_numpy(pb.Mod1D(minval=-1.0, maxval=1.0).forward(_temp_samples[:,_index]-_circ_mean))
                    _circ_var = np.average(_tmp**2, weights=self.chain_weights)
                    # define bijectors:
                    _temp_temp_bijectors = [pb.Mod1D(minval=-1.0, maxval=1.0), bj.Shift(_circ_mean), pb.Mod1D(minval=-1.0, maxval=1.0)]
                    # if the variance is small (the distribution is well localized inside a period) then rescale to variance 1:
                    if np.sqrt(_circ_var) / 2.0 < 0.05:
                        _temp_temp_bijectors.append(bj.Scale(float(np.sqrt(_circ_var))))
                    else:
                        self.trainable_periodic_params.append(name)
                    temp_bijectors.append(bj.Chain(_temp_temp_bijectors, name='ShiftMod1D'))
                else:
                    temp_bijectors.append(bj.Identity())
            bijector = bj.Blockwise(temp_bijectors, name='ModBijector')
            self.bijectors.append(bijector)

        self.fixed_bijector = bj.Chain(self.bijectors)

        # feedback:
        if self.feedback > 0:
            print('    - time taken: {0:.4f} seconds'.format(time.time() - _time))

        #
        return None

    def _init_trainable_bijector(self, trainable_bijector, **kwargs):
        """
        Initialize trainable part of the bijector
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing trainable bijector')
        _time = time.time()

        # add periodic parameters to kwargs:
        if 'periodic_params' not in kwargs.keys():
            kwargs['periodic_params'] = [True if name in self.trainable_periodic_params else False for name in self.param_names]

        # calculate minimum ranges in training space:
        with torch.no_grad():
            _training_samples = tu.to_numpy(self.fixed_bijector.inverse(self.chain_samples))
        _training_space_min = np.amin(_training_samples, axis=0).astype(self.np_prec)
        _training_space_max = np.amax(_training_samples, axis=0).astype(self.np_prec)
        kwargs['parameters_min'] = _training_space_min
        kwargs['parameters_max'] = _training_space_max
        # select model for trainable transformation:
        if isinstance(trainable_bijector, str) and trainable_bijector == 'AutoregressiveFlow':
            self.trainable_transformation = tb.AutoregressiveFlow(self.num_params,
                                                                  feedback=self.feedback,
                                                                  device='cpu',
                                                                  **kwargs)
        elif isinstance(trainable_bijector, tb.TrainableTransformation):
            self.trainable_transformation = trainable_bijector
        elif isinstance(trainable_bijector, bj.Bijector):
            self.trainable_transformation = None
        elif trainable_bijector is None or trainable_bijector is False:
            self.trainable_transformation = None
        else:
            raise ValueError('trainable_bijector must be "AutoregressiveFlow", a TrainableTransformation, '
                             'a bijector or None, got ' + repr(trainable_bijector))

        # initialize bijector:
        if self.trainable_transformation is not None:
            self.trainable_bijector = self.trainable_transformation.bijector
        elif isinstance(trainable_bijector, bj.Bijector):
            self.trainable_bijector = trainable_bijector
        else:
            self.trainable_bijector = bj.Identity()

        self.bijectors.append(self.trainable_bijector)
        self.bijector = bj.Chain(self.bijectors)

        # move all the bijectors to the flow device:
        self.bijector.to(self.device)
        if self.trainable_transformation is not None and hasattr(self.trainable_transformation, 'device'):
            self.trainable_transformation.device = self.device

        # feedback:
        if self.feedback > 0:
            print('    - time taken: {0:.4f} seconds'.format(time.time() - _time))
        #
        return None

    def _init_training_dataset(self, validation_split=0.1, rng=None, validation_training_idx=None):
        """
        Initialize the training dataset, splitting training and validation.
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing training dataset')
        _time = time.time()

        # split training/test:
        n = self.chain_samples.shape[0]
        if validation_training_idx is None:
            if rng is not None:
                indices = rng.permutation(n)
            else:
                indices = np.random.permutation(n)
            n_split = int(validation_split * n)
            self.test_idx, self.training_idx = indices[:n_split], indices[n_split:]
        else:
            self.test_idx, self.training_idx = validation_training_idx

        with torch.no_grad():
            # training samples:
            self.training_samples = tu.to_numpy(self.fixed_bijector.inverse(
                self.chain_samples[self.training_idx, :])).astype(self.np_prec)
            self.num_training_samples = len(self.training_samples)

            if self.has_loglikes:
                _jac_true_preabs = tu.to_numpy(self.fixed_bijector.inverse_log_det_jacobian(
                    self.chain_samples[self.training_idx, :], event_ndims=1))
                self.training_logP_preabs = (-1. * self.chain_loglikes[self.training_idx] - _jac_true_preabs).astype(self.np_prec)
            else:
                self.training_logP_preabs = None

            self.training_weights = _normalize_weights(self.chain_weights[self.training_idx])
            self.has_weights = bool(np.any(self.training_weights != self.training_weights[0])) if len(self.training_weights) > 0 else False

            # test samples:
            self.test_samples = tu.to_numpy(self.fixed_bijector.inverse(self.chain_samples[self.test_idx, :])).astype(self.np_prec)
            self.num_test_samples = len(self.test_samples)

            if self.has_loglikes:
                _jac_true_test_preabs = tu.to_numpy(self.fixed_bijector.inverse_log_det_jacobian(
                    self.chain_samples[self.test_idx, :], event_ndims=1))
                self.test_logP_preabs = (-1. * self.chain_loglikes[self.test_idx] - _jac_true_test_preabs).astype(self.np_prec)
            else:
                self.test_logP_preabs = None

            self.test_weights = _normalize_weights(self.chain_weights[self.test_idx])

        # initialize the tensors used for training:
        self._init_dataset_tensors()

        # final feedback
        if self.feedback > 1:
            if self.has_weights:
                print(
                    '    - {}/{} training/test samples and non-uniform weights'.format(
                        self.num_training_samples, self.num_test_samples))
                print(
                    '    - {0:.6g} effective number of training samples'.format(
                        np.sum(self.training_weights)**2 / np.sum(self.training_weights**2)))
                print(
                    '    - {0:.6g} effective number of test samples'.format(
                        np.sum(self.test_weights)**2 / np.sum(self.test_weights**2)))
            else:
                print(
                    '    - {}/{} training/test samples and uniform weights'.format(
                        self.num_training_samples, self.num_test_samples))
        if self.feedback > 0:
            print('    - time taken: {0:.4f} seconds'.format(time.time() - _time))
        #
        return None

    def _init_dataset_tensors(self):
        """
        Create the training and validation tensors ``(samples, logP or None, weights)`` on the
        flow device from the stored numpy arrays.
        """
        def _tensor(value):
            if value is None:
                return None
            return tu.to_tensor(value, device=self.device)
        self.training_dataset = (
            _tensor(self.training_samples),
            _tensor(getattr(self, 'training_logP_preabs', None)),
            _tensor(self.training_weights))
        self.validation_dataset = (
            _tensor(self.test_samples),
            _tensor(getattr(self, 'test_logP_preabs', None)),
            _tensor(self.test_weights))
        #
        return None

    def _build_distributions(self):
        """Build the base, full and abstract space distributions on the flow device."""
        self.base_distribution = ds.standard_normal(self.num_params, device=self.device)
        # samples from std gaussian mapped to original space:
        self.distribution = ds.TransformedDistribution(
            distribution=self.base_distribution, bijector=self.bijector)
        # abstract space distribution:
        self.trained_distribution = ds.TransformedDistribution(
            distribution=self.base_distribution, bijector=self.trainable_bijector)

    def _init_distribution(self):
        """
        Initialize the transformed distributions
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing transformed distribution')
        _time = time.time()
        self._build_distributions()
        # feedback:
        if self.feedback > 0:
            print('    - time taken: {0:.4f} seconds'.format(time.time() - _time))
        #
        return None

    def _reset_optimizer(self):
        """
        Reset the loss function state and create a fresh optimizer.
        """
        # reset loss function:
        self.loss.reset()
        # new optimizer:
        self.trainer.reset_optimizer()
        #
        return None

    def _init_loss_function(
            self,
            learning_rate=1.e-3,
            global_clipnorm=1.0,
            alpha_lossv=0.5,
            beta_lossv=0.0,
            loss_mode='standard',
            **kwargs):
        """
        Initialize the loss function.

        mode can be standard, fixed or variable
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing loss function')
        _time = time.time()

        # set loss functions relative weights:
        if not self.has_loglikes and not loss_mode == 'standard':
            raise ValueError(
                'Cannot use posterior based loss functions if the input chain does not have posterior values')
        # save in:
        self.alpha_lossv = alpha_lossv
        self.beta_lossv = beta_lossv
        self.initial_learning_rate = learning_rate
        self.final_learning_rate = kwargs.get('final_learning_rate', learning_rate / 1000.)
        self.global_clipnorm = global_clipnorm
        self.loss_mode = loss_mode
        # allocate and initialize loss model:
        if self.loss_mode == 'standard':
            self.loss = loss.standard_loss()
        elif self.loss_mode == 'fixed':
            self.loss = loss.constant_weight_loss(self.alpha_lossv, self.beta_lossv)
        elif self.loss_mode == 'random':
            self.loss = loss.random_weight_loss(**kwargs)
        elif self.loss_mode == 'annealed':
            self.loss = loss.annealed_weight_loss(**kwargs)
        elif self.loss_mode == 'softadapt':
            self.loss = loss.SoftAdapt_weight_loss(**kwargs)
        elif self.loss_mode == 'sharpstep':
            self.loss = loss.SharpStep(**kwargs)
        else:
            raise ValueError('Unknown loss_mode ' + repr(self.loss_mode))
        # print feedback:
        if self.feedback > 1:
            self.loss.print_feedback(padding='    - ')
        # feedback:
        if self.feedback > 0:
            print('    - time taken: {0:.4f} seconds'.format(time.time() - _time))
        #
        return None

    def _init_trainer(self):
        """
        Initialize the trainer (optimizer and training loop).
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing trainer')
        _time = time.time()

        # build trainer:
        self._trainer_initialized = False
        self.trainer = training.Trainer(
            module=self.trainable_bijector,
            log_prob_fn=self.trained_distribution.log_prob,
            loss=self.loss,
            learning_rate=self.initial_learning_rate,
            global_clipnorm=self.global_clipnorm)
        self.loss.reset()
        num_model_params = training.count_parameters(self.trainable_bijector)
        self._trainer_initialized = True
        # feedback:
        if self.feedback > 1:
            print('    - trainable parameters :', num_model_params)
            print('    - maximum learning rate: %.3g' % (self.initial_learning_rate))
            print('    - minimum learning rate: %.3g' % (self.final_learning_rate))
        # check that number of parameters is less than data:
        num_data = self.training_samples.shape[0] * self.training_samples.shape[1]
        if num_data < num_model_params:
            print('WARNING: more parameters than data')
            print('    - trainable parameters :', num_model_params)
            print('    - number of data values:', num_data)
        # feedback:
        if self.feedback > 0:
            print('    - time taken: {0:.4f} seconds'.format(time.time() - _time))
        #
        return None

    def _check_trainable(self):
        """
        Raise if the flow cannot be trained.

        :raises RuntimeError: for flows loaded from light snapshots.
        """
        if self.is_light:
            raise RuntimeError('This flow was loaded from a light snapshot and cannot be trained. '
                               'Load a full snapshot or rebuild the flow from the chain.')

    def on_epoch_begin(self, epoch, logs=None):
        """
        Initialization to be done at the beginning of every epoch:
        updates the weights of variable weight losses.

        :param epoch: index of the current epoch.
        :param logs: dictionary of training metrics passed by the training loop (unused, the
            internal ``self.log`` is used instead).
        :returns: None
        """
        # update loss function if needed:
        if issubclass(type(self.loss), loss.variable_weight_loss):
            self.loss.update_lambda_values_on_epoch_begin(epoch, logs=self.log)
        #
        return None

    def train(self, epochs=100, batch_size=None, steps_per_epoch=None, callbacks=None, verbose=None, **kwargs):
        """
        Train the normalizing flow model. Internally, this runs the fit method of the
        :class:`~tensiometer.synthetic_probability.training.Trainer`, to which relevant `**kwargs` are passed.

        :param epochs: number of training epochs, defaults to 100.
        :type epochs: int, optional
        :param batch_size: number of samples per batch, defaults to None. If None, the training sample is divided into `steps_per_epoch` batches.
        :type batch_size: int, optional
        :param steps_per_epoch: number of steps per epoch, defaults to None. If None and `batch_size` is also None, then `steps_per_epoch` is set to 20.
        :type steps_per_epoch: int, optional
        :param callbacks: a list of additional :class:`~tensiometer.synthetic_probability.training.Callback`, defaults to None which uses the learning rate scheduler selected with the ``lr_scheduler`` keyword (default :class:`~tensiometer.synthetic_probability.lr_schedulers.LRAdaptLossSlopeEarlyStop`).
        :type callbacks: list, optional
        :param verbose: verbosity level: 0 silent, 1 one line per epoch, -1 progress bar. Defaults to 0 if ``feedback`` is 0 and 1 otherwise.
        :type verbose: int, optional
        :param kwargs: ``lr_scheduler``, name of the learning rate scheduler class in
            :mod:`~tensiometer.synthetic_probability.lr_schedulers` (default
            ``'LRAdaptLossSlopeEarlyStop'``, None for no scheduler), used only if ``callbacks`` is None,
            and the options of its constructor (``min_lr`` defaults to the final learning rate of the flow).
            Options of :meth:`~tensiometer.synthetic_probability.training.Trainer.fit` other than the
            explicit arguments above are also passed through.
        :return: A :class:`~tensiometer.synthetic_probability.training.History` object. Its `history` attribute is a dictionary of training and validation loss values and learning rates at successive epochs.
        :raises RuntimeError: for flows loaded from light snapshots.
        """
        self._check_trainable()
        # check that model is initialized:
        if not self._trainer_initialized:
            self._init_trainer()
        # We're trying to loop through the full sample each epoch
        if batch_size is None:
            if steps_per_epoch is None:
                steps_per_epoch = 20
            batch_size = int(self.num_training_samples / steps_per_epoch)
        else:
            if steps_per_epoch is None:
                steps_per_epoch = int(self.num_training_samples / batch_size)
        batch_size = max(int(batch_size), 1)
        steps_per_epoch = max(int(steps_per_epoch), 1)
        # get verbosity:
        if verbose is None:
            verbose = 0 if self.feedback == 0 else 1
        # set callbacks:
        if callbacks is None:
            callbacks = []
            # get lr_scheduler from kwargs if present
            lr_scheduler_name = kwargs.get('lr_scheduler', 'LRAdaptLossSlopeEarlyStop')
            if lr_scheduler_name is not None:
                if hasattr(lr, lr_scheduler_name):
                    _scheduler = getattr(lr, lr_scheduler_name)
                    _scheduler_kwargs = stutils.filter_kwargs(kwargs, _scheduler)
                    if 'min_lr' in inspect.signature(_scheduler).parameters:
                        _scheduler_kwargs.setdefault('min_lr', self.final_learning_rate)
                    callbacks.append(_scheduler(**_scheduler_kwargs))
                else:
                    print(f"Warning: lr_scheduler '{lr_scheduler_name}' not found in lr module.")
        # validation data:
        if self.num_test_samples > 0:
            validation_data = self.validation_dataset
        else:
            validation_data = None
        # other options of the trainer:
        _fit_kwargs = stutils.filter_kwargs(kwargs, self.trainer.fit)
        for key in ['x', 'y', 'sample_weight', 'validation_data', 'epochs', 'batch_size',
                    'steps_per_epoch', 'callbacks', 'verbose']:
            _fit_kwargs.pop(key, None)
        # Run training:
        x, y, w = self.training_dataset
        hist = self.trainer.fit(
            x=x,
            y=y,
            sample_weight=w,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            callbacks=[self] + list(callbacks),
            verbose=verbose,
            **_fit_kwargs)
        # model is now trained:
        self.is_trained = True
        #
        return hist

    def global_train(self, pop_size=10, **kwargs):
        """
        Training algorithm with some globalization strategy. Starts from multiple
        random weight initializations and selects the one that has the best
        performances on the validation set after training.

        The first member of the population starts from the current weights, the others from
        freshly initialized weights.

        :param pop_size: number of weight initializations. Time to solution scales linearly with this parameter.
        :param kwargs: options passed to :meth:`train` for every member of the population.
        :returns: training and validation loss of the best member.
        :raises RuntimeError: for flows loaded from light snapshots.
        """
        self._check_trainable()
        # check that model is initialized:
        if not self._trainer_initialized:
            self._init_trainer()
        # initialize:
        best_loss, best_val_loss, best_weights, best_log = None, None, None, None
        loss, val_loss, logs = [], [], []
        ind = 1
        # do the loop:
        while ind <= pop_size:
            # feedback:
            if self.feedback > 0:
                if pop_size > 1:
                    print('* Training population', ind)
                else:
                    print('* Training')

            # initialize logs:
            self.log = {_k: [] for _k in self.log.keys()}
            self.log['population'] = ind
            if best_loss is not None:
                self.log['best_loss'] = best_loss

            # draw new random weights:
            if ind > 1:
                training.reset_module_parameters(self.trainable_bijector)

            # reset optimizer:
            self._reset_optimizer()

            # train:
            history = self.train(**kwargs)
            # save log (a population whose training never completed an epoch has infinite loss):
            if len(history.history.get('loss', [])) > 0:
                _last_loss = history.history['loss'][-1]
                _last_val_loss = history.history.get('val_loss', history.history['loss'])[-1]
            else:
                _last_loss, _last_val_loss = np.inf, np.inf
            loss.append(_last_loss)
            val_loss.append(_last_val_loss)
            logs.append(copy.deepcopy(self.log))

            # if improvement save weights:
            if best_val_loss is None or _last_val_loss < best_val_loss:
                best_log = copy.deepcopy(self.log)
                best_loss = copy.deepcopy(_last_loss)
                best_val_loss = copy.deepcopy(_last_val_loss)
                best_weights = copy.deepcopy(self.trainable_bijector.state_dict())

            # update counter:
            ind += 1

        # select best:
        self.trainable_bijector.load_state_dict(best_weights)
        self.log = best_log
        self.population_logs = logs

        if self.feedback > 1:
            _best_idx = np.argmin(val_loss)
            print('* Population optimizer:')
            print('    - best model is number', _best_idx + 1)
            print('    - best loss function is', np.round(best_loss, 2))
            print('    - best validation loss function is', np.round(best_val_loss, 2))
            with np.printoptions(precision=2, suppress=True):
                print('    - population losses', np.array(val_loss))
        #
        return best_loss, best_val_loss

    ###############################################################################
    # Utility functions:

    def cast(self, v):
        """
        Convert a vector to a host (CPU) tensor with the precision of the flow.

        :param v: input vector (array-like or tensor).
        :returns: CPU tensor.
        """
        return tu.to_tensor(v, device='cpu')

    def _input(self, coord):
        """
        Convert an input of a non-derivative method.

        :param coord: array-like or tensor.
        :returns: tensor on the flow device and whether the caller wants a graph-attached result.
        """
        needs_graph = torch.is_tensor(coord) and coord.requires_grad
        return tu.to_tensor(coord, device=self.device), needs_graph

    def _grad_input(self, coord):
        """
        Convert an input of a derivative method.

        :param coord: array-like or tensor.
        :returns: tensor requiring gradients on the flow device and whether the caller wants a
            graph-attached result.
        """
        needs_graph = torch.is_tensor(coord) and coord.requires_grad
        return autodiff.prepare_input(coord, device=self.device), needs_graph

    @staticmethod
    def _output(result, needs_graph):
        """
        Convert a result for the public interface: detached CPU tensor unless a graph is needed.

        :param result: tensor.
        :param needs_graph: keep the result attached to the graph on its device.
        :returns: tensor.
        """
        if needs_graph:
            return result
        return result.detach().cpu()

    def _evaluate(self, method, coord):
        """Evaluate a device method on a public input without derivatives."""
        x, needs_graph = self._input(coord)
        with torch.set_grad_enabled(needs_graph):
            result = method(x)
        return self._output(result, needs_graph)

    def _differentiate(self, method, coord):
        """Evaluate a device derivative method ``method(x, create_graph)`` on a public input."""
        x, needs_graph = self._grad_input(coord)
        result = method(x, create_graph=needs_graph)
        return self._output(result, needs_graph)

    ###############################################################################
    # Probability methods:

    def _log_probability(self, x):
        """Log probability on the flow device."""
        return self.distribution.log_prob(x)

    def _log_probability_jacobian(self, x, create_graph=True):
        """Gradient of the log probability on the flow device."""
        return autodiff.gradient(self._log_probability, x, create_graph=create_graph)

    def _log_probability_hessian(self, x, create_graph=True):
        """Hessian of the log probability on the flow device."""
        return autodiff.batch_jacobian(
            lambda _x: self._log_probability_jacobian(_x, create_graph=True), x, create_graph=create_graph)

    def _log_probability_abs(self, z):
        """Log probability as a function of abstract coordinates, on the flow device."""
        temp_1 = self.distribution.distribution.log_prob(z)
        temp_2 = self.distribution.bijector.forward_log_det_jacobian(z, event_ndims=1)
        return temp_1 - temp_2

    def _log_probability_abs_jacobian(self, z, create_graph=True):
        """Gradient of the log probability in abstract coordinates."""
        return autodiff.gradient(self._log_probability_abs, z, create_graph=create_graph)

    def _log_probability_abs_hessian(self, z, create_graph=True):
        """Hessian of the log probability in abstract coordinates."""
        return autodiff.batch_jacobian(
            lambda _z: self._log_probability_abs_jacobian(_z, create_graph=True), z, create_graph=create_graph)

    def log_probability(self, coord):
        """
        Returns learned log probability in parameter space.

        :param coord: input parameter value, shape ``(N, D)``
        :returns: log probability, shape ``(N,)``
        """
        return self._evaluate(self._log_probability, coord)

    def log_probability_jacobian(self, coord):
        """
        Computes the Jacobian of log probability in parameter space.

        :param coord: input parameter value, shape ``(N, D)``
        :returns: gradient, shape ``(N, D)``
        """
        return self._differentiate(self._log_probability_jacobian, coord)

    def log_probability_hessian(self, coord):
        """
        Computes the Hessian of log probability in parameter space.

        :param coord: input parameter value, shape ``(N, D)``
        :returns: Hessian, shape ``(N, D, D)``
        """
        return self._differentiate(self._log_probability_hessian, coord)

    def log_probability_abs(self, abs_coord):
        """
        Returns learned log probability in original parameter space as a function of abstract coordinates.
        This can be used to perform maximization in abstract space (which has no bounds).

        :param abs_coord: input parameter value in abstract Gaussian coordinates
        """
        return self._evaluate(self._log_probability_abs, abs_coord)

    def log_probability_abs_jacobian(self, abs_coord):
        """
        Jacobian of the original parameter space log probability with respect to abstract coordinates.

        :param abs_coord: input parameter value in abstract Gaussian coordinates
        """
        return self._differentiate(self._log_probability_abs_jacobian, abs_coord)

    def log_probability_abs_hessian(self, abs_coord):
        """
        Hessian of the original parameter space log probability with respect to abstract coordinates.

        :param abs_coord: input parameter value in abstract Gaussian coordinates
        """
        return self._differentiate(self._log_probability_abs_hessian, abs_coord)

    def _sample(self, N):
        """
        Draw samples on the flow device.
        """
        return self.distribution.sample(int(N))

    def sample(self, N):
        """
        Return samples from the synthetic probablity.

        :param N: number of samples
        :returns: CPU tensor of shape ``(N, D)``
        """
        with torch.no_grad():
            samples = self._sample(int(N))
        return samples.detach().cpu()

    def MCSamples(self, size, logLikes=True, **kwargs):
        """
        Return MCSamples object from the syntetic probability.

        :param size: number of samples
        :param logLikes: logical, whether to include log-likelihoods or not.
        :param kwargs: options passed to :class:`~getdist.mcsamples.MCSamples`; ``name_tag``
            defaults to the flow name tag.
        :returns: :class:`~getdist.mcsamples.MCSamples` with the samples (non-finite samples removed).
        """
        # sample:
        samples = self.sample(size)
        finite_filter = torch.isfinite(samples).all(dim=-1)
        if logLikes:
            loglikes = -self.log_probability(samples)
            finite_filter = finite_filter & torch.isfinite(loglikes)
        else:
            loglikes = None
        # filter out non-finite values:
        if not bool(finite_filter.all()):
            samples = samples[finite_filter]
            if loglikes is not None:
                loglikes = loglikes[finite_filter]
            # feedback:
            if self.feedback > 0:
                print('    - found non-finite values, filtering out {0} samples'.format(size - len(samples)))
        # create MCSamples object:
        mc_samples = MCSamples(
            samples=tu.to_numpy(samples),
            loglikes=None if loglikes is None else tu.to_numpy(loglikes),
            names=self.param_names,
            labels=self.param_labels,
            ranges=self.parameter_ranges,
            name_tag=kwargs.pop('name_tag', self.name_tag),
            **kwargs)
        #
        return mc_samples

    def evidence(self, indexes=None, weighted=False):
        """
        Get evidence from the flow. Can pass indexes to use only some of the samples for the estimate.

        Each chain sample gives an estimate of the log evidence, ``-loglikes - log_probability``,
        the difference between the chain log posterior (up to normalization) and the normalized flow
        log probability. For a perfect flow all the estimates are equal to the log evidence.
        The chain log-likelihoods are needed, and they must refer to the flow parameters.

        - The returned value is the weighted average of these estimates. On posterior samples its
          expectation is the log evidence plus the Kullback-Leibler divergence
          ``KL(posterior || flow)``, so the estimate is biased high, by an amount that vanishes for
          a perfect flow.
        - The returned error is the weighted standard deviation of the estimates. It measures how
          well the flow reproduces the local values of the posterior (it vanishes for a perfect
          flow) and gives a conservative scale for the error of the estimate, which is dominated by
          the bias above. It is not the statistical error of the mean, which is smaller by about
          the square root of the number of samples and does not include the bias.

        :param indexes: indexes (or boolean mask) of the chain samples to use, defaults to None (all samples).
        :param weighted: if True, further weight the samples with the chi-squared survival function
            of their log-likelihood distance from the best sample, defaults to False.
        :returns: tuple with the log evidence estimate and the weighted standard deviation of the
            per-sample estimates.
        :raises ValueError: if the chain log-likelihoods of the flow parameters are not available
            (chains without log-likelihoods and transformed flows). The evidence does not depend on
            the parameterization, so for a transformed flow it can be computed on the original flow.
        """
        # the chain log-likelihoods of the flow parameters are needed (light flows have no chain at all):
        if not self.__dict__.get('is_light', False) and not self.has_loglikes:
            raise ValueError('The evidence needs the chain log-likelihoods of the flow parameters, which are not '
                             'available for this flow (chain without log-likelihoods or transformed flow). '
                             'The evidence does not depend on the parameterization: for a transformed flow '
                             'compute it on the original flow.')
        # filter by index:
        if indexes is not None:
            _samples = self.chain_samples[indexes, :]
            _loglikes = self.chain_loglikes[indexes]
            _weights = self.chain_weights[indexes]
        else:
            _samples = self.chain_samples
            _loglikes = self.chain_loglikes
            _weights = self.chain_weights
        # compute log likes:
        flow_log_likes = tu.to_numpy(self.log_probability(_samples))
        # use distance weights if required:
        if weighted:
            evidence_weights = scipy.stats.chi2.sf(2.0 * (_loglikes - np.amin(_loglikes)), self.num_params)
            _weights = _weights * evidence_weights
        # compute residuals:
        diffs = -_loglikes - flow_log_likes
        # compute average and error:
        average = np.average(diffs, weights=_weights)
        variance = np.average((diffs - average)**2, weights=_weights)
        #
        return (average, np.sqrt(variance))

    def smoothness_score(self):
        """
        Compute smoothness score for the flow. This measures how much the flow is non-linear in between neares neighbours.
        """
        # check if nearest neighbours are already initialized:
        if 'chain_nearest_index' not in self.__dict__:
            self._init_nearest_samples()
        # get delta log likes and delta params:
        delta_theta = self.chain_samples - self.chain_samples[self.chain_nearest_index[:, 1], :]
        delta_log_likes = -(self.chain_loglikes - self.chain_loglikes[self.chain_nearest_index[:, 1]])
        # compute the gradient:
        delta_1 = np.einsum(
            "...i, ...i -> ...", tu.to_numpy(self.log_probability_jacobian(self.chain_samples)),
            delta_theta) - delta_log_likes
        delta_2 = np.einsum(
            "...i, ...i -> ...",
            tu.to_numpy(self.log_probability_jacobian(self.chain_samples[self.chain_nearest_index[:, 1], :])),
            delta_theta) - delta_log_likes
        # average:
        score = np.average(np.abs(0.5 * (delta_1 + delta_2)), weights=self.chain_weights)
        #
        return score

    ###############################################################################
    # Information geometry base methods:

    def _map_to_abstract_coord(self, x):
        """Map from parameter space to abstract space on the flow device."""
        return self.bijector.inverse(x)

    def _map_to_original_coord(self, z):
        """Map from abstract space to parameter space on the flow device."""
        return self.bijector.forward(z)

    def _log_det_metric(self, x):
        """Log determinant of the metric on the flow device."""
        return 2. * self.bijector.inverse_log_det_jacobian(x, event_ndims=1)

    def _direct_jacobian(self, x, create_graph=True):
        """Jacobian of the map to original coordinates at ``x``."""
        abs_coord = self._map_to_abstract_coord(x)
        if not abs_coord.requires_grad:
            abs_coord = abs_coord.detach().requires_grad_(True)
        return autodiff.batch_jacobian(self._map_to_original_coord, abs_coord, create_graph=create_graph)

    def _inverse_jacobian(self, x, create_graph=True):
        """Jacobian of the map to abstract coordinates at ``x``."""
        return autodiff.batch_jacobian(self._map_to_abstract_coord, x, create_graph=create_graph)

    def _inverse_jacobian_coord_derivative(self, x, create_graph=True):
        """Coordinate derivative of the inverse Jacobian."""
        return autodiff.batch_jacobian(
            lambda _x: self._inverse_jacobian(_x, create_graph=True), x, create_graph=create_graph)

    def _metric(self, x, create_graph=True):
        """Metric ``J^T J`` with ``J`` the inverse Jacobian."""
        jac = self._inverse_jacobian(x, create_graph=create_graph)
        return jac.transpose(-1, -2) @ jac

    def _inverse_metric(self, x, create_graph=True):
        """Inverse metric ``J J^T`` with ``J`` the direct Jacobian."""
        jac = self._direct_jacobian(x, create_graph=create_graph)
        return jac @ jac.transpose(-1, -2)

    def _coord_metric_derivative(self, x, create_graph=True):
        """First coordinate derivative of the metric."""
        return autodiff.batch_jacobian(lambda _x: self._metric(_x, create_graph=True), x, create_graph=create_graph)

    def _coord_inverse_metric_derivative(self, x, create_graph=True):
        """First coordinate derivative of the inverse metric."""
        return autodiff.batch_jacobian(
            lambda _x: self._inverse_metric(_x, create_graph=True), x, create_graph=create_graph)

    def _coord_metric_derivative_2(self, x, create_graph=True):
        """Second coordinate derivative of the metric."""
        return autodiff.batch_jacobian(
            lambda _x: self._coord_metric_derivative(_x, create_graph=True), x, create_graph=create_graph)

    def _coord_inverse_metric_derivative_2(self, x, create_graph=True):
        """Second coordinate derivative of the inverse metric."""
        return autodiff.batch_jacobian(
            lambda _x: self._coord_inverse_metric_derivative(_x, create_graph=True), x, create_graph=create_graph)

    def _levi_civita_connection(self, x, create_graph=True):
        """Levi-Civita connection ``Gamma^i_jk``."""
        inv_metric = self._inverse_metric(x, create_graph=create_graph)
        metric_derivative = self._coord_metric_derivative(x, create_graph=create_graph)
        rank = metric_derivative.dim()
        leading = list(range(rank - 3))
        term_1 = metric_derivative.permute(*(leading + [rank - 2, rank - 3, rank - 1]))
        term_2 = metric_derivative.permute(*(leading + [rank - 2, rank - 1, rank - 3]))
        term_3 = metric_derivative.permute(*(leading + [rank - 1, rank - 3, rank - 2]))
        return 0.5 * torch.einsum("...ij,...jkl->...ikl", inv_metric, term_1 + term_2 - term_3)

    def map_to_abstract_coord(self, coord):
        """
        Map from parameter space to abstract space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: abstract coordinates, shape ``(N, D)``
        """
        return self._evaluate(self._map_to_abstract_coord, coord)

    def map_to_original_coord(self, coord):
        """
        Map from abstract space to parameter space

        :param coord: input abstract coordinates, shape ``(N, D)``
        :returns: parameter values, shape ``(N, D)``
        """
        return self._evaluate(self._map_to_original_coord, coord)

    def log_det_metric(self, coord):
        """
        Computes the log determinant of the metric

        :param coord: input parameter value, shape ``(N, D)``
        :returns: log determinant, shape ``(N,)``
        """
        return self._evaluate(self._log_det_metric, coord)

    def direct_jacobian(self, coord):
        """
        Computes the Jacobian of the parameter transformation at one point in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: Jacobian of the map from abstract to parameter space, shape ``(N, D, D)``
        """
        return self._differentiate(self._direct_jacobian, coord)

    def inverse_jacobian(self, coord):
        """
        Computes the inverse Jacobian of the parameter transformation at one point in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: Jacobian of the map from parameter to abstract space, shape ``(N, D, D)``
        """
        return self._differentiate(self._inverse_jacobian, coord)

    def inverse_jacobian_coord_derivative(self, coord):
        """
        Compute the coordinate derivative of the inverse Jacobian at a given point in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: derivative, shape ``(N, D, D, D)``, the last index being the derivative one
        """
        return self._differentiate(self._inverse_jacobian_coord_derivative, coord)

    def metric(self, coord):
        """
        Computes the metric at a given point or array of points in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: metric, shape ``(N, D, D)``
        """
        return self._differentiate(self._metric, coord)

    def inverse_metric(self, coord):
        """
        Computes the inverse metric at a given point or array of points in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: inverse metric, shape ``(N, D, D)``
        """
        return self._differentiate(self._inverse_metric, coord)

    def coord_metric_derivative(self, coord):
        """
        Compute the coordinate derivative of the metric at a given point in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: derivative, shape ``(N, D, D, D)``, the last index being the derivative one
        """
        return self._differentiate(self._coord_metric_derivative, coord)

    def coord_inverse_metric_derivative(self, coord):
        """
        Compute the coordinate derivative of the inverse metric at a given point in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: derivative, shape ``(N, D, D, D)``, the last index being the derivative one
        """
        return self._differentiate(self._coord_inverse_metric_derivative, coord)

    def coord_metric_derivative_2(self, coord):
        """
        Compute the second coordinate derivative of the metric at a given point in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: second derivative, shape ``(N, D, D, D, D)``, the last two indexes being the derivative ones
        """
        return self._differentiate(self._coord_metric_derivative_2, coord)

    def coord_inverse_metric_derivative_2(self, coord):
        """
        Compute the second coordinate derivative of the inverse metric at a given point in (original) parameter space

        :param coord: input parameter value, shape ``(N, D)``
        :returns: second derivative, shape ``(N, D, D, D, D)``, the last two indexes being the derivative ones
        """
        return self._differentiate(self._coord_inverse_metric_derivative_2, coord)

    def levi_civita_connection(self, coord):
        """
        Compute the Levi-Civita connection, gives Gamma^i_j_k

        :param coord: input parameter value, shape ``(N, D)``
        :returns: connection coefficients, shape ``(N, D, D, D)`` with indexes ``(i, j, k)``
        """
        return self._differentiate(self._levi_civita_connection, coord)

    def geodesic_distance(self, coord_1, coord_2, **kwargs):
        """
        Compute geodesic distance between pair of points.

        :param coord_1: first points.
        :param coord_2: second points.
        :param kwargs: options of ``torch.linalg.norm`` (``axis`` and ``keepdims`` are accepted
            as aliases of ``dim`` and ``keepdim``). Without ``dim`` the norm is taken over all
            the elements, as in the TensorFlow version.
        :returns: CPU tensor with the distance(s).
        """
        if 'axis' in kwargs:
            kwargs['dim'] = kwargs.pop('axis')
        if 'keepdims' in kwargs:
            kwargs['keepdim'] = kwargs.pop('keepdims')
        x_1, needs_graph_1 = self._input(coord_1)
        x_2, needs_graph_2 = self._input(coord_2)
        needs_graph = needs_graph_1 or needs_graph_2
        with torch.set_grad_enabled(needs_graph):
            # map to abstract coordinates:
            abs_coord_1 = self._map_to_abstract_coord(x_1)
            abs_coord_2 = self._map_to_abstract_coord(x_2)
            # metric there is Euclidean:
            result = torch.linalg.norm(abs_coord_1 - abs_coord_2, **kwargs)
        return self._output(result, needs_graph)

    def geodesic_bvp(self, pos_start, pos_end, num_points=1000):
        """
        Solve geodesic boundary value problem.

        :param pos_start: initial points ``(N, D)``.
        :param pos_end: final points ``(N, D)``.
        :param num_points: number of points along each geodesic.
        :returns: CPU tensor ``(N, num_points, D)``.
        """
        x_start, needs_graph_1 = self._input(pos_start)
        x_end, needs_graph_2 = self._input(pos_end)
        needs_graph = needs_graph_1 or needs_graph_2
        with torch.set_grad_enabled(needs_graph):
            # map initial and final positions to abstract space:
            _abs_pos_start = self._map_to_abstract_coord(x_start)
            _abs_pos_end = self._map_to_abstract_coord(x_end)
            # get the affine parameter along the geodesic:
            _alpha = torch.linspace(0.0, 1.0, num_points, dtype=_abs_pos_start.dtype, device=_abs_pos_start.device)
            # get the trajectory (a straight line) in abstract space:
            _traj = _abs_pos_start.unsqueeze(-1) + _alpha * (_abs_pos_end.unsqueeze(-1) - _abs_pos_start.unsqueeze(-1))
            _traj = _traj.transpose(-1, -2)
            # return map to parameter space:
            result = self._map_to_original_coord(_traj)
        return self._output(result, needs_graph)

    def geodesic_ivp(self, pos, velocity, solution_times):
        """
        Solve geodesic initial value problem.

        :param pos: initial points (unused).
        :param velocity: initial velocities (unused).
        :param solution_times: times at which the solution is required (unused).
        :raises NotImplementedError: always, not implemented yet.
        """
        raise NotImplementedError('geodesic_ivp is not implemented.')

    ###############################################################################
    # device handling:

    def to(self, device):
        """
        Move the flow to a device. The optimizer is re-created, so its moments are reset.

        :param device: device specification (``'cpu'``, ``'cuda'``, ``'cuda:N'``, ``'mps'``).
        :returns: the flow itself.
        :raises ValueError: if the device is not available or does not support the precision.
        """
        device = tu.check_device(device)
        self._move_to(device)
        return self

    def _move_to(self, device):
        """Move bijectors, distributions, data tensors and the trainer to ``device``."""
        self.bijector.to(device)
        self.device = device
        _transformation = self.__dict__.get('trainable_transformation', None)
        if _transformation is not None and hasattr(_transformation, 'device'):
            _transformation.device = device
        self._build_distributions()
        if not self.is_light and 'training_samples' in self.__dict__:
            self._init_dataset_tensors()
            if self.__dict__.get('_trainer_initialized', False):
                self.trainer.log_prob_fn = self.trained_distribution.log_prob
                self.trainer.reset_optimizer()
        #
        return None

    ###############################################################################
    # caching methods:

    def save(self, path, mode='full'):
        """
        Save the whole flow to one file, from which :meth:`load` restores it without the chain
        and without rebuilding anything.

        Two modes are available:

        - ``'full'`` (default): everything, including the chain samples, the training split,
          the loss and the optimizer state, so that training can be resumed;
        - ``'light'``: only what is needed to evaluate the flow (bijectors, distributions,
          parameter names and ranges, MAP estimates, training logs). The file is much
          smaller but the flow cannot be trained again, and methods that need the chain
          (``train``, ``global_train``, ``evidence``, ``smoothness_score``,
          ``compute_training_metrics``, ``training_plot``) fail with an explanatory error.

        The file is a pickle (written with ``torch.save``): it is tied to the package version
        that wrote it and must only be loaded from trusted sources. Everything stored in the
        flow must be picklable: user functions given to ``Inline`` bijectors must be module
        level functions, not lambdas or closures.

        :param path: file path.
        :param mode: ``'full'`` or ``'light'``.
        :raises ValueError: for an unknown mode, or ``mode='full'`` on a light flow.
        :raises pickle.PicklingError: if part of the flow cannot be pickled.
        """
        global _save_mode
        if mode not in ('full', 'light'):
            raise ValueError("mode must be 'full' or 'light', got " + repr(mode))
        if mode == 'full' and self.is_light:
            raise ValueError('A light flow cannot be saved in full mode: the chain data is not available.')
        self._snapshot_info = {
            'format_version': SNAPSHOT_FORMAT_VERSION,
            'tensiometer_version': _tensiometer_version,
            'torch_version': torch.__version__,
            'precision': _precision_name(self.prec),
            'device': str(self.device),
            'class': type(self).__name__,
            'mode': mode,
        }
        previous_mode = _save_mode
        _save_mode = mode
        try:
            tu.atomic_save(self, path)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise pickle.PicklingError(
                'The flow could not be pickled: ' + str(exc) + '. Everything stored in the flow must be '
                'picklable; functions given to Inline bijectors, AnalyticalDerivedParamsBijector or '
                'TransformedFlowCallback must be module level functions, not lambdas or closures.') from exc
        finally:
            _save_mode = previous_mode
        #
        return None

    @classmethod
    def load(cls, path, device=None):
        """
        Load a flow saved with :meth:`save`. No chain is needed and nothing is rebuilt.

        Loading executes pickle code: only load files from trusted sources. Files written by
        the TensorFlow version of tensiometer cannot be loaded (retrain the flow).

        :param path: file path.
        :param device: target device, defaults to the default device. Flows trained on a GPU
            can be loaded on a CPU-only machine and vice versa.
        :returns: the flow.
        :raises FileNotFoundError: if the file does not exist.
        :raises ValueError: if the file is not a flow snapshot, was written by a newer
            snapshot format, or has a precision different from a locked one.
        :raises TypeError: if the snapshot holds a flow that is not an instance of ``cls``.
        """
        target = tu.check_device(device)
        if not os.path.isfile(path):
            raise FileNotFoundError('Flow snapshot not found: ' + str(path))
        try:
            flow = torch.load(path, map_location=target, weights_only=False)
        except Exception as exc:
            raise ValueError(
                'Could not load a flow snapshot from ' + str(path) + ' (' + type(exc).__name__ + ': '
                + str(exc) + '). Caches written by the TensorFlow version of tensiometer cannot be '
                'loaded; retrain the flow.') from exc
        info = getattr(flow, '_snapshot_info', None) if isinstance(flow, FlowCallback) else None
        if info is None:
            raise ValueError(str(path) + ' is not a flow snapshot written by FlowCallback.save. '
                             'Caches written by the TensorFlow version of tensiometer cannot be loaded; retrain the flow.')
        if info.get('format_version', 0) > SNAPSHOT_FORMAT_VERSION:
            raise ValueError('The snapshot ' + str(path) + ' has format version ' + str(info.get('format_version'))
                             + ', newer than the supported version ' + str(SNAPSHOT_FORMAT_VERSION)
                             + '. Update tensiometer.')
        if not isinstance(flow, cls):
            raise TypeError('The snapshot ' + str(path) + ' contains a ' + type(flow).__name__
                            + ', not a ' + cls.__name__)
        if info.get('tensiometer_version') != _tensiometer_version:
            warnings.warn('The snapshot ' + str(path) + ' was written by tensiometer '
                          + str(info.get('tensiometer_version')) + ', this is ' + _tensiometer_version)
        # precision:
        saved_precision = tu._parse_precision(info.get('precision', 'float32'))
        if saved_precision != tu.get_precision():
            if tu.is_precision_locked():
                raise ValueError('The snapshot ' + str(path) + ' uses ' + _precision_name(saved_precision)
                                 + ' but the precision is locked to ' + _precision_name(tu.get_precision()) + '.')
            tu.set_precision(saved_precision)
        tu.lock_precision()
        tu._check_device_precision(target, saved_precision)
        #
        return flow

    def _upgrade_state(self, state):
        """
        Fill defaults for attributes added after the snapshot was written.

        :param state: unpickled state dictionary.
        :returns: the upgraded state dictionary.
        """
        return state

    @staticmethod
    def _state_device(state):
        """Device of the tensors of an unpickled state."""
        bijector = state.get('bijector', None)
        if isinstance(bijector, torch.nn.Module):
            device = tu.module_device(bijector)
            if device is not None:
                return device
        flows = state.get('flows', None)
        if flows:
            return flows[0].device
        base = state.get('base_distribution', None)
        if base is not None and hasattr(base, 'mean'):
            return base.mean.device
        return state.get('device', tu.get_device())

    def __getstate__(self):
        """
        State to pickle: drops the figure and the training tensors (re-created on load) and,
        when writing a light snapshot, the attributes in ``_full_only_attributes``.
        """
        state = self.__dict__.copy()
        for key in ('fig', 'training_dataset', 'validation_dataset'):
            state.pop(key, None)
        if _save_mode == 'light':
            for key in type(self)._full_only_attributes:
                state.pop(key, None)
            state['is_light'] = True
            state['_trainer_initialized'] = False
            if '_snapshot_info' in state:
                state['_snapshot_info'] = dict(state['_snapshot_info'], mode='light')
        return state

    def __setstate__(self, state):
        """
        Restore a pickled state and re-create the training tensors on the device of the
        restored bijectors.
        """
        state = self._upgrade_state(state)
        self.__dict__.update(state)
        self.device = self._state_device(state)
        if not self.__dict__.get('is_light', False) and 'training_samples' in state:
            self._init_dataset_tensors()

    def __getattr__(self, name):
        """
        Explain missing attributes of light flows.

        :raises AttributeError: always (only called for missing attributes).
        """
        _dict = object.__getattribute__(self, '__dict__')
        if _dict.get('is_light', False) and name in type(self)._full_only_attributes:
            raise AttributeError(
                "'" + name + "' is not stored in a light flow snapshot; load a full snapshot or "
                "rebuild the flow from the chain")
        raise AttributeError("'" + type(self).__name__ + "' object has no attribute '" + name + "'")

    ###############################################################################
    # Training statistics:

    def _init_training_monitoring(self):
        """
        Initialize training monitoring quantities and logs
        """
        # training metrics:
        if issubclass(type(self.loss), loss.standard_loss):
            self.training_metrics = ["loss", "val_loss", "lr", "chi2Z_ks", "chi2Z_ks_p", "loss_rate", "val_loss_rate"]
        elif issubclass(type(self.loss), loss.constant_weight_loss):
            self.training_metrics = [
                "loss",
                "val_loss",
                "lr",
                # loss breakdown:
                "rho_loss",
                "ee_loss",
                "val_rho_loss",
                "val_ee_loss",
                # loss improvement rate:
                "loss_rate",
                "rho_loss_rate",
                "ee_loss_rate",
                # KS test:
                "chi2Z_ks",
                "chi2Z_ks_p",
                # evidence estimates:
                "training_evidence",
                "training_evidence_error",
                "test_evidence",
                "test_evidence_error",
                "evidence",
                "evidence_error",
            ]
        elif issubclass(type(self.loss), loss.variable_weight_loss):
            self.training_metrics = [
                "loss",
                "val_loss",
                "lr",
                # loss breakdown:
                "rho_loss",
                "ee_loss",
                "val_rho_loss",
                "val_ee_loss",
                # loss improvement rate:
                "loss_rate",
                "rho_loss_rate",
                "ee_loss_rate",
                # KS test:
                "chi2Z_ks",
                "chi2Z_ks_p",
                # evidence estimates:
                "training_evidence",
                "training_evidence_error",
                "test_evidence",
                "test_evidence_error",
                "evidence",
                "evidence_error",
                # moo coefficients:
                "lambda_1",
                "lambda_2",
            ]

        # initialize logs:
        self.log = {_k: [] for _k in self.training_metrics}

        # compute initial chi2:
        if self.num_test_samples < 1:
            self.chi2Y = np.array([])
            self.chi2Y_ks, self.chi2Y_ks_p = 0.0, 0.0
            return None

        _temp_mean = np.average(self.test_samples, axis=0, weights=self.test_weights)
        if self.num_test_samples < 2:
            _temp_invcov = np.eye(self.num_params)
        else:
            try:
                _temp_cov = np.cov(self.test_samples.T, aweights=self.test_weights)
                if not np.all(np.isfinite(_temp_cov)):
                    raise ValueError('Non-finite covariance for chi2 statistics')
                _temp_invcov = np.linalg.inv(scipy.linalg.sqrtm(_temp_cov))
            except Exception:
                _temp_invcov = np.eye(self.num_params)
        _temp = np.dot(_temp_invcov, (self.test_samples - _temp_mean).T)
        self.chi2Y = np.sum((_temp)**2, axis=0)
        if len(self.chi2Y) > 0 and self.has_weights:
            _weight_sum = np.sum(self.test_weights)
            if np.isfinite(_weight_sum) and _weight_sum > 0:
                self.chi2Y = np.random.choice(
                    self.chi2Y, size=len(self.chi2Y), replace=True, p=self.test_weights / _weight_sum)
            else:
                self.chi2Y = np.random.choice(self.chi2Y, size=len(self.chi2Y), replace=True)
        self.chi2Y_ks, self.chi2Y_ks_p = scipy.stats.kstest(self.chi2Y, 'chi2', args=(self.num_params,))
        #
        return None

    def _trainable_inverse(self, samples):
        """
        Map samples of the training space to the abstract space, as a numpy array.

        :param samples: numpy array ``(N, D)``.
        :returns: numpy array ``(N, D)``.
        """
        with torch.no_grad():
            return tu.to_numpy(self.trainable_bijector.inverse(tu.to_tensor(samples, device=self.device)))

    def _loss_components(self, logP, samples, weights):
        """
        Loss components on a whole data set, with numpy arrays for the per-sample components.

        :param logP: true log posterior in the training space.
        :param samples: samples in the training space.
        :param weights: sample weights.
        :returns: tuple of loss components.
        """
        with torch.no_grad():
            _logP = tu.to_tensor(logP, device=self.device)
            _pred = self.trained_distribution.log_prob(tu.to_tensor(samples, device=self.device))
            _weights = tu.to_tensor(weights, device=self.device)
            components = self.loss.compute_loss_components(_logP, _pred, _weights)
        return tuple(tu.to_numpy(_c) if torch.is_tensor(_c) else _c for _c in components)

    def compute_training_metrics(self, logs=None):
        """
        Compute training metrics and append results to internal logs

        :param logs: dictionary of training metrics for the epoch, from which ``loss``,
            ``val_loss`` and ``lr`` are read.
        :returns: None
        """
        if logs is None:
            logs = {}
        # update loss log:
        if "loss" in self.training_metrics:
            self.log["loss"].append(logs.get('loss'))
        if "val_loss" in self.training_metrics:
            _val_loss = logs.get('val_loss')
            self.log["val_loss"].append(np.nan if _val_loss is None else _val_loss)

        # update learning rate log:
        if "lr" in self.training_metrics:
            self.log["lr"].append(logs.get('lr'))

        # do KS test:
        if "chi2Z_ks" in self.training_metrics:
            self.chi2Z = np.sum(self._trainable_inverse(self.test_samples)**2, axis=1)
            # Run KS test
            try:
                # Note that scipy.stats.kstest does not handle weights yet so we need to resample.
                _s = np.isfinite(self.chi2Z)
                self.chi2Z = self.chi2Z[_s]
                if len(self.chi2Z) == 0:
                    chi2Z_ks, chi2Z_ks_p = 0.0, 0.0
                else:
                    if self.has_weights:
                        _weights = self.test_weights[_s]
                        _weight_sum = np.sum(_weights)
                        if np.isfinite(_weight_sum) and _weight_sum > 0:
                            self.chi2Z = np.random.choice(
                                self.chi2Z,
                                size=len(self.chi2Z),
                                replace=True,
                                p=_weights / _weight_sum)
                        else:
                            self.chi2Z = np.random.choice(
                                self.chi2Z,
                                size=len(self.chi2Z),
                                replace=True)
                    chi2Z_ks, chi2Z_ks_p = scipy.stats.kstest(self.chi2Z, 'chi2', args=(self.num_params,))
            except:
                chi2Z_ks, chi2Z_ks_p = 0., 0.
            self.log["chi2Z_ks"].append(chi2Z_ks)
        if "chi2Z_ks_p" in self.training_metrics:
            self.log["chi2Z_ks_p"].append(chi2Z_ks_p)

        # evidence:
        if "evidence" in self.training_metrics:
            evidence, evidence_error = self.evidence()
            self.log["evidence"].append(evidence)
        if "evidence_error" in self.training_metrics:
            self.log["evidence_error"].append(evidence_error)
        # compute evidence on training samples:
        if "training_evidence" in self.training_metrics:
            training_evidence, training_evidence_error = self.evidence(indexes=self.training_idx)
            self.log["training_evidence"].append(training_evidence)
        if "training_evidence_error" in self.training_metrics:
            self.log["training_evidence_error"].append(training_evidence_error)
        # compute evidence on validation samples:
        if "test_evidence" in self.training_metrics:
            test_evidence, test_evidence_error = self.evidence(indexes=self.test_idx)
            self.log["test_evidence"].append(test_evidence)
        if "test_evidence_error" in self.training_metrics:
            self.log["test_evidence_error"].append(test_evidence_error)

        # compute density loss on validation data:
        if "rho_loss" in self.training_metrics:
            _train_loss_components = self._loss_components(
                self.training_logP_preabs, self.training_samples, self.training_weights)
            _test_loss_components = self._loss_components(
                self.test_logP_preabs, self.test_samples, self.test_weights)
            if issubclass(type(self.loss), loss.constant_weight_loss):
                # average:
                temp_train_rho_loss = np.average(_train_loss_components[0], weights=self.training_weights)
                temp_train_ee_loss = np.average(_train_loss_components[1], weights=self.training_weights)
                temp_val_rho_loss = np.average(_test_loss_components[0], weights=self.test_weights)
                temp_val_ee_loss = np.average(_test_loss_components[1], weights=self.test_weights)
                # add to log:
                self.log["rho_loss"].append(temp_train_rho_loss)
                self.log["ee_loss"].append(temp_train_ee_loss)
                self.log["val_rho_loss"].append(temp_val_rho_loss)
                self.log["val_ee_loss"].append(temp_val_ee_loss)
            if issubclass(type(self.loss), loss.variable_weight_loss):
                # average:
                temp_train_rho_loss = np.average(_train_loss_components[0], weights=self.training_weights)
                temp_train_ee_loss = np.average(_train_loss_components[1], weights=self.training_weights)
                temp_val_rho_loss = np.average(_test_loss_components[0], weights=self.test_weights)
                temp_val_ee_loss = np.average(_test_loss_components[1], weights=self.test_weights)
                # add to log:
                self.log["lambda_1"].append(_test_loss_components[2])
                self.log["lambda_2"].append(_test_loss_components[3])
                self.log["rho_loss"].append(temp_train_rho_loss)
                self.log["ee_loss"].append(temp_train_ee_loss)
                self.log["val_rho_loss"].append(temp_val_rho_loss)
                self.log["val_ee_loss"].append(temp_val_ee_loss)

        # loss rate:
        if "loss_rate" in self.training_metrics:
            if len(self.log["loss"]) < 2:
                self.log["loss_rate"].append(0.0)
            else:
                self.log["loss_rate"].append(self.log["loss"][-1] - self.log["loss"][-2])
        if "val_loss_rate" in self.training_metrics:
            if len(self.log["val_loss"]) < 2:
                self.log["val_loss_rate"].append(0.0)
            else:
                self.log["val_loss_rate"].append(self.log["val_loss"][-1] - self.log["val_loss"][-2])
        if "rho_loss_rate" in self.training_metrics:
            if len(self.log["rho_loss"]) < 2:
                self.log["rho_loss_rate"].append(0.0)
                self.log["ee_loss_rate"].append(0.0)
            else:
                self.log["rho_loss_rate"].append(self.log["rho_loss"][-1] - self.log["rho_loss"][-2])
                self.log["ee_loss_rate"].append(self.log["ee_loss"][-1] - self.log["ee_loss"][-2])

    @matplotlib.rc_context(plot_options)
    def _plot_loss(self, ax, logs=None):
        """
        Utility function to plot loss for training and validation samples
        """
        # plot loss lines:
        ax.plot(self.log["loss"], ls='-', lw=1., color='k', label='training')
        ax.plot(self.log["val_loss"], ls='--', lw=1., color='k', label='validation')
        # plot best population loss so far (if any):
        if 'best_loss' in self.log.keys():
            ax.axhline(self.log['best_loss'], ls='--', lw=1., color='tab:blue', label='pop best')
        # finish plot:
        ax.set_title("Loss function")
        ax.set_xlabel(r"Epoch $\#$")
        # log scale if positive:
        if self.log["loss"][-1] > 0.0:
            ax.set_yscale('log')
        else:
            ax.set_yscale('symlog', linthresh=1.e-3, linscale=0.5)
            ax.axhline(0.0, ls=':', lw=1., color='k')
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_lr(self, ax, logs=None):
        """
        Utility function to plot learning rate per epoch
        """
        ax.plot(self.log["lr"], ls='-', lw=1.)
        ax.set_ylim([0.8 * self.final_learning_rate, 1.2 * self.initial_learning_rate])
        ax.set_title("Learning rate")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_yscale('log')
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_chi2_dist(self, ax, logs=None, fast=False):
        """
        Utility function to plot chi2 distribution vs histogram.
        """
        xx = np.linspace(0, self.num_params * 4, 1000)
        bins = np.linspace(0, self.num_params * 4, 100)
        ax.plot(
            xx,
            scipy.stats.chi2.pdf(xx, df=self.num_params),
            label=r'$\chi^2_{{{}}}$ PDF'.format(self.num_params),
            c='k',
            lw=1.,
            ls='-')
        chi2y_vals = np.asarray(self.chi2Y)
        chi2y_mask = np.isfinite(chi2y_vals)
        chi2y_vals = chi2y_vals[chi2y_mask]
        chi2y_weights = None
        if chi2y_mask.size == self.test_weights.size:
            chi2y_weights = self.test_weights[chi2y_mask]
        if len(chi2y_vals) > 0:
            ax.hist(
                chi2y_vals,
                bins=bins,
                density=True,
                histtype='step',
                weights=chi2y_weights,
                label='Pre-NF ($D_n$={:.3f})'.format(self.chi2Y_ks),
                lw=1.,
                ls='-')
        chi2z_vals = np.asarray(self.chi2Z)
        chi2z_vals = chi2z_vals[np.isfinite(chi2z_vals)]
        if len(chi2z_vals) > 0:
            ax.hist(
                chi2z_vals,
                bins=bins,
                density=True,
                histtype='step',
                label='Post-NF val ($D_n$={:.3f})'.format(self.log["chi2Z_ks"][-1]),
                lw=1.,
                ls='-')
        if not fast:
            train_chi2Z = np.sum(self._trainable_inverse(self.training_samples)**2, axis=1)
            train_chi2Z = train_chi2Z[np.isfinite(train_chi2Z)]
            if len(train_chi2Z) > 0:
                ax.hist(
                    train_chi2Z,
                    bins=bins,
                    density=True,
                    histtype='step',
                    label='Post-NF train',
                    lw=1.,
                    ls='-')
        ax.set_title(r'$\chi^2_{{{}}}$ PDF'.format(self.num_params))
        ax.set_xlabel(r'$\chi^2$')
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_chi2_ks_p(self, ax, logs=None):
        """
        Utility function to plot the KS test results.
        """
        # KS result probability:
        ln1 = ax.plot(self.log["chi2Z_ks_p"], label='$p$', lw=1., ls='-', color='tab:blue')
        ax.set_title(r"KS test ($\chi^2$)")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_ylabel(r"$p$-value")
        # KL difference:
        ax2 = ax.twinx()
        ln2 = ax2.plot(self.log["chi2Z_ks"], label='$D_n$', lw=1., ls='--', color='tab:orange')
        ax2.set_ylabel(r'$D_n$')
        # legend:
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs)
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_density_evidence_error_losses(self, ax, logs=None):
        """
        Plot behavior of density and evidence-error loss as training progresses.
        """
        ax.plot(np.abs(self.log["rho_loss"]), lw=1., ls='-', color='tab:blue')
        ax.plot(np.abs(self.log["ee_loss"]), lw=1., ls='-', color='tab:orange')
        ax.plot(np.abs(self.log["val_rho_loss"]), lw=1., ls='--', color='tab:blue', label='density')
        ax.plot(np.abs(self.log["val_ee_loss"]), lw=1., ls='--', color='tab:orange', label='evidence error')
        ax.set_title(r"Loss breakdown")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_yscale('log')
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_lambda_values(self, ax, logs=None):
        """
        Plot balance between the two loss functions
        """
        ax.plot(np.abs(self.log["lambda_1"]), lw=1., ls='-', label=r'$\lambda_1$')
        ax.plot(np.abs(self.log["lambda_2"]), lw=1., ls='--', label=r'$\lambda_2$')
        ax.set_title(r"Loss function weights")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_ylim([-0.1, 1.1])
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_weighted_density_evidence_error_losses(self, ax, logs=None):
        """
        Plot behavior of density and evidence error loss as training progresses.
        """
        ax.plot(
            np.abs(np.array(self.log["lambda_1"]) * np.array(self.log["rho_loss"])), lw=1., ls='-', color='tab:blue')
        ax.plot(
            np.abs(np.array(self.log["lambda_2"]) * np.array(self.log["ee_loss"])), lw=1., ls='-', color='tab:orange')
        ax.plot(
            np.abs(np.array(self.log["lambda_1"]) * np.array(self.log["val_rho_loss"])),
            lw=1.,
            ls='--',
            color='tab:blue',
            label='density')
        ax.plot(
            np.abs(np.array(self.log["lambda_2"]) * np.array(self.log["val_ee_loss"])),
            lw=1.,
            ls='--',
            color='tab:orange',
            label='evidence error')
        ax.set_title(r"Wighted loss breakdown")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_yscale('log')
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_losses_rate(self, ax, logs=None, abs_value=False, epoch_range=20):
        """
        Plot evolution of loss function.
        """
        if abs_value:
            if issubclass(type(self.loss), loss.standard_loss):
                ax.plot(np.abs(self.log["loss_rate"]), lw=1., ls='-', label='training')
                ax.plot(np.abs(self.log["val_loss_rate"]), lw=1., ls='-', label='validation')
            elif issubclass(type(self.loss), loss.constant_weight_loss):
                ax.plot(np.abs(self.log["loss_rate"]), lw=1.2, color='k', ls='-', label='all', alpha=0.5, zorder=2)
                ax.plot(np.abs(self.log["rho_loss_rate"]), lw=1., ls='-', label='density', zorder=1)
                ax.plot(np.abs(self.log["ee_loss_rate"]), lw=1., ls='-', label='evidence error', zorder=0)
            elif issubclass(type(self.loss), loss.variable_weight_loss):
                ax.plot(np.abs(self.log["loss_rate"]), lw=1.2, color='k', ls='-', label='all', zorder=2)
                ax.plot(np.abs(self.log["rho_loss_rate"]), lw=1., ls='-', label='density', zorder=1)
                ax.plot(np.abs(self.log["ee_loss_rate"]), lw=1., ls='-', label='evidence error', zorder=0)
        else:
            if issubclass(type(self.loss), loss.standard_loss):
                ax.plot(self.log["loss_rate"], lw=1., ls='-', label='training')
                ax.plot(self.log["val_loss_rate"], lw=1., ls='-', label='validation')
            elif issubclass(type(self.loss), loss.constant_weight_loss):
                ax.plot(self.log["loss_rate"], lw=1.2, color='k', ls='-', label='all', alpha=0.5, zorder=2)
                ax.plot(self.log["rho_loss_rate"], lw=1., ls='-', label='density', zorder=1)
                ax.plot(self.log["ee_loss_rate"], lw=1., ls='-', label='evidence error', zorder=0)
            elif issubclass(type(self.loss), loss.variable_weight_loss):
                ax.plot(self.log["loss_rate"], lw=1.2, color='k', ls='-', label='all', alpha=0.5, zorder=2)
                ax.plot(self.log["rho_loss_rate"], lw=1., ls='-', label='density', zorder=1)
                ax.plot(self.log["ee_loss_rate"], lw=1., ls='-', label='evidence error', zorder=0)
        if abs_value:
            ax.set_yscale('log')
        else:
            # plot horizontal line at zero:
            ax.axhline(0., lw=1.0, ls='--', color='k')
            ## calculate variance of loss rate for last 30 epochs:
            #if 'val_loss_rate' in self.log.keys():
            #    if len(self.log["val_loss_rate"]) > epoch_range:
            #        loss_rate_sig = np.sqrt(np.var(self.log["val_loss_rate"][-epoch_range:]))
            #        ax.set_ylim([-3*loss_rate_sig, 3*loss_rate_sig])
            #elif 'loss_rate' in self.log.keys():
            #    if len(self.log["loss_rate"]) > epoch_range:
            #        loss_rate_sig = np.sqrt(np.var(self.log["loss_rate"][-epoch_range:]))
            #        ax.set_ylim([-3*loss_rate_sig, 3*loss_rate_sig])
            #else:
            #    ax.set_ylim([-1, 1])

            # calculate variance of loss rate for last 30 epochs:
            loss_rate_sig = 0.1
            if 'val_loss_rate' in self.log.keys():
                if len(self.log["val_loss_rate"]) > epoch_range:
                    loss_rate_sig = np.sqrt(np.var(self.log["val_loss_rate"][-epoch_range:]))
            elif 'loss_rate' in self.log.keys():
                if len(self.log["loss_rate"]) > epoch_range:
                    loss_rate_sig = np.sqrt(np.var(self.log["loss_rate"][-epoch_range:]))
            ax.set_yscale('symlog', linthresh=0.5*loss_rate_sig, linscale=0.5)
        ax.set_title(r"$\Delta$ Loss / epoch")
        ax.set_xlabel(r"Epoch $\#$")
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_evidence(self, ax, logs=None):
        """
        Utility function to plot the evidence and error on evidence as a function of training.
        """
        # evidence:
        ax.plot(np.abs(self.log["evidence"]), lw=1.2, ls='--', color='k', label='all')
        ax.plot(np.abs(self.log["training_evidence"]), lw=1., ls='-', label='training')
        ax.plot(np.abs(self.log["test_evidence"]), lw=1., ls='-', label='validation')
        ax.set_title(r"Flow |evidence|")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_yscale('log')
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_evidence_error(self, ax, logs=None):
        """
        Utility function to plot the evidence and error on evidence as a function of training.
        """
        # evidence error:
        ax.plot(self.log["evidence_error"], lw=1.2, ls='--', color='k', label='all')
        ax.plot(self.log["training_evidence_error"], lw=1., ls='-', label='training')
        ax.plot(self.log["test_evidence_error"], lw=1., ls='-', label='validation')
        ax.set_title(r"Flow evidence error")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_yscale('log')
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _create_figure(self):
        """
        Utility to create figure
        """
        if issubclass(type(self.loss), loss.standard_loss):
            self.fig = plt.figure(figsize=(16, 3))
        elif issubclass(type(self.loss), loss.constant_weight_loss):
            self.fig = plt.figure(figsize=(16, 6))
        elif issubclass(type(self.loss), loss.variable_weight_loss):
            self.fig = plt.figure(figsize=(16, 6))
        #
        return None

    @matplotlib.rc_context(plot_options)
    def on_train_begin(self, logs):
        """
        Execute on beginning of training: creates the figure unless plotting inline.

        :param logs: dictionary of training metrics passed by the training loop (unused).
        :returns: None
        """
        if not ipython_plotting:
            self._create_figure()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def on_train_end(self, logs):
        """
        Execute at end of training: closes the figure unless plotting inline.

        :param logs: dictionary of training metrics passed by the training loop (unused).
        :returns: None
        """
        if not ipython_plotting:
            del self.fig
            plt.close('all')
        #
        return None
    
    @matplotlib.rc_context(plot_options)
    def training_plot(self, logs=None, file_path=None, ipython_plotting=False, title=None, fast=False):
        """
        Method to produce training plot with training metrics

        :param logs: dictionary of training metrics passed to the plotting helpers, defaults to
            None (use ``self.log``). The panels currently plot the internal ``self.log``.
        :param file_path: path of the file to save the figure to, defaults to None (not saved).
            If given the figure is closed after saving.
        :param ipython_plotting: whether plotting is inline in IPython, in which case an existing
            figure is not cleared and re-created, defaults to False.
        :param title: figure title, defaults to None (training population number, if available).
        :param fast: skip the chi2 histogram of the training samples, defaults to False. Only used
            for variable weight losses.
        :returns: None
        """
        # check that self.fig exists:
        if not hasattr(self, 'fig'):
            self._create_figure()
        elif not ipython_plotting:
            plt.clf()
            self._create_figure()
            
        # initialize the log to use:
        if logs is None:
            _logs = self.log
        else:
            _logs = logs
        
        # plot figure:
        if issubclass(type(self.loss), loss.standard_loss):
            gs = self.fig.add_gridspec(nrows=1, ncols=5)
            axes = [self.fig.add_subplot(_g) for _g in gs]
            self._plot_loss(axes[0], logs=_logs)
            self._plot_losses_rate(axes[1], logs=_logs)
            self._plot_lr(axes[2], logs=_logs)
            self._plot_chi2_dist(axes[3], logs=_logs)
            self._plot_chi2_ks_p(axes[4], logs=_logs)
        elif issubclass(type(self.loss), loss.constant_weight_loss):
            gs = self.fig.add_gridspec(nrows=2, ncols=4)
            axes = [self.fig.add_subplot(_g) for _g in gs]
            self._plot_loss(axes[0], logs=_logs)
            self._plot_density_evidence_error_losses(axes[1], logs=_logs)
            self._plot_losses_rate(axes[2], logs=_logs)
            self._plot_lr(axes[3], logs=_logs)
            self._plot_evidence(axes[4], logs=_logs)
            self._plot_evidence_error(axes[5], logs=_logs)
            self._plot_chi2_dist(axes[6], logs=_logs)
            self._plot_chi2_ks_p(axes[7], logs=_logs)
        elif issubclass(type(self.loss), loss.variable_weight_loss):
            gs = self.fig.add_gridspec(nrows=2, ncols=5)
            axes = [self.fig.add_subplot(_g) for _g in gs]
            self._plot_loss(axes[0], logs=_logs)
            self._plot_density_evidence_error_losses(axes[1], logs=_logs)
            self._plot_lambda_values(axes[2], logs=_logs)
            self._plot_weighted_density_evidence_error_losses(axes[3], logs=_logs)
            self._plot_losses_rate(axes[4], logs=_logs)
            self._plot_lr(axes[5], logs=_logs)
            self._plot_evidence(axes[6], logs=_logs)
            self._plot_evidence_error(axes[7], logs=_logs)
            self._plot_chi2_dist(axes[8], logs=_logs, fast=fast)
            self._plot_chi2_ks_p(axes[9], logs=_logs)

        # plot title:
        if title is not None:
            plt.suptitle(title, fontweight='bold')
        else:
            if 'population' in self.log.keys():
                plt.suptitle('Training population ' + str(self.log['population']), fontweight='bold')
            
        # finalize plot:
        plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
        
        # save out:
        if file_path is not None:
            plt.savefig(file_path)
            plt.close('all')
        #
        return None

    @matplotlib.rc_context(plot_options)
    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called by the trainer at the end of every epoch to compute the training
        metrics and show progress during training if `feedback` is True.

        :param epoch: index of the current epoch.
        :param logs: dictionary of training metrics for the epoch. ``lr`` is set to 0 if missing and,
            with ``feedback > 2``, the internal training metrics are added to it.
        :returns: None
        """
        if logs is None:
            logs = {}
        if logs.get('lr', None) is None:
            logs['lr'] = 0.0

        # compute metrics:
        self.compute_training_metrics(logs=logs)

        # text monitoring of output:
        if self.feedback > 2:
            for met in self.training_metrics:
                if met in self.log.keys():
                    logs[met] = self.log[met][-1]

        # decide whether to plot:
        do_plots = self.feedback > 0 and self.plot_every > 0 and not cluster_plotting
        if do_plots:
            if ((epoch + 1) % self.plot_every) > 0:
                do_plots = False

        # do the plots:
        if do_plots:
            # clear output to restart the plot:
            if ipython_plotting:
                clear_output(wait=True)
                self._create_figure()
            else:
                plt.clf()

            # do the plot:
            self.training_plot(logs=logs, ipython_plotting=ipython_plotting, fast=True)

            # allow time for rendering and show:
            backend = matplotlib.get_backend().lower()
            if "agg" not in backend:
                plt.pause(0.00001)
                plt.show()
        #
        return None
    
    def print_training_summary(self):
        """
        Prints the summary of training metrics.
        """
        # length of maximum label for formatting:
        _max_label = max(len(_t) for _t in self.training_metrics)
        # cycle over metrics:        
        for _t in self.training_metrics:
            _v = self.log[_t][-1]
            if np.abs(_v) > 1e3 or np.abs(_v) < 1e-3:
                _pv = f"{_v:.4e}"
            else:
                _pv = f"{_v:.4f}"
            print(_t.ljust(_max_label)+':', _pv)


###############################################################################
# Transformed flow:


class DerivedParamsBijector(tb.AutoregressiveFlow):
    """Autoregressive bijector transforming between two parameterizations."""

    def __init__(self, chain, param_names_in, param_names_out, permutations=False, feedback=0, **kwargs):
        """
        Initialize the bijector.

        :param chain: reference chain providing parameter statistics.
        :param param_names_in: list of input parameter names.
        :param param_names_out: list of output parameter names.
        :param permutations: enable permutation of autoregressive order.
        :param feedback: verbosity level.
        :param kwargs: options passed to
            :class:`~tensiometer.synthetic_probability.trainable_bijectors.AutoregressiveFlow`.
        :raises ValueError: if ``param_names_in`` and ``param_names_out`` have different lengths.
        """
        self.num_params = len(param_names_in)
        if len(param_names_out) != self.num_params:
            raise ValueError('param_names_in and param_names_out must have the same length.')
        self.param_names_in = param_names_in
        self.param_names_out = param_names_out

        super().__init__(self.num_params, permutations=permutations, feedback=feedback, **kwargs)

        self.feedback = feedback

        seed = np.random.randint(0, 9999)

        self.flow_in = FlowCallback(
            chain,
            param_names=param_names_in,
            prior_bijector=None,
            trainable_bijector=None,
            rng=np.random.default_rng(seed=seed),
            apply_rescaling='independent',
            device=self.device,
            feedback=0)

        self.flow_out = FlowCallback(
            chain,
            param_names=param_names_out,
            prior_bijector=None,
            trainable_bijector=None,
            rng=np.random.default_rng(seed=seed),
            apply_rescaling='independent',
            device=self.device,
            feedback=0)

        self.num_training_samples = len(self.flow_in.training_samples)
        ind = [chain.index[name] for name in param_names_out]
        self.chain_samples = chain.samples[:, ind].astype(tu.np_prec)
        self.chain_loglikes = None
        self.has_loglikes = False
        self.chain_weights = chain.weights.astype(tu.np_prec)

        self.trainable_bijector = self.bijector
        self.bijector = bj.Chain([self.flow_out.bijector, self.trainable_bijector, bj.Invert(self.flow_in.bijector)])

        self.trainer = training.Trainer(
            module=self.trainable_bijector,
            log_prob_fn=self.trainable_bijector.forward,
            loss=loss.mean_squared_error(),
            learning_rate=1e-3)

    def train(self, epochs=100, batch_size=None, steps_per_epoch=None, callbacks=None, verbose=None, **kwargs):
        """
        Train the bijector to map the input to the output parameters (mean squared error).

        :param epochs: number of epochs.
        :param batch_size: batch size, defaults to ``num_training_samples / steps_per_epoch``.
        :param steps_per_epoch: steps per epoch, defaults to 20.
        :param callbacks: optional list of callbacks.
        :param verbose: verbosity level, defaults to 0 unless ``feedback >= 3``.
        :param kwargs: accepted for compatibility with :meth:`FlowCallback.train` and ignored.
        :returns: :class:`~tensiometer.synthetic_probability.training.History`.
        """
        # We're trying to loop through the full sample each epoch
        if batch_size is None:
            if steps_per_epoch is None:
                steps_per_epoch = 20
            batch_size = int(self.num_training_samples / steps_per_epoch)
        else:
            if steps_per_epoch is None:
                steps_per_epoch = int(self.num_training_samples / batch_size)

        if verbose is None:
            if self.feedback < 3:
                verbose = 0
            else:
                verbose = 1

        hist = self.trainer.fit(
            x=self.flow_in.training_samples,
            y=self.flow_out.training_samples,
            validation_data=(self.flow_in.test_samples, self.flow_out.test_samples, None),
            batch_size=max(int(batch_size), 1),
            epochs=epochs,
            steps_per_epoch=max(int(steps_per_epoch), 1),
            callbacks=callbacks,
            verbose=verbose)

        return hist


class AnalyticalDerivedParamsBijector:
    """Wrapper exposing a bijector that maps to analytically defined parameters."""
    def __init__(self, param_names_in, param_names_out, param_labels_out, **kwargs):
        """
        Initialize the analytical bijector wrapper.

        :param param_names_in: input parameter names.
        :param param_names_out: output parameter names.
        :param param_labels_out: labels for the derived parameters.
        :param kwargs: arguments of :class:`~tensiometer.synthetic_probability.bijectors.Inline`
            (``forward_fn``, ``inverse_fn``, ``inverse_log_det_jacobian_fn``, ...); the
            functions must be torch functions, and module level functions if the flow is saved.
        """
        self.num_params = len(param_names_in)
        if len(param_names_out) != self.num_params:
            raise ValueError('param_names_in and param_names_out must have the same length.')
        self.param_names_in = param_names_in
        self.param_names_out = param_names_out
        self.param_labels_out = param_labels_out

        _kwargs = stutils.filter_kwargs(kwargs, bj.Inline)
        _kwargs.setdefault('forward_min_event_ndims', 1)
        self.bijector = bj.Inline(**_kwargs)


class TransformedFlowCallback(FlowCallback):
    """Flow obtained by applying a transformation bijector to a trained flow."""

    def __init__(self, flow, transformation, transform_posterior=True):
        """
        Applies an analytic bijector to a flow to transform parameters.

        :param flow: trained flow.
        :param transformation: list with one elementwise bijector per parameter, a
            :class:`DerivedParamsBijector` or an :class:`AnalyticalDerivedParamsBijector`.
        :param transform_posterior: if False, the density is not corrected by the Jacobian
            of the transformation.
        :raises ValueError: for unsupported transformations.
        """

        infos = [
            'feedback',
            'plot_every',
            'num_params',
            'is_trained',
            'log',
        ]
        for info in infos:
            self.__dict__[info] = flow.__dict__[info]
        for info in ['training_metrics', 'prec', 'np_prec']:
            if info in flow.__dict__:
                self.__dict__[info] = flow.__dict__[info]
        self.device = flow.device
        self.is_light = bool(flow.__dict__.get('is_light', False))
        self._trainer_initialized = False

        self.transform_posterior = transform_posterior

        if isinstance(transformation, Iterable):
            transformation = list(transformation)
            if len(transformation) != self.num_params:
                raise ValueError('The transformation needs one bijector per parameter.')
            # new bijector
            b = bj.Blockwise(transformation)

            # parameter names and labels:
            self.param_names = []
            self.param_labels = []
            for t, name, label in zip(transformation, flow.param_names, flow.param_labels):
                if t.name != '':
                    self.param_names.append(t.name + '_' + name)
                    self.param_labels.append(t.name + ' ' + label)
                else:
                    self.param_names.append(name)
                    self.param_labels.append(label)
            # set ranges:
            if flow.parameter_ranges is not None:
                parameter_ranges = {}
                with torch.no_grad():
                    for i, name in enumerate(flow.param_names):
                        parameter_ranges[self.param_names[i]] = list(tu.to_numpy(transformation[i](flow.parameter_ranges[name])))
                self.parameter_ranges = parameter_ranges
            else:
                self.parameter_ranges = None

            if not self.is_light:
                with torch.no_grad():
                    self.chain_samples = tu.to_numpy(b.forward(flow.chain_samples))
                self.chain_loglikes = None
                self.has_loglikes = False
                self.chain_weights = flow.chain_weights.astype(tu.np_prec)

        elif isinstance(transformation, DerivedParamsBijector) or isinstance(transformation, AnalyticalDerivedParamsBijector):
            # first find the parameters that the DerivedParamsBijector is modifying
            mod_params = [flow.param_names.index(name) for name in transformation.param_names_in]
            fix_params = [i for i in range(flow.num_params) if i not in mod_params]
            perm = mod_params + [i for i in range(flow.num_params) if i not in mod_params]
            # new bijector
            s = len(transformation.param_names_in)
            b = bj.Chain([
                bj.Blockwise([transformation.bijector] + [bj.Identity() for _ in range(flow.num_params - s)],
                             block_sizes=[s] + [1] * (flow.num_params - s)),
                bj.Permute(perm),
            ])

            # parameter names and labels:
            self.param_names = transformation.param_names_out + [flow.param_names[i] for i in fix_params]

            if not self.is_light:
                if flow.chain_loglikes is not None:
                    self.chain_loglikes = flow.chain_loglikes.astype(tu.np_prec)
                else:
                    self.chain_loglikes = None
                self.has_loglikes = False
                self.chain_weights = flow.chain_weights.astype(tu.np_prec)

        else:
            raise ValueError('Unsupported transformation of type ' + type(transformation).__name__)

        if isinstance(transformation, DerivedParamsBijector):
            self.param_labels = transformation.flow_out.param_labels + [flow.param_labels[i] for i in fix_params]

            if not self.is_light:
                self.chain_samples = np.concatenate(
                    [transformation.chain_samples,
                     np.take(flow.chain_samples, fix_params, axis=1)], axis=1)

            # set ranges:
            if flow.parameter_ranges is not None:
                if transformation.flow_out.parameter_ranges is not None:
                    self.parameter_ranges = copy.deepcopy(transformation.flow_out.parameter_ranges)
                    for i in fix_params:
                        name = flow.param_names[i]
                        self.parameter_ranges[name] = flow.parameter_ranges[name]
                else:
                    self.parameter_ranges = None
            else:
                self.parameter_ranges = None

        elif isinstance(transformation, AnalyticalDerivedParamsBijector):
            self.param_labels = transformation.param_labels_out + [flow.param_labels[i] for i in fix_params]

            if not self.is_light:
                with torch.no_grad():
                    temp_samples = tu.to_numpy(transformation.bijector.forward(
                        np.take(flow.chain_samples, mod_params, axis=1))).astype(tu.np_prec)
                self.chain_samples = np.concatenate(
                    [temp_samples,
                    np.take(flow.chain_samples, fix_params, axis=1)], axis=1)

                # set ranges:
                if flow.parameter_ranges is not None:
                    self.parameter_ranges = {p: (temp_samples[:, i].min(), temp_samples[:, i].max())
                                             for i, p in enumerate(transformation.param_names_out)}
                    for i in fix_params:
                        name = flow.param_names[i]
                        self.parameter_ranges[name] = flow.parameter_ranges[name]
                else:
                    self.parameter_ranges = None
            else:
                self.parameter_ranges = None

        # save bijector:
        b.to(self.device)
        self.transformer_bijector = b

        # set name tag:
        self.name_tag = flow.name_tag + '_transformed'
        with torch.no_grad():
            # set sample MAP:
            if flow.sample_MAP is not None:
                self.sample_MAP = tu.to_numpy(b.forward(np.atleast_2d(flow.sample_MAP)))
            else:
                self.sample_MAP = None
            # set chains MAP:
            if flow.chain_MAP is not None:
                self.chain_MAP = tu.to_numpy(b.forward(np.atleast_2d(flow.chain_MAP)))
            else:
                self.chain_MAP = None

        # set bijectors and distribution:
        self.bijectors = [b] + flow.bijectors
        self.bijector = bj.Chain(self.bijectors)
        self.trainable_bijector = flow.__dict__.get('trainable_bijector', None)
        self._build_distributions()

        # MAP:
        if flow.MAP_coord is not None:
            with torch.no_grad():
                self.MAP_coord = tu.to_numpy(b.forward(np.atleast_2d(flow.MAP_coord)))[0]
            self.MAP_logP = float(self.log_probability(np.atleast_2d(self.MAP_coord))[0])
        else:
            self.MAP_coord = None
            self.MAP_logP = None

    def _build_distributions(self):
        """Build the base and full distributions on the flow device."""
        self.base_distribution = ds.standard_normal(self.num_params, device=self.device)
        self.distribution = ds.TransformedDistribution(distribution=self.base_distribution, bijector=self.bijector)

    def _log_probability(self, x):
        """Log probability, optionally without the Jacobian of the transformation."""
        log_prob = self.distribution.log_prob(x)
        if not self.transform_posterior:
            log_prob = log_prob - self.transformer_bijector.inverse_log_det_jacobian(x, event_ndims=1)
        return log_prob

    def train(self, *args, **kwargs):
        """
        Transformed flows cannot be trained.

        :param args: ignored.
        :param kwargs: ignored.
        :raises NotImplementedError: transformed flows cannot be trained, train the original flow.
        """
        raise NotImplementedError('Transformed flows cannot be trained: train the original flow.')

    def global_train(self, *args, **kwargs):
        """
        Transformed flows cannot be trained.

        :param args: ignored.
        :param kwargs: ignored.
        :raises NotImplementedError: transformed flows cannot be trained, train the original flow.
        """
        raise NotImplementedError('Transformed flows cannot be trained: train the original flow.')

###############################################################################
# Average flow:


class average_flow(FlowCallback):
    """Mixture of flows trained on the same chain, weighted by their validation performance."""

    def __init__(self, flows, **kwargs):
        """
        Initialize the average flow class

        :param flows: list of flows with the same parameters, on the same device.
        :param kwargs: ``validation_training_idx``, the split shared by the flows.
        :raises ValueError: if the flows have different parameters or devices.
        """
        # check parameters and copy in info:
        for flow in flows:
            if flow.param_names != flows[0].param_names:
                raise ValueError(
                    'Flow', flow.name_tag, 'does not have the same parameters as', flows[0].name_tag,
                    '. Cannot average.')
            if torch.device(flow.device) != torch.device(flows[0].device):
                raise ValueError('All the flows of an average flow must be on the same device, got '
                                 + str(flow.device) + ' and ' + str(flows[0].device)
                                 + '. Use flow.to(device) to move them.')
        self.is_light = any(bool(flow.__dict__.get('is_light', False)) for flow in flows)
        self._trainer_initialized = False
        # copy in infos from the first flow:
        infos = [
            'name_tag',
            'feedback',
            'plot_every',
            'sample_MAP',
            'chain_MAP',
            'num_params',
            'param_names',
            'param_labels',
            'parameter_ranges',
            'periodic_params',
            'chain_samples',
            'chain_loglikes',
            'has_loglikes',
            'chain_weights',
            'is_trained',
            'MAP_coord',
            'MAP_logP',
            'prec',
            'np_prec',
            # bijectors and distribution:
            'prior_bijector',
            'fixed_bijector',
            'trainable_transformation'
        ]
        for info in infos:
            if self.is_light and info in FlowCallback._full_only_attributes:
                continue
            try:
                self.__dict__[info] = flows[0].__dict__[info]
            except KeyError:
                print("Flow does not have attribute :", info)
        self.device = flows[0].device
        # check and save training and validation indexes:
        if not self.is_light:
            validation_training_idx = kwargs.get('validation_training_idx')
            if validation_training_idx is None:
                print('Warning: validation_training_idx not found in kwargs. You should ensure that training and validation indexes are coherent across average flow.')
                self.test_idx, self.training_idx = None, None
            else:
                self.test_idx, self.training_idx = validation_training_idx
            # check consistency of training and validation split across average flows:
            for flow in flows:
                if 'test_idx' in flow.__dict__ and 'training_idx' in flow.__dict__:
                    if not np.array_equal(flow.test_idx, self.test_idx) or not np.array_equal(flow.training_idx, self.training_idx):
                        print('Warning: validation and training indexes are not consistent across average flows.')
        # copy in flows:
        self.flows = flows
        # process:
        self.num_flows = len(self.flows)
        # compute weights:
        self._set_flow_weights()
        #
        return None

    def _set_flow_weights(self, mode='val_loss'):
        """
        Compute the relative weights of flows based on validation loss

        :param mode: ``'val_loss'`` (default), ``'loss'``, ``'chi2Z_ks_p'``, ``'equal'`` or another log key.
        :raises ValueError: if a flow has no ``mode`` entry in its log.
        """
        # get weights:
        if mode == 'equal':
            self.weights = np.ones(self.num_flows) / self.num_flows
        else:
            _temp_weights = []
            for flow in self.flows:
                if mode not in flow.log.keys() or len(flow.log[mode]) == 0:
                    raise ValueError('Cannot initialize average flow weights. Key', mode, 'not found in flow', flow.name_tag)
                _temp_weights.append(flow.log[mode][-1])
            _temp_weights = np.array(_temp_weights, dtype=np.float64)

            if mode == 'loss' or mode == 'val_loss':
                _temp_weights = np.exp(np.amin(_temp_weights) - _temp_weights)
                self.weights = _temp_weights / np.sum(_temp_weights)
            elif mode == 'chi2Z_ks_p':
                _temp_weights = np.exp(np.log(_temp_weights) - np.amax(np.log(_temp_weights)))
                self.weights = _temp_weights / np.sum(_temp_weights)
            else:
                self.weights = _temp_weights / np.sum(_temp_weights)

        # save:
        self.weights = tu.to_tensor(self.weights, device='cpu')

        # initialize the mixture distribution:
        self._build_distributions()
        #
        return None

    def _build_distributions(self):
        """Build the mixture distribution of the flows."""
        self.distribution = ds.Mixture(self.weights, [flow.distribution for flow in self.flows])

    def _move_to(self, device):
        """Move all the flows to ``device``."""
        for flow in self.flows:
            flow.to(device)
        self.device = device
        self._build_distributions()

    def train(self, **kwargs):
        """
        Train all the flows.

        :param kwargs: options passed to :meth:`FlowCallback.train` of every flow.
        :returns: None
        :raises RuntimeError: for flows loaded from light snapshots.
        """
        self._check_trainable()
        for flow in self.flows:
            flow.train(**kwargs)
        return None

    def global_train(self, **kwargs):
        """
        Train all the flows with the population strategy.

        :param kwargs: options passed to :meth:`FlowCallback.global_train` of every flow.
        :returns: None
        :raises RuntimeError: for flows loaded from light snapshots.
        """
        self._check_trainable()
        for flow in self.flows:
            flow.global_train(**kwargs)
        return None

    ###############################################################################
    # Plotting and feedback:

    @matplotlib.rc_context(plot_options)
    def training_plot(self, logs=None, file_path=None, ipython_plotting=False):
        """
        Method to produce training plot with training metrics, one figure per flow.

        :param logs: dictionary of training metrics passed to the training plot of every flow,
            defaults to None.
        :param file_path: path of the file to save the figures to, defaults to None (not saved).
            The index of the flow is appended to the file name, e.g. ``plot_0.pdf``.
        :param ipython_plotting: whether plotting is inline in IPython, in which case every
            figure is shown, defaults to False.
        :returns: None
        """
        if file_path is not None:
            file_name, file_format = os.path.splitext(file_path)
        for _i, flow in enumerate(self.flows):
            if file_path is not None:
                _temp_name = file_name+'_'+str(_i)+file_format
            else:
                _temp_name = None
            flow.training_plot(logs=logs,
                               file_path=_temp_name,
                               ipython_plotting=ipython_plotting,
                               title='Training flow '+str(_i))
            if ipython_plotting:
                plt.show()
        #
        return None

    def print_training_summary(self):
        """
        Prints the summary of training metrics.
        """
        # print number of flows:
        print('Number of flows:', self.num_flows)
        # print flow weights:
        with np.printoptions(precision=2, suppress=True):
            print('Flow weights   :', tu.to_numpy(self.weights))
        # cycle over training metrics using the first flow as template:
        _max_label = max(len(_t) for _t in self.flows[0].training_metrics)
        # cycle over metrics:
        for _t in self.flows[0].training_metrics:
            _v = np.array([_f.log[_t][-1] for _f in self.flows])
            with np.printoptions(precision=2, suppress=False):
                print(_t.ljust(_max_label)+':', _v)

    ###############################################################################
    # Utility functions:

    def cast(self, v):
        """
        Convert to a CPU tensor with the flow precision.

        :param v: input vector (array-like or tensor).
        :returns: CPU tensor.
        """
        return self.flows[0].cast(v)

    def _log_probability_abs(self, abs_coord):
        """:raises NotImplementedError: average flows have no abstract coordinates."""
        raise NotImplementedError('Average flow does not have well defined abstract coordinates')

    def log_probability_abs(self, abs_coord):
        """
        Not available: average flows have no abstract coordinates.

        :param abs_coord: input parameter value in abstract coordinates (unused).
        :raises NotImplementedError: average flows have no abstract coordinates.
        """
        raise NotImplementedError('Average flow does not have well defined abstract coordinates')

    def log_probability_abs_jacobian(self, abs_coord):
        """
        Not available: average flows have no abstract coordinates.

        :param abs_coord: input parameter value in abstract coordinates (unused).
        :raises NotImplementedError: average flows have no abstract coordinates.
        """
        raise NotImplementedError('Average flow does not have well defined abstract coordinates')

    def log_probability_abs_hessian(self, abs_coord):
        """
        Not available: average flows have no abstract coordinates.

        :param abs_coord: input parameter value in abstract coordinates (unused).
        :raises NotImplementedError: average flows have no abstract coordinates.
        """
        raise NotImplementedError('Average flow does not have well defined abstract coordinates')

    def _sample(self, N):
        """Draw samples from the mixture on the flows device."""
        return self.distribution.sample(int(N))


###############################################################################
# Cache helpers:

# keyword arguments that do not change the flow and are ignored by the cache check:
_RUNTIME_KWARGS = ('feedback', 'plot_every', 'verbose', 'device')


def _hash_array(value):
    """sha256 of an array."""
    array = np.ascontiguousarray(tu.to_numpy(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _normalize_cache_value(value):
    """
    Convert a creation argument to a plain, comparable value.

    :param value: any value.
    :returns: JSON-like value (arrays by shape and hash, callables by qualified name, other
        objects by type name).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return {'array_shape': list(value.shape), 'sha256': _hash_array(value)}
    if isinstance(value, dict):
        return {str(_k): _normalize_cache_value(_v) for _k, _v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(_v) for _v in value]
    if callable(value) and hasattr(value, '__qualname__'):
        return 'callable:' + str(getattr(value, '__module__', '')) + '.' + value.__qualname__
    return 'object:' + type(value).__module__ + '.' + type(value).__name__


def _chain_fingerprint(chain, param_names=None):
    """
    sha256 fingerprint of the chain columns used by a flow.

    :param chain: :class:`~getdist.mcsamples.MCSamples`.
    :param param_names: parameter names, defaults to the running parameters.
    :returns: hexadecimal digest.
    """
    if param_names is None:
        param_names = chain.getParamNames().getRunningNames()
    param_names = list(param_names)
    digest = hashlib.sha256()
    digest.update(json.dumps(param_names).encode())
    indexes = [chain.index[name] for name in param_names]
    digest.update(np.ascontiguousarray(chain.samples[:, indexes], dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(chain.weights, dtype=np.float64).tobytes())
    if chain.loglikes is not None:
        digest.update(np.ascontiguousarray(chain.loglikes, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _cache_record(chain, kwargs):
    """
    Record identifying the chain and the creation arguments of a cached flow.

    :param chain: input chain.
    :param kwargs: creation arguments.
    :returns: dictionary with ``chain_fingerprint`` and normalized ``kwargs``.
    """
    return {
        'chain_fingerprint': _chain_fingerprint(chain, kwargs.get('param_names', None)),
        'kwargs': {_k: _normalize_cache_value(_v) for _k, _v in sorted(kwargs.items()) if _k not in _RUNTIME_KWARGS},
    }


def _check_cache_record(stored, record, cache_file):
    """
    Check that a cache was made from the same chain and arguments.

    :param stored: record stored in the cache (or None).
    :param record: record of the current call.
    :param cache_file: cache path, for the error message.
    :raises ValueError: listing the differences.
    """
    if stored is None:
        raise ValueError('The cache file ' + str(cache_file) + ' has no cache record; '
                         'use overwrite_cache=True to retrain and overwrite it.')
    differences = []
    if stored.get('chain_fingerprint') != record.get('chain_fingerprint'):
        differences.append('chain')
    _missing = object()
    stored_kwargs = stored.get('kwargs', {})
    record_kwargs = record.get('kwargs', {})
    for key in sorted(set(stored_kwargs.keys()) | set(record_kwargs.keys())):
        if stored_kwargs.get(key, _missing) != record_kwargs.get(key, _missing):
            differences.append(key)
    if len(differences) > 0:
        raise ValueError('The cache file ' + str(cache_file) + ' was created from different inputs ('
                         + ', '.join(differences) + '). Use overwrite_cache=True to retrain and overwrite it, '
                         'or use another cache_file.')


def _check_removed_cache_kwargs(kwargs):
    """
    :raises ValueError: if the removed ``cache_dir`` / ``root_name`` arguments are used.
    """
    for key in ['cache_dir', 'root_name']:
        if key in kwargs:
            raise ValueError(key + ' has been removed: pass the path of the cache file with cache_file.')


###############################################################################
# Flow utilities:


def flow_from_chain(chain, cache_file=None, overwrite_cache=False, cache_mode='full', **kwargs):
    """
    Helper to initialize and train a synthetic probability starting from a chain.

    If a cache file is given, the trained flow is saved there and loaded (without training)
    at later calls. A cache hit is accepted only if it was made from the same chain and the
    same arguments (``feedback``, ``plot_every``, ``verbose`` and ``device`` are ignored).

    :param chain: input chain.
    :param cache_file: optional path of the cache file.
    :param overwrite_cache: retrain and overwrite an existing cache file.
    :param cache_mode: ``'full'`` or ``'light'`` snapshot, see :meth:`FlowCallback.save`.
    :param kwargs: arguments of :class:`FlowCallback` and :meth:`FlowCallback.global_train`.
    :returns: trained flow.
    :raises ValueError: if the cache file was made from different inputs.
    """
    _check_removed_cache_kwargs(kwargs)
    record = _cache_record(chain, kwargs)
    # load from cache:
    if cache_file is not None and os.path.isfile(cache_file) and not overwrite_cache:
        flow = FlowCallback.load(cache_file, device=kwargs.get('device', None))
        _check_cache_record(getattr(flow, 'cache_record', None), record, cache_file)
        return flow
    # initialize posterior flow:
    flow = FlowCallback(chain, **kwargs)
    # train posterior flow:
    flow.global_train(**kwargs)
    # save trained model:
    if cache_file is not None:
        flow.cache_record = record
        flow.save(cache_file, mode=cache_mode)
    #
    return flow


def _remove_file(path):
    """Remove a file if it exists."""
    if path is not None and os.path.isfile(path):
        os.remove(path)


def average_flow_from_chain(chain, num_flows=1, cache_file=None, overwrite_cache=False, cache_mode='full',
                            use_mpi=False, **kwargs):
    """
    Helper to initialize and train an average of flows starting from a chain.

    The flows share the training / validation split. If a cache file is given the
    assembled average flow is saved there and loaded at later calls; while training, the
    split and the members are stored in ``cache_file + '.split'`` and ``cache_file + '.part<i>'``,
    so that interrupted or MPI runs resume. These files are removed once the average flow is saved.

    :param chain: input chain.
    :param num_flows: number of flows.
    :param cache_file: optional path of the cache file.
    :param overwrite_cache: retrain and overwrite existing cache files.
    :param cache_mode: ``'full'`` or ``'light'`` snapshot of the assembled flow.
    :param use_mpi: distribute the flows over MPI ranks (requires ``cache_file``).
    :param kwargs: arguments of :class:`FlowCallback` and :meth:`FlowCallback.global_train`.
    :returns: the average flow (the flow itself if ``num_flows`` is 1).
    :raises ValueError: if a cache file was made from different inputs.
    """
    _check_removed_cache_kwargs(kwargs)
    kwargs = dict(kwargs)

    # get feedback flag:
    feedback = kwargs.get('feedback', 0)
    if feedback is None:
        feedback = 0
    if 'feedback' in kwargs and type(kwargs['feedback']) == int:
        kwargs['feedback'] = max(kwargs['feedback'] - 1, 0)

    # record of the call:
    record = _cache_record(chain, dict(kwargs, num_flows=num_flows))

    # MPI is incompatible with no cache:
    if use_mpi and cache_file is None:
        use_mpi = False
        print('Warning: MPI is incompatible with no cache. Disabling MPI.')

    # check if we want to use MPI:
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank is None:
            rank = 0
        if size is None:
            size = 1
    else:
        rank = 0
        size = 1

    if size > 1 and rank == 0 and feedback > 0 and use_mpi:
        print('Training average flow with MPI enabled', flush=use_mpi)
        print('MPI size:', size, flush=use_mpi)

    # device of the assembled flow and of the flows trained by this rank:
    target_device = kwargs.get('device', None)
    member_device = tu.resolve_device(target_device)
    if use_mpi and member_device.type == 'cuda':
        member_device = torch.device('cuda:' + str(rank % torch.cuda.device_count()))

    # load the assembled flow from cache:
    if cache_file is not None and os.path.isfile(cache_file) and not overwrite_cache:
        if feedback > 0 and rank == 0:
            print('Loading average flow from cache', flush=use_mpi)
        flow = FlowCallback.load(cache_file, device=target_device)
        _check_cache_record(getattr(flow, 'cache_record', None), record, cache_file)
        return flow

    # cache files of the split and of the members:
    split_file = None if cache_file is None else cache_file + '.split'
    part_files = [None if cache_file is None else cache_file + '.part' + str(i) for i in range(num_flows)]

    # draw training and test indexes so that they are shared across flows:
    if 'validation_training_idx' not in kwargs:
        validation_training_idx = None
        if rank == 0:
            if split_file is not None and os.path.isfile(split_file) and not overwrite_cache:
                _split = torch.load(split_file, weights_only=False)
                _check_cache_record(_split.get('cache_record', None), record, split_file)
                validation_training_idx = _split['validation_training_idx']
            else:
                # get number of samples:
                n = chain.samples.shape[0]
                # draw:
                indices = np.random.permutation(n)
                # get validation split:
                validation_split = kwargs.get('validation_split', 0.1)
                n_split = int(validation_split * n)
                validation_training_idx = indices[:n_split], indices[n_split:]
                # save to file:
                if split_file is not None:
                    tu.atomic_save({'cache_record': record, 'validation_training_idx': validation_training_idx}, split_file)
        # broadcast:
        if use_mpi:
            validation_training_idx = comm.bcast(validation_training_idx, root=0)
        kwargs['validation_training_idx'] = validation_training_idx

    # load each flow from cache or compute:
    flows = []
    for i in range(num_flows):

        # skip if not in rank:
        if i % size != rank:
            continue

        _member_kwargs = dict(kwargs)
        _member_kwargs['device'] = member_device
        _member_record = _cache_record(chain, dict(kwargs, member_index=i))
        # do the list of flows:
        if part_files[i] is not None and os.path.isfile(part_files[i]) and not overwrite_cache:
            if feedback > 0:
                print('Loading flow', i, 'from cache', flush=use_mpi)
            flow = FlowCallback.load(part_files[i], device=member_device)
            _check_cache_record(getattr(flow, 'cache_record', None), _member_record, part_files[i])
        else:
            # proceed:
            if feedback > 0:
                if use_mpi and size > 1:
                    print('Training flow', i, 'on MPI worker', rank, flush=use_mpi)
                else:
                    print('Training flow', i, flush=use_mpi)
            # initialize posterior flow:
            flow = FlowCallback(chain, **_member_kwargs)
            # train posterior flow:
            flow.global_train(**_member_kwargs)
            # save trained model:
            if part_files[i] is not None:
                flow.cache_record = _member_record
                flow.save(part_files[i], mode='full')
        flows.append(flow)

    # collect the flows of all the ranks from the part files:
    if use_mpi and size > 1:
        comm.Barrier()
        flows = [FlowCallback.load(part_files[i], device=target_device) for i in range(num_flows)]
    elif tu.resolve_device(target_device) != member_device:
        flows = [flow.to(tu.resolve_device(target_device)) for flow in flows]

    # initialize the average flow:
    if len(flows) == 1:
        _avg_flow = flows[0]
    else:
        _avg_flow = average_flow(flows, **kwargs)

    # save the assembled flow and remove the temporary files:
    if cache_file is not None:
        if rank == 0:
            _avg_flow.cache_record = record
            _avg_flow.save(cache_file, mode=cache_mode)
            _remove_file(split_file)
            for part_file in part_files:
                _remove_file(part_file)
        if use_mpi and size > 1:
            comm.Barrier()
    #
    return _avg_flow
