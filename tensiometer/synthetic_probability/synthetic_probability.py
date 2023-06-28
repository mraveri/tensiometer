"""
Main file containing the synthetic probability class and methods.
"""

###############################################################################
# initial imports and set-up:

import os
import copy
import numpy as np
import getdist.chains as gchains
from getdist import MCSamples
import scipy
import scipy.integrate
from scipy.spatial import cKDTree
import scipy.stats
from collections.abc import Iterable
import pickle

# plotting:
import matplotlib
from matplotlib import pyplot as plt

# local imports:
from . import lr_schedulers as lr
from . import loss_functions as loss
from . import trainable_bijectors as tb
from . import prior_bijectors as pb

from .. import utilities as utils
from .. import gaussian_tension

gchains.print_load_details = False

# tensorflow imports:
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfb = tfp.bijectors
    tfd = tfp.distributions
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.callbacks import Callback
    import tensorflow.keras.callbacks as keras_callbacks
    import tensorflow.python.eager.def_function as tf_defun

    HAS_FLOW = True
    # tensorflow precision:
    prec = tf.float32
    np_prec = np.float32
except Exception as e:
    print("Could not import tensorflow or tensorflow_probability: ", e)
    Callback = object
    HAS_FLOW = False

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


class FlowCallback(Callback):
    """
    A class to compute the normalizing flow interpolation of a probability density given the samples.

    A normalizing flow is trained to approximate the distribution and then used to numerically evaluate the probablity of a parameter shift (see REF). To do so, it defines a bijective mapping that is optimized to gaussianize the difference chain samples. This mapping is performed in two steps, using the gaussian approximation as pre-whitening. The notations used in the code are:

    * `X` designates samples in the original parameter difference space;
    * `Y` designates samples in the gaussian approximation space, `Y` is obtained by shifting and scaling `X` by its mean and covariance (like a PCA);
    * `Z` designates samples in the gaussianized space, connected to `Y` with a normalizing flow denoted `trainable_bijector`.

    The user may provide the `trainable_bijector` as a :class:`~tfp.bijectors.Bijector` object from `Tensorflow Probability <https://www.tensorflow.org/probability/>`_ or make use of the utility class :class:`~.MaskedAutoregressiveFLow` to instantiate a Masked Autoregressive Flow (with `trainable_bijector='MAF'`).

    This class derives from :class:`~tf.keras.callbacks.Callback` from Keras, which allows for visualization during training. The normalizing flows (X->Y->Z) are implemented as :class:`~tfp.bijectors.Bijector` objects and encapsulated in a Keras :class:`~tf.keras.Model`.

    Here is an example:

    .. code-block:: python

        # Initialize the flow and model
        diff_flow_callback = FlowCallback(chain, trainable_bijector='MAF')
        # Train the model
        diff_flow_callback.train()
        # Compute the shift probability and confidence interval
        p, p_low, p_high = diff_flow_callback.estimate_shift_significance()

    :param chain: input parameter difference chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :type param_names: list, optional
    :param trainable_bijector: either a :class:`~tfp.bijectors.Bijector` object
        representing the mapping from `Z` to `Y`, or 'MAF' to instantiate a :class:`~.MaskedAutoregressiveFLow`, defaults to 'MAF'.
    :type trainable_bijector: optional
    :param learning_rate: initial learning rate, defaults to 1e-3.
    :type learning_rate: float, optional
    :param feedback: feedback level, defaults to 1. Zero is no feedback (including training plotting). One is a little feedback. Two and higher is a lot of feedback (useful for debug).
    :type feedback: int, optional
    :param plot_every: how much to plot during training. This quantifies how many epochs should pass before plotting.
    :type plot_every: int, optional
    :param validation_split: fraction of samples to use for the validation sample, defaults to 0.1
    :type validation_split: float, optional
    :reference: George Papamakarios, Theo Pavlakou, Iain Murray (2017). Masked Autoregressive Flow for Density Estimation. `arXiv:1705.07057 <https://arxiv.org/abs/1705.07057>`_
    """

    def __init__(
            self,
            chain,
            param_names=None,
            param_ranges=None,
            feedback=1,
            plot_every=10,
            prior_bijector='ranges',
            apply_pregauss=True,
            trainable_bijector='AutoregressiveFlow',
            validation_split=0.1,
            **kwargs
        ):

        # check input:
        if feedback < 0 or not isinstance(feedback, int):
            raise ValueError('feedback needs to be a positive integer')
        if plot_every < 0 or not isinstance(plot_every, int):
            raise ValueError('plot_every needs to be a positive integer')
        # read in varaiables:
        self.feedback = feedback
        self.plot_every = plot_every

        # initialize internal samples from chain:
        self._init_chain(chain, param_names=param_names, param_ranges=param_ranges, **kwargs)
        # initialize fixed bijector:
        self._init_fixed_bijector(prior_bijector=prior_bijector, apply_pregauss=apply_pregauss)
        # initialize trainable bijector:
        self._init_trainable_bijector(trainable_bijector=trainable_bijector, **kwargs)
        # initialize training dataset:
        self._init_training_dataset(validation_split=validation_split)
        # initialize distribution:
        self._init_distribution()
        # initialize loss function:
        self._init_model(**kwargs)
        # initialize training metrics and plotting:
        self._init_training_monitoring()

        # initialize internal variables:
        self.is_trained = False
        self.MAP_coord = None
        self.MAP_logP = None

    def _init_chain(self, chain=None, param_names=None, param_ranges=None, init_nearest=False, **kwargs):
        """
        Read in MCMC sample chain and save internal quantities.
        """
        # return if we have no chain:
        if chain is None:
            return None
        # feedback:
        if self.feedback > 0:
            print('* Initializing samples')

        # save name of the flow:
        if chain.name_tag is not None:
            self.name_tag = chain.name_tag + '_flow'
        else:
            self.name_tag = 'flow'
        # feedback:
        if self.feedback > 1:
            print('    - flow name:', self.name_tag)

        # initialize param names:
        if param_names is None:
            param_names = chain.getParamNames().getRunningNames()
        else:
            chain_params = chain.getParamNames().list()
            if not np.all([name in chain_params for name in param_names]):
                raise ValueError(
                    'Input parameter is not in the chain.\n', 'Input parameters ', param_names, '\n'
                    'Possible parameters', chain_params
                    )
        # save param names:
        self.param_names = param_names
        # save param labels:
        self.param_labels = [name.label for name in chain.getParamNames().parsWithNames(param_names)]

        # initialize ranges:
        self.parameter_ranges = {}
        for name in param_names:
            # get ranges from user:
            if param_ranges is not None:
                if name not in param_ranges.keys():
                    raise ValueError(
                        'Range for parameter ', name, ' is not specified.\n',
                        'When passing ranges explicitly all parameters have to be included.'
                        )
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

        # feedback:
        if self.feedback > 1:
            print('    - flow parameters and ranges:')
            for name in param_names:
                print('      ' + name + ' : [{0:.6g}, {1:.6g}]'.format(*self.parameter_ranges[name]))

        # initialize sample MAP:
        temp = chain.samples[np.argmin(chain.loglikes), :]
        self.sample_MAP = np.array([temp[chain.index[name]] for name in param_names])
        # try to get real best fit:
        try:
            self.chain_MAP = np.array([name.best_fit for name in chain.getBestFit().parsWithNames(param_names)])
        except:
            self.chain_MAP = None

        # initialize the samples:
        ind = [chain.index[name] for name in param_names]
        self.num_params = len(ind)
        self.chain_samples = chain.samples[:, ind].astype(np_prec)
        self.chain_loglikes = chain.loglikes.astype(np_prec)
        self.has_loglikes = self.chain_loglikes is not None
        self.chain_weights = chain.weights.astype(np_prec)

        # initialize nearest neighbours:
        if init_nearest:
            self._init_nearest_samples()
        #
        return None

    def _init_nearest_samples(self):
        """
        Initializes samples in a tree for nearest neighbour searches. Takes a while in high dimensions...
        """
        # cache nearest neighbours indexes, in whitened coordinates:
        temp_cov = np.cov(self.chain_samples.T, aweights=self.chain_weights)
        white_samples = np.dot(scipy.linalg.sqrtm(np.linalg.inv(temp_cov)), self.chain_samples.T).T
        data_tree = cKDTree(white_samples, balanced_tree=True)
        r2, idx = data_tree.query(white_samples, (2), workers=-1)
        self.chain_nearest_index = idx.copy()
        #
        return None

    def _init_fixed_bijector(self, prior_bijector='ranges', apply_pregauss=True):
        """
        Intitialize prior and whitening bijector.
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing fixed bijector')

        # Prior bijector setup:
        if prior_bijector == 'ranges':
            # extend slightly the ranges to avoid overflows:
            temp_ranges = []
            for name in self.param_names:
                temp_range = self.parameter_ranges[name]
                center = 0.5 * (temp_range[0] + temp_range[1])
                length = temp_range[1] - temp_range[0]
                eps = 0.001
                temp_ranges.append(
                    {
                        'lower': self.cast(center - 0.5 * length * (1. + eps)),
                        'upper': self.cast(center + 0.5 * length * (1. + eps))
                        }
                    )
            # define bijector:
            self.prior_bijector = pb.prior_bijector_helper(temp_ranges)
        elif isinstance(prior_bijector, tfp.bijectors.Bijector):
            self.prior_bijector = prior_bijector
        elif prior_bijector is None or prior_bijector is False:
            self.prior_bijector = tfb.Identity()
        self.bijectors = [self.prior_bijector]

        # feedback:
        if self.feedback > 1:
            print('    - using prior bijector:', self.prior_bijector)

        # Whitening bijector:
        if apply_pregauss:
            temp_X = self.prior_bijector.inverse(self.chain_samples).numpy()
            temp_chain = MCSamples(samples=temp_X, weights=self.chain_weights, names=self.param_names)
            temp_gaussian_approx = gaussian_tension.gaussian_approximation(temp_chain, param_names=self.param_names)
            temp_dist = tfd.MultivariateNormalTriL(
                loc=self.cast(temp_gaussian_approx.means[0]),
                scale_tril=tf.linalg.cholesky(self.cast(temp_gaussian_approx.covs[0]))
                )
            self.bijectors.append(temp_dist.bijector)

        # feedback:
        if self.feedback > 1:
            if apply_pregauss:
                print('    - whitening samples')
            else:
                print('    - not whitening samples')

        self.fixed_bijector = tfb.Chain(self.bijectors)

        #
        return None

    def _init_trainable_bijector(self, trainable_bijector, trainable_bijector_path=None, **kwargs):
        """
        Initialize trainable part of the bijector
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing trainable bijector')

        # select model for trainable transformation:
        if trainable_bijector == 'AutoregressiveFlow':
            self.trainable_transformation = tb.AutoregressiveFlow(self.num_params, feedback=self.feedback, **kwargs)
        elif isinstance(trainable_bijector, tb.TrainableTransformation):
            self.trainable_transformation = trainable_bijector
        elif isinstance(trainable_bijector, tfp.bijectors.Bijector):
            self.trainable_transformation = None
        elif trainable_bijector is None or trainable_bijector is False:
            self.trainable_transformation = None
        else:
            raise ValueError

        # load from file:
        if trainable_bijector_path is not None:
            if self.trainable_transformation is not None:
                self.trainable_transformation = self.trainable_transformation.load(trainable_bijector_path, **kwargs)

        # initialize bijector:
        if self.trainable_transformation is not None:
            self.trainable_bijector = self.trainable_transformation.bijector
        elif isinstance(trainable_bijector, tfp.bijectors.Bijector):
            self.trainable_bijector = trainable_bijector
        elif trainable_bijector is None or trainable_bijector is False:
            self.trainable_bijector = tfb.Identity()

        self.bijectors.append(self.trainable_bijector)
        self.bijector = tfb.Chain(self.bijectors)

        #
        return None

    def _init_training_dataset(self, validation_split=0.1):
        """
        Initialize the training dataset, splitting training and validation.
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing training dataset')

        # split training/test:
        n = self.chain_samples.shape[0]
        indices = np.random.permutation(n)
        n_split = int(validation_split * n)
        self.test_idx, self.training_idx = indices[:n_split], indices[n_split:]

        # training samples:
        self.training_samples = self.fixed_bijector.inverse(self.chain_samples[self.training_idx, :]
                                                           ).numpy().astype(np_prec)
        self.num_training_samples = len(self.training_samples)

        if self.has_loglikes:
            _jac_true_preabs = self.fixed_bijector.inverse_log_det_jacobian(
                self.chain_samples[self.training_idx, :], event_ndims=1
                )
            self.training_logP_preabs = -1. * self.chain_loglikes[self.training_idx] - _jac_true_preabs

        self.training_weights = self.chain_weights[self.training_idx]
        self.training_weights *= len(self.training_weights) / np.sum(
            self.training_weights
            )  # weights normalized to number of training samples
        self.has_weights = np.any(self.training_weights != self.training_weights[0])

        # test samples:
        self.test_samples = self.fixed_bijector.inverse(self.chain_samples[self.test_idx, :]).numpy().astype(np_prec)
        self.num_test_samples = len(self.test_samples)

        if self.has_loglikes:
            _jac_true_test_preabs = self.fixed_bijector.inverse_log_det_jacobian(
                self.chain_samples[self.test_idx, :], event_ndims=1
                )
            self.test_logP_preabs = -1. * self.chain_loglikes[self.test_idx] - _jac_true_test_preabs

        self.test_weights = self.chain_weights[self.test_idx]
        self.test_weights *= len(self.test_weights
                                ) / np.sum(self.test_weights)  # weights normalized to number of validation samples

        # initialize tensorflow sample generator:
        if self.has_loglikes:
            self.training_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    self.cast(self.training_samples),
                    self.cast(self.training_logP_preabs),
                    self.cast(self.training_weights),
                    )
                )
        else:
            self.training_dataset = tf.data.Dataset.from_tensor_slices(
                (
                    self.cast(self.training_samples),
                    self.cast(self.training_weights),
                    )
                )
        self.training_dataset = self.training_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()
        self.training_dataset = self.training_dataset.shuffle(self.num_training_samples,
                                                              reshuffle_each_iteration=True).repeat()

        # initialize validation data:
        if self.has_loglikes:
            self.validation_dataset = (
                self.cast(self.test_samples), self.cast(self.test_logP_preabs), self.cast(self.test_weights)
                )
        else:
            self.validation_dataset = (self.cast(self.test_samples), self.cast(self.test_weights))

        # final feedback
        if self.feedback > 1:
            if self.has_weights:
                print(
                    '    - {}/{} training/test samples and non-uniform weights'.format(
                        self.num_training_samples, self.num_test_samples
                        )
                    )
                print(
                    '    - {0:.6g} effective number of training samples'.format(
                        np.sum(self.training_weights)**2 / np.sum(self.training_weights**2)
                        )
                    )
                print(
                    '    - {0:.6g} effective number of test samples'.format(
                        np.sum(self.test_weights)**2 / np.sum(self.test_weights**2)
                        )
                    )
            else:
                print(
                    '    - {}/{} training/test samples and uniform weights'.format(
                        self.num_training_samples, self.num_test_samples
                        )
                    )
        #
        return None

    def _init_distribution(self):
        """
        Initialize the transformed distributions
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing transformed distribution')

        # full distribution:
        self.base_distribution = tfd.MultivariateNormalDiag(
            tf.zeros(self.num_params, dtype=prec), tf.ones(self.num_params, dtype=prec)
            )
        self.distribution = tfd.TransformedDistribution(
            distribution=self.base_distribution, bijector=self.bijector
            )  # samples from std gaussian mapped to original space
        # abstract space distribution:
        self.trained_distribution = tfd.TransformedDistribution(
            distribution=self.base_distribution, bijector=self.trainable_bijector
            )
        #
        return None

    def _compile_model(self):
        """
        Utility function to compile model
        """
        # reset loss function:
        self.loss.reset()
        # compile model:
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.initial_learning_rate, global_clipnorm=self.global_clipnorm),
            loss=self.loss,
            weighted_metrics=[]
            )
        # we need to rebuild all the self methods that are tf.functions otherwise they might do unwanted caching...
        _self_functions = [func for func in dir(self) if callable(getattr(self, func))]
        # get the methods that are tensorflow functions:
        _tf_functions = [func for func in _self_functions if isinstance(getattr(self, func), tf_defun.Function)]
        # rebuild them (with cloning). I can see this causing memory leaks but I am blaming it on tf
        for func in _tf_functions:
            setattr(self, func, getattr(self, func)._clone(None))
        #
        return None

    def _init_model(
            self, learning_rate=1.e-3, global_clipnorm=1.0, alpha_lossv=1.0, beta_lossv=0.0, loss_mode='standard', **kwargs
        ):
        """
        Initialize the loss function.

        mode can be standard, fixed or variable
        """
        # feedback:
        if self.feedback > 0:
            print('* Initializing loss function')

        # set loss functions relative weights:
        if not self.has_loglikes and not loss_mode == 'standard':
            raise ValueError(
                'Cannot use posterior based loss functions if the input chain does not have posterior values'
                )
        # save in:
        self.alpha_lossv = alpha_lossv
        self.beta_lossv = beta_lossv
        self.initial_learning_rate = learning_rate
        self.final_learning_rate = learning_rate / 1000.
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
        # print feedback:
        if self.feedback > 1:
            self.loss.print_feedback(padding='    - ')
        # build model:
        x_ = Input(shape=(self.num_params,), dtype=prec)
        self.model = Model(x_, self.trained_distribution.log_prob(x_))
        # compile model:
        self._compile_model()
        num_model_params = self.model.count_params()
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
        #
        return None

    def on_epoch_begin(self, epoch, logs):
        """
        Initialization to be done at the beginning of every epoch:
        """
        # update loss function if needed:
        if issubclass(type(self.loss), loss.variable_weight_loss):
            self.model.loss.update_lambda_values_on_epoch_begin(epoch, logs=self.log)
        #
        return None

    def train(self, epochs=100, batch_size=None, steps_per_epoch=None, callbacks=None, verbose=None, **kwargs):
        """
        Train the normalizing flow model. Internally, this runs the fit method of the Keras :class:`~tf.keras.Model`, to which `**kwargs are passed`.

        :param epochs: number of training epochs, defaults to 100.
        :type epochs: int, optional
        :param batch_size: number of samples per batch, defaults to None. If None, the training sample is divided into `steps_per_epoch` batches.
        :type batch_size: int, optional
        :param steps_per_epoch: number of steps per epoch, defaults to None. If None and `batch_size` is also None, then `steps_per_epoch` is set to 100.
        :type steps_per_epoch: int, optional
        :param callbacks: a list of additional Keras callbacks, such as :class:`~tf.keras.callbacks.ReduceLROnPlateau`, defaults to None which contains a selection of useful callbacks.
        :type callbacks: list, optional
        :param verbose: verbosity level, defaults to 1.
        :type verbose: int, optional
        :return: A :class:`~tf.keras.callbacks.History` object. Its `history` attribute is a dictionary of training and validation loss values and metrics values at successive epochs: `"shift0_chi2"` is the squared norm of the zero-shift point in the gaussianized space, with the probability-to-exceed and corresponding tension in `"shift0_pval"` and `"shift0_nsigma"`; `"chi2Z_ks"` and `"chi2Z_ks_p"` contain the :math:`D_n` statistic and probability-to-exceed of the Kolmogorov-Smironov test that squared norms of the transformed samples `Z` are :math:`\\chi^2` distributed (with a number of degrees of freedom equal to the number of parameters).

        batch_size = None
        steps_per_epoch = None
        verbose = None
        callbacks = None
        epochs = 2
        """
        # We're trying to loop through the full sample each epoch
        if batch_size is None:
            if steps_per_epoch is None:
                steps_per_epoch = 20
            batch_size = int(self.num_training_samples / steps_per_epoch)
        else:
            if steps_per_epoch is None:
                steps_per_epoch = int(self.num_training_samples / batch_size)
        # get tensorflow verbosity:
        if verbose is None:
            if self.feedback == 0:
                verbose = 0
            elif self.feedback > 0:
                verbose = 1
        # set callbacks:
        if callbacks is None:
            callbacks = []
            # learning rate scheduler:
            lr_schedule = lr.LRAdaptLossSlopeEarlyStop(**utils.filter_kwargs(kwargs, lr.LRAdaptLossSlopeEarlyStop))
            callbacks.append(lr_schedule)

        # Run training:
        hist = self.model.fit(
            x=self.training_dataset.batch(batch_size),
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.validation_dataset,
            verbose=verbose,
            callbacks=[tf.keras.callbacks.TerminateOnNaN(), self] + callbacks,
            **utils.filter_kwargs(kwargs, self.model.fit)
            )
        # model is now trained:
        self.is_trained = True
        #
        return hist

    def global_train(self, pop_size=10, **kwargs):
        """
        Training algorithm with some globalization strategy. Starts from multiple
        random weight initializations and selects the one that has the best
        performances on the validation set after training.

        :param pop_size: number of weight initializations. Time to solution
        scales linearly with this parameter.
        """
        # initialize:
        best_loss, best_val_loss, best_weights, best_log = None, None, None, None
        loss, val_loss, logs = [], [], []
        ind = 1
        # do the loop:
        while ind <= pop_size:
            # feedback:
            if self.feedback > 0:
                print('* Training population', ind)

            # initialize logs:
            self.log = {_k: [] for _k in self.log.keys()}
            self.log['population'] = ind
            if best_loss is not None:
                self.log['best_loss'] = best_loss

            # build the random weights:
            for layer in self.model.layers:
                layer.build(layer.input_shape)

            # re-compile model:
            self._compile_model()

            # train:
            history = self.train(**kwargs)
            # save log:
            loss.append(history.history['loss'][-1])
            val_loss.append(history.history['val_loss'][-1])
            logs.append(copy.deepcopy(self.log))

            # if improvement save weights:
            if best_val_loss is None:
                best_log = copy.deepcopy(self.log)
                best_loss = copy.deepcopy(history.history['loss'][-1])
                best_val_loss = copy.deepcopy(history.history['val_loss'][-1])
                best_weights = copy.deepcopy(self.model.get_weights())
            else:
                if history.history['val_loss'][-1] < best_val_loss:
                    best_log = copy.deepcopy(self.log)
                    best_loss = copy.deepcopy(history.history['loss'][-1])
                    best_val_loss = copy.deepcopy(history.history['val_loss'][-1])
                    best_weights = copy.deepcopy(self.model.get_weights())

            # update counter:
            ind += 1

        # select best:
        self.model.set_weights(best_weights)
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
        Cast vector/tensor to tensorflow tensor with internal precision of the flow.

        :param v: input vector
        """
        return tf.cast(v, dtype=prec)

    @tf.function()
    def log_probability(self, coord):
        """
        Returns learned log probability in parameter space.

        :param coord: input parameter value
        """
        return self.distribution.log_prob(coord)

    @tf.function()
    def log_probability_jacobian(self, coord):
        """
        Computes the Jacobian of log probability in parameter space.

        :param coord: input parameter value
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.log_probability(coord)
        return tape.gradient(f, coord)

    @tf.function()
    def log_probability_hessian(self, coord):
        """
        Computes the Hessian of log probability in parameter space.

        :param coord: input parameter value
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.log_probability_jacobian(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def log_probability_abs(self, abs_coord):
        """
        Returns learned log probability in original parameter space as a function of abstract coordinates.
        This can be used to perform maximization in abstract space (which has no bounds).

        :param abs_coord: input parameter value in abstract Gaussian coordinates
        """
        temp_1 = self.distribution.distribution.log_prob(abs_coord)
        temp_2 = self.distribution.bijector.forward_log_det_jacobian(abs_coord, event_ndims=1)
        return temp_1 - temp_2

    @tf.function()
    def log_probability_abs_jacobian(self, abs_coord):
        """
        Jacobian of the original parameter space log probability with respect to abstract coordinates.

        :param abs_coord: input parameter value in abstract Gaussian coordinates
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(abs_coord)
            f = self.log_probability_abs(abs_coord)
        return tape.gradient(f, abs_coord)

    @tf.function()
    def log_probability_abs_hessian(self, abs_coord):
        """
        Hessian of the original parameter space log probability with respect to abstract coordinates.

        :param abs_coord: input parameter value in abstract Gaussian coordinates
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(abs_coord)
            f = self.log_probability_abs_jacobian(abs_coord)
        return tape.batch_jacobian(f, abs_coord)

    @tf.function()
    def sample(self, N):
        """
        Return samples from the synthetic probablity.

        :param N: number of samples
        """
        return self.distribution.sample(N)

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

    def evidence(self, indexes=None, weighted=False):
        """
        Get evidence from the flow. Can pass indexes to use only some of the samples for the estimate.
        """
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
        flow_log_likes = self.log_probability(self.cast(_samples))
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
        if not hasattr(self, 'chain_nearest_index'):
            self._init_nearest_samples()
        # get delta log likes and delta params:
        delta_theta = self.chain_samples - self.chain_samples[self.chain_nearest_index[:, 1], :]
        delta_log_likes = -(self.chain_loglikes - self.chain_loglikes[self.chain_nearest_index[:, 1]])
        # compute the gradient:
        delta_1 = tf.einsum(
            "...i, ...i -> ...", self.log_probability_jacobian(self.cast(self.chain_samples)), delta_theta
            ) - delta_log_likes
        delta_2 = tf.einsum(
            "...i, ...i -> ...",
            self.log_probability_jacobian(self.cast(self.chain_samples[self.chain_nearest_index[:, 1], :])), delta_theta
            ) - delta_log_likes
        # average:
        score = np.average(np.abs(0.5 * (delta_1 + delta_2)), weights=self.chain_weights)
        #
        return score

    ###############################################################################
    # Information geometry base methods:

    @tf.function()
    def map_to_abstract_coord(self, coord):
        """
        Map from parameter space to abstract space
        """
        return self.bijector.inverse(coord)

    @tf.function()
    def map_to_original_coord(self, coord):
        """
        Map from abstract space to parameter space
        """
        return self.bijector(coord)

    @tf.function()
    def log_det_metric(self, coord):
        """
        Computes the log determinant of the metric
        """
        log_det = self.bijector.inverse_log_det_jacobian(coord, event_ndims=1)
        if len(log_det.shape) == 0:
            return 2. * log_det * tf.ones_like(coord[..., 0])
        else:
            return 2. * log_det

    @tf.function()
    def direct_jacobian(self, coord):
        """
        Computes the Jacobian of the parameter transformation at one point in (original) parameter space
        """
        abs_coord = self.map_to_abstract_coord(coord)
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(abs_coord)
            f = self.map_to_original_coord(abs_coord)
        return tape.batch_jacobian(f, abs_coord)

    @tf.function()
    def inverse_jacobian(self, coord):
        """
        Computes the inverse Jacobian of the parameter transformation at one point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.map_to_abstract_coord(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def inverse_jacobian_coord_derivative(self, coord):
        """
        Compute the coordinate derivative of the inverse Jacobian at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.inverse_jacobian(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def metric(self, coord):
        """
        Computes the metric at a given point or array of points in (original) parameter space
        """
        # compute Jacobian:
        jac = self.inverse_jacobian(coord)
        # take the transpose (we need to calculate the indexes that we want to swap):
        trailing_axes = [-1, -2]
        leading = tf.range(tf.rank(jac) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(jac)
        new_order = tf.concat([leading, trailing], axis=0)
        jac_T = tf.transpose(jac, new_order)
        # compute metric:
        metric = tf.linalg.matmul(jac_T, jac)
        #
        return metric

    @tf.function()
    def inverse_metric(self, coord):
        """
        Computes the inverse metric at a given point or array of points in (original) parameter space
        """
        # compute Jacobian:
        jac = self.direct_jacobian(coord)
        # take the transpose (we need to calculate the indexes that we want to swap):
        trailing_axes = [-1, -2]
        leading = tf.range(tf.rank(jac) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(jac)
        new_order = tf.concat([leading, trailing], axis=0)
        jac_T = tf.transpose(jac, new_order)
        # compute metric:
        metric = tf.linalg.matmul(jac, jac_T)
        #
        return metric

    @tf.function()
    def coord_metric_derivative(self, coord):
        """
        Compute the coordinate derivative of the metric at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.metric(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def coord_inverse_metric_derivative(self, coord):
        """
        Compute the coordinate derivative of the inverse metric at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.inverse_metric(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def coord_metric_derivative_2(self, coord):
        """
        Compute the second coordinate derivative of the metric at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.coord_metric_derivative(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def coord_inverse_metric_derivative_2(self, coord):
        """
        Compute the second coordinate derivative of the inverse metric at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.coord_inverse_metric_derivative(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def levi_civita_connection(self, coord):
        """
        Compute the Levi-Civita connection, gives Gamma^i_j_k
        """
        inv_metric = self.inverse_metric(coord)
        metric_derivative = self.coord_metric_derivative(coord)
        # first transpose:
        trailing_axes = [-2, -3, -1]
        leading = tf.range(tf.rank(metric_derivative) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(metric_derivative)
        new_order = tf.concat([leading, trailing], axis=0)
        term_1 = tf.transpose(metric_derivative, new_order)
        # second transpose:
        trailing_axes = [-2, -1, -3]
        leading = tf.range(tf.rank(metric_derivative) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(metric_derivative)
        new_order = tf.concat([leading, trailing], axis=0)
        term_2 = tf.transpose(metric_derivative, new_order)
        # third transpose:
        trailing_axes = [-1, -3, -2]
        leading = tf.range(tf.rank(metric_derivative) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(metric_derivative)
        new_order = tf.concat([leading, trailing], axis=0)
        term_3 = tf.transpose(metric_derivative, new_order)
        # compute
        connection = 0.5 * tf.einsum("...ij,...jkl-> ...ikl", inv_metric, term_1 + term_2 - term_3)
        #
        return connection

    @tf.function()
    def geodesic_distance(self, coord_1, coord_2, **kwargs):
        """
        Compute geodesic distance between pair of points
        """
        # map to abstract coordinates:
        abs_coord_1 = self.map_to_abstract_coord(coord_1)
        abs_coord_2 = self.map_to_abstract_coord(coord_2)
        # metric there is Euclidean:
        return tf.linalg.norm(abs_coord_1 - abs_coord_2, **kwargs)

    @tf.function()
    def geodesic_bvp(self, pos_start, pos_end, num_points=1000):
        """
        Solve geodesic boundary value problem.
        """
        # map initial and final positions to abstract space:
        _abs_pos_start = self.map_to_abstract_coord(pos_start)
        _abs_pos_end = self.map_to_abstract_coord(pos_end)
        # get the affine parameter along the geodesic:
        _alpha = tf.linspace(0.0, 1.0, num_points)
        # get the trajectory (a straight line) in abstract space:
        _traj = tf.expand_dims(
            _abs_pos_start, axis=-1
            ) + _alpha * (tf.expand_dims(_abs_pos_end, axis=-1) - tf.expand_dims(_abs_pos_start, axis=-1))
        # take the transpose (we need to calculate the indexes that we want to swap):
        trailing_axes = [-1, -2]
        leading = tf.range(tf.rank(_traj) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(_traj)
        new_order = tf.concat([leading, trailing], axis=0)
        _traj = tf.transpose(_traj, new_order)
        # return map to parameter space:
        #
        return self.map_to_original_coord(_traj)

    @tf.function()
    def geodesic_ivp(self, pos, velocity, solution_times):
        """
        Solve geodesic initial value problem.
        """
        raise NotImplemented

    ###############################################################################
    # caching methods:

    def save(self, outroot):
        """
        Save the flow model to file
        """
        # we need to exclude some TF objects because they cannot be pickled:
        exclude_objects = [
            'prior_bijector', 'base_distribution', 'loss', 'model', 'bijectors', 'fixed_bijector',
            'trainable_transformation', 'trainable_bijector', 'bijector', 'training_dataset', 'distribution',
            'trained_distribution'
            ]

        # get properties that can be pickled and properties that cannot:
        pickle_objects = {}
        for el in self.__dict__:
            if el not in exclude_objects:
                if not type(self.__dict__[el]) == type(tf.function(lambda x: x)):
                    pickle_objects[el] = self.__dict__[el]
        # group and save to pickle all the objects that can be pickled:
        pickle.dump(pickle_objects, open(outroot + '_flow_cache.pickle', 'wb'))

        # save out trainable transformation:
        if self.trainable_transformation is not None:
            self.trainable_transformation.save(outroot)
        #
        return None

    @classmethod
    def load(cls, chain, outroot, **kwargs):
        """
        Load the flow model from file
        """
        # remove trainable bijector path:
        temp = kwargs.pop('trainable_bijector_path', None)
        if temp is not None:
            print('WARNING: trainable_bijector_path is set and will be ignored by load function')
        # re-create the object (we have to do this because we cannot pickle all TF things)
        flow = FlowCallback(chain, trainable_bijector_path=outroot, **kwargs)
        # load the pickle file:
        pickle_objects = pickle.load(open(outroot + '_flow_cache.pickle', 'rb'))
        # load to self:
        for key in pickle_objects:
            setattr(flow, key, pickle_objects[key])
        #
        return flow

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
        _temp_mean = np.average(self.test_samples, axis=0, weights=self.test_weights)
        _temp_invcov = np.linalg.inv(scipy.linalg.sqrtm(np.cov(self.test_samples.T, aweights=self.test_weights)))
        _temp = np.dot(_temp_invcov, (self.test_samples - _temp_mean).T)
        self.chi2Y = np.sum((_temp)**2, axis=0)
        if self.has_weights:
            self.chi2Y = np.random.choice(
                self.chi2Y, size=len(self.chi2Y), replace=True, p=self.test_weights / np.sum(self.test_weights)
                )
        self.chi2Y_ks, self.chi2Y_ks_p = scipy.stats.kstest(self.chi2Y, 'chi2', args=(self.num_params,))
        #
        return None

    def compute_training_metrics(self, logs={}):
        """
        Compute training metrics and append results to internal logs
        """
        # update loss log:
        if "loss" in self.training_metrics:
            self.log["loss"].append(logs.get('loss'))
        if "val_loss" in self.training_metrics:
            self.log["val_loss"].append(logs.get('val_loss'))

        # update learning rate log:
        if "lr" in self.training_metrics:
            self.log["lr"].append(logs.get('lr'))

        # do KS test:
        if "chi2Z_ks" in self.training_metrics:
            self.chi2Z = np.sum(np.array(self.trainable_bijector.inverse(self.test_samples))**2, axis=1)
            # Run KS test
            try:
                # Note that scipy.stats.kstest does not handle weights yet so we need to resample.
                _s = np.isfinite(self.chi2Z)
                self.chi2Z = self.chi2Z[_s]
                if self.has_weights:
                    self.chi2Z = np.random.choice(
                        self.chi2Z,
                        size=len(self.chi2Z),
                        replace=True,
                        p=self.test_weights[_s] / np.sum(self.test_weights[_s])
                        )
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
            # import pdb; pdb.set_trace()
            _train_loss_components = self.loss.compute_loss_components(
                self.cast(self.training_logP_preabs), self.model.call(self.cast(self.training_samples)),
                self.cast(self.training_weights)
                )
            _test_loss_components = self.loss.compute_loss_components(
                self.cast(self.test_logP_preabs), self.model.call(self.cast(self.test_samples)),
                self.cast(self.test_weights)
                )
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
                # self.log["rho_loss"].append(self.model.loss.loss1_top)
                # self.log["ee_loss"].append(self.model.loss.loss2_top)
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
    def _plot_loss(self, ax, logs={}):
        """
        Utility function to plot loss for training and validation samples
        """
        # plot loss lines:
        ax.plot(self.log["loss"], ls='-', lw=1., color='k', label='training')
        ax.plot(self.log["val_loss"], ls='--', lw=1., color='k', label='testing')
        # plot best population loss so far (if any):
        if 'best_loss' in self.log.keys():
            ax.axhline(self.log['best_loss'], ls='--', lw=1., color='tab:blue', label='pop best')
        # finish plot:
        ax.set_title("Loss function")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_yscale('log')
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_lr(self, ax, logs={}):
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
    def _plot_chi2_dist(self, ax, logs={}):
        """
        Utility function to plot chi2 distribution vs histogram.
        """
        xx = np.linspace(0, self.num_params * 4, 1000)
        bins = np.linspace(0, self.num_params * 4, 100)
        ax.plot(
            xx,
            scipy.stats.chi2.pdf(xx, df=self.num_params),
            label='$\\chi^2_{{{}}}$ PDF'.format(self.num_params),
            c='k',
            lw=1.,
            ls='-'
            )
        ax.hist(
            self.chi2Y,
            bins=bins,
            density=True,
            histtype='step',
            weights=self.test_weights,
            label='Pre-NF ($D_n$={:.3f})'.format(self.chi2Y_ks),
            lw=1.,
            ls='-'
            )
        ax.hist(
            self.chi2Z,
            bins=bins,
            density=True,
            histtype='step',
            label='Post-NF ($D_n$={:.3f})'.format(self.log["chi2Z_ks"][-1]),
            lw=1.,
            ls='-'
            )
        ax.set_title(r'$\chi^2_{{{}}}$ PDF'.format(self.num_params))
        ax.set_xlabel(r'$\chi^2$')
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_chi2_ks_p(self, ax, logs={}):
        """
        Utility function to plot the KS test results.
        """
        # KS result probability:
        ln1 = ax.plot(self.log["chi2Z_ks_p"], label='$p$', lw=1., ls='-')
        ax.set_title(r"KS test ($\chi^2$)")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_ylabel(r"$p$-value")
        # KL difference:
        ax2 = ax.twinx()
        ln2 = ax2.plot(self.log["chi2Z_ks"], label='$D_n$', lw=1., ls='--')
        ax2.set_ylabel(r'$D_n$')
        # legend:
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs)
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_density_evidence_error_losses(self, ax, logs={}):
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
    def _plot_lambda_values(self, ax, logs={}):
        """
        Plot balance between the two loss functions
        """
        ax.plot(np.abs(self.log["lambda_1"]), lw=1., ls='-', label='$\\lambda_1$')
        ax.plot(np.abs(self.log["lambda_2"]), lw=1., ls='--', label='$\\lambda_2$')
        ax.set_title(r"Loss function weights")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_ylim([-0.1, 1.1])
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_weighted_density_evidence_error_losses(self, ax, logs={}):
        """
        Plot behavior of density and evidence error loss as training progresses.
        """
        ax.plot(
            np.abs(np.array(self.log["lambda_1"]) * np.array(self.log["rho_loss"])), lw=1., ls='-', color='tab:blue'
            )
        ax.plot(
            np.abs(np.array(self.log["lambda_2"]) * np.array(self.log["ee_loss"])), lw=1., ls='-', color='tab:orange'
            )
        ax.plot(
            np.abs(np.array(self.log["lambda_1"]) * np.array(self.log["val_rho_loss"])),
            lw=1.,
            ls='--',
            color='tab:blue',
            label='density'
            )
        ax.plot(
            np.abs(np.array(self.log["lambda_2"]) * np.array(self.log["val_ee_loss"])),
            lw=1.,
            ls='--',
            color='tab:orange',
            label='evidence error'
            )
        ax.set_title(r"Wighted loss breakdown")
        ax.set_xlabel(r"Epoch $\#$")
        ax.set_yscale('log')
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_losses_rate(self, ax, logs={}, abs_value=False, epoch_range=20):
        """
        Plot evolution of loss function.
        """
        if abs_value:
            if issubclass(type(self.loss), loss.standard_loss):
                ax.plot(np.abs(self.log["loss_rate"]), lw=1., ls='-', label='training')
                ax.plot(np.abs(self.log["val_loss_rate"]), lw=1., ls='-', label='validation')
            elif issubclass(type(self.loss), loss.constant_weight_loss):
                ax.plot(np.abs(self.log["loss_rate"]), lw=1.2, color='k', ls='-', label='all', zorder=2)
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
                ax.plot(self.log["loss_rate"], lw=1.2, color='k', ls='-', label='all', zorder=2)
                ax.plot(self.log["rho_loss_rate"], lw=1., ls='-', label='density', zorder=1)
                ax.plot(self.log["ee_loss_rate"], lw=1., ls='-', label='evidence error', zorder=0)
            elif issubclass(type(self.loss), loss.variable_weight_loss):
                ax.plot(self.log["loss_rate"], lw=1.2, color='k', ls='-', label='all', zorder=2)
                ax.plot(self.log["rho_loss_rate"], lw=1., ls='-', label='density', zorder=1)
                ax.plot(self.log["ee_loss_rate"], lw=1., ls='-', label='evidence error', zorder=0)
        if abs_value:
            ax.set_yscale('log')
        else:
            # plot horizontal line at zero:
            ax.axhline(0., lw=1.0, ls='--', color='k')
            # calculate variance of loss rate for last 30 epochs:
            if 'val_loss_rate' in self.log.keys():
                if len(self.log["val_loss_rate"]) > epoch_range:
                    loss_rate_sig = np.sqrt(np.var(self.log["val_loss_rate"][-epoch_range:]))
                    ax.set_ylim([-3*loss_rate_sig, 3*loss_rate_sig])
            elif 'loss_rate' in self.log.keys():
                if len(self.log["loss_rate"]) > epoch_range:
                    loss_rate_sig = np.sqrt(np.var(self.log["loss_rate"][-epoch_range:]))
                    ax.set_ylim([-3*loss_rate_sig, 3*loss_rate_sig])
            else:
                ax.set_ylim([-1, 1])
        ax.set_title(r"$\Delta$ Loss / epoch")
        ax.set_xlabel(r"Epoch $\#$")
        # legend:
        ax.legend()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def _plot_evidence(self, ax, logs={}):
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
    def _plot_evidence_error(self, ax, logs={}):
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
        Execute on beginning of training
        """
        if not ipython_plotting:
            self._create_figure()
        #
        return None

    @matplotlib.rc_context(plot_options)
    def on_train_end(self, logs):
        """
        Execute at end of training
        """
        if not ipython_plotting:
            del self.fig
            plt.close('all')
        #
        return None

    @matplotlib.rc_context(plot_options)
    def on_epoch_end(self, epoch, logs={}):
        """
        This method is used by Keras to show progress during training if `feedback` is True.
        """

        # update log:
        try:
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            logs['lr'] = 0.0

        # compute metrics:
        self.compute_training_metrics(logs=logs)

        # text monitoring of output:
        if self.feedback > 1:
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

            # create figure:
            if issubclass(type(self.loss), loss.standard_loss):
                gs = self.fig.add_gridspec(nrows=1, ncols=5)
                axes = [self.fig.add_subplot(_g) for _g in gs]
                self._plot_loss(axes[0], logs=logs)
                self._plot_losses_rate(axes[1], logs=logs)
                self._plot_lr(axes[2], logs=logs)
                self._plot_chi2_dist(axes[3], logs=logs)
                self._plot_chi2_ks_p(axes[4], logs=logs)
            elif issubclass(type(self.loss), loss.constant_weight_loss):
                gs = self.fig.add_gridspec(nrows=2, ncols=4)
                axes = [self.fig.add_subplot(_g) for _g in gs]
                self._plot_loss(axes[0], logs=logs)
                self._plot_density_evidence_error_losses(axes[1], logs=logs)
                self._plot_losses_rate(axes[2], logs=logs)
                self._plot_lr(axes[3], logs=logs)
                self._plot_evidence(axes[4], logs=logs)
                self._plot_evidence_error(axes[5], logs=logs)
                self._plot_chi2_dist(axes[6], logs=logs)
                self._plot_chi2_ks_p(axes[7], logs=logs)
            elif issubclass(type(self.loss), loss.variable_weight_loss):
                gs = self.fig.add_gridspec(nrows=2, ncols=5)
                axes = [self.fig.add_subplot(_g) for _g in gs]
                self._plot_loss(axes[0], logs=logs)
                self._plot_density_evidence_error_losses(axes[1], logs=logs)
                self._plot_lambda_values(axes[2], logs=logs)
                self._plot_weighted_density_evidence_error_losses(axes[3], logs=logs)
                self._plot_losses_rate(axes[4], logs=logs)
                self._plot_lr(axes[5], logs=logs)
                self._plot_evidence(axes[6], logs=logs)
                self._plot_evidence_error(axes[7], logs=logs)
                self._plot_chi2_dist(axes[8], logs=logs)
                self._plot_chi2_ks_p(axes[9], logs=logs)

            # plot title:
            if 'population' in self.log.keys():
                plt.suptitle('Training population ' + str(self.log['population']), fontweight='bold')
            # finalize plot:
            plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
            plt.pause(0.00001)
            plt.show()
        #
        return None


###############################################################################
# Transformed flow:


class TransformedFlowCallback(FlowCallback):

    def __init__(self, flow, transformation):
        """
        Applies an analytic bijector to a flow to transform parameters.
        """

        self.num_params = flow.num_params
        if isinstance(transformation, Iterable):
            tmap = transformation
        else:
            tmap = [transformation] * self.num_params

        # New bijector
        split = tfb.Split(self.num_params, axis=-1)
        b = tfb.Chain([tfb.Invert(split), tfb.JointMap(tmap), split])

        # parameter names and labels:
        self.param_names = []
        self.param_labels = []
        for t, name, label in zip(tmap, flow.param_names, flow.param_labels):
            if t.name != '':
                self.param_names.append(t.name + '_' + name)
                self.param_labels.append(t.name + ' ' + label)
            else:
                self.param_names.append(name)
                self.param_labels.append(label)
        # set ranges:
        if flow.parameter_ranges is not None:
            parameter_ranges = {}
            for i, name in enumerate(flow.param_names):
                parameter_ranges[self.param_names[i]] = list(tmap[i](flow.parameter_ranges[name]).numpy())
            self.parameter_ranges = parameter_ranges
        else:
            self.parameter_ranges = None
        # set name tag:
        self.name_tag = flow.name_tag + '_transformed'
        # set sample MAP:
        if flow.sample_MAP is not None:
            self.sample_MAP = np.array([trans(par).numpy() for par, trans in zip(flow.sample_MAP, tmap)])
        else:
            self.sample_MAP = None
        # set chains MAP:
        if flow.chain_MAP is not None:
            self.chain_MAP = np.array([trans(par).numpy() for par, trans in zip(flow.chain_MAP, tmap)])
        else:
            self.chain_MAP = None
        # set bijectors and distribution:
        self.bijectors = [b] + flow.bijectors
        self.bijector = tfb.Chain(self.bijectors)
        self.distribution = tfd.TransformedDistribution(
            distribution=flow.distribution.distribution, bijector=self.bijector
            )
        # MAP:
        if flow.MAP_coord is not None:
            self.MAP_coord = np.array([trans(par).numpy() for par, trans in zip(flow.MAP_coord, tmap)])
            self.MAP_logP = self.log_probability(self.cast(self.MAP_coord))
        else:
            self.MAP_coord = flow.MAP_coord
            self.MAP_logP = flow.MAP_logP

###############################################################################
# Average flow:


class average_flow(FlowCallback):
    
    def __init__(self, flows, **kwargs):
        """
        Initialize the average flow class
        """
        # check parameters and copy in info:
        for flow in flows:
            if flow.param_names != flows[0].param_names:
                raise ValueError('Flow', flow.name_tag, 'does not have the same parameters as', flows[0].name_tag, '. Cannot average.')       
        # copy in infos from the first flow:
        infos = [
                 'name_tag',
                 'feedback',
                 'plot_every',
                 'sample_MAP',
                 'num_params',
                 'param_names',
                 'param_labels',
                 'parameter_ranges',
                 'chain_samples',
                 'chain_loglikes',
                 'has_loglikes',
                 'chain_weights',
                 'is_trained',
                 'MAP_coord',
                 'MAP_logP',
                 ]
        for info in infos:
            self.__dict__[info] = flows[0].__dict__[info]             

        # copy in flows:
        self.flows = flows
        # process:
        self.num_flows = len(self.flows)
        # compute weights:
        self._set_flow_weights()
        #
        return None

    def _set_flow_weights(self, key='val_loss'):
        """
        Compute the relative weights of flows based on validation loss
        """
        # get weights:
        _temp_weights = []
        for flow in self.flows:
            if key not in flow.log.keys():
                _temp_weights.append(np.inf)
            else:
                _temp_weights.append(flow.log[key][-1])
        # compute weighting factors:
        _temp_weights = _temp_weights / np.amin(_temp_weights)
        _temp_weights = np.exp(-_temp_weights)
        # normalize and save:
        self.weights = _temp_weights/np.sum(_temp_weights)
        self.weights = self.cast(self.weights)
        # initialize multinomial over weights for sampling:
        self.weights_prob = tfp.distributions.Multinomial(1, probs=self.weights, validate_args=True)
        self.distribution = tfp.distributions.Mixture(cat=tfp.distributions.Categorical(probs=self.weights),
                                                      components=[flow.distribution for flow in self.flows])
        #
        return None

    def train(self, **kwargs):
        for flow in self.flows:
            flow.train(**kwargs)
        return None

    def global_train(self, **kwargs):
        for flow in self.flows:
            flow.global_train(**kwargs)
        return None

    ###############################################################################
    # Utility functions:

    def cast(self, v):
        return self.flows[0].cast(v)

    @tf.function()
    def log_probability_abs(self, abs_coord):
        raise NotImplementedError('Average flow does not have wel defined abstract coordinates')

    @tf.function()
    def log_probability_abs_jacobian(self, abs_coord):
        raise NotImplementedError('Average flow does not have wel defined abstract coordinates')

    @tf.function()
    def log_probability_abs_hessian(self, abs_coord):
        raise NotImplementedError('Average flow does not have wel defined abstract coordinates')

    @tf.function()
    def sample(self, N):
        # sample from the weights:
        temp_weights = tf.cast(self.weights_prob.sample(N), dtype=tf.int32)
        # count samples:
        counts = tf.reduce_sum(temp_weights, axis=0)
        # go through the flows:
        temp_samples = tf.concat([self.flows[i].sample(counts[i]) for i in range(self.num_flows)], axis=0)
        #
        return tf.random.shuffle(temp_samples)

    ###############################################################################
    # Caching functions:

    def save(self, outroot):
        for i, flow in enumerate(self.flows):
            _outroot = outroot + '_'+str(i)
            flow.save(_outroot)
        return None

    @classmethod
    def load(cls, chain, outroot, num_flows=1, **kwargs):
        # load each flow:
        flows = []
        for i in range(num_flows):
            _outroot = outroot + '_'+str(i)
            flows.append(FlowCallback.load(chain, _outroot, **kwargs))
        # initialize average flow:
        flow = average_flow(flows, **kwargs)
        #
        return flow

###############################################################################
# Flow utilities:


def flow_from_chain(chain, cache_dir=None, root_name='sprob', **kwargs):
    """
    Helper to initialize and train a synthetic probability starting from a chain.
    If a cache directory is specified then training results are cached and
    retreived at later calls.
    """

    # check if we want to create a cache folder:
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    # load from cache:
    if cache_dir is not None and os.path.isfile(cache_dir + '/' + root_name + '_flow_cache.pickle'):
        flow = FlowCallback.load(chain, cache_dir + '/' + root_name, **kwargs)
    else:
        # initialize posterior flow:
        flow = FlowCallback(chain, **kwargs)
        # train posterior flow:
        flow.global_train(**kwargs)
        # save trained model:
        if cache_dir is not None:
            flow.save(cache_dir + '/' + root_name)
    #
    return flow


def average_flow_from_chain(chain, num_flows=1, cache_dir=None, root_name='sprob', **kwargs):
    """
    Helper to initialize and train a synthetic probability starting from a chain.
    If a cache directory is specified then training results are cached and
    retreived at later calls.
    """

    # check if we want to create a cache folder:
    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    # load each flow from cache or compute:
    flows = []
    for i in range(num_flows):
        # get output root:
        if cache_dir is not None:
            _outroot = cache_dir + '/' + root_name + '_'+str(i)
        else:
            _outroot = ''
        # do the list of flows:
        if cache_dir is not None and os.path.isfile(_outroot + '_flow_cache.pickle'):
            flow = FlowCallback.load(chain, cache_dir + '/' + root_name, **kwargs)
            flows.append(flow)
        else:
            # initialize posterior flow:
            flow = FlowCallback(chain, **kwargs)
            # train posterior flow:
            flow.global_train(**kwargs)
            # save trained model:
            if cache_dir is not None:
                flow.save(_outroot)
            flows.append(flow)
    # initialize the average flow:
    _avg_flow = average_flow(flows, **kwargs)
    #
    return _avg_flow