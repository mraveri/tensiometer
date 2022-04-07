"""
For testing purposes:

import getdist
chains_dir = './tensiometer/test_chains/'
settings = {'ignore_rows':0, 'smooth_scale_1D':0.3, 'smooth_scale_2D':0.3}
chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'DES', no_cache=True, settings=settings)
param_names = ['omegam', 'sigma8']
from tensorflow.keras.callbacks import Callback
self = Callback()

from tensiometer import utilities as utils
from tensiometer import gaussian_tension
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
from scipy.optimize import minimize
import scipy.stats
import matplotlib
from matplotlib import pyplot as plt
from collections.abc import Iterable

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

    HAS_FLOW = True
    # tensorflow precision:
    prec = tf.float32
    np_prec = np.float32
except Exception as e:
    print("Could not import tensorflow or tensorflow_probability: ", e)
    Callback = object
    HAS_FLOW = False

try:
    from IPython.display import clear_output
except ModuleNotFoundError:
    pass

# local imports:
from . import lr_schedulers as lr
from . import trainable_bijectors as tb
from . import prior_bijectors as pb


###############################################################################
# loss function for hybrid density and log-likelihood optimization:
def custom_loss(alv=1.0, blv=0.0):
    def loss(y_true_inp, y_pred):
        # unpack the weights:
        y_true, weights = tf.unstack(y_true_inp, axis=1)
        # compute difference between true and predicted likelihoods:
        diffs = (y_true - y_pred)
        # sum weights:
        tot_weights = tf.reduce_sum(weights)
        # compute overall offset:
        mean_diff = tf.reduce_sum(diffs*weights) / tot_weights
        # compute its variance:
        var_diff = tf.reduce_sum((tf.square(diffs - mean_diff))*weights) / tot_weights
        std_diff = tf.sqrt(var_diff)
        # compute density loss function:
        loss_orig = y_pred
        # combine into total loss function:
        loss_comb = -alv*(loss_orig + blv) + (1. - alv)*std_diff
        return loss_comb
    return loss

###############################################################################
# main class to compute NF-based tension:


class FlowCallback(Callback):
    """
    A class to compute the normalizing flow interpolation of a probability density given the samples.

    A normalizing flow is trained to approximate the distribution and then used to numerically evaluate the probablity of a parameter shift (see REF). To do so, it defines a bijective mapping that is optimized to gaussianize the difference chain samples. This mapping is performed in two steps, using the gaussian approximation as pre-whitening. The notations used in the code are:

    * `X` designates samples in the original parameter difference space;
    * `Y` designates samples in the gaussian approximation space, `Y` is obtained by shifting and scaling `X` by its mean and covariance (like a PCA);
    * `Z` designates samples in the gaussianized space, connected to `Y` with a normalizing flow denoted `trainable_bijector`.

    The user may provide the `trainable_bijector` as a :class:`~tfp.bijectors.Bijector` object from `Tensorflow Probability <https://www.tensorflow.org/probability/>`_ or make use of the utility class :class:`~.SimpleMAF` to instantiate a Masked Autoregressive Flow (with `trainable_bijector='MAF'`).

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
        representing the mapping from `Z` to `Y`, or 'MAF' to instantiate a :class:`~.SimpleMAF`, defaults to 'MAF'.
    :type trainable_bijector: optional
    :param learning_rate: initial learning rate, defaults to 1e-3.
    :type learning_rate: float, optional
    :param feedback: feedback level, defaults to 1.
    :type feedback: int, optional
    :param validation_split: fraction of samples to use for the validation sample, defaults to 0.1
    :type validation_split: float, optional
    :reference: George Papamakarios, Theo Pavlakou, Iain Murray (2017). Masked Autoregressive Flow for Density Estimation. `arXiv:1705.07057 <https://arxiv.org/abs/1705.07057>`_
    """

    """
    For testing purposes:
    feedback = 1
    validation_split=0.1
    trainable_bijector='MAF'
    learning_rate=1e-3
    feedback=1
    validation_split=0.1
    kwargs={}
    param_ranges = None
    param_names
    self = FlowCallback(chain, param_names=param_names, feedback=1)
    """

    def __init__(self, chain, param_names=None, param_ranges=None, prior_bijector='ranges', apply_pregauss=True, trainable_bijector='MAF', learning_rate=1e-3, feedback=1, validation_split=0.1, alpha_lossv=1.0, **kwargs):

        # read in varaiables:
        self.feedback = feedback

        # initialize internal samples from chain:
        self._init_chain(chain, param_names=param_names, param_ranges=param_ranges, validation_split=validation_split, prior_bijector=prior_bijector, apply_pregauss=apply_pregauss, trainable_bijector=trainable_bijector, alpha_lossv=alpha_lossv)

        # Transformed distribution
        self._init_transf_dist(trainable_bijector, learning_rate=learning_rate, **kwargs)
        if feedback > 0:
            print("Building flow")
            print("    - trainable parameters:", self.model.count_params())
        # Metrics
        if self.has_loglikes:
            keys = ["loss", "val_loss", "rho_loss", "like_loss", "lr", "chi2Z_ks", "chi2Z_ks_p", "evidence", "evidence_error", "smoothness_score"]
        else:
            keys = ["loss", "val_loss", "lr", "chi2Z_ks", "chi2Z_ks_p"]
        self.log = {_k: [] for _k in keys}

        # compute initial chi2:
        _temp_mean = np.average(self.samples_test, axis=0, weights=self.weights_test)
        _temp_invcov = np.linalg.inv(scipy.linalg.sqrtm(np.cov(self.samples_test.T, aweights=self.weights_test)))
        _temp = np.dot(_temp_invcov, (self.samples_test-_temp_mean).T)
        self.chi2Y = np.sum((_temp)**2, axis=0)
        if self.has_weights:
            self.chi2Y = np.random.choice(self.chi2Y, size=len(self.chi2Y), replace=True, p=self.weights_test/np.sum(self.weights_test))
        self.chi2Y_ks, self.chi2Y_ks_p = scipy.stats.kstest(self.chi2Y, 'chi2', args=(self.num_params,))

        # internal variables:
        self.is_trained = False
        self.MAP_coord = None
        self.MAP_logP = None

    def _init_chain(self, chain, param_names=None, param_ranges=None, validation_split=0.1, prior_bijector='ranges', apply_pregauss=True, trainable_bijector='MAF', alpha_lossv=1.0):
        """
        Add documentation
        """
        # initialize param names:
        if param_names is None:
            param_names = chain.getParamNames().getRunningNames()
        else:
            chain_params = chain.getParamNames().list()
            if not np.all([name in chain_params for name in param_names]):
                raise ValueError('Input parameter is not in the chain.\n',
                                 'Input parameters ', param_names, '\n'
                                 'Possible parameters', chain_params)
        # save name of the flow:
        if chain.name_tag is not None:
            self.name_tag = chain.name_tag+'_flow'
        else:
            self.name_tag = 'flow'
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
                    raise ValueError('Range for parameter ', name, ' is not specified.\n',
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
        # save sample MAP:
        temp = chain.samples[np.argmin(chain.loglikes), :]
        self.sample_MAP = np.array([temp[chain.index[name]] for name in param_names])
        # try to get real best fit:
        try:
            self.chain_MAP = np.array([name.best_fit for name in chain.getBestFit().parsWithNames(param_names)])
        except:
            self.chain_MAP = None
        # Prior bijector setup:
        if prior_bijector == 'ranges':
            # extend slightly the ranges to avoid overflows:
            temp_ranges = []
            for name in param_names:
                temp_range = self.parameter_ranges[name]
                center = 0.5 * (temp_range[0]+temp_range[1])
                length = temp_range[1] - temp_range[0]
                eps = 0.001
                temp_ranges.append({'lower': self.cast(center - 0.5*length*(1.+eps)), 'upper': self.cast(center + 0.5*length*(1.+eps))})
            # define bijector:
            self.prior_bijector = pb.prior_bijector_helper(temp_ranges)
        elif isinstance(prior_bijector, tfp.bijectors.Bijector):
            self.prior_bijector = prior_bijector
        elif prior_bijector is None or prior_bijector is False:
            self.prior_bijector = tfb.Identity()
        self.bijectors = [self.prior_bijector]

        # Samples indices:
        ind = [chain.index[name] for name in param_names]
        self.num_params = len(ind)

        # cache chain samples and log likes:
        self.chain_samples = chain.samples[:, ind]
        self.chain_loglikes = chain.loglikes
        self.has_loglikes = self.chain_loglikes is not None
        self.chain_weights = chain.weights
        # cache nearest neighbours indexes, in whitened coordinates:
        temp_cov = np.cov(self.chain_samples.T, aweights=self.chain_weights)
        white_samples = np.dot(scipy.linalg.sqrtm(np.linalg.inv(temp_cov)), self.chain_samples.T).T
        data_tree = cKDTree(white_samples, balanced_tree=True)
        r2, idx = data_tree.query(white_samples, (2), workers=-1)
        self.chain_nearest_index = idx.copy()

        # Gaussian approximation (full chain)
        if apply_pregauss:
            temp_X = self.prior_bijector.inverse(chain.samples[:, ind]).numpy()
            temp_chain = MCSamples(samples=temp_X, weights=chain.weights, names=param_names)
            temp_gaussian_approx = gaussian_tension.gaussian_approximation(temp_chain, param_names=param_names)
            temp_dist = tfd.MultivariateNormalTriL(loc=self.cast(temp_gaussian_approx.means[0]), scale_tril=tf.linalg.cholesky(self.cast(temp_gaussian_approx.covs[0])))
            self.bijectors.append(temp_dist.bijector)

        self.fixed_bijector = tfb.Chain(self.bijectors)

        # set loss functions relative weights:
        if not self.has_loglikes and alpha_lossv != 1.0:
            raise ValueError('Cannot use likelihood based loss function in absence if the input chain does not have likelihood values')
        # compute beta loss and save in:
        self.alpha_lossv = alpha_lossv
        # feedback:
        if self.feedback:
            print('Weight of density loss: %.3g, weight of likelihood-loss: %.3g' % (self.alpha_lossv, 1.-alpha_lossv))
        # rescale beta term by Gaussian expectation of the first term (so that roughly alpha controls the precision of the Evidence estimate)
        temp_X = self.fixed_bijector.inverse(self.chain_samples).numpy()
        temp_chain = MCSamples(samples=temp_X, weights=self.chain_weights, names=param_names)
        temp_gaussian_approx = gaussian_tension.gaussian_approximation(temp_chain, param_names=param_names)
        self.beta_lossv = np.log(np.linalg.det(2.*np.pi*np.exp(1)*temp_gaussian_approx.covs[0]))/2.
        # feedback:
        if self.feedback:
            print(f'Gaussian density loss asymptotic constant: %.4g' % (self.beta_lossv))

        # Split training/test:
        n = chain.samples.shape[0]
        indices = np.random.permutation(n)
        n_split = int(validation_split*n)
        self.test_idx, self.training_idx = indices[:n_split], indices[n_split:]

        # Training:
        self.samples = self.fixed_bijector.inverse(chain.samples[self.training_idx, :][:, ind]).numpy().astype(np_prec)
        self.jac_true_preabs = self.fixed_bijector.inverse_log_det_jacobian(chain.samples[self.training_idx, :][:, ind], event_ndims=1)
        if self.has_loglikes:
            self.logP_preabs = -1.*self.chain_loglikes[self.training_idx] - self.jac_true_preabs

        self.weights = chain.weights[self.training_idx]
        self.weights *= len(self.weights) / np.sum(self.weights)  # weights normalized to number of samples
        self.has_weights = np.any(self.weights != self.weights[0])
        self.num_training_samples = len(self.samples)

        # Test:
        self.samples_test = self.fixed_bijector.inverse(chain.samples[self.test_idx, :][:, ind]).numpy().astype(np_prec)
        self.jac_true_test_preabs = self.fixed_bijector.inverse_log_det_jacobian(chain.samples[self.test_idx, :][:, ind], event_ndims=1)
        if self.has_loglikes:
            self.logP_preabs_test = -1.*self.chain_loglikes[self.test_idx] - self.jac_true_test_preabs

        # self.Y_test = np.array(self.Y2X_bijector.inverse(self.samples_test.astype(np_prec)))
        self.weights_test = chain.weights[self.test_idx]
        self.weights_test *= len(self.weights_test) / np.sum(self.weights_test)  # weights normalized to number of samples

        # Training sample generator:
        self.training_dataset = tf.data.Dataset.from_tensor_slices((self.cast(self.samples),
                                                                    self.cast(np.array([(self.logP_preabs), (self.weights)]).T),
                                                                    self.cast(self.weights),))
        self.training_dataset = self.training_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()
        self.training_dataset = self.training_dataset.shuffle(self.num_training_samples, reshuffle_each_iteration=True).repeat()

        if self.feedback:
            print("Building training/test samples")
            if self.has_weights:
                print("    - {}/{} training/test samples and non-uniform weights.".format(self.num_training_samples, self.samples_test.shape[0]))
            else:
                print("    - {}/{} training/test samples and uniform weights.".format(self.num_training_samples, self.samples_test.shape[0]))

    def _init_transf_dist(self, trainable_bijector, learning_rate=1e-4, **kwargs):
        """
        Add documentation
        """
        # Model
        if trainable_bijector == 'MAF':
            self.MAF = tb.SimpleMAF(self.num_params, feedback=self.feedback, **kwargs)
            self.trainable_bijector = self.MAF.bijector
        elif isinstance(trainable_bijector, tfp.bijectors.Bijector):
            self.trainable_bijector = trainable_bijector
        elif trainable_bijector is None or trainable_bijector is False:
            self.trainable_bijector = tfb.Identity()
        else:
            raise ValueError

        # Bijector
        self.bijectors.append(self.trainable_bijector)
        self.bijector = tfb.Chain(self.bijectors)

        # Full distribution
        base_distribution = tfd.MultivariateNormalDiag(tf.zeros(self.num_params, dtype=prec), tf.ones(self.num_params, dtype=prec))
        self.distribution = tfd.TransformedDistribution(distribution=base_distribution, bijector=self.bijector)  # samples from std gaussian mapped to original space

        # Construct model (using only trainable bijector)
        x_ = Input(shape=(self.num_params,), dtype=prec)
        log_prob_ = tfd.TransformedDistribution(distribution=base_distribution, bijector=self.trainable_bijector).log_prob(x_)
        self.model = Model(x_, log_prob_)

        # compile model:
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=custom_loss(alv=self.alpha_lossv, blv=self.beta_lossv))

        # feedback:
        if self.feedback:
            print('Maximum learning rate: %.3g' % (learning_rate))

    def train(self, epochs=100, batch_size=None, steps_per_epoch=None, callbacks=None, verbose=1, **kwargs):
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
        """
        # We're trying to loop through the full sample each epoch
        if batch_size is None:
            if steps_per_epoch is None:
                steps_per_epoch = 100
            batch_size = int(self.num_training_samples/steps_per_epoch)
        else:
            if steps_per_epoch is None:
                steps_per_epoch = int(self.num_training_samples/batch_size)
        # set callbacks:
        if callbacks is None:
            callbacks = []
            # learning rate scheduler:
            total_steps = steps_per_epoch * epochs
            initial_lr = self.model.optimizer.lr.numpy()
            # lr_schedule = lr.OneCycleScheduler(self.model.optimizer.lr.numpy(), total_steps, **utils.filter_kwargs(kwargs, lr.OneCycleScheduler))
            lr_schedule = lr.ExponentialDecayScheduler(initial_lr, initial_lr/100., 0.8*total_steps, total_steps, **utils.filter_kwargs(kwargs, lr.ExponentialDecayScheduler))
            # lr_schedule = lr.PowerLawDecayScheduler(initial_lr, initial_lr/100., 0.5, total_steps, **utils.filter_kwargs(kwargs, lr.PowerLawDecayScheduler))
            callbacks.append(lr_schedule)
            # callback that reduces learning rate when it stops improving:
            callbacks.append(keras_callbacks.ReduceLROnPlateau(**utils.filter_kwargs(kwargs, keras_callbacks.ReduceLROnPlateau)))
            # callback to stop if weights start getting worse:
            callbacks.append(keras_callbacks.EarlyStopping(patience=20, restore_best_weights=True,
                                                           **utils.filter_kwargs(kwargs, keras_callbacks.EarlyStopping)))
        # # Run training:
        hist = self.model.fit(x=self.training_dataset.batch(batch_size),
                              batch_size=batch_size,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=(self.samples_test, self.cast(np.array([self.logP_preabs_test, self.weights_test]).T), self.weights_test),
                              verbose=verbose,
                              callbacks=[tf.keras.callbacks.TerminateOnNaN(), self]+callbacks,
                              **utils.filter_kwargs(kwargs, self.model.fit))
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
        # generate starting population of weights:
        population = [self.model.get_weights()]
        for i in range(pop_size-1):
            for layer in self.model.layers:
                layer.build(layer.input_shape)
            population.append(self.model.get_weights())
        # evolve:
        loss, val_loss = [], []
        for i in range(pop_size):
            # feedback:
            if self.feedback:
                print('Training population', i+1)
            # train:
            self.model.set_weights(population[i])
            history = self.train(**kwargs)
            # update stored weights:
            population[i] = self.model.get_weights()
            # save log:
            loss.append(history.history['loss'][-1])
            val_loss.append(history.history['val_loss'][-1])
        loss = np.array(loss)
        val_loss = np.array(val_loss)
        # select best:
        self.model.set_weights(population[np.argmin(val_loss)])
        #
        return population, loss, val_loss

    ###############################################################################
    # Utility functions:

    def cast(self, v):
        """
        Cast vector/tensor to tensorflow tensor with internal precision of the flow.

        :param v: input vector
        """
        return tf.cast(v, dtype=prec)

    @tf.function()
    def sample(self, N):
        """
        Return samples from the synthetic probablity.

        :param N: number of samples
        """
        return self.distribution.sample(N)

    @tf.function()
    def log_probability(self, coord):
        """
        Returns learned log probability.

        :param coord: input parameter value
        """
        return self.distribution.log_prob(coord)

    @tf.function()
    def log_probability_jacobian(self, coord):
        """
        Computes the Jacobian of log probability.

        :param coord: input parameter value
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.log_probability(coord)
        return tape.gradient(f, coord)

    @tf.function()
    def log_probability_abs(self, abs_coord):
        """
        Returns learned log probability in original parameter space as a function of abstract coordinates.

        :param abs_coord: input parameter value in abstract Gaussian coordinates
        """
        temp_1 = self.distribution.distribution.log_prob(abs_coord)
        temp_2 = self.distribution.bijector.forward_log_det_jacobian(abs_coord, event_ndims=1)
        return temp_1 - temp_2

    @tf.function()
    def log_probability_abs_jacobian(self, abs_coord):
        """
        Jacobian of the log probability with respect to abstract coordinates.

        :param abs_coord: input parameter value in abstract Gaussian coordinates
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(abs_coord)
            f = self.log_probability_abs(abs_coord)
        return tape.gradient(f, abs_coord)

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
        mc_samples = MCSamples(samples=samples.numpy(), loglikes=loglikes.numpy(),
                               names=self.param_names, labels=self.param_labels,
                               ranges=self.parameter_ranges,
                               name_tag=self.name_tag, **utils.filter_kwargs(kwargs, MCSamples))
        #
        return mc_samples

    def MAP_finder(self, x0=None, bounds='ranges', **kwargs):
        """
        Function that uses scipy optimizer to find the maximum of the synthetic posterior.
        """
        # find initial guess:
        if x0 is None:
            num_samples = 10000*self.num_params
            temp_samples = self.sample(num_samples)
            temp_probs = self.log_probability(temp_samples)
            x0 = temp_samples[tf.argmax(temp_probs)]
        # find ranges:
        if bounds == 'ranges':
            bounds_to_use = list(self.parameter_ranges.values())
        else:
            bounds_to_use = bounds
        # call to minimizer:
        result = minimize(lambda x: -self.log_probability(self.cast(x)).numpy().astype(np.float64),
                          x0=x0,
                          jac=lambda x: -self.log_probability_jacobian(self.cast(x)).numpy().astype(np.float64),
                          bounds=bounds_to_use,
                          **utils.filter_kwargs(kwargs, minimize)
                          )
        # test result:
        if not result.success:
            print('fast map finder failed')
        # cache MAP value:
        self.MAP_coord = result.x
        self.MAP_logP = -result.fun
        #
        return result

    def sigma_to_length(self, nsigma):
        """
        Approximate proper length of events separated by given number of sigmas.
        This is the inverse of from_chi2_to_sigma, should implement it as such, with the inverse of the asynth expansion.
        """
        return np.sqrt(scipy.stats.chi2.isf(1. - utils.from_sigma_to_confidence(nsigma), self.num_params))

    def evidence(self):
        """
        Get evidence from the flow
        """
        # compute log likes:
        flow_log_likes = self.log_probability(self.cast(self.chain_samples))
        # compute residuals:
        diffs = -self.chain_loglikes - flow_log_likes
        # compute average and error:
        average = np.average(diffs, weights=self.chain_weights)
        variance = np.average((diffs-average)**2, weights=self.chain_weights)
        #
        return (average, np.sqrt(variance))

    def smoothness_score(self):
        """
        Compute smoothness score for the flow. This measures how much the flow is non-linear in between neares neighbours.
        """
        # get delta log likes and delta params:
        delta_theta = self.chain_samples - self.chain_samples[self.chain_nearest_index[:, 1], :]
        delta_log_likes = -(self.chain_loglikes - self.chain_loglikes[self.chain_nearest_index[:, 1]])
        # compute the gradient:
        delta_1 = tf.einsum("...i, ...i -> ...", self.log_probability_jacobian(self.cast(self.chain_samples)), delta_theta) - delta_log_likes
        delta_2 = tf.einsum("...i, ...i -> ...", self.log_probability_jacobian(self.cast(self.chain_samples[self.chain_nearest_index[:, 1], :])), delta_theta) - delta_log_likes
        # average:
        score = np.average(np.abs(0.5*(delta_1 + delta_2)), weights=self.chain_weights)
        #
        return score

    ###############################################################################
    # Information geometry methods:

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
            return 2.*log_det*tf.ones_like(coord[..., 0])
        else:
            return 2.*log_det

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
        # rearrange indexes:
        # term_1 = tf.einsum("...kjl -> ...jkl", metric_derivative)
        # term_2 = tf.einsum("...lik -> ...ikl", metric_derivative)
        # term_3 = tf.einsum("...kli -> ...ikl", metric_derivative)
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
        connection = 0.5*tf.einsum("...ij,...jkl-> ...ikl", inv_metric, term_1 + term_2 - term_3)
        #
        return connection

    @tf.function()
    def geodesic_ode(self, t, y):
        """
        """
        # unpack position and velocity:
        pos = y[:self.num_params]
        vel = y[self.num_params:]
        # compute geodesic equation:
        acc = -tf.einsum("...ijk, ...j, ...k -> ...i", self.levi_civita_connection(tf.convert_to_tensor([pos])), tf.convert_to_tensor([vel]), tf.convert_to_tensor([vel]))
        #
        return tf.concat([vel, acc[0]], axis=0)

    @tf.function()
    def solve_geodesic(self, y_init, yprime_init, solution_times, **kwargs):
        """
        """
        # prepare initial conditions:
        y0 = tf.concat([y_init, yprime_init], axis=0)
        # solve with explicit solver:
        results = tfp.math.ode.DormandPrince(rtol=1.e-4).solve(self.geodesic_ode, initial_time=0., initial_state=y0, solution_times=solution_times, **kwargs)
        #
        return results

    def solve_geodesics_scipy(self, y_init, yprime_init, solution_times, **kwargs):
        """
        """
        # prepare initial conditions:
        y0 = tf.concat([y_init, yprime_init], axis=0)
        # solve with scipy ivp solver:
        results = scipy.integrate.solve_ivp(self.geodesic_ode,
                                            t_span=(np.amin(solution_times), np.amax(solution_times)),
                                            y0=y0,
                                            t_eval=solution_times,
                                            **kwargs)
        #
        return results

    @tf.function()
    def geodesic_distance(self, coord_1, coord_2, **kwargs):
        """
        """
        # map to abstract coordinates:
        abs_coord_1 = self.map_to_abstract_coord(coord_1)
        abs_coord_2 = self.map_to_abstract_coord(coord_2)
        # metric there is Euclidean:
        return tf.linalg.norm(abs_coord_1 - abs_coord_2, **kwargs)

    @tf.function()
    def fast_geodesic_ivp(self, pos, velocity, solution_times):
        """
        """
        pass

    @tf.function()
    def fast_geodesic_bvp(self, pos_start, pos_end, solution_times):
        """
        """
        pass

    def _naive_eigenvalue_ode_abs(self, t, y, reference):
        """
        Solve naively the dynamical equation for eigenvalues in abstract space.
        """
        # preprocess:
        x = self.cast([y])
        # map to original space to compute Jacobian (without inversion):
        x_par = self.map_to_original_coord(x)
        # precompute Jacobian and its derivative:
        jac = self.inverse_jacobian(x_par)[0]
        jac_T = tf.transpose(jac)
        jac_jac_T = tf.matmul(jac, jac_T)
        # compute eigenvalues:
        eig, eigv = tf.linalg.eigh(jac_jac_T)
        temp = tf.matmul(tf.transpose(eigv), tf.transpose([reference]))
        idx = tf.math.argmax(tf.abs(temp))[0]
        w = tf.convert_to_tensor([tf.math.sign(temp[idx]) * eigv[:, idx]])
        #
        return w

    def solve_eigenvalue_ode_abs(self, y0, n, length=1.5, side='both', integrator_options=None, num_points=100, **kwargs):
        """
        Solve eigenvalue problem in abstract space
        side = '+', '-', 'both'
        """
        # define solution points:
        solution_times = tf.linspace(0., length, num_points)
        # compute initial PCA:
        x_abs = tf.convert_to_tensor([y0])
        x_par = self.map_to_original_coord(x_abs)
        jac = self.inverse_jacobian(x_par)[0]
        jac_T = tf.transpose(jac)
        jac_jac_T = tf.matmul(jac, jac_T)
        # compute eigenvalues at initial point:
        eig, eigv = tf.linalg.eigh(jac_jac_T)
        # solve forward:
        if side == '+' or side == 'both':
            # initialize solution:
            temp_sol_1 = np.zeros((num_points-1, self.num_params))
            temp_sol_dot_1 = np.zeros((num_points-1, self.num_params))
            # initialize forward integration:
            solver = scipy.integrate.ode(self._naive_eigenvalue_ode_abs)
            if integrator_options is not None:
                solver.set_integrator(**integrator_options)
            solver.set_initial_value(y0, 0.)
            reference = eigv[:, n]
            yt = y0.numpy()
            yprime = reference
            # do the time steps:
            for ind, t in enumerate(solution_times[1:]):
                # set the reference:
                solver.set_f_params(reference)
                # advance solver:
                try:
                    yt = solver.integrate(t)
                    yprime = self._naive_eigenvalue_ode_abs(t, yt, reference)
                except:
                    pass
                # update reference:
                reference = yprime[0]
                # save out:
                temp_sol_1[ind] = yt.copy()
                temp_sol_dot_1[ind] = yprime.numpy().copy()
            # return if needed:
            if side == '+':
                traj = np.concatenate((x_abs.numpy(), temp_sol_1))
                vel = np.concatenate(([eigv[:, n].numpy()], temp_sol_dot_1))
                return solution_times, traj, vel
        # solve backward:
        if side == '-' or side == 'both':
            # initialize solution:
            temp_sol_2 = np.zeros((num_points-1, self.num_params))
            temp_sol_dot_2 = np.zeros((num_points-1, self.num_params))
            # initialize backward integration:
            solver = scipy.integrate.ode(self._naive_eigenvalue_ode_abs)
            if integrator_options is not None:
                solver.set_integrator(**integrator_options)
            solver.set_initial_value(y0, 0.)
            reference = - eigv[:, n]
            yt = y0.numpy()
            yprime = reference
            for ind, t in enumerate(solution_times[1:]):
                # set the reference:
                solver.set_f_params(reference)
                # advance solver:
                try:
                    yt = solver.integrate(t)
                    yprime = self._naive_eigenvalue_ode_abs(t, yt, reference)
                except:
                    pass
                # update reference:
                reference = yprime[0]
                # save out:
                temp_sol_2[ind] = yt.copy()
                temp_sol_dot_2[ind] = yprime.numpy().copy()
            # return if needed:
            if side == '-':
                traj = np.concatenate((temp_sol_2[::-1], x_abs.numpy()))
                vel = np.concatenate((-temp_sol_dot_2[::-1], [eigv[:, n].numpy()]))
                return -solution_times, traj, vel
        # patch solutions:
        times = np.concatenate((-solution_times[::-1], solution_times[1:]))
        traj = np.concatenate((temp_sol_2[::-1], x_abs.numpy(), temp_sol_1))
        vel = np.concatenate((-temp_sol_dot_2[::-1], [eigv[:, n].numpy()], temp_sol_dot_1))
        #
        return times, traj, vel

    def solve_eigenvalue_ode_par(self, y0, n, **kwargs):
        """
        Solve eigenvalue ODE in parameter space
        """
        # go to abstract space:
        x_abs = self.map_to_abstract_coord(self.cast([y0]))[0]
        # call solver:
        times, traj, vel = self.solve_eigenvalue_ode_abs(x_abs, n, **kwargs)
        # convert back:
        traj = self.map_to_original_coord(self.cast(traj))
        #
        return times, traj

    # solve full transport in abstract space:
    @tf.function()
    def eigenvalue_ode_abs_temp_3(self, t, y):
        # unpack y:
        x = self.cast([y[:self.num_params]])
        w = self.cast([y[self.num_params:-1]])
        alpha = self.cast([y[-1]])
        # map to original space to compute Jacobian (without inversion):
        x_par = self.map_to_original_coord(x)
        # precompute Jacobian and its derivative:
        jac = self.inverse_jacobian(x_par)[0]
        #djac = coord_jacobian_derivative(x_par)[0]
        jacm1 = self.direct_jacobian(x_par)[0]
        jac_T = tf.transpose(jac)
        jac_jac_T = tf.matmul(jac, jac_T)
        Id = tf.eye(self.num_params)
        # select the eigenvector that we want to follow based on the solution to the continuity equation:
        eig, eigv = tf.linalg.eigh(jac_jac_T)
        idx = tf.math.argmax(tf.abs(tf.matmul(tf.transpose(eigv), tf.transpose(w))))[0]
        tilde_w = tf.convert_to_tensor([eigv[:, idx]])
        dot_J = tf.einsum('k, lk, ijl -> ji', tilde_w[0], jacm1, djac)
        # equation for alpha:
        alpha_dot = 2.*tf.matmul(tf.matmul(tilde_w, jac), tf.matmul(dot_J, tf.transpose(tilde_w)))
        # equation for wdot:
        wdot_lhs = (jac_jac_T - tf.matmul(tf.matmul(tilde_w, jac_jac_T), tf.transpose(tilde_w))*Id)
        wdot_rhs = tf.matmul(alpha_dot - tf.matmul(dot_J, jac_T) - tf.matmul(jac, tf.transpose(dot_J)), tf.transpose(tilde_w))
        w_dot = tf.linalg.lstsq(wdot_lhs, wdot_rhs, fast=False)
        w_dot = tf.matmul((Id - tf.einsum('i,j->ij', tilde_w[0], tf.transpose(tilde_w[0]))), w_dot)
        # equation for w:
        x_dot = tf.transpose(tilde_w)
        #
        return tf.transpose(tf.concat([x_dot, w_dot, alpha_dot], axis=0))[0]

    ###############################################################################
    # Training statistics:

    def _compute_shift_proba(self):
        zero = np.array(self.bijector.inverse(np.zeros(self.num_params, dtype=np_prec)))
        chi2Z0 = np.sum(zero**2)
        pval = scipy.stats.chi2.cdf(chi2Z0, df=self.num_params)
        nsigma = utils.from_confidence_to_sigma(pval)
        return zero, chi2Z0, pval, nsigma

    def _plot_loss(self, ax, logs={}):
        """
        Utility function to plot loss for training and validation samples
        """
        self.log["loss"].append(logs.get('loss'))
        self.log["val_loss"].append(logs.get('val_loss'))
        if ax is not None:
            ax.plot(np.abs(self.log["loss"]), ls='-', lw=1., label='Training')
            ax.plot(np.abs(self.log["val_loss"]), ls='-', lw=1., label='Testing')
            ax.set_title("Training Loss")
            ax.set_xlabel(r"Epoch $\#$")
            ax.set_ylabel("Loss")
            ax.set_yscale('log')
            ax.legend()

    def _plot_lr(self, ax, logs={}):
        """
        Utility function to plot learning rate per epoch
        """
        self.log["lr"].append(logs.get('lr'))
        if ax is not None:
            ax.plot(self.log["lr"], ls='-', lw=1.)
            ax.set_title("Learning rate")
            ax.set_xlabel(r"Epoch $\#$")
            ax.set_ylabel("Rate")
            ax.set_yscale('log')

    def _plot_chi2_dist(self, ax, logs={}):
        """
        Utility function to compute KS test and plot chi2 distribution vs histogram.
        """
        # Compute chi2:
        chi2Z = np.sum(np.array(self.trainable_bijector.inverse(self.samples_test))**2, axis=1)
        # make sure some are finite:
        _s = np.isfinite(chi2Z)
        assert np.any(_s)
        chi2Z = chi2Z[_s]

        # Run KS test
        try:
            # Note that scipy.stats.kstest does not handle weights yet so we need to resample.
            if self.has_weights:
                chi2Z = np.random.choice(chi2Z, size=len(chi2Z), replace=True, p=self.weights_test[_s]/np.sum(self.weights_test[_s]))
            chi2Z_ks, chi2Z_ks_p = scipy.stats.kstest(chi2Z, 'chi2', args=(self.num_params,))
        except:
            chi2Z_ks, chi2Z_ks_p = 0., 0.
        self.log["chi2Z_ks"].append(chi2Z_ks)
        self.log["chi2Z_ks_p"].append(chi2Z_ks_p)

        # Plot PDF results:
        if ax is not None:
            xx = np.linspace(0, self.num_params*4, 1000)
            bins = np.linspace(0, self.num_params*4, 100)
            ax.plot(xx, scipy.stats.chi2.pdf(xx, df=self.num_params), label='$\\chi^2_{{{}}}$ PDF'.format(self.num_params), c='k', lw=1., ls='-')
            ax.hist(self.chi2Y, bins=bins, density=True, histtype='step', weights=self.weights_test, label='Pre-gauss ($D_n$={:.3f})'.format(self.chi2Y_ks), lw=1., ls='-')
            ax.hist(chi2Z, bins=bins, density=True, histtype='step', weights=self.weights_test[_s], label='Post-gauss ($D_n$={:.3f})'.format(chi2Z_ks), lw=1., ls='-')
            ax.set_title(r'$\chi^2_{{{}}}$ PDF'.format(self.num_params))
            ax.set_xlabel(r'$\chi^2$')
            ax.legend(fontsize=8)

    def _plot_chi2_ks_p(self, ax, logs={}):
        """
        Utility function to plot the KS test results.
        """
        # Plot
        if ax is not None:
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
            lns = ln1+ln2
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc=1)

    def _plot_losses(self, ax, logs={}):
        """
        Plot behavior of density and likelihood loss as training progresses.
        """
        # compute density loss on validation data:
        temp_rho_loss = custom_loss(1.0, self.beta_lossv)(self.cast(np.array([self.logP_preabs_test, self.weights_test]).T), self.model.predict(self.samples_test))
        temp_rho_loss = np.average(temp_rho_loss.numpy(), weights=self.weights_test)
        # compute likelihood loss on validation data:
        temp_like_loss = custom_loss(0.0, self.beta_lossv)(self.cast(np.array([self.logP_preabs_test, self.weights_test]).T), self.model.predict(self.samples_test))
        temp_like_loss = np.average(temp_like_loss.numpy(), weights=self.weights_test)
        # add to log:
        self.log["rho_loss"].append(temp_rho_loss)
        self.log["like_loss"].append(temp_like_loss)
        # plot:
        if ax is not None:
            # evidence error:
            ax.plot(np.abs(self.log["rho_loss"]), lw=1., ls='-', label='density loss')
            ax.plot(np.abs(self.log["like_loss"]), lw=1., ls='-', label='likelihood loss')
            ax.set_title(r"Loss breakdown")
            ax.set_xlabel(r"Epoch $\#$")
            ax.set_yscale('log')
            # legend:
            ax.legend(loc=1)

    def _plot_evidence_error(self, ax, logs={}):
        """
        Utility function to plot the evidence and error on evidence as a function of training.
        """
        # compute evidence:
        evidence, evidence_error = self.evidence()
        self.log["evidence"].append(evidence)
        self.log["evidence_error"].append(evidence_error)
        # plot:
        if ax is not None:
            # evidence error:
            ln1 = ax.plot(self.log["evidence_error"], lw=1., ls='-', label='var $\\log \\mathcal{E}$')
            ax.set_title(r"Flow evidence")
            ax.set_xlabel(r"Epoch $\#$")
            ax.set_ylabel(r"Evidence error")
            ax.set_yscale('log')
            # evidence value:
            ax2 = ax.twinx()
            ln2 = ax2.plot(self.log["evidence"], lw=1., ls='--', label='$\\log \\mathcal{E}$')
            ax2.set_ylabel(r'Evidence')
            # legend:
            lns = ln1+ln2
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc=1)

    def _plot_smoothness_score(self, ax, logs={}):
        """
        Utility function to plot the smoothness score as a function of training.
        """
        # compute and plot smoothness score during training:
        score = self.smoothness_score()
        self.log["smoothness_score"].append(score)
        # plot:
        if ax is not None:
            ax.plot(self.log["smoothness_score"], ls='-', lw=1.)
            ax.set_title(r"Smoothness score")
            ax.set_xlabel("Epoch #")
            ax.set_ylabel(r"$\langle \, J\cdot \Delta \theta - \Delta \log P \, \rangle$")
            ax.set_yscale('log')

    def on_epoch_end(self, epoch, logs={}):
        """
        This method is used by Keras to show progress during training if `feedback` is True.
        """

        # update log:
        try:
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            logs['lr'] = 0.0

        # decide whether to plot:
        do_plots = self.feedback and matplotlib.get_backend() != 'agg'
        if do_plots and isinstance(self.feedback, int):
            if ((epoch + 1) % self.feedback) > 0:
                do_plots = False

        # do the plots:
        if do_plots:
            clear_output(wait=True)
            if self.has_loglikes:
                fig, axes = plt.subplots(2, 4, figsize=(16, 6))
            else:
                fig, axes = plt.subplots(1, 4, figsize=(16, 3))
        else:
            if self.has_loglikes:
                axes = np.array([[None]*4, [None]*4])
            else:
                axes = np.array([None]*4)

        # do the plot and update logs:
        self._plot_loss(axes[0, 0], logs=logs)
        self._plot_lr(axes[0, 1], logs=logs)
        self._plot_chi2_dist(axes[0, 2], logs=logs)
        self._plot_chi2_ks_p(axes[0, 3], logs=logs)
        if self.has_loglikes:
            self._plot_losses(axes[1, 0], logs=logs)
            self._plot_evidence_error(axes[1, 1], logs=logs)
            self._plot_smoothness_score(axes[1, 2], logs=logs)
            if axes[1, 3] is not None:
                axes[1, 3].axis('off')

        # save out log:
        for k in self.log.keys():
            logs[k] = self.log[k][-1]

        # show figure:
        if do_plots:
            plt.tight_layout()
            plt.show()
            return fig

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
            tmap = [transformation]*self.num_params

        # New bijector
        split = tfb.Split(self.num_params, axis=-1)
        b = tfb.Chain([tfb.Invert(split), tfb.JointMap(tmap), split])

        # parameter names and labels:
        self.param_names = []
        self.param_labels = []
        for t, name, label in zip(tmap, flow.param_names, flow.param_labels):
            if t.name != '':
                self.param_names.append(t.name+'_'+name)
                self.param_labels.append(t.name+' '+label)
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
        self.name_tag = flow.name_tag+'_transformed'
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
        self.distribution = tfd.TransformedDistribution(distribution=flow.distribution.distribution, bijector=self.bijector)
        # MAP:
        if flow.MAP_coord is not None:
            self.MAP_coord = np.array([trans(par).numpy() for par, trans in zip(flow.MAP_coord, tmap)])
            self.MAP_logP = self.log_probability(self.cast(self.MAP_coord))
        else:
            self.MAP_coord = flow.MAP_coord
            self.MAP_logP = flow.MAP_logP

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
    if cache_dir is not None and os.path.isfile(cache_dir+'/'+root_name+'_permutations.pickle'):
        # load trained model:
        temp_MAF = tb.SimpleMAF.load(cache_dir+'/'+root_name, **kwargs)
        # initialize flow:
        if 'trainable_bijector' in kwargs:
            kwargs.pop('trainable_bijector')
        flow = FlowCallback(chain, trainable_bijector=temp_MAF.bijector, **kwargs)
    else:
        # initialize posterior flow:
        flow = FlowCallback(chain, **kwargs)
        # train posterior flow:
        flow.global_train(**kwargs)
        # save trained model:
        if cache_dir is not None:
            flow.MAF.save(cache_dir+'/'+root_name)
    #
    return flow
