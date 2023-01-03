"""
This file contains the loss functions for the normalizing flow training.

Since we are combining different loss functions we have different options.
"""

###############################################################################
# initial imports and set-up:

import numpy as np

# tensorflow imports:
import tensorflow as tf
from keras.utils import tf_utils
from keras import backend
from keras.utils import losses_utils

###############################################################################
# standard normalizing flow loss function:


class standard_loss(tf.keras.losses.Loss):

    def __init__(self):
        """
        Initialize loss function
        """
        # initialize:
        super(standard_loss, self).__init__()
        #
        return None

    def compute_loss_components(self, y_true, y_pred, sample_weight):
        """
        Compute different components of the loss function
        """
        return -y_pred

    def call(self, y_true, y_pred):
        """
        Standard normalizing flow loss function is KL divergence of two abstract
        distributions.
        """
        return -y_pred

###############################################################################
# density and evidence loss with constant weights:


class constant_weight_loss(tf.keras.losses.Loss):

    def __init__(self, alpha=1.0, beta=0.0):
        """
        Initialize loss function
        """
        # initialize:
        super(constant_weight_loss, self).__init__()
        # set parameters:
        self.alpha = alpha
        self.beta = beta
        #
        return None

    def compute_loss_components(self, y_true, y_pred, sample_weight):
        """
        Compute different components of the loss function
        """
        # compute difference between true and predicted likelihoods:
        diffs = (y_true - y_pred)
        # sum weights:
        tot_weights = tf.reduce_sum(sample_weight)
        # compute overall offset:
        mean_diff = tf.reduce_sum(diffs*sample_weight) / tot_weights
        # compute its variance:
        var_diff = tf.abs(diffs - mean_diff)
        # compute density loss function:
        loss_orig = -(y_pred + self.beta)
        #
        return loss_orig, var_diff

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Combine density and likelihood loss
        """
        # get components:
        loss_1, loss_2 = self.compute_loss_components(y_true, y_pred, sample_weight)
        #
        return +self.alpha*loss_1 + (1. - self.alpha)*loss_2

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        This function overrides the standard tensorflow one to pass along weights.
        """
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight
            )
        with backend.name_scope(self._name_scope), graph_ctx:
            if tf.executing_eagerly():
                call_fn = self.compute_loss
            else:
                call_fn = tf.__internal__.autograph.tf_convert(self.compute_loss, tf.__internal__.autograph.control_status_ctx())
            losses = call_fn(y_true, y_pred, sample_weight)
            return losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=self._get_reduction()
                )

###############################################################################
# density and evidence loss with variable weights:


class variable_weight_loss(tf.keras.losses.Loss):

    def __init__(self, lambda_1=1.0, lambda_2=0.0, beta=0.0):
        """
        Initialize loss function
        """
        # initialize:
        super(variable_weight_loss, self).__init__()
        # set parameters:
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.beta = beta
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        # base class is empty...
        return None

    def compute_loss_components(self, y_true, y_pred, sample_weight):
        """
        Compute different components of the loss function
        """
        # compute difference between true and predicted likelihoods:
        diffs = (y_true - y_pred)
        # sum weights:
        tot_weights = tf.reduce_sum(sample_weight)
        # compute overall offset:
        mean_diff = tf.reduce_sum(diffs*sample_weight) / tot_weights
        # compute its variance:
        var_diff = tf.abs(diffs - mean_diff)
        # compute density loss function:
        loss_orig = -(y_pred + self.beta)
        #
        return loss_orig, var_diff, self.lambda_1, self.lambda_2

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Combine density and likelihood loss
        """
        # get components:
        loss_1, loss_2, lambda_1, lambda_2 = self.compute_loss_components(y_true, y_pred, sample_weight)
        #
        return lambda_1*loss_1 + lambda_2*loss_2

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        This function overrides the standard tensorflow one to pass along weights.
        """
        graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
            y_true, y_pred, sample_weight
            )
        with backend.name_scope(self._name_scope), graph_ctx:
            if tf.executing_eagerly():
                call_fn = self.compute_loss
            else:
                call_fn = tf.__internal__.autograph.tf_convert(self.compute_loss, tf.__internal__.autograph.control_status_ctx())
            losses = call_fn(y_true, y_pred, sample_weight)
            return losses_utils.compute_weighted_loss(
                losses, sample_weight, reduction=self._get_reduction()
                )


class random_weight_loss(variable_weight_loss):
    """
    Random weighting of the two loss functions.
    """

    def __init__(self, initial_random_epoch=0, lambda_1=1.0, beta=0.0, **kwargs):
        """
        Initialize loss function
        """
        # initialize:
        super(variable_weight_loss, self).__init__()
        # set parameters:
        self.lambda_1 = lambda_1
        self.lambda_2 = 1. - self.lambda_1
        self.initial_random_epoch = initial_random_epoch
        self.beta = beta
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        if epoch > self.initial_random_epoch:
            self.lambda_1 = np.random.randint(2)
            self.lambda_2 = 1. - self.lambda_1
        #
        return None


class annealed_weight_loss(variable_weight_loss):
    """
    Slowly go from density to likelihood loss.
    """

    def __init__(self, initial_random_epoch=50, lambda_1=1.0, beta=0.0, roll_off_nepoch=10, **kwargs):
        """
        Initialize loss function
        """
        # initialize:
        super(variable_weight_loss, self).__init__()
        # set parameters:
        self.lambda_1 = lambda_1
        self.lambda_2 = 1. - self.lambda_1
        self.initial_random_epoch = initial_random_epoch
        self.beta = beta
        self.roll_off_nepoch = roll_off_nepoch
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        if epoch > self.initial_random_epoch:
            self.lambda_1 *= np.exp(-1.*(epoch - self.initial_random_epoch)/self.roll_off_nepoch)
            self.lambda_2 = 1. - self.lambda_1
        #
        return None


class SoftAdapt_weight_loss(variable_weight_loss):
    """
    Implement SoftAdapt as in arXiv:1912.12355, with optional smoothing
    """

    def __init__(self, tau=1.0, beta=0.0, smoothing=True, smoothing_tau=5, **kwargs):
        """
        Initialize loss function
        """
        # initialize:
        super(variable_weight_loss, self).__init__()
        # set parameters:
        self.tau = tau
        self.beta = beta
        self.smoothing = smoothing
        self.smoothing_alpha = 1. / smoothing_tau
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        # get logs:
        logs = kwargs.get('logs')
        # get the two rates:
        like_loss_rate = logs['like_loss_rate']
        rho_loss_rate = logs['rho_loss_rate']
        # protect for initial phase:
        if len(rho_loss_rate) == 0 or rho_loss_rate[-1] == 0.0:
            self.lambda_1 = 1.0
        else:
            lambda_1 = np.exp(self.tau * rho_loss_rate[-1])
            lambda_2 = np.exp(self.tau * like_loss_rate[-1])
            _tot = lambda_1 + lambda_2
            if self.smoothing:
                self.lambda_1 = self.smoothing_alpha * lambda_1 / _tot + (1. - self.smoothing_alpha) * self.lambda_1
            else:
                self.lambda_1 = lambda_1 / _tot
        # set second by enforcing sum to one:
        self.lambda_2 = 1. - self.lambda_1
        #
        return None
