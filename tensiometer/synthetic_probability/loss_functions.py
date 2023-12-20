"""
This file contains the loss functions for the normalizing flow training.

Since we are combining different loss functions we have different options.
"""

###############################################################################
# initial imports and set-up:

import numpy as np

# tensorflow imports:
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils

###############################################################################
# standard normalizing flow loss function:


class standard_loss(tf.keras.losses.Loss):

    def __init__(self):
        """
        Standard density loss function for the normalizing flow.
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
    
    def print_feedback(self, padding=''):
        """
        Print feedback to screen
        """
        print(padding+'using standard loss function')

    def reset(self):
        """
        Reset loss functions hyper parameters
        """
        pass


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
        # compute difference between true and predicted posterior values:
        diffs = (y_true - y_pred)
        # sum weights:
        tot_weights = tf.reduce_sum(sample_weight)
        # compute overall offset:
        mean_diff = tf.reduce_sum(diffs*sample_weight) / tot_weights
        # compute its variance:
        var_diff = (diffs - mean_diff)**2
        # compute density loss function:
        loss_orig = -(y_pred + self.beta)
        #
        return loss_orig, var_diff

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Combine density and evidence-error loss
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

    def print_feedback(self, padding=''):
        """
        Print feedback to screen
        """
        print(padding+'using combined density and evidence-error loss function')
        print(padding+'weight of density loss: %.3g, weight of evidence-error-loss: %.3g' % (self.alpha, 1.-self.alpha))

    def reset(self):
        """
        Reset loss functions hyper parameters
        """
        pass

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
        self.lambda_1 = tf.Variable(lambda_1, trainable=False, name='loss_lambda_1', dtype=type(lambda_1))
        self.lambda_2 = tf.Variable(lambda_2, trainable=False, name='loss_lambda_2', dtype=type(lambda_2))
        self.beta = tf.Variable(beta, trainable=False, name='loss_beta', dtype=type(beta))
        # save initial parameters:
        self.initial_lambda_1 = lambda_1
        self.initial_lambda_2 = lambda_2
        self.initial_beta = beta
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        # base class is empty...
        # use the following sintax:
        # tf.keras.backend.set_value(self.lambda_1, tf.constant(0.5) * epoch)
        raise NotImplementedError

    def compute_loss_components(self, y_true, y_pred, sample_weight, lambda_1=None, lambda_2=None):
        """
        Compute different components of the loss function
        """
        # compute difference between true and predicted posterior values:
        diffs = (y_true - y_pred)
        # sum weights:
        tot_weights = tf.reduce_sum(sample_weight)
        # compute overall offset:
        mean_diff = tf.reduce_sum(diffs*sample_weight) / tot_weights
        # compute its variance:
        var_diff = (diffs - mean_diff)**2
        # compute density loss function:
        loss_orig = -(y_pred + self.beta)
        # get weights if not passed:
        if lambda_1 is None:
            lambda_1 = tf.keras.backend.get_value(self.lambda_1)
        if lambda_2 is None:
            lambda_2 = tf.keras.backend.get_value(self.lambda_2)
        #
        return loss_orig, var_diff, lambda_1, lambda_2

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Combine density and evidence-error loss
        """
        # get components:
        loss_1, loss_2, lambda_1,lambda_2 = self.compute_loss_components(y_true, y_pred, sample_weight, self.lambda_1, self.lambda_2)
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
        
    def print_feedback(self, padding=''):
        """
        Print feedback to screen
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset loss functions hyper parameters
        """
        self.__init__(lambda_1=self.initial_lambda_1, lambda_2=self.initial_lambda_2, beta=self.initial_beta)


class random_weight_loss(variable_weight_loss):
    """
    Random weighting of the two loss functions.
    """

    def __init__(self, initial_random_epoch=0, lambda_1=1.0, beta=0.0, **kwargs):
        """
        Initialize loss function
        """
        # initialize:
        super(random_weight_loss, self).__init__(lambda_1, 1.-lambda_1, beta)
        # set parameters:
        self.initial_random_epoch = initial_random_epoch
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        if epoch > self.initial_random_epoch:
            _temp_rand = np.random.randint(2)
            tf.keras.backend.set_value(self.lambda_1, _temp_rand)
            tf.keras.backend.set_value(self.lambda_2, 1.-_temp_rand)
        #        
        return None

    def print_feedback(self, padding=''):
        """
        Print feedback to screen
        """
        print(padding+'using randomized loss function')


class annealed_weight_loss(variable_weight_loss):
    """
    Slowly go from density to evidence-error loss.
    """

    def __init__(self, anneal_epoch=125, lambda_1=1.0, beta=0.0, roll_off_nepoch=10, **kwargs):
        """
        Initialize loss function
        """
        # initialize:
        super(annealed_weight_loss, self).__init__(lambda_1, 1.-lambda_1, beta)
        # set parameters:
        self.anneal_epoch = anneal_epoch
        self.roll_off_nepoch = roll_off_nepoch
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        if epoch > self.anneal_epoch:
            _lambda_1 = tf.keras.backend.get_value(self.lambda_1)
            _lambda_1 *= np.exp(-1.*(epoch - self.anneal_epoch)/self.roll_off_nepoch)
            tf.keras.backend.set_value(self.lambda_1, _lambda_1)
            tf.keras.backend.set_value(self.lambda_2, 1.-_lambda_1)
        #
        return None

    def print_feedback(self, padding=''):
        """
        Print feedback to screen
        """
        print(padding+'using annealed loss function')


class SoftAdapt_weight_loss(variable_weight_loss):
    """
    Implement SoftAdapt as in arXiv:1912.12355, with optional smoothing
    """

    def __init__(self, tau=1.0, beta=0.0, smoothing=True, smoothing_tau=20, quantity_1='val_rho_loss', quantity_2='val_ee_loss', **kwargs):
        """
        Initialize loss function
        """
        # initialize:
        super(SoftAdapt_weight_loss, self).__init__()
        # set parameters:
        self.tau = tau
        self.beta = beta
        self.smoothing = smoothing
        self.smoothing_alpha = 1. / smoothing_tau
        # quantrities to be monitored:
        self.quantity_1 = quantity_1 
        self.quantity_2 = quantity_2 
        self.rate_1_buffer = 0.0
        self.rate_2_buffer = 0.0
        self.lambda_1_buffer = 1.0
        self.lambda_2_buffer = 0.0
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        # get logs:
        logs = kwargs.get('logs')
        quantity_1 = logs[self.quantity_1]
        quantity_2 = logs[self.quantity_2]
        # get the two rates:
        if len(quantity_1) < 2:
            rate_1 = 0.0
        else:
            rate_1 = quantity_1[-1] - quantity_1[-2]
        if len(quantity_2) < 2:
            rate_2 = 0.0
        else:
            rate_2 = quantity_2[-1] - quantity_2[-2]
        # smooth the two rates:
        if self.smoothing:
            rate_1 = self.smoothing_alpha * rate_1 + (1. - self.smoothing_alpha) * self.rate_1_buffer
            rate_2 = self.smoothing_alpha * rate_2 + (1. - self.smoothing_alpha) * self.rate_2_buffer
            self.rate_1_buffer = rate_1
            self.rate_2_buffer = rate_2
        # protect for initial phase:
        if rate_1 == 0.0:
            _lambda_1 = 1.0
        else:
            lambda_1 = np.exp(self.tau * rate_1)
            lambda_2 = np.exp(self.tau * rate_2)
            _tot = lambda_1 + lambda_2
            if self.smoothing:
                _lambda_1 = self.smoothing_alpha * lambda_1 / _tot + (1. - self.smoothing_alpha) * self.lambda_1_buffer
                _lambda_2 = self.smoothing_alpha * lambda_2 / _tot + (1. - self.smoothing_alpha) * self.lambda_2_buffer
                _lambda_1 = _lambda_1 / (_lambda_1 + _lambda_2)
                _lambda_2 = _lambda_2 / (_lambda_1 + _lambda_2)
                self.lambda_1_buffer = _lambda_1
                self.lambda_2_buffer = _lambda_2
            else:
                _lambda_1 = lambda_1 / _tot
        # set second by enforcing sum to one:
        tf.keras.backend.set_value(self.lambda_1, _lambda_1)
        tf.keras.backend.set_value(self.lambda_2, 1.-_lambda_1)
        #
        return None

    def print_feedback(self, padding=''):
        """
        Print feedback to screen
        """
        print(padding+'using SoftAdapt loss function')
        if self.smoothing:
           print(padding+' with smoothing')
           print(padding+' with smoothing_tau = ', 1./self.smoothing_alpha)
        print(padding+' with tau = ', self.tau)


class SharpStep(variable_weight_loss):
    """
    Implement sharp stepping between two values
    """

    def __init__(self, step_epoch=50, value_1=1.0, value_2=0.1, beta=0., **kwargs):
        """
        Initialize loss function
        """
        # initialize:
        super(SharpStep, self).__init__()
        # set parameters:
        self.step_epoch = step_epoch
        self.value_1 = value_1
        self.value_2 = value_2
        # initialize:
        self.beta = beta
        #
        return None

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...
        """
        if epoch < self.step_epoch:
            lambda_1 = self.value_1
        else:
            lambda_1 = self.value_2
        # set second by enforcing sum to one:
        lambda_2 = 1. - lambda_1
        #
        tf.keras.backend.set_value(self.lambda_1, lambda_1)
        tf.keras.backend.set_value(self.lambda_2, lambda_2)
        #
        return None

    def print_feedback(self, padding=''):
        """
        Print feedback to screen
        """
        print(padding+'using sharp step loss function')
