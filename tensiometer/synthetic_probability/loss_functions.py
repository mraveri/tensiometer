"""
This file contains the loss functions for the normalizing flow training.

Since we are combining different loss functions we have different options.

All losses are called as ``loss(y_true, y_pred, sample_weight=None)``, where ``y_pred`` is
the flow log probability of the batch and ``y_true`` the (optional) true log posterior.
With the default ``'weighted_mean'`` reduction the result is ``sum(l * w) / sum(w)``.
Keras divided by the batch size instead; since the flows normalize the weights to sum to
the number of samples the two agree in expectation.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import torch

###############################################################################
# helpers:

_REDUCTIONS = ['none', 'sum', 'weighted_mean', 'sum_over_batch_size']


def _broadcast_sample_weight(sample_weight, losses):
    """
    Broadcast sample weights to match loss tensor shape.

    :param sample_weight: weights to apply.
    :param losses: loss tensor.
    :returns: broadcasted weights matching ``losses``.
    """
    if sample_weight is None:
        return None
    sample_weight = torch.as_tensor(sample_weight, dtype=losses.dtype, device=losses.device)
    if sample_weight.dim() < losses.dim():
        sample_weight = sample_weight.reshape(tuple(sample_weight.shape) + (1,) * (losses.dim() - sample_weight.dim()))
    return torch.broadcast_to(sample_weight, losses.shape)


def _reduce_weighted_loss(losses, sample_weight, reduction):
    """
    Reduce weighted losses.

    :param losses: per-sample losses.
    :param sample_weight: sample weights for each loss.
    :param reduction: ``'none'``, ``'sum'``, ``'weighted_mean'`` (sum of weighted losses over
        the sum of weights) or ``'sum_over_batch_size'`` (sum of weighted losses over the
        number of losses, the Keras default).
    :returns: reduced loss tensor.
    :raises ValueError: for an unknown reduction.
    """
    if reduction not in _REDUCTIONS:
        raise ValueError('Unknown reduction ' + repr(reduction) + '. Use one of ' + ', '.join(_REDUCTIONS))
    weights = _broadcast_sample_weight(sample_weight, losses)
    if weights is not None:
        losses = losses * weights
    if reduction == 'none':
        return losses
    total_loss = torch.sum(losses)
    if reduction == 'sum':
        return total_loss
    if weights is None or reduction == 'sum_over_batch_size':
        denom = torch.as_tensor(float(losses.numel()), dtype=losses.dtype, device=losses.device)
    else:
        denom = torch.sum(weights)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return total_loss / denom

###############################################################################
# base class:


class Loss(object):
    """
    Base class of the loss functions.

    :param reduction: reduction of the per-sample losses: ``'none'``, ``'sum'``,
        ``'weighted_mean'`` (default, sum of weighted losses over the sum of weights) or
        ``'sum_over_batch_size'`` (sum of weighted losses over the number of losses).
    """

    def __init__(self, reduction='weighted_mean'):
        if reduction not in _REDUCTIONS:
            raise ValueError('Unknown reduction ' + repr(reduction) + '. Use one of ' + ', '.join(_REDUCTIONS))
        self.reduction = reduction

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Per-sample losses.

        :param y_true: target values.
        :param y_pred: predicted values.
        :param sample_weight: weights for each sample.
        :returns: per-sample loss tensor.
        """
        raise NotImplementedError

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Evaluate the reduced loss.

        :param y_true: target values (may be None for losses that do not use them).
        :param y_pred: predicted values.
        :param sample_weight: optional weights for each sample.
        :returns: scalar loss tensor (per-sample tensor with ``reduction='none'``).
        """
        if sample_weight is None:
            sample_weight = torch.ones_like(y_pred)
        losses = self.compute_loss(y_true, y_pred, sample_weight)
        return _reduce_weighted_loss(losses, sample_weight, self.reduction)

    def reset(self):
        """Reset the loss state (no-op by default)."""


class mean_squared_error(Loss):
    """Mean squared error over the last axis, used to train derived parameter bijectors."""

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Mean squared difference over the last axis.

        :param y_true: target values ``(N, D)``.
        :param y_pred: predicted values ``(N, D)``.
        :param sample_weight: weights for each sample (applied by the reduction).
        :returns: per-sample losses ``(N,)``.
        """
        return torch.mean((y_true - y_pred)**2, dim=-1)

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Evaluate the reduced loss.

        :param y_true: target values ``(N, D)``.
        :param y_pred: predicted values ``(N, D)``.
        :param sample_weight: optional weights ``(N,)``.
        :returns: scalar loss tensor.
        """
        losses = self.compute_loss(y_true, y_pred, sample_weight)
        if sample_weight is None:
            sample_weight = torch.ones_like(losses)
        return _reduce_weighted_loss(losses, sample_weight, self.reduction)

###############################################################################
# standard normalizing flow loss function:


class standard_loss(Loss):
    """KL-based density loss for normalizing flow training."""

    def __init__(self, reduction='weighted_mean'):
        """
        Standard density loss function for the normalizing flow.

        :param reduction: reduction of the per-sample losses.
        """
        super(standard_loss, self).__init__(reduction=reduction)

    def compute_loss_components(self, y_true, y_pred, sample_weight):
        """
        Compute the signed log-probability contribution.

        :param y_true: target log density (unused).
        :param y_pred: predicted log density.
        :param sample_weight: sample weights.
        :returns: negative predicted log density.
        """
        return -y_pred

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Per-sample loss, the negative log density.

        :param y_true: target log density (unused).
        :param y_pred: predicted log density.
        :param sample_weight: sample weights (applied by the reduction).
        :returns: negative predicted log density.
        """
        return -y_pred

    def call(self, y_true, y_pred):
        """
        Standard normalizing flow loss function is KL divergence of two abstract
        distributions.

        :param y_true: target log density (unused).
        :param y_pred: predicted log density.
        :returns: negative predicted log density.
        """
        return -y_pred

    def print_feedback(self, padding=''):
        """
        Print the configured loss details.

        :param padding: string prepended to every printed line.
        """
        print(padding+'using standard loss function')

    def reset(self):
        """Reset loss hyperparameters (no-op for standard loss)."""
        pass


###############################################################################
# density and evidence loss with constant weights:


def _density_and_evidence_error(y_true, y_pred, sample_weight, beta):
    """
    Density loss and squared deviation of the log posterior residuals from their weighted mean.

    :param y_true: target log density.
    :param y_pred: predicted log density.
    :param sample_weight: weights for each sample.
    :param beta: additive offset applied to the predicted log density.
    :returns: tuple of per-sample density loss and squared residual deviation.
    """
    if sample_weight is None:
        sample_weight = torch.ones_like(y_pred)
    sample_weight = torch.as_tensor(sample_weight, dtype=y_pred.dtype, device=y_pred.device)
    y_true = torch.as_tensor(y_true, dtype=y_pred.dtype, device=y_pred.device)
    # compute difference between true and predicted posterior values:
    diffs = (y_true - y_pred)
    # sum weights:
    tot_weights = torch.sum(sample_weight)
    # compute overall offset:
    mean_diff = torch.sum(diffs * sample_weight) / tot_weights
    # compute its variance:
    var_diff = (diffs - mean_diff)**2
    # compute density loss function:
    loss_orig = -(y_pred + beta)
    #
    return loss_orig, var_diff


class constant_weight_loss(Loss):
    """Combined density and evidence-error loss with fixed weights."""

    def __init__(self, alpha=1.0, beta=0.0, reduction='weighted_mean'):
        """
        Initialize the fixed-weight loss.

        :param alpha: weight for the density component.
        :param beta: additive offset applied to predicted log density.
        :param reduction: reduction of the per-sample losses.
        """
        # initialize:
        super(constant_weight_loss, self).__init__(reduction=reduction)
        # set parameters:
        self.alpha = alpha
        self.beta = beta

    def compute_loss_components(self, y_true, y_pred, sample_weight):
        """
        Compute density and evidence-error components.

        :param y_true: target log density.
        :param y_pred: predicted log density.
        :param sample_weight: weights for each sample.
        :returns: tuple of density loss and variance of residuals.
        """
        return _density_and_evidence_error(y_true, y_pred, sample_weight, self.beta)

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Combine density and evidence-error loss.

        :param y_true: target log density.
        :param y_pred: predicted log density.
        :param sample_weight: weights for each sample.
        :returns: weighted sum of loss components.
        """
        # get components:
        loss_1, loss_2 = self.compute_loss_components(y_true, y_pred, sample_weight)
        #
        return +self.alpha*loss_1 + (1. - self.alpha)*loss_2

    def print_feedback(self, padding=''):
        """
        Print the configured fixed-weight loss settings.

        :param padding: string prepended to every printed line.
        """
        print(padding+'using combined density and evidence-error loss function')
        print(padding+'weight of density loss: %.3g, weight of evidence-error-loss: %.3g' % (self.alpha, 1.-self.alpha))

    def reset(self):
        """Reset loss hyperparameters (no-op for constant weights)."""
        pass

###############################################################################
# density and evidence loss with variable weights:


class variable_weight_loss(Loss):
    """Combined loss with weights updated during training."""

    def __init__(self, lambda_1=1.0, lambda_2=0.0, beta=0.0, reduction='weighted_mean'):
        """
        Initialize the variable-weight loss.

        :param lambda_1: initial weight for the density term.
        :param lambda_2: initial weight for the evidence-error term.
        :param beta: additive offset applied to predicted log density.
        :param reduction: reduction of the per-sample losses.
        """
        # initialize:
        super(variable_weight_loss, self).__init__(reduction=reduction)
        # set parameters:
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = float(lambda_2)
        self.beta = float(beta)
        # save initial parameters:
        self.initial_lambda_1 = float(lambda_1)
        self.initial_lambda_2 = float(lambda_2)
        self.initial_beta = float(beta)

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...

        :param epoch: current epoch index.
        :param kwargs: unused passthrough arguments for compatibility.
        :raises NotImplementedError: expected to be overridden in subclasses.
        """
        # base class is empty...
        # use the following sintax:
        # self.lambda_1 = 0.5 * epoch
        raise NotImplementedError

    def compute_loss_components(self, y_true, y_pred, sample_weight, lambda_1=None, lambda_2=None):
        """
        Compute density and evidence-error components with configurable weights.

        :param y_true: target log density.
        :param y_pred: predicted log density.
        :param sample_weight: weights for each sample.
        :param lambda_1: optional override for density weight.
        :param lambda_2: optional override for evidence-error weight.
        :returns: tuple of density loss, variance of residuals, and active weights.
        """
        loss_orig, var_diff = _density_and_evidence_error(y_true, y_pred, sample_weight, self.beta)
        # get weights if not passed:
        if lambda_1 is None:
            lambda_1 = self.lambda_1
        if lambda_2 is None:
            lambda_2 = self.lambda_2
        #
        return loss_orig, var_diff, lambda_1, lambda_2

    def compute_loss(self, y_true, y_pred, sample_weight):
        """
        Combine density and evidence-error loss using current weights.

        :param y_true: target log density.
        :param y_pred: predicted log density.
        :param sample_weight: weights for each sample.
        :returns: weighted loss value.
        """
        # get components:
        loss_1, loss_2, lambda_1, lambda_2 = self.compute_loss_components(
            y_true, y_pred, sample_weight, self.lambda_1, self.lambda_2)
        #
        return lambda_1*loss_1 + lambda_2*loss_2

    def print_feedback(self, padding=''):
        """
        Print feedback to screen

        :param padding: string prepended to every printed line.
        :raises NotImplementedError: expected to be overridden in subclasses.
        """
        raise NotImplementedError

    def _reset_state(self):
        """Reset the subclass specific state (no-op by default)."""

    def reset(self):
        """
        Reset the loss weights and state to their initial values, keeping the hyperparameters.
        """
        self.lambda_1 = self.initial_lambda_1
        self.lambda_2 = self.initial_lambda_2
        self.beta = self.initial_beta
        self._reset_state()


class random_weight_loss(variable_weight_loss):
    """
    Random weighting of the two loss functions.
    """

    def __init__(self, initial_random_epoch=0, lambda_1=1.0, beta=0.0, **kwargs):
        """
        Initialize loss function

        :param initial_random_epoch: epoch after which the weights are randomized.
        :param lambda_1: initial weight for the density term.
        :param beta: additive offset applied to predicted log density.
        :param kwargs: additional keyword arguments (the flow passes all its options), ignored
            and not forwarded to the parent class.
        """
        # initialize:
        super(random_weight_loss, self).__init__(lambda_1, 1.-lambda_1, beta)
        # set parameters:
        self.initial_random_epoch = initial_random_epoch

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...

        :param epoch: current epoch index; after ``initial_random_epoch`` the weights are set at
            random to either ``(1, 0)`` or ``(0, 1)``.
        :param kwargs: additional keyword arguments (e.g. ``logs``), ignored.
        :returns: None.
        """
        if epoch > self.initial_random_epoch:
            _temp_rand = np.random.randint(2)
            self.lambda_1 = float(_temp_rand)
            self.lambda_2 = 1. - float(_temp_rand)
        #
        return None

    def print_feedback(self, padding=''):
        """
        Print feedback to screen

        :param padding: string prepended to every printed line.
        """
        print(padding+'using randomized loss function')


class annealed_weight_loss(variable_weight_loss):
    """
    Slowly go from density to evidence-error loss.
    """

    def __init__(self, anneal_epoch=125, lambda_1=1.0, beta=0.0, roll_off_nepoch=10, **kwargs):
        """
        Initialize loss function

        :param anneal_epoch: epoch at which the annealing starts.
        :param lambda_1: initial weight for the density term.
        :param beta: additive offset applied to predicted log density.
        :param roll_off_nepoch: e-folding number of epochs of the annealing.
        :param kwargs: additional keyword arguments (the flow passes all its options), ignored
            and not forwarded to the parent class.
        """
        # initialize:
        super(annealed_weight_loss, self).__init__(lambda_1, 1.-lambda_1, beta)
        # set parameters:
        self.anneal_epoch = anneal_epoch
        self.roll_off_nepoch = roll_off_nepoch

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...

        :param epoch: current epoch index; after ``anneal_epoch`` the current density weight is
            multiplied by ``exp(-(epoch - anneal_epoch) / roll_off_nepoch)``.
        :param kwargs: additional keyword arguments (e.g. ``logs``), ignored.
        :returns: None.
        """
        if epoch > self.anneal_epoch:
            _lambda_1 = self.lambda_1
            _lambda_1 *= np.exp(-1.*(epoch - self.anneal_epoch)/self.roll_off_nepoch)
            self.lambda_1 = float(_lambda_1)
            self.lambda_2 = 1. - float(_lambda_1)
        #
        return None

    def print_feedback(self, padding=''):
        """
        Print feedback to screen

        :param padding: string prepended to every printed line.
        """
        print(padding+'using annealed loss function')


class SoftAdapt_weight_loss(variable_weight_loss):
    """
    Implement SoftAdapt as in arXiv:1912.12355, with optional smoothing
    """

    def __init__(self, tau=1.0, beta=0.0, smoothing=True, smoothing_tau=20, quantity_1='val_rho_loss', quantity_2='val_ee_loss', **kwargs):
        """
        Initialize loss function

        :param tau: temperature of the softmax of the loss rates.
        :param beta: additive offset applied to predicted log density.
        :param smoothing: smooth the rates and weights in time.
        :param smoothing_tau: smoothing time scale in epochs.
        :param quantity_1: log key of the first monitored loss.
        :param quantity_2: log key of the second monitored loss.
        :param kwargs: additional keyword arguments (the flow passes all its options), ignored
            and not forwarded to the parent class.
        """
        # initialize:
        super(SoftAdapt_weight_loss, self).__init__(beta=beta)
        # set parameters:
        self.tau = tau
        self.smoothing = smoothing
        self.smoothing_alpha = 1. / smoothing_tau
        # quantrities to be monitored:
        self.quantity_1 = quantity_1
        self.quantity_2 = quantity_2
        self._reset_state()

    def _reset_state(self):
        """Reset the smoothing buffers."""
        self.rate_1_buffer = 0.0
        self.rate_2_buffer = 0.0
        self.lambda_1_buffer = 1.0
        self.lambda_2_buffer = 0.0

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...

        :param epoch: current epoch index.
        :param kwargs: must contain ``logs``, the flow training log.
        :returns: None.
        :raises ValueError: if ``logs`` is missing or does not contain the monitored quantities.
        """
        # get logs:
        logs = kwargs.get('logs')
        if logs is None:
            raise ValueError('SoftAdapt_weight_loss needs the training logs, pass them as logs=...')
        for _quantity in [self.quantity_1, self.quantity_2]:
            if _quantity not in logs:
                raise ValueError('SoftAdapt_weight_loss monitors ' + _quantity + ' which is not in the training logs.')
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
                _lambda_tot = _lambda_1 + _lambda_2
                _lambda_1 = _lambda_1 / _lambda_tot
                _lambda_2 = _lambda_2 / _lambda_tot
                self.lambda_1_buffer = _lambda_1
                self.lambda_2_buffer = _lambda_2
            else:
                _lambda_1 = lambda_1 / _tot
        # set second by enforcing sum to one:
        self.lambda_1 = float(_lambda_1)
        self.lambda_2 = 1. - float(_lambda_1)
        #
        return None

    def print_feedback(self, padding=''):
        """
        Print feedback to screen

        :param padding: string prepended to every printed line.
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

        :param step_epoch: epoch of the step.
        :param value_1: density weight before the step.
        :param value_2: density weight after the step.
        :param beta: additive offset applied to predicted log density.
        :param kwargs: additional keyword arguments (the flow passes all its options), ignored
            and not forwarded to the parent class.
        """
        # initialize:
        super(SharpStep, self).__init__(beta=beta)
        # set parameters:
        self.step_epoch = step_epoch
        self.value_1 = value_1
        self.value_2 = value_2

    def update_lambda_values_on_epoch_begin(self, epoch, **kwargs):
        """
        Update values of lambda at epoch start. Takes in every kwargs to not
        crowd the interface...

        :param epoch: current epoch index; the density weight is ``value_1`` before
            ``step_epoch`` and ``value_2`` from it on.
        :param kwargs: additional keyword arguments (e.g. ``logs``), ignored.
        :returns: None.
        """
        if epoch < self.step_epoch:
            lambda_1 = self.value_1
        else:
            lambda_1 = self.value_2
        # set second by enforcing sum to one:
        self.lambda_1 = float(lambda_1)
        self.lambda_2 = 1. - float(lambda_1)
        #
        return None

    def print_feedback(self, padding=''):
        """
        Print feedback to screen

        :param padding: string prepended to every printed line.
        """
        print(padding+'using sharp step loss function')
