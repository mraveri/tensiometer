"""
This file contains learning rate schedulers for the flow training.

The schedulers are :class:`~tensiometer.synthetic_probability.training.Callback` objects and
change the learning rate of the trainer optimizer (a ``torch.optim`` optimizer).
"""

###############################################################################
# initial imports and set-up:

import logging

import numpy as np

from .training import Callback

###############################################################################
# optimizer helpers:

def _get_optimizer_lr(optimizer):
    """
    Return the optimizer learning rate value.

    :param optimizer: ``torch.optim`` optimizer.
    :returns: learning rate of the first parameter group.
    :raises AttributeError: if the optimizer has no parameter groups (for example None).
    """
    return float(optimizer.param_groups[0]['lr'])


def _set_optimizer_lr(optimizer, lr_value):
    """
    Update the optimizer learning rate.

    :param optimizer: ``torch.optim`` optimizer.
    :param lr_value: new learning rate, applied to all parameter groups.
    :raises AttributeError: if the optimizer has no parameter groups (for example None).
    """
    for group in optimizer.param_groups:
        group['lr'] = float(lr_value)

###############################################################################
# Exponential decay:


class ExponentialDecayAnnealer():
    """Exponential learning-rate annealer with smooth roll-off."""

    def __init__(self, start, end, roll_off_step, steps):
        """
        Initialize the annealer.

        :param start: starting learning rate.
        :param end: final learning rate.
        :param roll_off_step: step where decay begins.
        :param steps: total number of steps to reach ``end``.
        """
        self.start = start
        self.end = end
        self.roll_off_step = float(roll_off_step)
        self.steps = float(steps)
        self.n = 0

    def step(self):
        """
        Advance one step and compute the decayed learning rate.

        :returns: updated learning rate.
        """
        self.n += 1
        return self.start / (
            1. + (self.end / (self.start - self.end))**((self.n - self.roll_off_step) /
                                                        (self.roll_off_step - self.steps)))


class ExponentialDecayScheduler(Callback):
    """Callback applying exponential decay to the optimizer learning rate."""

    def __init__(self, lr_max, lr_min, roll_off_step, steps):
        """
        Exponentially decaying learning rate.
        
        :param lr_max: maximum learning rate
        :param lr_min: minimum earning rate
        :param roll_off_step: step at which the scheduler starts rolling off
        :param steps: total number of steps
        """
        
        super(ExponentialDecayScheduler, self).__init__()

        self.step = 0
        self.Annealer = ExponentialDecayAnnealer(lr_max, lr_min, roll_off_step, steps)
        self.lrs = []

    def on_train_begin(self, logs=None):
        """
        Reset state and set the initial learning rate.

        :param logs: training logs (the trainer passes an empty dict); unused.
        """
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        """
        Record the current learning rate at batch start.

        :param batch: index of the optimization step within the current epoch; unused.
        :param logs: batch logs (the trainer passes an empty dict); unused.
        """
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        """
        Update the learning rate at batch end.

        :param batch: index of the optimization step within the current epoch; unused.
        :param logs: batch logs (``{'loss': batch_loss}`` from the trainer); unused.
        """
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        """Retrieve the current optimizer learning rate."""
        try:
            return _get_optimizer_lr(self.trainer.optimizer)
        except AttributeError:
            return None

    def set_lr(self, lr):
        """
        Set the optimizer learning rate if available.

        :param lr: learning rate value to assign.
        """
        try:
            _set_optimizer_lr(self.trainer.optimizer, lr)
        except AttributeError:
            pass  # ignore


###############################################################################
# Power law decay:


class PowerLawDecayAnnealer():
    """Power-law learning-rate annealer."""

    def __init__(self, start, end, power, steps):
        """
        Initialize the annealer.

        :param start: starting learning rate.
        :param end: final learning rate.
        :param power: power-law exponent.
        :param steps: total number of steps to reach ``end``.
        """
        self.start = start
        self.end = end
        self.p = power
        self.steps = float(steps)
        self.n = 0

    def step(self):
        """
        Advance one step and compute the decayed learning rate.

        :returns: updated learning rate.
        """
        self.n += 1
        return self.start / (self.n / (self.steps * (self.end / self.start)**(1 / self.p)))**self.p


class PowerLawDecayScheduler(Callback):
    """Callback applying power-law decay to the learning rate."""

    def __init__(self, lr_max, lr_min, power, steps):
        """
        Power law decaying learning rate.
            
        :param lr_max: maximum learning rate
        :param lr_min: minimum earning rate
        :param power: power law index
        :param steps: total number of steps
        """

        super(PowerLawDecayScheduler, self).__init__()

        self.step = 0
        self.Annealer = PowerLawDecayAnnealer(lr_max, lr_min, power, steps)
        self.lrs = []

    def on_train_begin(self, logs=None):
        """
        Reset state and set the initial learning rate.

        :param logs: training logs (the trainer passes an empty dict); unused.
        """
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        """
        Record the current learning rate at batch start.

        :param batch: index of the optimization step within the current epoch; unused.
        :param logs: batch logs (the trainer passes an empty dict); unused.
        """
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        """
        Update the learning rate at batch end.

        :param batch: index of the optimization step within the current epoch; unused.
        :param logs: batch logs (``{'loss': batch_loss}`` from the trainer); unused.
        """
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        """Retrieve the current optimizer learning rate."""
        try:
            return _get_optimizer_lr(self.trainer.optimizer)
        except AttributeError:
            return None

    def set_lr(self, lr):
        """
        Set the optimizer learning rate if available.

        :param lr: learning rate value to assign.
        """
        try:
            _set_optimizer_lr(self.trainer.optimizer, lr)
        except AttributeError:
            pass  # ignore

###############################################################################
# Step decay:


class StepDecayAnnealer():
    """Piecewise constant learning-rate annealer."""

    def __init__(self, start=None, change_every=None, steps=None, steps_per_epoch=None, boundaries=None, values=None):
        """
        Initialize the step decay schedule.

        :param start: initial learning rate.
        :param change_every: number of steps between decay events.
        :param steps: total number of steps.
        :param steps_per_epoch: steps contained in one epoch.
        :param boundaries: optional explicit boundary steps.
        :param values: optional explicit values at boundaries.
        """
        self.steps_per_epoch = steps_per_epoch if steps_per_epoch is not None else 1
        self.start = start
        if boundaries is not None:
            self.boundaries = boundaries
            self.values = values
            if self.start is None and self.values is not None and len(self.values) > 0:
                self.start = self.values[0]
        else:
            self.ch = change_every
            total_changes = int(steps / self.ch)
            self.end = self.start / (10**total_changes)
            self.steps = float(steps)

            self.boundaries = [self.ch * i for i in range(1, total_changes)]
            self.values = [self.start / (10**i) for i in range(0, total_changes)]

        self.n = 0

    def step(self):
        """
        Advance one step and return the appropriate learning rate.

        :returns: learning rate selected by the current step.
        """
        self.n += 1
        for i in range(len(self.boundaries)):
            if self.n < self.boundaries[i] * self.steps_per_epoch:
                return self.values[i]
            else:
                pass
        return self.values[-1]


class StepDecayScheduler(Callback):
    """Callback applying step-wise changes to the learning rate."""

    def __init__(self, lr_max=None, change_every=None, steps=None, steps_per_epoch=None, boundaries=None, values=None):
        """
        Learning rate step function changes.
        
        :param lr_max:
        :param change_every:
        :param steps:
        :param steps_per_epoch:
        :param boundaries:
        :param values:
        """

        super(StepDecayScheduler, self).__init__()

        self.step = 0
        self.Annealer = StepDecayAnnealer(lr_max, change_every, steps, steps_per_epoch, boundaries, values)
        self.lrs = []

    def on_train_begin(self, logs=None):
        """
        Reset state and set the initial learning rate.

        :param logs: training logs (the trainer passes an empty dict); unused.
        """
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        """
        Record the current learning rate at batch start.

        :param batch: index of the optimization step within the current epoch; unused.
        :param logs: batch logs (the trainer passes an empty dict); unused.
        """
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        """
        Update the learning rate at batch end.

        :param batch: index of the optimization step within the current epoch; unused.
        :param logs: batch logs (``{'loss': batch_loss}`` from the trainer); unused.
        """
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        """ """
        try:
            return _get_optimizer_lr(self.trainer.optimizer)
        except AttributeError:
            return None

    def set_lr(self, lr):
        """
        Set the optimizer learning rate if available.

        :param lr: learning rate value to assign.
        """
        try:
            _set_optimizer_lr(self.trainer.optimizer, lr)
        except AttributeError:
            pass  # ignore


###############################################################################
# Adaptive learning rate (loss slope) and early stopping:


class LRAdaptLossSlopeEarlyStop(Callback):
    """Adaptive learning-rate scheduler with optional early stopping based on loss slope."""

    def __init__(
            self,
            monitor="val_loss",
            factor=1. / np.sqrt(10.),
            patience=25,
            cooldown=10,
            verbose=0,
            min_lr=1e-5,
            threshold=0.0,
            **kwargs,
    ):
        """
        Adaptive reduction of learning rate when likelihood improvement stalls for a given number of epochs.

        :param monitor: metric name to monitor.
        :param factor: multiplicative decay factor (<1).
        :param patience: epochs to wait before reducing the rate.
        :param cooldown: epochs to wait after a reduction.
        :param verbose: verbosity level.
        :param min_lr: lower bound for the learning rate.
        :param threshold: slope threshold (loss change per epoch). A line is fitted to the
            monitored metric over the last ``patience`` epochs and the learning rate is
            reduced when the fitted slope exceeds this value (default 0, i.e. the loss is rising).
        :param kwargs: additional keyword arguments, accepted for API compatibility and ignored.
        :raises ValueError: if ``factor >= 1``.
        """

        super().__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError("LRAdaptLossSlopeEarlyStop does not support a factor >= 1.0. Got {}".format(factor))

        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.threshold = threshold

        self._reset()

    def _reset(self):
        """Reset cooldown, wait counters, and loss history."""
        self.cooldown_counter = 0
        self.wait = 0
        self.last_losses = []

    def on_train_begin(self, logs=None):
        """
        Reset scheduler state at the start of training.

        :param logs: training logs (the trainer passes an empty dict); unused.
        """
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        """
        Update learning rate based on monitored loss slope at epoch end.

        :param epoch: current epoch index.
        :param logs: training logs containing monitored metrics.
        """
        logs = logs or {}
        logs["lr"] = _get_optimizer_lr(self.trainer.optimizer)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Learning rate reduction is conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.wait = 0
            else:
                self.last_losses.append(current)
                self.wait += 1
                if self.wait >= self.patience:
                    a = np.polyfit(np.arange(self.patience), self.last_losses[-self.patience:], 1)[0] # fits a line to `val_loss` in the last `patience` epochs
                    if a > self.threshold: # tests if val_loss is going up
                        old_lr = _get_optimizer_lr(self.trainer.optimizer)
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            _set_optimizer_lr(self.trainer.optimizer, new_lr)
                            if self.verbose > 0:
                                print(
                                    f"\nEpoch {epoch +1}: "
                                    "LRAdaptLossSlopeEarlyStop reducing "
                                    f"learning rate to {new_lr}.")
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                            self.last_losses = []
                        else:
                            self.trainer.stop_training = True

###############################################################################
# Adaptive learning rate (loss slope) with globalization and early stopping:


class LRSeesawAdaptLossSlopeEarlyStop(Callback):
    """Adaptive scheduler that decays on plateau but allows gentle increases when improving."""

    def __init__(
            self,
            monitor="val_loss",
            reduction_factor=1./np.sqrt(10.),
            increase_factor=0.003,
            patience=25,
            cooldown=10,
            verbose=0,
            min_lr=1e-5,
            threshold=0.0,
            **kwargs,
    ):
        """
        Adaptive reduction of learning rate when likelihood improvement stalls for a given number of epochs.

        :param monitor: metric name to monitor.
        :param reduction_factor: multiplicative decay factor (<1).
        :param increase_factor: small increment applied when loss improves.
        :param patience: epochs to wait before reducing the rate.
        :param cooldown: epochs to wait after a reduction.
        :param verbose: verbosity level.
        :param min_lr: lower bound for the learning rate.
        :param threshold: slope threshold (loss change per epoch). A line is fitted to the
            monitored metric over the last ``patience`` epochs and the learning rate is
            reduced when the fitted slope exceeds this value (default 0, i.e. the loss is rising).
        :param kwargs: additional keyword arguments, accepted for API compatibility and ignored.
        :raises ValueError: if ``reduction_factor >= 1``.
        """

        super().__init__()

        self.monitor = monitor
        if reduction_factor >= 1.0:
            raise ValueError("LRSeesawAdaptLossSlopeEarlyStop does not support a factor >= 1.0. Got {}".format(reduction_factor))

        self.reduction_factor = reduction_factor
        self.increase_factor = increase_factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.threshold = threshold

        self._reset()

    def _reset(self):
        """Reset cooldown, wait counters, and loss history."""
        self.cooldown_counter = 0
        self.wait = 0
        self.last_losses = []

    def on_train_begin(self, logs=None):
        """
        Reset scheduler state at the start of training.

        :param logs: training logs (the trainer passes an empty dict); unused.
        """
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        """
        Update the learning rate at epoch end.

        If the monitored metric is available, the learning rate is first multiplied by
        ``1 + increase_factor``. Then, outside cooldown, once ``patience`` epochs have been
        collected, a line is fitted to the last ``patience`` values of the metric; if its slope
        exceeds ``threshold`` the learning rate is multiplied by ``reduction_factor`` (clipped at
        ``min_lr``), or training is stopped if it is already at ``min_lr``.

        :param epoch: current epoch index (only used in the verbose message).
        :param logs: epoch logs containing the monitored metric; ``logs['lr']`` is set in place
            to the learning rate at the start of the call.
        """
        logs = logs or {}
        logs["lr"] = _get_optimizer_lr(self.trainer.optimizer)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning(
                "Learning rate reduction is conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        else:
            # increase learning rate:
            old_lr = _get_optimizer_lr(self.trainer.optimizer)
            new_lr = old_lr * (1.0 + self.increase_factor)
            _set_optimizer_lr(self.trainer.optimizer, new_lr)
            # decrease learning rate:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.wait = 0
            else:
                self.last_losses.append(current)
                self.wait += 1
                if self.wait >= self.patience:
                    a = np.polyfit(np.arange(self.patience), self.last_losses[-self.patience:], 1)[0] # fits a line to `val_loss` in the last `patience` epochs
                    if a > self.threshold: # tests if val_loss is going up
                        old_lr = _get_optimizer_lr(self.trainer.optimizer)
                        if old_lr > self.min_lr:
                            new_lr = old_lr * self.reduction_factor
                            new_lr = max(new_lr, self.min_lr)
                            _set_optimizer_lr(self.trainer.optimizer, new_lr)
                            if self.verbose > 0:
                                print(
                                    f"\nEpoch {epoch +1}: "
                                    "LRSeesawAdaptLossSlopeEarlyStop reducing "
                                    f"learning rate to {new_lr}.")
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                            self.last_losses = []
                        else:
                            self.trainer.stop_training = True
                            
