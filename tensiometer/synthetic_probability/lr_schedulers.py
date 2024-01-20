"""
This file contains learning rate schedulers for tensorflow optimization.
"""

###############################################################################
# initial imports and set-up:

import numpy as np

# tensorflow imports:
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.platform import tf_logging as logging

###############################################################################
# Exponential decay:


class ExponentialDecayAnnealer():

    def __init__(self, start, end, roll_off_step, steps):
        self.start = start
        self.end = end
        self.roll_off_step = float(roll_off_step)
        self.steps = float(steps)
        self.n = 0

    def step(self):
        self.n += 1
        return self.start / (
            1. + (self.end / (self.start - self.end))**((self.n - self.roll_off_step) /
                                                        (self.roll_off_step - self.steps)))


class ExponentialDecayScheduler(Callback):

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
        """ """
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        """ """
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        """ """
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        """ """
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def set_lr(self, lr):
        """ """
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore


###############################################################################
# Power law decay:


class PowerLawDecayAnnealer():

    def __init__(self, start, end, power, steps):
        self.start = start
        self.end = end
        self.p = power
        self.steps = float(steps)
        self.n = 0

    def step(self):
        self.n += 1
        return self.start / (self.n / (self.steps * (self.end / self.start)**(1 / self.p)))**self.p


class PowerLawDecayScheduler(Callback):

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
        """ """
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        """ """
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        """ """
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        """ """
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def set_lr(self, lr):
        """ """
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore

###############################################################################
# Step decay:


class StepDecayAnnealer():

    def __init__(self, start=None, change_every=None, steps=None, steps_per_epoch=None, boundaries=None, values=None):
        self.start = start
        self.steps_per_epoch = steps_per_epoch
        if boundaries is not None:
            self.boundaries = boundaries
            self.values = values
        else:
            self.ch = change_every
            total_changes = int(steps / self.ch)
            self.end = self.start / (10**total_changes)
            self.steps = float(steps)

            self.boundaries = [self.ch * i for i in range(1, total_changes)]
            self.values = [self.start / (10**i) for i in range(0, total_changes)]

        self.n = 0

    def step(self):
        self.n += 1
        for i in range(len(self.boundaries)):
            if self.n < self.boundaries[i] * self.steps_per_epoch:
                return self.values[i]
            else:
                pass
        return self.values[-1]


class StepDecayScheduler(Callback):

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
        """ """
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        """ """
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        """ """
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        """ """
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def set_lr(self, lr):
        """ """
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore


###############################################################################
# Adaptive learning rate (loss slope) and early stopping:


class LRAdaptLossSlopeEarlyStop(Callback):

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

        :param monitor:
        :param factor:
        :param patience:
        :param cooldown:
        :param verbose:
        :param min_lr:
        """

        super().__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError("LRAdaptLossSlopeEarlyStop does not support a factor >= 1.0. Got {factor}")

        self.factor = factor
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.threshold = threshold

        self._reset()

    def _reset(self):
        self.cooldown_counter = 0
        self.wait = 0
        self.last_losses = []

    def on_train_begin(self, logs=None):
        """ """
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        """ """
        logs = logs or {}
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)
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
                        old_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                        if old_lr > np.float32(self.min_lr):
                            new_lr = old_lr * self.factor
                            new_lr = max(new_lr, self.min_lr)
                            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                            if self.verbose > 0:
                                tf.print(
                                    f"\nEpoch {epoch +1}: "
                                    "LRAdaptLossSlopeEarlyStop reducing "
                                    f"learning rate to {new_lr}.")
                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                            self.last_losses = []
                        else:
                            self.model.stop_training = True
                            