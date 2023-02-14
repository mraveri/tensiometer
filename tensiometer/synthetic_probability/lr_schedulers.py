"""
This file contains the learning rate schedulers.
"""

###############################################################################
# initial imports and set-up:

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np

###############################################################################
# Cosine learning rate scheduler:


class CosineAnnealer():
    """
    Code taken from https://www.kaggle.com/avanwyk/tf2-super-convergence-with-the-1cycle-policy
    """

    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos


class OneCycleScheduler(Callback):
    """
    `Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.
    Code taken from https://www.kaggle.com/avanwyk/tf2-super-convergence-with-the-1cycle-policy
    """

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)],
                       [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1

        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass  # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]

    def mom_schedule(self):
        return self.phases[self.phase][1]

###############################################################################
# Exponential decay:


class ExponentialDecayAnnealer():
    """
    Utility function describing the exponential decay
    """

    def __init__(self, start, end, roll_off_step, steps):
        self.start = start
        self.end = end
        self.roll_off_step = float(roll_off_step)
        self.steps = float(steps)
        self.n = 0

    def step(self):
        self.n += 1
        return self.start/(1. + (self.end/(self.start-self.end))**((self.n - self.roll_off_step)/(self.roll_off_step - self.steps)))


class ExponentialDecayScheduler(Callback):
    """
    Exponential decay learning rate
    """

    def __init__(self, lr_max, lr_min, roll_off_step, steps):
        super(ExponentialDecayScheduler, self).__init__()

        self.step = 0
        self.Annealer = ExponentialDecayAnnealer(lr_max, lr_min, roll_off_step, steps)
        self.lrs = []

    def on_train_begin(self, logs=None):
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore

###############################################################################
# Power law decay:


class PowerLawDecayAnnealer():
    """
    Utility function describing the power law decay
    """

    def __init__(self, start, end, power, steps):
        self.start = start
        self.end = end
        self.p = power
        self.steps = float(steps)
        self.n = 0

    def step(self):
        self.n += 1
        return self.start/(self.n/(self.steps*(self.end/self.start)**(1/self.p)))**self.p


class PowerLawDecayScheduler(Callback):
    """
    Power law decay in learning rate
    """

    def __init__(self, lr_max, lr_min, power, steps):
        super(PowerLawDecayScheduler, self).__init__()

        self.step = 0
        self.Annealer = PowerLawDecayAnnealer(lr_max, lr_min, power, steps)
        self.lrs = []

    def on_train_begin(self, logs=None):
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore

###############################################################################
# Adaptive learning rate scheduler:


class AdaptiveScheduler(Callback):
    """
    """

    def __init__(self, lr_max, lr_min, smoothing=True, smoothing_tau=5):
        super(AdaptiveScheduler, self).__init__()

        self.lr_max = lr_max
        self.lr_min = lr_min
        self.smoothing = smoothing
        self.smoothing_alpha = 1. / smoothing_tau

    def on_train_begin(self, logs=None):
        self.step = 0

        _initial_lr = 10**(0.5*(np.log10(self.lr_max) + np.log10(self.lr_min)))
        

        self.set_lr(_initial_lr)
        #self.set_momentum(self.mom_schedule().start)

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):

        self.step += 1
        _temp_lr = logs['lr'] * 10.**(0.001+logs['loss_rate'])
        #if self.smoothing:
        #    _temp_lr = self.smoothing_alpha * _temp_lr + (1. - self.smoothing_alpha) * logs['lr']
        self.set_lr(_temp_lr)

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore

    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass  # ignore


###############################################################################
# Step decay:

class StepDecayAnnealer():
    """
    Utility function describing the step decay
    """

    def __init__(self, start=None, change_every=None, steps=None, steps_per_epoch=None, boundaries=None, values=None):
        self.start = start
        self.steps_per_epoch = steps_per_epoch
        if boundaries is not None:
            self.boundaries = boundaries
            self.values = values            
        else:            
            self.ch = change_every
            total_changes = int(steps/self.ch)
            self.end = self.start/(10**total_changes)                
            self.steps = float(steps)

            self.boundaries = [self.ch*i for i in range(1, total_changes)]
            self.values = [self.start/(10**i) for i in range(0, total_changes)]
            
        self.n = 0

    def step(self):
        self.n += 1
        for i in range(len(self.boundaries)):
            if self.n < self.boundaries[i]*self.steps_per_epoch:
                return self.values[i]
            else:
                pass
        return self.values[-1]


class StepDecayScheduler(Callback):
    """
    Step decay in learning rate
    """

    def __init__(self, lr_max=None, change_every=None, steps=None, steps_per_epoch=None, boundaries=None, values=None):
        super(StepDecayScheduler, self).__init__()

        self.step = 0
        self.Annealer = StepDecayAnnealer(lr_max, change_every, steps, steps_per_epoch, boundaries, values)
        self.lrs = []

    def on_train_begin(self, logs=None):
        self.step = 0
        self.set_lr(self.Annealer.start)

    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        self.set_lr(self.Annealer.step())

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None

    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        except AttributeError:
            pass  # ignore