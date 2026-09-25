"""Tests for the learning rate schedulers of the flow training (PyTorch implementation)."""

#########################################################################################################
# Imports

import unittest
from unittest.mock import patch

import numpy as np

from tensiometer.synthetic_probability import lr_schedulers as lrs
from tensiometer.synthetic_probability import training

#########################################################################################################
# Helper stubs


class DummyOptimizer:
    """Minimal stand-in for a ``torch.optim`` optimizer."""

    def __init__(self, lr=0.01, num_groups=1):
        """Create ``num_groups`` parameter groups with learning rate ``lr``."""
        self.param_groups = [{'lr': lr} for _ in range(num_groups)]


class DummyTrainer:
    """Minimal stand-in for :class:`~tensiometer.synthetic_probability.training.Trainer`."""

    def __init__(self, lr=0.01):
        """Hold a dummy optimizer and the stop flag."""
        self.optimizer = DummyOptimizer(lr)
        self.stop_training = False


class NoOptimizerTrainer:
    """Trainer stub without an optimizer attribute."""


def _lr(trainer):
    """Learning rate of the first parameter group of a trainer."""
    return trainer.optimizer.param_groups[0]['lr']


def _run_epochs(callback, values, monitor='val_loss', start_epoch=0):
    """Call ``on_epoch_end`` with one monitored value per epoch and return the logs."""
    all_logs = []
    for index, value in enumerate(values):
        logs = {monitor: value}
        callback.on_epoch_end(start_epoch + index, logs=logs)
        all_logs.append(logs)
    return all_logs

#########################################################################################################
# Optimizer helpers


class TestOptimizerHelpers(unittest.TestCase):
    """Tests of the optimizer learning rate helpers."""

    def test_get_and_set_lr(self):
        """The helpers read and write ``param_groups[*]['lr']``."""
        optimizer = DummyOptimizer(0.1, num_groups=3)
        self.assertEqual(lrs._get_optimizer_lr(optimizer), 0.1)
        lrs._set_optimizer_lr(optimizer, 0.05)
        self.assertEqual([group['lr'] for group in optimizer.param_groups], [0.05, 0.05, 0.05])
        self.assertIsInstance(lrs._get_optimizer_lr(optimizer), float)

    def test_set_lr_converts_to_float(self):
        """Learning rates are stored as python floats."""
        optimizer = DummyOptimizer()
        lrs._set_optimizer_lr(optimizer, np.float32(0.25))
        self.assertIs(type(optimizer.param_groups[0]['lr']), float)

    def test_missing_optimizer_raises_attribute_error(self):
        """A missing optimizer raises AttributeError (tolerated by the schedulers)."""
        with self.assertRaises(AttributeError):
            lrs._get_optimizer_lr(None)
        with self.assertRaises(AttributeError):
            lrs._set_optimizer_lr(None, 0.1)

    def test_schedulers_are_callbacks(self):
        """All schedulers are training callbacks attached with ``set_trainer``."""
        callbacks = [
            lrs.ExponentialDecayScheduler(0.1, 0.01, 1, 10),
            lrs.PowerLawDecayScheduler(0.1, 0.01, 2, 10),
            lrs.StepDecayScheduler(boundaries=[1], values=[0.1]),
            lrs.LRAdaptLossSlopeEarlyStop(),
            lrs.LRSeesawAdaptLossSlopeEarlyStop(),
        ]
        trainer = DummyTrainer()
        for callback in callbacks:
            self.assertIsInstance(callback, training.Callback)
            self.assertIsNone(callback.trainer)
            callback.set_trainer(trainer)
            self.assertIs(callback.trainer, trainer)

#########################################################################################################
# Annealers


class TestAnnealers(unittest.TestCase):
    """Tests of the learning rate annealers."""

    def test_exponential_annealer(self):
        """Exponential decay reaches half the start at ``roll_off_step`` and ``end`` at ``steps``."""
        annealer = lrs.ExponentialDecayAnnealer(start=0.1, end=0.01, roll_off_step=2, steps=5)
        values = [annealer.step() for _ in range(5)]
        self.assertAlmostEqual(values[1], 0.05)
        self.assertAlmostEqual(values[-1], 0.01)
        self.assertTrue(np.all(np.diff(values) < 0.0))
        self.assertEqual(annealer.n, 5)

    def test_power_law_annealer(self):
        """Power law decay reaches ``end`` at ``steps``."""
        annealer = lrs.PowerLawDecayAnnealer(start=0.1, end=0.01, power=2, steps=5)
        values = [annealer.step() for _ in range(5)]
        self.assertTrue(np.all(np.array(values) > 0.0))
        self.assertTrue(np.all(np.diff(values) < 0.0))
        self.assertAlmostEqual(values[-1], 0.01)

    def test_step_annealer_change_every(self):
        """Step decay divides the learning rate by ten at each boundary."""
        annealer = lrs.StepDecayAnnealer(start=0.1, change_every=1, steps=4, steps_per_epoch=1)
        self.assertEqual(annealer.boundaries, [1, 2, 3])
        np.testing.assert_allclose(annealer.values, [0.1, 0.01, 0.001, 0.0001])
        values = [annealer.step() for _ in range(4)]
        np.testing.assert_allclose(values, [0.01, 0.001, 0.0001, 0.0001])

    def test_step_annealer_boundaries(self):
        """Explicit boundaries are counted in epochs of ``steps_per_epoch`` steps."""
        annealer = lrs.StepDecayAnnealer(steps_per_epoch=2, boundaries=[1, 2], values=[0.3, 0.1])
        self.assertEqual(annealer.start, 0.3)
        values = [annealer.step() for _ in range(5)]
        self.assertEqual(values, [0.3, 0.1, 0.1, 0.1, 0.1])
        annealer = lrs.StepDecayAnnealer(start=0.5, boundaries=[1], values=[0.4])
        self.assertEqual(annealer.start, 0.5)
        self.assertEqual(annealer.steps_per_epoch, 1)

#########################################################################################################
# Per-batch schedulers


class TestBatchSchedulers(unittest.TestCase):
    """Tests of the schedulers acting on every batch."""

    def _check_batch_scheduler(self, callback, annealer_copy, num_steps=4):
        """Drive the per-batch hooks and compare with the annealer sequence."""
        trainer = DummyTrainer(lr=123.0)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        self.assertEqual(callback.step, 0)
        self.assertEqual(_lr(trainer), annealer_copy.start)
        expected = [annealer_copy.start]
        for batch in range(num_steps):
            callback.on_train_batch_begin(batch)
            callback.on_train_batch_end(batch)
            expected.append(annealer_copy.step())
            self.assertAlmostEqual(_lr(trainer), expected[-1])
        self.assertEqual(callback.step, num_steps)
        np.testing.assert_allclose(callback.lrs, expected[:-1])
        self.assertEqual(callback.get_lr(), _lr(trainer))

    def _check_missing_optimizer(self, callback):
        """Hooks do nothing without an optimizer."""
        self.assertIsNone(callback.get_lr())
        callback.set_trainer(NoOptimizerTrainer())
        callback.on_train_begin()
        callback.on_train_batch_begin(0)
        callback.on_train_batch_end(0)
        self.assertIsNone(callback.get_lr())
        self.assertEqual(callback.lrs, [None])
        callback.set_lr(0.05)

    def test_exponential_scheduler(self):
        """Exponential decay scheduler follows its annealer."""
        callback = lrs.ExponentialDecayScheduler(lr_max=0.1, lr_min=0.01, roll_off_step=2, steps=10)
        self._check_batch_scheduler(callback, lrs.ExponentialDecayAnnealer(0.1, 0.01, 2, 10))
        self._check_missing_optimizer(lrs.ExponentialDecayScheduler(0.1, 0.01, 2, 10))

    def test_power_law_scheduler(self):
        """Power law decay scheduler follows its annealer."""
        callback = lrs.PowerLawDecayScheduler(lr_max=0.1, lr_min=0.01, power=2, steps=10)
        self._check_batch_scheduler(callback, lrs.PowerLawDecayAnnealer(0.1, 0.01, 2, 10))
        self._check_missing_optimizer(lrs.PowerLawDecayScheduler(0.1, 0.01, 2, 10))

    def test_step_scheduler(self):
        """Step decay scheduler follows its annealer, with both constructors."""
        callback = lrs.StepDecayScheduler(lr_max=0.1, change_every=2, steps=6, steps_per_epoch=1)
        self._check_batch_scheduler(callback, lrs.StepDecayAnnealer(0.1, 2, 6, 1))
        callback = lrs.StepDecayScheduler(boundaries=[1, 2], values=[0.1, 0.05])
        self._check_batch_scheduler(callback, lrs.StepDecayAnnealer(boundaries=[1, 2], values=[0.1, 0.05]))
        self._check_missing_optimizer(lrs.StepDecayScheduler(boundaries=[1], values=[0.1]))

    def test_train_begin_resets_step(self):
        """A new training restarts the step counter and the learning rate."""
        trainer = DummyTrainer()
        callback = lrs.PowerLawDecayScheduler(lr_max=0.1, lr_min=0.01, power=1, steps=10)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        callback.on_train_batch_end(0)
        self.assertLess(_lr(trainer), 0.1)
        callback.on_train_begin()
        self.assertEqual(callback.step, 0)
        self.assertEqual(_lr(trainer), 0.1)

#########################################################################################################
# Adaptive schedulers


class TestLRAdaptLossSlopeEarlyStop(unittest.TestCase):
    """Tests of :class:`LRAdaptLossSlopeEarlyStop`."""

    def test_factor_validation(self):
        """A factor >= 1 raises ValueError."""
        with self.assertRaises(ValueError):
            lrs.LRAdaptLossSlopeEarlyStop(factor=1.0)
        with self.assertRaises(ValueError):
            lrs.LRAdaptLossSlopeEarlyStop(factor=1.1)

    def test_defaults_and_extra_kwargs(self):
        """Defaults, and unknown keyword arguments are accepted."""
        callback = lrs.LRAdaptLossSlopeEarlyStop(some_other_option=3)
        self.assertEqual(callback.monitor, 'val_loss')
        self.assertAlmostEqual(callback.factor, 1. / np.sqrt(10.))
        self.assertEqual((callback.patience, callback.cooldown), (25, 10))
        self.assertEqual(callback.min_lr, 1e-5)

    def test_reduction_on_increasing_loss(self):
        """An increasing monitored loss reduces the learning rate after ``patience`` epochs."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.5, patience=2, cooldown=0, min_lr=1e-6)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        logs = _run_epochs(callback, [1.0])
        self.assertEqual(_lr(trainer), 0.01)
        self.assertEqual(logs[0]['lr'], 0.01)
        _run_epochs(callback, [2.0], start_epoch=1)
        self.assertAlmostEqual(_lr(trainer), 0.005)
        self.assertEqual(callback.wait, 0)
        self.assertEqual(callback.last_losses, [])
        self.assertFalse(trainer.stop_training)

    def test_no_reduction_on_decreasing_loss(self):
        """A decreasing monitored loss keeps the learning rate."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.5, patience=2, cooldown=0, min_lr=1e-6)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        _run_epochs(callback, [2.0, 1.5, 1.4, 1.0])
        self.assertEqual(_lr(trainer), 0.01)
        self.assertEqual(callback.wait, 4)

    def test_threshold(self):
        """Only slopes above ``threshold`` trigger a reduction."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.5, patience=2, cooldown=0, threshold=1.0)
        callback.set_trainer(trainer)
        _run_epochs(callback, [1.0, 1.5])
        self.assertEqual(_lr(trainer), 0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.5, patience=2, cooldown=0, threshold=-1.0)
        callback.set_trainer(trainer)
        _run_epochs(callback, [2.0, 1.5])
        self.assertAlmostEqual(_lr(trainer), 0.005)

    def test_reduction_clipped_at_min_lr(self):
        """The reduced learning rate never goes below ``min_lr``."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.1, patience=2, cooldown=0, min_lr=0.004)
        callback.set_trainer(trainer)
        _run_epochs(callback, [1.0, 2.0])
        self.assertAlmostEqual(_lr(trainer), 0.004)

    def test_stop_training_at_min_lr(self):
        """At ``min_lr`` a stalled loss stops the training."""
        trainer = DummyTrainer(lr=1e-6)
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.5, patience=2, cooldown=0, min_lr=1e-6)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        _run_epochs(callback, [2.0, 3.0])
        self.assertTrue(trainer.stop_training)
        self.assertEqual(_lr(trainer), 1e-6)

    def test_cooldown(self):
        """After a reduction the loss is not monitored for ``cooldown`` epochs."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.5, patience=2, cooldown=2, min_lr=1e-6)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        _run_epochs(callback, [1.0, 2.0])
        self.assertAlmostEqual(_lr(trainer), 0.005)
        self.assertEqual(callback.cooldown_counter, 2)
        _run_epochs(callback, [3.0, 4.0], start_epoch=2)
        self.assertEqual(callback.cooldown_counter, 0)
        self.assertEqual(callback.last_losses, [])
        self.assertAlmostEqual(_lr(trainer), 0.005)
        _run_epochs(callback, [5.0, 6.0], start_epoch=4)
        self.assertAlmostEqual(_lr(trainer), 0.0025)

    def test_train_begin_resets_state(self):
        """``on_train_begin`` resets counters and loss history."""
        callback = lrs.LRAdaptLossSlopeEarlyStop(patience=5)
        callback.set_trainer(DummyTrainer())
        _run_epochs(callback, [1.0, 2.0])
        callback.cooldown_counter = 3
        callback.on_train_begin()
        self.assertEqual((callback.cooldown_counter, callback.wait, callback.last_losses), (0, 0, []))

    def test_missing_monitor_warning(self):
        """A missing monitored metric logs a warning and changes nothing."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(monitor='val_loss', patience=1)
        callback.set_trainer(trainer)
        logs = {'loss': 1.0}
        with self.assertLogs(level='WARNING') as captured:
            callback.on_epoch_end(0, logs=logs)
        self.assertIn('val_loss', captured.output[0])
        self.assertEqual(logs['lr'], 0.01)
        self.assertEqual(callback.wait, 0)

    def test_custom_monitor(self):
        """The monitored key can be changed."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(monitor='loss', factor=0.5, patience=2, cooldown=0)
        callback.set_trainer(trainer)
        _run_epochs(callback, [1.0, 2.0], monitor='loss')
        self.assertAlmostEqual(_lr(trainer), 0.005)

    def test_verbose_message(self):
        """With ``verbose > 0`` the reduction is printed."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.5, patience=2, cooldown=0, verbose=1)
        callback.set_trainer(trainer)
        with patch('builtins.print') as mock_print:
            _run_epochs(callback, [1.0, 2.0])
        self.assertEqual(mock_print.call_count, 1)
        message = mock_print.call_args[0][0]
        self.assertIn('Epoch 2', message)
        self.assertIn('reducing learning rate to 0.005', message)

    def test_silent_reduction(self):
        """With ``verbose=0`` nothing is printed."""
        callback = lrs.LRAdaptLossSlopeEarlyStop(factor=0.5, patience=2, cooldown=0, verbose=0)
        callback.set_trainer(DummyTrainer())
        with patch('builtins.print') as mock_print:
            _run_epochs(callback, [1.0, 2.0])
        mock_print.assert_not_called()


class TestLRSeesawAdaptLossSlopeEarlyStop(unittest.TestCase):
    """Tests of :class:`LRSeesawAdaptLossSlopeEarlyStop`."""

    def test_factor_validation(self):
        """A reduction factor >= 1 raises ValueError."""
        with self.assertRaises(ValueError):
            lrs.LRSeesawAdaptLossSlopeEarlyStop(reduction_factor=1.0)
        with self.assertRaises(ValueError):
            lrs.LRSeesawAdaptLossSlopeEarlyStop(reduction_factor=1.1)

    def test_increase_on_improvement(self):
        """Every monitored epoch increases the learning rate by ``1 + increase_factor``."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRSeesawAdaptLossSlopeEarlyStop(increase_factor=0.1, patience=10)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        logs = _run_epochs(callback, [3.0, 2.0, 1.0])
        self.assertAlmostEqual(_lr(trainer), 0.01 * 1.1**3)
        self.assertAlmostEqual(logs[1]['lr'], 0.01 * 1.1)

    def test_reduction_on_increasing_loss(self):
        """An increasing loss reduces the (increased) learning rate."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRSeesawAdaptLossSlopeEarlyStop(reduction_factor=0.5, increase_factor=0.1,
                                                       patience=2, cooldown=1, min_lr=1e-6)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        _run_epochs(callback, [1.0, 2.0])
        self.assertAlmostEqual(_lr(trainer), 0.01 * 1.1**2 * 0.5)
        self.assertEqual(callback.cooldown_counter, 1)
        _run_epochs(callback, [3.0], start_epoch=2)
        self.assertEqual(callback.cooldown_counter, 0)
        self.assertEqual(callback.last_losses, [])

    def test_no_reduction_with_threshold(self):
        """Slopes below ``threshold`` only increase the learning rate."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRSeesawAdaptLossSlopeEarlyStop(reduction_factor=0.5, increase_factor=0.0,
                                                       patience=2, cooldown=0, threshold=1.0)
        callback.set_trainer(trainer)
        _run_epochs(callback, [2.0, 1.5, 1.4])
        self.assertEqual(_lr(trainer), 0.01)

    def test_stop_training_at_min_lr(self):
        """At ``min_lr`` a stalled loss stops the training."""
        trainer = DummyTrainer(lr=1e-6)
        callback = lrs.LRSeesawAdaptLossSlopeEarlyStop(reduction_factor=0.5, increase_factor=0.0,
                                                       patience=2, cooldown=0, min_lr=1e-6)
        callback.set_trainer(trainer)
        callback.on_train_begin()
        _run_epochs(callback, [2.0, 3.0])
        self.assertTrue(trainer.stop_training)

    def test_missing_monitor_warning(self):
        """A missing monitored metric logs a warning and leaves the learning rate unchanged."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRSeesawAdaptLossSlopeEarlyStop(increase_factor=0.5)
        callback.set_trainer(trainer)
        with self.assertLogs(level='WARNING'):
            callback.on_epoch_end(0, logs={'other': 1.0})
        self.assertEqual(_lr(trainer), 0.01)

    def test_verbose_message(self):
        """With ``verbose > 0`` the reduction is printed."""
        trainer = DummyTrainer(lr=0.01)
        callback = lrs.LRSeesawAdaptLossSlopeEarlyStop(reduction_factor=0.5, increase_factor=0.0,
                                                       patience=2, cooldown=0, verbose=2)
        callback.set_trainer(trainer)
        with patch('builtins.print') as mock_print:
            _run_epochs(callback, [1.0, 2.0])
        self.assertEqual(mock_print.call_count, 1)
        self.assertIn('reducing learning rate', mock_print.call_args[0][0])
        self.assertAlmostEqual(_lr(trainer), 0.005)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
