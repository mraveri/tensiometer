"""Tests for the PyTorch training loop and for the training of the normalizing flows."""

#########################################################################################################
# Imports

import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('agg')
import numpy as np
import torch
from getdist import MCSamples

import tensiometer.synthetic_probability.synthetic_probability as sp
from tensiometer.synthetic_probability import loss_functions as lf
from tensiometer.synthetic_probability import lr_schedulers as lrs
from tensiometer.synthetic_probability import tensor_utilities as tu
from tensiometer.synthetic_probability import training
from tensiometer.utilities import stats_utilities as stutils

#########################################################################################################
# Toy model helpers


class ShiftModel(torch.nn.Module):
    """Gaussian with unit covariance and a learnable mean."""

    def __init__(self, dimension=2):
        """Create the learnable shift, initialized at zero."""
        super(ShiftModel, self).__init__()
        self.shift = torch.nn.Parameter(torch.zeros(dimension, dtype=tu.get_precision(), device=tu.get_device()))

    def reset_parameters(self):
        """Set the shift back to zero."""
        with torch.no_grad():
            self.shift.zero_()

    def log_prob(self, x):
        """Gaussian log density of ``x`` with shape ``(N, D)``."""
        dimension = x.shape[-1]
        return -0.5 * torch.sum((x - self.shift)**2, dim=-1) - 0.5 * dimension * np.log(2. * np.pi)


class RecordingCallback(training.Callback):
    """Callback recording the hooks it receives."""

    def __init__(self, name='recorder', events=None):
        """Store events in ``events`` (a new list by default)."""
        self.name = name
        self.events = events if events is not None else []

    def on_train_begin(self, logs=None):
        """Record the event."""
        self.events.append((self.name, 'train_begin'))

    def on_train_end(self, logs=None):
        """Record the event."""
        self.events.append((self.name, 'train_end'))

    def on_epoch_begin(self, epoch, logs=None):
        """Record the event."""
        self.events.append((self.name, 'epoch_begin', epoch))

    def on_epoch_end(self, epoch, logs=None):
        """Record the event and the epoch log keys."""
        self.events.append((self.name, 'epoch_end', epoch, tuple(sorted(logs.keys()))))

    def on_train_batch_begin(self, batch, logs=None):
        """Record the event."""
        self.events.append((self.name, 'batch_begin', batch))

    def on_train_batch_end(self, batch, logs=None):
        """Record the event."""
        self.events.append((self.name, 'batch_end', batch))


def _toy_data(num_samples=200, dimension=2, center=3.0, seed=0):
    """Gaussian samples centered at ``center`` as a numpy array of the active precision."""
    rng = np.random.default_rng(seed)
    return (center + rng.normal(size=(num_samples, dimension))).astype(tu.np_prec)


def _toy_trainer(learning_rate=0.1, **kwargs):
    """Trainer of a :class:`ShiftModel` with the standard loss."""
    model = ShiftModel()
    trainer = training.Trainer(model, model.log_prob, lf.standard_loss(), learning_rate=learning_rate, **kwargs)
    return model, trainer

#########################################################################################################
# Flow helpers


def _gaussian_samples(num_samples=600, seed=0):
    """Correlated 2-D Gaussian samples and their log posterior."""
    rng = np.random.default_rng(seed)
    mean = np.array([0.0, 1.0])
    cov = np.array([[1.0, 0.6], [0.6, 2.0]])
    samples = rng.multivariate_normal(mean, cov, size=num_samples)
    diffs = samples - mean
    chi2 = np.einsum('ij,jk,ik->i', diffs, np.linalg.inv(cov), diffs)
    return samples, 0.5 * chi2


def _make_chain(with_loglikes=True, weights=None, num_samples=600, seed=0):
    """Getdist chain of a correlated 2-D Gaussian."""
    samples, loglikes = _gaussian_samples(num_samples=num_samples, seed=seed)
    return MCSamples(samples=samples,
                     loglikes=loglikes if with_loglikes else None,
                     weights=weights,
                     names=['a', 'b'],
                     labels=['a', 'b'],
                     sampler='uncorrelated',
                     settings={'ignore_rows': 0.0})


def _make_flow(chain, seed=0, **kwargs):
    """Small affine MAF on ``chain``, silent and seeded."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    options = {
        'feedback': 0,
        'plot_every': 0,
        'trainable_bijector': 'AutoregressiveFlow',
        'transformation_type': 'affine',
        'n_transformations': 1,
        'hidden_units': [4],
        'rng': np.random.default_rng(seed),
    }
    options.update(kwargs)
    return sp.FlowCallback(chain, **options)


def _state_arrays(module):
    """Copy of a module state dict as numpy arrays."""
    return {key: tu.to_numpy(value).copy() for key, value in module.state_dict().items()}


def _states_equal(state_1, state_2):
    """Whether two numpy state dicts are identical."""
    if state_1.keys() != state_2.keys():
        return False
    return all(np.array_equal(state_1[key], state_2[key]) for key in state_1)

#########################################################################################################
# Trainer tests


class TestTrainer(unittest.TestCase):
    """Tests of :class:`~tensiometer.synthetic_probability.training.Trainer` on a toy model."""

    def setUp(self):
        """Seed the random generators."""
        torch.manual_seed(0)
        np.random.seed(0)

    def test_loss_decreases(self):
        """The loss decreases over 20 epochs and the shift approaches the data mean."""
        data = _toy_data()
        model, trainer = _toy_trainer(learning_rate=0.1)
        history = trainer.fit(data, epochs=20, batch_size=32, verbose=0)
        losses = history.history['loss']
        self.assertEqual(len(losses), 20)
        self.assertLess(losses[-1], losses[0])
        self.assertLess(losses[-1], losses[0] - 3.0)
        np.testing.assert_allclose(tu.to_numpy(model.shift), np.mean(data, axis=0), atol=0.3)
        self.assertEqual(model.shift.dtype, tu.get_precision())

    def test_history_keys_and_lengths(self):
        """History records loss, val_loss and lr for every epoch."""
        data = _toy_data()
        _, trainer = _toy_trainer()
        validation = (_toy_data(num_samples=50, seed=1), None)
        history = trainer.fit(data, validation_data=validation, epochs=3, batch_size=50, verbose=0)
        self.assertIsInstance(history, training.History)
        self.assertEqual(set(history.history.keys()), {'loss', 'val_loss', 'lr'})
        self.assertEqual(history.epoch, [0, 1, 2])
        for key in ['loss', 'val_loss', 'lr']:
            self.assertEqual(len(history.history[key]), 3)
            self.assertTrue(all(isinstance(value, float) for value in history.history[key]))
        self.assertEqual(history.history['lr'], [0.1, 0.1, 0.1])

    def test_history_without_validation(self):
        """Without validation data there is no val_loss."""
        _, trainer = _toy_trainer()
        history = trainer.fit(_toy_data(), epochs=2, batch_size=64, verbose=0)
        self.assertEqual(set(history.history.keys()), {'loss', 'lr'})

    def test_history_callback(self):
        """History appends every log key and resets the epoch list on train begin."""
        history = training.History()
        history.on_epoch_end(0, {'loss': 1.0, 'extra': 2.0})
        history.on_epoch_end(1, None)
        self.assertEqual(history.history, {'loss': [1.0], 'extra': [2.0]})
        self.assertEqual(history.epoch, [0, 1])
        history.on_train_begin()
        self.assertEqual(history.epoch, [])

    def test_callbacks_fire_in_order(self):
        """Hooks fire in order, for every callback in list order."""
        events = []
        callbacks = [RecordingCallback('first', events), RecordingCallback('second', events)]
        _, trainer = _toy_trainer()
        data = _toy_data(num_samples=8)
        trainer.fit(data, validation_data=(data,), epochs=2, batch_size=4, callbacks=callbacks, verbose=0)
        for callback in callbacks:
            self.assertIs(callback.trainer, trainer)
        log_keys = ('loss', 'lr', 'val_loss')
        expected = [('train_begin',)]
        for epoch in range(2):
            expected.append(('epoch_begin', epoch))
            for batch in range(2):
                expected.append(('batch_begin', batch))
                expected.append(('batch_end', batch))
            expected.append(('epoch_end', epoch, log_keys))
        expected.append(('train_end',))
        expanded = []
        for event in expected:
            expanded.append(('first',) + event)
            expanded.append(('second',) + event)
        self.assertEqual(events, expanded)

    def test_base_callback_hooks_are_noops(self):
        """The base callback hooks do nothing."""
        callback = training.Callback()
        self.assertIsNone(callback.trainer)
        self.assertIsNone(callback.on_train_begin({}))
        self.assertIsNone(callback.on_epoch_begin(0, {}))
        self.assertIsNone(callback.on_train_batch_begin(0, {}))
        self.assertIsNone(callback.on_train_batch_end(0, {}))
        self.assertIsNone(callback.on_epoch_end(0, {}))
        self.assertIsNone(callback.on_train_end({}))

    def test_nan_termination(self):
        """A non-finite loss at the start of an epoch stops training and that epoch is not logged."""
        model = ShiftModel()
        calls = {'count': 0}

        def log_prob_fn(x):
            calls['count'] += 1
            values = model.log_prob(x)
            if calls['count'] >= 3:
                values = values * float('nan')
            return values

        trainer = training.Trainer(model, log_prob_fn, lf.standard_loss(), learning_rate=0.1)
        recorder = RecordingCallback()
        with patch('builtins.print') as mock_print:
            history = trainer.fit(_toy_data(num_samples=8), epochs=5, batch_size=4, callbacks=[recorder], verbose=0)
        self.assertTrue(trainer.stop_training)
        self.assertEqual(calls['count'], 3)
        self.assertEqual(history.epoch, [0])
        self.assertEqual(len(history.history['loss']), 1)
        self.assertTrue(np.isfinite(history.history['loss'][0]))
        epoch_ends = [event[2] for event in recorder.events if event[1] == 'epoch_end']
        self.assertEqual(epoch_ends, [0])
        self.assertTrue(np.all(np.isfinite(tu.to_numpy(model.shift))))
        self.assertIn('Invalid loss', mock_print.call_args_list[0][0][0])
        self.assertEqual(recorder.events[-1], ('recorder', 'train_end'))
        batch_ends = [event for event in recorder.events if event[1] == 'batch_end']
        self.assertEqual(len(batch_ends), 2)

    def test_nan_termination_mid_epoch(self):
        """A non-finite loss mid epoch logs the epoch from its finite batch losses only."""
        model = ShiftModel()
        calls = {'count': 0}
        batch_losses = []

        def log_prob_fn(x):
            calls['count'] += 1
            values = model.log_prob(x)
            if calls['count'] >= 4:
                values = values * float('nan')
            return values

        class LossRecorder(training.Callback):
            """Record the batch losses."""
            def on_train_batch_end(self, batch, logs=None):
                """Record the loss."""
                batch_losses.append(logs['loss'])

        trainer = training.Trainer(model, log_prob_fn, lf.standard_loss(), learning_rate=0.1)
        with patch('builtins.print'):
            history = trainer.fit(_toy_data(num_samples=8), epochs=5, batch_size=4,
                                  callbacks=[LossRecorder()], verbose=1)
        self.assertTrue(trainer.stop_training)
        self.assertEqual(history.epoch, [0, 1])
        self.assertEqual(len(batch_losses), 3)
        self.assertTrue(np.all(np.isfinite(history.history['loss'])))
        self.assertAlmostEqual(history.history['loss'][1], batch_losses[2])

    def test_stop_training_resets_on_fit(self):
        """A new fit clears a previous stop request."""
        _, trainer = _toy_trainer()
        trainer.stop_training = True
        history = trainer.fit(_toy_data(), epochs=2, batch_size=100, verbose=0)
        self.assertFalse(trainer.stop_training)
        self.assertEqual(len(history.epoch), 2)

    def test_wrap_around_batching(self):
        """When ``steps * batch_size > n`` the permutations are concatenated."""
        num_samples, batch_size, steps = 5, 4, 3
        model = ShiftModel(dimension=1)
        batches = []

        def log_prob_fn(x):
            batches.append(tu.to_numpy(x[:, 0]).astype(int).tolist())
            return model.log_prob(x)

        trainer = training.Trainer(model, log_prob_fn, lf.standard_loss(), learning_rate=1e-6)
        data = np.arange(num_samples, dtype=tu.np_prec).reshape(-1, 1)
        trainer.fit(data, epochs=1, batch_size=batch_size, steps_per_epoch=steps, verbose=0)
        self.assertEqual(len(batches), steps)
        self.assertTrue(all(len(batch) == batch_size for batch in batches))
        flat = np.concatenate(batches)
        counts = np.bincount(flat, minlength=num_samples)
        self.assertEqual(int(np.sum(counts)), steps * batch_size)
        self.assertTrue(np.all(counts >= 2))
        np.testing.assert_array_equal(np.sort(flat[:num_samples]), np.arange(num_samples))
        np.testing.assert_array_equal(np.sort(flat[num_samples:2 * num_samples]), np.arange(num_samples))

    def test_default_batching(self):
        """Default batch size is 32 and steps cover the data set once."""
        model = ShiftModel()
        sizes = []

        def log_prob_fn(x):
            sizes.append(x.shape[0])
            return model.log_prob(x)

        trainer = training.Trainer(model, log_prob_fn, lf.standard_loss())
        trainer.fit(_toy_data(num_samples=70), epochs=1, verbose=0)
        self.assertEqual(sizes, [32, 32, 32])

    def test_sample_weights_and_targets_are_batched(self):
        """Targets and weights are sliced with the same indices as the samples."""
        model = ShiftModel(dimension=1)
        seen = []

        def loss(y_true, y_pred, sample_weight=None):
            seen.append((tu.to_numpy(y_true).copy(), tu.to_numpy(sample_weight).copy()))
            return lf.standard_loss()(y_true, y_pred, sample_weight=sample_weight)

        x_seen = []

        def log_prob_fn(x):
            x_seen.append(tu.to_numpy(x[:, 0]).copy())
            return model.log_prob(x)

        trainer = training.Trainer(model, log_prob_fn, loss)
        data = np.arange(6, dtype=tu.np_prec).reshape(-1, 1)
        trainer.fit(data, y=10. * data[:, 0], sample_weight=100. * data[:, 0], epochs=1, batch_size=3, verbose=0)
        for x_batch, (y_batch, w_batch) in zip(x_seen, seen):
            np.testing.assert_allclose(y_batch, 10. * x_batch)
            np.testing.assert_allclose(w_batch, 100. * x_batch)

    def test_filter_kwargs_with_fit(self):
        """``filter_kwargs(kwargs, trainer.fit)`` keeps fit options and drops the others."""
        _, trainer = _toy_trainer()
        kwargs = {'epochs': 3, 'batch_size': 8, 'pop_size': 2, 'feedback': 1, 'verbose': 0}
        filtered = stutils.filter_kwargs(kwargs, trainer.fit)
        self.assertEqual(filtered, {'epochs': 3, 'batch_size': 8, 'verbose': 0})
        history = trainer.fit(_toy_data(), **filtered)
        self.assertEqual(len(history.epoch), 3)

    def test_verbose_one_prints_epoch_lines(self):
        """``verbose=1`` prints one line per epoch."""
        _, trainer = _toy_trainer()
        data = _toy_data()
        with patch('builtins.print') as mock_print:
            trainer.fit(data, validation_data=(data,), epochs=2, batch_size=100, verbose=1)
        messages = [call[0][0] for call in mock_print.call_args_list]
        self.assertEqual(len(messages), 2)
        self.assertTrue(messages[0].startswith('Epoch 1/2 - loss: '))
        self.assertIn('val_loss', messages[0])
        self.assertIn('lr', messages[1])

    def test_verbose_zero_is_silent(self):
        """``verbose=0`` prints nothing."""
        _, trainer = _toy_trainer()
        with patch('builtins.print') as mock_print:
            trainer.fit(_toy_data(), epochs=2, batch_size=100, verbose=0)
        mock_print.assert_not_called()

    def test_verbose_progress_bar(self):
        """``verbose=-1`` runs with a tqdm progress bar."""
        _, trainer = _toy_trainer()
        data = _toy_data()
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream), patch('builtins.print') as mock_print:
            history = trainer.fit(data, validation_data=(data,), epochs=3, batch_size=100, verbose=-1)
        mock_print.assert_not_called()
        self.assertEqual(len(history.epoch), 3)
        self.assertIn('Training', stream.getvalue())

    def test_module_without_parameters(self):
        """A module without trainable parameters has no optimizer and cannot be fit."""
        module = torch.nn.Identity()
        trainer = training.Trainer(module, lambda x: -torch.sum(x**2, dim=-1), lf.standard_loss())
        self.assertIsNone(trainer.optimizer)
        self.assertIsNone(trainer.get_learning_rate())
        with self.assertRaises(ValueError):
            trainer.fit(_toy_data(), epochs=1, verbose=0)
        self.assertTrue(np.isfinite(trainer.evaluate(_toy_data())))

    def test_frozen_parameters_are_not_optimized(self):
        """Parameters with ``requires_grad=False`` are left out of the optimizer."""
        model = ShiftModel()
        model.shift.requires_grad_(False)
        trainer = training.Trainer(model, model.log_prob, lf.standard_loss())
        self.assertIsNone(trainer.optimizer)

    def test_reset_optimizer(self):
        """``reset_optimizer`` creates a fresh Adam with the Keras epsilon."""
        _, trainer = _toy_trainer(learning_rate=0.05)
        trainer.fit(_toy_data(), epochs=1, batch_size=50, verbose=0)
        self.assertGreater(len(trainer.optimizer.state), 0)
        old_optimizer = trainer.optimizer
        trainer.reset_optimizer()
        self.assertIsNot(trainer.optimizer, old_optimizer)
        self.assertIsInstance(trainer.optimizer, torch.optim.Adam)
        self.assertEqual(len(trainer.optimizer.state), 0)
        self.assertEqual(trainer.optimizer.param_groups[0]['eps'], 1e-7)
        self.assertEqual(trainer.get_learning_rate(), 0.05)

    def test_global_clipnorm(self):
        """With ``global_clipnorm`` the gradient norm is clipped at every step."""
        clipnorm = 1e-3
        model, trainer = _toy_trainer(learning_rate=0.01, global_clipnorm=clipnorm)
        with patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as mock_clip:
            trainer.fit(_toy_data(), epochs=1, batch_size=50, verbose=0)
        self.assertEqual(mock_clip.call_count, 4)
        self.assertEqual(mock_clip.call_args[0][1], clipnorm)
        gradient_norm = float(torch.linalg.norm(model.shift.grad))
        self.assertLessEqual(gradient_norm, clipnorm * (1. + 1e-4))

    def test_no_clipping_by_default(self):
        """Without ``global_clipnorm`` the gradients are not clipped."""
        _, trainer = _toy_trainer()
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            trainer.fit(_toy_data(), epochs=1, batch_size=50, verbose=0)
        mock_clip.assert_not_called()

    def test_evaluate(self):
        """``evaluate`` averages the batch losses weighted by the batch sizes."""
        data = _toy_data(num_samples=10)
        weights = np.linspace(1.0, 2.0, 10)
        model, trainer = _toy_trainer()
        with torch.no_grad():
            per_sample = -tu.to_numpy(model.log_prob(tu.to_tensor(data, device=model.shift.device)))
        expected = 0.
        for start, stop in [(0, 4), (4, 8), (8, 10)]:
            chunk = np.average(per_sample[start:stop], weights=weights[start:stop])
            expected += chunk * (stop - start) / 10.
        value = trainer.evaluate(data, sample_weight=weights, batch_size=4)
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, expected, places=4)
        whole = trainer.evaluate(data, sample_weight=weights)
        self.assertAlmostEqual(whole, np.average(per_sample, weights=weights), places=4)
        self.assertTrue(np.isnan(trainer.evaluate(data[:0])))

    def test_evaluate_does_not_build_graph(self):
        """``evaluate`` runs without gradients and leaves the parameters untouched."""
        model, trainer = _toy_trainer()
        trainer.evaluate(_toy_data())
        self.assertIsNone(model.shift.grad)

    def test_validation_uses_evaluate(self):
        """The epoch val_loss equals ``evaluate`` on the validation data after the epoch."""
        data = _toy_data()
        validation = _toy_data(num_samples=30, seed=3)
        _, trainer = _toy_trainer()
        history = trainer.fit(data, validation_data=(validation, None, None), epochs=1, batch_size=16, verbose=0)
        expected = trainer.evaluate(validation, batch_size=16)
        self.assertAlmostEqual(history.history['val_loss'][0], expected, places=5)

    def test_device_follows_module(self):
        """The trainer runs on the device of the module."""
        model, trainer = _toy_trainer()
        self.assertEqual(trainer.device, model.shift.device)

#########################################################################################################
# Schedulers with a real trainer


class TestSchedulersWithTrainer(unittest.TestCase):
    """Learning rate schedulers attached to a real trainer with Adam."""

    def setUp(self):
        """Seed the random generators."""
        torch.manual_seed(0)
        np.random.seed(0)

    def test_exponential_decay_records_lrs(self):
        """ExponentialDecayScheduler records decreasing per-batch learning rates."""
        _, trainer = _toy_trainer(learning_rate=0.1)
        scheduler = lrs.ExponentialDecayScheduler(lr_max=0.1, lr_min=0.001, roll_off_step=2, steps=6)
        history = trainer.fit(_toy_data(), epochs=2, batch_size=40, steps_per_epoch=3,
                              callbacks=[scheduler], verbose=0)
        self.assertEqual(len(scheduler.lrs), 6)
        self.assertEqual(scheduler.lrs[0], 0.1)
        self.assertTrue(np.all(np.diff(scheduler.lrs) < 0.0))
        self.assertAlmostEqual(trainer.get_learning_rate(), 0.001)
        self.assertAlmostEqual(history.history['lr'][-1], 0.001)
        self.assertAlmostEqual(history.history['lr'][0], scheduler.lrs[3])

    def test_adapt_loss_slope_stops_at_min_lr(self):
        """LRAdaptLossSlopeEarlyStop sets ``trainer.stop_training`` at ``min_lr``."""
        _, trainer = _toy_trainer(learning_rate=1e-3)
        scheduler = lrs.LRAdaptLossSlopeEarlyStop(monitor='loss', patience=2, cooldown=0,
                                                  min_lr=1e-3, threshold=-np.inf)
        history = trainer.fit(_toy_data(), epochs=10, batch_size=100, callbacks=[scheduler], verbose=0)
        self.assertTrue(trainer.stop_training)
        self.assertEqual(history.epoch, [0, 1])

    def test_adapt_loss_slope_reduces_lr(self):
        """LRAdaptLossSlopeEarlyStop reduces the learning rate of the Adam optimizer."""
        _, trainer = _toy_trainer(learning_rate=1e-2)
        scheduler = lrs.LRAdaptLossSlopeEarlyStop(monitor='loss', factor=0.5, patience=2, cooldown=100,
                                                  min_lr=1e-6, threshold=-np.inf)
        history = trainer.fit(_toy_data(), epochs=4, batch_size=100, callbacks=[scheduler], verbose=0)
        self.assertFalse(trainer.stop_training)
        self.assertAlmostEqual(trainer.get_learning_rate(), 5e-3)
        self.assertEqual(history.history['lr'][:2], [1e-2, 1e-2])
        self.assertAlmostEqual(history.history['lr'][-1], 5e-3)

#########################################################################################################
# Module helpers


class TestModuleHelpers(unittest.TestCase):
    """Tests of ``count_parameters`` and ``reset_module_parameters``."""

    def test_count_parameters(self):
        """Only trainable parameters are counted."""
        module = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        self.assertEqual(training.count_parameters(module), 3 * 2 + 2 + 2 + 1)
        module[1].weight.requires_grad_(False)
        self.assertEqual(training.count_parameters(module), 3 * 2 + 2 + 1)
        self.assertEqual(training.count_parameters(torch.nn.Identity()), 0)
        self.assertIsInstance(training.count_parameters(module), int)

    def test_reset_module_parameters(self):
        """Every submodule with ``reset_parameters`` is re-initialized."""
        torch.manual_seed(0)
        shift = ShiftModel()
        module = torch.nn.ModuleList([torch.nn.Linear(3, 2), shift])
        with torch.no_grad():
            shift.shift.fill_(5.0)
        before = module[0].weight.detach().clone()
        training.reset_module_parameters(module)
        self.assertFalse(torch.equal(before, module[0].weight))
        np.testing.assert_array_equal(tu.to_numpy(shift.shift), [0.0, 0.0])

#########################################################################################################
# Flow training tests


class TestFlowTraining(unittest.TestCase):
    """Training of a small affine MAF on a 2-D Gaussian chain."""

    @classmethod
    def setUpClass(cls):
        """Build the chains shared by the tests."""
        cls.chain = _make_chain(with_loglikes=True)

    def setUp(self):
        """Seed the random generators."""
        torch.manual_seed(0)
        np.random.seed(0)

    def test_flow_uses_trainer(self):
        """The flow is a training callback with a trainer on its trainable bijector."""
        flow = _make_flow(self.chain)
        self.assertIsInstance(flow, training.Callback)
        self.assertIsInstance(flow.trainer, training.Trainer)
        self.assertIs(flow.trainer.module, flow.trainable_bijector)
        self.assertGreater(training.count_parameters(flow.trainable_bijector), 0)
        x, y, w = flow.training_dataset
        self.assertEqual(x.dtype, tu.get_precision())
        self.assertEqual(tuple(x.shape), (flow.num_training_samples, 2))
        self.assertIsNotNone(y)
        self.assertEqual(tuple(w.shape), (flow.num_training_samples,))

    def test_val_loss_decreases(self):
        """A few epochs reduce the validation loss and fill the flow log."""
        flow = _make_flow(self.chain)
        history = flow.train(epochs=8, verbose=0)
        val_loss = history.history['val_loss']
        self.assertEqual(len(val_loss), 8)
        self.assertLess(val_loss[-1], val_loss[0])
        self.assertTrue(flow.is_trained)
        self.assertEqual(flow.log['val_loss'], val_loss)
        for key in flow.training_metrics:
            self.assertEqual(len(flow.log[key]), 8, key)

    def test_train_verbose_zero(self):
        """``train(verbose=0)`` runs silently (UnboundLocalError in the TensorFlow version)."""
        flow = _make_flow(self.chain)
        with patch('builtins.print') as mock_print:
            history = flow.train(epochs=1, verbose=0)
        self.assertIsInstance(history, training.History)
        mock_print.assert_not_called()

    def test_train_verbose_one(self):
        """``train(verbose=1)`` prints one line per epoch."""
        flow = _make_flow(self.chain)
        with patch('builtins.print') as mock_print:
            flow.train(epochs=2, verbose=1)
        messages = [call[0][0] for call in mock_print.call_args_list if call[0]]
        self.assertEqual(len([message for message in messages if str(message).startswith('Epoch ')]), 2)

    def test_train_batching_options(self):
        """``batch_size`` and ``steps_per_epoch`` are forwarded to the trainer."""
        flow = _make_flow(self.chain)
        recorded = {}
        original_fit = flow.trainer.fit

        def recording_fit(*args, **kwargs):
            recorded.update(kwargs)
            return original_fit(*args, **kwargs)

        flow.trainer.fit = recording_fit
        flow.train(epochs=1, verbose=0)
        self.assertEqual(recorded['steps_per_epoch'], 20)
        self.assertEqual(recorded['batch_size'], int(flow.num_training_samples / 20))
        flow.train(epochs=1, batch_size=100, verbose=0)
        self.assertEqual(recorded['batch_size'], 100)
        self.assertEqual(recorded['steps_per_epoch'], int(flow.num_training_samples / 100))
        flow.train(epochs=1, batch_size=10000, verbose=0, pop_size=3, feedback=2)
        self.assertEqual(recorded['steps_per_epoch'], 1)
        self.assertNotIn('pop_size', recorded)
        self.assertNotIn('feedback', recorded)

    def test_callbacks_empty_disables_scheduler(self):
        """``callbacks=[]`` trains with the flow as the only callback."""
        flow = _make_flow(self.chain)
        recorded = []
        original_fit = flow.trainer.fit

        def recording_fit(*args, **kwargs):
            recorded.append(kwargs['callbacks'])
            return original_fit(*args, **kwargs)

        flow.trainer.fit = recording_fit
        flow.train(epochs=1, verbose=0)
        self.assertEqual(len(recorded[-1]), 2)
        self.assertIs(recorded[-1][0], flow)
        self.assertIsInstance(recorded[-1][1], lrs.LRAdaptLossSlopeEarlyStop)
        self.assertEqual(recorded[-1][1].min_lr, flow.final_learning_rate)
        flow.train(epochs=1, verbose=0, callbacks=[])
        self.assertEqual(recorded[-1], [flow])
        extra = RecordingCallback()
        flow.train(epochs=1, verbose=0, callbacks=[extra])
        self.assertEqual(recorded[-1], [flow, extra])
        self.assertEqual(extra.events[0], ('recorder', 'train_begin'))

    def test_lr_scheduler_selection(self):
        """The ``lr_scheduler`` keyword selects the scheduler and forwards its options."""
        flow = _make_flow(self.chain)
        recorded = []
        original_fit = flow.trainer.fit

        def recording_fit(*args, **kwargs):
            recorded.append(kwargs['callbacks'])
            return original_fit(*args, **kwargs)

        flow.trainer.fit = recording_fit
        flow.train(epochs=1, verbose=0, lr_scheduler='LRSeesawAdaptLossSlopeEarlyStop',
                   increase_factor=0.01, patience=7)
        scheduler = recorded[-1][1]
        self.assertIsInstance(scheduler, lrs.LRSeesawAdaptLossSlopeEarlyStop)
        self.assertEqual(scheduler.increase_factor, 0.01)
        self.assertEqual(scheduler.patience, 7)
        self.assertEqual(scheduler.min_lr, flow.final_learning_rate)
        flow.train(epochs=1, verbose=0, lr_scheduler=None)
        self.assertEqual(recorded[-1], [flow])
        with patch('builtins.print') as mock_print:
            flow.train(epochs=1, verbose=0, lr_scheduler='NotAScheduler')
        self.assertEqual(recorded[-1], [flow])
        self.assertIn('NotAScheduler', mock_print.call_args_list[0][0][0])

    def test_learning_rate_options(self):
        """``learning_rate`` sets the Adam learning rate and the default ``final_learning_rate``."""
        flow = _make_flow(self.chain, learning_rate=0.01, global_clipnorm=0.5)
        self.assertEqual(flow.trainer.get_learning_rate(), 0.01)
        self.assertAlmostEqual(flow.final_learning_rate, 1e-5)
        self.assertEqual(flow.trainer.global_clipnorm, 0.5)

    def test_global_train_population(self):
        """``global_train`` re-initializes the members and restores the best one."""
        flow = _make_flow(self.chain)
        initial_states, final_states, val_losses = [], [], []
        original_train = flow.train

        def recording_train(**kwargs):
            initial_states.append(_state_arrays(flow.trainable_bijector))
            history = original_train(**kwargs)
            final_states.append(_state_arrays(flow.trainable_bijector))
            val_losses.append(history.history['val_loss'][-1])
            return history

        constructed_state = _state_arrays(flow.trainable_bijector)
        flow.train = recording_train
        best_loss, best_val_loss = flow.global_train(pop_size=2, epochs=2, verbose=0)
        del flow.train
        self.assertEqual(len(initial_states), 2)
        self.assertTrue(_states_equal(initial_states[0], constructed_state))
        self.assertFalse(_states_equal(initial_states[0], initial_states[1]))
        best_index = int(np.argmin(val_losses))
        self.assertAlmostEqual(best_val_loss, val_losses[best_index])
        self.assertTrue(_states_equal(_state_arrays(flow.trainable_bijector), final_states[best_index]))
        self.assertEqual(flow.log['population'], best_index + 1)
        self.assertEqual(len(flow.population_logs), 2)
        self.assertEqual([log['population'] for log in flow.population_logs], [1, 2])
        self.assertEqual(len(flow.log['loss']), 2)
        self.assertAlmostEqual(best_loss, flow.log['loss'][-1])

    def test_global_train_resets_optimizer(self):
        """Each population member starts with a fresh optimizer."""
        flow = _make_flow(self.chain)
        optimizers = []
        original_train = flow.train

        def recording_train(**kwargs):
            optimizers.append(flow.trainer.optimizer)
            self.assertEqual(len(flow.trainer.optimizer.state), 0)
            return original_train(**kwargs)

        flow.train = recording_train
        flow.global_train(pop_size=2, epochs=1, verbose=0)
        del flow.train
        self.assertIsNot(optimizers[0], optimizers[1])

    def test_train_after_save_and_load(self):
        """A flow reloaded from a full snapshot resumes training."""
        flow = _make_flow(self.chain)
        flow.train(epochs=2, verbose=0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'flow.pt')
            flow.save(path)
            loaded = sp.FlowCallback.load(path)
        self.assertGreater(len(loaded.trainer.optimizer.state), 0)
        self.assertIs(loaded.trainer.module, loaded.trainable_bijector)
        before = _state_arrays(loaded.trainable_bijector)
        history = loaded.train(epochs=1, verbose=0)
        self.assertEqual(len(history.epoch), 1)
        self.assertEqual(len(loaded.log['loss']), 3)
        self.assertFalse(_states_equal(before, _state_arrays(loaded.trainable_bijector)))
        self.assertEqual(len(np.intersect1d(loaded.training_idx, loaded.test_idx)), 0)


class TestFlowWeights(unittest.TestCase):
    """Sample weights of chains without log likelihoods are used in training."""

    def test_weights_change_training_loss(self):
        """Non-uniform weights give a different training loss than uniform weights."""
        num_samples = 600
        rng = np.random.default_rng(5)
        weights = rng.uniform(0.1, 3.0, size=num_samples)
        uniform_chain = _make_chain(with_loglikes=False, num_samples=num_samples)
        weighted_chain = _make_chain(with_loglikes=False, weights=weights, num_samples=num_samples)
        options = {'prior_bijector': None, 'apply_rescaling': False}
        uniform_flow = _make_flow(uniform_chain, seed=3, **options)
        weighted_flow = _make_flow(weighted_chain, seed=3, **options)
        np.testing.assert_array_equal(uniform_flow.training_samples, weighted_flow.training_samples)
        self.assertTrue(_states_equal(_state_arrays(uniform_flow.trainable_bijector),
                                      _state_arrays(weighted_flow.trainable_bijector)))
        self.assertFalse(uniform_flow.has_weights)
        self.assertTrue(weighted_flow.has_weights)
        self.assertIsNone(weighted_flow.training_dataset[1])
        np.testing.assert_allclose(tu.to_numpy(weighted_flow.training_dataset[2]),
                                   weighted_flow.training_weights, rtol=1e-6)
        torch.manual_seed(11)
        uniform_history = uniform_flow.train(epochs=1, verbose=0, callbacks=[])
        torch.manual_seed(11)
        weighted_history = weighted_flow.train(epochs=1, verbose=0, callbacks=[])
        uniform_loss = uniform_history.history['loss'][0]
        weighted_loss = weighted_history.history['loss'][0]
        self.assertGreater(abs(uniform_loss - weighted_loss), 1e-4 * abs(uniform_loss))
        self.assertNotEqual(uniform_history.history['val_loss'][0], weighted_history.history['val_loss'][0])

    def test_uniform_weights_reproducible(self):
        """Two identical uniform-weight flows with the same seeds give the same loss."""
        chain = _make_chain(with_loglikes=False)
        options = {'prior_bijector': None, 'apply_rescaling': False}
        losses = []
        for _ in range(2):
            flow = _make_flow(chain, seed=3, **options)
            torch.manual_seed(11)
            losses.append(flow.train(epochs=1, verbose=0, callbacks=[]).history['loss'][0])
        self.assertEqual(losses[0], losses[1])


class TestFlowLossModes(unittest.TestCase):
    """Training with the posterior based loss functions."""

    @classmethod
    def setUpClass(cls):
        """Build the chain shared by the tests."""
        cls.chain = _make_chain(with_loglikes=True)

    def _train_mode(self, loss_mode, **kwargs):
        """Train a flow for two epochs with ``loss_mode`` and return it."""
        np.random.seed(0)
        flow = _make_flow(self.chain, loss_mode=loss_mode, **kwargs)
        history = flow.train(epochs=2, verbose=0)
        self.assertEqual(len(history.epoch), 2)
        self.assertTrue(np.all(np.isfinite(history.history['loss'])))
        for key in flow.training_metrics:
            self.assertEqual(len(flow.log[key]), 2, key)
        return flow

    def _check_lambda_logs(self, flow):
        """Lambda logs hold two python floats summing to one."""
        for lambda_1, lambda_2 in zip(flow.log['lambda_1'], flow.log['lambda_2']):
            self.assertIsInstance(lambda_1, float)
            self.assertAlmostEqual(lambda_1 + lambda_2, 1.0)

    def test_fixed(self):
        """Fixed weights log the loss components."""
        flow = self._train_mode('fixed', alpha_lossv=0.7)
        self.assertIsInstance(flow.loss, lf.constant_weight_loss)
        self.assertEqual(flow.loss.alpha, 0.7)
        self.assertNotIn('lambda_1', flow.log)
        self.assertEqual(len(flow.log['val_ee_loss']), 2)

    def test_annealed(self):
        """Annealed weights decay after ``anneal_epoch``."""
        flow = self._train_mode('annealed', anneal_epoch=0, roll_off_nepoch=1)
        self.assertIsInstance(flow.loss, lf.annealed_weight_loss)
        self._check_lambda_logs(flow)
        self.assertEqual(flow.log['lambda_1'][0], 1.0)
        self.assertAlmostEqual(flow.log['lambda_1'][1], np.exp(-1.0))

    def test_softadapt(self):
        """SoftAdapt weights are logged."""
        flow = self._train_mode('softadapt', tau=2.0)
        self.assertIsInstance(flow.loss, lf.SoftAdapt_weight_loss)
        self.assertEqual(flow.loss.tau, 2.0)
        self._check_lambda_logs(flow)
        self.assertEqual(flow.log['lambda_1'][0], 1.0)

    def test_sharpstep(self):
        """SharpStep switches at ``step_epoch``."""
        flow = self._train_mode('sharpstep', step_epoch=1, value_1=1.0, value_2=0.2)
        self.assertIsInstance(flow.loss, lf.SharpStep)
        self._check_lambda_logs(flow)
        self.assertEqual(flow.log['lambda_1'], [1.0, 0.2])

    def test_random(self):
        """Random weights are zero or one."""
        flow = self._train_mode('random', initial_random_epoch=0)
        self.assertIsInstance(flow.loss, lf.random_weight_loss)
        self._check_lambda_logs(flow)
        self.assertEqual(flow.log['lambda_1'][0], 1.0)
        self.assertIn(flow.log['lambda_1'][1], [0.0, 1.0])

    def test_reset_keeps_loss_hyperparameters(self):
        """Resetting the optimizer resets the lambdas but keeps the loss hyperparameters."""
        flow = self._train_mode('annealed', anneal_epoch=0, roll_off_nepoch=3)
        self.assertLess(flow.loss.lambda_1, 1.0)
        flow._reset_optimizer()
        self.assertEqual(flow.loss.lambda_1, 1.0)
        self.assertEqual(flow.loss.anneal_epoch, 0)
        self.assertEqual(flow.loss.roll_off_nepoch, 3)

    def test_posterior_loss_requires_loglikes(self):
        """Posterior based losses need a chain with log likelihoods."""
        chain = _make_chain(with_loglikes=False, num_samples=200)
        with self.assertRaises(ValueError):
            _make_flow(chain, loss_mode='fixed')

    def test_unknown_loss_mode(self):
        """An unknown loss mode raises ValueError."""
        with self.assertRaises(ValueError):
            _make_flow(self.chain, loss_mode='not_a_mode')

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
