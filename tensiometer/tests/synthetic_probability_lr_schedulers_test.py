"""Tests for learning rate schedulers."""

#########################################################################################################
# Imports

import unittest
from unittest.mock import patch

import tensorflow as tf

from tensiometer.synthetic_probability import lr_schedulers as lrs

#########################################################################################################
# Helper stubs


class DummyOptimizer:
    """Dummy Optimizer test suite."""
    def __init__(self, lr=0.01):
        """Init."""
        self.lr = tf.Variable(lr, dtype=tf.float32)
        self.learning_rate = self.lr


class DummyModel:
    """Dummy Model test suite."""
    def __init__(self):
        """Init."""
        self.optimizer = DummyOptimizer()
        self.stop_training = False

#########################################################################################################
# Scheduler tests


class TestLrSchedulers(unittest.TestCase):
    """Learning-rate schedulers test suite."""
    def test_set_optimizer_lr_fallback(self):
        """Test optimizer lr fallback setter."""
        class LegacyOptimizer:
            """Legacy optimizer with only lr attribute."""
            def __init__(self, lr=0.1):
                """Init."""
                self.lr = tf.Variable(lr, dtype=tf.float32)

        opt = LegacyOptimizer()
        lrs._set_optimizer_lr(opt, 0.05)
        self.assertAlmostEqual(float(lrs._get_optimizer_lr(opt)), 0.05)

    def test_annealers(self):
        """Test Annealers."""
        exp = lrs.ExponentialDecayAnnealer(start=0.1, end=0.01, roll_off_step=1, steps=5)
        val1 = exp.step()
        self.assertLess(val1, 0.1)
        pl = lrs.PowerLawDecayAnnealer(start=0.1, end=0.01, power=2, steps=5)
        val2 = pl.step()
        val3 = pl.step()
        self.assertGreater(val2, 0.0)
        self.assertGreater(val3, 0.0)
        step = lrs.StepDecayAnnealer(start=0.1, change_every=1, steps=4, steps_per_epoch=1)
        self.assertLessEqual(step.step(), 0.1)

    def test_exponential_scheduler(self):
        """Test Exponential scheduler."""
        model = DummyModel()
        cb = lrs.ExponentialDecayScheduler(lr_max=0.1, lr_min=0.01, roll_off_step=1, steps=10)
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_train_batch_begin(batch=0)
        cb.on_train_batch_end(batch=0)
        self.assertTrue(lrs._get_optimizer_lr(model.optimizer) <= 0.1)
        # missing optimizer branch
        cb_no_opt = lrs.ExponentialDecayScheduler(lr_max=0.1, lr_min=0.01, roll_off_step=1, steps=10)
        cb_no_opt.set_model(type("NoOpt", (), {})())
        self.assertIsNone(cb_no_opt.get_lr())
        cb_no_opt.set_lr(0.05)

    def test_powerlaw_scheduler(self):
        """Test Powerlaw scheduler."""
        model = DummyModel()
        cb = lrs.PowerLawDecayScheduler(lr_max=0.1, lr_min=0.01, power=2, steps=10)
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_train_batch_begin(batch=0)
        initial_lr = lrs._get_optimizer_lr(model.optimizer)
        for _ in range(5):
            cb.on_train_batch_end(batch=0)
        self.assertLess(lrs._get_optimizer_lr(model.optimizer), initial_lr)
        # missing optimizer branch
        cb_no_opt = lrs.PowerLawDecayScheduler(lr_max=0.1, lr_min=0.01, power=2, steps=10)
        cb_no_opt.set_model(type("NoOpt", (), {})())
        self.assertIsNone(cb_no_opt.get_lr())
        cb_no_opt.set_lr(0.05)

    def test_step_scheduler(self):
        """Test Step scheduler."""
        model = DummyModel()
        cb = lrs.StepDecayScheduler(lr_max=0.1, change_every=1, steps=5, steps_per_epoch=1)
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_train_batch_end(batch=0)
        self.assertTrue(lrs._get_optimizer_lr(model.optimizer) <= 0.1)
        # boundaries/values override
        cb = lrs.StepDecayScheduler(boundaries=[1, 2], values=[0.1, 0.05])
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_train_batch_end(batch=0)
        self.assertTrue(lrs._get_optimizer_lr(model.optimizer) <= 0.1)
        # start inferred from values branch
        ann = lrs.StepDecayAnnealer(start=None, change_every=None, steps=None, steps_per_epoch=1, boundaries=[1], values=[0.2])
        self.assertEqual(ann.start, 0.2)
        # missing optimizer branch
        cb_no_opt = lrs.StepDecayScheduler(boundaries=[1], values=[0.1])
        cb_no_opt.set_model(type("NoOpt", (), {})())
        cb_no_opt.on_train_begin()
        cb_no_opt.on_train_batch_begin(batch=0)
        cb_no_opt.on_train_batch_end(batch=0)
        self.assertIsNone(cb_no_opt.get_lr())
        cb_no_opt.set_lr(0.05)
        # step decay init paths
        ann2 = lrs.StepDecayAnnealer(start=None, change_every=None, steps=None, steps_per_epoch=2, boundaries=[1, 2], values=[0.3, 0.1])
        self.assertEqual(ann2.start, 0.3)
        ann2.step()
        ann2.step()
        ann2.step()
        ann3 = lrs.StepDecayAnnealer(start=0.5, change_every=None, steps=None, steps_per_epoch=1, boundaries=[1], values=[0.4])
        self.assertEqual(ann3.start, 0.5)

    def test_adapt_loss_schedulers(self):
        """Test Adapt loss schedulers."""
        model = DummyModel()
        cb1 = lrs.LRAdaptLossSlopeEarlyStop(monitor="val_loss", factor=0.5, patience=2, cooldown=1, min_lr=1e-6, verbose=0, threshold=-1e-3)
        cb1.set_model(model)
        cb1.on_train_begin()
        cb1.cooldown_counter = 1
        cb1.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
        cb1.on_epoch_end(epoch=0, logs={"other": 1.0})
        cb1.on_epoch_end(epoch=1, logs={"val_loss": 1.1})
        # slope decreasing branch (no lr change)
        cb1.on_epoch_end(epoch=2, logs={"val_loss": 0.9})
        # min_lr stop_training branch
        low_lr_model = DummyModel()
        low_lr_model.optimizer.learning_rate.assign(1e-6)
        cb_stop = lrs.LRAdaptLossSlopeEarlyStop(monitor="val_loss", factor=0.5, patience=2, cooldown=0, min_lr=1e-6, verbose=0, threshold=-1e-6)
        cb_stop.set_model(low_lr_model)
        cb_stop.on_train_begin()
        cb_stop.on_epoch_end(epoch=0, logs={"val_loss": 2.0})
        cb_stop.on_epoch_end(epoch=1, logs={"val_loss": 3.0})
        self.assertTrue(low_lr_model.stop_training)
        # factor >= 1 error
        with self.assertRaises(ValueError):
            lrs.LRAdaptLossSlopeEarlyStop(factor=1.1)

        cb2 = lrs.LRSeesawAdaptLossSlopeEarlyStop(monitor="val_loss", reduction_factor=0.5, increase_factor=0.001,
                                                  patience=2, cooldown=1, min_lr=1e-6, verbose=0)
        cb2.set_model(model)
        cb2.on_train_begin()
        cb2.cooldown_counter = 1
        cb2.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
        cb2.on_epoch_end(epoch=1, logs={"val_loss": 1.1})
        cb2.on_epoch_end(epoch=2, logs={"val_loss": 1.2})
        # missing metric branch
        cb2.on_epoch_end(epoch=3, logs={"other": 1.0})
        # reduction_factor validation
        with self.assertRaises(ValueError):
            lrs.LRSeesawAdaptLossSlopeEarlyStop(reduction_factor=1.1)
        # stop_training when at min_lr
        min_model = DummyModel()
        min_model.optimizer.learning_rate.assign(1e-6)
        cb_stop2 = lrs.LRSeesawAdaptLossSlopeEarlyStop(monitor="val_loss", reduction_factor=0.5, increase_factor=0.0,
                                                       patience=2, cooldown=0, min_lr=1e-6, verbose=0)
        cb_stop2.set_model(min_model)
        cb_stop2.on_train_begin()
        cb_stop2.on_epoch_end(epoch=0, logs={"val_loss": 2.0})
        cb_stop2.on_epoch_end(epoch=1, logs={"val_loss": 3.0})
        self.assertTrue(min_model.stop_training)

    def test_adapt_loss_reduction_verbose_and_no_reduction_branch(self):
        """Test Adapt loss reduction verbose and no reduction branch."""
        model = DummyModel()
        # explicit reduction path with verbose
        cb = lrs.LRAdaptLossSlopeEarlyStop(monitor="val_loss", factor=0.5, patience=2, cooldown=0, min_lr=1e-6, verbose=1, threshold=-1e-6)
        cb.set_model(model)
        cb.on_train_begin()
        cb.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
        cb.on_epoch_end(epoch=1, logs={"val_loss": 2.0})
        cb.on_epoch_end(epoch=2, logs={"val_loss": 2.5})
        # branch where slope does not trigger reduction
        cb_no_reduce = lrs.LRAdaptLossSlopeEarlyStop(monitor="val_loss", factor=0.5, patience=2, cooldown=0, min_lr=1e-6, verbose=0, threshold=1.0)
        cb_no_reduce.set_model(model)
        cb_no_reduce.on_train_begin()
        cb_no_reduce.on_epoch_end(epoch=0, logs={"val_loss": 2.0})
        cb_no_reduce.on_epoch_end(epoch=1, logs={"val_loss": 1.5})
        cb_no_reduce.on_epoch_end(epoch=2, logs={"val_loss": 1.4})

        silent_model = DummyModel()
        cb_silent_reduce = lrs.LRAdaptLossSlopeEarlyStop(monitor="val_loss", factor=0.5, patience=2, cooldown=0, min_lr=1e-6, verbose=0, threshold=-1.0)
        cb_silent_reduce.set_model(silent_model)
        cb_silent_reduce.on_train_begin()
        cb_silent_reduce.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
        cb_silent_reduce.on_epoch_end(epoch=1, logs={"val_loss": 2.0})
        cb_silent_reduce.on_epoch_end(epoch=2, logs={"val_loss": 3.0})

        # Seesaw: cover no reduction due to low slope and verbose print path
        model2 = DummyModel()
        cb_seesaw = lrs.LRSeesawAdaptLossSlopeEarlyStop(monitor="val_loss", reduction_factor=0.5, increase_factor=0.0,
                                                        patience=2, cooldown=0, min_lr=1e-6, verbose=1, threshold=1.0)
        cb_seesaw.set_model(model2)
        cb_seesaw.on_train_begin()
        cb_seesaw.on_epoch_end(epoch=0, logs={"val_loss": 2.0})
        cb_seesaw.on_epoch_end(epoch=1, logs={"val_loss": 1.5})
        cb_seesaw.on_epoch_end(epoch=2, logs={"val_loss": 1.4})
        # reduction with verbose
        cb_seesaw_reduce = lrs.LRSeesawAdaptLossSlopeEarlyStop(monitor="val_loss", reduction_factor=0.5, increase_factor=0.0,
                                                               patience=2, cooldown=0, min_lr=1e-6, verbose=1, threshold=-1.0)
        cb_seesaw_reduce.set_model(model2)
        cb_seesaw_reduce.on_train_begin()
        cb_seesaw_reduce.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
        cb_seesaw_reduce.on_epoch_end(epoch=1, logs={"val_loss": 2.0})
        cb_seesaw_reduce.on_epoch_end(epoch=2, logs={"val_loss": 2.5})
        # additional verbose reduction coverage
        verbose_model = DummyModel()
        verbose_cb = lrs.LRAdaptLossSlopeEarlyStop(monitor="val_loss", factor=0.5, patience=2, cooldown=0, min_lr=1e-6, verbose=2, threshold=-1.0)
        verbose_cb.set_model(verbose_model)
        verbose_cb.on_train_begin()
        with patch("tensorflow.print", lambda *args, **kwargs: None):
            verbose_cb.on_epoch_end(epoch=0, logs={"val_loss": 1.0})
            verbose_cb.on_epoch_end(epoch=1, logs={"val_loss": 2.0})
            verbose_cb.on_epoch_end(epoch=2, logs={"val_loss": 3.0})
        self.assertLess(lrs._get_optimizer_lr(verbose_model.optimizer), 0.01)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
