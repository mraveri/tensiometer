"""
Tests of the flow snapshots (save / load, full and light modes) and of the cache helpers.
"""

#########################################################################################################
# Imports

import contextlib
import copy
import io
import os
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import matplotlib
matplotlib.use('agg')

import numpy as np
import scipy.stats
import torch
from getdist import MCSamples

import tensiometer.synthetic_probability.synthetic_probability as sp
import tensiometer.synthetic_probability.flow_profiler as fp
from tensiometer.synthetic_probability import bijectors as bj
from tensiometer.synthetic_probability import distributions as ds
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Helpers

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))


def _chain(num_samples=2000, seed=0, shift=0.):
    """
    Correlated 3D Gaussian chain with log posterior values.

    :param num_samples: number of samples.
    :param seed: random seed.
    :param shift: shift of the mean, to build a different chain.
    :returns: :class:`getdist.MCSamples`.
    """
    rng = np.random.default_rng(seed)
    mean = np.array([0.5, -0.5, 1.]) + shift
    cov = np.array([[1., 0.3, 0.1], [0.3, 0.5, 0.2], [0.1, 0.2, 2.]])
    samples = rng.multivariate_normal(mean, cov, size=num_samples)
    loglikes = -scipy.stats.multivariate_normal(mean, cov).logpdf(samples)
    names = ['a', 'b', 'c']
    return MCSamples(samples=samples, loglikes=loglikes, names=names, labels=names,
                     ranges={'a': [-10., 10.], 'b': [-10., 10.], 'c': [-10., 10.]})


_FLOW_KWARGS = {'feedback': 0, 'plot_every': 0, 'hidden_units': [8, 8], 'n_transformations': 2}


def _trained_flow(chain=None, **kwargs):
    """
    Small flow trained for a few epochs.

    :param chain: chain, defaults to :func:`_chain`.
    :returns: flow.
    """
    if chain is None:
        chain = _chain()
    _kwargs = dict(_FLOW_KWARGS)
    _kwargs.update(kwargs)
    flow = sp.FlowCallback(chain, **_kwargs)
    flow.train(epochs=3, verbose=0)
    return flow


def _points(flow, num=10):
    """Test points from the chain range."""
    rng = np.random.default_rng(3)
    return rng.normal(size=(num, flow.num_params)).astype(tu.np_prec)


def _run_python(code):
    """
    Run code in a fresh interpreter (clean precision state).

    :param code: python source.
    :returns: completed process.
    """
    env = dict(os.environ)
    env.pop('TENSIOMETER_PRECISION', None)
    env.pop('TENSIOMETER_DEVICE', None)
    return subprocess.run([sys.executable, '-c', code], cwd=_REPO, env=env, capture_output=True, text=True)


def _module_level_forward(x):
    """Picklable Inline forward function."""
    return 2. * x


def _module_level_inverse(y):
    """Picklable Inline inverse function."""
    return 0.5 * y


def _other_precision():
    """Name of the precision that is not active."""
    if tu.get_precision() == torch.float32:
        return 'float64'
    return 'float32'


class FakeCommunicator:
    """Stand-in for ``MPI.COMM_WORLD`` as seen by one rank."""

    def __init__(self, rank, size, on_first_barrier=None):
        """
        :param rank: value returned by ``Get_rank``.
        :param size: value returned by ``Get_size``.
        :param on_first_barrier: optional function called at the first ``Barrier``, to run
            the other ranks while this one waits.
        """
        self.rank = rank
        self.size = size
        self.on_first_barrier = on_first_barrier
        self.broadcast_value = None
        self.broadcasts = []
        self.barrier_calls = 0

    def Get_rank(self):
        """Rank of this process."""
        return self.rank

    def Get_size(self):
        """Number of processes."""
        return self.size

    def bcast(self, value, root=0):
        """Record the value; non-root ranks receive ``broadcast_value``."""
        self.broadcasts.append(value)
        if self.broadcast_value is not None:
            return self.broadcast_value
        return value

    def Barrier(self):
        """Count the call and run the other ranks at the first one."""
        self.barrier_calls += 1
        if self.on_first_barrier is not None:
            callback = self.on_first_barrier
            self.on_first_barrier = None
            callback()


@contextlib.contextmanager
def _fake_mpi(communicator):
    """
    Replace ``mpi4py`` with a fake package whose ``MPI.COMM_WORLD`` is ``communicator``.

    :param communicator: :class:`FakeCommunicator`.
    :returns: context manager.
    """
    mpi = types.ModuleType('mpi4py.MPI')
    mpi.COMM_WORLD = communicator
    package = types.ModuleType('mpi4py')
    package.MPI = mpi
    saved = {name: sys.modules.get(name, None) for name in ('mpi4py', 'mpi4py.MPI')}
    sys.modules['mpi4py'] = package
    sys.modules['mpi4py.MPI'] = mpi
    try:
        yield mpi
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

#########################################################################################################
# Full snapshots


class TestFullSnapshots(unittest.TestCase):
    """Round trips of full snapshots."""

    @classmethod
    def setUpClass(cls):
        """Train one flow for all the tests."""
        np.random.seed(0)
        torch.manual_seed(0)
        cls.flow = _trained_flow()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, 'flow.pt')
        cls.flow.save(cls.path)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary directory."""
        cls.tmp.cleanup()

    def test_load_without_chain(self):
        """Loading restores the same flow without rebuilding or retraining."""
        with mock.patch.object(sp.FlowCallback, 'global_train', side_effect=AssertionError('no training')), \
                mock.patch.object(sp.FlowCallback, '__init__', side_effect=AssertionError('no rebuild')):
            flow = sp.FlowCallback.load(self.path)
        x = _points(flow)
        self.assertTrue(torch.equal(flow.log_probability(x), self.flow.log_probability(x)))
        self.assertTrue(torch.allclose(flow.metric(x), self.flow.metric(x)))
        torch.manual_seed(5)
        samples_1 = flow.sample(10)
        torch.manual_seed(5)
        samples_2 = self.flow.sample(10)
        self.assertTrue(torch.equal(samples_1, samples_2))
        self.assertEqual(flow.log.keys(), self.flow.log.keys())
        self.assertEqual(flow.log['loss'], self.flow.log['loss'])
        self.assertTrue(np.array_equal(flow.training_idx, self.flow.training_idx))
        self.assertTrue(np.array_equal(flow.test_idx, self.flow.test_idx))
        self.assertEqual(flow._snapshot_info['mode'], 'full')
        self.assertEqual(flow._snapshot_info['format_version'], sp.SNAPSHOT_FORMAT_VERSION)

    def test_resume_training(self):
        """Training resumes with the stored optimizer state and the same split."""
        flow = sp.FlowCallback.load(self.path)
        self.assertTrue(flow._trainer_initialized)
        self.assertGreater(len(flow.trainer.optimizer.state), 0)
        # the optimizer acts on the parameters of the loaded bijector:
        loaded_parameters = set(id(_p) for _p in flow.trainable_bijector.parameters())
        optimizer_parameters = set(id(_p) for _g in flow.trainer.optimizer.param_groups for _p in _g['params'])
        self.assertEqual(loaded_parameters, optimizer_parameters)
        before = copy.deepcopy(flow.trainable_bijector.state_dict())
        flow.train(epochs=1, verbose=0)
        after = flow.trainable_bijector.state_dict()
        self.assertTrue(any(not torch.equal(before[_k], after[_k]) for _k in before))
        self.assertEqual(len(np.intersect1d(flow.training_idx, flow.test_idx)), 0)
        self.assertEqual(flow.training_dataset[0].shape[0], len(flow.training_idx))

    def test_copy(self):
        """Copies of a flow are independent and equivalent."""
        flow = copy.deepcopy(self.flow)
        x = _points(flow)
        self.assertTrue(torch.equal(flow.log_probability(x), self.flow.log_probability(x)))
        self.assertIsNot(flow.trainable_bijector, self.flow.trainable_bijector)

    def test_average_flow_load_rejects_single_flow(self):
        """Loading with a subclass checks the type of the snapshot."""
        with self.assertRaises(TypeError):
            sp.average_flow.load(self.path)

    def test_missing_file(self):
        """A missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            sp.FlowCallback.load(os.path.join(self.tmp.name, 'missing.pt'))

    def test_foreign_files(self):
        """TensorFlow-era caches and foreign files are rejected."""
        tf_path = os.path.join(self.tmp.name, 'sprob_flow_cache.pickle')
        with open(tf_path, 'wb') as handle:
            pickle.dump({'num_params': 3, 'log': {}}, handle)
        with self.assertRaises(ValueError):
            sp.FlowCallback.load(tf_path)
        foreign_path = os.path.join(self.tmp.name, 'foreign.pt')
        torch.save({'a': 1}, foreign_path)
        with self.assertRaises(ValueError):
            sp.FlowCallback.load(foreign_path)

    def test_newer_format_rejected(self):
        """Snapshots with a newer format version are rejected."""
        flow = copy.deepcopy(self.flow)
        path = os.path.join(self.tmp.name, 'newer.pt')
        flow.save(path)
        with mock.patch.object(sp, 'SNAPSHOT_FORMAT_VERSION', 0):
            with self.assertRaises(ValueError):
                sp.FlowCallback.load(path)

    def test_other_version_warns(self):
        """Snapshots written by another tensiometer version load with a warning."""
        flow = copy.deepcopy(self.flow)
        path = os.path.join(self.tmp.name, 'old_version.pt')
        with mock.patch.object(sp, '_tensiometer_version', '0.0.1'):
            flow.save(path)
        with self.assertWarns(UserWarning) as context:
            loaded = sp.FlowCallback.load(path)
        self.assertIn('0.0.1', str(context.warning))
        x = _points(flow)
        self.assertTrue(torch.equal(loaded.log_probability(x), flow.log_probability(x)))

    def _save_with_other_precision(self, name):
        """Save a copy of the flow whose snapshot declares the other precision."""
        path = os.path.join(self.tmp.name, name)
        with mock.patch.object(sp, '_precision_name', return_value=_other_precision()):
            copy.deepcopy(self.flow).save(path)
        return path

    def test_other_precision_rejected_when_locked(self):
        """A snapshot with another precision is rejected once the precision is locked."""
        path = self._save_with_other_precision('other_precision.pt')
        self.assertTrue(tu.is_precision_locked())
        with self.assertRaises(ValueError) as context:
            sp.FlowCallback.load(path)
        self.assertIn('locked', str(context.exception))
        self.assertIn(_other_precision(), str(context.exception))

    def test_other_precision_adopted_when_unlocked(self):
        """A snapshot with another precision sets the process precision when it is not locked yet."""
        path = self._save_with_other_precision('adopted_precision.pt')
        with mock.patch.object(tu, 'is_precision_locked', return_value=False), \
                mock.patch.object(tu, 'set_precision') as set_precision:
            loaded = sp.FlowCallback.load(path, device='cpu')
        set_precision.assert_called_once_with(tu._parse_precision(_other_precision()))
        self.assertIsInstance(loaded, sp.FlowCallback)

    def test_state_device_fallbacks(self):
        """The device of a state comes from the bijector, the base distribution or the stored device."""
        cpu = torch.device('cpu')
        self.assertEqual(sp.FlowCallback._state_device({'bijector': self.flow.bijector}), self.flow.device)
        state = {'bijector': bj.Identity(), 'base_distribution': ds.standard_normal(2)}
        self.assertEqual(sp.FlowCallback._state_device(state), cpu)
        self.assertEqual(sp.FlowCallback._state_device({'base_distribution': object(), 'device': 'stored'}), 'stored')
        self.assertEqual(sp.FlowCallback._state_device({}), tu.get_device())

    def test_atomic_save(self):
        """A failure while writing leaves no file behind."""
        path = os.path.join(self.tmp.name, 'failed.pt')
        with mock.patch.object(tu.torch, 'save', side_effect=RuntimeError('disk full')):
            with self.assertRaises(RuntimeError):
                self.flow.save(path)
        self.assertFalse(os.path.exists(path))
        self.assertEqual([_f for _f in os.listdir(self.tmp.name) if _f.endswith('.tmp')], [])

    def test_invalid_mode(self):
        """Unknown save modes raise."""
        with self.assertRaises(ValueError):
            self.flow.save(os.path.join(self.tmp.name, 'bad.pt'), mode='medium')

#########################################################################################################
# Light snapshots


class TestLightSnapshots(unittest.TestCase):
    """Light snapshots keep the flow and drop the chain."""

    @classmethod
    def setUpClass(cls):
        """Train one flow on a larger chain and save it in both modes."""
        np.random.seed(1)
        torch.manual_seed(1)
        cls.flow = _trained_flow(_chain(num_samples=20000))
        cls.tmp = tempfile.TemporaryDirectory()
        cls.full_path = os.path.join(cls.tmp.name, 'full.pt')
        cls.light_path = os.path.join(cls.tmp.name, 'light.pt')
        cls.flow.save(cls.full_path)
        cls.flow.save(cls.light_path, mode='light')
        cls.light = sp.FlowCallback.load(cls.light_path)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary directory."""
        cls.tmp.cleanup()

    def test_same_outputs(self):
        """A light flow evaluates like the full one."""
        x = _points(self.flow)
        self.assertTrue(self.light.is_light)
        self.assertFalse(self.flow.is_light)
        self.assertTrue(torch.equal(self.light.log_probability(x), self.flow.log_probability(x)))
        self.assertTrue(torch.allclose(self.light.metric(x), self.flow.metric(x)))
        torch.manual_seed(2)
        samples_1 = self.light.sample(10)
        torch.manual_seed(2)
        samples_2 = self.flow.sample(10)
        self.assertTrue(torch.equal(samples_1, samples_2))
        self.assertEqual(self.light.param_names, self.flow.param_names)
        self.assertEqual(self.light.log['loss'], self.flow.log['loss'])
        self.assertEqual(self.light._snapshot_info['mode'], 'light')

    def test_file_size(self):
        """The light file is much smaller than the full one."""
        self.assertLess(os.path.getsize(self.light_path), 0.1 * os.path.getsize(self.full_path))

    def test_no_training(self):
        """Light flows cannot be trained."""
        with self.assertRaises(RuntimeError):
            self.light.train(epochs=1)
        with self.assertRaises(RuntimeError):
            self.light.global_train(pop_size=1, epochs=1)

    def test_missing_attributes(self):
        """Dropped attributes raise an explanatory AttributeError."""
        with self.assertRaises(AttributeError) as context:
            self.light.chain_samples
        self.assertIn('light flow snapshot', str(context.exception))
        self.assertFalse(hasattr(self.light, 'chain_samples'))
        self.assertFalse(hasattr(self.light, 'loss'))
        with self.assertRaises(AttributeError):
            self.light.evidence()
        with self.assertRaises(AttributeError):
            self.light.not_an_attribute

    def test_copy_and_pickle(self):
        """Copies and pickles of light flows work."""
        x = _points(self.flow)
        for flow in [copy.copy(self.light), copy.deepcopy(self.light), pickle.loads(pickle.dumps(self.light))]:
            self.assertTrue(flow.is_light)
            self.assertTrue(torch.equal(flow.log_probability(x), self.flow.log_probability(x)))
        path = os.path.join(self.tmp.name, 'light_again.pt')
        self.light.save(path, mode='light')
        self.assertTrue(sp.FlowCallback.load(path).is_light)

    def test_full_save_of_light_flow(self):
        """A light flow cannot be saved in full mode."""
        with self.assertRaises(ValueError):
            self.light.save(os.path.join(self.tmp.name, 'bad.pt'), mode='full')

    def test_full_to_light_conversion(self):
        """A full snapshot converts to a light one without the chain."""
        path = os.path.join(self.tmp.name, 'converted.pt')
        sp.FlowCallback.load(self.full_path).save(path, mode='light')
        converted = sp.FlowCallback.load(path)
        self.assertTrue(converted.is_light)
        x = _points(self.flow)
        self.assertTrue(torch.equal(converted.log_probability(x), self.flow.log_probability(x)))

    def test_light_flow_in_other_tools(self):
        """Light flows work with transformed and average flows, the profiler and KL divergence."""
        from tensiometer.synthetic_probability import flow_utilities as fu
        x = _points(self.flow)
        # transformed flow:
        transformed = sp.TransformedFlowCallback(self.light, [bj.Identity(), bj.Shift(1.), bj.Scale(2.)])
        self.assertTrue(transformed.is_light)
        self.assertEqual(tuple(transformed.log_probability(x).shape), (len(x),))
        path = os.path.join(self.tmp.name, 'transformed.pt')
        transformed.save(path, mode='light')
        self.assertTrue(torch.allclose(sp.TransformedFlowCallback.load(path).log_probability(x),
                                       transformed.log_probability(x)))
        # average flow of light members:
        average = sp.average_flow([self.light, copy.deepcopy(self.light)])
        self.assertTrue(average.is_light)
        self.assertTrue(torch.allclose(average.log_probability(x), self.flow.log_probability(x), atol=1e-5))
        path = os.path.join(self.tmp.name, 'average.pt')
        average.save(path, mode='light')
        loaded = sp.average_flow.load(path)
        self.assertEqual(loaded.num_flows, 2)
        self.assertTrue(all(_f.is_light for _f in loaded.flows))
        # profiler (margestats from flow samples):
        profiler = fp.posterior_profile_plotter(self.light, feedback=0, num_minimization_samples=500)
        profiler.update_cache(update_2D=False, num_points_1D=8, polish=False, pre_polish=False)
        self.assertEqual(len(profiler.profile_density_1D), 3)
        # KL divergence:
        mean, _ = fu.KL_divergence(self.light, self.light, num_samples=100, num_batches=2)
        self.assertAlmostEqual(mean, 0., places=4)

    def test_light_flow_analytical_transformation(self):
        """Analytical transformations of light flows have no chain and no ranges."""
        analytical = sp.AnalyticalDerivedParamsBijector(['a'], ['exp_a'], ['e^a'],
                                                        forward_fn=torch.exp, inverse_fn=torch.log)
        transformed = sp.TransformedFlowCallback(self.light, analytical)
        self.assertTrue(transformed.is_light)
        self.assertIsNone(transformed.parameter_ranges)
        self.assertEqual(transformed.param_names, ['exp_a', 'b', 'c'])
        self.assertNotIn('chain_samples', transformed.__dict__)
        x = torch.as_tensor(_points(self.flow))
        y = torch.cat([torch.exp(x[:, :1]), x[:, 1:]], dim=1)
        expected = self.flow.log_probability(x) - x[:, 0]
        self.assertTrue(torch.allclose(transformed.log_probability(y), expected, atol=1e-4))

#########################################################################################################
# Other flows


class TestOtherSnapshots(unittest.TestCase):
    """Round trips of the flow variants."""

    def setUp(self):
        """Temporary directory."""
        self.tmp = tempfile.TemporaryDirectory()
        np.random.seed(2)
        torch.manual_seed(2)

    def tearDown(self):
        """Remove the temporary directory."""
        self.tmp.cleanup()

    def _round_trip(self, flow, cls=sp.FlowCallback):
        """Save, load and compare log probabilities."""
        path = os.path.join(self.tmp.name, 'flow.pt')
        flow.save(path)
        loaded = cls.load(path)
        x = _points(flow)
        self.assertTrue(torch.allclose(loaded.log_probability(x), flow.log_probability(x)))
        return loaded

    def test_no_trainable_bijector(self):
        """Flows without trainable transformation save and load."""
        flow = sp.FlowCallback(_chain(500), trainable_bijector=None, prior_bijector=None, feedback=0)
        loaded = self._round_trip(flow)
        self.assertIsNone(loaded.trainable_transformation)

    def test_parameter_free_flow(self):
        """A flow whose bijectors hold no tensors takes its device from the base distribution."""
        flow = sp.FlowCallback(_chain(500), prior_bijector=None, apply_rescaling=False, trainable_bijector=None,
                               feedback=0)
        self.assertIsNone(tu.module_device(flow.bijector))
        loaded = self._round_trip(flow)
        self.assertEqual(loaded.device, flow.base_distribution.mean.device)
        self.assertEqual(loaded.training_dataset[0].device, loaded.device)

    def test_custom_trainable_bijector(self):
        """Flows with a raw trainable bijector save and load."""
        flow = sp.FlowCallback(_chain(500), trainable_bijector=bj.Shift(0.5), feedback=0)
        self._round_trip(flow)

    def test_transformed_flow(self):
        """Transformed flows save and load."""
        flow = _trained_flow(_chain(500))
        transformed = sp.TransformedFlowCallback(flow, [bj.Identity(), bj.Scale(2.), bj.Shift(-1.)])
        loaded = self._round_trip(transformed, cls=sp.TransformedFlowCallback)
        self.assertEqual(loaded.param_names, transformed.param_names)

    def test_average_flow(self):
        """Average flows save and load with their members."""
        chain = _chain(500)
        split = (np.arange(50), np.arange(50, 500))
        flows = [_trained_flow(chain, validation_training_idx=split) for _ in range(2)]
        average = sp.average_flow(flows, validation_training_idx=split)
        loaded = self._round_trip(average, cls=sp.average_flow)
        self.assertEqual(loaded.num_flows, 2)
        self.assertTrue(torch.allclose(loaded.weights.cpu(), average.weights.cpu()))

    def test_profiler(self):
        """Profilers save and load together with their flow."""
        flow = _trained_flow(_chain(500))
        profiler = fp.posterior_profile_plotter(flow, feedback=0, num_minimization_samples=500)
        profiler.update_cache(update_2D=False, num_points_1D=8, polish=False, pre_polish=False)
        path = os.path.join(self.tmp.name, 'profiler.pt')
        profiler.savePickle(path)
        loaded = fp.posterior_profile_plotter.loadPickle(path)
        self.assertIsInstance(loaded.flow, sp.FlowCallback)
        self.assertTrue(np.allclose(loaded.flow_MAP, profiler.flow_MAP))
        x = _points(flow)
        self.assertTrue(torch.allclose(loaded.flow.log_probability(x), flow.log_probability(x)))
        self.assertEqual(sorted(loaded.profile_density_1D.keys()), sorted(profiler.profile_density_1D.keys()))

    def test_picklable_inline(self):
        """Inline bijectors with module level functions save; lambdas raise an explanatory error."""
        flow = sp.FlowCallback(_chain(500), trainable_bijector=bj.Inline(
            _module_level_forward, _module_level_inverse, forward_min_event_ndims=0), feedback=0)
        self._round_trip(flow)
        flow = sp.FlowCallback(_chain(500), trainable_bijector=bj.Inline(
            lambda x: 2. * x, lambda y: 0.5 * y, forward_min_event_ndims=0), feedback=0)
        with self.assertRaises(pickle.PicklingError) as context:
            flow.save(os.path.join(self.tmp.name, 'lambda.pt'))
        self.assertIn('module level functions', str(context.exception))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, 'lambda.pt')))

#########################################################################################################
# Cache helpers


class TestFlowFromChainCache(unittest.TestCase):
    """Cache checks of flow_from_chain."""

    def setUp(self):
        """Temporary directory and chain."""
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_file = os.path.join(self.tmp.name, 'cache.pt')
        self.chain = _chain(500)
        self.kwargs = dict(_FLOW_KWARGS, epochs=2, pop_size=1, verbose=0)
        np.random.seed(3)
        torch.manual_seed(3)

    def tearDown(self):
        """Remove the temporary directory."""
        self.tmp.cleanup()

    def test_cache_hit(self):
        """The second call loads the cached flow without training."""
        flow = sp.flow_from_chain(self.chain, cache_file=self.cache_file, **self.kwargs)
        self.assertTrue(os.path.isfile(self.cache_file))
        with mock.patch.object(sp.FlowCallback, 'global_train', side_effect=AssertionError('no training')):
            cached = sp.flow_from_chain(self.chain, cache_file=self.cache_file, **self.kwargs)
            # runtime-only arguments do not invalidate the cache:
            _kwargs = dict(self.kwargs, feedback=1)
            sp.flow_from_chain(self.chain, cache_file=self.cache_file, **_kwargs)
        x = _points(flow)
        self.assertTrue(torch.equal(cached.log_probability(x), flow.log_probability(x)))

    def test_cache_mismatch(self):
        """Different chains or settings raise with the differing keys."""
        sp.flow_from_chain(self.chain, cache_file=self.cache_file, **self.kwargs)
        with self.assertRaises(ValueError) as context:
            sp.flow_from_chain(_chain(500, shift=1.), cache_file=self.cache_file, **self.kwargs)
        self.assertIn('chain', str(context.exception))
        for key, value in [('epochs', 3), ('activation', torch.tanh), ('range_max', 7.), ('hidden_units', [4, 4])]:
            _kwargs = dict(self.kwargs)
            _kwargs[key] = value
            with self.assertRaises(ValueError) as context:
                sp.flow_from_chain(self.chain, cache_file=self.cache_file, **_kwargs)
            self.assertIn(key, str(context.exception))

    def test_overwrite_cache(self):
        """overwrite_cache retrains and replaces the file."""
        sp.flow_from_chain(self.chain, cache_file=self.cache_file, **self.kwargs)
        _kwargs = dict(self.kwargs, epochs=3)
        flow = sp.flow_from_chain(self.chain, cache_file=self.cache_file, overwrite_cache=True, **_kwargs)
        self.assertEqual(len(flow.log['loss']), 3)
        cached = sp.flow_from_chain(self.chain, cache_file=self.cache_file, **_kwargs)
        self.assertEqual(len(cached.log['loss']), 3)

    def test_light_cache(self):
        """cache_mode='light' returns light flows on later calls, and the chain check still applies."""
        sp.flow_from_chain(self.chain, cache_file=self.cache_file, cache_mode='light', **self.kwargs)
        cached = sp.flow_from_chain(self.chain, cache_file=self.cache_file, cache_mode='light', **self.kwargs)
        self.assertTrue(cached.is_light)
        with self.assertRaises(ValueError):
            sp.flow_from_chain(_chain(500, shift=1.), cache_file=self.cache_file, **self.kwargs)

    def test_removed_arguments(self):
        """cache_dir and root_name were replaced by cache_file."""
        with self.assertRaises(ValueError):
            sp.flow_from_chain(self.chain, cache_dir=self.tmp.name, **self.kwargs)
        with self.assertRaises(ValueError):
            sp.average_flow_from_chain(self.chain, root_name='sprob', **self.kwargs)


class TestAverageFlowFromChainCache(unittest.TestCase):
    """Cache files of average_flow_from_chain."""

    def setUp(self):
        """Temporary directory and chain."""
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_file = os.path.join(self.tmp.name, 'average.pt')
        self.chain = _chain(500)
        self.kwargs = dict(_FLOW_KWARGS, epochs=2, pop_size=1, verbose=0)
        np.random.seed(4)
        torch.manual_seed(4)

    def tearDown(self):
        """Remove the temporary directory."""
        self.tmp.cleanup()

    def test_resume_from_parts(self):
        """An interrupted run resumes from the part files, which are removed after assembling."""
        original_global_train = sp.FlowCallback.global_train
        calls = []

        def failing_second_member(flow, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt('interrupted')
            return original_global_train(flow, **kwargs)

        with mock.patch.object(sp.FlowCallback, 'global_train', failing_second_member):
            with self.assertRaises(KeyboardInterrupt):
                sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=self.cache_file, **self.kwargs)
        self.assertTrue(os.path.isfile(self.cache_file + '.split'))
        self.assertTrue(os.path.isfile(self.cache_file + '.part0'))
        self.assertFalse(os.path.isfile(self.cache_file + '.part1'))
        calls.clear()

        def counting(flow, **kwargs):
            calls.append(1)
            return original_global_train(flow, **kwargs)

        with mock.patch.object(sp.FlowCallback, 'global_train', counting):
            average = sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=self.cache_file, **self.kwargs)
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(average, sp.average_flow)
        self.assertTrue(np.array_equal(average.flows[0].test_idx, average.flows[1].test_idx))
        self.assertTrue(os.path.isfile(self.cache_file))
        self.assertEqual([_f for _f in os.listdir(self.tmp.name) if _f != 'average.pt'], [])
        # cache hit:
        with mock.patch.object(sp.FlowCallback, 'global_train', side_effect=AssertionError('no training')):
            cached = sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=self.cache_file, **self.kwargs)
        x = _points(average)
        self.assertTrue(torch.allclose(cached.log_probability(x), average.log_probability(x)))
        # different number of flows:
        with self.assertRaises(ValueError):
            sp.average_flow_from_chain(self.chain, num_flows=3, cache_file=self.cache_file, **self.kwargs)

    def test_single_flow(self):
        """With one flow the helper returns the flow itself."""
        flow = sp.average_flow_from_chain(self.chain, num_flows=1, cache_file=self.cache_file, **self.kwargs)
        self.assertNotIsInstance(flow, sp.average_flow)
        self.assertTrue(os.path.isfile(self.cache_file))

    def test_feedback_reports_cache_loads(self):
        """With feedback the members and the assembled flow loaded from cache are reported."""
        original_global_train = sp.FlowCallback.global_train
        calls = []

        def failing_second_member(flow, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt('interrupted')
            return original_global_train(flow, **kwargs)

        _kwargs = dict(self.kwargs, feedback=1)
        with mock.patch.object(sp.FlowCallback, 'global_train', failing_second_member):
            with self.assertRaises(KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
                sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=self.cache_file, **_kwargs)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=self.cache_file, **_kwargs)
        self.assertIn('Loading flow 0 from cache', output.getvalue())
        self.assertIn('Training flow 1', output.getvalue())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cached = sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=self.cache_file, **_kwargs)
        self.assertIn('Loading average flow from cache', output.getvalue())
        self.assertIsInstance(cached, sp.average_flow)

#########################################################################################################
# MPI runs of average_flow_from_chain, with a fake communicator


class TestAverageFlowFromChainMPI(unittest.TestCase):
    """MPI distribution of the members of an average flow, simulated in one process."""

    def setUp(self):
        """Temporary directory and chain."""
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_file = os.path.join(self.tmp.name, 'average.pt')
        self.chain = _chain(500)
        self.kwargs = dict(_FLOW_KWARGS, epochs=2, pop_size=1, verbose=0)
        np.random.seed(5)
        torch.manual_seed(5)

    def tearDown(self):
        """Remove the temporary directory."""
        self.tmp.cleanup()

    def test_two_ranks(self):
        """Each of two ranks trains one member and both assemble the same average flow."""
        results = {}
        rank_1 = FakeCommunicator(1, 2)

        def run_rank_1():
            """Run rank 1 while rank 0 waits at its first barrier."""
            rank_1.broadcast_value = rank_0.broadcasts[0]
            with _fake_mpi(rank_1):
                results[1] = sp.average_flow_from_chain(
                    self.chain, num_flows=2, cache_file=self.cache_file, use_mpi=True, **self.kwargs)

        rank_0 = FakeCommunicator(0, 2, on_first_barrier=run_rank_1)
        output = io.StringIO()
        with _fake_mpi(rank_0), contextlib.redirect_stdout(output):
            results[0] = sp.average_flow_from_chain(
                self.chain, num_flows=2, cache_file=self.cache_file, use_mpi=True, **dict(self.kwargs, feedback=1))
        text = output.getvalue()
        self.assertIn('Training average flow with MPI enabled', text)
        self.assertIn('MPI size: 2', text)
        self.assertIn('Training flow 0 on MPI worker 0', text)
        self.assertNotIn('Training flow 1', text)
        self.assertEqual(rank_0.barrier_calls, 2)
        self.assertEqual(rank_1.barrier_calls, 2)
        self.assertIsNone(rank_1.broadcasts[0])
        for result in results.values():
            self.assertIsInstance(result, sp.average_flow)
            self.assertEqual(result.num_flows, 2)
            self.assertTrue(np.array_equal(result.flows[0].test_idx, result.flows[1].test_idx))
        x = _points(results[0])
        self.assertTrue(torch.allclose(results[0].log_probability(x), results[1].log_probability(x)))
        self.assertEqual(os.listdir(self.tmp.name), ['average.pt'])
        with mock.patch.object(sp.FlowCallback, 'global_train', side_effect=AssertionError('no training')):
            cached = sp.average_flow_from_chain(self.chain, num_flows=2, cache_file=self.cache_file, **self.kwargs)
        self.assertTrue(torch.allclose(cached.log_probability(x), results[0].log_probability(x)))

    def test_undefined_rank_and_size(self):
        """A communicator without rank and size runs everything as a single process."""
        communicator = FakeCommunicator(None, None)
        with _fake_mpi(communicator):
            average = sp.average_flow_from_chain(
                self.chain, num_flows=2, cache_file=self.cache_file, use_mpi=True, **self.kwargs)
        self.assertIsInstance(average, sp.average_flow)
        self.assertEqual(average.num_flows, 2)
        self.assertEqual(len(communicator.broadcasts), 1)
        self.assertEqual(communicator.barrier_calls, 0)
        self.assertEqual(os.listdir(self.tmp.name), ['average.pt'])

    def test_cuda_member_device_follows_rank(self):
        """On cuda each rank trains on device ``rank % device_count``; results move to the target device."""
        target = torch.device('cuda:1')
        communicator = FakeCommunicator(0, 1)
        with _fake_mpi(communicator), mock.patch.object(sp.tu, 'resolve_device', return_value=target), \
                mock.patch.object(sp.torch.cuda, 'device_count', return_value=2), \
                mock.patch.object(sp, 'FlowCallback') as flow_class:
            result = sp.average_flow_from_chain(
                self.chain, num_flows=1, cache_file=self.cache_file, use_mpi=True, device='cuda:1', **self.kwargs)
        member = flow_class.return_value
        self.assertEqual(flow_class.call_args.kwargs['device'], torch.device('cuda:0'))
        self.assertEqual(member.global_train.call_args.kwargs['device'], torch.device('cuda:0'))
        member.save.assert_called_once_with(self.cache_file + '.part0', mode='full')
        member.to.assert_called_once_with(target)
        self.assertIs(result, member.to.return_value)
        result.save.assert_called_once_with(self.cache_file, mode='full')
        self.assertFalse(os.path.isfile(self.cache_file + '.split'))

#########################################################################################################
# Precision of snapshots


class TestSnapshotPrecision(unittest.TestCase):
    """Precision adoption and mismatch when loading, in fresh interpreters."""

    def test_adopt_and_mismatch(self):
        """A float64 snapshot is adopted by a fresh process and rejected by a locked float32 one."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'flow64.pt')
            save_code = (
                "import numpy as np\n"
                "from getdist import MCSamples\n"
                "from tensiometer.synthetic_probability import tensor_utilities as tu\n"
                "tu.set_precision('float64')\n"
                "import tensiometer.synthetic_probability.synthetic_probability as sp\n"
                "s = np.random.default_rng(0).normal(size=(300, 2))\n"
                "chain = MCSamples(samples=s, names=['a', 'b'])\n"
                "flow = sp.FlowCallback(chain, feedback=0, hidden_units=[4], n_transformations=1)\n"
                "flow.save(%r)\n" % path)
            result = _run_python(save_code)
            self.assertEqual(result.returncode, 0, result.stderr)
            load_code = (
                "import torch\n"
                "import tensiometer.synthetic_probability.synthetic_probability as sp\n"
                "from tensiometer.synthetic_probability import tensor_utilities as tu\n"
                "flow = sp.FlowCallback.load(%r)\n"
                "assert tu.get_precision() == torch.float64\n"
                "assert flow.log_probability([[0., 0.]]).dtype == torch.float64\n" % path)
            result = _run_python(load_code)
            self.assertEqual(result.returncode, 0, result.stderr)
            mismatch_code = (
                "import tensiometer.synthetic_probability.synthetic_probability as sp\n"
                "from tensiometer.synthetic_probability import bijectors as bj\n"
                "bj.Identity()\n"
                "try:\n"
                "    sp.FlowCallback.load(%r)\n"
                "except ValueError:\n"
                "    print('mismatch detected')\n" % path)
            result = _run_python(mismatch_code)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('mismatch detected', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
