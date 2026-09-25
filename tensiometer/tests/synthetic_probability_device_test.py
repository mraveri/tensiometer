"""
Tests of the device handling (cpu, cuda, mps) of the synthetic probability package.

The accelerator tests run only where the device is available and, for MPS, in float32.
"""

#########################################################################################################
# Imports

import copy
import os
import tempfile
import unittest
import warnings
from unittest import mock

import matplotlib
matplotlib.use('agg')

import numpy as np
import torch
from getdist import MCSamples

import tensiometer.synthetic_probability.synthetic_probability as sp
from tensiometer.synthetic_probability import autodiff
from tensiometer.synthetic_probability import bijectors as bj
from tensiometer.synthetic_probability import tensor_utilities as tu
from tensiometer.synthetic_probability import trainable_bijectors as tb

#########################################################################################################
# Helpers


def _accelerators():
    """Available accelerator devices that support the active precision."""
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    if torch.backends.mps.is_available() and tu.get_precision() == torch.float32:
        devices.append(torch.device('mps'))
    return devices


def _chain(num_samples=1000, seed=0):
    """Small correlated Gaussian chain."""
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal([0., 1.], [[1., 0.4], [0.4, 0.5]], size=num_samples)
    return MCSamples(samples=samples, names=['a', 'b'], labels=['a', 'b'])


_FLOW_KWARGS = {'feedback': 0, 'plot_every': 0, 'hidden_units': [8, 8], 'n_transformations': 2}

#########################################################################################################
# Device selection


class TestDeviceSelection(unittest.TestCase):
    """Parsing and validation of device names."""

    def test_cpu(self):
        """cpu is always available."""
        self.assertEqual(tu.resolve_device('cpu'), torch.device('cpu'))
        self.assertEqual(tu.resolve_device(torch.device('cpu')), torch.device('cpu'))
        self.assertEqual(tu.resolve_device(None), tu.get_device())

    def test_auto(self):
        """auto picks cuda, then mps, then cpu."""
        with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                mock.patch.object(torch.backends.mps, 'is_available', return_value=False):
            self.assertEqual(tu.resolve_device('auto'), torch.device('cpu'))
        with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                mock.patch.object(torch.backends.mps, 'is_available', return_value=True):
            self.assertEqual(tu.resolve_device('auto'), torch.device('mps', 0))
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'current_device', return_value=0):
            self.assertEqual(tu.resolve_device('auto'), torch.device('cuda', 0))

    def test_explicit_indices(self):
        """Accelerator devices get explicit indices so that they compare equal to tensor devices."""
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'device_count', return_value=2), \
                mock.patch.object(torch.cuda, 'current_device', return_value=1):
            self.assertEqual(tu.resolve_device('cuda'), torch.device('cuda', 1))
            self.assertEqual(tu.resolve_device('cuda:0'), torch.device('cuda', 0))
        with mock.patch.object(torch.backends.mps, 'is_available', return_value=True):
            self.assertEqual(tu.resolve_device('mps'), torch.device('mps', 0))

    def test_unavailable_devices(self):
        """Unavailable or unknown devices raise ValueError."""
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(ValueError):
                tu.resolve_device('cuda')
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.object(torch.cuda, 'device_count', return_value=1):
            with self.assertRaises(ValueError):
                tu.resolve_device('cuda:3')
        with mock.patch.object(torch.backends.mps, 'is_available', return_value=False):
            with self.assertRaises(ValueError):
                tu.resolve_device('mps')
        for name in ['tpu', 'not a device', 'xla']:
            with self.assertRaises(ValueError):
                tu.resolve_device(name)

    def test_mps_float64(self):
        """MPS does not support float64."""
        with self.assertRaises(ValueError):
            tu._check_device_precision(torch.device('mps'), torch.float64)
        tu._check_device_precision(torch.device('mps'), torch.float32)

    def test_set_device(self):
        """set_device changes the default device for flows built afterwards."""
        previous = tu.get_device()
        try:
            self.assertEqual(tu.set_device('cpu'), torch.device('cpu'))
            self.assertEqual(tu.get_device(), torch.device('cpu'))
            with mock.patch.object(torch.cuda, 'is_available', return_value=False):
                with self.assertRaises(ValueError):
                    tu.set_device('cuda')
            self.assertEqual(tu.get_device(), torch.device('cpu'))
        finally:
            tu._default_device = previous

    def test_call_with_cpu_fallback(self):
        """Missing device kernels fall back to the host and are remembered."""
        calls = []

        class FakeTensor(object):
            """Minimal stand-in for a tensor on an accelerator."""
            device = torch.device('meta')

        def fake_operator(x):
            calls.append(x)
            if not isinstance(x, torch.Tensor):
                raise NotImplementedError('no kernel')
            return x + 1.

        with mock.patch.object(tu, 'run_on_cpu', side_effect=lambda fn, *t: fn(torch.zeros(1))) as fallback:
            tu._missing_operators.discard(('fake', 'meta'))
            result = tu.call_with_cpu_fallback('fake', fake_operator, FakeTensor())
            self.assertEqual(float(result), 1.)
            self.assertIn(('fake', 'meta'), tu._missing_operators)
            # the second call goes directly to the host:
            tu.call_with_cpu_fallback('fake', fake_operator, FakeTensor())
            self.assertEqual(fallback.call_count, 2)
            self.assertEqual(len(calls), 3)
        tu._missing_operators.discard(('fake', 'meta'))
        # on the cpu the operator runs directly:
        x = torch.ones(2, dtype=tu.get_precision())
        self.assertTrue(torch.equal(tu.call_with_cpu_fallback('fake', fake_operator, x), x + 1.))

    def test_run_on_cpu(self):
        """run_on_cpu evaluates on the host and returns on the input device."""
        x = torch.linspace(0.1, 0.9, 5, dtype=tu.get_precision())
        result = tu.run_on_cpu(torch.special.ndtri, x)
        self.assertEqual(result.device, x.device)
        self.assertTrue(torch.allclose(result, torch.special.ndtri(x)))

    def test_run_on_cpu_tuple_outputs(self):
        """run_on_cpu keeps the container type of tuple outputs."""
        x = torch.tensor([[2., 0.], [0., 3.]], dtype=tu.get_precision())
        sign, log_abs_det = tu.run_on_cpu(torch.linalg.slogdet, x)
        self.assertAlmostEqual(float(sign), 1.)
        self.assertAlmostEqual(float(log_abs_det), float(np.log(6.)), places=5)
        pair = tu.run_on_cpu(lambda a, b: (a + b, a - b), x, x)
        self.assertIsInstance(pair, tuple)
        self.assertTrue(torch.equal(pair[0], 2. * x))
        self.assertTrue(torch.equal(pair[1], torch.zeros_like(x)))

    def test_cuda_float64_warning(self):
        """float64 on CUDA warns once per process."""
        previous = tu._cuda_float64_warned
        try:
            tu._cuda_float64_warned = False
            with self.assertWarns(UserWarning):
                tu._check_device_precision(torch.device('cuda', 0), torch.float64)
            self.assertTrue(tu._cuda_float64_warned)
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                tu._check_device_precision(torch.device('cuda', 0), torch.float64)
                tu._check_device_precision(torch.device('cuda', 0), torch.float32)
        finally:
            tu._cuda_float64_warned = previous

    def test_init_from_environment(self):
        """TENSIOMETER_PRECISION and TENSIOMETER_DEVICE are forwarded to the setters, blank values ignored."""
        with mock.patch.object(tu, 'set_precision') as set_precision, \
                mock.patch.object(tu, 'set_device') as set_device:
            with mock.patch.dict(os.environ, {'TENSIOMETER_PRECISION': 'double', 'TENSIOMETER_DEVICE': 'cpu'}):
                tu._init_from_environment()
            set_precision.assert_called_once_with('double')
            set_device.assert_called_once_with('cpu')
            with mock.patch.dict(os.environ, {'TENSIOMETER_PRECISION': ' ', 'TENSIOMETER_DEVICE': ''}):
                tu._init_from_environment()
            self.assertEqual(set_precision.call_count, 1)
            self.assertEqual(set_device.call_count, 1)

#########################################################################################################
# Conversion and saving helpers


class TestTensorHelpers(unittest.TestCase):
    """to_tensor and atomic_save."""

    def test_to_tensor_list_of_tensors(self):
        """A list of tensors is stacked on the target device with the active precision."""
        rows = [torch.tensor([1., 2.], dtype=torch.float64), torch.tensor([3., 4.], dtype=torch.float32)]
        stacked = tu.to_tensor(rows)
        self.assertEqual(tuple(stacked.shape), (2, 2))
        self.assertEqual(stacked.dtype, tu.get_precision())
        self.assertEqual(stacked.device, torch.device('cpu'))
        self.assertTrue(torch.equal(stacked, torch.tensor([[1., 2.], [3., 4.]], dtype=tu.get_precision())))

    def test_atomic_save_creates_directory(self):
        """atomic_save creates missing directories and leaves no temporary file."""
        with tempfile.TemporaryDirectory() as tmp:
            directory = os.path.join(tmp, 'nested', 'folder')
            path = os.path.join(directory, 'object.pt')
            tu.atomic_save({'value': torch.arange(3)}, path)
            self.assertEqual(os.listdir(directory), ['object.pt'])
            loaded = torch.load(path)
            self.assertTrue(torch.equal(loaded['value'], torch.arange(3)))

#########################################################################################################
# CPU flows


class TestCPUFlows(unittest.TestCase):
    """Public outputs and device bookkeeping on the CPU."""

    def test_flow_device(self):
        """Flows record their device and return CPU tensors."""
        flow = sp.FlowCallback(_chain(), device='cpu', **_FLOW_KWARGS)
        self.assertEqual(flow.device, torch.device('cpu'))
        self.assertEqual(flow.trainable_transformation.device, torch.device('cpu'))
        x = flow.sample(5)
        for result in [x, flow.log_probability(x), flow.log_probability_jacobian(x), flow.metric(x)]:
            self.assertEqual(result.device.type, 'cpu')
        self.assertEqual(flow.to('cpu'), flow)

    def test_invalid_device(self):
        """Flows on unavailable devices are rejected."""
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(ValueError):
                sp.FlowCallback(_chain(), device='cuda', **_FLOW_KWARGS)

#########################################################################################################
# Accelerators


@unittest.skipIf(len(_accelerators()) == 0, 'no accelerator available for the active precision')
class TestAcceleratorFlows(unittest.TestCase):
    """Flows on GPUs (CUDA or MPS)."""

    def test_bijectors_on_device(self):
        """Every bijector works on the device, with third derivatives, matching the CPU."""
        torch.manual_seed(0)
        for device in _accelerators():
            flows = [
                tb.AutoregressiveFlow(3, transformation_type='affine', n_transformations=2, scale_roto_shift=True,
                                      device='cpu'),
                tb.AutoregressiveFlow(3, transformation_type='spline', n_transformations=2, device='cpu'),
                tb.AutoregressiveFlow(3, transformation_type='spline', periodic_params=[True, False, False],
                                      n_transformations=2, device='cpu'),
                tb.AutoregressiveFlow(3, transformation_type='spline', map_to_unitcube=True, n_transformations=1,
                                      device='cpu'),
            ]
            fixed = [bj.Chain([bj.Shift(1.), bj.Scale(2.)]), bj.Permute([2, 0, 1]), bj.Invert(bj.NormalCDF()),
                     bj.AffineTriL(np.zeros(3), np.array([[1., 0., 0.], [0.5, 2., 0.], [0.1, 0.2, 0.3]]))]
            x_cpu = 0.5 * torch.randn(8, 3, dtype=tu.get_precision())
            for bijector in [_f.bijector for _f in flows] + fixed:
                moved = copy.deepcopy(bijector).to(device)
                # the inverse normal CDF acts on (0, 1):
                if isinstance(bijector, bj.Invert):
                    x_cpu_input = torch.sigmoid(x_cpu)
                else:
                    x_cpu_input = x_cpu
                x = x_cpu_input.to(device)
                y = moved.forward(x)
                self.assertEqual(y.device.type, device.type)
                self.assertTrue(torch.allclose(moved.inverse(y), x, atol=1e-4))
                self.assertTrue(torch.allclose(y.cpu(), bijector.forward(x_cpu_input), atol=1e-4))
                xg = x.clone().requires_grad_(True)
                jacobian = autodiff.batch_jacobian(moved.forward, xg)
                # slogdet has no MPS kernel in older torch versions, compare on the host:
                self.assertTrue(torch.allclose(torch.linalg.slogdet(jacobian.cpu())[1],
                                               moved.forward_log_det_jacobian(x).cpu(), atol=1e-3))
                third = autodiff.batch_jacobian(
                    lambda z: autodiff.batch_jacobian(
                        lambda w: autodiff.gradient(lambda v: moved.forward_log_det_jacobian(v), w), z), xg,
                    create_graph=False)
                self.assertTrue(bool(torch.isfinite(third).all()))

    def test_flow_on_device(self):
        """Training, evaluation, CPU/device agreement, save and load across devices."""
        torch.manual_seed(1)
        np.random.seed(1)
        for device in _accelerators():
            flow = sp.FlowCallback(_chain(), device='cpu', **_FLOW_KWARGS)
            flow.train(epochs=1, verbose=0)
            x = flow.sample(20)
            cpu_log_prob = flow.log_probability(x)
            # move to the device:
            moved = copy.deepcopy(flow).to(device)
            self.assertEqual(moved.device.type, device.type)
            self.assertEqual(moved.training_dataset[0].device.type, device.type)
            result = moved.log_probability(x)
            self.assertEqual(result.device.type, 'cpu')
            self.assertTrue(torch.allclose(result, cpu_log_prob, atol=1e-4))
            self.assertEqual(moved.sample(5).device.type, 'cpu')
            self.assertEqual(moved.log_probability_hessian(x[:3]).device.type, 'cpu')
            # graph-attached outputs stay on the device:
            xg = x.to(device).requires_grad_(True)
            self.assertEqual(moved.log_probability(xg).device.type, device.type)
            # training on the device:
            history = moved.train(epochs=2, verbose=0)
            self.assertEqual(len(history.history['loss']), 2)
            # save on the device and load on the cpu:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'flow.pt')
                moved.save(path)
                loaded = sp.FlowCallback.load(path, device='cpu')
                self.assertEqual(loaded.device, torch.device('cpu'))
                self.assertEqual(tu.module_device(loaded.trainable_bijector), torch.device('cpu'))
                self.assertTrue(torch.allclose(loaded.log_probability(x), moved.log_probability(x), atol=1e-4))
                loaded.train(epochs=1, verbose=0)
                # and back on the device:
                again = sp.FlowCallback.load(path, device=device)
                self.assertEqual(again.device.type, device.type)
                again.train(epochs=1, verbose=0)
            # back to the cpu:
            moved.to('cpu')
            self.assertEqual(tu.module_device(moved.trainable_bijector), torch.device('cpu'))
            # transformed and average flows move with their parts:
            transformed = sp.TransformedFlowCallback(copy.deepcopy(flow), [bj.Identity(), bj.Scale(2.)])
            reference = transformed.log_probability(x)
            transformed.to(device)
            self.assertEqual(transformed.device.type, device.type)
            self.assertTrue(torch.allclose(transformed.log_probability(x), reference, atol=1e-4))
            average = sp.average_flow([copy.deepcopy(flow), copy.deepcopy(flow)])
            reference = average.log_probability(x)
            average.to(device)
            self.assertTrue(all(_f.device.type == device.type for _f in average.flows))
            self.assertTrue(torch.allclose(average.log_probability(x), reference, atol=1e-4))
            self.assertEqual(average.sample(4).device.type, 'cpu')


if __name__ == '__main__':
    unittest.main(verbosity=2)
