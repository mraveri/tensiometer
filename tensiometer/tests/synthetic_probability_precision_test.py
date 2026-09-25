"""
Tests of the precision settings of the synthetic probability package.

The precision is process-wide and locked once a bijector exists, so most checks run in
fresh interpreters.
"""

#########################################################################################################
# Imports

import os
import subprocess
import sys
import unittest

import numpy as np
import torch

import tensiometer.synthetic_probability.synthetic_probability as sp
from tensiometer.synthetic_probability import bijectors as bj
from tensiometer.synthetic_probability import tensor_utilities as tu

#########################################################################################################
# Helpers

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))


def _run_python(code, precision=None):
    """
    Run code in a fresh interpreter.

    :param code: python source.
    :param precision: value of ``TENSIOMETER_PRECISION``, None to unset it.
    :returns: completed process.
    """
    env = dict(os.environ)
    env.pop('TENSIOMETER_PRECISION', None)
    env.pop('TENSIOMETER_DEVICE', None)
    if precision is not None:
        env['TENSIOMETER_PRECISION'] = precision
    return subprocess.run([sys.executable, '-c', code], cwd=_REPO, env=env, capture_output=True, text=True)

#########################################################################################################
# Parsing


class TestPrecisionParsing(unittest.TestCase):
    """Precision specifications."""

    def test_accepted_names(self):
        """All the accepted names map to float32 or float64."""
        for name in ['float32', '32', 'single', 'FLOAT32', ' float ', np.float32, torch.float32, np.dtype('float32')]:
            self.assertEqual(tu._parse_precision(name), torch.float32)
        for name in ['float64', '64', 'double', 'Double', np.float64, torch.float64]:
            self.assertEqual(tu._parse_precision(name), torch.float64)

    def test_invalid_names(self):
        """Other values raise ValueError."""
        for name in ['float16', 'half', '128', torch.float16, np.int32, 'bfloat16']:
            with self.assertRaises(ValueError):
                tu._parse_precision(name)

    def test_abstract_numpy_types(self):
        """Abstract numpy types that are not dtypes raise ValueError."""
        for name in [np.floating, np.generic, np.number]:
            with self.assertRaises(ValueError):
                tu._parse_precision(name)

    def test_forwarding(self):
        """sp.prec and sp.np_prec forward to the live values."""
        self.assertEqual(sp.prec, tu.get_precision())
        self.assertEqual(sp.np_prec, tu.np_prec)
        self.assertEqual(np.dtype(sp.np_prec), np.dtype(str(sp.prec).replace('torch.', '')))
        with self.assertRaises(AttributeError):
            sp.not_a_module_attribute

    def test_locked_after_bijector(self):
        """Building a bijector locks the precision; changing it then raises."""
        bj.Identity()
        self.assertTrue(tu.is_precision_locked())
        other = torch.float64 if tu.get_precision() == torch.float32 else torch.float32
        with self.assertRaises(RuntimeError):
            tu.set_precision(other)
        # setting the same precision is allowed:
        self.assertEqual(tu.set_precision(tu.get_precision()), tu.get_precision())

    def test_set_precision_when_unlocked(self):
        """An unlocked precision can be switched in process; the MPS default device rejects float64."""
        previous = (tu.prec, tu.np_prec, tu._precision_locked, tu._default_device)
        other = torch.float64 if tu.get_precision() == torch.float32 else torch.float32
        try:
            tu._precision_locked = False
            tu._default_device = torch.device('cpu')
            self.assertEqual(tu.set_precision(other), other)
            self.assertEqual(tu.get_precision(), other)
            self.assertEqual(np.dtype(tu.np_prec), np.dtype(str(other).replace('torch.', '')))
            self.assertEqual(tu.set_precision('float32'), torch.float32)
            tu._default_device = torch.device('mps', 0)
            with self.assertRaises(ValueError):
                tu.set_precision('float64')
            self.assertEqual(tu.get_precision(), torch.float32)
        finally:
            tu.prec, tu.np_prec, tu._precision_locked, tu._default_device = previous

    def test_dtypes_follow_precision(self):
        """Parameters, buffers and outputs use the active precision."""
        shift = bj.Shift(1.)
        self.assertEqual(shift.shift.dtype, tu.get_precision())
        self.assertEqual(shift(np.zeros((2, 3))).dtype, tu.get_precision())
        self.assertEqual(shift(torch.zeros(2, 3, dtype=torch.float16)).dtype, tu.get_precision())
        self.assertEqual(tu.to_tensor([1, 2]).dtype, tu.get_precision())

#########################################################################################################
# Fresh interpreters


class TestPrecisionSelection(unittest.TestCase):
    """Selection of the precision at import and with set_precision."""

    def test_environment_variable(self):
        """TENSIOMETER_PRECISION selects the precision at import."""
        code = (
            "import torch\n"
            "from tensiometer.synthetic_probability import tensor_utilities as tu, bijectors as bj\n"
            "import tensiometer.synthetic_probability.synthetic_probability as sp\n"
            "assert tu.get_precision() == torch.float64\n"
            "assert sp.prec == torch.float64\n"
            "assert bj.Shift(1.)(torch.zeros(2)).dtype == torch.float64\n")
        result = _run_python(code, precision='double')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_environment_variable(self):
        """An invalid TENSIOMETER_PRECISION fails at import."""
        result = _run_python("import tensiometer.synthetic_probability.tensor_utilities", precision='float16')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ValueError', result.stderr)

    def test_default_and_set_precision(self):
        """The default is float32; set_precision before building flows switches to float64 and locks."""
        code = (
            "import numpy as np, torch\n"
            "from getdist import MCSamples\n"
            "from tensiometer.synthetic_probability import tensor_utilities as tu\n"
            "assert tu.get_precision() == torch.float32 and tu.np_prec == np.float32\n"
            "tu.set_precision('float64')\n"
            "assert tu.np_prec == np.float64\n"
            "import tensiometer.synthetic_probability.synthetic_probability as sp\n"
            "s = np.random.default_rng(0).normal(size=(300, 2))\n"
            "flow = sp.FlowCallback(MCSamples(samples=s, names=['a', 'b']), feedback=0, hidden_units=[4], n_transformations=1)\n"
            "assert flow.prec == torch.float64\n"
            "assert flow.log_probability(s[:3]).dtype == torch.float64\n"
            "assert flow.chain_samples.dtype == np.float64\n"
            "try:\n"
            "    tu.set_precision('float32')\n"
            "    raise AssertionError('no error')\n"
            "except RuntimeError:\n"
            "    pass\n")
        result = _run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_float64_accuracy(self):
        """float64 flows are accurate well beyond float32 round off."""
        code = (
            "import torch\n"
            "from tensiometer.synthetic_probability import tensor_utilities as tu\n"
            "tu.set_precision('float64')\n"
            "from tensiometer.synthetic_probability import trainable_bijectors as tb\n"
            "torch.manual_seed(0)\n"
            "flow = tb.AutoregressiveFlow(3, transformation_type='spline', n_transformations=2, scale_roto_shift=True)\n"
            "x = torch.randn(20, 3, dtype=torch.float64)\n"
            "error = (flow.bijector.inverse(flow.bijector(x)) - x).abs().max()\n"
            "assert error < 1e-10, error\n")
        result = _run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
