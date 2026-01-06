"""Tests for subprocess runner utilities."""

#########################################################################################################
# Imports

import time
import unittest
from unittest.mock import patch

from tensiometer.utilities import subprocess_runner as spr

#########################################################################################################
# Helper fakes


class _FakeMemoryInfo:
    """Fake Memory Info test suite."""
    def __init__(self, rss):
        """Init."""
        self.rss = rss


class _FakePsutilProcess:
    """Fake Psutil Process test suite."""
    def __init__(self, pid):
        """Init."""
        self.pid = pid

    def memory_info(self):
        """Memory info."""
        return _FakeMemoryInfo(rss=1024 * 1024)


class _FakePipeConn:
    """Fake Pipe Conn test suite."""
    def __init__(self, shared):
        """Init."""
        self._shared = shared
        self.closed = False

    def send(self, obj):
        """Send."""
        self._shared.append(obj)

    def recv(self):
        """Recv."""
        return self._shared.pop(0)

    def poll(self):
        """Poll."""
        return bool(self._shared)

    def close(self):
        """Close."""
        self.closed = True


class _FakeProcess:
    """Fake Process test suite."""
    def __init__(self, target, args=(), kwargs=None, keep_alive=True, exitcode=0):
        """Init."""
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self._alive = keep_alive
        self.exitcode = exitcode
        self.pid = 12345

    def start(self):
        """Start."""
        self._target(*self._args, **self._kwargs)

    def is_alive(self):
        """Is alive."""
        return self._alive

    def join(self):
        """Join."""
        self._alive = False

    def terminate(self):
        """Terminate."""
        self._alive = False


class _NonZeroExitProcess(_FakeProcess):
    """Non Zero Exit Process test suite."""
    def __init__(self, target, args=(), kwargs=None):
        """Init."""
        super().__init__(target, args, kwargs, keep_alive=False, exitcode=2)

    def start(self):
        """Start."""
        self._alive = False


class _FakeContext:
    """Fake Context test suite."""
    def __init__(self, process_factory=_FakeProcess):
        """Init."""
        self._process_factory = process_factory

    def Pipe(self):
        """Pipe."""
        shared = []
        return _FakePipeConn(shared), _FakePipeConn(shared)

    def Process(self, target, args=(), kwargs=None):
        """Process."""
        return self._process_factory(target=target, args=args, kwargs=kwargs or {})

#########################################################################################################
# Runner tests


class TestSubprocessRunner(unittest.TestCase):
    """Subprocess runner test suite."""
    def test_fake_process_terminate(self):
        """Test Fake process terminate branch."""
        proc = _FakeProcess(target=lambda: None)
        proc.terminate()
        self.assertFalse(proc.is_alive())

    def test_validation_errors(self):
        """Test Validation errors."""
        with self.assertRaises(ValueError):
            spr.run_in_process(subprocess="yes")
        with self.assertRaises(ValueError):
            spr.run_in_process(feedback_level="loud")
        with self.assertRaises(ValueError):
            spr.run_in_process(context="invalid")
        with self.assertRaises(ValueError):
            spr.run_in_process(monitoring="nope")
        with self.assertRaises(ValueError):
            spr.run_in_process(monitoring_frequency=1.5)
        with self.assertRaises(ValueError):
            spr.run_in_process(timeout="soon")

    def test_passthrough_no_subprocess(self):
        """Test passthrough path when subprocess execution is disabled."""

        @spr.run_in_process(subprocess=False)
        def add(a, b):
            """Add."""
            return a + b

        self.assertEqual(add(2, 3), 5)

    def test_exception_propagates(self):
        """Ensure exceptions raised in subprocess propagate to caller."""

        @spr.run_in_process(subprocess=True, monitoring=False, feedback_level=0)
        def boom():
            """Boom."""
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            boom()

    def test_timeout_enforced(self):
        """Confirm timeouts enforce process termination."""

        @spr.run_in_process(subprocess=True, monitoring=True, monitoring_frequency=0, timeout=1)
        def sleepy():
            """Sleepy."""
            time.sleep(2.0)
            return "done"

        with self.assertRaises(TimeoutError):
            sleepy()

        with patch("tensiometer.tests.utilities_subprocess_runner_test.time.sleep") as mock_sleep:
            result = sleepy.__wrapped__()
            self.assertEqual(result, "done")
            mock_sleep.assert_called_once()

    def test_monitoring_feedback_flow(self):
        """Verify monitoring feedback path."""

        @spr.run_in_process(subprocess=True, monitoring=True, monitoring_frequency=0, feedback_level=3)
        def echo(x):
            """Echo."""
            return x * 2

        self.assertEqual(echo(4), 8)
        self.assertEqual(echo.__wrapped__(4), 8)

    def test_extra_kwargs_are_ignored(self):
        """Ensure unknown kwargs are ignored by decorator."""

        @spr.run_in_process(subprocess=False, unknown_key=True)
        def identity(val):
            """Identity."""
            return val

        self.assertEqual(identity("ok"), "ok")

    def test_feedback_level_range_error(self):
        """Test Feedback level range error."""
        with self.assertRaises(ValueError):
            spr.run_in_process(feedback_level=5)

    @patch("tensiometer.utilities.subprocess_runner.psutil.Process", _FakePsutilProcess)
    @patch("tensiometer.utilities.subprocess_runner.mp.get_context")
    def test_inline_process_success_with_monitoring(self, mock_get_context):
        """Test Inline process success with monitoring."""
        mock_get_context.return_value = _FakeContext()

        @spr.run_in_process(subprocess=True, monitoring=True, monitoring_frequency=0, feedback_level=1)
        def add(a, b):
            """Add."""
            return a + b

        self.assertEqual(add(1, 2), 3)

    @patch("tensiometer.utilities.subprocess_runner.psutil.Process", _FakePsutilProcess)
    @patch("tensiometer.utilities.subprocess_runner.mp.get_context")
    def test_inline_process_exception_reraised_after_monitoring(self, mock_get_context):
        """Test Inline process exception reraised after monitoring."""
        mock_get_context.return_value = _FakeContext()

        @spr.run_in_process(subprocess=True, monitoring=True, monitoring_frequency=0, feedback_level=1)
        def explode():
            """Explode."""
            raise RuntimeError("kaboom")

        with self.assertRaises(RuntimeError):
            explode()

    @patch("tensiometer.utilities.subprocess_runner.psutil.Process", _FakePsutilProcess)
    @patch("tensiometer.utilities.subprocess_runner.mp.get_context")
    def test_monitoring_without_feedback_prints(self, mock_get_context):
        """Test Monitoring without feedback prints."""
        mock_get_context.return_value = _FakeContext()

        @spr.run_in_process(subprocess=True, monitoring=True, monitoring_frequency=0, feedback_level=0)
        def identity(val):
            """Identity."""
            return val

        self.assertEqual(identity("silent"), "silent")

    @patch("tensiometer.utilities.subprocess_runner.mp.get_context")
    def test_nonzero_exitcode_raises(self, mock_get_context):
        """Test Nonzero exitcode raises."""
        mock_get_context.return_value = _FakeContext(process_factory=_NonZeroExitProcess)

        @spr.run_in_process(subprocess=True, monitoring=False, feedback_level=3)
        def noop():
            """Noop."""
            return None

        with self.assertRaises(Exception):
            noop()
        self.assertIsNone(noop.__wrapped__())

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
