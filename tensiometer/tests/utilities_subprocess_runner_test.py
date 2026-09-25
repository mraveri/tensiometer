"""Tests for subprocess runner utilities."""

#########################################################################################################
# Imports

import contextlib
import io
import sys
import time
import types
import unittest
import warnings
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


class _FakeResource:
    """Fake resource module reporting a fixed peak memory."""
    RUSAGE_SELF = 0

    def __init__(self, maxrss):
        """Init."""
        self._maxrss = maxrss

    def getrusage(self, who):
        """Getrusage."""
        return types.SimpleNamespace(ru_maxrss=self._maxrss)


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
# Functions run in real subprocesses (a new interpreter imports them from this module)

_PARENT_STATE = {"value": 0}


@spr.run_in_process(subprocess=True, monitoring=False, feedback_level=0)
def _spawn_boom():
    """Raise in the subprocess."""
    raise ValueError("boom")


@spr.run_in_process(subprocess=True, monitoring=True, monitoring_frequency=0, timeout=1)
def _spawn_sleepy():
    """Sleep longer than the timeout."""
    time.sleep(2.0)
    return "done"


@spr.run_in_process(subprocess=True, monitoring=True, monitoring_frequency=0, feedback_level=3)
def _spawn_echo(x):
    """Double the input."""
    return x * 2


@spr.run_in_process(subprocess=True, monitoring=False, feedback_level=0)
def _spawn_parent_state():
    """Module state seen by a spawned subprocess."""
    return _PARENT_STATE["value"]


@spr.run_in_process(subprocess=True, context="forkserver", monitoring=False, feedback_level=0)
def _forkserver_parent_state():
    """Module state seen by a forkserver subprocess."""
    return _PARENT_STATE["value"]


@spr.run_in_process(subprocess=True, context="spawn", monitoring=False, feedback_level=0)
def _spawn_square(x):
    """Square."""
    return x * x


@spr.run_in_process(subprocess=True, monitoring=True, monitoring_frequency=1, feedback_level=1)
def _spawn_allocate(size_mb):
    """Allocate memory in the subprocess and return before the first monitoring sample."""
    data = b"x" * (size_mb * 1024 * 1024)
    return len(data)


def _plain_double(x):
    """Undecorated function."""
    return 2 * x

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
        with self.assertRaises(ValueError):
            _spawn_boom()

    def test_timeout_enforced(self):
        """Confirm timeouts enforce process termination."""
        with self.assertRaises(TimeoutError):
            _spawn_sleepy()

        with patch("tensiometer.tests.utilities_subprocess_runner_test.time.sleep") as mock_sleep:
            result = _spawn_sleepy.__wrapped__()
            self.assertEqual(result, "done")
            mock_sleep.assert_called_once()

    def test_monitoring_feedback_flow(self):
        """Verify monitoring feedback path."""
        self.assertEqual(_spawn_echo(4), 8)
        self.assertEqual(_spawn_echo.__wrapped__(4), 8)

    def test_spawn_is_independent(self):
        """spawn and forkserver subprocesses do not inherit the state of the calling process."""
        _PARENT_STATE["value"] = 1
        try:
            self.assertEqual(_spawn_parent_state.__wrapped__(), 1)
            self.assertEqual(_spawn_parent_state(), 0)
            self.assertEqual(_forkserver_parent_state(), 0)
        finally:
            _PARENT_STATE["value"] = 0

    def test_spawn_needs_importable_function(self):
        """With spawn and forkserver, functions a new interpreter cannot import are rejected when decorated."""

        def nested():
            """Nested function."""
            return None

        for context in ("spawn", "forkserver"):
            with self.assertRaises(ValueError):
                spr.run_in_process(context=context)(nested)
        interactive = types.FunctionType(nested.__code__, {}, "interactive")
        interactive.__module__ = "__main__"
        interactive.__qualname__ = "interactive"
        with patch.dict(sys.modules, {"__main__": types.ModuleType("__main__")}):
            with self.assertRaises(ValueError):
                spr.run_in_process()(interactive)
        self.assertTrue(callable(spr.run_in_process(context="fork")(nested)))

    def test_resolve_function(self):
        """A new interpreter finds the original function through the decorated one."""
        module_name = _spawn_echo.__module__
        self.assertIs(spr._resolve_function(module_name, "_spawn_echo"), _spawn_echo.__wrapped__)
        self.assertIs(spr._resolve_function(module_name, "_plain_double"), _plain_double)

    def test_subprocess_target_with_reference(self):
        """The subprocess target imports the function and sends back its result or its exception."""
        module_name = _plain_double.__module__
        shared = []
        spr._subprocess_target(_FakePipeConn(shared), (module_name, "_plain_double"), (3,), {})
        result, peak_memory = shared
        self.assertEqual(result, 6)
        self.assertGreater(peak_memory, 0.0)
        shared = []
        spr._subprocess_target(_FakePipeConn(shared), (module_name, "_spawn_boom"), (), {})
        error, peak_memory = shared
        self.assertIsInstance(error, ValueError)
        self.assertIn("boom", error.traceback)
        self.assertGreater(peak_memory, 0.0)

    def test_peak_memory_units(self):
        """ru_maxrss is in bytes on macOS and in kilobytes on the other systems."""
        with patch.object(spr, "resource", _FakeResource(2 * 1024 * 1024)):
            with patch.object(spr, "sys", types.SimpleNamespace(platform="darwin")):
                self.assertAlmostEqual(spr._peak_memory(), 2.0)
            with patch.object(spr, "sys", types.SimpleNamespace(platform="linux")):
                self.assertAlmostEqual(spr._peak_memory(), 2048.0)
        with patch.object(spr, "resource", None):
            self.assertIsNone(spr._peak_memory())
        self.assertGreater(spr._peak_memory(), 0.0)

    def test_peak_memory_of_short_subprocess(self):
        """The peak memory of a subprocess that ends between monitoring samples is measured."""
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(_spawn_allocate(200), 200 * 1024 * 1024)
        peak_lines = [line for line in buffer.getvalue().splitlines() if "Peak memory usage" in line]
        self.assertEqual(len(peak_lines), 1)
        peak_memory = float(peak_lines[0].split(":")[1].split()[0])
        self.assertGreater(peak_memory, 200.0)

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

        @spr.run_in_process(subprocess=True, context="fork", monitoring=True, monitoring_frequency=0, feedback_level=1)
        def add(a, b):
            """Add."""
            return a + b

        self.assertEqual(add(1, 2), 3)

    @patch("tensiometer.utilities.subprocess_runner.psutil.Process", _FakePsutilProcess)
    @patch("tensiometer.utilities.subprocess_runner.mp.get_context")
    def test_inline_process_exception_reraised_after_monitoring(self, mock_get_context):
        """Test Inline process exception reraised after monitoring."""
        mock_get_context.return_value = _FakeContext()

        @spr.run_in_process(subprocess=True, context="fork", monitoring=True, monitoring_frequency=0, feedback_level=1)
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

        @spr.run_in_process(subprocess=True, context="fork", monitoring=True, monitoring_frequency=0, feedback_level=0)
        def identity(val):
            """Identity."""
            return val

        self.assertEqual(identity("silent"), "silent")

    @patch("tensiometer.utilities.subprocess_runner.mp.get_context")
    def test_nonzero_exitcode_raises(self, mock_get_context):
        """Test Nonzero exitcode raises."""
        mock_get_context.return_value = _FakeContext(process_factory=_NonZeroExitProcess)

        @spr.run_in_process(subprocess=True, context="fork", monitoring=False, feedback_level=3)
        def noop():
            """Noop."""
            return None

        with self.assertRaises(Exception):
            noop()
        self.assertIsNone(noop.__wrapped__())

    def test_cuda_initialized_without_torch(self):
        """Test that CUDA is reported as not initialized when torch is not imported."""
        with patch.dict(sys.modules, {"torch": None}):
            self.assertFalse(spr._cuda_initialized())

    def test_cuda_initialized_query(self):
        """Test the CUDA initialization query, including a failing torch query."""
        import torch
        with patch.object(torch.cuda, "is_initialized", return_value=True):
            self.assertTrue(spr._cuda_initialized())
        with patch.object(torch.cuda, "is_initialized", return_value=False):
            self.assertFalse(spr._cuda_initialized())
        with patch.object(torch.cuda, "is_initialized", side_effect=RuntimeError("broken")):
            self.assertFalse(spr._cuda_initialized())

    @patch("tensiometer.utilities.subprocess_runner._cuda_initialized", return_value=True)
    @patch("tensiometer.utilities.subprocess_runner.mp.get_context")
    def test_fork_with_cuda_initialized_warns(self, mock_get_context, mock_cuda):
        """Test the warning when forking a process after CUDA was initialized."""
        mock_get_context.return_value = _FakeContext()

        @spr.run_in_process(subprocess=True, context="fork", monitoring=False, feedback_level=0)
        def square(x):
            """Square."""
            return x * x

        with self.assertWarns(UserWarning) as caught:
            self.assertEqual(square(3), 9)
        self.assertIn("square", str(caught.warning))
        mock_get_context.assert_called_once_with("fork")

    @patch("tensiometer.utilities.subprocess_runner._cuda_initialized", return_value=True)
    @patch("tensiometer.utilities.subprocess_runner.mp.get_context")
    def test_spawn_with_cuda_initialized_does_not_warn(self, mock_get_context, mock_cuda):
        """Test that non-fork contexts do not warn about CUDA."""
        mock_get_context.return_value = _FakeContext()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(_spawn_square(4), 16)
        mock_cuda.assert_not_called()
        mock_get_context.assert_called_once_with("spawn")

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
