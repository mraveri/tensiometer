###############################################################################
# initial imports:

import unittest
from multiprocessing import ProcessError
import time

from tensiometer.utilities.subprocess_runner import run_in_process, default_settings

###############################################################################

class TestRunInProcess(unittest.TestCase):

    def test_basic_function_execution(self):
        @run_in_process()
        def add(a, b):
            return a + b

        result = add(2, 3)
        self.assertEqual(result, 5, "Function should return the correct result when running in a subprocess")

    def test_feedback_level(self):
        @run_in_process(feedback_level=0)
        def echo(value):
            return value

        result = echo("test")
        self.assertEqual(result, "test", "Function should return the input value even with minimal feedback")

    def test_timeout(self):
        @run_in_process(timeout=1)
        def long_running_task():
            time.sleep(2)
            return "Completed"

        with self.assertRaises(TimeoutError, msg="Function should raise TimeoutError when exceeding the timeout"):
            long_running_task()

    def test_invalid_feedback_level(self):
        with self.assertRaises(ValueError, msg="Should raise ValueError for invalid feedback_level"):
            @run_in_process(feedback_level=4)
            def dummy():
                pass

    def test_context_validation(self):
        with self.assertRaises(ValueError, msg="Should raise ValueError for invalid context"):
            @run_in_process(context="invalid_context")
            def dummy():
                pass

    def test_no_subprocess_execution(self):
        @run_in_process(subprocess=False)
        def multiply(a, b):
            return a * b

        result = multiply(3, 4)
        self.assertEqual(result, 12, "Function should execute directly without a subprocess")

    def test_monitoring_enabled(self):
        @run_in_process(monitoring=True, monitoring_frequency=1)
        def quick_task():
            return "Quick task completed"

        result = quick_task()
        self.assertEqual(result, "Quick task completed", "Function should execute successfully with monitoring enabled")

    def test_exception_handling(self):
        @run_in_process()
        def faulty_function():
            raise ValueError("This is an intentional error")

        with self.assertRaises(ValueError, msg="Function should propagate exceptions raised in the subprocess"):
            faulty_function()

if __name__ == '__main__':
    unittest.main()
