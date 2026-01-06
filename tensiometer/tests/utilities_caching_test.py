"""Tests for caching utility decorator."""

#########################################################################################################
# Imports

import ctypes
import inspect
import os
import pickle
import unittest
from functools import wraps
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tensiometer.utilities.caching import cache_input

#########################################################################################################
# Test cases

class TestCacheInput(unittest.TestCase):
    """Cache input test suite."""
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TemporaryDirectory()
        self.cache_dir = self.temp_dir.name

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_cache_creation(self):
        """Test cache creation and reuse."""
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            """Sample function."""
            return a + b

        result1 = sample_function(2, 3, cache_dir=self.cache_dir, root_name="test_cache")
        self.assertEqual(result1, 5, "Function should return the correct result.")

        expected_cache_file = os.path.join(self.cache_dir, "test_cache_function_cache.plk")
        self.assertTrue(os.path.exists(expected_cache_file), "Cache file should be created.")

        result2 = sample_function(2, 3, cache_dir=self.cache_dir, root_name="test_cache")
        self.assertEqual(result2, 5, "Function should reuse cached arguments.")

    def test_cache_mismatch(self):
        """Test cache mismatch detection."""
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            """Sample function."""
            return a + b

        sample_function(2, 3, cache_dir=self.cache_dir, root_name="test_cache")

        with self.assertRaises(ValueError, msg="Should raise ValueError for mismatched arguments"):
            sample_function(3, 4, cache_dir=self.cache_dir, root_name="test_cache")

    def test_cache_kwargs_mismatch(self):
        """Test kwargs mismatch validation."""
        @cache_input
        def sample_function(a, cache_dir=None, root_name=None, flag=True):
            """Sample function."""
            return a

        sample_function(1, cache_dir=self.cache_dir, root_name="kwargs_cache", flag=True)
        with self.assertRaises(ValueError):
            sample_function(1, cache_dir=self.cache_dir, root_name="kwargs_cache", flag=False)

    def test_missing_cache_dir_or_root_name(self):
        """Test missing cache_dir or root_name handling."""
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            """Sample function."""
            return a + b

        with self.assertRaises(ValueError, msg="Should raise ValueError when cache_dir is missing"):
            sample_function(2, 3, root_name="test_cache")

        with self.assertRaises(ValueError, msg="Should raise ValueError when root_name is missing"):
            sample_function(2, 3, cache_dir=self.cache_dir)

        ok = sample_function(2, 3, cache_dir=self.cache_dir, root_name="missing_ok")
        self.assertEqual(ok, 5)

    def test_function_execution_without_cache(self):
        """Test missing caching parameters."""
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            """Sample function."""
            return a * b

        with self.assertRaises(ValueError, msg="Should raise ValueError when caching parameters are missing"):
            sample_function(4, 5)

        ok = sample_function(4, 5, cache_dir=self.cache_dir, root_name="exec_ok")
        self.assertEqual(ok, 20)

    def test_cache_persistence(self):
        """Test cache persistence across calls."""
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            """Sample function."""
            return a - b

        result1 = sample_function(10, 5, cache_dir=self.cache_dir, root_name="persistent_cache")
        self.assertEqual(result1, 5, "Function should return the correct result.")

        result2 = sample_function(10, 5, cache_dir=self.cache_dir, root_name="persistent_cache")
        self.assertEqual(result2, 5, "Cached result should persist and match the original result.")

    def test_nested_function_calls(self):
        """Test nested function calls with caching."""
        @cache_input
        def inner_function(x, cache_dir=None, root_name=None):
            """Inner function."""
            return x * x

        @cache_input
        def outer_function(y, cache_dir=None, root_name=None):
            """Outer function."""
            return inner_function(y, cache_dir=cache_dir, root_name="inner_cache")

        result1 = outer_function(3, cache_dir=self.cache_dir, root_name="outer_cache")
        self.assertEqual(result1, 9, "Outer function should correctly call and cache the inner function.")

        result2 = outer_function(3, cache_dir=self.cache_dir, root_name="outer_cache")
        self.assertEqual(result2, 9, "Outer function should reuse the cached result.")

    def test_cache_check_input_disabled(self):
        """Test cache reuse when input checks are disabled."""
        def sample_function(val, cache_dir=None, root_name=None):
            """Sample function."""
            return val

        cached_fn = cache_input(sample_function, check_input=False)

        first = cached_fn(1, cache_dir=self.cache_dir, root_name="ignore_changes")
        second = cached_fn(2, cache_dir=self.cache_dir, root_name="ignore_changes")
        self.assertEqual(first, second)

    def test_cache_directory_creation(self):
        """Test cache directory creation."""
        nested_dir = os.path.join(self.cache_dir, "nested", "path")

        @cache_input
        def sample_function(a, cache_dir=None, root_name=None):
            """Sample function."""
            return a

        result = sample_function(10, cache_dir=nested_dir, root_name="auto_create")
        self.assertEqual(result, 10)
        expected_cache = os.path.join(nested_dir, "auto_create_function_cache.plk")
        self.assertTrue(os.path.exists(expected_cache))

    def test_cached_kwargs_empty_skips_validation(self):
        """Test kwargs validation skip when kwargs are cleared."""
        @cache_input
        def sample_function(value="default", cache_dir=None, root_name=None):
            """Sample function."""
            return value

        root_name = "empty_kwargs"
        cache_file = os.path.join(self.cache_dir, f"{root_name}_function_cache.plk")
        cached_args = ("from_cache",)
        cached_kwargs = {}
        # Create the cache file so os.path.exists is True when the wrapper runs.
        with open(cache_file, "wb") as f:
            f.write(b"placeholder")

        def clearing_load(_file):
            """Clearing load."""
            frame = inspect.currentframe()
            while frame and frame.f_code.co_name != "wrapper":
                frame = frame.f_back
            if frame is None:
                raise RuntimeError("wrapper frame not found")

            # Clear kwargs so the kwargs != {} branch is skipped.
            locals_dict = frame.f_locals
            locals_dict["kwargs"] = {}
            ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(1))
            return cached_args, cached_kwargs

        with patch("tensiometer.utilities.caching.pickle.load", side_effect=clearing_load):
            result = sample_function(cache_dir=self.cache_dir, root_name=root_name)

        self.assertEqual(result, "from_cache")

        with self.assertRaises(RuntimeError):
            clearing_load(None)

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
