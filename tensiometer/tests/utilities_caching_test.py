###############################################################################
# initial imports:

import os
import pickle
import unittest
from functools import wraps
from tempfile import TemporaryDirectory

from tensiometer.utilities.caching import cache_input

###############################################################################

class TestCacheInput(unittest.TestCase):

    def setUp(self):
        """
        Set up a temporary directory for testing the cache.
        """
        self.temp_dir = TemporaryDirectory()
        self.cache_dir = self.temp_dir.name

    def tearDown(self):
        """
        Clean up the temporary directory after each test.
        """
        self.temp_dir.cleanup()

    def test_cache_creation(self):
        """
        Test that the cache file is created and used correctly.
        """
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            return a + b

        # First call: creates the cache
        result1 = sample_function(2, 3, cache_dir=self.cache_dir, root_name="test_cache")
        self.assertEqual(result1, 5, "Function should return the correct result.")

        # Check that the cache file exists
        expected_cache_file = os.path.join(self.cache_dir, "test_cache_function_cache.plk")
        self.assertTrue(os.path.exists(expected_cache_file), "Cache file should be created.")

        # Second call: uses the cache
        result2 = sample_function(2, 3, cache_dir=self.cache_dir, root_name="test_cache")
        self.assertEqual(result2, 5, "Function should reuse cached arguments.")

    def test_cache_mismatch(self):
        """
        Test that a ValueError is raised when cached arguments do not match the provided arguments.
        """
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            return a + b

        # First call: creates the cache
        sample_function(2, 3, cache_dir=self.cache_dir, root_name="test_cache")

        # Second call: mismatched arguments
        with self.assertRaises(ValueError, msg="Should raise ValueError for mismatched arguments"):
            sample_function(3, 4, cache_dir=self.cache_dir, root_name="test_cache")

    def test_missing_cache_dir_or_root_name(self):
        """
        Test that a ValueError is raised when cache_dir or root_name is not provided.
        """
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            return a + b

        with self.assertRaises(ValueError, msg="Should raise ValueError when cache_dir is missing"):
            sample_function(2, 3, root_name="test_cache")

        with self.assertRaises(ValueError, msg="Should raise ValueError when root_name is missing"):
            sample_function(2, 3, cache_dir=self.cache_dir)

    def test_function_execution_without_cache(self):
        """
        Test that the function executes correctly when caching parameters are not provided.
        """
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            return a * b

        with self.assertRaises(ValueError, msg="Should raise ValueError when caching parameters are missing"):
            sample_function(4, 5)

    def test_cache_persistence(self):
        """
        Test that the cached arguments persist across multiple function calls.
        """
        @cache_input
        def sample_function(a, b, cache_dir=None, root_name=None):
            return a - b

        # Create a cache
        result1 = sample_function(10, 5, cache_dir=self.cache_dir, root_name="persistent_cache")
        self.assertEqual(result1, 5, "Function should return the correct result.")

        # Simulate a new function call that uses the cached arguments
        result2 = sample_function(10, 5, cache_dir=self.cache_dir, root_name="persistent_cache")
        self.assertEqual(result2, 5, "Cached result should persist and match the original result.")

    def test_nested_function_calls(self):
        """
        Test that nested function calls are handled correctly.
        """
        @cache_input
        def inner_function(x, cache_dir=None, root_name=None):
            return x * x

        @cache_input
        def outer_function(y, cache_dir=None, root_name=None):
            return inner_function(y, cache_dir=cache_dir, root_name="inner_cache")

        # First call to outer function
        result1 = outer_function(3, cache_dir=self.cache_dir, root_name="outer_cache")
        self.assertEqual(result1, 9, "Outer function should correctly call and cache the inner function.")

        # Second call to outer function, should use both outer and inner caches
        result2 = outer_function(3, cache_dir=self.cache_dir, root_name="outer_cache")
        self.assertEqual(result2, 9, "Outer function should reuse the cached result.")

if __name__ == '__main__':
    unittest.main()
