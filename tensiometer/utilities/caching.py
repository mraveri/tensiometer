"""
This file contains utilities that are useful for caching
"""

###########################################################################################
# Initial imports:

import os
import pickle
from functools import wraps

###########################################################################################

def cache_input(func):
    """
    A decorator to cache the input arguments of a function to a file and reuse them
    in subsequent function calls, so that one can call a function by only specifying the cache details. 

    The decorator expects the function to be called with two specific keyword arguments:
    `cache_dir` (the directory to store the cache file) and `root_name` (the base name for the cache file).

    Behavior:
        - If `cache_dir` or `root_name` is not provided in the function's keyword arguments, 
          the decorator raises a `ValueError`.
        - If the specified `cache_dir` does not exist, it is created.
        - The cache file is saved with the name `<root_name>_function_cache.plk` in the specified `cache_dir`.
        - If the cache file exists, the function's input arguments are loaded from the cache instead
          of using the provided arguments.

    Parameters:
        func (function): The function to decorate.

    Returns:
        function: The decorated function with caching capability.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Retrieve caching essentials from function's keyword arguments
        cache_dir = kwargs.get('cache_dir', None)
        root_name = kwargs.get('root_name', None)

        # If no cache_dir or root_name is provided, raise an error
        if cache_dir is None or root_name is None:
            raise ValueError('cache_dir and root_name must be provided for input to be cached')

        # Ensure the cache directory exists
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Define the cache file path
        cache_file = os.path.join(cache_dir, root_name + '_function_cache.plk')

        # If the cache file exists, load the cached arguments
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_args, cached_kwargs = pickle.load(f)
            # check whether the cached arguments are the same as the current ones:
            if args != () and args != cached_args:
                raise ValueError(
                    f"Cached arguments are different from current ones. "
                    f"args: {args}, cached_args: {cached_args}. "
                    f"Please remove the cache file and run the function again."
                )
            if kwargs != {}:
                for key in kwargs:
                    if kwargs[key] != cached_kwargs[key]:
                        raise ValueError(
                            f"Cached arguments are different from current ones. "
                            f"Error in key: {key}. "
                            f"Provided: {kwargs[key]}, Cached: {cached_kwargs[key]}. "
                            f"Please remove the cache file and run the function again."
                        )
            # if all checks pass, return the cached result:
            args = cached_args
            kwargs = cached_kwargs
        else:
            with open(cache_file, 'wb') as f:
                pickle.dump((args, kwargs), f)

        # Call the original function with the arguments
        return func(*args, **kwargs)

    return wrapper
