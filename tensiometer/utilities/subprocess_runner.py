"""
This file contains some utilities that are useful to run a function inside an independent sub-process.

When the subprocess is finished all its memory is returned to the operating system, which is useful
to contain memory leaks (for example from large machine learning models), e.g. on clusters without
memory defragmentation mechanisms.

By default the subprocess is a new Python interpreter (the ``spawn`` context), completely independent
from the calling process: it shares no memory, threads, locks or CUDA state with it. The decorated
function is then imported by the new interpreter, so it has to be defined at the top level of a module
or of a script (with the main code of the script protected by ``if __name__ == '__main__':``), not in a
notebook or inside another function, and its arguments and result have to be picklable. The
``forkserver`` context has the same requirements, with subprocesses forked from a clean server process.

The ``fork`` context runs any function, including the ones defined in notebooks, but the subprocess
starts as a copy of the calling process: it inherits its memory (pages are copied as they are
modified, so a large calling process can make the subprocess grow), its threads and locks (Python
warns that this can deadlock when the calling process runs other threads) and it cannot use CUDA
once the calling process has initialized it (in that case a warning is issued).
"""

###########################################################################################
# Initial imports:

import copy
import datetime
import importlib
import multiprocessing as mp
import sys
import time
import traceback
import warnings
from functools import wraps

import psutil

try:
    import resource
except ImportError:  # pragma: no cover - not available on Windows
    resource = None

###########################################################################################
# Default Settings:

# These settings define the default behavior of the `run_in_process` decorator.

default_settings = {
    'subprocess': True,         # Whether to run the function in a subprocess.
    'feedback_level': 1,        # Level of feedback provided:
                                # 0 - None, 1 - Minimal, 2 - Medium, 3 - Full.
    'context': 'spawn',         # Multiprocessing context: 'spawn' (independent interpreter),
                                # 'forkserver' or 'fork' (copy of the calling process).
    'monitoring': True,         # Enable or disable monitoring of the subprocess.
    'monitoring_frequency': 1,  # Frequency of monitoring updates, in seconds.
    'timeout': None,            # Maximum allowed runtime in seconds. None means no timeout.
                                # The timeout is rounded to the nearest monitoring frequency.
}

###########################################################################################
# Hard-Coded Settings:

# These constants define formatting options for feedback messages.

feedback_offset = '  '         # Indentation for feedback messages.
feedback_offset_2 = '   | '    # Indentation for nested feedback messages.
feedback_separator = '****************************************************************'
                                # Separator line for formatting feedback output.

###########################################################################################
# Helpers:


def _cuda_initialized():
    """
    Tell whether PyTorch has initialized CUDA in this process, without importing torch.

    :returns: True if torch is imported and CUDA is initialized.
    """
    torch = sys.modules.get('torch', None)
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return False


def _peak_memory():
    """
    Peak resident memory of the current process, measured by the operating system.

    :returns: peak resident memory in MB, or None where the ``resource`` module is not available.
    """
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on the other systems:
    if sys.platform == 'darwin':
        return peak / 1024 / 1024
    return peak / 1024


def _function_reference(func):
    """
    Module and qualified name that let a new interpreter import a function decorated with
    :func:`run_in_process`.

    :param func: function to run in the subprocess.
    :returns: tuple ``(module name, qualified name)``.
    :raises ValueError: if a new interpreter cannot import the function (defined inside another
        function, or in an interactive session such as a notebook).
    """
    if '<locals>' in func.__qualname__:
        raise ValueError('Function ' + func.__qualname__ + ' is defined inside another function and a new '
                         'interpreter cannot import it: define it at the top level of a module, or use '
                         "context='fork'.")
    if func.__module__ == '__main__' and not hasattr(sys.modules['__main__'], '__file__'):
        raise ValueError('Function ' + func.__qualname__ + ' is defined in an interactive session (e.g. a '
                         'notebook) and a new interpreter cannot import it: define it in a module, or use '
                         "context='fork'.")
    return func.__module__, func.__qualname__


def _resolve_function(module_name, qualname):
    """
    Import, in the subprocess, a function decorated with :func:`run_in_process`.

    :param module_name: name of the module that defines the function.
    :param qualname: qualified name of the function in the module.
    :returns: the undecorated function.
    """
    obj = importlib.import_module(module_name)
    for name in qualname.split('.'):
        obj = getattr(obj, name)
    # the module attribute is the decorated function, that keeps a reference to the original one:
    return getattr(obj, '_run_in_process_function', obj)


def _subprocess_target(pipe_conn, function, args, kwargs):
    """
    Run a function in the subprocess and send to the calling process its result, or the exception
    it raised, followed by the peak memory of the subprocess (see :func:`_peak_memory`).

    :param pipe_conn: subprocess end of the pipe to the calling process.
    :param function: the function (``fork`` context) or its ``(module name, qualified name)``
        reference (``spawn`` and ``forkserver`` contexts, see :func:`_function_reference`).
    :param args: positional arguments of the function.
    :param kwargs: keyword arguments of the function.
    """
    try:
        try:
            if isinstance(function, tuple):
                function = _resolve_function(*function)
            # execute the function and send the result through the pipe:
            result = function(*args, **kwargs)
            pipe_conn.send(result)
        except Exception as e:
            # send exceptions, with their traceback, through the pipe for error handling:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            e.traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            pipe_conn.send(e)
        # then send the peak memory, that includes sending the result:
        pipe_conn.send(_peak_memory())
    finally:
        pipe_conn.close()

###########################################################################################
# `run_in_process` Decorator:
# This decorator modifies a function to optionally run it in a separate subprocess
# with configurable feedback and monitoring.

def run_in_process(**kwargs):
    """
    Decorator to run a function in a subprocess with optional feedback,
    monitoring, and timeout capabilities.

    :param subprocess: whether to execute in a subprocess.
    :param feedback_level: verbosity level from 0 to 3.
    :param context: multiprocessing context: ``spawn`` (default, a new independent interpreter),
        ``forkserver`` or ``fork`` (a copy of the calling process); see the module documentation.
    :param monitoring: enable monitoring of the subprocess: elapsed time and peak memory. The peak
        memory is the peak resident memory of the subprocess measured by the operating system
        (for the ``fork`` context it includes the memory shared with the calling process); where
        this is not available, the maximum of the memory sampled during monitoring.
    :param monitoring_frequency: interval, in seconds, for monitoring updates.
    :param timeout: maximum runtime in seconds; enables monitoring when set.
    :param kwargs: keyword overrides of ``default_settings``; the accepted
        keys are the settings listed above, unknown keys are silently ignored.
    :returns: wrapped function with subprocess capabilities.
    :raises ValueError: if supplied configuration values are invalid, or, with the ``spawn`` and
        ``forkserver`` contexts, when decorating a function that a new interpreter cannot import.
    """
    # Update settings with the provided overrides.
    settings = copy.deepcopy(default_settings)
    for _k in kwargs.keys():
        if _k in settings.keys():
            settings[_k] = kwargs[_k]

    # Validate settings.
    if not isinstance(settings['subprocess'], bool):
        raise ValueError('subprocess must be a boolean.')
    if not isinstance(settings['feedback_level'], int):
        raise ValueError('feedback_level must be an integer.')
    if not settings['feedback_level'] in [0, 1, 2, 3]:
        raise ValueError('feedback_level must be 0, 1, 2, or 3.')
    if not settings['context'] in ['fork', 'spawn', 'forkserver']:
        raise ValueError("context must be 'fork', 'spawn', or 'forkserver'.")
    if not isinstance(settings['monitoring'], bool):
        raise ValueError('monitoring must be a boolean.')
    if not isinstance(settings['monitoring_frequency'], int):
        raise ValueError('monitoring_frequency must be an integer (in seconds).')
    if settings['timeout'] is not None and not isinstance(settings['timeout'], int):
        raise ValueError('timeout must be an integer (in seconds).')

    # Expand settings into individual variables.
    subprocess = settings['subprocess']
    feedback_level = settings['feedback_level']
    context = settings['context']
    monitoring = settings['monitoring']
    monitoring_frequency = settings['monitoring_frequency']
    timeout = settings['timeout']

    # If a timeout is set, monitoring must be enabled.
    if timeout is not None:
        monitoring = True

    def decorator(func):
        """
        Inner decorator to wrap the target function.

        :param func: target function to decorate.
        :returns: wrapped function respecting the configured subprocess options.
        :raises ValueError: with the ``spawn`` and ``forkserver`` contexts, if a new interpreter
            cannot import ``func``.
        """
        if not subprocess:
            # If not running in a subprocess, return the original function.
            return func

        # A forked subprocess runs the function directly, a new interpreter imports it:
        if context == 'fork':
            function_reference = func
        else:
            function_reference = _function_reference(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function to run the target function in a subprocess.

            :param args: positional arguments for the target function.
            :param kwargs: keyword arguments for the target function.
            :returns: result of the target function.
            """
            # Record the start time of the process.
            global_start_time = datetime.datetime.now()
            func_name = func.__name__
            result = None

            # Provide initial feedback based on the feedback level.
            if feedback_level > 0:
                print(feedback_separator, flush=True)
                print(f'* Running subprocess for function: {func_name}', flush=True)
                print(feedback_offset + f'Start time: {global_start_time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
                print(feedback_separator, flush=True)
                if feedback_level > 1:
                    print('* Settings:', flush=True)
                    for key, value in settings.items():
                        print(feedback_offset + f'- {key}: {value}', flush=True)
                    print(feedback_separator, flush=True)
                if feedback_level > 2:
                    print('* Function arguments      :', args, flush=True)
                    print('* Function keyword args   :', kwargs, flush=True)
                    print(feedback_separator, flush=True)

            # CUDA cannot be re-initialized in a forked child:
            if context == 'fork' and _cuda_initialized():
                warnings.warn('CUDA is initialized in the parent process and cannot be used in the forked '
                              'subprocess of ' + func_name + '. Initialize CUDA inside the function or run it '
                              'without a subprocess.')

            # Set up and start the subprocess.
            ctx = mp.get_context(context)
            parent_conn, child_conn = ctx.Pipe()
            
            # create the process:
            process = ctx.Process(target=_subprocess_target, args=(child_conn, function_reference, args, kwargs))
            if feedback_level > 2:
                print('* Sub-process created.', flush=True)
            
            process.start()
            if feedback_level > 2:
                print('* Sub-process started.', flush=True)    

            # initial memory usage:
            if monitoring:
                initial_memory = psutil.Process(process.pid).memory_info().rss / 1024 / 1024
                if feedback_level > 1:
                    print('* Initial memory usage:', initial_memory, 'MB', flush=True)
                    print(feedback_separator, flush=True)
                peak_memory = initial_memory
                    
            # Monitor the process and handle timeout.
            initial_time = time.time() if monitoring else None
 
            while True:

                # exit by process status:
                _process_status = process.is_alive()
                if feedback_level > 2:
                    print('* Process is alive:', _process_status, flush=True)
                if not _process_status:
                    if feedback_level > 2:
                        print(f'* Process exited with code {process.exitcode}.', flush=True)
                    if process.exitcode == 0:
                        break
                    else:
                        raise Exception('Process exited with code %d' % process.exitcode)                    

                # monitoring:
                if monitoring:
                    current_memory = psutil.Process(process.pid).memory_info().rss / 1024 / 1024
                    if feedback_level > 2:
                        print('* Current memory usage:', current_memory, 'MB', flush=True)
                    peak_memory = max(peak_memory, current_memory)

                # break if pipe is full:
                if parent_conn.poll():
                    if feedback_level > 2:
                        print('* Breaking, sub-process pipe is full.', flush=True)
                    break

                # break by timeout:
                if timeout and time.time() - initial_time > timeout:
                    process.terminate()
                    process.join()
                    raise TimeoutError("Process timed out.")

                # sleep for monitoring frequency:
                time.sleep(monitoring_frequency)

            # Receive the result from the pipe
            result = parent_conn.recv()
            if feedback_level > 2:
                print('* Result received.', flush=True)

            # Wait for the process to finish and get the result:
            process.join()
            if feedback_level > 2:
                print('* Process joined.', flush=True)

            # Exact peak memory, sent by a subprocess that completed normally:
            if monitoring and process.exitcode == 0 and parent_conn.poll():
                subprocess_peak_memory = parent_conn.recv()
                if subprocess_peak_memory is not None:
                    peak_memory = max(peak_memory, subprocess_peak_memory)

            # monitoring stats if needed:
            if monitoring:
                final_time = time.time()
                total_time = final_time - initial_time
                if feedback_level > 0:
                    print(feedback_separator, flush=True)
                    print(f'* Total time elapsed: {total_time:.2f} seconds', flush=True)
                    print(f'* Peak memory usage: {peak_memory:.2f} MB', flush=True)
                    print(feedback_separator, flush=True)

            # Raise exceptions received from the subprocess.
            if isinstance(result, Exception):
                raise result

            return result

        # The subprocess finds the original function through the decorated one:
        wrapper._run_in_process_function = func

        return wrapper

    return decorator
