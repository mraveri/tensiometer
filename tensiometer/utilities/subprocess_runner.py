"""
This file contains some utilities that are useful to run a function inside an independent sub-process.

When the subprocess is finished the memory is fully released, which is useful to avoid memory leaks, especially from tensorflow.
"""

###########################################################################################
# Initial imports:

import multiprocessing as mp
from functools import wraps
import psutil
import datetime
import time
import copy

###########################################################################################
# Default Settings:

# These settings define the default behavior of the `run_in_process` decorator.

default_settings = {
    'subprocess': True,         # Whether to run the function in a subprocess.
    'feedback_level': 1,        # Level of feedback provided:
                                # 0 - None, 1 - Minimal, 2 - Medium, 3 - Full.
    'context': 'fork',          # Multiprocessing context: 'fork', 'spawn', or 'forkserver'.
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
# `run_in_process` Decorator:
# This decorator modifies a function to optionally run it in a separate subprocess
# with configurable feedback and monitoring.

def run_in_process(**kwargs):
    """
    Decorator to run a function in a subprocess with optional feedback, monitoring, 
    and timeout capabilities.

    Parameters:
        kwargs (dict): Overrides for default settings. Valid keys are:
            - 'subprocess' (bool): Whether to run the function in a subprocess.
            - 'feedback_level' (int): Feedback verbosity level (0 to 3).
            - 'context' (str): Multiprocessing context ('fork', 'spawn', 'forkserver').
            - 'monitoring' (bool): Enable/disable monitoring.
            - 'monitoring_frequency' (int): Monitoring update frequency in seconds.
            - 'timeout' (int or None): Maximum allowed runtime in seconds.

    Returns:
        function: A wrapped version of the input function with subprocess capabilities.
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

        Parameters:
            func (function): The target function to decorate.

        Returns:
            function: The wrapped function.
        """
        if not subprocess:
            # If not running in a subprocess, return the original function.
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function to run the target function in a subprocess.

            Parameters:
                args: Positional arguments for the target function.
                kwargs: Keyword arguments for the target function.

            Returns:
                Any: The result of the target function.
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

            # Define the target function to be executed in the subprocess.
            def target_function(pipe_conn, *args, **kwargs):
                """
                Execute the target function in a subprocess and send the result back.

                Parameters:
                    pipe_conn: Pipe connection for inter-process communication.
                    args: Positional arguments for the target function.
                    kwargs: Keyword arguments for the target function.
                """
                try:
                    # Execute the function and send the result through the pipe.
                    result = func(*args, **kwargs)
                    pipe_conn.send(result)
                except Exception as e:
                    # Send exceptions through the pipe for error handling.
                    import sys
                    import traceback
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback_string = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                    e.traceback = traceback_string
                    pipe_conn.send(e)
                finally:
                    pipe_conn.close()

            # Set up and start the subprocess.
            ctx = mp.get_context(context)
            parent_conn, child_conn = ctx.Pipe()
            
            # create the process:
            process = ctx.Process(target=target_function, args=(child_conn, *args), kwargs=kwargs)
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
                peak_memory = initial_memory / 1024 / 1024
                    
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

        return wrapper

    return decorator
