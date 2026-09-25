"""
Precision, device and conversion helpers shared by the synthetic probability modules.

The floating point precision and the default compute device are process-wide settings:

- precision: ``float32`` (default) or ``float64``, selected at import with the environment
  variable ``TENSIOMETER_PRECISION`` or later with :func:`set_precision`, before any bijector
  or flow is built;
- device: ``cpu`` (default), ``cuda``, ``cuda:N``, ``mps`` or ``auto``, selected with the
  environment variable ``TENSIOMETER_DEVICE`` or with :func:`set_device`. The default device
  is used by the flows (``FlowCallback``, ``AutoregressiveFlow``) built without an explicit
  ``device`` argument. Standalone building blocks follow the torch convention: new tensors,
  bijectors and base distributions live on the CPU unless a device is given, and bijectors
  move their inputs to their own device.

Other modules read the module attributes ``prec`` (torch dtype) and ``np_prec`` (numpy dtype)
through this module at call time (``tensor_utilities.prec``) and never bind them at import.

This module imports only numpy and torch so that every other module can import it.
"""

###############################################################################
# initial imports and set-up:

import os
import tempfile
import warnings

import numpy as np
import torch

###############################################################################
# precision state:

_PRECISION_NAMES = {
    'float32': torch.float32,
    '32': torch.float32,
    'single': torch.float32,
    'float': torch.float32,
    'float64': torch.float64,
    '64': torch.float64,
    'double': torch.float64,
}

_NUMPY_PRECISION = {
    torch.float32: np.float32,
    torch.float64: np.float64,
}

prec = torch.float32
np_prec = np.float32
_precision_locked = False
_cuda_float64_warned = False


def _parse_precision(name):
    """
    Convert a precision specification to a torch floating dtype.

    :param name: ``'float32'``, ``'float64'``, ``'32'``, ``'64'``, ``'single'``, ``'double'``,
        a torch dtype or a numpy dtype.
    :returns: ``torch.float32`` or ``torch.float64``.
    :raises ValueError: if the precision is not supported.
    """
    if isinstance(name, torch.dtype):
        if name in _NUMPY_PRECISION:
            return name
        raise ValueError('Unsupported precision ' + str(name) + '. Use float32 or float64.')
    if isinstance(name, type) or isinstance(name, np.dtype):
        try:
            numpy_name = np.dtype(name).name
        except TypeError:
            numpy_name = None
        if numpy_name in _PRECISION_NAMES:
            return _PRECISION_NAMES[numpy_name]
        raise ValueError('Unsupported precision ' + str(name) + '. Use float32 or float64.')
    key = str(name).strip().lower()
    if key not in _PRECISION_NAMES:
        raise ValueError(
            'Unsupported precision ' + repr(name) + '. Accepted values are: '
            + ', '.join(sorted(_PRECISION_NAMES.keys())))
    return _PRECISION_NAMES[key]


def get_precision():
    """
    Return the active floating point precision.

    :returns: the torch dtype used by all flows (``torch.float32`` or ``torch.float64``).
    """
    return prec


def is_precision_locked():
    """
    Tell whether the precision can still be changed.

    :returns: True once a bijector or flow has been constructed.
    """
    return _precision_locked


def lock_precision():
    """
    Lock the precision. Called by every bijector constructor so that later calls to
    :func:`set_precision` with a different value fail instead of mixing dtypes.
    """
    global _precision_locked
    _precision_locked = True


def set_precision(name):
    """
    Set the process-wide floating point precision.

    Must be called before any bijector or flow is constructed.

    :param name: precision specification: ``'float32'``, ``'float64'``, ``'32'``, ``'64'``,
        ``'single'``, ``'double'``, a torch dtype or a numpy dtype.
    :returns: the new torch dtype.
    :raises RuntimeError: if the precision is locked and ``name`` differs from it.
    :raises ValueError: if the precision is not supported or is float64 with an MPS default device.
    """
    global prec, np_prec
    new_prec = _parse_precision(name)
    if new_prec == prec:
        return prec
    if _precision_locked:
        raise RuntimeError(
            'The precision is locked to ' + str(prec) + ' because bijectors or flows have already been '
            'built. Set TENSIOMETER_PRECISION or call set_precision before building any flow.')
    _check_device_precision(_default_device, new_prec)
    prec = new_prec
    np_prec = _NUMPY_PRECISION[new_prec]
    return prec

###############################################################################
# device state:

_default_device = torch.device('cpu')


def _check_device_precision(device, precision):
    """
    Validate a device and precision combination.

    :param device: torch device.
    :param precision: torch dtype.
    :raises ValueError: for float64 on MPS, which has no float64 support.
    """
    global _cuda_float64_warned
    if device.type == 'mps' and precision == torch.float64:
        raise ValueError('The MPS device does not support float64. Use float32 or a cpu/cuda device.')
    if device.type == 'cuda' and precision == torch.float64 and not _cuda_float64_warned:
        _cuda_float64_warned = True
        warnings.warn('float64 on CUDA devices can be much slower than float32 on consumer GPUs.')


def resolve_device(name=None):
    """
    Convert a device specification to an available torch device.

    :param name: ``None`` (current default), ``'cpu'``, ``'cuda'``, ``'cuda:N'``, ``'mps'``,
        ``'auto'`` (cuda, then mps, then cpu) or a ``torch.device``.
    :returns: the corresponding ``torch.device``, with an explicit index for accelerators
        (``cuda:N``, ``mps:0``) so that it compares equal to the device of tensors.
    :raises ValueError: if the device is unknown or not available on this machine.
    """
    if name is None:
        return _default_device
    if isinstance(name, str) and name.strip().lower() == 'auto':
        if torch.cuda.is_available():
            name = 'cuda'
        elif torch.backends.mps.is_available():
            name = 'mps'
        else:
            return torch.device('cpu')
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError) as exc:
        raise ValueError('Unknown device ' + repr(name)) from exc
    if device.type == 'cpu':
        return device
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('Device ' + str(device) + ' requested but CUDA is not available.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                'Device ' + str(device) + ' requested but only ' + str(torch.cuda.device_count())
                + ' CUDA devices are available.')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        return device
    if device.type == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError('Device mps requested but the MPS backend is not available.')
        return torch.device('mps', 0)
    raise ValueError('Unsupported device type ' + repr(device.type) + '. Use cpu, cuda or mps.')


def get_device():
    """
    Return the default compute device for flows built without an explicit device.

    :returns: a ``torch.device``.
    """
    return _default_device


def set_device(name):
    """
    Set the default compute device for flows built afterwards.

    :param name: device specification, see :func:`resolve_device`.
    :returns: the new default ``torch.device``.
    :raises ValueError: if the device is not available or does not support the active precision.
    """
    global _default_device
    device = resolve_device(name)
    _check_device_precision(device, prec)
    _default_device = device
    return device


def check_device(device):
    """
    Resolve a device and check it against the active precision.

    :param device: device specification or None for the default device.
    :returns: the resolved ``torch.device``.
    :raises ValueError: if the device is not available or does not support the active precision.
    """
    device = resolve_device(device)
    _check_device_precision(device, prec)
    return device


def module_device(module, default=None):
    """
    Return the device of the first parameter or buffer of a module.

    :param module: a ``torch.nn.Module``.
    :param default: value returned when the module holds no tensors.
    :returns: a ``torch.device`` or ``default``.
    """
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return default

###############################################################################
# conversion helpers:


def to_tensor(x, device=None, dtype=None):
    """
    Convert an array-like to a tensor with the active precision.

    Tensors keep their device unless ``device`` is given; other inputs are placed on
    ``device`` or on the CPU. Conversions of tensors are differentiable.

    :param x: numpy array, list, scalar or tensor.
    :param device: target device, None to keep the device of tensors and use the CPU otherwise.
    :param dtype: target dtype, defaults to the active precision.
    :returns: a tensor.
    """
    if dtype is None:
        dtype = prec
    if torch.is_tensor(x):
        if device is None:
            return x.to(dtype=dtype)
        return x.to(device=resolve_device(device), dtype=dtype)
    target = torch.device('cpu') if device is None else resolve_device(device)
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(torch.is_tensor(_x) for _x in x):
        return torch.stack([to_tensor(_x, device=target, dtype=dtype) for _x in x])
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=target)


def to_numpy(x):
    """
    Convert a tensor (on any device, with or without graph) to a numpy array on the host.

    :param x: tensor or array-like.
    :returns: a numpy array.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def run_on_cpu(fn, *tensors):
    """
    Evaluate ``fn`` on the host and move the result back to the device of the first input.

    Used for operators that have no kernel on a device (for example ``ndtri`` on MPS).
    The round trip is differentiable.

    :param fn: callable taking and returning tensors.
    :param tensors: input tensors.
    :returns: output of ``fn`` on the device of the first input.
    """
    device = tensors[0].device
    result = fn(*[tensor.cpu() for tensor in tensors])
    if isinstance(result, (tuple, list)):
        return type(result)(_r.to(device) for _r in result)
    return result.to(device)


# (operator name, device type) pairs known to have no kernel:
_missing_operators = set()


def call_with_cpu_fallback(name, fn, *tensors):
    """
    Evaluate ``fn`` on the device of the inputs, falling back to the host when the
    device has no kernel for it.

    Operator coverage of accelerator backends (in particular MPS) changes between torch
    releases, so the operator is first tried on the device; after a ``NotImplementedError``
    it is remembered as missing and always run with :func:`run_on_cpu`.

    :param name: name of the operator, used to remember missing kernels.
    :param fn: callable taking and returning tensors.
    :param tensors: input tensors.
    :returns: output of ``fn`` on the device of the first input.
    """
    device_type = tensors[0].device.type
    if device_type == 'cpu':
        return fn(*tensors)
    if (name, device_type) in _missing_operators:
        return run_on_cpu(fn, *tensors)
    try:
        return fn(*tensors)
    except NotImplementedError:
        _missing_operators.add((name, device_type))
        return run_on_cpu(fn, *tensors)


def atomic_save(obj, path):
    """
    Save an object with ``torch.save`` without ever leaving a partial file at ``path``.

    The object is written to a temporary file in the same directory and then renamed.

    :param obj: object to save.
    :param path: destination file path.
    """
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    handle, temp_path = tempfile.mkstemp(dir=directory, prefix='.' + os.path.basename(path) + '.', suffix='.tmp')
    os.close(handle)
    try:
        torch.save(obj, temp_path)
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

###############################################################################
# initialization from the environment:


def _init_from_environment():
    """Read ``TENSIOMETER_PRECISION`` and ``TENSIOMETER_DEVICE``."""
    _precision = os.environ.get('TENSIOMETER_PRECISION', None)
    if _precision is not None and _precision.strip() != '':
        set_precision(_precision)
    _device = os.environ.get('TENSIOMETER_DEVICE', None)
    if _device is not None and _device.strip() != '':
        set_device(_device)


_init_from_environment()
