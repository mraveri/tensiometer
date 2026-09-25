"""
This module contains all the methods to build synthetic models for posterior distributions.

These models start from samples from a given posterior distribution and 
build machine learning normalizing flow models for the distribution.

The synthetic distribution can then be evaluated at arbitrary points, is differentiable and we can sample from it. 

The flows are implemented in `PyTorch <https://pytorch.org>`_. Public methods accept numpy
arrays or tensors and return CPU tensors (call ``.numpy()`` to get arrays).

Precision and device
--------------------

- Precision: float32 by default. Set the environment variable ``TENSIOMETER_PRECISION=float64``
  before starting Python, or call
  :func:`tensiometer.synthetic_probability.tensor_utilities.set_precision` before building any
  flow, to work in float64.
- Device: flows run on the CPU by default. Set ``TENSIOMETER_DEVICE`` to ``cuda``, ``cuda:N``,
  ``mps`` (Apple Silicon GPU) or ``auto``, call
  :func:`tensiometer.synthetic_probability.tensor_utilities.set_device`, or pass ``device=`` to
  :class:`~tensiometer.synthetic_probability.synthetic_probability.FlowCallback`. A flow can be
  moved later with ``flow.to(device)``. MPS does not support float64.
- Performance: the default networks are small, so training is usually fastest on the CPU.
  On an Apple M5 Max, one epoch of a 6-parameter flow trained on 9000 samples took 0.06 s
  (affine) and 0.45 s (spline) on the CPU, and 0.15 s and 0.55 s on the MPS GPU. GPUs pay
  off for much larger networks and batches.

Saving flows
------------

``flow.save(path)`` writes the whole flow to one file and
``FlowCallback.load(path)`` restores it without the chain. ``flow.save(path, mode='light')``
writes a much smaller file that can be evaluated, sampled and profiled but not trained again.
Snapshots are pickles: load them only from trusted sources.

Note that documentation is spotty at places and might need to be improved.
"""
