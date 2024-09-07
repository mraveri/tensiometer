"""
This module contains the functions and utilities to compute non-Gaussian
Monte Carlo tension estimators.

The submodule `param_diff` contains the functions and utilities to compute the distribution
of parameter differences from the parameter posterior of two experiments.

The submodule `kde` contains the functions to compute the statistical significance
of a difference in parameters with KDE methods.

This submodule `flow` contains the functions and utilities to compute the statistical significance
of a difference in parameters with normalizing flow methods.

For more details on the method implemented see
`arxiv 1806.04649 <https://arxiv.org/pdf/1806.04649.pdf>`_
and `arxiv 2105.03324 <https://arxiv.org/pdf/2105.03324.pdf>`_.
"""

# parameter difference module import:
from .param_diff import parameter_diff_chain, parameter_diff_weighted_samples

# kde module import:
from .kde import kde_parameter_shift_1D_fft, kde_parameter_shift_2D_fft, kde_parameter_shift

# flow module import:
from .flow import estimate_shift, estimate_shift_from_samples, flow_parameter_shift
