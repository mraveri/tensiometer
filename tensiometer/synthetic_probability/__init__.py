"""
This module contains all the methods to build synthetic models for posterior distributions.

These models start from samples from a given posterior distribution and 
build machine learning normalizing flow models for the distribution.

The synthetic distribution can then be evaluated at arbitrary points, is differentiable and we can sample from it. 

Note that documentation is spotty at places and might need to be improved.
"""

# module imports:
from . import synthetic_probability
from . import flow_utilities
from . import flow_profiler
from . import analytic_flow
from . import flow_CPCA

# internal modules:
from . import fixed_bijectors
from . import trainable_bijectors
from . import loss_functions
from . import lr_schedulers
