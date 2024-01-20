"""
This module contains all the methods to build synthetic models for posterior distributions.

These models start from samples from a given posterior distribution and 
build machine learning normalizing flow models for the distribution.

The synthetic distribution can then be evaluated at arbitrary points, is differentiable and we can sample from it. 
"""

#
from .synthetic_probability import FlowCallback, flow_from_chain, average_flow_from_chain
