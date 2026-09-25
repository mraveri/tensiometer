"""
This file contains the code to perform profiling on the normalizing flow models.
"""

###############################################################################
# initial imports and set-up:

import gc
import time
import tqdm
import sys
import copy
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from itertools import combinations

from numba import njit
import torch

# getdist imports:
import getdist.mcsamples as mcsamples
import getdist.types as types
from getdist.paramnames import ParamInfo
from getdist.densities import Density1D, Density2D

# tensiometer imports:
from ..utilities import stats_utilities as stutils
from . import autodiff
from . import bijectors as bj
from . import fixed_bijectors as pb
from . import synthetic_probability as sp
from . import tensor_utilities as tu
from .optimizers import batched_minimize

###############################################################################
# utility functions:

# numpy 2 renamed trapz to trapezoid (and later removed trapz):
_trapezoid = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


@njit
def _binned_argmax_1D(bins, vals, num_bins):
    """ 
    Utility function to find index of maximum in a bin.
    Bins is the bin number that every vals falls in (i.e. the output of digitize).
    """
    # initialize vectors:
    out = -np.ones(num_bins, np.int64)  # this is initialized to -1 so we can filter out
    trk = np.empty(num_bins, vals.dtype)
    trk.fill(-np.inf)
    # do the for loop:
    for i in range(len(vals)):
        v = vals[i]
        b = bins[i]
        if v >= trk[b]:
            trk[b] = v
            out[b] = i
    #
    return out


@njit
def _binned_argmax_2D(x_bins, y_bins, vals, num_bins_x, num_bins_y):
    """
    Utility function to find index of maximum in a 2D bin.
    Bins is the bin number that every vals falls in (i.e. the output of digitize).
    """
    # initialize vectors:
    out = -np.ones((num_bins_x, num_bins_y), np.int64)  # this is initialized to -1 so we can filter out
    trk = np.empty((num_bins_x, num_bins_y), vals.dtype)
    trk.fill(-np.inf)
    # do the for loop:
    for i in range(len(vals)):
        bx = x_bins[i]
        by = y_bins[i]
        v = vals[i]
        if v >= trk[bx, by]:
            trk[bx, by] = v
            out[bx, by] = i
    #
    return out


###############################################################################
# minimizer functions


def points_minimizer(func, jac, x0, bounds=None, feedback=0, use_scipy=True, **kwargs):
    """
    Minimize a population of points. Has default options to deal with 32 bit precisions.
    Note that the scipy implementation can be run with boundaries
    while the torch one (:func:`~tensiometer.synthetic_probability.optimizers.batched_minimize`) cannot.
    The scipy branch minimizes one point at a time with numpy functions; the torch branch
    minimizes all the points at once and ``func`` / ``jac`` act on tensors of shape ``(B, d)``.

    :param func: objective function.
    :param jac: gradient of the objective.
    :param x0: initial points, shape ``(B, d)``.
    :param bounds: optional list of ``(min, max)`` bounds (scipy only).
    :param feedback: feedback level.
    :param use_scipy: use scipy (default) or the batched torch minimizer.
    :param kwargs: options of the minimizer.
    :returns: numpy arrays ``success (B,)``, ``min_value (B,)`` and ``min_point (B, d)``.
    """
    if use_scipy:
        # read in options:
        _temp_kwargs = kwargs
        _method = _temp_kwargs.pop('method', 'L-BFGS-B')
        _options = _temp_kwargs.pop('options', {
            'ftol': 1.e-6,
            'gtol': 1.e-05,
            'maxls': 100,
        })
        _use_jac = _temp_kwargs.pop('use_jac', True)
        # feedback:
        if feedback > 2:
            print('      method:', _method)
            print('      options:', _options)
            print('      use_jac:', _use_jac)
            print('      using bounds:', bounds is not None)
        # prepare:
        success, min_value, min_point = [], [], []
        # do the loop:
        for i, _x0 in enumerate(x0):
            # feedback:
            if feedback > 1:
                print('  * sample', i + 1)
            if feedback > 2:
                print('    - initial point', _x0)
                print('    - initial loss function', func(_x0))
            # main minimizer call:
            result = minimize(
                func, x0=_x0,
                jac=jac if _use_jac else None,
                bounds=bounds,
                method=_method, 
                options=_options, 
                **stutils.filter_kwargs(_temp_kwargs, minimize))
            # save results:
            success.append(result.success)
            min_value.append(result.fun)
            min_point.append(result.x)
            # feedback:
            if feedback > 1:
                print('    - Success', result.success)
                print('    - Loss function', result.fun)
                if hasattr(result, 'nfev') and hasattr(result, 'njev'):
                    print('    - Function /Jacobian evaluations', result.nfev, '/', result.njev)
                if not result.success:
                    print('    - Message', result.message)
        # convert to numpy array:
        success = np.array(success)
        min_value = np.array(min_value)
        min_point = np.array(min_point)
    else:
        if bounds is not None:
            print('WARNING: torch minimizer does not support bounds. Ignoring bounds.')
        result = batched_minimize(
            lambda x: (func(x), jac(x)),
            tu.to_tensor(x0, device='cpu'),
            **stutils.filter_kwargs(kwargs, batched_minimize))
        success = tu.to_numpy(result.converged)
        min_value = tu.to_numpy(result.objective_value)
        min_point = tu.to_numpy(result.position)
    #
    return success, min_value, min_point


###############################################################################
# MAP finding method:


def find_flow_MAP(
        flow,
        feedback=1,
        abstract=True,
        num_samples=None,
        num_best_to_follow=10,
        initial_points=None,
        use_scipy=True,
        box_bijector=None,
        **kwargs):
    """
    We want to do precise maximization in a large number of dimensions.
    This is achieved by first sampling. Selecting the best samples.
    Mapping them to abstract coordinates (where we have no bounds)
    and finding the MAP.

    :param flow: flow to maximize.
    :param feedback: feedback level.
    :param abstract: minimize in the abstract (Gaussian) coordinates of the flow.
    :param num_samples: number of samples of the initial random search.
    :param num_best_to_follow: number of best samples that are minimized.
    :param initial_points: optional initial points in parameter space (skips the random search).
    :param use_scipy: use scipy (one point at a time) or the batched torch minimizer.
    :param box_bijector: optional bijector from unbounded coordinates to the prior box.
    :param kwargs: options of the minimizer.
    :returns: best log probability and best point, in parameter space.
    :raises ValueError: if both ``abstract`` and ``box_bijector`` are used, or no minimization succeeded.
    """
    # branch depending on wether initial points are provided:
    if initial_points is None:
        # initialize options:
        if num_samples is None:
            num_samples = 1000 * flow.num_params
        # feedback:
        if feedback > 0:
            print('  number of minimization initial samples =', num_samples)
            print('  number of best points to follow =', num_best_to_follow)
        # find the initial population of points:
        temp_samples = tu.to_numpy(flow.sample(num_samples))
        temp_probs = tu.to_numpy(flow.log_probability(temp_samples))
        # find the best ones:
        _best_indexes = np.argpartition(temp_probs, -num_best_to_follow)[-num_best_to_follow:]
        # select population:
        best_population = temp_samples[_best_indexes]
        # delete samples (these can be heavy):
        del (temp_samples, temp_probs)
    else:
        best_population = initial_points
    # feedback:
    if feedback > 0:
        print('      using abstract coordinates =', abstract)
        print('      using box bijector =', box_bijector is not None)  
    # abstract and box bijector are incompatible:
    if abstract and box_bijector is not None:
        raise ValueError('Cannot use abstract coordinates and box bijector at the same time.')      
    # map to abstract space:
    if abstract:
        best_population = flow.map_to_abstract_coord(flow.cast(best_population))
    elif box_bijector is not None:
        with torch.no_grad():
            best_population = box_bijector.inverse(flow.cast(best_population)).cpu()
    best_population = tu.to_numpy(best_population)
    # objective functions acting on tensors:
    if abstract:

        def func(x):
            return -flow.log_probability_abs(x)

        def jac(x):
            return -flow.log_probability_abs_jacobian(x)
        
    elif box_bijector is not None:
        
        def func(x):
            with torch.no_grad():
                return -flow.log_probability(box_bijector(x))

        def jac(x):
            return autodiff.gradient(
                lambda _x: -flow.log_probability(box_bijector(_x)), autodiff.prepare_input(x), create_graph=False)

    else:

        def func(x):
            return -flow.log_probability(x)

        def jac(x):
            return -flow.log_probability_jacobian(x)

    # do explicit maximization for all the best points:
    if use_scipy:

        def func2(x):
            return tu.to_numpy(func(flow.cast([x])))[0].astype(np.float64)

        def jac2(x):
            return tu.to_numpy(jac(flow.cast([x])))[0].astype(np.float64)
    else:
        func2, jac2 = func, jac
    # now set the ranges:
    if hasattr(flow, 'parameter_ranges') and flow.parameter_ranges is not None:
        bounds = [flow.parameter_ranges[name] for name in flow.param_names]
    else:
        bounds = None
    if box_bijector is not None or abstract or not use_scipy:
        bounds = None
    # now do the minimization
    success, min_value, min_point = points_minimizer(
        func2, jac2, best_population, 
        bounds=bounds,
        feedback=feedback-1, 
        use_scipy=use_scipy, 
        **kwargs)
    # we need to filter for finite values:
    _filter = np.all(np.isfinite(min_point), axis=1)
    _filter = np.logical_and(_filter, np.isfinite(min_value))
    if np.any(np.logical_and(_filter, success)):
        _filter = np.logical_and(_filter, success)
    elif np.any(_filter):
        if feedback > 0:
            print('  * WARNING: no minimization converged, using the best finite point')
    else:
        raise ValueError('find_flow_MAP: no minimization gave a finite result.')
    min_value = min_value[_filter]
    min_point = min_point[_filter]
    # then find best solution and send out:
    _min_idx = np.argmin(min_value)
    _value = -min_value[_min_idx]
    _solution = min_point[_min_idx]
    # map the solution back to parameter space:
    if abstract:
        _solution = tu.to_numpy(flow.map_to_original_coord(flow.cast([_solution])))[0]
    elif box_bijector is not None:
        with torch.no_grad():
            _solution = tu.to_numpy(box_bijector(flow.cast([_solution])))[0]
    # feedback:
    if feedback > 0:
        print('  * best value found =', _value)
        print('  * best point found =', _solution)    
    #
    return _value, _solution


###############################################################################
# hijack getdist plotting:


class posterior_profile_plotter(mcsamples.MCSamples):
    """
    Profile posterior of a flow, exposed as a getdist ``MCSamples`` object so that the
    getdist plotting functions show profiles instead of marginals.

    The profiles are computed by a random search over flow samples followed by optional
    gradient ascent (``pre_polish``) and minimization (``polish``) with scipy
    (``use_scipy=True``, default) or with the batched torch minimizer.

    The profiler can be saved with its flow with :meth:`savePickle` and restored with
    :meth:`loadPickle`. Files written by the TensorFlow version of tensiometer cannot be loaded.
    """

    # default options:
    options = {
               'initialize_cache': False,
               'use_scipy': True,
               'pre_polish': True,
               'polish': True,
               'smoothing': True,
               'box_prior': False,
               'update_MAP': True,
               'update_1D': True,
               'update_2D': True,
               # maximization options:
               'num_minimization_samples': 20000,
               # 1D profile options:               
               'num_points_1D': 64,
               'learning_rate_1D': 0.1,
               'num_gd_interactions_1D': 100,
               # 2D profile options:
               'num_points_2D': 32,
               'learning_rate_2D': 0.1,
               'num_gd_interactions_2D': 100,
               # scipy minimizer options:
               'scipy_method': 'L-BFGS-B',
               'scipy_options': {'ftol': 1.e-6, 'gtol': 1.e-05},
               'scipy_use_jac': True,
               # torch minimizer options:
               'torch_tolerance': 1.e-5,
               'torch_max_iterations': 1000,
               'torch_max_line_search_iterations': 20,
            }

    def __init__(self, flow, feedback=1, **kwargs):
        """
        Initialize the profile posterior plotter.
        Pass a flow and the number of samples to use for base MCSamples.

        The base ``MCSamples`` is built from ``flow.chain_samples`` / ``flow.chain_loglikes``
        when available, otherwise from ``1000 * flow.num_params`` flow samples.

        :param flow: the flow to profile.
        :param feedback: feedback level (default 1); higher values print more.
        :param kwargs: options, see ``posterior_profile_plotter.options``; they update the
            instance options and, if ``initialize_cache`` is ``True``, are forwarded to
            :meth:`update_cache`. Also accepts ``name_tag`` (default ``flow.name_tag + '_profiler'``)
            and the named arguments of the getdist ``MCSamples`` constructor (e.g. ``settings``).
        """
        # copy default options so per-instance updates do not leak
        self.options = copy.deepcopy(self.options)
        # initialize settings:
        self.feedback = feedback
        # initialize internal variables:
        initialize_cache = kwargs.get('initialize_cache', self.options['initialize_cache'])
        name_tag = kwargs.get('name_tag', flow.name_tag + '_profiler')
        # initialize global options:
        self.use_scipy = kwargs.get('use_scipy', self.options['use_scipy'])
        self.pre_polish = kwargs.get('pre_polish', self.options['pre_polish'])
        self.polish = kwargs.get('polish', self.options['polish'])
        self.smoothing = kwargs.get('smoothing', self.options['smoothing'])
        self.box_prior = kwargs.get('box_prior', self.options['box_prior'])
        # feedback:
        if self.feedback > 0:
            print('Flow profiler')
            print('  * use_scipy  =', self.use_scipy)
            print('  * pre_polish =', self.pre_polish)
            print('  * polish     =', self.polish)
            print('  * smoothing  =', self.smoothing)
            print('  * box_prior  =', self.box_prior)
            print('  * parameters =', flow.param_names)
            print('  * feedback   =', self.feedback)
        # initialize the parent with the flow (so we can do margestats)
        if getattr(flow, 'chain_samples', None) is not None:
            _samples = flow.chain_samples
            _loglikes = flow.chain_loglikes
        else:
            _samples = tu.to_numpy(flow.sample(1000 * flow.num_params))
            _loglikes = -tu.to_numpy(flow.log_probability(flow.cast(_samples)))
        super().__init__(
            samples=_samples,
            loglikes=_loglikes,
            names=flow.param_names,
            labels=flow.param_labels,
            ranges=flow.parameter_ranges,
            name_tag=name_tag,
            **stutils.filter_kwargs(kwargs, mcsamples.MCSamples))
        # save the flow so that we can query it:
        self.flow = flow
        # define the empty caches:
        self.reset_cache()
        # initialize options:
        self.options.update(kwargs)
        # call the initial calculation:
        if initialize_cache:
            self.update_cache(**kwargs)
        #
        return None

    def update_cache(self, params=None, **kwargs):
        """
        Initialize the profiler cache. This does all the maximization
        calculations and it's the heavy part of the whole thing.

        Draws the random search population, then (depending on ``update_MAP``, ``update_1D``
        and ``update_2D``) finds the MAP and computes the 1D profiles and the 2D profiles of all
        pairs of ``params``. The random search samples are deleted at the end.

        :param params: list of parameter names (or indices) to profile, defaults to all parameters.
        :param kwargs: options, see ``posterior_profile_plotter.options``; they update the instance
            options and are forwarded to :meth:`sample_profile_population`, :meth:`find_MAP`,
            :meth:`get1DDensityGridData` and :meth:`get2DDensityGridData`.
        :returns: None
        """
        # initial feedback:
        if self.feedback > 0:
            print('  * initializing profiler data.')
            
        # update options:
        self.options.update(kwargs)

        # get default options:
        _update_MAP = kwargs.get('update_MAP', self.options.get('update_MAP'))
        _update_1D = kwargs.get('update_1D', self.options.get('update_1D'))
        _update_2D = kwargs.get('update_2D', self.options.get('update_2D'))

        # draw the initial population of samples:
        if _update_MAP or \
           _update_1D or \
           _update_2D:
            self.sample_profile_population(**kwargs)

        # get parameter names:
        if params is not None:
            indexes = [self._parAndNumber(name)[0] for name in params]
        else:
            indexes = list(range(self.n))

        # initialize best fit:
        if _update_MAP:
            if self.feedback > 0:
                print('  * finding MAP')
            self.find_MAP(**kwargs)

        # initialize 1D profiles:
        if _update_1D:
            if self.feedback > 0:
                print('  * initializing 1D profiles')
            for ind in tqdm.tqdm(indexes, file=sys.stdout, desc='    1D profiles'):
                self.get1DDensityGridData(ind, **kwargs)

        # initialize 2D profiles:
        if _update_2D:
            if self.feedback > 0:
                print('  * initializing 2D profiles')
            # prepare indexes:
            idxs = list(combinations(indexes, 2))
            # run the loop:
            for idx in tqdm.tqdm(idxs, file=sys.stdout, desc='    2D profiles'):
                ind1, ind2 = idx
                self.get2DDensityGridData(ind1, ind2, **kwargs)

        # do clean up:
        del (self.temp_samples)
        del (self.temp_probs)
        self.temp_samples = None
        self.temp_probs = None
        # force memory cleanup:
        gc.collect()
        #
        return None

    def update_cache_iterative(self, params=None, niter=1, **kwargs):
        """
        Initialize the profiler cache accumulating the random search over several populations.

        The 1D and 2D bins are taken from the getdist marginal densities. For ``niter`` rounds a new
        population is drawn with :meth:`sample_profile_population` and the best sample of each bin
        is kept (stored in ``_1d_samples`` / ``_1d_logP`` and ``_2d_samples`` / ``_2d_logP``).
        The profiles are then computed with :meth:`get1DDensityGridData` and
        :meth:`get2DDensityGridData`, which use these accumulated samples as starting points.
        The accumulated samples are kept until :meth:`reset_cache`; parameters (or pairs) they do
        not cover are profiled with a new random search. The MAP is not updated. The random search
        samples are deleted at the end.

        :param params: list of parameter names to profile, defaults to all parameters.
        :param niter: number of random search populations (default 1).
        :param kwargs: options, see ``posterior_profile_plotter.options`` (``update_1D``,
            ``update_2D``, ``num_points_1D``, ``num_points_2D`` and ``num_minimization_samples``
            are read here); forwarded to the getdist ``get1DDensityGridData`` /
            ``get2DDensityGridData`` used for the binning, to :meth:`sample_profile_population`
            and to the profile methods. Unlike :meth:`update_cache` they do not update the
            instance options.
        :returns: None
        """
        # initial feedback:
        if self.feedback > 0:
            print('  * initializing profiler data.')

        # get parameter names:
        if params is None:
            params = [name.name for name in self.paramNames.names]
        indexes = [self._parAndNumber(name)[0] for name in params]

        # initialize 1D profiles:

        if kwargs.get('update_1D', self.options.get('update_1D')):
            self._1d_bins = {}
            # prepare indexes:
            self._1d_name_idx = list(zip(params, indexes))
            self._1d_samples = {}
            self._1d_logP = {}
            if self.feedback > 0:
                print('* initializing 1D profiles')
                print('    - parameters:', [x[0] for x in self._1d_name_idx])
            for name, ind in tqdm.tqdm(self._1d_name_idx, file=sys.stdout, desc='    1D profiles'):
                # call the MCSamples method to have the marginalized density:
                marge_density = super().get1DDensityGridData(
                    name, 
                    fine_bins=kwargs.get('num_points_1D', self.options.get('num_points_1D')), 
                    num_bins=kwargs.get('num_points_1D', self.options.get('num_points_1D')), 
                    **kwargs)
                # get valid bins:
                if self.flow.parameter_ranges is not None:
                    _rang = self.flow.parameter_ranges[name]
                    x_bins = marge_density.x[np.logical_and(_rang[0] <= marge_density.x, _rang[1] >= marge_density.x)]
                else:
                    x_bins = marge_density.x
                self._1d_bins[name] = x_bins
                self._1d_samples[name] = np.nan * np.ones((len(x_bins) - 1, self.n), dtype=np.float32)
                self._1d_logP[name] = np.full(len(x_bins) - 1, -np.inf, dtype=np.float32)

        # initialize 2D profiles:
        if kwargs.get('update_2D', self.options.get('update_2D')):
            self._2d_bins = {}
            # prepare indexes:
            names = list(combinations(params, 2))
            idxs = list(combinations(indexes, 2))
            self._2d_name_idx = list(zip(names, idxs))
            self._2d_samples = {}
            self._2d_logP = {}
            if self.feedback > 0:
                print('* initializing 2D profiles')
                print('    - parameter pairs:', [x[0] for x in self._2d_name_idx])
            # run the loop:
            for name, idx in tqdm.tqdm(zip(names, idxs), file=sys.stdout, desc='    2D profiles'):
                ind1, ind2 = idx
                name1, name2 = name
                # call the MCSamples method to have the marginalized density:
                marge_density = super().get2DDensityGridData(
                    ind1, ind2, fine_bins_2D=kwargs.get('num_points_2D', self.options.get('num_points_2D')), **kwargs)
                # get valid bins:
                if self.flow.parameter_ranges is not None:
                    _rang = self.flow.parameter_ranges[name1]
                    x_bins = marge_density.x[np.logical_and(_rang[0] <= marge_density.x, _rang[1] >= marge_density.x)]
                    _rang = self.flow.parameter_ranges[name2]
                    y_bins = marge_density.y[np.logical_and(_rang[0] <= marge_density.y, _rang[1] >= marge_density.y)]
                else:
                    x_bins = marge_density.x
                    y_bins = marge_density.y
                self._2d_bins[name1, name2] = (x_bins, y_bins)
                self._2d_samples[name1, name2] = np.nan * np.ones((len(x_bins) - 1, len(y_bins) - 1, self.n), dtype=np.float32)
                self._2d_logP[name1, name2] = np.full((len(x_bins) - 1, len(y_bins) - 1), -np.inf, dtype=np.float32)


        for _ in tqdm.trange(niter, desc='iterative sampling'):
            # sample
            self.sample_profile_population(**kwargs)
            # do 1d
            if kwargs.get('update_1D', self.options.get('update_1D')):
                for name, idx in self._1d_name_idx:
                    # protect for samples inside bins:
                    x_bins = self._1d_bins[name]
                    _filter_range = np.logical_and(
                        self.temp_samples[:, idx] > np.amin(x_bins), self.temp_samples[:, idx] < np.amax(x_bins))
                    
                    # first find best samples in the bin:
                    _indexes = np.digitize(self.temp_samples[_filter_range, idx], x_bins) - 1
                    _max_bins_idx = _binned_argmax_1D(_indexes, self.temp_probs[_filter_range], len(x_bins) - 1)
                    
                    # check bins that have points
                    _filter_valid_bins = _max_bins_idx >= 0  
                    
                    # get indices of best points
                    _max_valid_bins_idx = _max_bins_idx[_filter_valid_bins]  
                    
                    # and their logP value
                    _max_valid_bins_logP = self.temp_probs[_filter_range][_max_valid_bins_idx]
                    
                    # check where new points are better than old ones
                    _filter_valid_bins_update = self._1d_logP[name][
                        _filter_valid_bins] < _max_valid_bins_logP  
                    
                    # update best logP values
                    self._1d_logP[name][_filter_valid_bins] = np.where(
                        _filter_valid_bins_update, _max_valid_bins_logP, self._1d_logP[name][_filter_valid_bins])  
                    
                    # update best samples
                    self._1d_samples[name][_filter_valid_bins] = np.where(
                        _filter_valid_bins_update[:, None],
                        copy.deepcopy(self.temp_samples[_filter_range][_max_valid_bins_idx]),
                        self._1d_samples[name][_filter_valid_bins])
                
            
            if kwargs.get('update_2D', self.options.get('update_2D')):
                for names, idxs in self._2d_name_idx:
                    idx1, idx2 = idxs
                    name1, name2 = names

                    # protect for samples inside bins:
                    x_bins, y_bins = self._2d_bins[names]
                    _filter_range_x = np.logical_and(
                        self.temp_samples[:, idx1] > np.amin(x_bins), self.temp_samples[:, idx1] < np.amax(x_bins))
                    _filter_range_y = np.logical_and(
                        self.temp_samples[:, idx2] > np.amin(y_bins), self.temp_samples[:, idx2] < np.amax(y_bins))
                    _filter_range = np.logical_and(_filter_range_x, _filter_range_y)

                    # first find best samples in the bin:
                    _indexes_x = np.digitize(self.temp_samples[_filter_range, idx1], x_bins) - 1
                    _indexes_y = np.digitize(self.temp_samples[_filter_range, idx2], y_bins) - 1
                    _max_bins_idx = _binned_argmax_2D(
                        _indexes_x, _indexes_y,
                        self.temp_probs[_filter_range], len(x_bins) - 1, len(y_bins) - 1)
                    
                    # check bins that have points
                    _filter_valid_bins = _max_bins_idx >= 0  
                    
                    # get indices of best points
                    _max_valid_bins_idx = _max_bins_idx[_filter_valid_bins]  
                    
                    # and their logP value
                    _max_valid_bins_logP = self.temp_probs[_filter_range][_max_valid_bins_idx]
                    
                    # check where new points are better than old ones
                    _filter_valid_bins_update = self._2d_logP[names][
                        _filter_valid_bins] < _max_valid_bins_logP  
                    
                    # update best logP values
                    self._2d_logP[names][_filter_valid_bins] = np.where(
                        _filter_valid_bins_update, _max_valid_bins_logP, self._2d_logP[names][_filter_valid_bins])  
                    
                    # update best samples
                    self._2d_samples[names][_filter_valid_bins] = np.where(
                        _filter_valid_bins_update[:, None],
                        copy.deepcopy(self.temp_samples[_filter_range][_max_valid_bins_idx]),
                        self._2d_samples[names][_filter_valid_bins])
                    

        # initialize 1D profiles:
        if kwargs.get('update_1D', self.options.get('update_1D')):
            if self.feedback > 0:
                print('  * initializing 1D profiles')
            for _, ind in tqdm.tqdm(self._1d_name_idx, file=sys.stdout, desc='    1D profiles'):
                self.get1DDensityGridData(ind, **kwargs)
                
        # initialize 2D profiles:
        if kwargs.get('update_2D', self.options.get('update_2D')):
            if self.feedback > 0:
                print('  * initializing 2D profiles')
            # run the loop:
            for _, idxs in tqdm.tqdm(self._2d_name_idx, file=sys.stdout, desc='    2D profiles'):
                ind1, ind2 = idxs
                self.get2DDensityGridData(ind1, ind2, **kwargs)

        # do clean up:
        del(self.temp_samples)
        del(self.temp_probs)
        self.temp_samples = None
        self.temp_probs = None
        # force memory cleanup:
        gc.collect()
        #
        return None

    def reset_cache(self):
        """
        Delete and re-initialize profile posterior (empty) caches, including the samples
        accumulated by :meth:`update_cache_iterative`.
        """
        # temporary storage for samples for minimization. These can be heavy and should not be stored.
        if hasattr(self, 'temp_samples'):
            del (self.temp_samples)
        if hasattr(self, 'temp_probs'):
            del (self.temp_probs)
        self.temp_samples = None
        self.temp_probs = None
        # best fit storage:
        self.bestfit = None
        self.likeStats = None
        self.flow_MAP = None
        self.flow_MAP_logP = None
        # profile posterior storage:
        if hasattr(self, 'profile_density_1D'):
            del (self.profile_density_1D)
        if hasattr(self, 'profile_density_2D'):
            del (self.profile_density_2D)
        self.profile_density_1D = dict()
        self.profile_density_2D = dict()
        # samples accumulated by the iterative search:
        for _name in ['_1d_bins', '_1d_name_idx', '_1d_samples', '_1d_logP',
                      '_2d_bins', '_2d_name_idx', '_2d_samples', '_2d_logP']:
            if hasattr(self, _name):
                delattr(self, _name)
        # force memory cleanup:
        gc.collect()
        #
        return None

    def sample_profile_population(self, **kwargs):
        """
        Generate the initial samples for minimzation.
        When calling this function the samples are cached (and can be heavy)

        Sets ``temp_samples`` ``(N, D)``, ``temp_probs`` ``(N,)`` (flow log probability),
        ``temp_cov`` and ``temp_inv_cov``. With ``box_prior`` only samples inside the prior box
        are kept and the covariance is computed in the unbounded coordinates.

        :param kwargs: options; only ``num_minimization_samples`` (number of samples ``N``,
            defaults to the instance option) is used.
        :returns: None
        """
        if self.feedback > 1:
            print('    * generating samples for profiler')
        # settings:
        num_minimization_samples = kwargs.get('num_minimization_samples', self.options.get('num_minimization_samples'))
        # feedback:
        if self.feedback > 1:
            print('    - number of random search samples =', num_minimization_samples)
        # initial population of points:
        if self.feedback > 1:
            print('    - sampling the distribution')
        t0 = time.time()
        # sample:
        self.temp_samples = tu.to_numpy(self.flow.sample(num_minimization_samples))
        # calculate probability:
        self.temp_probs = tu.to_numpy(self.flow.log_probability(self.temp_samples))
        # process the samples:
        if self.box_prior:
            _box_bijector = self._get_masked_box_bijector()
            with torch.no_grad():
                temp_samples = tu.to_numpy(_box_bijector.inverse(self.temp_samples))
            _temp_filter = np.all(np.isfinite(temp_samples), axis=1)
            if self.feedback > 1:
                print('    - number of random search samples inside box =', np.sum(_temp_filter))
            _temp_filter = np.arange(len(_temp_filter))[_temp_filter]
            self.temp_samples = self.temp_samples[_temp_filter]
            self.temp_probs = self.temp_probs[_temp_filter]
            self.temp_cov = np.atleast_2d(np.cov(temp_samples[_temp_filter, :].T))
        else:
            self.temp_cov = np.atleast_2d(np.cov(self.temp_samples.T))
        self.temp_inv_cov = np.linalg.inv(self.temp_cov)

        # feedback:
        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken to sample the distribution {0:.4g} (s)'.format(t1))
        #
        return None

    def _get_masked_box_bijector(self, mask=None):
        """
        If we have ranges get bijector + inverse that sends input unbounded coordinates in box coordinates and vice-versa
        Mask means that some coordinates will not be present to account for some coordinates fixed.
        """
        # get names (with mask possibly):
        if mask is None:
            _names = [name for name in self.flow.param_names]
        else:
            _names = [name for i, name in enumerate(self.flow.param_names) if mask[i] > 0]
        # define prior bound bijector to optimize in a space without boundaries:
        if self.flow.parameter_ranges is not None:
            temp_ranges = [{
                'lower': self.flow.cast(self.flow.parameter_ranges[name][0]),
                'upper': self.flow.cast(self.flow.parameter_ranges[name][1]),
                'mode': 'uniform'
            } for name in _names]
            bound_bijector = pb.prior_bijector_helper(temp_ranges)
        else:
            bound_bijector = bj.Identity()
        #
        return bound_bijector

    def find_MAP(self, x0=None, randomize=True, num_best_to_follow=100, abstract=False, **kwargs):
        """
        Find the flow MAP. Strategy is sample and then minimize.

        The result is cached in ``flow_MAP`` / ``flow_MAP_logP`` and used to initialize the
        getdist best fit (see :meth:`getBestFit`).

        :param x0: optional initial points in parameter space, shape ``(B, D)``; if given the
            initial search is skipped.
        :param randomize: if ``True`` (default) start from the ``num_best_to_follow`` best points of
            the random search population (drawn with :meth:`sample_profile_population` if not cached),
            otherwise from the best points of the base ``MCSamples`` samples.
        :param num_best_to_follow: number of initial points that are minimized (default 100).
        :param abstract: minimize in the abstract (Gaussian) coordinates of the flow (default ``False``).
        :param kwargs: options; ``scipy_method``, ``scipy_options`` and ``scipy_use_jac`` (scipy)
            or ``torch_tolerance``, ``torch_max_iterations`` and ``torch_max_line_search_iterations``
            (torch) configure the minimizer, ``num_minimization_samples`` is used when sampling.
            The remaining ones are forwarded to :func:`find_flow_MAP`.
        :returns: best log probability and best point ``(D,)``, in parameter space.
        :raises ValueError: if no minimization gave a finite result (from :func:`find_flow_MAP`).
        """
        # if not cached redo the calculation:
        if self.feedback > 1:
            print('    * finding global best fit')
        # search initial point if not passed:
        if x0 is not None:
            # samples:
            initial_population = self.flow.cast(x0)
        else:
            # initial points search:
            if randomize:
                # feedback:
                if self.feedback > 1:
                    print('    - doing initial randomized search')
                t0 = time.time()

                # check that we have cached samples, otherwise generate:
                if self.temp_samples is None:
                    self.sample_profile_population(**kwargs)

                # find the best ones:
                _best_indexes = np.argpartition(self.temp_probs, -num_best_to_follow)[-num_best_to_follow:]
                # get best samples:
                initial_population = self.temp_samples[_best_indexes]

                # feedback:
                t1 = time.time() - t0
                if self.feedback > 1:
                    print('    - time taken for random initial selection {0:.4g} (s)'.format(t1))
            else:
                # find the best ones:
                _best_indexes = np.argpartition(self.loglikes, num_best_to_follow)[:num_best_to_follow]
                # get best samples:
                initial_population = self.flow.cast(self.samples[_best_indexes, :])
        # feedback:
        if self.feedback > 1:
            if self.use_scipy:
                print('    - doing minimization (scipy)')
            else:
                print('    - doing minimization (torch)')
        t0 = time.time()
        # prepare kwargs:
        _temp_kwargs = kwargs
        if self.use_scipy:
            _temp_kwargs['options'] = kwargs.pop('scipy_options', self.options.get('scipy_options'))
            _temp_kwargs['method'] = kwargs.pop('scipy_method', self.options.get('scipy_method'))
            _temp_kwargs['use_jac'] = kwargs.pop('scipy_use_jac', self.options.get('scipy_use_jac'))
        else:
            _temp_kwargs.update(self._minimizer_options(kwargs))
        # call population minimizer:
        _value, _solution = find_flow_MAP(
            self.flow,
            feedback=self.feedback - 2,
            abstract=abstract,
            initial_points=initial_population,
            use_scipy=self.use_scipy,
            box_bijector=self._get_masked_box_bijector() if self.box_prior else None,
            **_temp_kwargs)
        # feedback:
        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken for minimization {0:.4g} (s)'.format(t1))

        # cache results:
        self.flow_MAP = _solution
        self.flow_MAP_logP = _value
        # initialize getdist things:
        self._initialize_bestfit()
        #
        return _value, _solution

    def getLikeStats(self, profile_lims=True):
        """
        Get likelihood statistics

        Overrides the getdist ``MCSamples.getLikeStats`` method, computing the statistics from the
        flow MAP and the random search population (drawn if not cached). As in getdist, the
        log-likelihood statistics refer to minus the (flow) log probability.

        :param profile_lims: if ``True`` (default) the N-dimensional limits of each parameter are
            obtained from the likelihood ratio of its 1D profile, otherwise from the region of the
            random search samples, as in getdist.
        :returns: getdist ``LikeStats`` object, also cached in ``likeStats``.
        :raises ValueError: if the MAP has not been computed (call :meth:`find_MAP` first).
        """
        return self._initialize_likestats(profile_lims=profile_lims)
            
    def _initialize_likestats(self, profile_lims=True):
        """
        Initialize likestats
        """
        # check if we can run:
        if self.flow_MAP_logP is None:
            raise ValueError('_initialize_likestats can only run after MAP finder')
        # initialize likestats, with the getdist convention that loglikes are minus the log probability:
        m = types.LikeStats()
        best_loglike = -self.flow_MAP_logP
        m.logLike_sample = best_loglike
        # check that we have cached samples, otherwise generate:
        if self.temp_samples is None:
            self.sample_profile_population()
        # get samples and loglikes from flow:
        _temp_samples = self.temp_samples
        _temp_loglikes = -np.asarray(self.temp_probs, dtype=np.float64)
        # compute likelihood statistics as in getdist:
        m.meanLogLike = np.mean(_temp_loglikes)
        m.complexity = 2 * (m.meanLogLike - best_loglike)
        m.logMeanInvLike = np.log(np.mean(np.exp(_temp_loglikes - best_loglike))) + best_loglike
        m.logMeanLike = -np.log(np.mean(np.exp(-(_temp_loglikes - best_loglike)))) + best_loglike
        m.varLogLike = np.var(_temp_loglikes)
        m.names = self.paramNames.names

        if profile_lims:

            ncontours = len(self.contours)
            for j, par in enumerate(self.paramNames.names):
                # get the 1D profile:
                _prof = self.get1DDensityGridData(j)
                par.ND_limit_bot = np.empty(ncontours)
                par.ND_limit_top = np.empty(ncontours)
                # get minimum and maximum:
                _min_val = par.limmin if hasattr(par, 'limmin') else _prof.x[0]
                _max_val = par.limmax if hasattr(par, 'limmax') else _prof.x[-1]
                # get the likelihood ratio contours:                
                for i, cont in enumerate(self.contours):
                    # initialize:
                    _lim_bot = _min_val
                    _lim_top = _max_val
                    # prepare the likelihood ratio:
                    _filter = _prof.P > 0
                    _log_P = np.log(_prof.P[_filter])
                    _temp_x = _prof.x[_filter]
                    # likelihood ratio: 2 Delta log P follows a chi2 with one degree of freedom:
                    _temp_y = _log_P.max() - _log_P - 0.5 * scipy.stats.chi2.ppf(cont, 1)
                    # get the limits:
                    zero_crossings = np.where(np.diff(np.sign(_temp_y)))[0]
                    # discriminate several cases:
                    if len(zero_crossings) == 1:
                        if zero_crossings[0] != 0:
                            # detect if growing or decreasing:
                            if _temp_y[zero_crossings[0]] - _temp_y[zero_crossings[0] - 1] < 0:
                                _lim_bot = _temp_x[zero_crossings[0]]
                            else:
                                _lim_top = _temp_x[zero_crossings[0]]
                    elif len(zero_crossings) > 1:
                        _lim_bot = _temp_x[zero_crossings[0]]
                        _lim_top = _temp_x[zero_crossings[-1]]
                    # save limits:
                    par.ND_limit_bot[i] = _lim_bot
                    par.ND_limit_top[i] = _lim_top
                                        
                par.bestfit_sample = self.flow_MAP[j]
                    
        else:

            # get N-dimensional confidence region, standard getdist code
            indexes = _temp_loglikes.argsort()
            _num_samples = len(indexes)
            cumsum = np.cumsum(np.ones(_num_samples))
            ncontours = len(self.contours)
            m.ND_contours = np.searchsorted(cumsum, _num_samples * self.contours[0:ncontours])

            for j, par in enumerate(self.paramNames.names):
                par.ND_limit_bot = np.empty(ncontours)
                par.ND_limit_top = np.empty(ncontours)
                for i, cont in enumerate(m.ND_contours):
                    region = _temp_samples[indexes[:cont], j]
                    par.ND_limit_bot[i] = np.min(region)
                    par.ND_limit_top[i] = np.max(region)
                par.bestfit_sample = self.flow_MAP[j]

        # save out:
        self.likeStats = m
        #
        return self.likeStats

    def _initialize_bestfit(self):
        """
        Initialize the best fit type
        """
        # check if we can run:
        if self.flow_MAP_logP is None:
            raise ValueError('_initialize_bestfit can only run after MAP finder')
        # initialize best fit:
        bf = types.BestFit(max_posterior=True)
        bf.weight = 1.0
        self.logLike = -self.flow_MAP_logP
        self.chiSquareds = []
        # cycle trough parameters:
        for j, par in enumerate(self.paramNames.names):
            param = ParamInfo()
            param.isFixed = False
            param.isDerived = False
            param.number = j
            param.best_fit = self.flow_MAP[j]
            param.name = par.name
            param.label = par.label
            # save out:
            bf.names.append(param)
        # save:
        self.bestfit = bf
        #
        return None

    def getBestFit(self, max_posterior=True):
        """
        Override standard behavior to return cached result

        Overrides the getdist ``MCSamples.getBestFit`` method, returning the best fit built from the
        flow MAP instead of reading it from a file.

        :param max_posterior: ignored, kept for compatibility with the getdist signature.
        :returns: getdist ``BestFit`` object.
        :raises ValueError: if the MAP has not been computed (call :meth:`find_MAP` first).
        """
        # check if we can run:
        if self.bestfit is None:
            raise ValueError('getBestFit needs a cached MAP. Call find_MAP first.')
        #
        return self.bestfit

    def precompute_1D(self, params, **kwargs):
        """
        Precompute profiles for given params.

        :param params: list of parameter names (or indices).
        :param kwargs: options forwarded to :meth:`get1DDensityGridData`.
        :returns: None
        """
        for name in params:
            self.get1DDensityGridData(name, **kwargs)
        #
        return None
    
    def normalize(self, by='max', **kwargs):
        """
        Normalize the cached profile posterior.

        :param by: ``'max'`` (default) to normalize the maximum to one, or ``'integral'``.
        :param kwargs: options forwarded to the getdist ``Density1D.normalize`` /
            ``Density2D.normalize`` methods (e.g. ``in_place``).
        :returns: None
        """
        # normalize 1D profiles:
        for key in self.profile_density_1D.keys():
            self.profile_density_1D[key].normalize(by=by, **kwargs)
        # normalize 2D profiles:
        for key in self.profile_density_2D.keys():
            for key2 in self.profile_density_2D[key].keys():
                self.profile_density_2D[key][key2].normalize(by=by, **kwargs)
        #
        return None

    def get1DDensityGridData(self, name, num_points_1D=64, **kwargs):
        """
        Compute 1D profile posteriors and return it as a grid density data
        for plotting and analysis.
        Note that this ensures that margestats are from the profile.

        Overrides the getdist ``MCSamples.get1DDensityGridData`` method. Results are cached in
        ``profile_density_1D``. The best random search sample of each bin (or, for the parameters it
        covered, the samples accumulated by :meth:`update_cache_iterative`) is polished by gradient
        ascent (``pre_polish``) and minimization (``polish``), as set at construction.
        For a flow with a single parameter the profile is the density, evaluated directly on the grid.

        :param name: name or index of the parameter.
        :param num_points_1D: number of points in the 1D profile (default 64).
        :param kwargs: options; ``num_minimization_samples``, ``learning_rate_1D``,
            ``num_gd_interactions_1D``, ``smooth_scale_1D`` and the scipy / torch minimizer options
            (see ``posterior_profile_plotter.options``) are read; the remaining ones are forwarded
            to ``scipy.optimize.minimize`` (filtered to its arguments) when polishing with scipy.
        :returns: getdist ``Density1D`` normalized to its maximum, with the additional attributes
            ``maximum`` (maximum log probability), ``profile_subspace`` (profile points ``(M, D)``)
            and ``profile_smoothing_scale_1D``; None if the parameter is not found.
        """
        # get the number of the parameter:
        idx, par_name = self._parAndNumber(name)
        
        # check:
        if idx is None:
            return None

        # look for cached results:
        if idx in self.profile_density_1D.keys():
            return self.profile_density_1D[idx]

        # if not cached redo the calculation:
        if self.feedback > 1:
            print('')
            print('    * calculating the 1D profile for:', par_name.name, 'with index', idx)

        # call the MCSamples method to have the marginalized density:
        marge_density = super().get1DDensityGridData(name, 
                                                     fine_bins=num_points_1D, # these have to be kept the same
                                                     num_bins=num_points_1D, # these have to be kept the same
                                                     smooth_scale_1D=10.0, # this should not matter here
                                                     )
        
        # with a single parameter the profile is the density itself:
        if self.flow.num_params == 1:
            return self._profile_1D_on_grid(idx, marge_density, num_points_1D)

        if hasattr(self, '_1d_logP') and par_name.name in self._1d_logP:

            _result_logP = self._1d_logP[par_name.name]
            _filter_finite = np.isfinite(_result_logP)
            _result_logP = _result_logP[_filter_finite]
            _flow_full_samples = self._1d_samples[par_name.name][_filter_finite,:]

        else:
            
            # check that we have cached samples, otherwise generate:
            if self.temp_samples is None:
                self.sample_profile_population(**kwargs)
                       
            # randomized maximizer algorithm:
         
            # feedback:
            if self.feedback > 1:
                print('    - doing initial randomized search')
            t0 = time.time()

            # get valid bins:
            if self.flow.parameter_ranges is not None:
                _rang = self.flow.parameter_ranges[par_name.name]
                x_bins = marge_density.x[np.logical_and(_rang[0] <= marge_density.x, _rang[1] >= marge_density.x)]
            else:
                x_bins = marge_density.x

            # we need to re-draw the grid since it might have been changed by getdist:
            x_bins = np.linspace(np.min(x_bins), np.max(x_bins), num_points_1D + 1)

            # protect for samples inside bins:
            _valid_filter = np.logical_and(
                self.temp_samples[:, idx] > np.amin(x_bins), self.temp_samples[:, idx] < np.amax(x_bins))

            # first find best samples in the bin:
            _indexes = np.digitize(self.temp_samples[_valid_filter, idx], x_bins)
            _max_idx = _binned_argmax_1D(_indexes, self.temp_probs[_valid_filter], len(x_bins))
            _max_idx = _max_idx[_max_idx >= 0]
            # get the global (un-filtered indexes):
            _max_idx = np.arange(len(self.temp_samples))[_valid_filter][_max_idx]
            # set data:
            _result_logP = self.temp_probs[_max_idx]
            _flow_full_samples = self.temp_samples[_max_idx]
            
            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for random algorithm {0:.4g} (s)'.format(t1))     
        
        if self.feedback > 1:
            print('    - number of 1D bins', num_points_1D)
            print('    - number of empty/filled 1D bins', num_points_1D - len(_result_logP), '/', len(_result_logP))
                                    
        # prepare mask:
        _mask = np.ones(self.flow.num_params)
        _mask[idx] = 0

        # gradient descent polishing:
        if self.pre_polish:

            # feedback:
            if self.feedback > 1:
                print('    - doing gradient descent pre-polishing')
            t0 = time.time()

            # do the iterations:
            _learning_rate = kwargs.get('learning_rate_1D', self.options.get('learning_rate_1D'))
            _num_iterations = kwargs.get('num_gd_interactions_1D', self.options.get('num_gd_interactions_1D'))
            _ensemble = copy.deepcopy(_flow_full_samples)
            _ensemble, temp_probs, num_moving, num_iter = self._masked_gradient_ascent(
                learning_rate=_learning_rate, num_iterations=_num_iterations, ensemble=_ensemble, mask=_mask)
            _filter = temp_probs > _result_logP
            _result_logP = np.where(_filter, temp_probs, _result_logP)
            _flow_full_samples = np.where(_filter[:, None], _ensemble, _flow_full_samples)
            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for gradient descents {0:.4g} (s)'.format(t1))
                print(
                    '      at the end of descent after', num_iter, 'steps', num_moving,
                    'samples were still beeing optimized.')
                if np.sum(_filter) < len(_result_logP):
                    print('      gradient descents did not improve', len(_result_logP) - np.sum(_filter), 'points')

        # branch for minimizer polishing:
        if self.polish:

            t0 = time.time()

            # scipy algorithm:
            if self.use_scipy:

                # polishing with scipy minimizer:
                if self.feedback > 1:
                    print('    - doing minimization polishing (scipy)')

                # compute bounds:
                if self.flow.parameter_ranges is not None:
                    _temp_ranges = [self.flow.parameter_ranges[name] for name in self.flow.param_names]
                    del _temp_ranges[idx]
                else:
                    _temp_ranges = None

                # read in options:
                _temp_kwargs = kwargs
                _method = _temp_kwargs.pop('scipy_method', self.options.get('scipy_method'))
                _options = _temp_kwargs.pop('scipy_options', self.options.get('scipy_options'))
                _use_jac = _temp_kwargs.pop('scipy_use_jac', self.options.get('scipy_use_jac'))
                if self.feedback > 2:
                    print('      method:', _method)
                    print('      options:', _options)
                    print('      use_jac:', _use_jac)

                # polish:
                _polished_ensemble, _polished_logP = [], []
                success = 0
                for _samp, _val in zip(_flow_full_samples, _result_logP):
                    _initial_x = np.delete(_samp, idx)
                    x0 = _samp[idx]

                    def temp_func(x):
                        _x = np.insert(x, idx, x0)
                        return tu.to_numpy(-self.flow.log_probability(self.flow.cast([_x]))[0]).astype(np.float64)

                    def temp_jac(x):
                        _x = np.insert(x, idx, x0)
                        _jac = tu.to_numpy(-self.flow.log_probability_jacobian(self.flow.cast([_x]))[0]).astype(np.float64)
                        return np.delete(_jac, idx)
                    if not _use_jac:
                        temp_jac = None

                    result = minimize(
                        temp_func,
                        x0=_initial_x,
                        jac=temp_jac,
                        bounds=_temp_ranges,
                        method=_method,
                        options=_options,
                        **stutils.filter_kwargs(_temp_kwargs, minimize))
                    
                    if -result.fun > _val:
                        _polished_logP.append(-result.fun)
                        _polished_ensemble.append(np.insert(result.x, idx, x0))
                        success += 1
                    else:
                        _polished_logP.append(_val)
                        _polished_ensemble.append(_samp)

                    if not result.success:
                        if self.feedback > 2:
                            print(result.message)

                if success < len(_result_logP) and self.feedback > 1:
                    print('      Warning minimization failed for ', len(_result_logP) - success, 'points')
                    print('      but ', success, 'samples were still better.')

                _result_logP = np.array(_polished_logP)
                _flow_full_samples = np.array(_polished_ensemble)

            else:

                # polishing with torch minimizer:
                if self.feedback > 1:
                    print('    - doing minimization polishing (torch)')

                _result_logP, _flow_full_samples = self._torch_polish(
                    _flow_full_samples, _result_logP, [idx], _mask, **kwargs)

            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for polishing {0:.4g} (s)'.format(t1))

        # now 1D interpolate on a fixed grid:
        if self.feedback > 1:
            print('    - doing interpolation on regular grid')
        t0 = time.time()

        # convert to numpy array:
        _result_logP = tu.to_numpy(_result_logP)
        _flow_full_samples = tu.to_numpy(_flow_full_samples)

        # now interpolate:
        _max_log_P = np.amax(_result_logP)
        _temp_interp = interp1d(
            _flow_full_samples[:, idx], np.exp(_result_logP - _max_log_P), kind='cubic', bounds_error=False, fill_value=0.0)
        _temp_x = np.linspace(marge_density.view_ranges[0], marge_density.view_ranges[1], num_points_1D + 1)
        _temp_P = _temp_interp(_temp_x)

        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken for interpolation {0:.4g} (s)'.format(t1))

        # smooth the results:
        if self.smoothing:
            if self.feedback > 1:
                print('    - smoothing results')
            t0 = time.time()

            # get analysis for the parameter:
            par = self._initParamRanges(idx, None)
            # get overall smoothing factor:
            _smooth_scale_1D = kwargs.get('smooth_scale_1D', self.smooth_scale_1D)
            if _smooth_scale_1D <= 0.:
                _smooth_scale_1D = 0.2
            # get smoothing factor:
            _smoothing_sigma = _smooth_scale_1D * len(_temp_x) / (_temp_x[-1] - _temp_x[0]) * par.err
            # do the smoothing:
            _temp_P = gaussian_filter1d(_temp_P, _smoothing_sigma, mode='reflect')

            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for smoothing {0:.4g} (s)'.format(t1))

        # save out smoothing scale:
        if self.smoothing:
            _smoothing_scale = _smoothing_sigma
        else:
            _smoothing_scale = None
        # initialize and cache the density:
        return self._save_1D_profile(idx, _temp_x, _temp_P, _max_log_P, marge_density.view_ranges,
                                     _flow_full_samples, _smoothing_scale)
    
    def get_1d_profile_variance(self, param, normalize_by='integral', **kwargs):
        """
        In case we are using an average flow we can calculate the variance of the profile

        The log probability of each flow of the average is evaluated on the profile points,
        interpolated (and smoothed as the profile) on the profile grid, normalized, and the
        weighted standard deviation across flows is returned.

        :param param: name or index of the parameter.
        :param normalize_by: ``'integral'`` (default) or ``'max'``, normalization of the profiles.
        :param kwargs: options forwarded to :meth:`get1DDensityGridData`.
        :returns: grid ``x``, normalized profile and its standard deviation, each of shape ``(M,)``.
        :raises ValueError: if the flow is not an ``average_flow``, ``normalize_by`` is not valid
            or ``param`` is not found.
        """
        # check that the flow is an average flow:
        if not isinstance(self.flow, sp.average_flow):
            raise ValueError('get_1d_profile_variance can only be used with an average flow.')
        if normalize_by not in ('integral', 'max'):
            raise ValueError('normalize_by should be either integral or max')
        # get parameter index and density:
        _idx, _ = self._parAndNumber(param)
        if _idx is None:
            raise ValueError('Parameter ' + str(param) + ' not found in the profiler parameters.')
        _density = self.get1DDensityGridData(param, **kwargs)
        # get the profile data:        
        _profile_subspace = _density.profile_subspace
        _x = _density.x
        # calculate log probabilities on the profile subspace:
        _log_probabilities, _probabilities = [], []
        for _f in self.flow.flows:
            # calculate the log probability on the profile subspace:
            _temp_log_P = tu.to_numpy(_f.log_probability(_f.cast(_profile_subspace)))
            # interpolate and evaluate on the density grid:
            _max_log_P = np.amax(_temp_log_P)
            _temp_interp = interp1d(_profile_subspace[:, _idx], 
                                    np.exp(_temp_log_P - _max_log_P), kind='cubic', bounds_error=False, fill_value=0.0)
            _temp_P = _temp_interp(_x)
            # if the profile is smoothed apply the same smoothing:
            if hasattr(_density, 'profile_smoothing_scale_1D') and \
                _density.profile_smoothing_scale_1D is not None:
                 _temp_P = gaussian_filter1d(_temp_P, _density.profile_smoothing_scale_1D, mode='reflect')
            # normalize:
            if normalize_by == 'integral':
                _temp_P = _temp_P / _trapezoid(_temp_P, _x)
            else:
                _temp_P = _temp_P / np.amax(_temp_P)
            # save out probabilities:
            _probabilities.append(_temp_P)
        # calculate the variance of probability (taking into account flow weights):
        _flow_weights = tu.to_numpy(self.flow.weights)
        _temp_average = np.average(_probabilities, axis=0, weights=_flow_weights)
        _temp_variance = np.average((_probabilities-_temp_average)**2, axis=0, weights=_flow_weights)
        _temp_std = np.sqrt(_temp_variance)
        # calculate the profile:
        _profile = _density.Prob(_x)
        if normalize_by == 'integral':
            _profile = _profile / _trapezoid(_profile, _x)
        else:
            _profile = _profile / np.amax(_profile)
        #
        return _x, _profile, _temp_std      

    def precompute_2D(self, param_pairs, **kwargs):
        """
        Precompute profiles for given names.

        :param param_pairs: list of pairs of parameter names (or indices).
        :param kwargs: options forwarded to :meth:`get2DDensityGridData`.
        :returns: None
        """
        for names in param_pairs:
            self.get2DDensityGridData(names[0], names[1], **kwargs)
        #
        return None

    def get2DDensityGridData(self, name1, name2, num_points_2D=32, **kwargs):
        """
        Compute 2D profile posteriors and return it as a grid density data
        for plotting and analysis.

        Overrides the getdist ``MCSamples.get2DDensityGridData`` method. Results are cached in
        ``profile_density_2D`` (together with the transposed density). The best random search sample
        of each bin (or, for the parameter pairs it covered, the samples accumulated by
        :meth:`update_cache_iterative`) is polished by gradient ascent (``pre_polish``) and
        minimization (``polish``), as set at construction.
        For a flow with two parameters the profile is the density, evaluated directly on the grid.

        :param name1: name or index of the x parameter.
        :param name2: name or index of the y parameter.
        :param num_points_2D: number of points per dimension of the 2D profile (default 32).
        :param kwargs: options; ``num_minimization_samples``, ``learning_rate_2D``,
            ``num_gd_interactions_2D``, ``smooth_scale_2D`` and the scipy / torch minimizer options
            (see ``posterior_profile_plotter.options``) are read; the remaining ones are forwarded
            to ``scipy.optimize.minimize`` (filtered to its arguments) when polishing with scipy.
        :returns: getdist ``Density2D`` normalized to its maximum, with the additional attributes
            ``maximum`` (maximum log probability) and ``profile_subspace`` (profile points ``(M, D)``);
            None if a parameter is not found.
        """
        # get the number of the parameter:
        idx1, parx = self._parAndNumber(name1)
        idx2, pary = self._parAndNumber(name2)

        # check:
        if idx1 is None or idx2 is None:
            return None

        # look for cached results:
        if idx1 in self.profile_density_2D.keys():
            if idx2 in self.profile_density_2D[idx1]:
                return self.profile_density_2D[idx1][idx2]

        # if not cached redo the calculation:
        if self.feedback > 1:
            print('')
            print('    * calculating the 2D profile for: ' + parx.name + ', ' + pary.name + ' with indexes ', idx1, idx2)

        # call the MCSamples method to have the marginalized density:
        marge_density = super().get2DDensityGridData(idx1, idx2, 
                                                     fine_bins_2D=num_points_2D, 
                                                     smooth_scale_2D=10.0,
                                                     )

        # with two parameters the profile is the density itself:
        if self.flow.num_params == 2:
            return self._profile_2D_on_grid(idx1, idx2, marge_density, num_points_2D)

        if hasattr(self, '_2d_logP') and (parx.name, pary.name) in self._2d_logP:
            
            _result_logP = self._2d_logP[parx.name, pary.name]
            _filter_finite = np.isfinite(_result_logP)
            _result_logP = _result_logP[_filter_finite]
            _flow_full_samples = self._2d_samples[parx.name, pary.name][_filter_finite,:]
            
        else:
            
            # check that we have cached samples, otherwise generate:
            if self.temp_samples is None:
                self.sample_profile_population(**kwargs)

            # randomized maximizer algorithm:

            # feedback:
            if self.feedback > 1:
                print('    - doing initial randomized search')
            t0 = time.time()

            # get valid bins:
            if self.flow.parameter_ranges is not None:
                _rang = self.flow.parameter_ranges[parx.name]
                x_bins = marge_density.x[np.logical_and(_rang[0] <= marge_density.x, _rang[1] >= marge_density.x)]
                _rang = self.flow.parameter_ranges[pary.name]
                y_bins = marge_density.y[np.logical_and(_rang[0] <= marge_density.y, _rang[1] >= marge_density.y)]
            else:
                x_bins = marge_density.x
                y_bins = marge_density.y

            # we need to re-draw the grid since it might have been changed by getdist:
            x_bins = np.linspace(np.amin(x_bins), np.amax(x_bins), num_points_2D + 1)
            y_bins = np.linspace(np.amin(y_bins), np.amax(y_bins), num_points_2D + 1)

            # protect for samples inside bins:
            _valid_filter_x = np.logical_and(
                self.temp_samples[:, idx1] > np.amin(x_bins), self.temp_samples[:, idx1] < np.amax(x_bins))
            _valid_filter_y = np.logical_and(
                self.temp_samples[:, idx2] > np.amin(y_bins), self.temp_samples[:, idx2] < np.amax(y_bins))
            _valid_filter = np.logical_and(_valid_filter_x, _valid_filter_y)

            # first find best samples in the bin:
            _indexes_x = np.digitize(self.temp_samples[_valid_filter, idx1], x_bins)
            _indexes_y = np.digitize(self.temp_samples[_valid_filter, idx2], y_bins)
            _max_idx = _binned_argmax_2D(
                _indexes_x, _indexes_y,
                self.temp_probs[_valid_filter], len(x_bins), len(y_bins))
            _max_idx = _max_idx[_max_idx >= 0]
            # get the global (un-filtered indexes):
            _max_idx = np.arange(len(self.temp_samples))[_valid_filter][_max_idx]
            # set data:
            _result_logP = self.temp_probs[_max_idx]
            _flow_full_samples = self.temp_samples[_max_idx]
            
            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for random algorithm {0:.4g} (s)'.format(t1))

        if self.feedback > 1:
            print('    - number of 2D bins', num_points_2D**2)
            print('    - number of empty/filled 2D bins', num_points_2D**2 - len(_result_logP), '/', len(_result_logP))
            
        # prepare mask:
        _mask = np.ones(self.flow.num_params)
        _mask[idx1] = 0
        _mask[idx2] = 0

        # gradient descent polishing:
        if self.pre_polish:

            # feedback:
            if self.feedback > 1:
                print('    - doing gradient descent pre-polishing')
            t0 = time.time()

            # do the iterations:
            _learning_rate = kwargs.get('learning_rate_2D', self.options.get('learning_rate_2D'))
            _num_iterations = kwargs.get('num_gd_interactions_2D', self.options.get('num_gd_interactions_2D'))
            _ensemble = copy.deepcopy(_flow_full_samples)
            _ensemble, temp_probs, num_moving, num_iter = self._masked_gradient_ascent(
                learning_rate=_learning_rate, num_iterations=_num_iterations, ensemble=_ensemble, mask=_mask)
            _filter = temp_probs > _result_logP
            _result_logP = np.where(_filter, temp_probs, _result_logP)
            _flow_full_samples = np.where(_filter[:, None], _ensemble, _flow_full_samples)
            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for gradient descents {0:.4g} (s)'.format(t1))
                print(
                    '      at the end of descent after', num_iter, 'steps', num_moving,
                    'samples were still beeing optimized.')
                if np.sum(_filter) < len(_result_logP):
                    print('      gradient descents did not improve', len(_result_logP) - np.sum(_filter), 'points')

        # branch for minimizer polishing:
        if self.polish:

            t0 = time.time()

            # scipy algorithm:
            if self.use_scipy:

                # polishing with scipy minimizer:
                if self.feedback > 1:
                    print('    - doing minimization polishing (scipy)')

                # compute bounds:
                if self.flow.parameter_ranges is not None:
                    _temp_ranges = [self.flow.parameter_ranges[name] for name in self.flow.param_names]
                    for index in sorted([idx1, idx2], reverse=True):
                        del _temp_ranges[index]
                else:
                    _temp_ranges = None

                # read in options:
                _temp_kwargs = kwargs
                _method = _temp_kwargs.pop('scipy_method', self.options.get('scipy_method'))
                _options = _temp_kwargs.pop('scipy_options', self.options.get('scipy_options'))
                _use_jac = _temp_kwargs.pop('scipy_use_jac', self.options.get('scipy_use_jac'))
                if self.feedback > 2:
                    print('      method:', _method)
                    print('      options:', _options)
                    print('      use_jac:', _use_jac)

                # polish:
                _polished_ensemble, _polished_logP = [], []
                success = 0
                for _samp, _val in zip(_flow_full_samples, _result_logP):
                    _initial_x = np.delete(_samp, [idx1, idx2])
                    x0_1 = _samp[idx1]
                    x0_2 = _samp[idx2]

                    def temp_func(x):
                        if idx1 < idx2:
                            _x = np.insert(x, [idx1, idx2-1], [x0_1, x0_2])
                        else:
                            _x = np.insert(x, [idx2, idx1-1], [x0_2, x0_1])
                        return tu.to_numpy(-self.flow.log_probability(self.flow.cast([_x]))[0]).astype(np.float64)

                    def temp_jac(x):
                        if idx1 < idx2:
                            _x = np.insert(x, [idx1, idx2-1], [x0_1, x0_2])
                        else:
                            _x = np.insert(x, [idx2, idx1-1], [x0_2, x0_1])
                        _jac = tu.to_numpy(-self.flow.log_probability_jacobian(self.flow.cast([_x]))[0]).astype(np.float64)
                        return np.delete(_jac, [idx1, idx2])
                    
                    if not _use_jac:
                        temp_jac = None

                    result = minimize(
                        temp_func,
                        x0=_initial_x,
                        jac=temp_jac,
                        bounds=_temp_ranges,
                        method=_method,
                        options=_options,
                        **stutils.filter_kwargs(_temp_kwargs, minimize))
                    if -result.fun > _val:
                        _polished_logP.append(-result.fun)
                        if idx1 < idx2:
                            _polished_ensemble.append(np.insert(result.x, [idx1, idx2-1], [x0_1, x0_2]))
                        else:
                            _polished_ensemble.append(np.insert(result.x, [idx2, idx1-1], [x0_2, x0_1]))
                        success += 1
                    else:
                        _polished_logP.append(_val)
                        _polished_ensemble.append(_samp)

                    if not result.success:
                        if self.feedback > 2:
                            print(result.message)

                if success < len(_result_logP) and self.feedback > 1:
                    print('      Warning minimization failed for ', len(_result_logP) - success, 'points')
                    print('      but ', success, 'samples were still better.')

                _result_logP = np.array(_polished_logP)
                _flow_full_samples = np.array(_polished_ensemble)

            else:

                # polishing with torch minimizer:
                if self.feedback > 1:
                    print('    - doing minimization polishing (torch)')

                _result_logP, _flow_full_samples = self._torch_polish(
                    _flow_full_samples, _result_logP, [idx1, idx2], _mask, **kwargs)

            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for polishing {0:.4g} (s)'.format(t1))

        # now 2D interpolate on a fixed grid:
        if self.feedback > 1:
            print('    - doing interpolation on regular grid')
        t0 = time.time()

        # cast to numpy:
        _result_logP = tu.to_numpy(_result_logP)
        _flow_full_samples = tu.to_numpy(_flow_full_samples)

        # now interpolate:
        _max_log_P = np.amax(_result_logP)
        _temp_interp = LinearNDInterpolator(
            _flow_full_samples[:, [idx1, idx2]], np.exp(_result_logP - _max_log_P), fill_value=0.0, rescale=True)
        _temp_x = np.linspace(marge_density.view_ranges[0][0], marge_density.view_ranges[0][1], num_points_2D)
        _temp_y = np.linspace(marge_density.view_ranges[1][0], marge_density.view_ranges[1][1], num_points_2D)
        _temp_P = _temp_interp(*np.meshgrid(_temp_x, _temp_y))

        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken for interpolation {0:.4g} (s)'.format(t1))

        # smooth the results:
        if self.smoothing:
            if self.feedback > 1:
                print('    - smoothing results')
            t0 = time.time()

            # get analysis for the two parameters:
            par_1 = self._initParamRanges(idx1, None)
            par_2 = self._initParamRanges(idx2, None)
            # get overall smoothing factor:
            _smooth_scale_2D = kwargs.get('smooth_scale_2D', self.smooth_scale_2D)
            if _smooth_scale_2D <= 0.:
                _smooth_scale_2D = 0.2
            # get smoothing factor:
            _smoothing_sigma_1 = _smooth_scale_2D * len(_temp_x) / (_temp_x[-1] - _temp_x[0]) * par_1.err
            _smoothing_sigma_2 = _smooth_scale_2D * len(_temp_y) / (_temp_y[-1] - _temp_y[0]) * par_2.err
            # do the smoothing:
            _temp_P = gaussian_filter(_temp_P, [_smoothing_sigma_2, _smoothing_sigma_1], mode='reflect')

            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for smoothing {0:.4g} (s)'.format(t1))

        # initialize and cache the densities:
        return self._save_2D_profile(idx1, idx2, _temp_x, _temp_y, _temp_P, _max_log_P,
                                     marge_density.view_ranges, _flow_full_samples)

    def _probability_on_points(self, points):
        """
        Flow probability on a set of points, normalized to its maximum.

        Points where the flow log probability is not finite get zero probability.

        :param points: points in parameter space, shape ``(M, D)``.
        :returns: probability normalized to its maximum, shape ``(M,)``, and maximum log probability.
        :raises ValueError: if the flow log probability is not finite on any of the points.
        """
        _log_P = np.asarray(tu.to_numpy(self.flow.log_probability(self.flow.cast(points))), dtype=np.float64)
        _log_P[np.logical_not(np.isfinite(_log_P))] = -np.inf
        _max_log_P = np.amax(_log_P)
        if not np.isfinite(_max_log_P):
            raise ValueError('The flow log probability is not finite on the profile grid.')
        return np.exp(_log_P - _max_log_P), _max_log_P

    def _profile_1D_on_grid(self, idx, marge_density, num_points_1D):
        """
        1D profile of a flow with a single parameter, where the profile is the density itself:
        the flow is evaluated directly on the grid, without random search, polishing or smoothing.

        :param idx: index of the parameter.
        :param marge_density: getdist marginal density, providing the ``view_ranges`` of the grid.
        :param num_points_1D: number of grid intervals.
        :returns: the cached profile ``Density1D`` (see :meth:`get1DDensityGridData`).
        """
        _temp_x = np.linspace(marge_density.view_ranges[0], marge_density.view_ranges[1], num_points_1D + 1)
        _grid_points = _temp_x[:, None]
        _temp_P, _max_log_P = self._probability_on_points(_grid_points)
        return self._save_1D_profile(idx, _temp_x, _temp_P, _max_log_P, marge_density.view_ranges,
                                     _grid_points, None)

    def _profile_2D_on_grid(self, idx1, idx2, marge_density, num_points_2D):
        """
        2D profile of a flow with two parameters, where the profile is the density itself:
        the flow is evaluated directly on the grid, without random search, polishing or smoothing.

        :param idx1: index of the x parameter.
        :param idx2: index of the y parameter.
        :param marge_density: getdist marginal density, providing the ``view_ranges`` of the grid.
        :param num_points_2D: number of grid points per dimension.
        :returns: the cached profile ``Density2D`` (see :meth:`get2DDensityGridData`).
        """
        _temp_x = np.linspace(marge_density.view_ranges[0][0], marge_density.view_ranges[0][1], num_points_2D)
        _temp_y = np.linspace(marge_density.view_ranges[1][0], marge_density.view_ranges[1][1], num_points_2D)
        # grid points, with the getdist ``P[y, x]`` layout:
        _mesh_x, _mesh_y = np.meshgrid(_temp_x, _temp_y)
        _grid_points = np.zeros((_mesh_x.size, 2))
        _grid_points[:, idx1] = _mesh_x.ravel()
        _grid_points[:, idx2] = _mesh_y.ravel()
        _temp_P, _max_log_P = self._probability_on_points(_grid_points)
        return self._save_2D_profile(idx1, idx2, _temp_x, _temp_y, _temp_P.reshape(_mesh_x.shape), _max_log_P,
                                     marge_density.view_ranges, _grid_points)

    def _save_1D_profile(self, idx, x, P, max_log_P, view_ranges, profile_subspace, smoothing_scale):
        """
        Build the 1D profile density and cache it in ``profile_density_1D``.

        :param idx: index of the parameter.
        :param x: grid, shape ``(M,)``.
        :param P: profile on the grid, shape ``(M,)``.
        :param max_log_P: maximum log probability of the profile.
        :param view_ranges: getdist view ranges of the density.
        :param profile_subspace: profile points, shape ``(N, D)``.
        :param smoothing_scale: smoothing scale applied to the profile, None if not smoothed.
        :returns: getdist ``Density1D`` normalized to its maximum.
        """
        density1D = Density1D(x, P=P, view_ranges=view_ranges)
        # save out maximum value:
        density1D.maximum = max_log_P
        # normalize:
        density1D.normalize('max', in_place=True)
        # save out the samples and the smoothing scale:
        density1D.profile_subspace = profile_subspace
        density1D.profile_smoothing_scale_1D = smoothing_scale
        # cache result:
        self.profile_density_1D[idx] = density1D
        #
        return density1D

    def _save_2D_profile(self, idx1, idx2, x, y, P, max_log_P, view_ranges, profile_subspace):
        """
        Build the 2D profile density, and its transpose, and cache them in ``profile_density_2D``.

        :param idx1: index of the x parameter.
        :param idx2: index of the y parameter.
        :param x: x grid, shape ``(Mx,)``.
        :param y: y grid, shape ``(My,)``.
        :param P: profile on the grid, shape ``(My, Mx)`` (getdist ``P[y, x]`` layout).
        :param max_log_P: maximum log probability of the profile.
        :param view_ranges: getdist view ranges of the density.
        :param profile_subspace: profile points, shape ``(N, D)``.
        :returns: getdist ``Density2D`` normalized to its maximum.
        """
        density2D = Density2D(x, y, P=P, view_ranges=view_ranges)
        density2D_T = Density2D(y, x, P=P.T, view_ranges=[view_ranges[1], view_ranges[0]])
        for density in [density2D, density2D_T]:
            # save out maximum value:
            density.maximum = max_log_P
            # normalize:
            density.normalize('max', in_place=True)
            # save out the samples:
            density.profile_subspace = profile_subspace
        # cache results:
        if idx1 not in self.profile_density_2D.keys():
            self.profile_density_2D[idx1] = dict()
        if idx2 not in self.profile_density_2D.keys():
            self.profile_density_2D[idx2] = dict()
        self.profile_density_2D[idx1][idx2] = density2D
        self.profile_density_2D[idx2][idx1] = density2D_T
        #
        return density2D

    def _minimizer_options(self, kwargs):
        """
        Options of the torch minimizer, from ``kwargs`` or the profiler options.

        :param kwargs: keyword arguments that may contain ``torch_tolerance``,
            ``torch_max_iterations`` and ``torch_max_line_search_iterations``.
        :returns: dictionary with the options of :func:`~tensiometer.synthetic_probability.optimizers.batched_minimize`.
        """
        return {
            'tolerance': kwargs.get('torch_tolerance', self.options.get('torch_tolerance')),
            'max_iterations': kwargs.get('torch_max_iterations', self.options.get('torch_max_iterations')),
            'max_line_search_iterations': kwargs.get(
                'torch_max_line_search_iterations', self.options.get('torch_max_line_search_iterations')),
        }

    def _torch_polish(self, samples, logP, fixed_indices, mask, **kwargs):
        """
        Maximize the flow probability over the free coordinates of all the points at once,
        with the batched torch minimizer. Points are updated only if they improve.

        :param samples: profile points, shape ``(B, D)``.
        :param logP: their log probability, shape ``(B,)``.
        :param fixed_indices: indices of the coordinates that are kept fixed.
        :param mask: array ``(D,)`` with 0 on the fixed coordinates and 1 elsewhere.
        :returns: updated ``logP`` and ``samples`` as numpy arrays.
        """
        samples = np.array(tu.to_numpy(samples), dtype=tu.np_prec)
        logP = np.array(tu.to_numpy(logP), dtype=tu.np_prec)
        if len(samples) == 0:
            return logP, samples
        # get the range bijector and transform initial points:
        if self.box_prior:
            box_bijector = self._get_masked_box_bijector()
            with torch.no_grad():
                _temp_samples = tu.to_numpy(box_bijector.inverse(self.flow.cast(samples)))
        else:
            box_bijector = None
            _temp_samples = samples
        # split fixed and free coordinates:
        free_indices = [i for i in range(self.flow.num_params) if i not in fixed_indices]
        fixed_values = tu.to_tensor(_temp_samples[:, fixed_indices], device='cpu')
        initial_x = tu.to_tensor(_temp_samples[:, free_indices], device='cpu')

        def _objective(x):
            _x = _insert_fixed_columns(x, fixed_indices, fixed_values)
            if box_bijector is not None:
                _x = box_bijector(_x)
            return -tu.to_tensor(self.flow.log_probability(_x))

        def func(x):
            with torch.no_grad():
                return _objective(x)

        def jac(x):
            return autodiff.gradient(_objective, autodiff.prepare_input(x), create_graph=False)

        # initial inverse Hessian from the masked Fisher matrix:
        _keep = np.asarray(mask) > 0
        _masked_fisher = np.asarray(self.temp_inv_cov)[np.ix_(_keep, _keep)]
        try:
            _inverse_hessian = np.linalg.inv(_masked_fisher)
        except np.linalg.LinAlgError:
            _inverse_hessian = np.eye(int(np.sum(_keep)))
        _inverse_hessian = 0.5 * (_inverse_hessian + _inverse_hessian.T)
        # minimize:
        _results = batched_minimize(
            lambda x: (func(x), jac(x)),
            initial_x,
            initial_inverse_hessian=tu.to_tensor(_inverse_hessian, device='cpu'),
            **self._minimizer_options(kwargs))
        _new_logP = -tu.to_numpy(_results.objective_value)
        _filter = _new_logP > logP
        # feedback:
        _converged = tu.to_numpy(_results.converged)
        if np.sum(_converged) < len(_converged) and self.feedback > 1:
            print('      Warning minimization failed for ', len(_converged) - np.sum(_converged), 'points')
            print('      but ', np.sum(_filter), 'samples were still better.')
        # get samples:
        _new_samples = tu.to_numpy(_insert_fixed_columns(_results.position, fixed_indices, fixed_values))
        _temp_samples = np.where(_filter[:, None], _new_samples, _temp_samples)
        logP = np.where(_filter, _new_logP, logP)
        # transform back:
        if box_bijector is not None:
            with torch.no_grad():
                samples = tu.to_numpy(box_bijector(self.flow.cast(_temp_samples)))
        else:
            samples = _temp_samples
        #
        return logP, samples

    def _masked_gradient_ascent(self, learning_rate, num_iterations, ensemble, mask, atol=0.0):
        """
        Masked gradient ascent for an ensemble of points.
        Mask is a [0, 1] vector that decides which coordinates are updated.

        :param learning_rate: step size in units of the inverse largest eigenvalue of the masked Fisher matrix.
        :param num_iterations: maximum number of iterations.
        :param ensemble: points, shape ``(B, D)``.
        :param mask: array ``(D,)`` of zeros (fixed) and ones (updated).
        :param atol: tolerance on the probability decrease of accepted steps.
        :returns: numpy ensemble, numpy log probabilities, number of points still moving at
            the end and number of iterations.
        """
        _mask = np.asarray(tu.to_numpy(mask), dtype=np.float64)
        # get the prior box bijector:
        if self.box_prior:

            box_bijector = self._get_masked_box_bijector()
            with torch.no_grad():
                _ensemble = box_bijector.inverse(self.flow.cast(ensemble)).cpu()

            def _func(x):
                with torch.no_grad():
                    return tu.to_tensor(self.flow.log_probability(box_bijector(x)), device='cpu')

            def _jacobian(x):
                return tu.to_tensor(autodiff.gradient(
                    lambda _x: self.flow.log_probability(box_bijector(_x)), autodiff.prepare_input(x),
                    create_graph=False), device='cpu')

        else:

            _ensemble = self.flow.cast(ensemble)

            def _func(x):
                return tu.to_tensor(self.flow.log_probability(x), device='cpu')

            def _jacobian(x):
                return tu.to_tensor(self.flow.log_probability_jacobian(x), device='cpu')

        # initialize probability values:
        val = _func(_ensemble)
        # initialize step:
        _keep = _mask > 0
        _masked_fisher = np.asarray(self.temp_inv_cov)[np.ix_(_keep, _keep)]
        _lambda_max = np.amax(np.abs(np.linalg.eigvalsh(_masked_fisher)))
        _h = 2. * learning_rate / _lambda_max
        _mask_tensor = tu.to_tensor(_mask, device='cpu')

        # do the loop:
        num_iter = 0
        num_moving = 1
        while num_iter < num_iterations and num_moving > 0:
            # compute Jacobian and apply mask:
            _jac = _mask_tensor * _jacobian(_ensemble)
            # update positions, do not move the mask:
            _ensemble_temp = _ensemble + _h * _jac
            # check new probability values:
            _val = _func(_ensemble_temp)
            # mask points that did not improve:
            _filter = _val > val - atol
            num_moving = int(torch.sum(_filter))
            # update:
            val = torch.where(_filter, _val, val)
            _ensemble = torch.where(_filter[:, None], _ensemble_temp, _ensemble)
            num_iter += 1

        # transform back:
        if self.box_prior:
            with torch.no_grad():
                ensemble = tu.to_numpy(box_bijector(_ensemble))
        else:
            ensemble = tu.to_numpy(_ensemble)

        #
        return ensemble, tu.to_numpy(val), num_moving, num_iter

    ####################################################################################################
    # methods to save and load the object:

    def savePickle(self, filename):
        """
        Save the profiler, together with its flow, to one file (written with ``torch.save``).

        The file is a pickle: only load it from trusted sources. Profiler files written by
        the TensorFlow version of tensiometer cannot be loaded.

        :param filename: The file to write to
        """
        tu.atomic_save(self, filename)

    @classmethod
    def loadPickle(cls, filename, device=None):
        """
        Load a profiler saved with :meth:`savePickle`, including its flow.

        :param filename: The file to read from
        :param device: device of the flow, defaults to the default device.
        :returns: the profiler.
        :raises TypeError: if the file does not contain a profiler.
        """
        target = tu.check_device(device)
        _profiler = torch.load(filename, map_location=target, weights_only=False)
        if not isinstance(_profiler, cls):
            raise TypeError(str(filename) + ' does not contain a ' + cls.__name__)
        return _profiler


###############################################################################
# helpers for the torch minimizer:


def _insert_fixed_columns(x_free, fixed_indices, fixed_values):
    """
    Build full points from the free coordinates and the fixed ones.

    :param x_free: tensor ``(B, D - k)`` with the free coordinates.
    :param fixed_indices: list of the ``k`` fixed coordinate indices.
    :param fixed_values: tensor ``(B, k)`` with the fixed values, in the order of ``fixed_indices``.
    :returns: tensor ``(B, D)``.
    """
    num_params = x_free.shape[-1] + len(fixed_indices)
    fixed_values = fixed_values.to(device=x_free.device, dtype=x_free.dtype)
    columns = []
    free_position = 0
    for i in range(num_params):
        if i in fixed_indices:
            j = fixed_indices.index(i)
            columns.append(fixed_values[..., j:j + 1])
        else:
            columns.append(x_free[..., free_position:free_position + 1])
            free_position += 1
    return torch.cat(columns, dim=-1)
