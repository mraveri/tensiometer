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
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from itertools import combinations

from numba import njit

# getdist imports:
import getdist.mcsamples as mcsamples
import getdist.types as types
from getdist.paramnames import ParamInfo
from getdist.densities import Density1D, Density2D

# tensiometer imports:
from .. import utilities as utils
from . import fixed_bijectors as pb

###############################################################################
# utility functions:


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


def points_minimizer(func, jac, x0, feedback=0, use_scipy=True, **kwargs):
    """
    Minimize one point. Has default options to deal with 32 bit precisions.
    Note that the scipy implementation can be run with boundaries
    while the tensorflow one cannot.
    Note that using tensorflow methods require everything to be tensors.
    """
    if use_scipy:
        # read in options:
        _method = kwargs.get('method', 'L-BFGS-B')
        _options = kwargs.get('options', {
            'ftol': 1.e-6,
            'gtol': 1e-05,
        })
        # prepare:
        success, min_value, min_point = [], [], []
        # do the loop:
        for i, _x0 in enumerate(x0):
            # feedback:
            if feedback > 0:
                print('  * sample', i + 1)
            # main minimizer call:
            result = minimize(
                func, x0=_x0, jac=jac, method=_method, options=_options, **utils.filter_kwargs(kwargs, minimize))
            # save results:
            success.append(result.success)
            min_value.append(result.fun)
            min_point.append(result.x)
            # feedback:
            if feedback > 0:
                print('    - Success', result.success)
                print('    - Loss function', result.fun)
                print('    - Function /Jacobian evaluations', result.nfev, '/', result.njev)
        # convert to numpy array:
        success = np.array(success)
        min_value = np.array(min_value)
        min_point = np.array(min_point)
    else:
        result = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=lambda x: [func(x), jac(x)],
            initial_position=x0,
        )
        success = result.converged
        min_value = result.objective_value
        min_point = result.position
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
        **kwargs):
    """
    We want to do precise maximization in a large number of dimensions.
    This is achieved by first sampling. Selecting the best samples.
    Mapping them to abstract coordinates (where we have no bounds)
    and finding the MAP.
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
        temp_samples = flow.sample(num_samples)
        temp_probs = flow.log_probability(temp_samples)
        # find the best ones:
        _best_indexes = np.argpartition(temp_probs, -num_best_to_follow)[-num_best_to_follow:]
        # select population:
        best_population = tf.gather(temp_samples, _best_indexes)
        # delete samples (these can be heavy):
        del (temp_samples, temp_probs)
    else:
        best_population = initial_points
    # feedback:
    if feedback > 0:
        print('  using abstract coordinates =', abstract)
    # map to abstract space:
    if abstract:
        best_population = flow.map_to_abstract_coord(best_population)
    # create the tensorflow functions (so that tracing happens here)
    if abstract:

        def func(x):
            return -flow.log_probability_abs(x)

        def jac(x):
            return -flow.log_probability_abs_jacobian(x)
    else:

        def func(x):
            return -flow.log_probability(x)

        def jac(x):
            return -flow.log_probability_jacobian(x)

    # do explicit maximization for all the best points:
    if use_scipy:

        def func2(x):
            return -flow.log_probability(flow.cast(x)).numpy().astype(np.float64)

        def jac2(x):
            return -flow.log_probability_jacobian(flow.cast(x)).numpy().astype(np.float64)
    else:
        func2, jac2 = func, jac
    # now do the minimization
    success, min_value, min_point = points_minimizer(
        func2, jac2, best_population, feedback=feedback, use_scipy=use_scipy, **kwargs)
    # then find best solution and send out:
    _value = min_value[np.argmin(min_value)]
    _solution = min_point[np.argmin(min_value)]
    #
    return _value, _solution


###############################################################################
# hijack getdist plotting:


class posterior_profile_plotter(mcsamples.MCSamples):

    def __init__(self, flow, feedback=1, **kwargs):
        """
        Initialize the profile posterior plotter.
        Pass a flow and the number of samples to use for base MCSamples.
        """
        # initialize settings:
        self.feedback = feedback
        # initialize internal variables:
        initialize_cache = kwargs.get('initialize_cache', True)
        name_tag = kwargs.get('name_tag', flow.name_tag + '_profiler')
        # initialize global options:
        self.use_scipy = kwargs.get('use_scipy', False)
        self.pre_polish = kwargs.get('pre_polish', True)
        self.polish = kwargs.get('polish', True)
        self.smoothing = kwargs.get('smoothing', True)
        self.box_prior = kwargs.get('box_prior', True)
        # feedback:
        if self.feedback > 0:
            print('Flow profiler')
            print('  * use_scipy  =', self.use_scipy)
            print('  * pre_polish =', self.pre_polish)
            print('  * polish     =', self.polish)
            print('  * smoothing  =', self.smoothing)
            print('  * box_prior  =', self.box_prior)
            print('  * parameters =', flow.param_names)
        # initialize the parent with the flow (so we can do margestats)
        _samples = flow.chain_samples
        _loglikes = flow.chain_loglikes
        super().__init__(
            samples=_samples,
            loglikes=_loglikes,
            names=flow.param_names,
            labels=flow.param_labels,
            ranges=flow.parameter_ranges,
            name_tag=name_tag,
            **utils.filter_kwargs(kwargs, mcsamples.MCSamples))
        # save the flow so that we can query it:
        self.flow = flow
        # define the empty caches:
        self.reset_cache()
        # call the initial calculation:
        if initialize_cache:
            self.update_cache(**kwargs)
        #
        return None

    def update_cache(self, params=None, **kwargs):
        """
        Initialize the profiler cache. This does all the maximization
        calculations and it's the heavy part of the whole thing.
        """
        # initial feedback:
        if self.feedback > 0:
            print('  * initializing profiler data.')

        # draw the initial population of samples:
        if 'update_MAP' in kwargs.keys() or \
           'update_1D' in kwargs.keys() or \
           'update_2D' in kwargs.keys():
            self.sample_profile_population(**kwargs)

        # get parameter names:
        if params is not None:
            indexes = [self._parAndNumber(name)[0] for name in params]
        else:
            indexes = list(range(self.n))

        # initialize best fit:
        if kwargs.get('update_MAP', True):
            if self.feedback > 0:
                print('  * finding MAP')
            self.find_MAP(**kwargs)

        # initialize 1D profiles:
        if kwargs.get('update_1D', True):
            if self.feedback > 0:
                print('  * initializing 1D profiles')
            for ind in tqdm.tqdm(indexes, file=sys.stdout, desc='    1D profiles'):
                self.get1DDensityGridData(ind, **kwargs)

        # initialize 2D profiles:
        if kwargs.get('update_2D', True):
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
        # initial feedback:
        if self.feedback > 0:
            print('  * initializing profiler data.')

        # # draw the initial population of samples:
        # if 'update_MAP' in kwargs.keys() or \
        #    'update_1D' in kwargs.keys() or \
        #    'update_2D' in kwargs.keys():
        #     self.sample_profile_population(**kwargs)

        # get parameter names:
        if params is not None:
            indexes = [self._parAndNumber(name)[0] for name in params]
        else:
            indexes = list(range(self.n))

        # # initialize best fit:
        # if kwargs.get('update_MAP', True):
        #     if self.feedback > 0:
        #         print('  * finding MAP')
        #     self.find_MAP(**kwargs)

        # initialize 1D profiles:

        if kwargs.get('update_1D', True):
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
                    name, fine_bins=kwargs.get('num_points_1D', 64), num_bins=kwargs.get('num_points_1D', 64), **kwargs)
                # get valid bins:
                if self.flow.parameter_ranges is not None:
                    _rang = self.flow.parameter_ranges[name]
                    x_bins = marge_density.x[np.logical_and(_rang[0] <= marge_density.x, _rang[1] >= marge_density.x)]
                else:
                    x_bins = marge_density.x
                self._1d_bins[name] = x_bins
                self._1d_samples[name] = np.nan * np.ones((len(x_bins), self.n), dtype=np.float32)
                self._1d_logP[name] = np.full(len(x_bins), -np.inf, dtype=np.float32)

        # initialize 2D profiles:
        if kwargs.get('update_2D', True):
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
                    ind1, ind2, fine_bins_2D=kwargs.get('num_points_2D', 32), **kwargs)
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
                self._2d_samples[name1, name2] = np.nan * np.ones((len(x_bins), len(y_bins), self.n), dtype=np.float32)
                self._2d_logP[name1, name2] = np.full((len(x_bins), len(y_bins)), -np.inf, dtype=np.float32)


        for _ in tqdm.trange(niter, desc='iterative sampling'):
            # sample
            self.sample_profile_population(**kwargs)
            # do 1d
            if kwargs.get('update_1D', True):
                for name, idx in self._1d_name_idx:
                    # protect for samples inside bins:
                    x_bins = self._1d_bins[name]
                    _filter_range = np.logical_and(
                        self.temp_samples[:, idx] > np.amin(x_bins), self.temp_samples[:, idx] < np.amax(x_bins))
                    
                    # first find best samples in the bin:
                    _indexes = np.digitize(self.temp_samples.numpy()[_filter_range, idx], x_bins)
                    _max_bins_idx = _binned_argmax_1D(_indexes, self.temp_probs.numpy()[_filter_range], len(x_bins))
                    
                    # check bins that have points
                    _filter_valid_bins = _max_bins_idx > 0  
                    
                    # get indices of best points
                    _max_valid_bins_idx = _max_bins_idx[_filter_valid_bins]  
                    
                    # and their logP value
                    _max_valid_bins_logP = tf.gather(self.temp_probs,
                                                    _max_valid_bins_idx).numpy()  
                    
                    # check where new points are better than old ones
                    _filter_valid_bins_update = self._1d_logP[name][
                        _filter_valid_bins] < _max_valid_bins_logP  
                    
                    # update best logP values
                    self._1d_logP[name][_filter_valid_bins] = np.where(
                        _filter_valid_bins_update, _max_valid_bins_logP, self._1d_logP[name][_filter_valid_bins])  
                    
                    # update best samples
                    self._1d_samples[name][_filter_valid_bins] = np.where(
                        _filter_valid_bins_update[:, None],
                        copy.deepcopy(self.temp_samples.numpy()[_filter_range][_max_valid_bins_idx]),
                        self._1d_samples[name][_filter_valid_bins])
                
            
            if kwargs.get('update_2D', True):
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
                    _indexes_x = np.digitize(self.temp_samples.numpy()[_filter_range, idx1], x_bins)
                    _indexes_y = np.digitize(self.temp_samples.numpy()[_filter_range, idx2], y_bins)
                    _max_bins_idx = _binned_argmax_2D(
                        _indexes_x, _indexes_y,
                        self.temp_probs.numpy()[_filter_range], len(x_bins), len(y_bins))
                    
                    # check bins that have points
                    _filter_valid_bins = _max_bins_idx > 0  
                    
                    # get indices of best points
                    _max_valid_bins_idx = _max_bins_idx[_filter_valid_bins]  
                    
                    # and their logP value
                    _max_valid_bins_logP = tf.gather(self.temp_probs,
                                                    _max_valid_bins_idx).numpy()  
                    
                    # check where new points are better than old ones
                    _filter_valid_bins_update = self._2d_logP[names][
                        _filter_valid_bins] < _max_valid_bins_logP  
                    
                    # update best logP values
                    self._2d_logP[names][_filter_valid_bins] = np.where(
                        _filter_valid_bins_update, _max_valid_bins_logP, self._2d_logP[names][_filter_valid_bins])  
                    
                    # update best samples
                    self._2d_samples[names][_filter_valid_bins] = np.where(
                        _filter_valid_bins_update[:, None],
                        copy.deepcopy(self.temp_samples.numpy()[_filter_range][_max_valid_bins_idx]),
                        self._2d_samples[names][_filter_valid_bins])
                    

        # initialize 1D profiles:
        if kwargs.get('update_1D', True):
            if self.feedback > 0:
                print('  * initializing 1D profiles')
            for _, ind in tqdm.tqdm(self._1d_name_idx, file=sys.stdout, desc='    1D profiles'):
                self.get1DDensityGridData(ind, **kwargs)
                
        # initialize 2D profiles:
        if kwargs.get('update_2D', True):
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
        Delete and re-initialize profile posterior (empty) caches.
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
        # force memory cleanup:
        gc.collect()
        #
        return None

    def sample_profile_population(self, **kwargs):
        """
        Generate the initial samples for minimzation.
        When calling this function the samples are cached (and can be heavy)
        """
        if self.feedback > 1:
            print('    * generating samples for profiler')
        # settings:
        num_minimization_samples = kwargs.get('num_minimization_samples', 20000 * self.flow.num_params)
        # feedback:
        if self.feedback > 1:
            print('    - number of random search samples =', num_minimization_samples)
        # initial population of points:
        if self.feedback > 1:
            print('    - sampling the distribution')
        t0 = time.time()
        # sample:
        self.temp_samples = self.flow.sample(num_minimization_samples)
        # calculate probability:
        self.temp_probs = self.flow.log_probability(self.temp_samples)
        # process the samples:
        if self.box_prior:
            _box_bijector = self._get_masked_box_bijector()
            temp_samples = _box_bijector.inverse(self.temp_samples)
            _temp_filter = np.all(np.isfinite(temp_samples.numpy()), axis=1)
            if self.feedback > 1:
                print('    - number of random search samples inside box =', np.sum(_temp_filter))
            _temp_filter = np.arange(len(_temp_filter))[_temp_filter]
            self.temp_samples = tf.gather(self.temp_samples, _temp_filter, axis=0)
            self.temp_probs = tf.gather(self.temp_probs, _temp_filter, axis=0)
            self.temp_cov = self.flow.cast(np.cov(temp_samples.numpy()[_temp_filter, :].T))
            self.temp_inv_cov = tf.linalg.inv(self.temp_cov)
        else:
            self.temp_cov = tfp.stats.covariance(self.temp_samples)
            self.temp_inv_cov = tf.linalg.inv(self.temp_cov)

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
                'upper': self.flow.cast(self.flow.parameter_ranges[name][1])
            } for name in _names]
            bound_bijector = pb.prior_bijector_helper(temp_ranges)
        else:
            bound_bijector = tfp.bijectors.Identity()
        #
        return bound_bijector

    def find_MAP(self, x0=None, randomize=True, num_best_to_follow=10, abstract=False, **kwargs):
        """
        Find the flow MAP. Strategy is sample and then minimize.
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
                initial_population = tf.gather(self.temp_samples, _best_indexes, axis=0)

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
                print('    - doing minimization (tensorflow)')
        t0 = time.time()
        # call population minimizer:
        _value, _solution = find_flow_MAP(
            self.flow,
            feedback=self.feedback - 2,
            abstract=abstract,
            initial_points=initial_population,
            use_scipy=self.use_scipy,
            **kwargs)
        # feedback:
        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken for minimization {0:.4g} (s)'.format(t1))

        # cache results:
        self.flow_MAP = _solution
        self.flow_MAP_logP = _value
        # initialize getdist things:
        self._initialize_bestfit()
        self._initialize_likestats()
        #
        return _value, _solution

    def _initialize_likestats(self):
        """
        Initialize likestats
        """
        # check if we can run:
        if self.flow_MAP_logP is None:
            raise ValueError('_initialize_likestats can only run after MAP finder')
        # initialize likestats:
        m = types.LikeStats()
        maxlike = -self.flow_MAP_logP
        m.logLike_sample = maxlike
        # samples statistics:
        if self.temp_samples is None:
            _temp_samples = self.temp_samples
            _temp_loglikes = -self.temp_probs
        else:
            _temp_samples = self.samples
            _temp_loglikes = self.loglikes
        try:
            if np.max(self.loglikes) - maxlike < 30:
                m.logMeanInvLike = np.log(self.mean(np.exp(_temp_loglikes - maxlike))) + maxlike
            else:
                m.logMeanInvLike = None
        except:
            raise
        m.meanLogLike = np.mean(_temp_loglikes)
        m.logMeanLike = -np.log(self.mean(np.exp(-(_temp_loglikes - maxlike)))) + maxlike
        # assuming maxlike is well determined
        m.complexity = 2 * (m.meanLogLike - maxlike)
        m.names = self.paramNames.names

        # get N-dimensional confidence region
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
                par.ND_limit_bot[i] = np.max(region)
            par.bestfit_sample = self.flow_MAP[j]

        # save out:
        self.likeStats = m
        #
        return None

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
        Override standard behavior to reutn cached result
        """
        # check if we can run:
        if self.bestfit is None:
            raise ValueError('getBestFit needs a cached MAP. Call find_MAP first.')
        #
        return self.bestfit

    def precompute_1D(self, params, **kwargs):
        """
        Precompute profiles for given params.
        """
        for name in params:
            self.get1DDensityGridData(name, **kwargs)
        #
        return None

    def get1DDensityGridData(self, name, num_points_1D=64, **kwargs):
        """
        Compute 1D profile posteriors and return it as a grid density data
        for plotting and analysis.
        Note that this ensures that margestats are from the profile.

        num_points_1D number of points in the 1D profile
        randomize default true do initial randomization
        polish do exact minimization polishing by either gradient descent or minimization
        """
        # get the number of the parameter:
        idx, par_name = self._parAndNumber(name)
        
        # check:
        if name is None:
            return None

        # look for cached results:
        if idx in self.profile_density_1D.keys():
            return self.profile_density_1D[idx]
        
        # call the MCSamples method to have the marginalized density:
        marge_density = super().get1DDensityGridData(name, fine_bins=num_points_1D, num_bins=num_points_1D, **kwargs)
        
        if hasattr(self, '_1d_logP'):
            _result_logP = self._1d_logP[par_name.name]
            _filter_finite = np.isfinite(_result_logP)
            _result_logP = _result_logP[_filter_finite]
            _flow_full_samples = self._1d_samples[par_name.name][_filter_finite,:]

        else:
            # check that we have cached samples, otherwise generate:
            if self.temp_samples is None:
                self.sample_profile_population(**kwargs)
                
            # if not cached redo the calculation:
            if self.feedback > 1:
                print('    * calculating the 1D profile for:', par_name.name)
                
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
            # protect for samples inside bins:
            _valid_filter = np.logical_and(
                self.temp_samples[:, idx] > np.amin(x_bins), self.temp_samples[:, idx] < np.amax(x_bins))
            # first find best samples in the bin:
            _indexes = np.digitize(self.temp_samples.numpy()[_valid_filter, idx], x_bins)
            _max_idx = _binned_argmax_1D(_indexes, self.temp_probs.numpy()[_valid_filter], len(x_bins))
            _max_idx = _max_idx[_max_idx > 0]
            # get the global (un-filtered indexes):
            _max_idx = np.arange(len(self.temp_samples))[_valid_filter][_max_idx]
            # set data:
            _result_logP = tf.gather(self.temp_probs, _max_idx)
            _flow_full_samples = tf.gather(self.temp_samples, _max_idx, axis=0)
            
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
        _mask = tf.constant(_mask, dtype=_flow_full_samples.dtype)

        # gradient descent polishing:
        if self.pre_polish:

            # feedback:
            if self.feedback > 1:
                print('    - doing gradient descent pre-polishing')
            t0 = time.time()

            # do the iterations:
            _learning_rate = kwargs.get('learning_rate_1D', 0.1)
            _num_iterations = kwargs.get('num_gd_interactions_1D', 400)
            _ensemble = copy.deepcopy(_flow_full_samples)
            _ensemble, temp_probs, num_moving, num_iter = self._masked_gradient_ascent(
                learning_rate=_learning_rate, num_iterations=_num_iterations, ensemble=_ensemble, mask=_mask)
            _filter = temp_probs > _result_logP
            _result_logP = tf.where(_filter, temp_probs, _result_logP)
            _filter2 = tf.tile(tf.expand_dims(_filter, 1), [1, self.flow.num_params])
            _flow_full_samples = tf.where(_filter2, _ensemble, _flow_full_samples)
            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for gradient descents {0:.4g} (s)'.format(t1))
                print(
                    '      at the end of descent after', num_iter.numpy(), 'steps', num_moving.numpy(),
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
                    _temp_ranges = list(self.flow.parameter_ranges.values())
                    del _temp_ranges[idx]
                else:
                    _temp_ranges = None

                # read in options:
                _method = kwargs.get('method', 'L-BFGS-B')
                _options = kwargs.get('options', {
                    'ftol': 1.e-6,
                    'gtol': 1e-05,
                })

                # polish:
                _polished_ensemble, _polished_logP = [], []
                success = 0
                for _samp, _val in zip(_flow_full_samples, _result_logP):
                    _initial_x = np.delete(_samp, idx)
                    x0 = _samp[idx]

                    def temp_func(x):
                        _x = np.insert(x, idx, x0)
                        return -self.flow.log_probability(self.flow.cast(_x)).numpy().astype(np.float64)

                    def temp_jac(x):
                        _x = np.insert(x, idx, x0)
                        _jac = -self.flow.log_probability_jacobian(self.flow.cast(_x)).numpy().astype(np.float64)
                        return np.delete(_jac, idx)

                    result = minimize(
                        temp_func,
                        x0=_initial_x,
                        jac=temp_jac,
                        bounds=_temp_ranges,
                        method=_method,
                        options=_options,
                        **utils.filter_kwargs(kwargs, minimize))
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

                # polishing with tensorflow minimizer:
                if self.feedback > 1:
                    print('    - doing minimization polishing (tensorflow)')

                # get the range bijector and transform initial points:
                if self.box_prior:
                    box_bijector = self._get_masked_box_bijector()
                    _temp_samples = box_bijector.inverse(_flow_full_samples)
                else:
                    _temp_samples = _flow_full_samples
                # get fixed coordinates:
                _x0 = _temp_samples[:, idx]
                _x0 = tf.expand_dims(_x0, -1)
                # get indexes of varying coordinates:
                _idxs = tf.constant([i for i in range(self.flow.num_params) if i != idx])
                # get initial points:
                _initial_x = tf.gather(_temp_samples, _idxs, axis=1)
                # get number of samples:
                _num_samps = len(_flow_full_samples)

                # define interfaces:
                if self.box_prior:

                    @tf.function
                    def func(x):
                        # insert fixed coordinate:
                        left_slice = tf.slice(x, [0, 0], [_num_samps, idx])
                        right_slice = tf.slice(x, [0, idx], [_num_samps, self.flow.num_params - 1 - idx])
                        _x = tf.concat([left_slice, _x0, right_slice], axis=1)
                        #
                        _logP = -self.flow.log_probability(box_bijector(_x))
                        #
                        return _logP
                else:

                    @tf.function
                    def func(x):
                        # insert fixed coordinate:
                        left_slice = tf.slice(x, [0, 0], [_num_samps, idx])
                        right_slice = tf.slice(x, [0, idx], [_num_samps, self.flow.num_params - 1 - idx])
                        _x = tf.concat([left_slice, _x0, right_slice], axis=1)
                        #
                        _logP = -self.flow.log_probability(_x)
                        #
                        return _logP

                @tf.function
                def jac(x):
                    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                        tape.watch(x)
                        f = func(x)
                    return tape.gradient(f, x)

                # minimize:
                _masked_fisher = tf.boolean_mask(tf.boolean_mask(self.temp_inv_cov, _mask, axis=0), _mask, axis=1)
                _masked_fisher = tf.linalg.inv(_masked_fisher)
                _tf_results = tfp.optimizer.bfgs_minimize(
                    value_and_gradients_function=lambda x: [func(x), jac(x)],
                    initial_position=_initial_x,
                    initial_inverse_hessian_estimate=0.5 * (_masked_fisher + tf.transpose(_masked_fisher)),
                    tolerance=kwargs.get('tolerance', 1.e-5),
                    max_iterations=kwargs.get('max_iterations', 100),
                    max_line_search_iterations=kwargs.get('max_line_search_iterations', 100),
                )
                _filter = -_tf_results.objective_value > _result_logP
                # feedback:
                if np.sum(_tf_results.converged) < len(_tf_results.converged) and self.feedback > 1:
                    print(
                        '      Warning minimization failed for ',
                        len(_tf_results.converged) - np.sum(_tf_results.converged), 'points')
                    print('      but ', np.sum(_filter.numpy()), 'samples were still better.')

                # get samples:
                _result_logP = tf.where(_filter, -_tf_results.objective_value, _result_logP)
                # insert fixed coordinate:
                x = _tf_results.position
                left_slice = tf.slice(x, [0, 0], [_num_samps, idx])
                right_slice = tf.slice(x, [0, idx], [_num_samps, self.flow.num_params - 1 - idx])
                x = tf.concat([left_slice, _x0, right_slice], axis=1)
                _filter = tf.tile(tf.expand_dims(_filter, 1), [1, self.flow.num_params])
                _temp_samples = tf.where(_filter, x, _temp_samples)
                # transform back:
                if self.box_prior:
                    _flow_full_samples = box_bijector(_temp_samples)
                else:
                    _flow_full_samples = _temp_samples

            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for polishing {0:.4g} (s)'.format(t1))

        # now 1D interpolate on a fixed grid:
        if self.feedback > 1:
            print('    - doing interpolation on regular grid')
        t0 = time.time()

        # convert to numpy array:
        if tf.is_tensor(_result_logP):
            _result_logP = _result_logP.numpy()
        if tf.is_tensor(_flow_full_samples):
            _flow_full_samples = _flow_full_samples.numpy()

        # now interpolate:
        _temp_interp = interp1d(
            _flow_full_samples[:, idx], np.exp(_result_logP), kind='cubic', bounds_error=False, fill_value=0.0)
        _temp_x = np.linspace(marge_density.view_ranges[0], marge_density.view_ranges[1], num_points_1D)
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

        # initialize the density:
        density1D = Density1D(_temp_x, P=_temp_P, view_ranges=marge_density.view_ranges)
        density1D.normalize('max', in_place=True)
        density1D.profile_subspace = _flow_full_samples

        # cache result:
        self.profile_density_1D[idx] = density1D
        #
        return density1D

    def precompute_2D(self, param_pairs, **kwargs):
        """
        Precompute profiles for given names.
        """
        for names in param_pairs:
            self.get2DDensityGridData(names[0], names[1], **kwargs)
        #
        return None

    def get2DDensityGridData(self, name1, name2, num_points_2D=32, **kwargs):
        """
        Compute 2D profile posteriors and return it as a grid density data
        for plotting and analysis.
        """
        # get the number of the parameter:
        idx1, parx = self._parAndNumber(name1)
        idx2, pary = self._parAndNumber(name2)

        # check:
        if name1 is None or name2 is None:
            return None

        # look for cached results:
        if idx1 in self.profile_density_2D.keys():
            if idx2 in self.profile_density_2D[idx1]:
                return self.profile_density_2D[idx1][idx2]

        # if not cached redo the calculation:
        if self.feedback > 1:
            print('    * calculating the 2D profile for: ' + parx.name + ', ' + pary.name)
        # call the MCSamples method to have the marginalized density:
        marge_density = super().get2DDensityGridData(idx1, idx2, fine_bins_2D=num_points_2D, **kwargs)

        if hasattr(self, '_2d_logP'):
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

            # protect for samples inside bins:
            _valid_filter_x = np.logical_and(
                self.temp_samples[:, idx1] > np.amin(x_bins), self.temp_samples[:, idx1] < np.amax(x_bins))
            _valid_filter_y = np.logical_and(
                self.temp_samples[:, idx2] > np.amin(y_bins), self.temp_samples[:, idx2] < np.amax(y_bins))
            _valid_filter = np.logical_and(_valid_filter_x, _valid_filter_y)

            # first find best samples in the bin:
            _indexes_x = np.digitize(self.temp_samples.numpy()[_valid_filter, idx1], x_bins)
            _indexes_y = np.digitize(self.temp_samples.numpy()[_valid_filter, idx2], y_bins)
            _max_idx = _binned_argmax_2D(
                _indexes_x, _indexes_y,
                self.temp_probs.numpy()[_valid_filter], len(x_bins), len(y_bins))
            _max_idx = _max_idx[_max_idx > 0]
            # get the global (un-filtered indexes):
            _max_idx = np.arange(len(self.temp_samples))[_valid_filter][_max_idx]
            # set data:
            _result_logP = tf.gather(self.temp_probs, _max_idx)
            _flow_full_samples = tf.gather(self.temp_samples, _max_idx, axis=0)
            
            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for random algorithm {0:.4g} (s)'.format(t1))
            
        # prepare mask:
        _mask = np.ones(self.flow.num_params)
        _mask[idx1] = 0
        _mask[idx2] = 0
        _mask = tf.constant(_mask, dtype=_flow_full_samples.dtype)

        if self.feedback > 1:
            print('    - number of 2D bins', num_points_2D**2)
            print('    - number of empty/filled 2D bins', num_points_2D**2 - len(_result_logP), '/', len(_result_logP))

        # gradient descent polishing:
        if self.pre_polish:

            # feedback:
            if self.feedback > 1:
                print('    - doing gradient descent pre-polishing')
            t0 = time.time()

            # do the iterations:
            _learning_rate = kwargs.get('learning_rate_2D', 0.1)
            _num_iterations = kwargs.get('num_gd_interactions_2D', 400)
            _ensemble = copy.deepcopy(_flow_full_samples)
            _ensemble, temp_probs, num_moving, num_iter = self._masked_gradient_ascent(
                learning_rate=_learning_rate, num_iterations=_num_iterations, ensemble=_ensemble, mask=_mask)
            _filter = temp_probs > _result_logP
            _result_logP = tf.where(_filter, temp_probs, _result_logP)
            _filter2 = tf.tile(tf.expand_dims(_filter, 1), [1, self.flow.num_params])
            _flow_full_samples = tf.where(_filter2, _ensemble, _flow_full_samples)
            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for gradient descents {0:.4g} (s)'.format(t1))
                print(
                    '      at the end of descent after', num_iter.numpy(), 'steps', num_moving.numpy(),
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
                    _temp_ranges = list(self.flow.parameter_ranges.values())
                    for index in sorted([idx1, idx2], reverse=True):
                        del _temp_ranges[index]
                else:
                    _temp_ranges = None

                # read in options:
                _method = kwargs.get('method', 'L-BFGS-B')
                _options = kwargs.get('options', {
                    'ftol': 1.e-6,
                    'gtol': 1e-05,
                })

                # polish:
                _polished_ensemble, _polished_logP = [], []
                success = 0
                for _samp, _val in zip(_flow_full_samples, _result_logP):
                    _initial_x = np.delete(_samp, [idx1, idx2])
                    x0_1 = _samp[idx1]
                    x0_2 = _samp[idx2]

                    def temp_func(x):
                        _x = np.insert(x, [idx1, idx2], [x0_1, x0_2])
                        return -self.flow.log_probability(self.flow.cast(_x)).numpy().astype(np.float64)

                    def temp_jac(x):
                        _x = np.insert(x, [idx1, idx2], [x0_1, x0_2])
                        _jac = -self.flow.log_probability_jacobian(self.flow.cast(_x)).numpy().astype(np.float64)
                        return np.delete(_jac, [idx1, idx2])

                    result = minimize(
                        temp_func,
                        x0=_initial_x,
                        jac=temp_jac,
                        bounds=_temp_ranges,
                        method=_method,
                        options=_options,
                        **utils.filter_kwargs(kwargs, minimize))
                    if -result.fun > _val:
                        _polished_logP.append(-result.fun)
                        _polished_ensemble.append(np.insert(result.x, [idx1, idx2], [x0_1, x0_2]))
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

                # polishing with tensorflow minimizer:
                if self.feedback > 1:
                    print('    - doing minimization polishing (tensorflow)')

                # sort fixed indexes:
                idx_min = min(idx1, idx2)
                idx_max = max(idx1, idx2)

                # get the range bijector and transform initial points:
                if self.box_prior:
                    box_bijector = self._get_masked_box_bijector()
                    _temp_samples = box_bijector.inverse(_flow_full_samples)
                else:
                    _temp_samples = _flow_full_samples
                # get fixed coordinates:
                _x0_min = _temp_samples[:, idx_min]
                _x0_min = tf.expand_dims(_x0_min, -1)
                _x0_max = _temp_samples[:, idx_max]
                _x0_max = tf.expand_dims(_x0_max, -1)
                # get indexes of varying coordinates:
                _idxs = tf.constant([i for i in range(self.flow.num_params) if i != idx_min and i != idx_max])
                # get initial points:
                _initial_x = tf.gather(_temp_samples, _idxs, axis=1)
                # get number of samples:
                _num_samps = len(_temp_samples)

                # define interfaces:
                if self.box_prior:

                    @tf.function
                    def func(x):
                        # insert fixed coordinate:
                        left_slice = tf.slice(x, [0, 0], [_num_samps, idx_min])
                        mid_slice = tf.slice(x, [0, idx_min], [_num_samps, idx_max - idx_min - 1])
                        right_slice = tf.slice(
                            x, [0, idx_max - 1], [_num_samps, self.flow.num_params - 2 - idx_max + 1])
                        _x = tf.concat([left_slice, _x0_min, mid_slice, _x0_max, right_slice], axis=1)
                        #
                        _logP = -self.flow.log_probability(box_bijector(_x))
                        #
                        return _logP
                else:

                    @tf.function
                    def func(x):
                        # insert fixed coordinate:
                        left_slice = tf.slice(x, [0, 0], [_num_samps, idx_min])
                        mid_slice = tf.slice(x, [0, idx_min], [_num_samps, idx_max - idx_min - 1])
                        right_slice = tf.slice(
                            x, [0, idx_max - 1], [_num_samps, self.flow.num_params - 2 - idx_max + 1])
                        _x = tf.concat([left_slice, _x0_min, mid_slice, _x0_max, right_slice], axis=1)
                        #
                        _logP = -self.flow.log_probability(_x)
                        #
                        return _logP

                @tf.function
                def jac(x):
                    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                        tape.watch(x)
                        f = func(x)
                    return tape.gradient(f, x)

                # minimize:
                _masked_fisher = tf.boolean_mask(tf.boolean_mask(self.temp_inv_cov, _mask, axis=0), _mask, axis=1)
                _masked_fisher = tf.linalg.inv(_masked_fisher)
                _tf_results = tfp.optimizer.bfgs_minimize(
                    value_and_gradients_function=lambda x: [func(x), jac(x)],
                    initial_position=_initial_x,
                    initial_inverse_hessian_estimate=0.5 * (_masked_fisher + tf.transpose(_masked_fisher)),
                    tolerance=kwargs.get('tolerance', 1.e-5),
                    max_iterations=kwargs.get('max_iterations', 100),
                    max_line_search_iterations=kwargs.get('max_line_search_iterations', 100),
                )

                _filter = -_tf_results.objective_value > _result_logP
                # feedback:
                if np.sum(_tf_results.converged) < len(_tf_results.converged) and self.feedback > 1:
                    print(
                        '      Warning minimization failed for ',
                        len(_tf_results.converged) - np.sum(_tf_results.converged), 'points')
                    print('      but ', np.sum(_filter.numpy()), 'samples were still better.')

                # get samples:
                _result_logP = tf.where(_filter, -_tf_results.objective_value, _result_logP)
                # insert fixed coordinate:
                x = _tf_results.position
                left_slice = tf.slice(x, [0, 0], [_num_samps, idx_min])
                mid_slice = tf.slice(x, [0, idx_min], [_num_samps, idx_max - idx_min - 1])
                right_slice = tf.slice(x, [0, idx_max - 1], [_num_samps, self.flow.num_params - 2 - idx_max + 1])
                x = tf.concat([left_slice, _x0_min, mid_slice, _x0_max, right_slice], axis=1)
                _filter = tf.tile(tf.expand_dims(_filter, 1), [1, self.flow.num_params])
                _temp_samples = tf.where(_filter, x, _temp_samples)
                # transform back:
                if self.box_prior:
                    _flow_full_samples = box_bijector(_temp_samples)
                else:
                    _flow_full_samples = _temp_samples

            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for polishing {0:.4g} (s)'.format(t1))

        # now 2D interpolate on a fixed grid:
        if self.feedback > 1:
            print('    - doing interpolation on regular grid')
        t0 = time.time()

        # cast to numpy:
        if tf.is_tensor(_result_logP):
            _result_logP = _result_logP.numpy()
        if tf.is_tensor(_flow_full_samples):
            _flow_full_samples = _flow_full_samples.numpy()

        # now interpolate:
        _temp_interp = LinearNDInterpolator(
            _flow_full_samples[:, [idx1, idx2]], np.exp(_result_logP), fill_value=0.0, rescale=True)
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
            _temp_P = gaussian_filter(_temp_P, [_smoothing_sigma_1, _smoothing_sigma_2], mode='reflect')

            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for smoothing {0:.4g} (s)'.format(t1))

        # initialize getdist densities:
        density2D = Density2D(_temp_x, _temp_y, P=_temp_P, view_ranges=marge_density.view_ranges)
        density2D.normalize('max', in_place=True)

        density2D_T = Density2D(
            _temp_y, _temp_x, P=_temp_P.T, view_ranges=[marge_density.view_ranges[1], marge_density.view_ranges[0]])
        density2D_T.normalize('max', in_place=True)
        density2D.profile_subspace = _flow_full_samples
        density2D_T.profile_subspace = _flow_full_samples

        # cache results:
        if idx1 not in self.profile_density_2D.keys():
            self.profile_density_2D[idx1] = dict()
        if idx2 not in self.profile_density_2D.keys():
            self.profile_density_2D[idx2] = dict()
        # save results:
        self.profile_density_2D[idx1][idx2] = density2D
        self.profile_density_2D[idx2][idx1] = density2D_T
        #
        return density2D

    def _masked_gradient_ascent(self, learning_rate, num_iterations, ensemble, mask, atol=1.0):
        """
        Masked gradient ascent for an ensemble of points.
        Mask is a [0, 1] vector that decides which coordinates are updated.
        """
        # loop variables:
        i = tf.constant(0)
        num_moving = tf.constant(1)

        # loop condition:
        def while_condition(i, num_moving, dummy_1, dummy_2):
            return tf.less(i, num_iterations) and num_moving > 0

        # get the prior box bijector:
        if self.box_prior:

            box_bijector = self._get_masked_box_bijector()
            _ensemble = box_bijector.inverse(ensemble)

            @tf.function
            def _func(x):
                return self.flow.log_probability(box_bijector(x))

            @tf.function
            def _jacobian(x):
                with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                    tape.watch(x)
                    f = _func(x)
                return tape.gradient(f, x)

        else:

            _ensemble = ensemble

            def _func(x):
                return self.flow.log_probability(x)

            def _jacobian(x):
                return self.flow.log_probability_jacobian(x)

        # initialize probability values:
        val = _func(_ensemble)
        # initialize step:
        _masked_fisher = tf.boolean_mask(tf.boolean_mask(self.temp_inv_cov, mask, axis=0), mask, axis=1)
        _lambda_max = np.amax(np.abs(np.linalg.eigh(_masked_fisher)[0]))
        _h = 2. * learning_rate / self.flow.cast(_lambda_max)
        _h = tf.expand_dims(_h, axis=-1)

        # loop body for Jacobian only optimization:
        def body_jacobian(i, num_moving, ensemb, val):
            # compute Jacobian:
            _jac = _jacobian(ensemb)
            # apply mask:
            _jac = mask * _jac
            # normalize Jacobian:
            #_norm = tf.norm(_jac, axis=1, keepdims=True)
            ## normalize if needed:
            #_norm_filter = _norm > 1.
            #_jac = tf.where(_norm_filter, _jac / _norm, _jac)
            # update positions, do not move the mask:
            ensemb_temp = ensemb + _h * _jac
            # check new probability values:
            _val = _func(ensemb_temp)
            # mask points that did not improve:
            _filter = _val > val - atol
            num_moving = tf.reduce_sum(tf.cast(_filter, tf.int32))
            # update:
            val = tf.where(_filter, _val, val)
            _filter = tf.tile(tf.expand_dims(_filter, 1), [1, self.flow.num_params])
            ensemb = tf.where(_filter, ensemb_temp, ensemb)
            #
            return tf.add(i, 1), num_moving, ensemb, val

        # do the loop:
        num_iter, num_moving, _ensemble, values = tf.while_loop(
            while_condition, body_jacobian, [i, num_moving, _ensemble, val])

        # transform back:
        if self.box_prior:
            ensemble = box_bijector(_ensemble)
        else:
            ensemble = _ensemble

        #
        return ensemble, values, num_moving, num_iter
