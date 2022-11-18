###############################################################################
# initial imports and set-up:

import gc
import time
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic_2d
from scipy.interpolate import LinearNDInterpolator

# getdist imports:
import getdist.mcsamples as mcsamples
import getdist.types as types
from getdist.paramnames import ParamInfo
from getdist.densities import Density1D, Density2D

# tensiometer imports:
from .. import utilities as utils

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
        _options = {
                    'ftol': 1.e-6,
                    'gtol': 1e-05,
                    }
        # prepare:
        success, min_value, min_point = [], [], []
        # do the loop:
        for i, _x0 in enumerate(x0):
            # feedback:
            if feedback > 0:
                print('  * sample', i+1)
            # main minimizer call:
            result = minimize(func, x0=_x0, jac=jac, method=_method,
                              options=_options, **utils.filter_kwargs(kwargs, minimize))
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


def find_flow_MAP(flow, feedback=1, abstract=True, num_samples=None,
                  num_best_to_follow=10, initial_points=None, use_scipy=True, **kwargs):
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
            num_samples = 1000*flow.num_params
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
        del(temp_samples, temp_probs)
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
        def func(x): return -flow.log_probability_abs(x)
        def jac(x): return -flow.log_probability_abs_jacobian(x)
    else:
        def func(x): return -flow.log_probability(x)
        def jac(x): return -flow.log_probability_jacobian(x)
    # do explicit maximization for all the best points:
    if use_scipy:
        def func2(x): return -flow.log_probability(flow.cast(x)).numpy().astype(np.float64)
        def jac2(x): return -flow.log_probability_jacobian(flow.cast(x)).numpy().astype(np.float64)
    else:
        func2, jac2 = func, jac
    # now do the minimization
    success, min_value, min_point = points_minimizer(func2, jac2, best_population,
                                                     feedback=feedback, use_scipy=use_scipy, **kwargs)
    # then find best solution and send out:
    _value = min_value[np.argmin(min_value)]
    _solution = min_point[np.argmin(min_value)]
    #
    return _value, _solution


###############################################################################
# hijack getdist plotting:


class posterior_profile_plotter(mcsamples.MCSamples):

    def __init__(self, flow, num_samples, feedback=1, **kwargs):
        """
        Initialize the profile posterior plotter.
        Pass a flow and the number of samples to use for base MCSamples.
        """
        # initialize settings:
        self.feedback = feedback
        # initialize internal variables:
        initialize_cache = kwargs.get('initialize_cache', True)
        name_tag = kwargs.get('name_tag', flow.name_tag+'_profiler')
        # initialize the parent with the flow (so we can do margestats)
        _samples = flow.sample(num_samples)
        _loglikes = -flow.log_probability(_samples)
        super(posterior_profile_plotter, self).__init__(samples=_samples.numpy(), loglikes=_loglikes.numpy(),
                                                        names=flow.param_names, labels=flow.param_labels,
                                                        ranges=flow.parameter_ranges,
                                                        name_tag=name_tag, **utils.filter_kwargs(kwargs, mcsamples.MCSamples))
        # save the flow so that we can query it:
        self.flow = flow
        # define the empty caches:
        self.reset_cache()
        # call the initial calculation:
        if initialize_cache:
            self.update_cache(**kwargs)
        #
        return None

    def update_cache(self, **kwargs):
        """
        Initialize the profiler cache. This does all the maximization
        calculations and it's the heavy part of the whole thing.
        """
        # initial feedback:
        if self.feedback > 0:
            print('  * initializing profiler data.')

        # draw the initial population of samples:
        self.sample_profile_population(**kwargs)

        # initialize best fit:
        if kwargs.get('update_MAP', True):
            self.find_MAP(**kwargs)

        # initialize 1D profiles:
        if kwargs.get('update_1D', True):
            for ind in range(self.n):
                self.get1DDensityGridData(ind, **kwargs)

        # initialize 2D profiles:
        if kwargs.get('update_2D', True):
            for ind1 in range(self.n):
                for ind2 in range(ind1+1, self.n):
                    self.get2DDensityGridData(ind1, ind2, **kwargs)

        # do clean up:
        del(self.temp_samples)
        del(self.temp_probs)
        self.temp_samples = None
        self.temp_probs = None
        #
        return None

    def reset_cache(self):
        """
        Delete and re-initialize profile posterior (empty) caches.
        """
        # temporary storage for samples for minimization. These can be heavy and should not be stored.
        if hasattr(self, 'temp_samples'):
            del(self.temp_samples)
        if hasattr(self, 'temp_probs'):
            del(self.temp_probs)
        self.temp_samples = None
        self.temp_probs = None
        # best fit storage:
        self.bestfit = None
        self.likeStats = None
        self.flow_MAP = None
        self.flow_MAP_logP = None
        # profile posterior storage:
        if hasattr(self, 'profile_density_1D'):
            del(self.profile_density_1D)
        if hasattr(self, 'profile_density_2D'):
            del(self.profile_density_2D)
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
        # settings:
        num_minimization_samples = kwargs.get('num_minimization_samples', 10000*self.flow.num_params)
        # feedback:
        if self.feedback > 1:
            print('    - number of random search samples =', num_minimization_samples)
        # initial population of points:
        if self.feedback > 1:
            print('    - sampling the distribution')
        t0 = time.time()
        self.temp_samples = self.flow.sample(num_minimization_samples)
        self.temp_probs = self.flow.log_probability(self.temp_samples)
        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken to sample the distribution {0:.4g} (s)'.format(t1))
        #
        return None

    def find_MAP(self, x0=None, randomize=True, num_best_to_follow=10, abstract=True, use_scipy=False, **kwargs):
        """
        Find the flow MAP. Strategy is sample and then minimize.
        """
        # if not cached redo the calculation:
        if self.feedback > 1:
            print('    - finding global best fit')
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
            print('    - doing minimization')
        t0 = time.time()
        # call population minimizer:
        _value, _solution = find_flow_MAP(self.flow, feedback=self.feedback-2,
                                          abstract=abstract,
                                          initial_points=initial_population,
                                          use_scipy=True, **kwargs)
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

    def get1DDensityGridData(self, name, num_points_1D=128, randomize=True, polish=True, **kwargs):
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

        # look for cached results:
        if idx in self.profile_density_1D.keys():
            return self.profile_density_1D[idx]

        # if not cached redo the calculation:
        if self.feedback > 1:
            print('    - calculating the 1D profile for:', par_name.name)
        # call the MCSamples method to have the marginalized density:
        marge_density = super().get1DDensityGridData(name,
                                                     fine_bins=num_points_1D,
                                                     num_bins=num_points_1D,
                                                     **kwargs)
        # initialize calculation
        _x = marge_density.x

        # two different algorithms for randomization or not:
        if randomize:
            # feedback:
            if self.feedback > 1:
                print('    - doing initial randomized search')
            t0 = time.time()

            # check that we have cached samples, otherwise generate:
            if self.temp_samples is None:
                self.sample_profile_population(**kwargs)

            # first find best samples in the bin:
            _indexes = np.digitize(self.temp_samples[:, idx], _x)

            # this code can definitely be improved:
            _flow_samples_x, _flow_samples_logP = [], []
            _flow_full_samples = []
            for j in range(len(_x)-1):
                _temp = self.temp_probs[_indexes == j]
                if len(_temp) > 0:
                    _idx = np.argmax(_temp)
                    _flow_full_samples.append(self.temp_samples[:, :][_indexes == j][_idx])
                    _flow_samples_x.append(self.temp_samples[:, idx][_indexes == j][_idx])
                    _flow_samples_logP.append(_temp[_idx])

            _flow_samples_x = np.array(_flow_samples_x)
            _flow_samples_logP = np.array(_flow_samples_logP)

            # feedback:
            t1 = time.time() - t0
            if self.feedback > 1:
                print('    - time taken for random algorithm {0:.4g} (s)'.format(t1))
                print('    - number of 1D bins', len(marge_density.x)-1)
                print('    - number of empty 1D bins', len(_flow_samples_logP))

            # save results:
            _result_x = _flow_samples_x
            _result_logP = _flow_samples_logP

            # branch for minimizer polishing:
            if polish:

                # feedback:
                if self.feedback > 1:
                    print('    - doing minimization polishing')
                t0 = time.time()

                # polish:
                _polished_x, _polished_logP = [], []
                for _samp in _flow_full_samples:
                    if _samp is not None:
                        _initial_x = np.delete(_samp, idx)
                        x0 = _samp[idx]
                        def temp_func(x):
                            _x = np.insert(x, idx, x0)
                            return -self.flow.log_probability(self.flow.cast(_x)).numpy().astype(np.float64)
                        def temp_jac(x):
                            _x = np.insert(x, idx, x0)
                            _jac = -self.flow.log_probability_jacobian(self.flow.cast(_x)).numpy().astype(np.float64)
                            return np.delete(_jac, idx)
                        result = minimize(temp_func,
                                          x0=_initial_x,
                                          jac=temp_jac,
                                          )
                        _polished_x.append(x0)
                        _polished_logP.append(-result.fun)
                _polished_x, _polished_logP = np.array(_polished_x), np.array(_polished_logP)

                # feedback:
                t1 = time.time() - t0
                if self.feedback > 1:
                    print('    - time taken for polishing {0:.4g} (s)'.format(t1))

                _result_x = _polished_x
                _result_logP = _polished_logP

        # initialize density (note that getdist assumes equispaced so we have to resample...)
        _temp_interp = interp1d(_result_x, np.exp(_result_logP), kind='cubic', bounds_error=False, fill_value=0.0)
        _temp_x = np.linspace(marge_density.view_ranges[0], marge_density.view_ranges[1], num_points_1D)
        _temp_P = _temp_interp(_temp_x)
        density1D = Density1D(_temp_x, P=np.exp(_temp_P), view_ranges=marge_density.view_ranges)
        density1D.normalize('max', in_place=True)

        # cache result:
        self.profile_density_1D[idx] = density1D
        #
        return density1D

    @tf.function()
    def _2D_gradient_descent(self, learning_rate, num_interactions, ensemble, mask):
        """
        Helper for 2D gradient descent.
        """
        # loop variables:
        i = tf.constant(0)

        # loop condition:
        def while_condition(i, _):
            return tf.less(i, num_interactions)

        # loop body:
        def body(i, ensemble):
            # compute Jacobian:
            _jac = self.flow.log_probability_jacobian(ensemble)
            # do not move the mask:
            ensemble = ensemble + learning_rate * mask * _jac
            #
            return tf.add(i, 1), ensemble
        # do the loop:
        _, ensemble = tf.while_loop(while_condition, body, [i, ensemble])
        #
        return ensemble

    def get2DDensityGridData(self, j, j2, num_points_2D=64, num_plot_contours=None, **kwargs):
        """
        Compute 2D profile posteriors and return it as a grid density data
        for plotting and analysis.
        """
        # get the number of the parameter:
        idx1, parx = self._parAndNumber(j)
        idx2, pary = self._parAndNumber(j2)

        # check:
        if j is None or j2 is None:
            return None

        # look for cached results:
        if idx1 in self.profile_density_2D.keys():
            if idx2 in self.profile_density_2D[idx1]:
                return self.profile_density_2D[idx1][idx2]

        # if not cached redo the calculation:
        if self.feedback > 1:
            print('    - calculating the 2D profile for: '+parx.name+', '+pary.name)
        # call the MCSamples method to have the marginalized density:
        marge_density = super().get2DDensityGridData(idx1, idx2,
                                                     fine_bins_2D=num_points_2D,
                                                     **kwargs)

        # check that we have cached samples, otherwise generate:
        if self.temp_samples is None:
            self.sample_profile_population(**kwargs)

        # feedback:
        if self.feedback > 1:
            print('    - doing initial randomized search')
        t0 = time.time()

        # get binned stats:
        binned_stats = binned_statistic_2d(self.temp_samples[:, idx1],
                                           self.temp_samples[:, idx2],
                                           self.temp_probs,
                                           'max',
                                           bins=[marge_density.x, marge_density.y],
                                           expand_binnumbers=False)

        # get indexes of best samples (this is super slow):
        _temp_indexes = []
        _temp_arange = np.arange(len(self.temp_probs))
        for k in range((len(marge_density.x)+1)*(len(marge_density.y)+1)):
            _sel_el = self.temp_probs[binned_stats.binnumber == k]
            if len(_sel_el) > 0:
                _temp_indexes.append(_temp_arange[binned_stats.binnumber==k][np.argmax(_sel_el)])
            else:
                _temp_indexes.append(np.nan)
        # mask:
        _temp_indexes = np.array(_temp_indexes)
        _mask = np.isfinite(_temp_indexes)

        # feedback:
        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken for random algorithm {0:.4g} (s)'.format(t1))
            print('    - number of 2D bins', num_points_2D**2)
            print('    - number of empty 2D bins', num_points_2D**2-np.sum(_mask))

        # do gradient descent iterations:

        # feedback:
        if self.feedback > 1:
            print('    - doing gradient descents')
        t0 = time.time()

        # prepare ensemble:
        ensemble = self.flow.cast(self.temp_samples.numpy()[_temp_indexes[_mask].astype(int), :])
        # prepare mask:
        _mask = np.ones(self.flow.num_params)
        _mask[idx1] = 0
        _mask[idx2] = 0
        _mask = tf.constant(_mask, dtype=ensemble.dtype)
        # do the iterations:
        ensemble = self._2D_gradient_descent(learning_rate=0.01, num_interactions=100, ensemble=ensemble, mask=_mask)
        # compute probabilities:
        _new_probabilities = self.flow.log_probability(ensemble).numpy()
        ensemble = ensemble.numpy()

        # feedback:
        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken for gradient descents {0:.4g} (s)'.format(t1))

        # now 2D interpolate on a fixed grid:
        if self.feedback > 1:
            print('    - doing interpolation on regular grid')
        t0 = time.time()

        _temp_interp = LinearNDInterpolator(ensemble[:, [idx1, idx2]], np.exp(_new_probabilities), fill_value=0.0, rescale=True)
        _temp_x = np.linspace(marge_density.view_ranges[0][0], marge_density.view_ranges[0][1], num_points_2D)
        _temp_y = np.linspace(marge_density.view_ranges[1][0], marge_density.view_ranges[1][1], num_points_2D)
        _temp_P = _temp_interp(*np.meshgrid(_temp_x, _temp_y))

        density2D = Density2D(_temp_x, _temp_y, P=_temp_P,
                              view_ranges=marge_density.view_ranges)
        density2D.normalize('max', in_place=True)

        density2D_T = Density2D(_temp_y, _temp_x, P=_temp_P.T,
                                view_ranges=[marge_density.view_ranges[1], marge_density.view_ranges[0]])
        density2D_T.normalize('max', in_place=True)

        t1 = time.time() - t0
        if self.feedback > 1:
            print('    - time taken for interpolation {0:.4g} (s)'.format(t1))

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
