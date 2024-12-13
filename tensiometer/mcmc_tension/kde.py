"""
This module contains functions to estimate the probability of a parameter shift given
with KDE methods.

For further details we refer to `arxiv 2105.03324 <https://arxiv.org/pdf/2105.03324.pdf>`_.
"""

###############################################################################
# initial imports and set-up:


import os
import time
import gc
from numba import jit
import numpy as np
import getdist.chains as gchains
gchains.print_load_details = False
from getdist import MCSamples
import scipy
from scipy.linalg import sqrtm
from scipy.integrate import simpson as simps 
from scipy.spatial import cKDTree

from ..utilities import stats_utilities as stutils

# imports for parallel calculations:
import multiprocessing
import joblib
# number of threads available:
if 'OMP_NUM_THREADS' in os.environ.keys():
    n_threads = int(os.environ['OMP_NUM_THREADS'])
else:
    n_threads = multiprocessing.cpu_count()

###############################################################################
# KDE bandwidth selection:


def Scotts_bandwidth(num_params, num_samples):
    """
    Compute Scott's rule of thumb bandwidth covariance scaling.
    This should be a fast approximation of the 1d MISE estimate.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :return: Scott's scaling matrix.
    :reference: Chacón, J. E., Duong, T. (2018). 
        Multivariate Kernel Smoothing and Its Applications. 
        United States: CRC Press.
    """
    return num_samples**(-2./(num_params+4.)) * np.identity(int(num_params))


def AMISE_bandwidth(num_params, num_samples):
    """
    Compute Silverman's rule of thumb bandwidth covariance scaling AMISE.
    This is the default scaling that is used to compute the KDE estimate of
    parameter shifts.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :return: AMISE bandwidth matrix.
    :reference: Chacón, J. E., Duong, T. (2018). 
        Multivariate Kernel Smoothing and Its Applications. 
        United States: CRC Press.
    """
    coeff = (num_samples * (num_params + 2.) / 4.)**(-2. / (num_params + 4.))
    return coeff * np.identity(int(num_params))


def MAX_bandwidth(num_params, num_samples):
    """
    Compute the maximum bandwidth matrix.
    This bandwidth is generally oversmoothing.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :return: MAX bandwidth matrix.
    :reference: Chacón, J. E., Duong, T. (2018). 
        Multivariate Kernel Smoothing and Its Applications. 
        United States: CRC Press.
    """
    d, n = num_params, num_samples
    coeff = (d + 8.)**((d + 6.) / (d + 4.)) / 4.
    coeff = coeff*(1./n/(d + 2.)/scipy.special.gamma(d/2. + 4))**(2./(d + 4.))
    return coeff*np.identity(int(num_params))


@jit(nopython=True, fastmath=True)
def _mise1d_optimizer(alpha, d, n):
    """
    Utility function that is minimized to obtain the MISE 1d bandwidth.
    """
    tmp = 2**(-d/2.) - 2/(2 + alpha)**(d/2.) + (2 + 2*alpha)**(-d/2.) \
        + (alpha**(-d/2.) - (1 + alpha)**(-d/2.))/(2**(d/2.)*n)
    return tmp


@jit(nopython=True, fastmath=True)
def _mise1d_optimizer_jac(alpha, d, n):
    """
    Jacobian of the MISE 1d bandwidth optimizer.
    """
    tmp = d*(2 + alpha)**(-1 - d/2.) - d*(2 + 2*alpha)**(-1 - d/2.) \
        + (2**(-1 - d/2.)*d*(-alpha**(-1 - d/2.)
                             + (1 + alpha)**(-1 - d/2.)))/n
    return tmp


def MISE_bandwidth_1d(num_params, num_samples, **kwargs):
    """
    Computes the MISE bandwidth matrix. All coordinates are considered the same
    so the MISE estimate just rescales the identity matrix.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :param kwargs: optional arguments to be passed to the optimizer algorithm.
    :return: MISE 1d bandwidth matrix.
    :reference: Chacón, J. E., Duong, T. (2018). 
        Multivariate Kernel Smoothing and Its Applications. 
        United States: CRC Press.
    """
    # initial calculations:
    alpha0 = kwargs.pop('alpha0', None)
    if alpha0 is None:
        alpha0 = AMISE_bandwidth(num_params, num_samples)[0, 0]
    d, n = num_params, num_samples
    # explicit optimization:
    opt = scipy.optimize.minimize(lambda alpha:
                                  _mise1d_optimizer(np.exp(alpha), d, n),
                                  np.log(alpha0),
                                  jac=lambda alpha:
                                  _mise1d_optimizer_jac(np.exp(alpha), d, n),
                                  **kwargs)
    # check for success:
    if not opt.success:
        print(opt)
    #
    return np.exp(opt.x[0]) * np.identity(num_params)


@jit(nopython=True, fastmath=True)
def _mise_optimizer(H, d, n):
    """
    Optimizer function to compute the MISE over the space of SPD matrices.
    """
    Id = np.identity(d)
    tmp = 1./np.sqrt(np.linalg.det(2.*H))/n
    tmp = tmp + (1.-1./n)/np.sqrt(np.linalg.det(2.*H + 2.*Id)) \
        - 2./np.sqrt(np.linalg.det(H + 2.*Id)) + np.power(2., -d/2.)
    return tmp


def MISE_bandwidth(num_params, num_samples, feedback=0, **kwargs):
    """
    Computes the MISE bandwidth matrix by numerically minimizing the MISE
    over the space of positive definite symmetric matrices.

    :param num_params: the number of parameters in the chain.
    :param num_samples: the number of samples in the chain.
    :param feedback: feedback level. If > 2 prints a lot of information.
    :param kwargs: optional arguments to be passed to the optimizer algorithm.
    :return: MISE bandwidth matrix.
    :reference: Chacón, J. E., Duong, T. (2018). 
        Multivariate Kernel Smoothing and Its Applications. 
        United States: CRC Press.
    """
    # initial calculations:
    alpha0 = kwargs.pop('alpha0', None)
    if alpha0 is None:
        alpha0 = MISE_bandwidth_1d(num_params, num_samples)
    alpha0 = stutils.PDM_to_vector(alpha0)
    d, n = num_params, num_samples
    # build a constraint:
    bounds = kwargs.pop('bounds', None)
    if bounds is None:
        bounds = np.array([[None, None] for i in range(d*(d+1)//2)])
        bounds[np.tril_indices(d, 0)[0] == np.tril_indices(d, 0)[1]] = [alpha0[0]/100, alpha0[0]*100]
    # explicit optimization:
    opt = scipy.optimize.minimize(lambda x: _mise_optimizer(stutils.vector_to_PDM(x), d, n),
                                  x0=alpha0, bounds=bounds, **kwargs)
    # check for success:
    if not opt.success or feedback > 2:
        print('MISE_bandwidth')
        print(opt)
    #
    return stutils.vector_to_PDM(opt.x)


@jit(nopython=True, fastmath=True, parallel=True)
def _UCV_optimizer_brute_force(H, weights, white_samples):
    """
    Optimizer for the cross validation bandwidth estimator.
    This does the computation with a brite force algorithm that scales as
    :math:`n_{\\rm samples}^2`. For this reason this is really never used.
    Note this solves for sqrt(H).
    """
    # digest:
    n, d = white_samples.shape
    fac = 2**(-d/2.)
    # compute the weights vectors:
    wtot = np.sum(weights)
    neff = wtot**2 / np.sum(weights**2)
    alpha = wtot / (wtot - weights)
    # compute determinant:
    detH = np.linalg.det(H)
    # whiten samples with inverse H:
    samps = white_samples.dot(np.linalg.inv(H))
    # brute force summation:
    res = 0.
    for i in range(1, n):
        for j in range(i):
            temp_samp = samps[i]-samps[j]
            r2 = np.dot(temp_samp, temp_samp)
            temp = fac*np.exp(-0.25*r2) - 2.*alpha[i]*np.exp(-0.5*r2)
            res += weights[i]*weights[j]*temp
    res = 2. * res / wtot**2
    #
    return (fac/neff + res)/detH


def _UCV_optimizer_nearest(H, weights, white_samples, n_nearest=20):
    """
    Optimizer for the cross validation bandwidth estimator.
    This does the computation uses a truncated KD-tree keeping only a limited
    number of nearest neighbours.
    Note this solves for sqrt(H).
    This is the algorithm that is always used in practice.
    """
    # digest:
    n, d = white_samples.shape
    fac = 2**(-d/2.)
    # compute the weights vectors:
    wtot = np.sum(weights)
    neff = wtot**2 / np.sum(weights**2)
    alpha = wtot / (wtot - weights)
    # compute determinant:
    detH = np.linalg.det(H)
    # whiten samples with inverse H:
    samps = white_samples.dot(np.linalg.inv(H))
    # KD-tree computation:
    data_tree = cKDTree(samps, balanced_tree=True)
    # query for nearest neighbour:
    r2, idx = data_tree.query(samps, np.arange(2, n_nearest), workers=-1)
    r2 = np.square(r2)
    temp = weights[:, None]*weights[idx]*(fac*np.exp(-0.25*r2)
                                          - 2.*np.exp(-0.5*r2)*alpha[:, None])
    res = np.sum(temp) / wtot**2
    #
    return (fac/neff + res)/detH


def UCV_bandwidth(weights, white_samples, alpha0=None, feedback=0, mode='full', **kwargs):
    """
    Computes the optimal unbiased cross validation bandwidth for the input samples
    by numerical minimization.

    :param weights: input sample weights.
    :param white_samples: pre-whitened samples (identity covariance)
    :param alpha0: (optional) initial guess for the bandwidth. If none is
        given then the AMISE band is used as the starting point for minimization.
    :param feedback: (optional) how verbose is the algorithm. Default is zero.
    :param mode: (optional) selects the space for minimization. Default is
        over the full space of SPD matrices. Other options are `diag` to perform
        minimization over diagonal matrices and `1d` to perform minimization
        over matrices that are proportional to the identity.
    :param kwargs: other arguments passed to :func:`scipy.optimize.minimize`
    :return: UCV bandwidth matrix.
    :reference: Chacón, J. E., Duong, T. (2018). 
        Multivariate Kernel Smoothing and Its Applications. 
        United States: CRC Press.
    """
    # digest input:
    n, d = white_samples.shape
    n_nearest = kwargs.pop('n_nearest', 20)
    # get number of effective samples:
    wtot = np.sum(weights)
    neff = wtot**2 / np.sum(weights**2)
    # initial guess calculations:
    t0 = time.time()
    if alpha0 is None:
        alpha0 = AMISE_bandwidth(d, neff)
    # select mode:
    if mode == '1d':
        opt = scipy.optimize.minimize(lambda alpha: _UCV_optimizer_nearest(np.sqrt(np.exp(alpha)) * np.identity(d), weights, white_samples, n_nearest),
                                      np.log(alpha0[0, 0]), **kwargs)
        res = np.exp(opt.x[0]) * np.identity(d)
    elif mode == 'diag':
        opt = scipy.optimize.minimize(lambda alpha: _UCV_optimizer_nearest(np.diag(np.sqrt(np.exp(alpha))), weights, white_samples, n_nearest),
                                      x0=np.log(np.diag(alpha0)), **kwargs)
        res = np.diag(np.exp(opt.x))
    elif mode == 'full':
        # build a constraint:
        bounds = kwargs.pop('bounds', None)
        if bounds is None:
            bounds = np.array([[None, None] for i in range(d*(d+1)//2)])
            bounds[np.tril_indices(d, 0)[0] == np.tril_indices(d, 0)[1]] = [alpha0[0, 0]/10, alpha0[0, 0]*10]
        # explicit optimization:
        alpha0 = stutils.PDM_to_vector(sqrtm(alpha0))
        opt = scipy.optimize.minimize(lambda alpha: _UCV_optimizer_nearest(stutils.vector_to_PDM(alpha), weights, white_samples, n_nearest),
                                      x0=alpha0, bounds=bounds, **kwargs)
        res = stutils.vector_to_PDM(opt.x)
        res = np.dot(res, res)
    # check for success and final feedback:
    if not opt.success or feedback > 2:
        print(opt)
    if feedback > 0:
        t1 = time.time()
        print('Time taken for UCV_bandwidth '+mode+' calculation:',
              round(t1-t0, 1), '(s)')
    #
    return res


def UCV_SP_bandwidth(white_samples, weights, feedback=0, near=1, near_max=20):
    """
    Computes the optimal unbiased cross validation bandwidth scaling for the
    BALL sampling point KDE estimator.

    :param white_samples: pre-whitened samples (identity covariance).
    :param weights: input sample weights.
    :param feedback: (optional) how verbose is the algorithm. Default is zero.
    :param near: (optional) number of nearest neighbour to use. Default is 1.
    :param near_max: (optional) number of nearest neighbour to use for the UCV calculation. Default is 20.
    """
    # digest input:
    n, d = white_samples.shape
    fac = 2**(-d/2.)
    t0 = time.time()
    # prepare the Tree with the samples:
    data_tree = cKDTree(white_samples, balanced_tree=True)
    # compute the weights vectors:
    wtot = np.sum(weights)
    weights2 = weights**2
    neff = wtot**2 / np.sum(weights2)
    alpha = wtot / (wtot - weights)
    # query the Tree for the maximum number of nearest neighbours:
    dist, idx = data_tree.query(white_samples, np.arange(2, near_max+1), workers=-1)
    r2 = np.square(dist)
    # do all sort of precomputations:
    R = dist[:, near]
    R2 = r2[:, near]
    R2s = R2[:, None] + R2[idx]
    term_1 = fac*np.sum(weights2/R**d)
    weight_term = weights[:, None]*weights[idx]
    R2sd = R2s**(-d/2)
    Rd = R[:, None]**d
    R21 = r2/R2s
    R22 = r2/R2[:, None]
    alpha_temp = alpha[:, None]

    # define helper for minimization:
    @jit(nopython=True)
    def _helper(gamma):
        # compute the i != j sum:
        temp = weight_term*(R2sd*gamma**(-d/2)*np.exp(-0.5*R21/gamma) - 2.*alpha_temp/Rd/gamma**d*np.exp(-0.5*R22/gamma))
        # sum:
        _ucv = term_1/gamma**d + np.sum(temp)
        _ucv = _ucv / wtot**2
        #
        return _ucv

    # initial guess:
    x0 = AMISE_bandwidth(d, neff)[0, 0]
    # call optimizer:
    res = scipy.optimize.minimize(lambda x: _helper(np.exp(x)), x0=np.log(x0), method='Nelder-Mead')
    res.x = np.exp(res.x)
    #
    if feedback > 0:
        t1 = time.time()
        print('Time taken for UCV_SP_bandwidth calculation:',
              round(t1-t0, 1), '(s)')
    #
    return res


def OptimizeBandwidth_1D(diff_chain, param_names=None, num_bins=1000):
    """
    Compute an estimate of an optimal bandwidth for covariance scaling as in
    GetDist. This is performed on whitened samples (with identity covariance),
    in 1D, and then scaled up with a dimensionality correction.

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param num_bins: number of bins used for the 1D estimate
    :return: scaling vector for the whitened parameters
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # some initial calculations:
    _samples_cov = diff_chain.cov(pars=param_names)
    _num_params = len(ind)
    # whiten the samples:
    _temp = sqrtm(stutils.QR_inverse(_samples_cov))
    white_samples = diff_chain.samples[:, ind].dot(_temp)
    # make these samples so that we can use GetDist band optization:
    temp_samples = MCSamples(samples=white_samples,
                             weights=diff_chain.weights,
                             ignore_rows=0, sampler=diff_chain.sampler)
    # now get optimal band for each parameter:
    bands = []
    for i in range(_num_params):
        # get the parameter:
        par = temp_samples._initParamRanges(i, paramConfid=None)
        # get the bins:
        temp_result = temp_samples._binSamples(temp_samples.samples[:, i],
                                               par, num_bins)
        bin_indices, bin_width, binmin, binmax = temp_result
        bins = np.bincount(bin_indices, weights=temp_samples.weights,
                           minlength=num_bins)
        # get the optimal smoothing scale:
        N_eff = temp_samples._get1DNeff(par, i)
        band = temp_samples.getAutoBandwidth1D(bins, par, i, N_eff=N_eff,
                                               mult_bias_correction_order=0,
                                               kernel_order=0) \
            * (binmax - binmin)
        # correction for dimensionality:
        dim_factor = Scotts_bandwidth(_num_params, N_eff)[0, 0]/Scotts_bandwidth(1., N_eff)[0, 0]
        #
        bands.append(band**2.*dim_factor)
    #
    return np.array(bands)

###############################################################################
# Parameter difference integrals:


def _gauss_kde_logpdf(x, samples, weights):
    """
    Utility function to compute the Gaussian log KDE probability at x from
    already whitened samples, possibly with weights.
    Normalization constants are ignored.
    """
    X = x-samples
    return scipy.special.logsumexp(-0.5*(X*X).sum(axis=1), b=weights)


def _gauss_ballkde_logpdf(x, samples, weights, distance_weights):
    """
    Utility function to compute the Gaussian log KDE probability
    with variable ball bandwidth at x from already whitened samples,
    possibly with weights. Each element has its own smoothing scale
    that is passed as `distance_weights`.
    Normalization constants are ignored.
    """
    X = x-samples
    return scipy.special.logsumexp(-0.5*(X*X).sum(axis=1)/distance_weights**2,
                                   b=weights)


def _gauss_ellkde_logpdf(x, samples, weights, distance_weights):
    """
    Utility function to compute the Gaussian log KDE probability
    with variable ellipsoid bandwidth at x from already whitened samples,
    possibly with weights. Each element has its own smoothing matrix
    that is passed as `distance_weights`.
    Normalization constants are ignored.
    """
    X = x-samples
    X = np.einsum('...j,...jk,...k', X, distance_weights, X)
    return scipy.special.logsumexp(-0.5*X, b=weights)


def _brute_force_kde_param_shift(white_samples, weights, zero_prob,
                                 num_samples, feedback, weights_norm=None,
                                 distance_weights=None):
    """
    Brute force parallelized algorithm for parameter shift.
    """
    # get feedback:
    if feedback > 1:
        from tqdm import tqdm
        def feedback_helper(x): return tqdm(x, ascii=True)
    else:
        def feedback_helper(x): return x
    # prepare:
    if distance_weights is not None:
        if len(distance_weights.shape) == 1:
            _num_params = white_samples.shape[1]
            weights_norm = weights/distance_weights**_num_params
            _log_pdf = _gauss_ballkde_logpdf
            _args = white_samples, weights_norm, distance_weights
        if len(distance_weights.shape) == 3:
            _log_pdf = _gauss_ellkde_logpdf
            _args = white_samples, weights_norm, distance_weights
    else:
        _log_pdf = _gauss_kde_logpdf
        _args = white_samples, weights
    # run:
    with joblib.Parallel(n_jobs=n_threads) as parallel:
        _kde_eval_pdf = parallel(joblib.delayed(_log_pdf)
                                 (samp, *_args)
                                 for samp in feedback_helper(white_samples))
    # filter for probability calculation:
    _filter = _kde_eval_pdf > zero_prob
    # compute number of filtered elements:
    _num_filtered = np.sum(weights[_filter])
    #
    return _num_filtered


def _neighbor_parameter_shift(white_samples, weights, zero_prob, num_samples,
                              feedback, weights_norm=None, distance_weights=None, **kwargs):
    """
    Parameter shift calculation through neighbour elimination.
    """
    # import specific for this function:
    if feedback > 1:
        from tqdm import tqdm
        def feedback_helper(x): return tqdm(x, ascii=True)
    else:
        def feedback_helper(x): return x
    # get options:
    stable_cycle = kwargs.get('stable_cycle', 2)
    chunk_size = kwargs.get('chunk_size', 40)
    smallest_improvement = kwargs.get('smallest_improvement', 1.e-4)
    # the tree elimination has to work with probabilities to go incremental:
    _zero_prob = np.exp(zero_prob)
    # build tree:
    if feedback > 1:
        print('Building KD-Tree with leafsize =', 10*chunk_size)
    data_tree = cKDTree(white_samples, leafsize=10*chunk_size,
                        balanced_tree=True)
    # make sure that the weights are floats:
    _weights = weights.astype(float)
    # initialize the calculation to zero:
    _num_elements = len(_weights)
    _kde_eval_pdf = np.zeros(_num_elements)
    _filter = np.ones(_num_elements, dtype=bool)
    _last_n = 0
    _stable_cycle = 0
    # loop over the neighbours:
    if feedback > 1:
        print('Neighbours elimination')
    for i in range(_num_elements//chunk_size):
        ind_min = chunk_size*i
        ind_max = chunk_size*i+chunk_size
        _dist, _ind = data_tree.query(white_samples[_filter],
                                      ind_max, workers=-1)
        if distance_weights is not None:
            if len(distance_weights.shape) == 1:
                # BALL case:
                _kde_eval_pdf[_filter] += np.sum(
                    weights_norm[_ind[:, ind_min:ind_max]]
                    * np.exp(-0.5*np.square(_dist[:, ind_min:ind_max]/distance_weights[_ind[:, ind_min:ind_max]])), axis=1)
            if len(distance_weights.shape) == 3:
                # ELL case:
                X = white_samples[_ind[:, ind_min:ind_max]] - white_samples[_ind[:, 0], np.newaxis, :]
                d2 = np.einsum('...j,...jk,...k', X, distance_weights[_ind[:, ind_min:ind_max]], X)
                _kde_eval_pdf[_filter] += np.sum(
                    weights_norm[_ind[:, ind_min:ind_max]] * np.exp(-0.5*d2), axis=1)
        else:
            # standard case:
            _kde_eval_pdf[_filter] += np.sum(
                _weights[_ind[:, ind_min:ind_max]]
                * np.exp(-0.5*np.square(_dist[:, ind_min:ind_max])), axis=1)
        _filter[_filter] = _kde_eval_pdf[_filter] < _zero_prob
        _num_filtered = np.sum(_filter)
        if feedback > 2:
            print('neighbor_elimination: chunk', i+1)
            print('    surviving elements', _num_filtered,
                  'of', _num_elements)
        # check if calculation has converged:
        _term_check = float(np.abs(_num_filtered-_last_n)) \
            / float(_num_elements) < smallest_improvement
        if _term_check and _num_filtered < _num_elements:
            _stable_cycle += 1
            if _stable_cycle >= stable_cycle:
                break
        elif not _term_check and _stable_cycle > 0:
            _stable_cycle = 0
        elif _num_filtered == 0:
            break
        else:
            _last_n = _num_filtered
    # clean up memory:
    del(data_tree)
    # brute force the leftovers:
    if feedback > 1:
        print('neighbor_elimination: polishing')
    # prepare:
    if distance_weights is not None:
        if len(distance_weights.shape) == 1:
            _num_params = white_samples.shape[1]
            weights_norm = weights/distance_weights**_num_params
            _log_pdf = _gauss_ballkde_logpdf
            _args = white_samples, weights_norm, distance_weights
        if len(distance_weights.shape) == 3:
            _log_pdf = _gauss_ellkde_logpdf
            _args = white_samples, weights_norm, distance_weights
    else:
        _log_pdf = _gauss_kde_logpdf
        _args = white_samples, weights
    # run:
    with joblib.Parallel(n_jobs=n_threads) as parallel:
        _kde_eval_pdf[_filter] = parallel(joblib.delayed(_log_pdf)
                                          (samp, *_args)
                                          for samp in feedback_helper(white_samples[_filter]))
        _filter[_filter] = _kde_eval_pdf[_filter] < np.log(_zero_prob)
    if feedback > 1:
        print('    surviving elements', np.sum(_filter),
              'of', _num_elements)
    # compute number of filtered elements:
    _num_filtered = num_samples - np.sum(weights[_filter])
    #
    return _num_filtered


def kde_parameter_shift_1D_fft(diff_chain, 
                               prior_diff_chain=None,
                               param_names=None,
                               scale=None, nbins=1024, feedback=1,
                               boundary_correction_order=1,
                               mult_bias_correction_order=1,
                               **kwarks):
    """
    Compute the MCMC estimate of the probability of a parameter shift given
    an input parameter difference chain in 1 dimension and by using FFT.
    This function uses GetDist 1D fft and optimal bandwidth estimates to
    perform the MCMC parameter shift integral discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param prior_diff_chain: :class:`~getdist.mcsamples.MCSamples`
        (optional) prior parameter difference chain. If present the code will use likelihood 
        thresholded tension calculation, giving a result that is parameter invariant.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        If none is provided the algorithm uses GetDist optimized bandwidth.
    :param nbins: (optional) number of 1D bins for the fft. Powers of 2 work best. Default is 1024.
    :param mult_bias_correction_order: (optional) multiplicative bias
        correction passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param boundary_correction_order: (optional) boundary correction
        passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :return: probability value and error estimate.
    :reference: `Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # check that param_names are in the prior_diff_chain:
    if prior_diff_chain is not None:
        prior_params = prior_diff_chain.getParamNames().list()
        if not np.all([name in prior_params for name in param_names]):
            raise ValueError('Input parameter is not in the prior diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', prior_params)
    # check that we only have one parameter:
    if len(param_names) != 1:
        raise ValueError('Calling 1D algorithm with more than 1 parameters')
    # initialize scale:
    if scale is None or isinstance(scale, str):
        scale = -1
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # compute the density with GetDist:
    t0 = time.time()
    density = diff_chain.get1DDensity(name=ind[0], normalized=True,
                                      num_bins=nbins,
                                      smooth_scale_1D=scale,
                                      boundary_correction_order=boundary_correction_order,
                                      mult_bias_correction_order=mult_bias_correction_order)
    # initialize the spline:
    density._initSpline()
    # compute prior density:
    if prior_diff_chain is not None:
        # indexes:
        prior_ind = [prior_diff_chain.index[name] for name in param_names]
        # density:
        prior_density = prior_diff_chain.get1DDensity(name=prior_ind[0], normalized=True,
                                                      num_bins=nbins,
                                                      smooth_scale_1D=scale,
                                                      boundary_correction_order=boundary_correction_order,
                                                      mult_bias_correction_order=mult_bias_correction_order)
        # initialize the spline:
        prior_density._initSpline()
    # compute probability of zero:
    prob_zero = density.Prob([0.])[0]
    if prior_diff_chain is not None:
        prob_zero = prob_zero / prior_density.Prob([0.])[0]
    # do the MC integral:
    probs = density.Prob(diff_chain.samples[:, ind[0]])
    if prior_diff_chain is not None:
        probs = probs / prior_density.Prob(diff_chain.samples[:, ind[0]])
    # filter:
    _filter = probs > prob_zero
    # if there are samples above zero then use MC:
    if np.sum(_filter) > 0:
        _num_filtered = float(np.sum(diff_chain.weights[_filter]))
        _num_samples = float(np.sum(diff_chain.weights))
        _P = float(_num_filtered)/float(_num_samples)
        _low, _upper = stutils.clopper_pearson_binomial_trial(_num_filtered,
                                                            _num_samples,
                                                            alpha=0.32)
    # if there are no samples try to do the integral:
    else:
        # normalize the density:
        norm = simps(density.P, density.x)
        # filter:
        if prior_diff_chain is None:
            _second_filter = density.P < prob_zero
        else:
            _second_filter = density.P / prior_density.Prob(density.x) < prob_zero
        # do the integral:
        density.P[_second_filter] = 0
        _P = simps(density.P, density.x)/norm
        _low, _upper = None, None
    #
    t1 = time.time()
    if feedback > 0:
        print('Time taken for 1D FFT-KDE calculation:', round(t1-t0, 1), '(s)')
    #
    return _P, _low, _upper


def kde_parameter_shift_2D_fft(diff_chain, 
                               prior_diff_chain=None,
                               param_names=None,
                               scale=None, nbins=1024, feedback=1,
                               boundary_correction_order=1,
                               mult_bias_correction_order=1,
                               **kwarks):
    """
    Compute the MCMC estimate of the probability of a parameter shift given
    an input parameter difference chain in 2 dimensions and by using FFT.
    This function uses GetDist 2D fft and optimal bandwidth estimates to
    perform the MCMC parameter shift integral discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param prior_diff_chain: :class:`~getdist.mcsamples.MCSamples`
        (optional) prior parameter difference chain. If present the code will use likelihood
        thresholded tension calculation, giving a result that is parameter invariant.
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        If none is provided the algorithm uses GetDist optimized bandwidth.
    :param nbins: (optional) number of 2D bins for the fft. Powers of 2 work best. Default is 1024.
    :param mult_bias_correction_order: (optional) multiplicative bias
        correction passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param boundary_correction_order: (optional) boundary correction
        passed to GetDist.
        See :meth:`~getdist.mcsamples.MCSamples.get2DDensity`.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :return: probability value and error estimate.
    :reference: `Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # check that param_names are in the prior_diff_chain:
    if prior_diff_chain is not None:
        prior_params = prior_diff_chain.getParamNames().list()
        if not np.all([name in prior_params for name in param_names]):
            raise ValueError('Input parameter is not in the prior diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', prior_params)
    # check that we only have two parameters:
    if len(param_names) != 2:
        raise ValueError('Calling 2D algorithm with more than 2 parameters')
    # initialize scale:
    if scale is None or isinstance(scale, str):
        scale = -1
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # compute the density with GetDist:
    t0 = time.time()
    density = diff_chain.get2DDensity(x=ind[0], y=ind[1], normalized=True,
                                      fine_bins_2D=nbins,
                                      smooth_scale_2D=scale,
                                      boundary_correction_order=boundary_correction_order,
                                      mult_bias_correction_order=mult_bias_correction_order)
    # initialize the spline:
    density._initSpline()
    # compute prior density:
    if prior_diff_chain is not None:
        # indexes:
        prior_ind = [prior_diff_chain.index[name] for name in param_names]
        # compute density:
        prior_density = prior_diff_chain.get2DDensity(x=prior_ind[0], y=prior_ind[1], normalized=True,
                                                      fine_bins_2D=nbins,
                                                      smooth_scale_2D=scale,
                                                      boundary_correction_order=boundary_correction_order,
                                                      mult_bias_correction_order=mult_bias_correction_order)
        # initialize the spline:
        prior_density._initSpline()
    # get density of zero:
    prob_zero = density.spl([0.], [0.])[0][0]
    if prior_diff_chain is not None:
        prob_zero = prob_zero / prior_density.spl([0.], [0.])[0][0]
    # do the MC integral:
    probs = density.spl.ev(diff_chain.samples[:, ind[0]],
                           diff_chain.samples[:, ind[1]])
    if prior_diff_chain is not None:
        probs = probs / prior_density.spl.ev(diff_chain.samples[:, ind[0]],
                                             diff_chain.samples[:, ind[1]])
    # filter:
    _filter = probs > prob_zero
    # if there are samples above zero then use MC:
    if np.sum(_filter) > 0:
        _num_filtered = float(np.sum(diff_chain.weights[_filter]))
        _num_samples = float(np.sum(diff_chain.weights))
        _P = float(_num_filtered)/float(_num_samples)
        _low, _upper = stutils.clopper_pearson_binomial_trial(_num_filtered,
                                                            _num_samples,
                                                            alpha=0.32)
    # if there are no samples try to do the integral:
    else:
        norm = simps(simps(density.P, density.y), density.x)
        _second_filter = density.P < prob_zero
        density.P[_second_filter] = 0
        _P = simps(simps(density.P, density.y), density.x)/norm
        _low, _upper = None, None
        if prior_diff_chain is not None:
            _P = 1.0
    #
    t1 = time.time()
    if feedback > 0:
        print('Time taken for 2D FFT-KDE calculation:', round(t1-t0, 1), '(s)')
    #
    return _P, _low, _upper


@jit(nopython=True)
def _ell_helper(_ind, _white_samples, _num_params):
    """
    Small helper for ellipse smoothing
    """
    mats = []
    dets = []
    for idx in _ind:
        temp_samp = _white_samples[idx]
        temp_samp = temp_samp[1:, :] - temp_samp[0, :]
        mat = np.zeros((_num_params, _num_params))
        for v in temp_samp:
            mat += np.outer(v, v)
        mats.append(np.linalg.inv(mat))
        dets.append(np.linalg.det(mat))
    return dets, mats


def kde_parameter_shift(diff_chain, param_names=None,
                        scale=None, method='neighbor_elimination',
                        feedback=1, **kwargs):
    """
    Compute the KDE estimate of the probability of a parameter shift given
    an input parameter difference chain.
    This function uses a Kernel Density Estimate (KDE) algorithm discussed in
    (`Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_).
    If the difference chain contains :math:`n_{\\rm samples}` this algorithm
    scales as :math:`O(n_{\\rm samples}^2)` and might require long run times.
    For this reason the algorithm is parallelized with the
    joblib library.
    If the problem is 1d or 2d use the fft algorithm in :func:`kde_parameter_shift_1D_fft`
    and :func:`kde_parameter_shift_2D_fft`.

    :param diff_chain: :class:`~getdist.mcsamples.MCSamples`
        input parameter difference chain
    :param param_names: (optional) parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :param scale: (optional) scale for the KDE smoothing.
        The scale is always referred to white samples with unit covariance.
        If none is provided the algorithm uses MISE estimate.
        Options are:

           #. a scalar for fixed scaling over all dimensions;
           #. a matrix from anisotropic smoothing;
           #. `MISE`, `AMISE`, `MAX` for the corresponding smoothing scale;
           #. `BALL` or `ELL` for variable adaptive smoothing with nearest neighbour;

    :param method: (optional) a string containing the indication for the method
        to use in the KDE calculation. This can be very intensive so different
        techniques are provided.

           #. method = `brute_force` is a parallelized brute force method. This
              method scales as :math:`O(n_{\\rm samples}^2)` and can be afforded
              only for small tensions. When suspecting a difference that is
              larger than 95% other methods are better.
           #. method = `neighbor_elimination` is a KD Tree based elimination method.
              For large tensions this scales as
              :math:`O(n_{\\rm samples}\\log(n_{\\rm samples}))`
              and in worse case scenarions, with small tensions, this can scale
              as :math:`O(n_{\\rm samples}^2)` but with significant overheads
              with respect to the brute force method.
              When expecting a statistically significant difference in parameters
              this is the recomended algorithm.

        Suggestion is to go with brute force for small problems, neighbor
        elimination for big problems with signifcant tensions.
        Default is `neighbor_elimination`.
    :param feedback: (optional) print to screen the time taken
        for the calculation.
    :param kwargs: extra options to pass to the KDE algorithm.
        The `neighbor_elimination` algorithm accepts the following optional
        arguments:

           #. stable_cycle: (default 2) number of elimination cycles that show
              no improvement in the result.
           #. chunk_size: (default 40) chunk size for elimination cycles.
              For best perfornamces this parameter should be tuned to result
              in the greatest elimination rates.
           #. smallest_improvement: (default 1.e-4) minimum percentage improvement
              rate before switching to brute force.
           #. near: (default 1) n-nearest neighbour to use for variable bandwidth KDE estimators.
           #. near_alpha: (default 1.0) scaling for nearest neighbour distance.

    :return: probability value and error estimate from binomial.
    :reference: `Raveri, Zacharegkas and Hu 19 <https://arxiv.org/pdf/1912.04880.pdf>`_
    """
    # initialize param names:
    if param_names is None:
        param_names = diff_chain.getParamNames().getRunningNames()
    else:
        chain_params = diff_chain.getParamNames().list()
        if not np.all([name in chain_params for name in param_names]):
            raise ValueError('Input parameter is not in the diff chain.\n',
                             'Input parameters ', param_names, '\n'
                             'Possible parameters', chain_params)
    # indexes:
    ind = [diff_chain.index[name] for name in param_names]
    # some initial calculations:
    _num_samples = np.sum(diff_chain.weights)
    _num_params = len(ind)
    # number of effective samples:
    _num_samples_eff = np.sum(diff_chain.weights)**2 / \
        np.sum(diff_chain.weights**2)
    # whighten samples:
    _white_samples = stutils.whiten_samples(diff_chain.samples[:, ind],
                                          diff_chain.weights)
    # scale for the kde:
    distance_weights = None
    weights_norm = None
    if (isinstance(scale, str) and scale == 'MISE') or scale is None:
        scale = MISE_bandwidth_1d(_num_params, _num_samples_eff, **kwargs)
    elif isinstance(scale, str) and scale == 'AMISE':
        scale = AMISE_bandwidth(_num_params, _num_samples_eff)
    elif isinstance(scale, str) and scale == 'MAX':
        scale = MAX_bandwidth(_num_params, _num_samples_eff)
    elif isinstance(scale, str) and scale == 'BALL':
        near = kwargs.pop('near', 1)
        near_alpha = kwargs.pop('near_alpha', 1.0)
        data_tree = cKDTree(_white_samples, balanced_tree=True)
        _dist, _ind = data_tree.query(_white_samples, near+1, workers=-1)
        distance_weights = np.sqrt(near_alpha)*_dist[:, near]
        weights_norm = diff_chain.weights/distance_weights**_num_params
        del(data_tree)
    elif isinstance(scale, str) and scale == 'ELL':
        # build tree:
        data_tree = cKDTree(_white_samples, balanced_tree=True)
        _dist, _ind = data_tree.query(_white_samples, _num_params+1, workers=-1)
        del(data_tree)
        # compute the covariances:
        dets, mats = _ell_helper(_ind, _white_samples, _num_params)
        weights_norm = diff_chain.weights/np.sqrt(dets)
        distance_weights = np.array(mats)
    elif isinstance(scale, int) or isinstance(scale, float):
        scale = scale*np.identity(int(_num_params))
    elif isinstance(scale, np.ndarray):
        if not scale.shape == (_num_params, _num_params):
            raise ValueError('Input scaling matrix does not have correct '
                             + 'size \n Input shape: '+str(scale.shape)
                             + '\nNumber of parameters: '+str(_num_params))
        scale = scale
    else:
        raise ValueError('Unrecognized option for scale')
    # feedback:
    if feedback > 0:
        with np.printoptions(precision=3):
            print(f'Dimension       : {int(_num_params)}')
            print(f'N    samples    : {int(_num_samples)}')
            print(f'Neff samples    : {_num_samples_eff:.2f}')
            if not isinstance(scale, str):
                if np.count_nonzero(scale - np.diag(np.diagonal(scale))) == 0:
                    print(f'Smoothing scale :', np.diag(scale))
                else:
                    print(f'Smoothing scale :', scale)
            elif scale == 'BALL':
                print(f'BALL smoothing scale')
            elif scale == 'ELL':
                print(f'ELL smoothing scale')
    # prepare the calculation:
    if not isinstance(scale, str):
        _kernel_cov = sqrtm(np.linalg.inv(scale))
        _white_samples = _white_samples.dot(_kernel_cov)
        _log_pdf = _gauss_kde_logpdf
        _args = _white_samples, diff_chain.weights
    elif scale == 'BALL':
        weights_norm = diff_chain.weights/distance_weights**_num_params
        _log_pdf = _gauss_ballkde_logpdf
        _args = _white_samples, weights_norm, distance_weights
    elif scale == 'ELL':
        _log_pdf = _gauss_ellkde_logpdf
        _args = _white_samples, weights_norm, distance_weights
    # probability of zero:
    _kde_prob_zero = _log_pdf(np.zeros(_num_params), *_args)
    # compute the KDE:
    t0 = time.time()
    if method == 'brute_force':
        _num_filtered = _brute_force_kde_param_shift(_white_samples,
                                                     diff_chain.weights,
                                                     _kde_prob_zero,
                                                     _num_samples,
                                                     feedback,
                                                     weights_norm=weights_norm,
                                                     distance_weights=distance_weights)
    elif method == 'neighbor_elimination':
        _num_filtered = _neighbor_parameter_shift(_white_samples,
                                                  diff_chain.weights,
                                                  _kde_prob_zero,
                                                  _num_samples,
                                                  feedback,
                                                  weights_norm=weights_norm,
                                                  distance_weights=distance_weights,
                                                  **kwargs)
    else:
        raise ValueError('Unknown method provided:', method)
    t1 = time.time()
    # clean up:
    gc.collect()
    # feedback:
    if feedback > 0:
        print('KDE method:', method)
        print('Time taken for KDE calculation:', round(t1-t0, 1), '(s)')
    # probability and binomial error estimate:
    _P = float(_num_filtered)/float(_num_samples)
    _low, _upper = stutils.clopper_pearson_binomial_trial(float(_num_filtered),
                                                        float(_num_samples),
                                                        alpha=0.32)
    #
    return _P, _low, _upper
