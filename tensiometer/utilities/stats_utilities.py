"""
This file contains some utilities that are used in the tensiometer package.
"""

###############################################################################
# initial imports:

import numpy as np
import scipy
import scipy.special
from scipy.linalg import sqrtm
from getdist import MCSamples
import inspect

###############################################################################


def from_confidence_to_sigma(P):
    """
    Transforms a probability to effective number of sigmas.
    This matches the input probability with the number of standard deviations
    that an event with the same probability would have had in a Gaussian
    distribution as in Eq. (G1) of
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_).

    .. math::
        n_{\\sigma}^{\\rm eff}(P) \\equiv \\sqrt{2} {\\rm Erf}^{-1}(P)

    :param P: the input probability.
    :return: the effective number of standard deviations.
    :reference: https://arxiv.org/1806.04649
    """
    if (np.all(P < 0.) or np.all(P > 1.)):
        raise ValueError('Input probability has to be between zero and one.\n',
                         'Input value is ', P)
    return np.sqrt(2.)*scipy.special.erfinv(P)

###############################################################################


def from_sigma_to_confidence(nsigma):
    """
    Gives the probability of an event at a given number of standard deviations
    in a Gaussian distribution.

    :param nsigma: the input number of standard deviations.
    :return: the probability to exceed the number of standard deviations.
    :reference: https://arxiv.org/1806.04649
    """
    if (np.all(nsigma < 0.)):
        raise ValueError('Input nsigma has to be positive.\n',
                         'Input value is ', nsigma)
    return scipy.special.erf(nsigma/np.sqrt(2.))

###############################################################################


def from_chi2_to_sigma(val, dofs, exact_threshold=6):
    """
    Computes the effective number of standard deviations for a chi squared
    variable.
    This matches the probability computed from the chi squared variable
    to the number of standard deviations that an event with the same
    probability would have had in a Gaussian
    distribution as in Eq. (G1) of
    (`Raveri and Hu 18 <https://arxiv.org/pdf/1806.04649.pdf>`_).

    .. math::
        n_{\\sigma}^{\\rm eff}(x, {\\rm dofs}) \\equiv
        \\sqrt{2} {\\rm Erf}^{-1}({\\rm CDF}(\\chi^2_{\\rm dofs}(x)))

    For very high statistical significant events this function
    switches from the direct formula to an accurate asyntotic expansion.

    :param val: value of the chi2 variable
    :param dofs: number of degrees of freedom of the chi2 variable
    :param exact_threshold: (default 6) threshold of value/dofs to switch to
        the asyntotic formula.
    :return: the effective number of standard deviations.
    :reference: https://arxiv.org/1806.04649
    """
    # check:
    if (np.all(val < 0.)):
        raise ValueError('Input chi2 value has to be positive.\n',
                         'Input value is ', val)
    if (np.all(dofs < 0.)):
        raise ValueError('Input number of dofs has to be positive.\n',
                         'Input value is ', dofs)
    # prepare:
    x = val/dofs
    # if value over dofs is low use direct calculation:
    if x < exact_threshold:
        res = from_confidence_to_sigma(scipy.stats.chi2.cdf(val, dofs))
    # if value is high use first order asyntotic expansion:
    else:
        lgamma = 2*np.log(scipy.special.gamma(dofs/2.))
        res = np.sqrt(dofs*(x + np.log(2)) - (-4 + dofs)*np.log(x*dofs)
                      - 2*np.log(-2 + dofs + x*dofs) + lgamma
                      - np.log(2*np.pi*(dofs*(x + np.log(2)) - np.log(2*np.pi)
                               - (-4 + dofs)*np.log(x*dofs)
                               - 2*np.log(-2 + dofs + x*dofs) + lgamma)))
    #
    return res

###############################################################################


def KL_decomposition(matrix_a, matrix_b):
    """
    Computes the Karhunen-Loeve (KL) decomposition of the matrix A and B. \n
    Notice that B has to be real, symmetric and positive. \n
    The algorithm is taken from
    `this link <http://fourier.eng.hmc.edu/e161/lectures/algebra/node7.html>`_.
    The algorithm is NOT optimized for speed but for precision.

    :param matrix_a: the first matrix.
    :param matrix_b: the second matrix.
    :return: the KL eigenvalues and the KL eigenvectors.
    """
    # compute the eigenvalues of b, lambda_b:
    _lambda_b, _phi_b = np.linalg.eigh(matrix_b)
    # check that this is positive:
    if np.any(_lambda_b < 0.):
        raise ValueError('B is not positive definite\n',
                         'eigenvalues are ', _lambda_b)
    _sqrt_lambda_b = np.diag(1./np.sqrt(_lambda_b))
    _phib_prime = np.dot(_phi_b, _sqrt_lambda_b)
    _a_prime = np.dot(np.dot(_phib_prime.T, matrix_a), _phib_prime)
    _lambda, _phi_a = np.linalg.eigh(_a_prime)
    _phi = np.dot(np.dot(_phi_b, _sqrt_lambda_b), _phi_a)
    return _lambda, _phi

###############################################################################


def QR_inverse(matrix):
    """
    Invert a matrix with the QR decomposition.
    This is much slower than standard inversion but has better accuracy
    for matrices with higher condition number.

    :param matrix: the input matrix.
    :return: the inverse of the matrix.
    """
    _Q, _R = np.linalg.qr(matrix)
    return np.dot(_Q, np.linalg.inv(_R.T))

###############################################################################


def clopper_pearson_binomial_trial(k, n, alpha=0.32):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected
    successes on n trials.

    :param k: number of success.
    :param n: total number of trials.
    :param alpha: (optional) confidence level.
    :return: lower and upper bound.
    """
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi

###############################################################################


def get_separate_mcsamples(chain):
    """
    Function that returns separate :class:`~getdist.mcsamples.MCSamples`
    for each sampler chain.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :return: list of :class:`~getdist.mcsamples.MCSamples` with the separate
        chains.
    """
    # get separate chains:
    _chains = chain.getSeparateChains()
    # copy the param names and ranges:
    _mc_samples = []
    for ch in _chains:
        temp = MCSamples()
        temp.paramNames = chain.getParamNames()
        temp.setSamples(ch.samples, weights=ch.weights, loglikes=ch.loglikes)
        temp.sampler = chain.sampler
        temp.ranges = chain.ranges
        temp.updateBaseStatistics()
        _mc_samples.append(temp.copy())
    #
    return _mc_samples

###############################################################################


def bernoulli_thin(chain, temperature=1, num_repeats=1):
    """
    Function that thins a chain with a Bernoulli process.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :param temperature: temperature of the Bernoulli process. If T=1 then
        this produces a unit weight chain.
    :param num_repeats: number of repetitions of the Bernoulli process.
    :return: a :class:`~getdist.mcsamples.MCSamples` chain with the
        reweighted chain.
    """
    # check input:
    # get the trial vector:
    test = np.log(chain.weights / np.sum(chain.weights))
    new_weights = np.exp((1. - temperature) * test)
    test = temperature*(test - np.amax(test))
    # do the trial:
    _filter = np.zeros(len(test)).astype(bool)
    _sample_repeat = np.zeros(len(test)).astype(int)
    for i in range(num_repeats):
        _temp = np.random.binomial(1, np.exp(test))
        _sample_repeat += _temp.astype(int)
        _filter = np.logical_or(_filter, _temp.astype(bool))
    new_weights = _sample_repeat*new_weights
    # filter the chain:
    chain.setSamples(samples=chain.samples[_filter, :],
                     weights=new_weights[_filter],
                     loglikes=chain.loglikes[_filter])
    # update:
    chain._weightsChanged()
    chain.updateBaseStatistics()
    #
    return chain

###############################################################################


def random_samples_reshuffle(chain):
    """
    Performs a coherent random reshuffle of the samples.

    :param chain: :class:`~getdist.mcsamples.MCSamples` the input chain.
    :return: a :class:`~getdist.mcsamples.MCSamples` chain with the
        reshuffled chain.
    """
    # check input:
    # get the reshuffling vector:
    _reshuffle_indexes = np.arange(len(chain.weights))
    np.random.shuffle(_reshuffle_indexes)
    # filter the chain:
    if hasattr(chain, 'loglikes') and chain.loglikes is not None:
        chain.setSamples(samples=chain.samples[_reshuffle_indexes, :],
                         weights=chain.weights[_reshuffle_indexes],
                         loglikes=chain.loglikes[_reshuffle_indexes])
    else:
        chain.setSamples(samples=chain.samples[_reshuffle_indexes, :],
                         weights=chain.weights[_reshuffle_indexes])
    # update:
    chain._weightsChanged()
    chain.updateBaseStatistics()
    #
    return chain

###############################################################################


def make_list(elements):
    """
    Checks if elements is a list.
    If yes returns elements without modifying it.
    If not creates and return a list with elements inside.

    :param elements: an element or a list of elements.
    :return: a list containing elements.
    """
    if isinstance(elements, (list, tuple)):
        return elements
    else:
        return [elements]

###############################################################################


def PDM_to_vector(pdm):
    """
    Transforms a positive definite matrix of dimension :math:`d \\times d`
    into an unconstrained vector of dimension :math:`d(d+1)/2`.
    This does not use the Cholesky decomposition since we need guarantee of
    strictly positive definiteness.

    The absolute values of the elements with indexes of the returned vector
    that satisfy:

    .. code-block:: python

        np.tril_indices(d, 0)[0] == np.tril_indices(d, 0)[1]

    are the eigenvalues of the matrix. The sign of these elements define
    the orientation of the eigenvectors.

    Note that this is not strictly the inverse of
    :meth:`tensiometer.utilities.stats_utilities.vector_to_PDM`
    since there are a number of discrete symmetries in the definition of the
    eigenvectors that we ignore since they are irrelevant for the sake of
    representing the matrix.

    :param pdm: the input positive definite matrix.
    :return: output vector representation.
    :reference: https://arxiv.org/abs/1906.00587
    """
    # get dimension:
    d = pdm.shape[0]
    # get the eigenvalues of the matrix:
    Lambda, Phi = np.linalg.eigh(pdm)
    # get triangular decomposition:
    (P, L, U) = scipy.linalg.lu(Phi.T, permute_l=False)
    # WR decomposition:
    Q, R = np.linalg.qr(L)
    L = np.dot(np.dot(R, U), L)
    # pivot the eigenvalues:
    Lambda2 = np.dot(np.dot(P.T, np.diag(Lambda)), P)
    # prepare output:
    mat = L
    mat[np.diag_indices(d)] = np.sign(mat[np.diag_indices(d)])*np.diag(Lambda2)
    #
    return mat[np.tril_indices(d, 0)]


def vector_to_PDM(vec):
    """
    Transforms an unconstrained vector of dimension :math:`d(d+1)/2`
    into a positive definite matrix of dimension :math:`d \\times d`.
    In the input vector the eigenvalues are in the positions where

    The absolute values of the elements with indexes of the input vector
    that satisfy:

    .. code-block:: python

        np.tril_indices(d, 0)[0] == np.tril_indices(d, 0)[1]

    are the eigenvalues of the matrix. The sign of these elements define
    the orientation of the eigenvectors.

    The purpose of this function is to allow optimization over the space
    of positive definite matrices that is either unconstrained or
    has constraints on the condition number of the matrix.

    :param pdm: the input vector.
    :return: output positive definite matrix.
    :reference: https://arxiv.org/abs/1906.00587
    """
    d = int(np.sqrt(1 + 8*len(vec)) - 1)//2
    L = np.zeros((d, d))
    # get the diagonal with eigenvalues:
    L[np.tril_indices(d, 0)] = vec
    Lambda2 = np.diag(np.abs(L[np.diag_indices(d)]))
    L[np.diag_indices(d)] = np.sign(L[np.diag_indices(d)]) * np.ones(d)
    # qr decompose L:
    Q, R = np.linalg.qr(L)
    Phi2 = np.dot(L, np.linalg.inv(R))
    # rebuild matrix
    return np.dot(np.dot(Phi2.T, Lambda2), Phi2)

###############################################################################


def whiten_samples(samples, weights):
    """
    Rescales samples by the square root of their inverse covariance.
    The resulting samples have identity covariance. This amounts to a change of
    coordinates so the physical meaning of different coordinates is changed.

    :param samples: the input samples.
    :param weights: the input weights of the samples.
    :return: whitened samples with identity covariance.
    """
    # compute sample covariance:
    _cov = np.cov(samples.T, aweights=weights)
    # compute its inverse square root:
    _temp = sqrtm(QR_inverse(_cov))
    # whiten the samples:
    white_samples = samples.dot(_temp)
    #
    return white_samples

###############################################################################


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    :param points: An num-observations by num-dimensions array of observations
    :param thresh: The modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.
    :return: A num-observations-length boolean array.
    :reference: Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

###############################################################################


def filter_kwargs(dict_to_filter, function_with_kwargs):
    """
    Inspects a function signature and returns the correct kwargs. Usefull to
    isolate kwargs for a third party library.

    :param dict_to_filter: dictionary to filter
    :param function_with_kwargs: function to inspect
    """
    sig = inspect.signature(function_with_kwargs)
    filter_keys = [param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    filtered_dict = {filter_key: dict_to_filter[filter_key] for filter_key in filter_keys if filter_key in dict_to_filter.keys()}
    return filtered_dict
