"""
This file contains a set of utilities to compute tensor eigenvalues
since there is no standard library to do so.
"""

###############################################################################
# initial imports:

from itertools import permutations
import numpy as np
import scipy.linalg
import scipy.integrate
import scipy
import sys
import functools

###############################################################################
# Utilities:


def random_symm_tensor(d, m, vmin=0.0, vmax=1.0):
    """
    Generate a random symmetric tensor of dimension d and rank m.
    There is no guarantee on the distribution of the elements, just that
    they are all different...

    :param d: number of dimensions
    :param m: rank of the tensor
    :param vmin: minimum value of the tensor elements
    :param vmax: maximum value of the tensor elements
    :returns: the random symmetric tensor
    """
    # output tensor:
    tensor_shape = [d for i in range(m)]
    out_tens = np.zeros(tensor_shape)
    # generate elements:
    in_tens = vmax*np.random.rand(*tensor_shape) + vmin
    # symmetrize:
    num_perm = 0
    for i in permutations([i for i in range(m)]):
        num_perm += 1
        out_tens += np.transpose(in_tens, axes=list(i))
    out_tens = out_tens/num_perm
    #
    return out_tens


def random_symm_positive_tensor(d, m, vmin=0.0, vmax=1.0):
    """
    Generate a random positive symmetric tensor of even order.
    There is no guarantee on the distribution of the elements, just that
    they are all different...

    :param d: number of dimensions
    :param m: rank of the tensor
    :param vmin: minimum value of the tensor elements
    :param vmax: maximum value of the tensor elements
    :returns: the random symmetric tensor
    """
    # Generate the starting tensor:
    A = random_symm_tensor(d, m, vmin=vmin, vmax=vmax)
    # Compute the product with itself:
    A = np.tensordot(A, A, ([i for i in range(m)][m//2:],
                            [i for i in range(m)][:m//2]))
    #
    return A


def identity_tensor(d, m):
    """
    Returns the identity tensor that has 1 on the (multidimensional) diagonal
    and 0 elsewhere.

    :param d: number of dimensions
    :param m: rank of the tensor
    :returns: the random symmetric tensor
    """
    # output tensor:
    tensor_shape = [d for i in range(m)]
    out_tens = np.zeros(tensor_shape)
    # initialize:
    for i in range(d):
        out_tens[tuple([i for j in range(m)])] = 1.
    #
    return out_tens


def number_eigenvalues(d, m):
    """
    Number of eigenvalues of a symmetric tensor of order m and dimension d.

    :param d: number of dimensions
    :param m: rank of the tensor
    :returns: the number of eigenvalues
    """
    return d*(m-1)**(d-1)


def tensor_deflation(A, l, x):
    """
    Deflates a tensor by a scalar multiplied my a vector.

    :param A: the input tensor
    :param l: the scalar to deflate
    :param x: the vector to deflate
    :returns: the deflated tensor :math:`A - l x^m`
    """
    # get dimension and rank:
    m = len(A.shape)
    # prepare the outer product of the input vector:
    vec = x
    for i in range(m-1):
        vec = np.multiply.outer(vec, x)
    #
    return A - l * vec


###############################################################################
# Tensor contractions utilities


def tensor_contraction_brute_1(A, x, n=1):
    """
    Contracts a symmetric tensor of rank m with a given vector n times.
    This function is meant to be as fast as possible, no check is
    performed.

    :param A: the inmput symmetric tensor of rank m
    :param x: the input vector to contract
    :param n: the number of times to contract
    :returns: the tensor contracted n times. This is a tensor of rank m-n.
    """
    res = A
    for i in range(n):
        res = np.dot(res, x)
    return res


def tensor_contraction_brute_2(A, x, n=1):
    """
    Contracts a symmetric tensor of rank m with a given vector n times.
    This function is meant to be as fast as possible, no check is
    performed.

    :param A: the inmput symmetric tensor of rank m
    :param x: the input vector to contract
    :param n: the number of times to contract
    :returns: the tensor contracted n times. This is a tensor of rank m-n.
    """
    return functools.reduce(np.dot, [A]+[x for i in range(n)])


# choose the contraction function to use:
tensor_contraction = tensor_contraction_brute_2

###############################################################################
# Optimization on the sphere:


def eu_to_sphere_grad(x, egrad):
    """
    Converts euclidean gradient to gradient on the n-sphere.

    :param x: vector x on the sphere
    :param egrad: euclidean gradient
    """
    return egrad - np.tensordot(x, egrad, axes=x.ndim) * x


def eu_to_sphere_hess(x, egrad, ehess, u):
    """
    Derivative of gradient in direction u (tangent to the sphere)

    :param x: vector x on the sphere
    :param egrad: euclidean gradient
    :param ehess: euclidean Hessian matrix
    :param u: direction vector that should belong on the tangent space
        of the sphere
    """
    ehess = np.dot(ehess, u)
    temp = ehess - np.tensordot(x, ehess, axes=x.ndim) * x \
        - np.tensordot(x, egrad, axes=x.ndim) * u
    return temp

###############################################################################
# Rayleight quotient definition and derivatives:


def tRq(x, A):
    """
    Symmetric Tensor Rayleigh quotient.

    :param x: the input vector
    :param A: the input tensor
    """
    # get dimension and rank:
    m = len(A.shape)
    # do the products:
    return tensor_contraction(A, x, m)


def tRq_nder(x, A, n):
    """
    Euclidean derivative of order n of the Tensor Rayleigh quotient problem.

    :param x: the input vector
    :param A: the input tensor
    :param n: the order of the derivative
    """
    # get dimension and rank:
    m = len(A.shape)
    # do the products:
    res = tensor_contraction(A, x, m-n)
    # get the prefactor:
    fac = np.prod([(m - j) for j in range(n)]).astype(np.float)
    #
    return fac*res

###############################################################################
# manifold brute force maximization:

import autograd.numpy as anp

# prevent pymanopt from running with tensorflow:
import pymanopt.tools.autodiff._tensorflow as ptf
ptf.tf = None
from pymanopt.manifolds import Sphere
from pymanopt import Problem
import pymanopt.solvers


def _tRq_brute_autograd(x, A):
    """
    Tensor Rayleigh quotient. Brute force implementation with autograd.
    """
    # get dimension and rank:
    m = len(A.shape)
    # do the products:
    res = functools.reduce(anp.dot, [A]+[x for i in range(m)])
    #
    return res


def max_tRq_brute(A, feedback=0, optimizer='ParticleSwarm', **kwargs):
    """
    Brute force maximization of the Tensor Rayleigh quotient on the sphere.
    Optimization is performed with Pymanopt.

    :param A: the input tensor
    :param feedback: the feedback level for pymanopt
    :param optimizer: the name of the pymanopt minimizer
    :param kwargs: keyword arguments to pass to the pymanopt solver
    """
    # get dimension and rank:
    d = A.shape[0]
    # initialize:
    manifold = Sphere(d)
    problem = Problem(manifold=manifold,
                      cost=lambda x: -_tRq_brute_autograd(x, A),
                      verbosity=feedback)
    # optimization:
    if optimizer == 'ParticleSwarm':
        solver = pymanopt.solvers.ParticleSwarm(logverbosity=0, **kwargs)
        Xopt = solver.solve(problem)
    elif optimizer == 'TrustRegions':
        solver = pymanopt.solvers.TrustRegions(logverbosity=0, **kwargs)
        Xopt = solver.solve(problem)
    # finalize:
    return _tRq_brute_autograd(Xopt, A), Xopt


def tRq_brute_2D(A, num_points=2000):
    """
    Brute force maximization of the Tensor Rayleigh quotient on the circle.
    Works for problems of any rank and 2 dimensions. Since the problem
    is one dimensional samples the function on num_points and returns the
    maximum.

    :param A: the input tensor
    :param num_points: the number of points of the search
    """
    theta = np.linspace(0., np.pi, num_points)
    res = np.array([tRq([x, y], A)
                    for x, y in zip(np.cos(theta), np.sin(theta))])
    sol = np.where(np.diff(np.sign(res[1:]-res[0:-1])))
    eig = np.concatenate((res[sol], res[sol]))
    eigv = np.concatenate(([[x, y] for x, y in zip(np.cos(theta[sol]),
                                                   np.sin(theta[sol]))],
                          [[x, y] for x, y in zip(np.cos(theta[sol]+np.pi),
                                                  np.sin(theta[sol]+np.pi))]
                           ))
    #
    return eig, eigv


###############################################################################
# power iterations:


def max_tRq_power(A, maxiter=1000, tol=1.e-10, x0=None, history=False):
    """
    Symmetric power iterations, also called S-HOPM, for tensors eigenvalues.
    Described in https://arxiv.org/abs/1007.1267
    The algorithm is not guaranteed to produce the global maximum but only
    a convex maximum. We advice to run the algorithm multiple times to
    make sure that the solution that is found is the global maximum.

    :param A: the input symmetric tensor
    :param maxiter: (default 500) maximum number of iterations
    :param tol: (default 1.e-10) tolerance on the solution of the eigenvalue
        problem
    :param x0: (default random on the sphere) starting point
    :param history: (default False) wether to return the history of the
        power iterations
    """
    # get dimension and rank:
    d, m = A.shape[0], len(A.shape)
    # get random (normalized) initial guess:
    if x0 is None:
        x = 2.*np.random.rand(d) - 1.
    else:
        x = x0
    x = x / np.sqrt(np.dot(x, x))
    # initialization:
    res_history = []
    # do the power iterations:
    for i in range(maxiter):
        # precomputations:
        Axmm1 = tensor_contraction(A, x, m-1)
        Axm = tensor_contraction(Axmm1, x)
        # save history:
        res_history.append(Axm)
        # check for termination:
        test = Axmm1 - Axm * x
        test = np.sqrt(np.dot(test, test))
        if test < tol:
            break
        # perform the iteration:
        x = Axmm1
        x = x / np.sqrt(np.dot(x, x))
    # check for rightful termination:
    if i == maxiter-1:
        print('WARNING(max_tRq_power)'
              + ' maximum number of iterations ('+str(maxiter)+') exceeded.')
    # return:
    if history:
        return Axm, x, np.array(res_history)
    else:
        return Axm, x


def max_tRq_shift_power(A, alpha, maxiter=1000, tol=1.e-10, x0=None,
                        history=False):
    """
    Shifted symmetric power iterations, also called SS-HOPM, for tensor
    eigenvalues. Described in https://arxiv.org/abs/1007.1267
    The algorithm is not guaranteed to produce the global maximum but only
    a convex maximum. We advice to run the algorithm multiple times to
    make sure that the solution that is found is the global maximum.

    :param A: the input symmetric tensor
    :param alpha: the input fixed shift
    :param maxiter: (default 500) maximum number of iterations.
    :param tol: (default 1.e-10) tolerance on the solution of the eigenvalue
        problem
    :param x0: (default random on the sphere) starting point
    :param history: (default False) wether to return the history of the
        power iterations
    """
    # get dimension and rank:
    d, m = A.shape[0], len(A.shape)
    # get random (normalized) initial guess:
    if x0 is None:
        x = 2.*np.random.rand(d) - 1.
    else:
        x = x0
    x = x / np.sqrt(np.dot(x, x))
    # initialization:
    res_history = []
    # do the power iterations:
    for i in range(maxiter):
        # precomputations:
        Axmm1 = tensor_contraction(A, x, m-1)
        Axm = tensor_contraction(Axmm1, x)
        # save history:
        res_history.append(Axm)
        # check for termination:
        test = Axmm1 - Axm * x
        test = np.sqrt(np.dot(test, test))
        if test < tol:
            break
        # perform the iteration:
        x = Axmm1 + alpha * x
        x = x / np.sqrt(np.dot(x, x))
    # check for rightful termination:
    if i == maxiter-1:
        print('WARNING(max_tRq_shift_power)'
              + ' maximum number of iterations ('+str(maxiter)+') exceeded.')
    # return:
    if history:
        return Axm, x, np.array(res_history)
    else:
        return Axm, x


def max_tRq_geap(A, tau=1.e-6, maxiter=1000, tol=1.e-10, x0=None,
                 history=False):
    """
    Shifted adaptive power iterations algorithm, also called GEAP, for tensor
    eigenvalues. Described in https://arxiv.org/pdf/1401.1183.pdf
    The algorithm is not guaranteed to produce the global maximum but only
    a convex maximum. We advice to run the algorithm multiple times to
    make sure that the solution that is found is the global maximum.

    :param A: the input symmetric tensor
    :param tau: (default 1.e-6) tolerance on being positive definite
    :param maxiter: (default 500) maximum number of iterations.
    :param tol: (default 1.e-10) tolerance on the solution of the eigenvalue
        problem
    :param x0: (default random on the sphere) starting point
    :param history: (default False) wether to return the history of the
        power iterations
    """
    # get dimension and rank:
    d, m = A.shape[0], len(A.shape)
    # get random (normalized) initial guess:
    if x0 is None:
        x = 2.*np.random.rand(d) - 1.
    else:
        x = x0
    x = x / np.sqrt(np.dot(x, x))
    # initialization:
    res_history = []
    # do the power iterations:
    for i in range(maxiter):
        # precompute:
        Axmm2 = tensor_contraction(A, x, m-2)
        Axmm1 = tensor_contraction(Axmm2, x)
        Axm = tensor_contraction(Axmm1, x)
        H_k = m*(m-1)*Axmm2
        alpha_k = max(0., (tau - np.amin(np.linalg.eigvals(H_k)))/m)
        # save history:
        res_history.append([Axm, alpha_k])
        # check for termination:
        test = Axmm1 - Axm * x
        test = np.sqrt(np.dot(test, test))
        if test < tol:
            break
        # iteration:
        x = Axmm1 + alpha_k * x
        x = x / np.sqrt(np.dot(x, x))
    # check for rightful termination:
    if i == maxiter-1:
        print('WARNING(max_tRq_geap_power)'
              + ' maximum number of iterations ('+str(maxiter)+') exceeded.')
    # return:
    if history:
        return Axm, x, np.array(res_history)
    else:
        return Axm, x


###############################################################################
# maximum Z-eigenvalue and Z-eigenvector of a tensor:


def tRq_dyn_sys_brute(t, x, A, d, m):
    """
    Dynamical system to solve for the biggest tensor eigenvalue.
    Derivative function.
    Described in https://arxiv.org/abs/1805.00903

    :param t: input time
    :param x: input position
    :param A: input symmetric tensor
    :param d: input number of dimensions
    :param m: input rank of the tensor A
    :return: the derivative of the dynamical system
    """
    # do the product and compute the 2D matrix:
    in_A = tensor_contraction(A, x, m-2)
    # eigenvalues:
    eig, eigv = np.linalg.eig(in_A)
    # selection:
    idx = np.argmax(np.real(eig))
    out_x = np.real(eigv[:, idx])
    out_x = out_x*np.sign(out_x[0])
    #
    return out_x - x


def max_tRq_dynsys(A, maxiter=1000, tol=1.e-10, x0=None, h0=0.5,
                   history=False):
    """
    Solves for the maximum eigenvalue with a dynamical system.
    Described in https://arxiv.org/abs/1805.00903
    Uses odeint to perform the differential equation evolution.

    :param A: the input symmetric tensor
    :param maxiter: (default 500) maximum number of iterations.
    :param tol: (default 1.e-10) tolerance on the solution of the eigenvalue
        problem
    :param x0: (default random on the sphere) starting point
    :param h0: (default 0.5) initial time step
    :param history: (default False) wether to return the history of the
        power iterations
    """
    # get dimension and rank:
    d, m = A.shape[0], len(A.shape)
    # get random (normalized) initial guess:
    if x0 is None:
        x = 2.*np.random.rand(d) - 1.
    else:
        x = x0
    x = x / np.sqrt(np.dot(x, x))
    # initialize:
    t1 = 10.*h0
    res_history = []
    # perform the dynamical system iterations:
    for i in range(maxiter):
        res = scipy.integrate.odeint(lambda x, t: tRq_dyn_sys_brute(t,
                                     x, A, d, m),
                                     y0=x, t=[0., t1], h0=h0,
                                     full_output=True)
        # process results:
        h0 = res[1]['hu'][0]
        t1 = 10.*h0
        x = res[0][-1, :]
        Axmm1 = tensor_contraction(A, x, m-1)
        Axm = tensor_contraction(Axmm1, x)
        # save history:
        res_history.append([Axm, h0])
        # termination 1, the Rayleight coefficient should not decrease:
        # if last_Axm > Axm:
        #     break
        # last_Axm = Axm
        # termination 2, the inverse stepsize cannot be smaller than machine
        # precision, an equilibrium has been reached
        if 1./h0 < 10*np.finfo(np.float32).eps:
            break
        # termination 3, the eigenvalue problem is solved to desired accuracy:
        test = Axmm1 - Axm * x
        test = np.sqrt(np.dot(test, test))
        if test < tol:
            break
    # check for rightful termination:
    if i == maxiter-1:
        print('WARNING(ds_max_eig)'
              + ' maximum number of iterations ('+str(maxiter)+') exceeded.')
    # return:
    if history:
        return Axm, x, np.array(res_history)
    else:
        return Axm, x


###############################################################################
# Generalized Rayleight quotient definition and derivatives:


def GtRq(x, A, B):
    """
    Generalized tensor Rayleigh quotient.

    :param x: the input vector
    :param A: the input tensor at the numerator
    :param B: the input tensor at the denumerator
    """
    # get rank:
    m = len(A.shape)
    #
    return tensor_contraction(A, x, m) / tensor_contraction(B, x, m)


def GtRq_Jac_brute(x, m, Axm, Bxm, Axmm1, Bxmm1):
    """
    The Euclidean Jacobian of the Generalized tensor Rayleigh quotient problem.
    Taken from https://arxiv.org/abs/1401.1183
    Requires precomputations since many things can be cached.

    :param x: the input vector
    :param m: the rank of the tensor
    :param Axm: first tensor contraction A*x^m
    :param Bxm: second tensor contraction B*x^m
    :param Axmm1: first tensor contraction A*x^(m-1)
    :param Bxmm1: second tensor contraction B*x^(m-1)
    """
    #
    return m / Bxm * (Axm*x + Axmm1 - Axm/Bxm*Bxmm1)


def GtRq_Hess_brute(x, m, Axm, Bxm, Axmm1, Bxmm1, Axmm2, Bxmm2):
    """
    The Euclidean Hessian of the Generalized tensor Rayleigh quotient problem.
    Taken from https://arxiv.org/abs/1401.1183
    Requires precomputations since many things can be cached.

    :param x: the input vector
    :param m: the rank of the tensor
    :param Axm: first tensor contraction A*x^m
    :param Bxm: second tensor contraction B*x^m
    :param Axmm1: first tensor contraction A*x^(m-1)
    :param Bxmm1: second tensor contraction B*x^(m-1)
    :param Axmm2: first tensor contraction A*x^(m-2)
    :param Bxmm2: second tensor contraction B*x^(m-2)
    """
    # get dimension:
    d = len(x)
    # start accumulating Hessian:
    Hess = m**2*Axm/Bxm**3*(np.outer(Bxmm1, Bxmm1) + np.outer(Bxmm1, Bxmm1))
    Hess += m/Bxm*((m-1.)*Axmm2 + Axm*(np.identity(d)+(m-2.)*np.outer(x, x))
                   + m*(np.outer(Axmm1, x) + np.outer(x, Axmm1)))
    Hess -= m/Bxm**2*((m-1.)*Axm*Bxmm2 + m*(np.outer(Axmm1, Bxmm1)
                      + np.outer(Bxmm1, Axmm1))
                      + m*Axm*(np.outer(x, Bxmm1) + np.outer(Bxmm1, x)))
    #
    return Hess

###############################################################################
# Brute force maximization:


def _GtRq_brute_autograd(x, A, B):
    """
    Generalized Tensor Rayleigh quotient. Brute force implementation.
    """
    # get dimension and rank:
    m = len(A.shape)
    # do the products:
    res1 = functools.reduce(anp.dot, [A]+[x for i in range(m)])
    res2 = functools.reduce(anp.dot, [B]+[x for i in range(m)])
    #
    return res1 / res2


def max_GtRq_brute(A, B, feedback=0, optimizer='ParticleSwarm', **kwargs):
    """
    Brute force maximization of the Generalized Tensor Rayleigh quotient
    on the sphere. Optimization is performed with Pymanopt.

    :param A: the input tensor
    :param B: the second input tensor
    :param feedback: the feedback level for pymanopt
    :param optimizer: the name of the pymanopt minimizer
    :param kwargs: keyword arguments to pass to the pymanopt solver
    """
    # get dimension:
    d = A.shape[0]
    # initialize:
    manifold = Sphere(d)
    problem = Problem(manifold=manifold,
                      cost=lambda x: -_GtRq_brute_autograd(x, A, B),
                      verbosity=feedback)
    # optimization:
    if optimizer == 'ParticleSwarm':
        solver = pymanopt.solvers.ParticleSwarm(logverbosity=0, **kwargs)
        Xopt = solver.solve(problem)
    elif optimizer == 'TrustRegions':
        solver = pymanopt.solvers.TrustRegions(logverbosity=0, **kwargs)
        Xopt = solver.solve(problem)
    # finalize:
    return _GtRq_brute_autograd(Xopt, A, B), Xopt


def GtRq_brute_2D(A, B, num_points=2000):
    """
    Brute force maximization of the Generalized Tensor Rayleigh quotient
    on the circle.
    Works for problems of any rank and 2 dimensions. Since the problem
    is one dimensional samples the function on num_points and returns the
    maximum.

    :param A: the first input tensor
    :param B: the second input tensor
    :param num_points: the number of points of the search
    """
    theta = np.linspace(0., np.pi, num_points)
    res = np.array([GtRq([x, y], A, B)
                   for x, y in zip(np.cos(theta), np.sin(theta))])
    sol = np.where(np.diff(np.sign(res[1:]-res[0:-1])))
    eig = np.concatenate((res[sol], res[sol]))
    eigv = np.concatenate(([[x, y] for x, y in zip(np.cos(theta[sol]),
                                                   np.sin(theta[sol]))],
                           [[x, y] for x, y in zip(np.cos(theta[sol]+np.pi),
                                                   np.sin(theta[sol]+np.pi))]
                           ))
    #
    return eig, eigv


###############################################################################
# Power method (GEAP):


def max_GtRq_geap_power(A, B, maxiter=1000, tau=1.e-6, tol=1.e-10, x0=None,
                        history=False):
    """
    Shifted adaptive power iterations algorithm, also called GEAP, for the
    Generalized Tensor Rayleigh quotient.
    Described in https://arxiv.org/pdf/1401.1183.pdf
    The algorithm is not guaranteed to produce the global maximum but only
    a convex maximum. We advice to run the algorithm multiple times to
    make sure that the solution that is found is the global maximum.

    :param A: the input symmetric tensor
    :param B: the second input symmetric tensor
    :param maxiter: (default 500) maximum number of iterations.
    :param tau: (default 1.e-6) tolerance on being positive definite
    :param tol: (default 1.e-10) tolerance on the solution of the eigenvalue
        problem
    :param x0: (default random on the sphere) starting point
    :param history: (default False) wether to return the history of the
        power iterations
    """
    # get dimension and rank:
    d, m = A.shape[0], len(A.shape)
    # get random (normalized) initial guess:
    if x0 is None:
        x = 2.*np.random.rand(d) - 1.
    else:
        x = x0
    x = x / np.sqrt(np.dot(x, x))
    # maximum iterations:
    if maxiter is None:
        maxiter = sys.maxsize
    # initialize:
    res_history = []
    # do the power iterations:
    for i in range(maxiter):
        # precomputations:
        Axmm2 = functools.reduce(np.dot, [A]+[x for i in range(m-2)])
        Bxmm2 = functools.reduce(np.dot, [B]+[x for i in range(m-2)])
        Axmm1 = np.dot(Axmm2, x)
        Bxmm1 = np.dot(Bxmm2, x)
        Axm = np.dot(Axmm1, x)
        Bxm = np.dot(Bxmm1, x)
        lambda_k = Axm/Bxm
        # termination check:
        term = Axmm1 - lambda_k * Bxmm1
        diff = np.sqrt(np.dot(term, term))
        if diff < tol:
            break
        # quantities:
        H_k = GtRq_Hess_brute(x, m, Axm, Bxm, Axmm1, Bxmm1, Axmm2, Bxmm2)
        alpha_k = max(0., (tau - np.amin(np.linalg.eigvals(H_k)))/m)
        # advance and normalize:
        x = (Axmm1 - lambda_k*Bxmm1 + (alpha_k + lambda_k)*Bxm*x)
        x = x / np.sqrt(np.dot(x, x))
        # history:
        res_history.append([lambda_k, alpha_k])
        # if going for too long warn:
        if i % 10000 == 0 and i > 1:
            print('WARNING(max_GtRq_geap_power)'
                  + ' large number of iterations ('+str(i)+').')
    # check for rightful termination:
    if i == maxiter-1:
        print('WARNING(max_GtRq_geap_power)'
              + ' maximum number of iterations ('+str(maxiter)+') exceeded.')
    # returns:
    if history:
        return lambda_k, x, np.array(res_history)
    else:
        return lambda_k, x
