"""
This file implements information geometry methods working on normalizing flows.

This file needs work and is not complete.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import scipy

# tensorflow imports:
import tensorflow as tf

###############################################################################
# CPC decomposition in tensorflow:


@tf.function
def tf_CPC_decomposition(matrix_a, matrix_b):
    """
    Covariant Principal Components decomposition impolemented in tensorflow.

    Args:
        matrix_a (tf.Tensor): Input matrix A.
        matrix_b (tf.Tensor): Input matrix B.

    Returns:
        tf.Tensor: Eigenvalues of A_prime.
        tf.Tensor: Eigenvectors of A_prime.
    """
    # compute the eigenvalues of b, lambda_b:
    _lambda_b, _phi_b = tf.linalg.eigh(matrix_b)
    _sqrt_lambda_b = tf.linalg.diag(1./tf.math.sqrt(_lambda_b))
    _phib_prime = tf.matmul(_phi_b, _sqrt_lambda_b)
    #
    trailing_axes = [-1, -2]
    leading = tf.range(tf.rank(_phib_prime) - len(trailing_axes))
    trailing = trailing_axes + tf.rank(_phib_prime)
    new_order = tf.concat([leading, trailing], axis=0)
    _phib_prime_T = tf.transpose(_phib_prime, new_order)
    #
    _a_prime = tf.matmul(tf.matmul(_phib_prime_T, matrix_a), _phib_prime)
    _lambda, _phi_a = tf.linalg.eigh(_a_prime)
    _phi = tf.matmul(tf.matmul(_phi_b, _sqrt_lambda_b), _phi_a)
    #
    return _lambda, _phi

###############################################################################
# Simple ODE solver:

# need here a simple ODE method

###############################################################################
# KL methods:


def _naive_eigenvalue_ode_abs(flow, t, y, reference):
    """
    Solve naively the dynamical equation for eigenvalues in abstract space.
    """
    # preprocess:
    x = flow.cast([y])
    # map to original space to compute Jacobian (without inversion):
    x_par = flow.map_to_original_coord(x)
    # precompute Jacobian and its derivative:
    jac = flow.inverse_jacobian(x_par)[0]
    jac_T = tf.transpose(jac)
    jac_jac_T = tf.matmul(jac, jac_T)
    # compute eigenvalues:
    eig, eigv = tf.linalg.eigh(jac_jac_T)
    temp = tf.matmul(tf.transpose(eigv), tf.transpose([reference]))
    idx = tf.math.argmax(tf.abs(temp))[0]
    w = tf.convert_to_tensor([tf.math.sign(temp[idx]) * eigv[:, idx]])
    #
    return w


def solve_eigenvalue_ode_abs(self, y0, n, length=1.5, side='both', integrator_options=None, num_points=100, **kwargs):
    """
    Solve eigenvalue problem in abstract space
    side = '+', '-', 'both'
    """
    # define solution points:
    solution_times = tf.linspace(0., length, num_points)
    # compute initial PCA:
    x_abs = tf.convert_to_tensor([y0])
    x_par = self.map_to_original_coord(x_abs)
    jac = self.inverse_jacobian(x_par)[0]
    jac_T = tf.transpose(jac)
    jac_jac_T = tf.matmul(jac, jac_T)
    # compute eigenvalues at initial point:
    eig, eigv = tf.linalg.eigh(jac_jac_T)
    # solve forward:
    if side == '+' or side == 'both':
        # initialize solution:
        temp_sol_1 = np.zeros((num_points-1, self.num_params))
        temp_sol_dot_1 = np.zeros((num_points-1, self.num_params))
        # initialize forward integration:
        solver = scipy.integrate.ode(self._naive_eigenvalue_ode_abs)
        if integrator_options is not None:
            solver.set_integrator(**integrator_options)
        solver.set_initial_value(y0, 0.)
        reference = eigv[:, n]
        yt = y0.numpy()
        yprime = reference
        # do the time steps:
        for ind, t in enumerate(solution_times[1:]):
            # set the reference:
            solver.set_f_params(reference)
            # advance solver:
            try:
                yt = solver.integrate(t)
                yprime = self._naive_eigenvalue_ode_abs(t, yt, reference)
            except:
                pass
            # update reference:
            reference = yprime[0]
            # save out:
            temp_sol_1[ind] = yt.copy()
            temp_sol_dot_1[ind] = yprime.numpy().copy()
        # return if needed:
        if side == '+':
            traj = np.concatenate((x_abs.numpy(), temp_sol_1))
            vel = np.concatenate(([eigv[:, n].numpy()], temp_sol_dot_1))
            return solution_times, traj, vel
    # solve backward:
    if side == '-' or side == 'both':
        # initialize solution:
        temp_sol_2 = np.zeros((num_points-1, self.num_params))
        temp_sol_dot_2 = np.zeros((num_points-1, self.num_params))
        # initialize backward integration:
        solver = scipy.integrate.ode(self._naive_eigenvalue_ode_abs)
        if integrator_options is not None:
            solver.set_integrator(**integrator_options)
        solver.set_initial_value(y0, 0.)
        reference = - eigv[:, n]
        yt = y0.numpy()
        yprime = reference
        for ind, t in enumerate(solution_times[1:]):
            # set the reference:
            solver.set_f_params(reference)
            # advance solver:
            try:
                yt = solver.integrate(t)
                yprime = self._naive_eigenvalue_ode_abs(t, yt, reference)
            except:
                pass
            # update reference:
            reference = yprime[0]
            # save out:
            temp_sol_2[ind] = yt.copy()
            temp_sol_dot_2[ind] = yprime.numpy().copy()
        # return if needed:
        if side == '-':
            traj = np.concatenate((temp_sol_2[::-1], x_abs.numpy()))
            vel = np.concatenate((-temp_sol_dot_2[::-1], [eigv[:, n].numpy()]))
            return -solution_times, traj, vel
    # patch solutions:
    times = np.concatenate((-solution_times[::-1], solution_times[1:]))
    traj = np.concatenate((temp_sol_2[::-1], x_abs.numpy(), temp_sol_1))
    vel = np.concatenate((-temp_sol_dot_2[::-1], [eigv[:, n].numpy()], temp_sol_dot_1))
    #
    return times, traj, vel

def solve_eigenvalue_ode_par(self, y0, n, **kwargs):
    """
    Solve eigenvalue ODE in parameter space
    """
    # go to abstract space:
    x_abs = self.map_to_abstract_coord(self.cast([y0]))[0]
    # call solver:
    times, traj, vel = self.solve_eigenvalue_ode_abs(x_abs, n, **kwargs)
    # convert back:
    traj = self.map_to_original_coord(self.cast(traj))
    #
    return times, traj

# solve full transport in abstract space:
@tf.function()
def eigenvalue_ode_abs_temp_3(self, t, y):
    # unpack y:
    x = self.cast([y[:self.num_params]])
    w = self.cast([y[self.num_params:-1]])
    alpha = self.cast([y[-1]])
    # map to original space to compute Jacobian (without inversion):
    x_par = self.map_to_original_coord(x)
    # precompute Jacobian and its derivative:
    jac = self.inverse_jacobian(x_par)[0]
    #djac = coord_jacobian_derivative(x_par)[0]
    jacm1 = self.direct_jacobian(x_par)[0]
    jac_T = tf.transpose(jac)
    jac_jac_T = tf.matmul(jac, jac_T)
    Id = tf.eye(self.num_params)
    # select the eigenvector that we want to follow based on the solution to the continuity equation:
    eig, eigv = tf.linalg.eigh(jac_jac_T)
    idx = tf.math.argmax(tf.abs(tf.matmul(tf.transpose(eigv), tf.transpose(w))))[0]
    tilde_w = tf.convert_to_tensor([eigv[:, idx]])
    dot_J = tf.einsum('k, lk, ijl -> ji', tilde_w[0], jacm1, djac)
    # equation for alpha:
    alpha_dot = 2.*tf.matmul(tf.matmul(tilde_w, jac), tf.matmul(dot_J, tf.transpose(tilde_w)))
    # equation for wdot:
    wdot_lhs = (jac_jac_T - tf.matmul(tf.matmul(tilde_w, jac_jac_T), tf.transpose(tilde_w))*Id)
    wdot_rhs = tf.matmul(alpha_dot - tf.matmul(dot_J, jac_T) - tf.matmul(jac, tf.transpose(dot_J)), tf.transpose(tilde_w))
    w_dot = tf.linalg.lstsq(wdot_lhs, wdot_rhs, fast=False)
    w_dot = tf.matmul((Id - tf.einsum('i,j->ij', tilde_w[0], tf.transpose(tilde_w[0]))), w_dot)
    # equation for w:
    x_dot = tf.transpose(tilde_w)
    #
    return tf.transpose(tf.concat([x_dot, w_dot, alpha_dot], axis=0))[0]




#def _naive_KL_ode(t, y, reference, flow, prior_flow):
#    """
#    Solve naively the dynamical equation for KL decomposition in abstract space.
#    """
#    # preprocess:
#    x = tf.convert_to_tensor([tf.cast(y, tf.float32)])
#    # compute metrics:
#    metric = flow.metric(x)[0]
#    prior_metric = prior_flow.metric(x)[0]
#    # compute KL decomposition:
#    eig, eigv = tf_KL_decomposition(metric, prior_metric)
#    # normalize to one to project and select direction:
#    temp = tf.matmul(tf.transpose(eigv), tf.transpose([reference]))[:, 0] / tf.linalg.norm(eigv, axis=0)
#    idx = tf.math.argmax(tf.abs(temp))
#    w = tf.math.sign(temp[idx]) * eigv[:, idx]
#    # normalize affine parameter:
#    s = tf.math.sqrt(tf.tensordot(w, tf.tensordot(metric, w, 1), 1))
#    #
#    return tf.convert_to_tensor([w / s])


def _naive_KL_ode(t, y, reference, flow, prior_flow):
    """
    Solve naively the dynamical equation for KL decomposition in abstract space.
    """
    # preprocess:
    x = flow.cast([y])
    # compute metrics:
    metric = flow.metric(x)[0]
    prior_metric = prior_flow.metric(x)[0]
    # compute KL decomposition:
    eig, eigv = tf_KL_decomposition(metric, prior_metric)
    # normalize to one to project and select direction:
    #temp = tf.linalg.matvec(tf.matmul(tf.transpose(eigv), metric), reference) - tf.tensordot(reference, tf.tensordot(metric, reference, 1), 1)
    #idx = tf.math.argmin(tf.abs(temp))
    #temp_2 = tf.tensordot(eigv[:, idx], reference, 1)
    #w = tf.math.sign(temp_2) * eigv[:, idx]
    #
    temp = tf.matmul(tf.transpose(eigv), tf.transpose([reference]))[:, 0] / tf.linalg.norm(eigv, axis=0) / tf.linalg.norm(reference)
    idx = tf.math.argmax(tf.abs(temp))
    w = tf.math.sign(temp[idx]) * eigv[:, idx]
    # normalize affine parameter:
    s = tf.math.sqrt(tf.tensordot(w, tf.tensordot(metric, w, 1), 1))
    #
    return tf.convert_to_tensor([w / s])


def solve_KL_ode(flow, prior_flow, y0, n, length=1.5, side='both', integrator_options=None, num_points=100, **kwargs):
    """
    Solve eigenvalue problem in abstract space
    side = '+', '-', 'both'
    length = 1.5
    num_points = 100
    n=0
    """
    # define solution points:
    solution_times = tf.linspace(0., length, num_points)
    # compute initial KL decomposition:
    x = flow.cast([y0])
    metric = flow.metric(x)[0]
    prior_metric = prior_flow.metric(x)[0]
    # compute KL decomposition:
    eig, eigv = tf_KL_decomposition(metric, prior_metric)
    # solve forward:
    if side == '+' or side == 'both':
        # initialize solution:
        temp_sol_1 = np.zeros((num_points-1, flow.num_params))
        temp_sol_dot_1 = np.zeros((num_points-1, flow.num_params))
        # initialize forward integration:
        solver = scipy.integrate.ode(_naive_KL_ode)
        if integrator_options is not None:
            solver.set_integrator(**integrator_options)
        solver.set_initial_value(y0, 0.)
        #reference = eigv[:, n] / tf.norm(eigv[:, n])
        reference = eigv[:, n]
        yt = y0.numpy()
        yprime = eigv[:, n]
        # do the time steps:
        for ind, t in enumerate(solution_times[1:]):
            # set the reference:
            solver.set_f_params(reference, flow, prior_flow)
            # advance solver:
            try:
                yt = solver.integrate(t)
                yprime = _naive_KL_ode(t, yt, reference, flow, prior_flow)
            except:
                pass
            # update reference:
            # reference = yprime[0] / tf.norm(yprime[0])
            reference = yprime[0]
            # save out:
            temp_sol_1[ind] = yt.copy()
            temp_sol_dot_1[ind] = yprime.numpy().copy()
        # return if needed:
        if side == '+':
            traj = np.concatenate((x.numpy(), temp_sol_1))
            vel = np.concatenate(([eigv[:, n].numpy()], temp_sol_dot_1))
            return solution_times, traj, vel
    # solve backward:
    if side == '-' or side == 'both':
        # initialize solution:
        temp_sol_2 = np.zeros((num_points-1, flow.num_params))
        temp_sol_dot_2 = np.zeros((num_points-1, flow.num_params))
        # initialize backward integration:
        solver = scipy.integrate.ode(_naive_KL_ode)
        if integrator_options is not None:
            solver.set_integrator(**integrator_options)
        solver.set_initial_value(y0, 0.)
        # reference = - eigv[:, n] / tf.norm(eigv[:, n])
        reference = - eigv[:, n]
        yt = y0.numpy()
        yprime = reference
        for ind, t in enumerate(solution_times[1:]):
            # set the reference:
            solver.set_f_params(reference, flow, prior_flow)
            # advance solver:
            try:
                yt = solver.integrate(t)
                yprime = _naive_KL_ode(t, yt, reference, flow, prior_flow)
            except:
                pass
            # update reference:
            # reference = yprime[0] / tf.norm(yprime[0])
            reference = yprime[0]
            # save out:
            temp_sol_2[ind] = yt.copy()
            temp_sol_dot_2[ind] = yprime.numpy().copy()
        # return if needed:
        if side == '-':
            traj = np.concatenate((temp_sol_2[::-1], x.numpy()))
            vel = np.concatenate((-temp_sol_dot_2[::-1], [eigv[:, n].numpy()]))
            return -solution_times, traj, vel
    # patch solutions:
    times = np.concatenate((-solution_times[::-1], solution_times[1:]))
    traj = np.concatenate((temp_sol_2[::-1], x.numpy(), temp_sol_1))
    vel = np.concatenate((-temp_sol_dot_2[::-1], [eigv[:, n].numpy()], temp_sol_dot_1))
    #
    return times, traj, vel
