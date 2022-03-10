###############################################################################
# initial imports and set-up:

import tensorflow as tf
import numpy as np
import scipy

###############################################################################
# KL methods:


@tf.function
def tf_KL_decomposition(matrix_a, matrix_b):
    """
    KL decomposition in tensorflow
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
    return _lambda, _phi


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
