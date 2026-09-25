"""
This file implements information geometry methods working on normalizing flows.

This file needs work and is not complete.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import scipy

import torch

from . import tensor_utilities as tu

###############################################################################
# CPC decomposition in torch:


def torch_CPC_decomposition(matrix_a, matrix_b):
    """
    Covariant Principal Components decomposition implemented in torch.

    :param matrix_a: input matrix ``A``, shape ``(..., D, D)``.
    :param matrix_b: input matrix ``B``, shape ``(..., D, D)``.
    :returns: eigenvalues and eigenvectors of the transformed matrix.
    """
    matrix_a = torch.as_tensor(matrix_a)
    matrix_b = torch.as_tensor(matrix_b)
    # compute the eigenvalues of b, lambda_b:
    _lambda_b, _phi_b = torch.linalg.eigh(matrix_b)
    _sqrt_lambda_b = torch.diag_embed(1./torch.sqrt(_lambda_b))
    _phib_prime = _phi_b @ _sqrt_lambda_b
    _phib_prime_T = _phib_prime.transpose(-1, -2)
    #
    _a_prime = _phib_prime_T @ matrix_a @ _phib_prime
    _lambda, _phi_a = torch.linalg.eigh(_a_prime)
    _phi = _phi_b @ _sqrt_lambda_b @ _phi_a
    #
    return _lambda, _phi

###############################################################################
# torch KL decomposition helper:


def torch_KL_decomposition(matrix_a, matrix_b):
    """
    Torch implementation of the KL decomposition used within flow-based
    utilities. Mirrors :func:`tensiometer.utilities.stats_utilities.KL_decomposition`.

    :param matrix_a: first matrix ``A``, shape ``(..., D, D)``.
    :param matrix_b: second matrix ``B`` (assumed positive definite), shape ``(..., D, D)``.
    :returns: eigenvalues and eigenvectors of the generalized problem.
    """
    return torch_CPC_decomposition(matrix_a, matrix_b)

###############################################################################
# Simple ODE solver:


def _flat(value):
    """
    Convert an ODE right hand side value (a ``(1, D)`` tensor) to the flat float64 array
    that the scipy integrators expect.

    :param value: right hand side value.
    :returns: numpy array of shape ``(D,)``.
    """
    return np.asarray(tu.to_numpy(value), dtype=np.float64).reshape(-1)


def _eigenvalue_rhs(flow):
    """
    Right hand side of the eigenvalue ODE of a flow for ``scipy.integrate.ode``.

    The wrapper has an explicit signature because the f2py based integrators count the
    positional arguments of the callback to pass the extra parameters.

    :param flow: flow with the ``_naive_eigenvalue_ode_abs`` method.
    :returns: function ``f(t, y, reference)``.
    """
    def _rhs(t, y, reference):
        return _flat(flow._naive_eigenvalue_ode_abs(t, y, reference))
    return _rhs


def _kl_rhs(t, y, reference, flow, prior_flow):
    """Right hand side of the KL ODE for ``scipy.integrate.ode`` (flat float64 array)."""
    return _flat(_naive_KL_ode(t, y, reference, flow, prior_flow))

# need here a simple ODE method

###############################################################################
# KL methods:


def _naive_eigenvalue_ode_abs(flow, t, y, reference):
    """
    Solve naively the dynamical equation for eigenvalues in abstract space.

    :param flow: flow object providing Jacobians.
    :param t: integration time.
    :param y: current position in abstract space.
    :param reference: reference eigenvector used to select the branch.
    :returns: eigenvector direction at the current step.
    """
    # preprocess:
    x = flow.cast([y])
    # map to original space to compute Jacobian (without inversion):
    x_par = flow.map_to_original_coord(x)
    # precompute Jacobian and its derivative:
    jac = flow.inverse_jacobian(x_par)[0]
    jac_T = jac.transpose(-1, -2)
    jac_jac_T = jac @ jac_T
    # compute eigenvalues:
    eig, eigv = torch.linalg.eigh(jac_jac_T)
    reference = torch.as_tensor(tu.to_numpy(reference), dtype=eigv.dtype)
    temp = eigv.transpose(-1, -2) @ reference.reshape(-1, 1)
    idx = int(torch.argmax(torch.abs(temp)))
    w = (torch.sign(temp[idx]) * eigv[:, idx]).unsqueeze(0)
    #
    return w


def solve_eigenvalue_ode_abs(self, y0, n, length=1.5, side='both', integrator_options=None, num_points=100, **kwargs):
    """
    Solve eigenvalue ODE in abstract space.

    Meant to be bound as a method of a flow (``self``) providing ``cast``,
    ``map_to_original_coord``, ``inverse_jacobian``, ``_naive_eigenvalue_ode_abs``
    and ``num_params``.

    :param y0: starting point in abstract coordinates, shape ``(D,)``.
    :param n: index of the eigenvector (of ``J J^T``, in ascending eigenvalue order) to follow.
    :param length: integration length for each direction.
    :param side: which direction to integrate; ``'+'``, ``'-'`` or ``'both'``.
    :param integrator_options: optional options forwarded to ``scipy.integrate.ode.set_integrator``.
    :param num_points: number of output points per direction.
    :param kwargs: ignored; accepted so that callers can pass extra options.
    :returns: solution times, trajectory ``(M, D)`` and velocity ``(M, D)`` along the path,
        with ``M = num_points`` for ``side='+'`` or ``'-'`` (times negated for ``'-'``) and
        ``M = 2 num_points - 1`` for ``'both'``.
    :raises ValueError: if ``side`` is not ``'+'``, ``'-'`` or ``'both'``.
    """
    if side not in ('+', '-', 'both'):
        raise ValueError("side must be '+', '-' or 'both', got " + repr(side))
    y0_np = np.asarray(tu.to_numpy(y0), dtype=np.float64)
    # define solution points:
    solution_times = np.linspace(0., length, num_points)
    # compute initial PCA:
    x_abs = self.cast([y0_np])
    x_par = self.map_to_original_coord(x_abs)
    jac = self.inverse_jacobian(x_par)[0]
    jac_T = jac.transpose(-1, -2)
    jac_jac_T = jac @ jac_T
    # compute eigenvalues at initial point:
    eig, eigv = torch.linalg.eigh(jac_jac_T)
    # solve forward:
    if side == '+' or side == 'both':
        # initialize solution:
        temp_sol_1 = np.zeros((num_points-1, self.num_params))
        temp_sol_dot_1 = np.zeros((num_points-1, self.num_params))
        # initialize forward integration:
        solver = scipy.integrate.ode(_eigenvalue_rhs(self))
        if integrator_options is not None:
            solver.set_integrator(**integrator_options)
        solver.set_initial_value(y0_np, 0.)
        reference = eigv[:, n]
        yt = y0_np.copy()
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
            temp_sol_dot_1[ind] = tu.to_numpy(yprime).copy()
        # return if needed:
        if side == '+':
            traj = np.concatenate((tu.to_numpy(x_abs), temp_sol_1))
            vel = np.concatenate(([tu.to_numpy(eigv[:, n])], temp_sol_dot_1))
            return solution_times, traj, vel
    # solve backward:
    if side == '-' or side == 'both':
        # initialize solution:
        temp_sol_2 = np.zeros((num_points-1, self.num_params))
        temp_sol_dot_2 = np.zeros((num_points-1, self.num_params))
        # initialize backward integration:
        solver = scipy.integrate.ode(_eigenvalue_rhs(self))
        if integrator_options is not None:
            solver.set_integrator(**integrator_options)
        solver.set_initial_value(y0_np, 0.)
        reference = - eigv[:, n]
        yt = y0_np.copy()
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
            temp_sol_dot_2[ind] = tu.to_numpy(yprime).copy()
        # return if needed:
        if side == '-':
            traj = np.concatenate((temp_sol_2[::-1], tu.to_numpy(x_abs)))
            vel = np.concatenate((-temp_sol_dot_2[::-1], [tu.to_numpy(eigv[:, n])]))
            return -solution_times, traj, vel
    # patch solutions:
    times = np.concatenate((-solution_times[::-1], solution_times[1:]))
    traj = np.concatenate((temp_sol_2[::-1], tu.to_numpy(x_abs), temp_sol_1))
    vel = np.concatenate((-temp_sol_dot_2[::-1], [tu.to_numpy(eigv[:, n])], temp_sol_dot_1))
    #
    return times, traj, vel

def solve_eigenvalue_ode_par(self, y0, n, **kwargs):
    """
    Solve the eigenvalue ODE in parameter space.

    :param y0: starting point in parameter coordinates, shape ``(D,)``.
    :param n: index of the eigenvector to follow.
    :param kwargs: options forwarded to :func:`solve_eigenvalue_ode_abs` (``length``,
        ``side``, ``integrator_options``, ``num_points``).
    :returns: times and mapped trajectory in parameter space.
    """
    # go to abstract space:
    x_abs = tu.to_numpy(self.map_to_abstract_coord(self.cast([y0]))[0])
    # call solver:
    times, traj, vel = self.solve_eigenvalue_ode_abs(x_abs, n, **kwargs)
    # convert back:
    traj = self.map_to_original_coord(self.cast(traj))
    #
    return times, traj

# solve full transport in abstract space:
def eigenvalue_ode_abs_temp_3(self, t, y):
    """
    Experimental right hand side of the full eigenvector transport equation in abstract space.

    :param t: integration time.
    :param y: state, position followed by eigenvector and eigenvalue.
    :returns: time derivative of the state.
    """
    # unpack y:
    y = tu.to_numpy(y)
    x = self.cast([y[:self.num_params]])
    w = self.cast([y[self.num_params:-1]])
    alpha = self.cast([y[-1]])
    # map to original space to compute Jacobian (without inversion):
    x_par = self.map_to_original_coord(x)
    # precompute Jacobian and its derivative:
    jac = self.inverse_jacobian(x_par)[0]
    # derivative of the Jacobian is not implemented; use zeros as a placeholder
    djac = torch.zeros((self.num_params, self.num_params, self.num_params), dtype=jac.dtype)
    jacm1 = self.direct_jacobian(x_par)[0]
    jac_T = jac.transpose(-1, -2)
    jac_jac_T = jac @ jac_T
    Id = torch.eye(self.num_params, dtype=jac_jac_T.dtype)
    # select the eigenvector that we want to follow based on the solution to the continuity equation:
    eig, eigv = torch.linalg.eigh(jac_jac_T)
    idx = int(torch.argmax(torch.abs(eigv.transpose(-1, -2) @ w.transpose(-1, -2))))
    tilde_w = eigv[:, idx].unsqueeze(0)
    dot_J = torch.einsum('k, lk, ijl -> ji', tilde_w[0], jacm1, djac)
    # equation for alpha:
    alpha_dot = 2. * ((tilde_w @ jac) @ (dot_J @ tilde_w.transpose(-1, -2)))
    # equation for wdot:
    wdot_lhs = (jac_jac_T - (tilde_w @ jac_jac_T @ tilde_w.transpose(-1, -2)) * Id)
    wdot_rhs = (alpha_dot - dot_J @ jac_T - jac @ dot_J.transpose(-1, -2)) @ tilde_w.transpose(-1, -2)
    w_dot = torch.linalg.lstsq(wdot_lhs, wdot_rhs).solution
    w_dot = (Id - torch.einsum('i,j->ij', tilde_w[0], tilde_w[0])) @ w_dot
    # equation for w:
    x_dot = tilde_w.transpose(-1, -2)
    #
    return torch.cat([x_dot, w_dot, alpha_dot], dim=0).transpose(-1, -2)[0]



def _naive_KL_ode(t, y, reference, flow, prior_flow):
    """
    Right hand side of the KL mode equation in (original) parameter space.

    At ``y`` the metric of ``flow`` is decomposed with respect to the metric of ``prior_flow``,
    both evaluated at the same parameter space point, and the KL eigenvector closest to
    ``reference`` is returned, with the sign of ``reference`` and unit length in the flow metric.

    :param t: integration time (unused, the equation does not depend on it).
    :param y: current position in parameter space, shape ``(D,)``.
    :param reference: direction used to select the eigenvector and its sign, shape ``(D,)``.
    :param flow: flow defining the target metric.
    :param prior_flow: flow defining the prior metric.
    :returns: velocity in parameter space, shape ``(1, D)``.
    """
    # preprocess:
    x = flow.cast([y])
    # compute metrics:
    metric = flow.metric(x)[0]
    prior_metric = prior_flow.metric(x)[0]
    # compute KL decomposition:
    eig, eigv = torch_KL_decomposition(metric, prior_metric)
    # normalize to one to project and select direction:
    reference = torch.as_tensor(tu.to_numpy(reference), dtype=eigv.dtype)
    temp = (eigv.transpose(-1, -2) @ reference.reshape(-1, 1))[:, 0] / torch.linalg.norm(eigv, dim=0) / torch.linalg.norm(reference)
    idx = int(torch.argmax(torch.abs(temp)))
    w = torch.sign(temp[idx]) * eigv[:, idx]
    # normalize affine parameter:
    s = torch.sqrt(w @ (metric @ w))
    #
    return (w / s).unsqueeze(0)


def solve_KL_ode(flow, prior_flow, y0, n, length=1.5, side='both', integrator_options=None, num_points=100, **kwargs):
    """
    Follow a KL eigenmode by solving its equation in (original) parameter space.

    The path moves along a KL eigenvector of the metric of ``flow`` with respect to the metric of
    ``prior_flow``, normalized to unit length in the flow metric. Both metrics are evaluated at the
    same parameter space point, the only coordinate system that the two flows share, so positions
    and velocities are in parameter space (no mapping to abstract coordinates is applied).

    :param flow: flow representing the target distribution.
    :param prior_flow: flow representing the prior distribution.
    :param y0: starting point in parameter space, shape ``(D,)``.
    :param n: index, in the order returned by :func:`torch_KL_decomposition` at ``y0``, of the KL
        eigenmode to follow; along the path the mode is followed by continuity.
    :param length: integration length for each direction.
    :param side: which direction to integrate; ``'+'``, ``'-'`` or ``'both'``.
    :param integrator_options: optional options forwarded to ``scipy.integrate.ode.set_integrator``.
    :param num_points: number of output points per direction.
    :param kwargs: ignored; accepted so that callers can pass extra options.
    :returns: solution times, trajectory and velocity along the path in parameter space, with the
        same layout as :func:`solve_eigenvalue_ode_abs`.
    :raises ValueError: if ``side`` is not ``'+'``, ``'-'`` or ``'both'``.
    """
    if side not in ('+', '-', 'both'):
        raise ValueError("side must be '+', '-' or 'both', got " + repr(side))
    y0_np = np.asarray(tu.to_numpy(y0), dtype=np.float64)
    # define solution points:
    solution_times = np.linspace(0., length, num_points)
    # compute initial KL decomposition:
    x = flow.cast([y0_np])
    metric = flow.metric(x)[0]
    prior_metric = prior_flow.metric(x)[0]
    # compute KL decomposition:
    eig, eigv = torch_KL_decomposition(metric, prior_metric)
    # solve forward:
    if side == '+' or side == 'both':
        # initialize solution:
        temp_sol_1 = np.zeros((num_points-1, flow.num_params))
        temp_sol_dot_1 = np.zeros((num_points-1, flow.num_params))
        # initialize forward integration:
        solver = scipy.integrate.ode(_kl_rhs)
        if integrator_options is not None:
            solver.set_integrator(**integrator_options)
        solver.set_initial_value(y0_np, 0.)
        #reference = eigv[:, n] / torch.linalg.norm(eigv[:, n])
        reference = eigv[:, n]
        yt = y0_np.copy()
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
            # reference = yprime[0] / torch.linalg.norm(yprime[0])
            reference = yprime[0]
            # save out:
            temp_sol_1[ind] = yt.copy()
            temp_sol_dot_1[ind] = tu.to_numpy(yprime).copy()
        # return if needed:
        if side == '+':
            traj = np.concatenate((tu.to_numpy(x), temp_sol_1))
            vel = np.concatenate(([tu.to_numpy(eigv[:, n])], temp_sol_dot_1))
            return solution_times, traj, vel
    # solve backward:
    if side == '-' or side == 'both':
        # initialize solution:
        temp_sol_2 = np.zeros((num_points-1, flow.num_params))
        temp_sol_dot_2 = np.zeros((num_points-1, flow.num_params))
        # initialize backward integration:
        solver = scipy.integrate.ode(_kl_rhs)
        if integrator_options is not None:
            solver.set_integrator(**integrator_options)
        solver.set_initial_value(y0_np, 0.)
        # reference = - eigv[:, n] / torch.linalg.norm(eigv[:, n])
        reference = - eigv[:, n]
        yt = y0_np.copy()
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
            # reference = yprime[0] / torch.linalg.norm(yprime[0])
            reference = yprime[0]
            # save out:
            temp_sol_2[ind] = yt.copy()
            temp_sol_dot_2[ind] = tu.to_numpy(yprime).copy()
        # return if needed:
        if side == '-':
            traj = np.concatenate((temp_sol_2[::-1], tu.to_numpy(x)))
            vel = np.concatenate((-temp_sol_dot_2[::-1], [tu.to_numpy(eigv[:, n])]))
            return -solution_times, traj, vel
    # patch solutions:
    times = np.concatenate((-solution_times[::-1], solution_times[1:]))
    traj = np.concatenate((temp_sol_2[::-1], tu.to_numpy(x), temp_sol_1))
    vel = np.concatenate((-temp_sol_dot_2[::-1], [tu.to_numpy(eigv[:, n])], temp_sol_dot_1))
    #
    return times, traj, vel
