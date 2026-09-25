"""
Batched quasi-Newton minimization of many independent problems at once.

:func:`batched_minimize` replaces the batched ``tfp.optimizer.bfgs_minimize`` and
``lbfgs_minimize`` calls of the profiler: every row of the position is an independent
problem with its own dense inverse Hessian estimate, line search and convergence flag.
"""

###############################################################################
# initial imports and set-up:

import torch

###############################################################################
# result container:


class MinimizeResult(object):
    """
    Result of :func:`batched_minimize`.

    :param converged: boolean tensor ``(B,)``, gradient sup-norm below tolerance (or no progress possible).
    :param failed: boolean tensor ``(B,)``, line search failure or non-finite values.
    :param num_iterations: number of iterations run.
    :param objective_value: tensor ``(B,)``.
    :param objective_gradient: tensor ``(B, d)``.
    :param position: tensor ``(B, d)``.
    :param inverse_hessian_estimate: tensor ``(B, d, d)``.
    """

    def __init__(self, converged, failed, num_iterations, objective_value, objective_gradient,
                 position, inverse_hessian_estimate):
        self.converged = converged
        self.failed = failed
        self.num_iterations = num_iterations
        self.objective_value = objective_value
        self.objective_gradient = objective_gradient
        self.position = position
        self.inverse_hessian_estimate = inverse_hessian_estimate

###############################################################################
# minimizer:


def _evaluate(value_and_grad_fn, x):
    """Evaluate the objective and its gradient, detached, on the device and dtype of ``x``."""
    value, grad = value_and_grad_fn(x)
    value = torch.as_tensor(value).detach().to(device=x.device, dtype=x.dtype).reshape(x.shape[0])
    grad = torch.as_tensor(grad).detach().to(device=x.device, dtype=x.dtype).reshape(x.shape)
    return value, grad


def _finite_rows(value, grad):
    """Rows with finite objective and gradient."""
    return torch.isfinite(value) & torch.isfinite(grad).all(dim=-1)


def batched_minimize(value_and_grad_fn, initial_position, initial_inverse_hessian=None, tolerance=1e-8,
                     max_iterations=50, max_line_search_iterations=50):
    """
    Minimize ``B`` independent functions with BFGS and backtracking (Armijo) line searches.

    :param value_and_grad_fn: callable mapping positions ``(B, d)`` to a tuple
        ``(values (B,), gradients (B, d))``. It is always called on the full batch.
    :param initial_position: tensor ``(B, d)``; its dtype and device are used throughout.
    :param initial_inverse_hessian: optional ``(d, d)`` or ``(B, d, d)`` initial inverse
        Hessian estimate, identity by default.
    :param tolerance: convergence threshold on the gradient sup-norm.
    :param max_iterations: maximum number of BFGS iterations.
    :param max_line_search_iterations: maximum number of step halvings per iteration.
    :returns: :class:`MinimizeResult`.
    """
    x = torch.as_tensor(initial_position).detach().clone()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    batch, dim = x.shape
    eye = torch.eye(dim, dtype=x.dtype, device=x.device)
    if initial_inverse_hessian is None:
        hessian = eye.expand(batch, dim, dim).clone()
    else:
        hessian = torch.as_tensor(initial_inverse_hessian).detach().to(device=x.device, dtype=x.dtype)
        hessian = torch.broadcast_to(hessian, (batch, dim, dim)).clone()
    value, grad = _evaluate(value_and_grad_fn, x)
    failed = ~_finite_rows(value, grad)
    converged = ~failed & (grad.abs().amax(dim=-1) <= tolerance)
    active = ~converged & ~failed
    iteration = 0
    c1 = 1e-4
    while iteration < max_iterations and bool(active.any()):
        iteration += 1
        with torch.no_grad():
            direction = -(hessian @ grad.unsqueeze(-1)).squeeze(-1)
            slope = (grad * direction).sum(dim=-1)
            # reset non-descent directions to steepest descent:
            reset = active & ~(slope < 0)
            if bool(reset.any()):
                hessian[reset] = eye
                direction[reset] = -grad[reset]
                slope = (grad * direction).sum(dim=-1)
        # backtracking line search:
        step = torch.ones(batch, dtype=x.dtype, device=x.device)
        searching = active.clone()
        accepted = torch.zeros_like(active)
        new_x, new_value, new_grad = x.clone(), value.clone(), grad.clone()
        for _ in range(max_line_search_iterations):
            if not bool(searching.any()):
                break
            trial_x = torch.where(searching[:, None], x + step[:, None] * direction, x)
            trial_value, trial_grad = _evaluate(value_and_grad_fn, trial_x)
            with torch.no_grad():
                ok = searching & _finite_rows(trial_value, trial_grad) \
                    & (trial_value <= value + c1 * step * slope)
                new_x = torch.where(ok[:, None], trial_x, new_x)
                new_value = torch.where(ok, trial_value, new_value)
                new_grad = torch.where(ok[:, None], trial_grad, new_grad)
                accepted = accepted | ok
                searching = searching & ~ok
                step = torch.where(searching, 0.5 * step, step)
        with torch.no_grad():
            failed = failed | (active & ~accepted)
            update = active & accepted
            s = new_x - x
            y = new_grad - grad
            sy = (s * y).sum(dim=-1)
            do_bfgs = update & (sy > 1e-10)
            rho = torch.where(do_bfgs, 1. / torch.where(do_bfgs, sy, torch.ones_like(sy)), torch.zeros_like(sy))
            v = eye - rho[:, None, None] * s[:, :, None] * y[:, None, :]
            new_hessian = v @ hessian @ v.transpose(-1, -2) + rho[:, None, None] * s[:, :, None] * s[:, None, :]
            hessian = torch.where(do_bfgs[:, None, None], new_hessian, hessian)
            no_progress = update & ((new_value >= value) | (s.abs().amax(dim=-1) == 0))
            x = torch.where(update[:, None], new_x, x)
            value = torch.where(update, new_value, value)
            grad = torch.where(update[:, None], new_grad, grad)
            converged = converged | (update & (grad.abs().amax(dim=-1) <= tolerance)) | no_progress
            active = ~converged & ~failed
    return MinimizeResult(converged=converged, failed=failed, num_iterations=iteration,
                          objective_value=value, objective_gradient=grad, position=x,
                          inverse_hessian_estimate=hessian)
