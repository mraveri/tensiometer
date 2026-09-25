"""
Nestable automatic differentiation helpers built on ``torch.autograd.grad``.

All functions assume that ``fn`` acts independently on the rows (first axis) of its input,
which holds for every bijector and flow in this package. Every level keeps
``create_graph=True`` by default so that the helpers can be nested to obtain Hessians and
third and fourth order derivatives.
"""

###############################################################################
# initial imports and set-up:

import torch

from . import tensor_utilities as tu

###############################################################################
# helpers:


def prepare_input(coord, device=None):
    """
    Prepare an input for differentiation.

    A tensor that already requires gradients is returned unchanged (nested call, the
    caller wants a graph-attached result); anything else is converted to a new leaf tensor
    with ``requires_grad=True``.

    :param coord: array-like or tensor.
    :param device: target device, None keeps tensors in place and uses the CPU otherwise.
    :returns: tensor requiring gradients.
    """
    if torch.is_tensor(coord) and coord.requires_grad:
        if device is not None and coord.device != torch.device(device):
            coord = coord.to(device)
        if coord.dtype != tu.prec:
            coord = coord.to(tu.prec)
        return coord
    return tu.to_tensor(coord, device=device).detach().requires_grad_(True)


def gradient(fn, x, create_graph=True):
    """
    Gradient of a row-wise scalar function.

    :param fn: callable mapping ``(N, ...)`` to ``(N,)``.
    :param x: input tensor requiring gradients.
    :param create_graph: keep the graph of the result for higher order derivatives.
    :returns: tensor with the shape of ``x``.
    """
    with torch.enable_grad():
        value = fn(x)
        if not value.requires_grad:
            return torch.zeros_like(x)
        (result,) = torch.autograd.grad(value.sum(), x, create_graph=create_graph, allow_unused=True)
    if result is None:
        return torch.zeros_like(x)
    return result


def batch_jacobian(fn, x, create_graph=True):
    """
    Jacobian of a row-wise function, in the layout of ``tf.GradientTape.batch_jacobian``.

    :param fn: callable mapping ``(N,) + in_shape`` to ``(N,) + out_shape``.
    :param x: input tensor requiring gradients, of shape ``(N,) + in_shape``.
    :param create_graph: keep the graph of the result for higher order derivatives.
    :returns: tensor of shape ``(N,) + out_shape + in_shape``.
    """
    with torch.enable_grad():
        value = fn(x)
        out_shape = value.shape[1:]
        flat_value = value.reshape(value.shape[0], -1)
        rows = []
        for i in range(flat_value.shape[1]):
            if not flat_value.requires_grad:
                rows.append(torch.zeros_like(x))
                continue
            (row,) = torch.autograd.grad(
                flat_value[:, i].sum(), x, create_graph=create_graph, retain_graph=True, allow_unused=True)
            if row is None:
                row = torch.zeros_like(x)
            rows.append(row)
    if len(rows) == 0:
        return x.new_zeros((x.shape[0],) + tuple(out_shape) + tuple(x.shape[1:]))
    jacobian = torch.stack(rows, dim=1)
    return jacobian.reshape((x.shape[0],) + tuple(out_shape) + tuple(x.shape[1:]))
