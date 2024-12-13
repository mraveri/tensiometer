"""
This file contains the definition of the trainable bijectors that are needed to define normalizing flows.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import pickle
from collections.abc import Iterable
import functools
import collections

# tensorflow imports:
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model, Sequential

# tensorflow internal imports:
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util

# tensorflow aliases:
tfb = tfp.bijectors
tfd = tfp.distributions

# local imports:
from ..utilities import stats_utilities as stutils
from . import fixed_bijectors as fixed_bijectors

###############################################################################
# utility function to generate random permutations with minimum stack variance:


def min_var_permutations(d, n, min_number=10000):
    """
    Find a random permutation sequence that has the minimum (sample) variance between components.
    This is useful for MAFs, in which we want to concatenate several triangular transformations.

    :param d: dimension of the problem
    :param n: number of dimensions
    :param min_number: number of random trials
    :return: the permutation that has less variance
    """
    permutation = None
    identity = np.arange(d)
    perm_var = np.inf
    for _ in range(max(min_number, 2 * d * n)):
        # draw the permutation ensemble:
        _n_perm = 0
        _temp_perm = []
        while _n_perm < n:
            _temp = np.random.permutation(d)
            if not np.all(_temp == identity):
                _temp_perm.append(_temp)
                _n_perm += 1
        # calculate variance:
        _temp_var = np.var(np.sum(_temp_perm, axis=0))
        # save minimum:
        if _temp_var < perm_var:
            perm_var = _temp_var
            permutation = _temp_perm
    #
    return permutation


###############################################################################
# generic class:


class TrainableTransformation(object):

    def save(self, path):
        """
        Save to file(s) the state of the bijector.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load the trainable bijector.
        """
        raise NotImplementedError


###############################################################################
# class to build a scaling, rotation and shift bijector:

class ScaleRotoShift(tfb.Bijector):

    def __init__(
            self,
            dimension,
            scale=True,
            roto=True,
            shift=True,
            validate_args=False,
            initializer='glorot_uniform',
            name='Affine',
            dtype=tf.float32):
        """
        Bijector performing a shift, scaling and rotation.
        Note that scale is exponential so that we can do unconstrained optimization.
        The bijector is initialized to identity but can be changed with optional argument.
        This bijector does not use the Cholesky decomposition since we need guarantee of
        strictly positive definiteness and invertibility.

        :param dimension: (int) number of dimensions.
        :param scale: (bool) include (or not) scaling.
        :param roto: (bool) include (or not) rotations.
        :param shift: (bool) include (or not) shifts.
        :param validate_args: validate input arguments or not.
        :param initializer: initializer for the bijector, defaults to zeros (identity).
        :param name: name of the bijector.
        :param dtype: data type for the bijector, defaults to tf.float32.      
        :reference: https://arxiv.org/abs/1906.00587
        """

        parameters = dict(locals())

        with tf.name_scope(name) as name:

            self.dimension = dimension
            if shift:
                self._shift = tfp.layers.VariableLayer(
                    dimension, initializer=initializer, dtype=dtype, name=name + '_shift')
            else:
                self._shift = lambda _: tf.zeros(dimension, dtype=dtype, name=name + '_shift')
            if scale:
                self._scalevec = tfp.layers.VariableLayer(
                    dimension, initializer=initializer, dtype=dtype, name=name + '_scale')
            else:
                self._scalevec = lambda _: tf.zeros(dimension, dtype=dtype, name=name + '_scale')
            if roto:
                self._rotvec = tfp.layers.VariableLayer(
                    dimension * (dimension - 1) // 2,
                    initializer=initializer,
                    trainable=True,
                    dtype=dtype,
                    name=name + '_roto')
            else:
                self._rotvec = lambda _: tf.zeros(dimension * (dimension - 1) // 2, dtype=dtype, name=name + '_roto')

            super(ScaleRotoShift, self).__init__(
                forward_min_event_ndims=0,
                is_constant_jacobian=False,
                validate_args=validate_args,
                parameters=parameters,
                dtype=dtype,
                name=name)

    @property
    def shift(self):
        return self._shift

    @classmethod
    def _is_increasing(cls):
        return True

    def _getaff_invaff(self, x):
        L = tf.zeros((self.dimension, self.dimension), dtype=self.dtype)
        L = tf.tensor_scatter_nd_update(L, np.array(np.tril_indices(self.dimension, -1)).T, self._rotvec(x))
        Lambda2 = tf.linalg.diag(tf.math.exp(self._scalevec(x)))
        val_update = tf.ones(self.dimension)
        L = tf.tensor_scatter_nd_update(L, np.array(np.diag_indices(self.dimension)).T, val_update)
        Q, R = tf.linalg.qr(L)
        Phi2 = tf.linalg.matmul(L, tf.linalg.inv(R))
        self.aff = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(Phi2), Lambda2), Phi2)
        self.invaff = tf.linalg.inv(self.aff)
        valdet = tf.linalg.logdet(self.aff)
        self.logdet = valdet / self.dimension

    def _forward(self, x):
        self._getaff_invaff(x)
        _aff = self.aff
        return tf.transpose(tf.linalg.matmul(_aff, tf.transpose(x))) + self._shift(x)[None, :]

    def _inverse(self, y):
        self._getaff_invaff(y)
        _invaff = self.invaff
        return tf.transpose(tf.linalg.matmul(_invaff, tf.transpose(y - self._shift(y)[None, :])))

    def _forward_log_det_jacobian(self, x):
        self._getaff_invaff(x)
        return self.logdet

    def _inverse_log_det_jacobian(self, y):
        return -self._forward_log_det_jacobian(self._inverse(y))

    @classmethod
    def _parameter_properties(cls, dtype):
        return {'shift': parameter_properties.ParameterProperties()}

###############################################################################
# class to build a circular spline:

_SplineShared = collections.namedtuple(
    'SplineShared', 'range_min,range_max,out_of_bounds,out_of_bounds_up,out_of_bounds_dn,x_k,y_k,d_k,d_kp1,h_k,w_k,s_k')

class CircularRationalQuadraticSpline(tfb.RationalQuadraticSpline):
    """
    Rational quadratic spline that has non-zero slope at the boundaries.
    """
    
    def __init__(self,
                 bin_widths,
                 bin_heights,
                 knot_slopes,
                 boundary_knot_slope,
                 range_min=-1,
                 range_max=1,
                 validate_args=False,
                 name=None):
        """
        """

        with tf.name_scope(name or 'RationalQuadraticSpline') as name:

            dtype = dtype_util.common_dtype(
                    [bin_widths, bin_heights, knot_slopes, range_min, boundary_knot_slope],
                    dtype_hint=tf.float32)
            
            self._boundary_knot_slope = tensor_util.convert_nonref_to_tensor(
                boundary_knot_slope, dtype=dtype, name='boundary_knot_slope')
            
            self._range_max = tensor_util.convert_nonref_to_tensor(
                range_max, dtype=dtype, name='range_max')

            super(CircularRationalQuadraticSpline, self).__init__(
                bin_widths=bin_widths,
                bin_heights=bin_heights,
                knot_slopes=knot_slopes,
                range_min=range_min,
                validate_args=validate_args,
                name=name)

    @property
    def range_max(self):
        return self._range_max

    def _compute_shared(self, x=None, y=None):
        """
        See documentation of tfb.RationalQuadraticSpline._compute_shared.
        """

        assert (x is None) != (y is None)
        is_x = x is not None


        range_min = tf.convert_to_tensor(self.range_min, name='range_min')
        range_max = tf.convert_to_tensor(self.range_max, name='range_max')  

        kx = tfb.rational_quadratic_spline._knot_positions(self.bin_widths, range_min)
        ky = tfb.rational_quadratic_spline._knot_positions(self.bin_heights, range_min)
        kd = tf.concat([tf.expand_dims(self._boundary_knot_slope, -1), 
                        self.knot_slopes, 
                        tf.expand_dims(self._boundary_knot_slope, -1)], axis=-1)

        kx_or_ky = kx if is_x else ky
                
        kx_or_ky_min = kx_or_ky[..., 0]
        kx_or_ky_max = kx_or_ky[..., -1]
        
        x_or_y = x if is_x else y
        
        out_of_bounds_up = x_or_y >= kx_or_ky_max
        out_of_bounds_dn = x_or_y <= kx_or_ky_min
        out_of_bounds = out_of_bounds_dn | out_of_bounds_up

        x_or_y = tf.where(out_of_bounds, kx_or_ky_min, x_or_y)

        shape = functools.reduce(
            tf.broadcast_dynamic_shape,
            (
                tf.shape(x_or_y[..., tf.newaxis]),  # Add a n_knots dim.
                tf.shape(kx),
                tf.shape(ky),
                tf.shape(kd)))

        bc_x_or_y = tf.broadcast_to(x_or_y, shape[:-1])
        bc_kx = tf.broadcast_to(kx, shape)
        bc_ky = tf.broadcast_to(ky, shape)
        bc_kd = tf.broadcast_to(kd, shape)
        bc_kx_or_ky = bc_kx if is_x else bc_ky
        indices = tf.maximum(
            tf.zeros([], dtype=tf.int64),
            tf.searchsorted(
                bc_kx_or_ky[..., :-1],
                bc_x_or_y[..., tf.newaxis],
                side='right',
                out_type=tf.int64) - 1)

        def gather_squeeze(params, indices):
            rank = tensorshape_util.rank(indices.shape)
            if rank is None:
                raise ValueError('`indices` must have statically known rank.')
            return tf.gather(params, indices, axis=-1, batch_dims=rank - 1)[..., 0]

        x_k = gather_squeeze(bc_kx, indices)
        x_kp1 = gather_squeeze(bc_kx, indices + 1)
        y_k = gather_squeeze(bc_ky, indices)
        y_kp1 = gather_squeeze(bc_ky, indices + 1)
        d_k = gather_squeeze(bc_kd, indices)
        d_kp1 = gather_squeeze(bc_kd, indices + 1)
        h_k = y_kp1 - y_k
        w_k = x_kp1 - x_k
        s_k = h_k / w_k
                        
        return _SplineShared(
            range_min=range_min,
            range_max=range_max,
            out_of_bounds=out_of_bounds,
            out_of_bounds_up=out_of_bounds_up,
            out_of_bounds_dn=out_of_bounds_dn,
            x_k=x_k,
            y_k=y_k,
            d_k=d_k,
            d_kp1=d_kp1,
            h_k=h_k,
            w_k=w_k,
            s_k=s_k)
        
    def _forward(self, x):
        """Compute the forward transformation (Appendix A.1)."""
        d = self._compute_shared(x=x)
        relx = (x - d.x_k) / d.w_k
        spline_val = (
            d.y_k + ((d.h_k * (d.s_k * relx**2 + d.d_k * relx * (1 - relx))) /
                    (d.s_k + (d.d_kp1 + d.d_k - 2 * d.s_k) * relx * (1 - relx))))
        # apply bounds:
        y_val = tf.where(d.out_of_bounds_up, 
                         self._boundary_knot_slope * (x-d.range_max) +d.range_max, 
                         spline_val)
        y_val = tf.where(d.out_of_bounds_dn, 
                         self._boundary_knot_slope * (x-d.range_min) +d.range_min, 
                         y_val)
        return y_val

    def _inverse(self, y):
        """Compute the inverse transformation (Appendix A.3)."""
        d = self._compute_shared(y=y)
        rely = tf.where(d.out_of_bounds, tf.zeros([], dtype=y.dtype), y - d.y_k)
        term2 = rely * (d.d_kp1 + d.d_k - 2 * d.s_k)
        # These terms are the a, b, c terms of the quadratic formula.
        a = d.h_k * (d.s_k - d.d_k) + term2
        b = d.h_k * d.d_k - term2
        c = -d.s_k * rely
        # The expression used here has better numerical behavior for small 4*a*c.
        relx = tf.where(
            tf.equal(rely, 0), tf.zeros([], dtype=a.dtype),
            (2 * c) / (-b - tf.sqrt(b**2 - 4 * a * c)))

        # apply bounds:
        x_val = tf.where(d.out_of_bounds_up, 
                         self._boundary_knot_slope * (y-d.range_max) +d.range_max, 
                         relx * d.w_k + d.x_k)
        x_val = tf.where(d.out_of_bounds_dn, 
                         self._boundary_knot_slope * (y-d.range_min) +d.range_min, 
                         x_val) 
        return x_val

    def _forward_log_det_jacobian(self, x):
        """Compute the forward derivative (Appendix A.2)."""
        d = self._compute_shared(x=x)
        relx = (x - d.x_k) / d.w_k
        relx = tf.where(d.out_of_bounds, tf.constant(.5, x.dtype), relx)
       
        grad = (
            2 * tf.math.log(d.s_k) +
            tf.math.log(d.d_kp1 * relx**2 + 2 * d.s_k * relx * (1 - relx) +  # newln
                        d.d_k * (1 - relx)**2) -
            2 * tf.math.log((d.d_kp1 + d.d_k - 2 * d.s_k) * relx *
                            (1 - relx) + d.s_k))
        return tf.where(d.out_of_bounds, tf.math.log(self._boundary_knot_slope), grad)

###############################################################################
# helper class to build a spline-autoregressive flow, base spline class:


class SplineHelper(tfb.MaskedAutoregressiveFlow):

    def __init__(
        self,
        shift_and_log_scale_fn=None,
        bijector_fn=None,
        is_constant_jacobian=False,
        validate_args=False,
        unroll_loop=False,
        event_ndims=1,
        name=None,
        spline_knots=8,
        range_max=5.,
        range_min=None,
        equispaced_x_knots=False,
        equispaced_y_knots=False,
        slope_min=0.0001,
        min_bin_width=0.0,
        min_bin_height=0.0,
        slope_std=None,
        softplus_alpha=10.,
        dtype=tf.float32,
        ):
        """
        """
        parameters = dict(locals())
        name = name or 'spline_flow'

        # set ranges:
        if range_min is None:
            assert range_max > 0.
            range_min = -range_max
        interval_width = range_max - range_min
        
        # equispaced knots handling:
        if equispaced_x_knots or equispaced_y_knots:
            delta = (range_max-range_min)/(spline_knots)
        if equispaced_x_knots and equispaced_y_knots:
            raise ValueError('Cannot have both x and y knots equispaced.')

        with tf.name_scope(name) as name:
            self._unroll_loop = unroll_loop
            self._event_ndims = event_ndims
            if bool(shift_and_log_scale_fn) == bool(bijector_fn):
                raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                                 '`bijector_fn` should be specified.')
            if shift_and_log_scale_fn:

                def _bijector_fn(x, **condition_kwargs):

                    def reshape(params):
                        
                        factor = tf.cast(interval_width, dtype=dtype)

                        # get x grid:
                        if not equispaced_x_knots:
                            bin_widths = min_bin_width +params[..., :spline_knots]
                            bin_widths = tf.math.softmax(bin_widths)
                            bin_widths = tf.math.scalar_mul(factor, bin_widths)
                        else:
                            bin_widths = delta * tf.ones_like(params[..., :spline_knots])

                        # get y grid:
                        if not equispaced_y_knots:
                            if equispaced_x_knots:    
                                bin_heights = min_bin_height +params[..., :spline_knots]
                            else:                                              
                                bin_heights = min_bin_height +params[..., spline_knots:spline_knots * 2]
                            bin_heights = tf.math.softmax(bin_heights)
                            bin_heights = tf.math.scalar_mul(factor, bin_heights)
                        else:
                            bin_heights = delta * tf.ones_like(params[..., :spline_knots])
                                
                        # get knot slopes:
                        _start_idx = 2*spline_knots
                        if equispaced_x_knots:
                            _start_idx = _start_idx - spline_knots
                        if equispaced_y_knots:
                            _start_idx = _start_idx - spline_knots
                        knot_slopes = params[..., _start_idx:]
                        
                        # treat different cases:
                        if slope_std is None:
                            # sigmoid:
                            knot_slopes = 2. * tf.math.sigmoid(knot_slopes)
                            # enforce minimum slope:
                            knot_slopes = slope_min + tf.math.scalar_mul(2.-slope_min,knot_slopes)
                        else:                        
                            # deviations around finite differences
                            avg_slope = (bin_heights[...,1:]+bin_heights[...,:-1])/(bin_widths[...,1:]+bin_widths[...,:-1]) # finite diff
                            knot_slopes = avg_slope + tf.math.scalar_mul(slope_std, tf.math.tanh(knot_slopes)) # small deviations around finite diff
                            knot_slopes = tf.math.softplus(knot_slopes*softplus_alpha)/softplus_alpha # ensure slope is positive

                        return bin_widths, bin_heights, knot_slopes

                    params = shift_and_log_scale_fn(x, **condition_kwargs)
                    bin_widths, bin_heights, knot_slopes = reshape(params)

                    return tfb.RationalQuadraticSpline(
                        bin_widths=bin_widths,
                        bin_heights=bin_heights,
                        knot_slopes=knot_slopes,
                        range_min=range_min,
                        validate_args=False)

                bijector_fn = _bijector_fn

            # Still do this assignment for variable tracking.
            self._shift_and_log_scale_fn = shift_and_log_scale_fn
            self._bijector_fn = bijector_fn

            # Call the init method of the Bijector class and not that of MaskedAutoregressiveFlow which we are overriding
            bijector_lib.Bijector.__init__(
                self,
                forward_min_event_ndims=self._event_ndims,
                is_constant_jacobian=is_constant_jacobian,
                validate_args=validate_args,
                parameters=parameters,
                name=name)


class CircularSplineHelper(tfb.MaskedAutoregressiveFlow):

    def __init__(
        self,
        shift_and_log_scale_fn=None,
        bijector_fn=None,
        is_constant_jacobian=False,
        validate_args=False,
        unroll_loop=False,
        event_ndims=1,
        name=None,
        spline_knots=8,
        range_max=5.,
        range_min=None,
        equispaced_x_knots=False,
        equispaced_y_knots=False,
        slope_min=0.0001,
        min_bin_width=0.0,
        min_bin_height=0.0,
        slope_std=None,
        softplus_alpha=10.,
        dtype=tf.float32,
        ):
        """
        """
        parameters = dict(locals())
        name = name or 'circular_spline_flow'

        # set ranges:
        if range_min is None:
            assert range_max > 0.
            range_min = -range_max
        interval_width = range_max - range_min
        
        # equispaced knots handling:
        if equispaced_x_knots or equispaced_y_knots:
            delta = (range_max-range_min)/(spline_knots)
        if equispaced_x_knots and equispaced_y_knots:
            raise ValueError('Cannot have both x and y knots equispaced.')

        with tf.name_scope(name) as name:
            self._unroll_loop = unroll_loop
            self._event_ndims = event_ndims
            if bool(shift_and_log_scale_fn) == bool(bijector_fn):
                raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                                 '`bijector_fn` should be specified.')
            if shift_and_log_scale_fn:

                def _bijector_fn(x, **condition_kwargs):

                    def reshape(params):
                        
                        factor = tf.cast(interval_width, dtype=dtype)

                        # get x grid:
                        if not equispaced_x_knots:
                            bin_widths = min_bin_width +params[..., :spline_knots]
                            bin_widths = tf.math.softmax(bin_widths)
                            bin_widths = tf.math.scalar_mul(factor, bin_widths)
                        else:
                            bin_widths = delta * tf.ones_like(params[..., :spline_knots])

                        # get y grid:
                        if not equispaced_y_knots:
                            if equispaced_x_knots:    
                                bin_heights = min_bin_height +params[..., :spline_knots]
                            else:                                              
                                bin_heights = min_bin_height +params[..., spline_knots:spline_knots * 2]
                            bin_heights = tf.math.softmax(bin_heights)
                            bin_heights = tf.math.scalar_mul(factor, bin_heights)
                        else:
                            bin_heights = delta * tf.ones_like(params[..., :spline_knots])

                        # get knot slopes:
                        _start_idx = 2*spline_knots
                        if equispaced_x_knots:
                            _start_idx -= spline_knots
                        if equispaced_y_knots:
                            _start_idx -= spline_knots
                        knot_slopes = params[..., _start_idx:]

                        # treat different cases:
                        if slope_std is None:
                            # sigmoid:
                            knot_slopes = 2. * tf.math.sigmoid(knot_slopes)
                            # enforce minimum slope:
                            knot_slopes = slope_min + tf.math.scalar_mul(2.-slope_min,knot_slopes)
                            # get boundary slope:
                            boundary_knot_slope = knot_slopes[..., -1]                        
                            knot_slopes = knot_slopes[..., :-1]
                        else:
                            raise NotImplementedError

                        return bin_widths, bin_heights, knot_slopes, boundary_knot_slope

                    params = shift_and_log_scale_fn(x, **condition_kwargs)
                    bin_widths, bin_heights, knot_slopes, boundary_knot_slope = reshape(params)
                                                                                
                    temp_bijector = CircularRationalQuadraticSpline(
                                        bin_widths=bin_widths,
                                        bin_heights=bin_heights,
                                        knot_slopes=knot_slopes,
                                        boundary_knot_slope=boundary_knot_slope,
                                        range_min=range_min,
                                        range_max=range_max,
                                        validate_args=False)

                    return temp_bijector

                bijector_fn = _bijector_fn

            # Still do this assignment for variable tracking.
            self._shift_and_log_scale_fn = shift_and_log_scale_fn
            self._bijector_fn = bijector_fn

            # Call the init method of the Bijector class and not that of MaskedAutoregressiveFlow which we are overriding
            bijector_lib.Bijector.__init__(
                self,
                forward_min_event_ndims=self._event_ndims,
                is_constant_jacobian=is_constant_jacobian,
                validate_args=validate_args,
                parameters=parameters,
                name=name)
            

###############################################################################
# Make separate NNs for each dimension:

def build_nn(dim_in, dim_out, hidden_units, activation=tf.math.asinh, **kwargs):
    if len(hidden_units) == 0 or dim_in == 0:
        model = Sequential(Dense(dim_out, activation=None, input_shape=(dim_in,)))
    else:
        model = Sequential()
        model.add(Dense(hidden_units[0], activation=activation, input_shape=(dim_in,), **kwargs))
        for n in hidden_units[1:]:
            model.add(Dense(n, activation=activation, **kwargs))
        model.add(Dense(dim_out, activation=None, **kwargs))
    return model

@tf.function
def const_zeros(tensor, dim):
    batch_size = tf.shape(tensor)[0]
    constant = tf.zeros(dim)
    constant = tf.expand_dims(constant, axis=0)
    return tf.broadcast_to(constant, shape=(batch_size, dim))

def build_AR_model(num_params, transf_params, hidden_units=[], scale_with_dim=True, identity_dims=None, **kwargs):
    x = Input(num_params)
    params = []
    for dim in range(num_params):
        if identity_dims is not None and dim in identity_dims:
            # params.append(Lambda(lambda _x: tf.broadcast_to(tf.zeros(dim), (tf.shape(_x)[0], dim)))(x))
            params.append(Lambda(lambda _x: const_zeros(_x, transf_params))(x))
        else:
            if dim==0:
                _h = []
            else:
                if scale_with_dim:
                    _h = [int(np.ceil(h * (dim+1) / num_params)) for h in hidden_units]
                else:
                    _h = hidden_units
            nn = build_nn(dim, transf_params, hidden_units=_h, **kwargs)
            params.append(nn(x[..., :dim]))
    params = Lambda(lambda x: tf.stack(x, axis=-2))(params)
    return Model(x, params)


###############################################################################
# helper class to build a spline-autoregressive flow:


class AutoregressiveFlow(TrainableTransformation):
    """
    """

    def __init__(
            self,
            num_params,
            transformation_type='affine',  # 'affine' or 'spline'
            autoregressive_type='masked',  # 'masked' or 'flex'
            n_transformations=None,
            hidden_units=None,
            periodic_params=None,
            activation=tf.math.asinh,
            kernel_initializer=None,
            permutations=True,
            scale_roto_shift=False,
            parameters_min=None,
            parameters_max=None,
            # spline parameters:
            map_to_unitcube=False,
            spline_knots=8,
            range_max=5.,
            equispaced_x_knots=False,
            equispaced_y_knots=False,
            # other parameters:
            autoregressive_scale_with_dim=True,
            autoregressive_identity_dims=None,
            int_np_prec=np.int32,
            np_prec=np.float32,
            feedback=0,
            **kwargs):
        """
        :param num_params: number of parameters of the distribution.
        :param transformation_type: type of transformation, either 'affine' or 'spline'.
        :param autoregressive_type: type of autoregressive network, either 'masked' or 'flex'.
        :param n_transformations: number of transformations to concatenate.
        :param hidden_units: list of hidden units for the autoregressive network.
        :param periodic_params: bool list of parameters that are periodic.
        """

        if n_transformations is None:
            n_transformations = int(np.ceil(2 * np.log2(num_params)) + 2)
        event_shape = (num_params,)

        if hidden_units is None:
            hidden_units = [num_params * 2] * 2

        if isinstance(transformation_type, str):
            _transformation_types = [transformation_type] * n_transformations
            _autoregressive_types = [autoregressive_type] * n_transformations
        else:
            _transformation_types = transformation_type
            _autoregressive_types = autoregressive_type
            assert len(transformation_type) == n_transformations
            assert len(autoregressive_type) == n_transformations

        # initialize permutations:
        if permutations is None:
            _permutations = False
        elif isinstance(permutations, Iterable):
            assert len(permutations) == n_transformations
            _permutations = permutations
        elif isinstance(permutations, bool):
            if permutations:
                _permutations = min_var_permutations(d=num_params, n=n_transformations)
            else:
                _permutations = False
        self.permutations = _permutations

        # check type of architecture:
        if map_to_unitcube:
            assert transformation_type == 'spline'
            
        # check periodic parameters:
        if periodic_params is not None: 
            # if all parameters are not periodic then set to None:
            if np.all(np.logical_not(periodic_params)):
                periodic_params = None
        if periodic_params is not None: 
            assert transformation_type == 'spline'
            
        # check ranges for non-periodic parameters:
        if parameters_min is not None and parameters_max is not None:
            if transformation_type == 'spline':
                # the spline range better enclose all the samples:
                _temp_range_max = max(np.abs(np.amin(parameters_min)), np.abs(np.amax(parameters_max))).astype(type(range_max))
                if range_max < _temp_range_max:
                    if feedback > 0:
                        print('WARNING: range_max should be larger than the maximum range of the data and is beeing adjusted.')
                        print('    range_max:', range_max)
                        print('    max range:', _temp_range_max)
                    range_max = (_temp_range_max + 1).astype(np_prec)
                    if feedback > 0:
                        print('    new range_max:', range_max)
                    range_max = tf.cast(range_max, dtype=np_prec)
        # save ramge max:
        self.range_max = None
        if transformation_type == 'spline':
            self.range_max = range_max
        
        # initialize kernel initializer:    
        if kernel_initializer is None:
            kernel_initializer = tf.keras.initializers.VarianceScaling(
                scale=1. / n_transformations,
                mode='fan_avg',
                distribution='truncated_normal',
                seed=np.random.randint(np.iinfo(int_np_prec).max))

        # Build transformed distribution
        bijectors = []

        # handle first bijectors for periodic parameters:
        if periodic_params is not None:
            # apply shift, modulus and scaling bijector to periodic parameters:
            temp_bijectors = []
            for i in range(num_params):
                if periodic_params[i]:
                    temp_bijector_2 = []
                    # different strategies for splines and mafs:
                    if transformation_type == 'spline':
                        temp_bijector_2.append(tfb.Scale(1./range_max))
                    elif transformation_type == 'affine':
                        temp_bijector_2.append(tfb.Tanh())
                    # add bijector to list:
                    temp_bijectors.append(tfb.Chain(temp_bijector_2))
                else:
                    temp_bijectors.append(tfb.Identity())
            split = tfb.Split(num_params, axis=-1)
            bijectors.append(tfb.Chain([tfb.Invert(split), tfb.JointMap(temp_bijectors), split], name='PeriodicPreprocessing'))
        
        for i in range(n_transformations):

            # add permutations:
            if _permutations:
                if periodic_params is not None:
                    if i > 0:
                        bijectors.append(tfb.Permute(_permutations[i].astype(int_np_prec)))
                else:
                    bijectors.append(tfb.Permute(_permutations[i].astype(int_np_prec)))

            # add map to unit cube
            if map_to_unitcube:
                bijectors.append(tfb.Invert(tfb.NormalCDF()))

            # add main transformation
            _transformation_type = _transformation_types[i]
            _autoregressive_type = _autoregressive_types[i]

            if _transformation_type == 'affine':
                transf_params = 2
            elif _transformation_type == 'spline':
                # number of parameters:
                if periodic_params is not None:
                    transf_params = 3 * spline_knots
                else:
                    transf_params = 3 * spline_knots - 1
                # adjust for equispaced knots:
                if equispaced_x_knots:
                    transf_params -= spline_knots
                if equispaced_y_knots:
                    transf_params -= spline_knots
            else:
                raise ValueError

            ## first, get networks that parametrize transformation
            if _autoregressive_type == 'flex':
                nn = build_AR_model(
                    num_params,
                    transf_params,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    scale_with_dim=autoregressive_scale_with_dim,
                    identity_dims=autoregressive_identity_dims,
                    **stutils.filter_kwargs(kwargs, Dense))
            elif _autoregressive_type == 'masked':
                nn = tfb.AutoregressiveNetwork(
                    params=transf_params,
                    event_shape=event_shape,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    **stutils.filter_kwargs(kwargs, tfb.AutoregressiveNetwork))
            else:
                raise ValueError

            if _transformation_type == 'affine':
                transformation = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=nn)
            elif _transformation_type == 'spline':
                if map_to_unitcube:
                    transformation = SplineHelper(
                        shift_and_log_scale_fn=nn, spline_knots=spline_knots, range_min=0., range_max=1.,
                        **stutils.filter_kwargs(kwargs, SplineHelper))
                else:
                    if periodic_params is not None:
                        transformation = CircularSplineHelper(
                            shift_and_log_scale_fn=nn, spline_knots=spline_knots, range_max=range_max,
                            equispaced_x_knots=equispaced_x_knots, equispaced_y_knots=equispaced_y_knots,
                            **stutils.filter_kwargs(kwargs, CircularSplineHelper))
                    else:
                        transformation = SplineHelper(
                            shift_and_log_scale_fn=nn, spline_knots=spline_knots, range_max=range_max,
                            equispaced_x_knots=equispaced_x_knots, equispaced_y_knots=equispaced_y_knots,
                            **stutils.filter_kwargs(kwargs, SplineHelper))
            bijectors.append(transformation)
            if map_to_unitcube:
                bijectors.append(tfb.NormalCDF())

            # add affine layer:
            if scale_roto_shift:
                bijectors.append(
                    ScaleRotoShift(
                        num_params,
                        name='affine_' + str(i) + '_' + str(np.random.randint(0, 100000)),
                        **stutils.filter_kwargs(kwargs, ScaleRotoShift)))

        self.bijector = tfb.Chain(bijectors)

        if feedback > 1:
            print("    Building Autoregressive Flow")
            print("    - # parameters          :", num_params)
            print("    - periodic parameters   :", periodic_params)
            print("    - # transformations     :", n_transformations)
            print("    - hidden_units          :", hidden_units)
            print("    - transformation_type   :", transformation_type)
            print("    - autoregressive_type   :", autoregressive_type)
            print("    - permutations          :", permutations)
            print("    - scale_roto_shift      :", scale_roto_shift)
            print("    - activation            :", activation)

    def save(self, path):
        """
        Save a `AutoregressiveFlow` object.

        :param path: path of the directory where to save.
        :type path: str
        """
        checkpoint = tf.train.Checkpoint(bijector=self.bijector)
        checkpoint.write(path)
        with open(path + '_permutations.pickle', 'wb') as f:
            pickle.dump(self.permutations, f)
            
    @classmethod
    def load(cls, path, **kwargs):
        """
        Load a saved `AutoregressiveFlow` object. The number of parameters and all other keyword arguments (except for `permutations`) must be included as the MAF is first created with random weights and then these weights are restored.

        :type num_params: int
        :param path: path of the directory from which to load.
        :type path: str
        """
        permutations = pickle.load(open(path + '_permutations.pickle', 'rb'))
        maf = AutoregressiveFlow(
            num_params=len(permutations[0]),
            permutations=permutations,
            **kwargs)
        checkpoint = tf.train.Checkpoint(bijector=maf.bijector)
        checkpoint.read(path).expect_partial()
        return maf


class BijectorLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer that applies a bijector transformation to the inputs.
    """

    def __init__(self, bijector, **kwargs):
        """
        Initializes the BijectorLayer.
        """
        super().__init__(**kwargs)
        self.bijector = bijector

    def call(self, inputs):
        """
        Applies the forward transformation of the bijector to the inputs.
        """
        return self.bijector.forward(inputs)


