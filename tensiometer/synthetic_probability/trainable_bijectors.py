"""
This file contains the definition of the trainable bijectors.
"""

###############################################################################
# initial imports and set-up:

import numpy as np
import pickle
from collections.abc import Iterable

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.internal import parameter_properties
from tensorflow.keras.layers import Input, Lambda, Dense
from tensorflow.keras.models import Model, Sequential

from .synthetic_probability import FlowCallback
from .. import utilities as utils

tfb = tfp.bijectors
tfd = tfp.distributions

###############################################################################
# utility function to generate random permutations with minimum stack variance:


def min_var_permutations(d, n, min_number=10000):
    """
    d = dimension of the problem
    n = number of stacks
    min_number = minimum number of random trials
    """
    permutation = None
    perm_var = np.inf
    for i in range(max(min_number, 2 * d * n)):
        _temp_perm = [np.random.permutation(d) for _ in range(n)]
        _temp_var = np.var(np.sum(_temp_perm, axis=0))
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
            initializer='zeros',
            name='Affine',
            dtype=tf.float32):
        """
        Bijector performing a shift, scaling and rotation.
        Note that scale is exponential so that we can do unconstrained optimization.
        Initialized to identity but can be changed with optional argument.
        This does not use the Cholesky decomposition since we need guarantee of
        strictly positive definiteness and invertibility.

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
        slope_min=0.1,
    ):
        parameters = dict(locals())
        name = name or 'masked_autoregressive_flow'

        if range_min is None:
            assert range_max > 0.
            range_min = -range_max
        interval_width = range_max - range_min

        with tf.name_scope(name) as name:
            self._unroll_loop = unroll_loop
            self._event_ndims = event_ndims
            if bool(shift_and_log_scale_fn) == bool(bijector_fn):
                raise ValueError('Exactly one of `shift_and_log_scale_fn` and '
                                 '`bijector_fn` should be specified.')
            if shift_and_log_scale_fn:

                def _bijector_fn(x, **condition_kwargs):

                    def reshape(params):
                        factor = tf.cast(interval_width, dtype=tf.float32)

                        bin_widths = params[..., :spline_knots]
                        bin_widths = tf.math.softmax(bin_widths)
                        bin_widths = tf.math.scalar_mul(factor, bin_widths)

                        bin_heights = params[..., spline_knots:spline_knots * 2]
                        bin_heights = tf.math.softmax(bin_heights)
                        bin_heights = tf.math.scalar_mul(factor, bin_heights)

                        knot_slopes = params[..., spline_knots * 2:]
                        # knot_slopes = tf.math.softplus(knot_slopes)
                        # knot_slopes =slope_min + tf.math.scalar_mul(2.-slope_min,knot_slopes)
                        knot_slopes = 2. * tf.math.sigmoid(knot_slopes)

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


def build_AR_model(num_params, transf_params, hidden_units=[], scale_with_dim=True, **kwargs):
    x = Input(num_params)
    params = []
    for dim in range(num_params):
        if scale_with_dim:
            _h = [np.ceil(h * dim / num_params) for h in hidden_units]
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
            activation=tf.math.asinh,
            kernel_initializer=None,
            permutations=True,
            scale_roto_shift=False,
            map_to_unitcube=False,
            spline_knots=8,
            range_max=5.,
            autoregressive_scale_with_dim=True,
            int_np_prec=np.int32,
            feedback=0,
            **kwargs):

        if n_transformations is None:
            #n_transformations = 2 * num_params
            n_transformations = int(np.ceil(2 * np.log2(num_params) + 2))
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

        if map_to_unitcube:
            assert transformation_type == 'spline'

        if kernel_initializer is None:
            kernel_initializer = tf.keras.initializers.VarianceScaling(
                scale=1. / n_transformations,
                mode='fan_avg',
                distribution='truncated_normal',
                seed=np.random.randint(np.iinfo(np.int32).max))

        # Build transformed distribution
        bijectors = []
        for i in range(n_transformations):

            # add permutations:
            if _permutations:
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
                transf_params = 3 * spline_knots - 1
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
                    **utils.filter_kwargs(kwargs, Dense))
            elif _autoregressive_type == 'masked':
                nn = tfb.AutoregressiveNetwork(
                    params=transf_params,
                    event_shape=event_shape,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    **utils.filter_kwargs(kwargs, tfb.AutoregressiveNetwork))
            else:
                raise ValueError

            if _transformation_type == 'affine':
                transformation = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=nn)
            elif _transformation_type == 'spline':
                if map_to_unitcube:
                    transformation = SplineHelper(
                        shift_and_log_scale_fn=nn, spline_knots=spline_knots, range_min=0., range_max=1.)
                else:
                    transformation = SplineHelper(
                        shift_and_log_scale_fn=nn, spline_knots=spline_knots, range_max=range_max)
            bijectors.append(transformation)
            if map_to_unitcube:
                bijectors.append(tfb.NormalCDF())

            # add affine layer:
            if scale_roto_shift:
                bijectors.append(
                    ScaleRotoShift(
                        num_params,
                        name='affine_' + str(i) + '_' + str(np.random.randint(0, 100000)),
                        **utils.filter_kwargs(kwargs, ScaleRotoShift)))

            # if _permutations:  # add the inverse permutation
            #     inv_perm = np.zeros_like(_permutations[i])
            #     inv_perm[_permutations[i]] = np.arange(len(inv_perm))
            #     bijectors.append(tfb.Permute(inv_perm.astype(int_np_prec)))

        self.bijector = tfb.Chain(bijectors)

        if feedback > 0:
            print("Building Autoregressive Flow")
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
        pickle.dump(self.permutations, open(path + '_permutations.pickle', 'wb'))

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load a saved `AutoregressiveFlow` object. The number of parameters and all other keyword arguments (except for `permutations`) must be included as the MAF is first created with random weights and then these weights are restored.

        :type num_params: int
        :param path: path of the directory from which to load.
        :type path: str
        :return: a :class:`~.SimpleMAF`.
        """
        permutations = pickle.load(open(path + '_permutations.pickle', 'rb'))
        maf = AutoregressiveFlow(
            num_params=len(permutations[0]),
            permutations=permutations,
            **utils.filter_kwargs(kwargs, AutoregressiveFlow))
        checkpoint = tf.train.Checkpoint(bijector=maf.bijector)
        checkpoint.read(path)
        return maf


class BijectorLayer(tf.keras.layers.Layer):

    def __init__(self, bijector, **kwargs):
        super().__init__(**kwargs)
        self.bijector = bijector

    def call(self, inputs):
        return self.bijector.forward(inputs)


class DerivedParamsBijector(AutoregressiveFlow):

    def __init__(self, chain, param_names_in, param_names_out, permutations=False, **kwargs):
        self.num_params = len(param_names_in)
        assert len(param_names_out) == self.num_params
        self.param_names_in = param_names_in
        self.param_names_out = param_names_out

        super().__init__(self.num_params, permutations=permutations, **kwargs)

        seed = np.random.randint(0, 9999)

        self.flow_in = FlowCallback(
            chain,
            param_names=param_names_in,
            prior_bijector=None,
            trainable_bijector=None,
            rng=np.random.default_rng(seed=seed),
            apply_pregauss='independent',
            feedback=0)

        self.flow_out = FlowCallback(
            chain,
            param_names=param_names_out,
            prior_bijector=None,
            trainable_bijector=None,
            rng=np.random.default_rng(seed=seed),
            apply_pregauss='independent',
            feedback=0)

        self.num_training_samples = len(self.flow_in.training_samples)

        # self.training_dataset = tf.data.Dataset.from_tensor_slices(
        #     (self.flow_in.cast(self.flow_in.training_samples), self.flow_in.cast(self.flow_out.training_samples)))

        # self.training_dataset = self.training_dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()
        # self.training_dataset = self.training_dataset.shuffle(
        #     self.num_training_samples, reshuffle_each_iteration=True).repeat()

        # self.validation_dataset = tf.data.Dataset.from_tensor_slices(
        #     (self.flow_in.cast(self.flow_in.test_samples), self.flow_in.cast(self.flow_out.test_samples)))

        self.trainable_bijector = self.bijector
        self.bijector = tfb.Chain([self.flow_out.bijector, self.trainable_bijector, tfb.Invert(self.flow_in.bijector)])

        x = Input(shape=(self.num_params,))
        y = BijectorLayer(self.trainable_bijector)(x)

        self.model = Model(x, y)

        self.model.compile('adam', 'mse')

    def train(self, epochs=100, batch_size=None, steps_per_epoch=None, callbacks=None, verbose=None, **kwargs):
        # We're trying to loop through the full sample each epoch
        if batch_size is None:
            if steps_per_epoch is None:
                steps_per_epoch = 20
            batch_size = int(self.num_training_samples / steps_per_epoch)
        else:
            if steps_per_epoch is None:
                steps_per_epoch = int(self.num_training_samples / batch_size)
                
        if verbose is None:
            if self.feedback == 0:
                verbose = 0
            elif self.feedback > 0:
                verbose = 1
                
        hist = self.model.fit(
            # x=self.training_dataset.batch(batch_size),
            x=self.flow_in.training_samples,
            y=self.flow_out.training_samples,
            validation_data=(self.flow_in.test_samples, self.flow_out.test_samples),
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            # validation_data=self.validation_dataset,
            verbose=verbose,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()] + callbacks,
            **utils.filter_kwargs(kwargs, self.model.fit))

        return hist