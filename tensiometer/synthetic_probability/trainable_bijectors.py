###############################################################################
# initial imports and set-up:

import numpy as np
import pickle
from collections.abc import Iterable

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.internal import parameter_properties

from .. import utilities as utils

tfb = tfp.bijectors
tfd = tfp.distributions

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

    def __init__(self, dimension, scale=True, roto=True, shift=True, validate_args=False, initializer='zeros', name='Affine', dtype=tf.float32):
        """
        Bijector performing a shift, scaling and rotation.
        Note that scale is exponential so that we can do unbounded optimization.
        Initialized to identity but can be changed with optional argument.
        """

        parameters = dict(locals())

        with tf.name_scope(name) as name:

            self.dimension = dimension
            if shift:
                self._shift = tfp.layers.VariableLayer(dimension, initializer=initializer, dtype=dtype, name=name+'_shift')
            else:
                self._shift = lambda _: tf.zeros(dimension, dtype=dtype, name=name+'_shift')
            if scale:
                self._scalevec = tfp.layers.VariableLayer(dimension, initializer=initializer, dtype=dtype, name=name+'_scale')
            else:
                self._scalevec = lambda _: tf.zeros(dimension, dtype=dtype, name=name+'_scale')
            if roto:
                self._rotvec = tfp.layers.VariableLayer(dimension*(dimension-1)//2, initializer=initializer, trainable=True, dtype=dtype, name=name+'_roto')
            else:
                self._rotvec = lambda _: tf.zeros(dimension*(dimension-1)//2, dtype=dtype, name=name+'_roto')

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
# helper class to build a masked-autoregressive flow:


class SimpleMAF(TrainableTransformation):
    """
    A class to implement a simple Masked AutoRegressive Flow (MAF) using the implementation :class:`tfp.bijectors.AutoregressiveNetwork` from from `Tensorflow Probability <https://www.tensorflow.org/probability/>`_. Additionally, this class provides utilities to load/save models, including random permutations.

    :param num_params: number of parameters, ie the dimension of the space of which the bijector is defined.
    :type num_params: int
    :param n_maf: number of MAFs to stack. Defaults to None, in which case it is set to `2*num_params`.
    :type n_maf: int, optional
    :param hidden_units: a list of the number of nodes per hidden layers. Defaults to None, in which case it is set to `[num_params*2]*2`.
    :type hidden_units: list, optional
    :param permutations: whether to use shuffle dimensions between stacked MAFs, defaults to True.
    :type permutations: bool, optional
    :param activation: activation function to use in all layers, defaults to :func:`tf.math.asinh`.
    :type activation: optional
    :param kernel_initializer: kernel initializer, defaults to 'glorot_uniform'.
    :type kernel_initializer: str, optional
    :param feedback: print the model architecture, defaults to 0.
    :type feedback: int, optional
    :reference: George Papamakarios, Theo Pavlakou, Iain Murray (2017). Masked Autoregressive Flow for Density Estimation. `arXiv:1705.07057 <https://arxiv.org/abs/1705.07057>`_
    """

    def __init__(self, num_params, n_maf=None, hidden_units=None, permutations=True,
                 activation=tf.math.asinh, kernel_initializer=None, int_np_prec=np.int32,
                 feedback=0, **kwargs):

        # initialize hidden units:
        if n_maf is None:
            n_maf = 2*num_params
        event_shape = (num_params,)

        if hidden_units is None:
            hidden_units = [num_params*2]*2

        # initialize permutations:
        _permutations = False
        if isinstance(permutations, Iterable):
            assert len(permutations) == n_maf
            _permutations = permutations
        elif isinstance(permutations, bool):
            if permutations:
                _permutations = [np.random.permutation(num_params) for _ in range(n_maf)]
            else:
                _permutations = False
        self.permutations = _permutations

        # Build transformed distribution
        bijectors = []
        for i in range(n_maf):
            # add permutations:
            if _permutations:
                bijectors.append(tfb.Permute(_permutations[i].astype(int_np_prec)))
            # add MAF layer:
            if kernel_initializer is None:
                kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1./n_maf, mode='fan_avg', distribution='truncated_normal')
            made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape, hidden_units=hidden_units, activation=activation,
                                             kernel_initializer=kernel_initializer, **utils.filter_kwargs(kwargs, tfb.AutoregressiveNetwork))
            maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made)
            bijectors.append(maf)
            # add the inverse permutation:
            if _permutations:
                inv_perm = np.zeros_like(_permutations[i])
                inv_perm[_permutations[i]] = np.arange(len(inv_perm))
                bijectors.append(tfb.Permute(inv_perm.astype(int_np_prec)))

        self.bijector = tfb.Chain(bijectors)

        if feedback > 0:
            print("Building MAF")
            print("    - permutations   :", permutations is not None)
            print("    - number of MAFs :", n_maf)
            print("    - activation     :", activation)
            print("    - hidden_units   :", hidden_units)

    def save(self, path):
        """
        Save a `SimpleMAF` object.

        :param path: path of the directory where to save.
        :type path: str
        """
        checkpoint = tf.train.Checkpoint(bijector=self.bijector)
        checkpoint.write(path)
        pickle.dump(self.permutations, open(path+'_permutations.pickle', 'wb'))

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load a saved `SimpleMAF` object. The number of parameters and all other keyword arguments (except for `permutations`) must be included as the MAF is first created with random weights and then these weights are restored.

        :type num_params: int
        :param path: path of the directory from which to load.
        :type path: str
        :return: a :class:`~.SimpleMAF`.
        """
        permutations = pickle.load(open(path+'_permutations.pickle', 'rb'))
        maf = SimpleMAF(num_params=len(permutations[0]), permutations=permutations, **utils.filter_kwargs(kwargs, SimpleMAF))
        checkpoint = tf.train.Checkpoint(bijector=maf.bijector)
        checkpoint.read(path)
        #
        return maf

###############################################################################
# helper class to build a spline-autoregressive flow, base spline class:


class MaskedAutoregressiveFlowSpline(tfb.MaskedAutoregressiveFlow):

    def __init__(self,
                 shift_and_log_scale_fn=None,
                 bijector_fn=None,
                 is_constant_jacobian=False,
                 validate_args=False,
                 unroll_loop=False,
                 event_ndims=1,
                 name=None,
                 spline_knots=2,
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

                        bin_widths = params[:, :, :spline_knots]
                        bin_widths = tf.math.softmax(bin_widths)
                        bin_widths = tf.math.scalar_mul(factor, bin_widths)

                        bin_heights = params[:, :, spline_knots:spline_knots*2]
                        bin_heights = tf.math.softmax(bin_heights)
                        bin_heights = tf.math.scalar_mul(factor, bin_heights)

                        knot_slopes = params[:, :, spline_knots*2:]
                        # knot_slopes=tf.math.softplus(knot_slopes)
                        # knot_slopes=slope_min + tf.math.scalar_mul(2.-slope_min,knot_slopes)
                        knot_slopes = 2.*tf.math.sigmoid(knot_slopes)

                        return bin_widths, bin_heights, knot_slopes

                    params = shift_and_log_scale_fn(x, **condition_kwargs)
                    bin_widths, bin_heights, knot_slopes = reshape(params)

                    return tfb.RationalQuadraticSpline(bin_widths=bin_widths, bin_heights=bin_heights, knot_slopes=knot_slopes, range_min=range_min, validate_args=False)

                bijector_fn = _bijector_fn

            # Still do this assignment for variable tracking.
            self._shift_and_log_scale_fn = shift_and_log_scale_fn
            self._bijector_fn = bijector_fn

            # Call the init method of the Bijector class and not that of MaskedAutoregressiveFlow which we are overriding
            bijector_lib.Bijector.__init__(self,
                                           forward_min_event_ndims=self._event_ndims,
                                           is_constant_jacobian=is_constant_jacobian,
                                           validate_args=validate_args,
                                           parameters=parameters,
                                           name=name)

###############################################################################
# helper class to build a spline-autoregressive flow:


class SplineMAF(object):
    """
    """

    def __init__(self, num_params, spline_knots, range_max=5., n_maf=None, hidden_units=None, permutations=True,
                 activation=tf.math.asinh, kernel_initializer='glorot_uniform', int_np_prec=np.int32,
                 feedback=0, map_to_unitsq=False, **kwargs):

        if n_maf is None:
            n_maf = 2*num_params
        event_shape = (num_params,)

        if hidden_units is None:
            hidden_units = [num_params*2]*2

        if permutations is None:
            _permutations = False
        elif isinstance(permutations, Iterable):
            assert len(permutations) == n_maf
            _permutations = permutations
        elif isinstance(permutations, bool):
            if permutations:
                _permutations = [np.random.permutation(num_params) for _ in range(n_maf)]
            else:
                _permutations = False

        self.permutations = _permutations

        # Build transformed distribution
        bijectors = []
        for i in range(n_maf):
            if _permutations:
                bijectors.append(tfb.Permute(_permutations[i].astype(int_np_prec)))
            if map_to_unitsq:
                bijectors.append(tfb.Invert(tfb.NormalCDF()))
            made = tfb.AutoregressiveNetwork(params=3*spline_knots - 1, event_shape=event_shape, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, **utils.filter_kwargs(kwargs, tfb.AutoregressiveNetwork))
            shift_and_log_scale_fn = made
            if map_to_unitsq:
                maf = MaskedAutoregressiveFlowSpline(shift_and_log_scale_fn=shift_and_log_scale_fn, spline_knots=spline_knots, range_min=0., range_max=1.)
            else:
                maf = MaskedAutoregressiveFlowSpline(shift_and_log_scale_fn=shift_and_log_scale_fn, spline_knots=spline_knots, range_max=range_max)
            bijectors.append(maf)
            if map_to_unitsq:
                bijectors.append(tfb.NormalCDF())
            if _permutations:  # add the inverse permutation
                inv_perm = np.zeros_like(_permutations[i])
                inv_perm[_permutations[i]] = np.arange(len(inv_perm))
                bijectors.append(tfb.Permute(inv_perm.astype(int_np_prec)))

        self.bijector = tfb.Chain(bijectors)

        if feedback > 0:
            print("Building MAF")
            print("    - permutations   :", permutations)
            print("    - number of Spline MAFs :", n_maf)
            print("    - activation     :", activation)
            print("    - hidden_units   :", hidden_units)

    def save(self, path):
        """
        Save a `SplineMAF` object.

        :param path: path of the directory where to save.
        :type path: str
        """
        checkpoint = tf.train.Checkpoint(bijector=self.bijector)
        checkpoint.write(path)
        pickle.dump(self.permutations, open(path+'_permutations.pickle', 'wb'))

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load a saved `SplineMAF` object. The number of parameters and all other keyword arguments (except for `permutations`) must be included as the MAF is first created with random weights and then these weights are restored.

        :type num_params: int
        :param path: path of the directory from which to load.
        :type path: str
        :return: a :class:`~.SimpleMAF`.
        """
        permutations = pickle.load(open(path+'_permutations.pickle', 'rb'))
        maf = SplineMAF(num_params=len(permutations[0]), permutations=permutations, **utils.filter_kwargs(kwargs, SplineMAF))
        checkpoint = tf.train.Checkpoint(bijector=maf.bijector)
        checkpoint.read(path)
        return maf


from tensorflow.keras.layers import Input, Layer, Lambda, Dense
from tensorflow.keras.models import Model, Sequential

def build_nn(dim_in, dim_out, hidden_units=[], activation='softplus', **kwargs):
    if len(hidden_units)==0 or dim_in==0:
        model = Sequential(Dense(dim_out, activation=None, input_shape=(dim_in,)))
    else:
        model = Sequential()
        model.add(Dense(hidden_units[0], activation=activation, input_shape=(dim_in,), **kwargs))
        for n in hidden_units[1:]:
            model.add(Dense(n, activation=activation, **kwargs))
        model.add(Dense(dim_out, activation=None, **kwargs))
    return model

def build_AR_model(num_params, spline_knots, hidden_units=[], **kwargs):
    x = Input(num_params)
    params = []
    for dim in range(num_params):
        nn = build_nn(dim, 3*spline_knots-1, hidden_units=hidden_units, **kwargs)
        params.append(nn(x[...,:dim]))
    params = Lambda(lambda x: tf.stack(x, axis=-2))(params)
    return Model(x, params)
    

    
class SplineAR(object):
    
    """
    A class to implement a simple Masked AutoRegressive Flow (MAF) using the implementation :class:`tfp.bijectors.AutoregressiveNetwork` from from `Tensorflow Probability <https://www.tensorflow.org/probability/>`_. Additionally, this class provides utilities to load/save models, including random permutations.

    :param num_params: number of parameters, ie the dimension of the space of which the bijector is defined.
    :type num_params: int
    :param n_maf: number of MAFs to stack. Defaults to None, in which case it is set to `2*num_params`.
    :type n_maf: int, optional
    :param hidden_units: a list of the number of nodes per hidden layers. Defaults to None, in which case it is set to `[num_params*2]*2`.
    :type hidden_units: list, optional
    :param permutations: whether to use shuffle dimensions between stacked MAFs, defaults to True.
    :type permutations: bool, optional
    :param activation: activation function to use in all layers, defaults to :func:`tf.math.asinh`.
    :type activation: optional
    :param kernel_initializer: kernel initializer, defaults to 'glorot_uniform'.
    :type kernel_initializer: str, optional
    :param feedback: print the model architecture, defaults to 0.
    :type feedback: int, optional
    :reference: George Papamakarios, Theo Pavlakou, Iain Murray (2017). Masked Autoregressive Flow for Density Estimation. `arXiv:1705.07057 <https://arxiv.org/abs/1705.07057>`_
    """

    def __init__(self, num_params, spline_knots, range_max=5., n_maf=None, hidden_units=None, permutations=True, activation='softplus', kernel_initializer='glorot_uniform', int_np_prec=np.int32,
                 feedback=0, map_to_unitsq=False, **kwargs):

        if n_maf is None:
            n_maf = 2*num_params
        event_shape = (num_params,)

        if hidden_units is None:
            hidden_units = [num_params*2]*2

        if permutations is None:
            _permutations = False
        elif isinstance(permutations, Iterable):
            assert len(permutations) == n_maf
            _permutations = permutations
        elif isinstance(permutations, bool):
            if permutations:
                _permutations = [np.random.permutation(num_params) for _ in range(n_maf)]
            else:
                _permutations = False

        self.permutations = _permutations

        # Build transformed distribution
        bijectors = []
        for i in range(n_maf):
            if _permutations:
                bijectors.append(tfb.Permute(_permutations[i].astype(int_np_prec)))
            if map_to_unitsq:
                bijectors.append(tfb.Invert(tfb.NormalCDF()))
            # made = tfb.AutoregressiveNetwork(params=3*spline_knots - 1, event_shape=event_shape, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, **utils.filter_kwargs(kwargs, tfb.AutoregressiveNetwork))
            shift_and_log_scale_fn = build_AR_model(num_params, spline_knots, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, **utils.filter_kwargs(kwargs, Dense))
            if map_to_unitsq:
                maf = MaskedAutoregressiveFlowSpline(shift_and_log_scale_fn=shift_and_log_scale_fn, spline_knots=spline_knots, range_min=0., range_max=1.)
            else:
                maf = MaskedAutoregressiveFlowSpline(shift_and_log_scale_fn=shift_and_log_scale_fn, spline_knots=spline_knots, range_max=range_max)
            bijectors.append(maf)
            if map_to_unitsq:
                bijectors.append(tfb.NormalCDF())
            if _permutations:  # add the inverse permutation
                inv_perm = np.zeros_like(_permutations[i])
                inv_perm[_permutations[i]] = np.arange(len(inv_perm))
                bijectors.append(tfb.Permute(inv_perm.astype(int_np_prec)))
                
        self.bijector = tfb.Chain(bijectors)
        
        if feedback > 0:
            print("Building MAF")
            print("    - permutations   :", permutations)
            print("    - number of MAFs :", n_maf)
            print("    - hidden_units   :", hidden_units)
    
    def save(self, path):
        """
        Save a `SplineMAF` object.

        :param path: path of the directory where to save.
        :type path: str
        """
        checkpoint = tf.train.Checkpoint(bijector=self.bijector)
        checkpoint.write(path)
        pickle.dump(self.permutations, open(path+'_permutations.pickle', 'wb'))

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load a saved `SplineMAF` object. The number of parameters and all other keyword arguments (except for `permutations`) must be included as the MAF is first created with random weights and then these weights are restored.

        :type num_params: int
        :param path: path of the directory from which to load.
        :type path: str
        :return: a :class:`~.SimpleMAF`.
        """
        permutations = pickle.load(open(path+'_permutations.pickle', 'rb'))
        maf = SplineMAF(num_params=len(permutations[0]), permutations=permutations, **utils.filter_kwargs(kwargs, SplineMAF))
        checkpoint = tf.train.Checkpoint(bijector=maf.bijector)
        checkpoint.read(path)
        return maf

###############################################################################
# helper class to build a masked-autoregressive affine flow:


class AMAF(object):
    """
    Stack of MADE and Affine transformations
    """

    def __init__(self, num_params, n_maf=None, hidden_units=None, affine=True,
                 activation=tf.math.asinh, kernel_initializer='glorot_uniform',
                 feedback=0, **kwargs):

        if n_maf is None:
            n_maf = 2*num_params
        event_shape = (num_params,)

        if hidden_units is None:
            hidden_units = [num_params*2]*2

        # Build transformed distribution
        bijectors = []
        for i, _ in enumerate(range(n_maf)):
            made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, **utils.filter_kwargs(kwargs, tfb.AutoregressiveNetwork))
            shift_and_log_scale_fn = made
            maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=shift_and_log_scale_fn)
            bijectors.append(maf)
            if affine:
                bijectors.append(ScaleRotoShift(num_params, name='affine_'+str(i), **utils.filter_kwargs(kwargs, ScaleRotoShift)))

        self.bijector = tfb.Chain(bijectors)

        if feedback > 0:
            print("Building Affine MAF")
            print("    - affine         :", affine)
            print("    - number of MAFs :", n_maf)
            print("    - activation     :", activation)
            print("    - hidden_units   :", hidden_units)
