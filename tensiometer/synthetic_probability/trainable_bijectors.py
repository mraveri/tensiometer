###############################################################################
# initial imports and set-up:

import numpy as np
import pickle
from collections.abc import Iterable

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import samplers
from tensorflow.keras.layers import Layer

from .. import utilities as utils

tfb = tfp.bijectors
tfd = tfp.distributions

###############################################################################
# class to build a scaling, rotation and shift bijector:


class ScaleRotoShift(tfb.Bijector):

    def __init__(self, dimension, scale=True, roto=True, shift=True, validate_args=False, name='Affine', dtype=tf.float32):
        parameters = dict(locals())

        with tf.name_scope(name) as name:

            self.dimension = dimension
            if shift:
                self._shift = tfp.layers.VariableLayer(dimension, dtype=dtype, name=name+'_shift')
            else:
                self._shift = lambda _: tf.zeros(dimension, dtype=dtype, name=name+'_shift')
            if scale:
                self._scalevec = tfp.layers.VariableLayer(dimension, initializer='zeros', dtype=dtype, name=name+'_scale')
            else:
                self._scalevec = lambda _: tf.zeros(dimension, dtype=dtype, name=name+'_scale')
            if roto:
                self._rotvec = tfp.layers.VariableLayer(dimension*(dimension-1)//2, initializer='random_normal', trainable=True, dtype=dtype, name=name+'_roto')
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
        L = tf.zeros((self.dimension, self.dimension), dtype=tf.float32)
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


class SimpleMAF(object):
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
                 activation=tf.math.asinh, kernel_initializer='glorot_uniform', int_np_prec=np.int32,
                 feedback=0, **kwargs):

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
            made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, **utils.filter_kwargs(kwargs, tfb.AutoregressiveNetwork))
            shift_and_log_scale_fn = made
            maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=shift_and_log_scale_fn)
            bijectors.append(maf)

            if _permutations:  # add the inverse permutation
                inv_perm = np.zeros_like(_permutations[i])
                inv_perm[_permutations[i]] = np.arange(len(inv_perm))
                bijectors.append(tfb.Permute(inv_perm.astype(int_np_prec)))

        self.bijector = tfb.Chain(bijectors)

        if feedback > 0:
            print("Building MAF")
            print("    - permutations   :", permutations)
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
        return maf
    

from tensorflow_probability.python.bijectors import bijector as bijector_lib

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
        assert range_max>0.
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
                factor=tf.cast(interval_width,dtype=tf.float32)
                
                bin_widths=params[:,:,:spline_knots]
                bin_widths=tf.math.softmax(bin_widths)
                bin_widths=tf.math.scalar_mul(factor,bin_widths)
        
                bin_heights=params[:,:,spline_knots:spline_knots*2]
                bin_heights=tf.math.softmax(bin_heights)
                bin_heights=tf.math.scalar_mul(factor,bin_heights)
        
                knot_slopes=params[:,:,spline_knots*2:]
                # knot_slopes=tf.math.softplus(knot_slopes)
                # knot_slopes=slope_min + tf.math.scalar_mul(2.-slope_min,knot_slopes)
                knot_slopes=2.*tf.math.sigmoid(knot_slopes)

                return bin_widths, bin_heights, knot_slopes
  
            params = shift_and_log_scale_fn(x, **condition_kwargs)
            bin_widths, bin_heights, knot_slopes = reshape(params)
      
            return tfb.RationalQuadraticSpline(bin_widths=bin_widths, bin_heights=bin_heights, knot_slopes=knot_slopes, range_min=range_min, validate_args=False)

        bijector_fn = _bijector_fn
        
      # if validate_args:
      #   bijector_fn = _validate_bijector_fn(bijector_fn)
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
      
class SplineMAF(object):
    
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


from tensorflow.keras.layers import Input, Layer, Lambda, Dense
from tensorflow.keras.models import Model, Sequential

def build_nn(dim_in, dim_out, hidden_units=[], **kwargs):
    if len(hidden_units)==0 or dim_in==0:
        model = Sequential(Dense(dim_out, activation=None, input_shape=(dim_in,)))
    else:
        model = Sequential()
        units = hidden_units + [dim_out]
        model.add(Dense(units[0], input_shape=(dim_in,), **kwargs))
        for n in units[1:]:
            model.add(Dense(n, **kwargs))
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

###############################################################################
# class to build trainable rational splines:


class TrainableRationalQuadraticSpline(tfb.Bijector):

    def __init__(self, nbins, range_min, range_max, validate_args=False, name="TRQS", min_bin_width=None, min_slope=1e-8):
        self._nbins = nbins
        self._range_min = range_min
        self._range_max = range_max
        self._interval_width = self._range_max - self._range_min
        self._built = False
        if min_bin_width is None:
            min_bin_width = self._interval_width / nbins / 100.
        self._min_bin_width = min_bin_width
        self._min_slope = min_slope
        super(TrainableRationalQuadraticSpline, self).__init__(validate_args=validate_args, forward_min_event_ndims=0, name=name)

    def _bin_positions(self, x):
        out_shape = tf.concat((tf.shape(x)[:-1], (self._nbins,)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softmax(x, axis=-1) * (self._interval_width - self._nbins * self._min_bin_width) + self._min_bin_width

    def _slopes(self, x):
        out_shape = tf.concat((tf.shape(x)[:-1], (self._nbins - 1,)), 0)
        x = tf.reshape(x, out_shape)
        return tf.math.softplus(x) + self._min_slope

    def _get_rqs(self, x):
        with tf.name_scope(self.name) as name:
            if not self._built:
                self._bin_widths = tfp.layers.VariableLayer(self._nbins, name=name+'w')
                self._bin_heights = tfp.layers.VariableLayer(self._nbins, name=name+'h')
                self._knot_slopes = tfp.layers.VariableLayer(self._nbins-1, name=name+'s')

            self._built = True

            return tfb.RationalQuadraticSpline(bin_widths=self._bin_positions(self._bin_widths(x)),
                                               bin_heights=self._bin_positions(self._bin_heights(x)),
                                               knot_slopes=self._slopes(self._knot_slopes(x)),
                                               range_min=self._range_min,
                                               name=name
                                               )

    def _inverse_log_det_jacobian(self, y):
        return self._get_rqs(y).inverse_log_det_jacobian(y)

    def forward(self, x):
        return self._get_rqs(x).forward(x)

    def inverse(self, y):
        return self._get_rqs(y).inverse(y)


def trainable_ndim_spline_bijector_helper(num_params, nbins=64, range_min=-4., range_max=4., name=None, **kwargs):
    # Build one-dimensional bijectors
    temp_bijectors = [TrainableRationalQuadraticSpline(nbins, range_min=range_min, range_max=range_max, name=name+f'TQRS{i}', **kwargs) for i in range(num_params)]
    # Need Split() to split/merge inputs
    split = tfb.Split(num_params, axis=-1)
    # Chain all
    return tfb.Chain([tfb.Invert(split), tfb.JointMap(temp_bijectors), split], name=name)

###############################################################################
# helper for spline flow 2:


def build_trainable_RQSpline(nbins, min_bin_width, interval_width, min_slope, range_min, seed, dtype, validate_args):
    """

    """
    bin_position_bijector = tfb.Chain([
        tfb.Shift(min_bin_width),
        tfb.Scale(interval_width - min_bin_width * nbins),
        tfb.SoftmaxCentered()
    ])
    slope_bijector = tfb.Softplus(low=min_slope)

    bin_widths_seed, bin_heights_seed, knot_slopes_seed = samplers.split_seed(seed, n=3)
    unconstrained_bin_widths_initial_values = samplers.normal(
        shape=[nbins-1], mean=0., stddev=.1, seed=bin_widths_seed)
    unconstrained_bin_heights_initial_values = samplers.normal(
        shape=[nbins-1], mean=0., stddev=.1, seed=bin_heights_seed)
    unconstrained_knot_slopes_initial_values = samplers.normal(
        shape=[nbins-1], mean=0., stddev=.01, seed=knot_slopes_seed)
    return tfb.RationalQuadraticSpline(
        bin_widths=tfp.util.TransformedVariable(
            initial_value=bin_position_bijector.forward(
                unconstrained_bin_widths_initial_values),
            bijector=bin_position_bijector,
            dtype=dtype),
        bin_heights=tfp.util.TransformedVariable(
            initial_value=bin_position_bijector.forward(
                unconstrained_bin_heights_initial_values),
            bijector=bin_position_bijector,
            dtype=dtype),
        knot_slopes=tfp.util.TransformedVariable(
            initial_value=slope_bijector.forward(
                unconstrained_knot_slopes_initial_values),
            bijector=slope_bijector,
            dtype=dtype),
        range_min=range_min,
        validate_args=validate_args
    )


class RQSplineFlow(Layer):

    def __init__(self, num_params, nbins, range_min, range_max, min_bin_width=None, min_slope=1e-8, seed=None, validate_args=False, name=None):
        """

        """
        super(RQSplineFlow, self).__init__(name=name)
        self.nbins = nbins
        self.range_min = range_min
        self.range_max = range_max
        self.interval_width = range_max - range_min
        if min_bin_width is None:
            min_bin_width = self.interval_width / nbins / 100.
        self.min_bin_width = min_bin_width
        self.min_slope = min_slope
        self.seed = seed
        self.validate_args = validate_args

        ## build bijector:
        #ndim = num_params
        #seeds = samplers.split_seed(self.seed, ndim)
        #flow_bijectors = []
        #for i in range(ndim):
        #    temp_bij = build_trainable_RQSpline(
        #        self.nbins, self.min_bin_width, self.interval_width, self.min_slope, self.range_min, seeds[i], self.dtype, self.validate_args)
        #    flow_bijectors.append(temp_bij)
        #self.bijectors = flow_bijectors
        #split = tfb.Split(ndim, axis=-1)
        #self.bijector = tfb.Chain([tfb.Invert(split), tfb.JointMap(self.bijectors), split])
        self.bijector = None

    def build(self, input_shape):
        # build bijector:
        ndim = input_shape[-1]
        seeds = samplers.split_seed(self.seed, ndim)
        flow_bijectors = []
        for i in range(ndim):
            temp_bij = build_trainable_RQSpline(
                self.nbins, self.min_bin_width, self.interval_width, self.min_slope, self.range_min, seeds[i], self.dtype, self.validate_args)
            flow_bijectors.append(temp_bij)
        self.bijectors = flow_bijectors

        split = tfb.Split(ndim, axis=-1)
        self.bijector = tfb.Chain([tfb.Invert(split), tfb.JointMap(self.bijectors), split])

###############################################################################
# helper class to build a spline masked-autoregressive affine flow:


class SplineAMAF(object):
    """
    Stack of Splines, MADE and Affine transformations
    """

    def __init__(self, num_params, n_maf=None, hidden_units=None, affine=True,
                 spline=True, activation=tf.math.asinh, kernel_initializer='glorot_uniform',
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
                bijectors.append(ScaleRotoShift(num_params, name='affine1_'+str(i), **utils.filter_kwargs(kwargs, ScaleRotoShift)))
            if spline:
                bijectors.append(RQSplineFlow(num_params, 8, -3., 3., name='spline_'+str(i)).bijector)
                #bijectors.append(trainable_ndim_spline_bijector_helper(num_params, name='spline_'+str(i), **utils.filter_kwargs(kwargs, trainable_ndim_spline_bijector_helper)))
            if affine:
                bijectors.append(ScaleRotoShift(num_params, name='affine2_'+str(i), **utils.filter_kwargs(kwargs, ScaleRotoShift)))

        self.bijector = tfb.Chain(bijectors)

        if feedback > 0:
            print("Building Spline Affine MAF")
            print("    - affine           :", affine)
            print("    - spline           :", spline)
            print("    - number of layers :", n_maf)
            print("    - activation       :", activation)
            print("    - hidden_units     :", hidden_units)
