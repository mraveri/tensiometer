###############################################################################
# initial imports and set-up:

import numpy as np
import pickle
from collections.abc import Iterable

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties

from .. import utilities as utils

tfb = tfp.bijectors
tfd = tfp.distributions

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
            print("    - number of MAFs:", n_maf)
            print("    - activation:", activation)
            print("    - hidden_units:", hidden_units)

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


def trainable_ndim_spline_bijector_helper(num_params, nbins=16, range_min=-3., range_max=3., name=None, **kwargs):
    # Build one-dimensional bijectors
    temp_bijectors = [TrainableRationalQuadraticSpline(nbins, range_min=range_min, range_max=range_max, name=f'TQRS{i}', **kwargs) for i in range(num_params)]
    # Need Split() to split/merge inputs
    split = tfb.Split(num_params, axis=-1)
    # Chain all
    return tfb.Chain([tfb.Invert(split), tfb.JointMap(temp_bijectors), split], name=name)

###############################################################################
# class to build a rotation and shift bijector:


class Rotoshift(tfb.Bijector):

    def __init__(self, dimension, validate_args=False, name='rotoshift', dtype=tf.float32):
        parameters = dict(locals())

        with tf.name_scope(name) as name:
            self.dimension = dimension
            self._shift = tfp.layers.VariableLayer(dimension, dtype=dtype)
            self._rotvec = tfp.layers.VariableLayer(dimension*(dimension-1)//2, initializer='random_normal', trainable=True, dtype=dtype)
            self.indexes_low_tri = np.array(np.tril_indices(self.dimension, -1)).T
            self.indexes_diag = np.array(np.diag_indices(self.dimension)).T

            super(Rotoshift, self).__init__(
                forward_min_event_ndims=0,
                is_constant_jacobian=True,
                validate_args=validate_args,
                parameters=parameters,
                dtype=dtype,
                name=name
                )

    @property
    def shift(self):
        return self._shift

    @classmethod
    def _is_increasing(cls):
        return True

    def _getrot_invrot(self, x):
        L = tf.zeros((self.dimension, self.dimension), dtype=self.dtype)
        L = tf.tensor_scatter_nd_update(L, self.indexes_low_tri, self._rotvec(x))
        L = tf.tensor_scatter_nd_update(L, self.indexes_diag, tf.ones(self.dimension))
        Q, R = tf.linalg.qr(L)
        self.rot = tf.linalg.matmul(L, tf.linalg.inv(R))
        self.invrot = tf.transpose(self.rot)

    def _forward(self, x):
        if hasattr(self, 'rot'):
            _rot = self.rot
        else:
            self._getrot_invrot(x)
            _rot = self.rot
        return tf.transpose(tf.linalg.matmul(_rot, tf.transpose(x))) + self._shift(x)[None, :]

    def _inverse(self, y):
        if hasattr(self, 'invrot'):
            _invrot = self.invrot
        else:
            self._getrot_invrot(y)
            _invrot = self.invrot
        return tf.transpose(tf.linalg.matmul(_invrot, tf.transpose(y - self._shift(y)[None, :])))

    def _forward_log_det_jacobian(self, x):
        return tf.zeros([], dtype=dtype_util.base_dtype(x.dtype))

    @classmethod
    def _parameter_properties(cls, dtype):
        return {'shift': parameter_properties.ParameterProperties()}

###############################################################################
# class to build an affine invertible bijector:


class Affine(tfb.Bijector):

    def __init__(self, dimension, validate_args=False, name='Affine', dtype=tf.float32):
        parameters = dict(locals())

        with tf.name_scope(name) as name:

            self.dimension = dimension
            self._shift = tfp.layers.VariableLayer(dimension, dtype=dtype)
            self._rotvec = tfp.layers.VariableLayer(dimension*(dimension+1)//2, initializer='random_normal', trainable=True, dtype=dtype)

            super(Affine, self).__init__(
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
        L = tf.tensor_scatter_nd_update(L, np.array(np.tril_indices(self.dimension, 0)).T, self._rotvec(x))
        Lambda2 = tf.linalg.diag(tf.math.abs(tf.linalg.tensor_diag_part(L)))
        val_update = tf.math.sign(tf.linalg.tensor_diag_part(L)) * tf.ones(self.dimension)
        L = tf.tensor_scatter_nd_update(L, np.array(np.diag_indices(self.dimension)).T, val_update)
        Q, R = tf.linalg.qr(L)
        Phi2 = tf.linalg.matmul(L, tf.linalg.inv(R))
        self.aff = tf.linalg.matmul(tf.linalg.matmul(tf.transpose(Phi2), Lambda2), Phi2)
        self.invaff = tf.linalg.inv(self.aff)
        valdet = tf.linalg.logdet(self.aff)
        self.logdet = valdet

    def _forward(self, x):
        if hasattr(self, 'rot'):
            _aff = self.aff
        else:
            self._getaff_invaff(x)
            _aff = self.aff
        return tf.transpose(tf.linalg.matmul(_aff, tf.transpose(x))) + self._shift(x)[None, :]

    def _inverse(self, y):
        if hasattr(self, 'invrot'):
            _invaff = self.invaff
        else:
            self._getaff_invaff(y)
            _invaff = self.invaff
        return tf.transpose(tf.linalg.matmul(_invaff, tf.transpose(y - self._shift(y)[None, :])))

    def _forward_log_det_jacobian(self, x):
        if not hasattr(self, 'logdet'):
            self._getaff_invaff(x)
        return self.logdet

    @classmethod
    def _parameter_properties(cls, dtype):
        return {'shift': parameter_properties.ParameterProperties()}
