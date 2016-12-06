from __future__ import absolute_import
import sugartensor as tf
# noinspection PyPackageRequirements
import numpy as np


__author__ = 'buriburisuri@gmail.com'


def constant(name, shape, value=0, dtype=tf.sg_floatx):
    r"""Returns an initializer of `shape` with all elements set to a scalar `value`.

    Args:
        name: name of tensor
        shape: shape to initialize
        value: value to initialize ( default : 0 )
        dtype: data type  ( default : floatx )

    Returns:
      A `Tensor` variable.

    """
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.constant_initializer(value))
    # add summary
    if not tf.get_variable_scope().reuse:
        tf.sg_summary_param(x)
    return x


def uniform(name, shape, scale=0.05, dtype=tf.sg_floatx):
    r"""Returns an initializer of random numbers based on uniform distribution.
    Note that the default value of `scale` (=0.05) is different from 
    the min/max values (=0.0, 1.0) of tf.random_uniform_initializer.

    Args:
        name: name of tensor
        shape: shape to initialize
        scale: scale to initialize ( default : 0.05 )
        dtype: data type  ( default : floatx )

    Returns:
      A `Tensor` variable.

    """
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
    # add summary
    if not tf.get_variable_scope().reuse:
        tf.sg_summary_param(x)
    return x


def he_uniform(name, shape, scale=1, dtype=tf.sg_floatx):
    r"""See He et al. 2015 `http://arxiv.org/pdf/1502.01852v1.pdf`

    Args:
        name: name of tensor
        shape: shape to initialize
        scale: scale to initialize ( default : 1 )
        dtype: data type  ( default : floatx )

    Returns:
      A `Tensor` variable.

    """
    fin, _ = _get_fans(shape)
    s = np.sqrt(1. * scale / fin)
    return uniform(name, shape, s, dtype)


def glorot_uniform(name, shape, scale=1, dtype=tf.sg_floatx):
    r"""See Glorot et al. 2010 `http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf`

    Args:
        name: name of tensor
        shape: shape to initialize
        scale: scale to initialize ( default : 1 )
        dtype: data type  ( default : floatx )

    Returns:
      A `Tensor` variable.

    """
    fin, fout = _get_fans(shape)
    s = np.sqrt(6. * scale / (fin + fout))
    return uniform(name, shape, s, dtype)


def identity(name, dim, scale=1, dtype=tf.sg_floatx):
    r"""Returns an initializer of a 2-D identity tensor.
    
    Args:
      name: A string. The name of the new or existing variable.
      dim: An int. The size of the first and second dimension of the output tensor
      scale: An int (optional). The value on the diagonal. ( default : 1 )
      dtype: A tensor datatype.
    
    Returns:
      A 2-D tensor variable with the value of `scale` on the diagonal and zeros elsewhere.
    """
    x = tf.get_variable(name,
                        initializer=tf.constant(np.eye(dim) * scale, dtype=dtype))
    # add summary
    if not tf.get_variable_scope().reuse:
        tf.sg_summary_param(x)
    return x


def orthogonal(name, shape, scale=1.1, dtype=tf.sg_floatx):
    r"""Returns a random orthogonal initializer.
    See Saxe et al. 2014 `http://arxiv.org/pdf/1312.6120.pdf`
    
    Args:
      name: A string. The name of the new or existing variable.
      shape: A list or tuple of integers.
      scale: A Python scalar.
      dtype: data type of tensor

    Returns:
      A `Tensor` variable.
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    # create variable
    x = tf.get_variable(name,
                        initializer=tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype))
    # add summary
    if not tf.get_variable_scope().reuse:
        tf.sg_summary_param(x)
    return x


def external(name, value, dtype=tf.sg_floatx):
    r"""Returns an initializer of `value`.
    Args:
      name: A string. The name of the new or existing variable.
      value: A constant value (or array) of output type `dtype`.
      dtype: The type of the elements of the resulting tensor. (optional)
    
    Returns:
      A `Tensor` variable.  
    """
    # create variable
    x = tf.get_variable(name,
                        initializer=tf.constant(value, dtype=dtype))
    # add summary
    if not tf.get_variable_scope().reuse:
        tf.sg_summary_param(x)
    return x


def _get_fans(shape):
    """Returns values of input dimension and output dimension, given `shape`.
    
    Args:
      shape: A list of integers.
    
    Returns:
      fan_in: An int. The value of input dimension.
      fan_out: An int. The value of output dimension.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # assuming convolution kernels (2D or 3D).
        kernel_size = np.prod(shape[:2])
        fan_in = shape[-2] * kernel_size
        fan_out = shape[-1] * kernel_size
    else:
        # no specific assumptions
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out
