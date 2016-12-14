from __future__ import absolute_import
import sugartensor as tf
# noinspection PyPackageRequirements
import numpy as np


__author__ = 'buriburisuri@gmail.com'


def constant(name, shape, value=0, dtype=tf.sg_floatx, summary=True):
    r"""Creates a tensor variable of which initial values are `value` and shape is `shape`.

    Args:
      name: The name of new variable.
      shape: A tuple/list of integers or an integer. 
        If shape is an integer, it is converted to a list.
      value: A Python scalar. All elements of the initialized variable
        will be set to this value. Default is 0.
      dtype: The data type. Only floating point types are supported. Default is float32.
      summary: If True, add this constant to tensor board summary.

    Returns:
      A `Variable`.

    """
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.constant_initializer(value))
    # add summary
    if not tf.get_variable_scope().reuse and summary:
        tf.sg_summary_param(x)
    return x


def uniform(name, shape, scale=0.05, dtype=tf.sg_floatx, summary=True):
    r"""Creates a tensor variable of which initial values are 
    random numbers based on uniform distribution.
    
    Note that the default value of `scale` (=0.05) is different from 
    the min/max values (=0.0, 1.0) of tf.random_uniform_initializer.
    
    Args:
      name: The name of the new variable.
      shape: A tuple/list of integers or an integer. 
        If shape is an integer, it's converted to a list.
      scale: A Python scalar. All initial values should be in range `[-scale, scale)`. Default is .05.
      dtype: The data type. Only floating point types are supported. Default is float32.
      summary: If True, add this constant to tensor board summary.
    
    Returns:
      A `Variable`.
    """
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
    # add summary
    if not tf.get_variable_scope().reuse and summary:
        tf.sg_summary_param(x)
    return x


def he_uniform(name, shape, scale=1, dtype=tf.sg_floatx, summary=True):
    r"""See [He et al. 2015](http://arxiv.org/pdf/1502.01852v1.pdf)

    Args:
      name: The name of new variable
      shape: A tuple/list of integers.
      scale: A Python scalar. Scale to initialize. Default is 1.
      dtype: The data type. Default is float32.
      summary: If True, add this constant to tensor board summary.

    Returns:
      A `Variable`.

    """
    fin, _ = _get_fans(shape)
    s = np.sqrt(1. * scale / fin)
    return uniform(name, shape, s, dtype, summary)


def glorot_uniform(name, shape, scale=1, dtype=tf.sg_floatx, summary=True):
    r"""See [Glorot & Bengio. 2010.](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

    Args:
      name: The name of new variable
      shape: A tuple/list of integers.
      scale: A Python scalar. Scale to initialize. Default is 1.
      dtype: The data type. Default is float32.
      summary: If True, add this constant to tensor board summary.

    Returns:
      A `Variable`.

    """
    fin, fout = _get_fans(shape)
    s = np.sqrt(6. * scale / (fin + fout))
    return uniform(name, shape, s, dtype, summary)


def identity(name, dim, scale=1, dtype=tf.sg_floatx, summary=True):
    r"""Creates a tensor variable of which initial values are of
    an identity matrix.
    
    Note that the default value of `scale` (=0.05) is different from 
    the min/max values (=0.0, 1.0) of tf.random_uniform_initializer.
    
    For example,
    
    ```
    identity("identity", 3, 2) =>
    [[2. 0. 0.]
     [0. 2. 0.]
     [0. 0. 2.]]
    ```
    
    Args:
      name: The name of new variable.
      dim: An int. The size of the first and second dimension of the output tensor.
      scale: A Python scalar. The value on the diagonal.
      dtype: The type of the elements of the resulting tensor.
      summary: If True, add this constant to tensor board summary.
    
    Returns:
      A 2-D `Variable`.
    """
    x = tf.get_variable(name,
                        initializer=tf.constant(np.eye(dim) * scale, dtype=dtype))
    # add summary
    if not tf.get_variable_scope().reuse and summary:
        tf.sg_summary_param(x)
    return x


def orthogonal(name, shape, scale=1.1, dtype=tf.sg_floatx, summary=True):
    r"""Creates a tensor variable of which initial values are of
    an orthogonal ndarray.
    
    See [Saxe et al. 2014.](http://arxiv.org/pdf/1312.6120.pdf)
    
    Args:
      name: The name of new variable.
      shape: A tuple/list of integers. 
      scale: A Python scalar.
      dtype: Either float32 or float64.
      summary: If True, add this constant to tensor board summary.
    
    Returns:
      A `Variable`.
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
    if not tf.get_variable_scope().reuse and summary:
        tf.sg_summary_param(x)
    return x


def external(name, value, dtype=tf.sg_floatx, summary=True):
    r"""Creates a tensor variable of which initial values are `value`.
    
    For example,
    
    ```
    external("external", [3,3,1,2])
    => [3. 3. 1. 2.]
    ```
    
    Args:
      name: The name of new variable.
      value: A constant value (or list) of output type `dtype`.
      dtype: The type of the elements of the resulting tensor.
      summary: If True, add this constant to tensor board summary.
    
    Returns:
      A `Variable`. Has the same contents as `value` of `dtype`. 
    """
    # create variable
    x = tf.get_variable(name,
                        initializer=tf.constant(value, dtype=dtype))
    # add summary
    if not tf.get_variable_scope().reuse and summary:
        tf.sg_summary_param(x)
    return x


def _get_fans(shape):
    r"""Returns the size of input dimension and output dimension, given `shape`.
    
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
