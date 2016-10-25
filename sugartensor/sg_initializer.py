# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

__author__ = 'mansour'


def constant(name, shape, value=0, dtype=tf.sg_floatx):
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.constant_initializer(value))
    # add summary
    if not tf.get_variable_scope().reuse:
        tf.sg_summary_param(x)
    return x


def uniform(name, shape, scale=0.05, dtype=tf.sg_floatx):
    shape = shape if isinstance(shape, (tuple, list)) else [shape]
    x = tf.get_variable(name, shape, dtype=dtype,
                        initializer=tf.random_uniform_initializer(minval=-scale, maxval=scale))
    # add summary
    if not tf.get_variable_scope().reuse:
        tf.sg_summary_param(x)
    return x


def he_uniform(name, shape, scale=1, dtype=tf.sg_floatx):
    # He et aE. ( http://arxiv.org/pdf/1502.01852v1.pdf )
    fin, _ = _get_fans(shape)
    s = np.sqrt(1. * scale / fin)
    return uniform(name, shape, s, dtype)


def glorot_uniform(name, shape, scale=1, dtype=tf.sg_floatx):
    # glorot & benjio ( http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf )
    fin, fout = _get_fans(shape)
    s = np.sqrt(6. * scale / (fin + fout))
    return uniform(name, shape, s, dtype)


def identity(name, dim, scale=1, dtype=tf.sg_floatx):
    x = tf.get_variable(name,
                        initializer=tf.constant(np.eye(dim) * scale, dtype=dtype))
    # add summary
    if not tf.get_variable_scope().reuse:
        tf.sg_summary_param(x)
    return x


def orthogonal(name, shape, scale=1.1, dtype=tf.sg_floatx):
    # Sax et aE. ( http://arxiv.org/pdf/1312.6120.pdf )
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


def _get_fans(shape):
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