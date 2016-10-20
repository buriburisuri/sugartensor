# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

__author__ = 'buriburisuri@gmail.com'


#
# transform sugar functions
#

@tf.sg_sugar_func
def sg_identity(tensor, opt):
    return tf.identity(tensor, name=opt.name)


@tf.sg_sugar_func
def sg_cast(tensor, opt):
    assert opt.dtype is not None, 'dtype is mandatory.'
    return tf.cast(tensor, opt.dtype, name=opt.name)


@tf.sg_sugar_func
def sg_float(tensor, opt):
    return tf.cast(tensor, tf.sg_floatx, name=opt.name)


@tf.sg_sugar_func
def sg_int(tensor, opt):
    return tf.cast(tensor, tf.sg_intx, name=opt.name)


@tf.sg_sugar_func
def sg_expand_dims(tensor, opt):
    opt += tf.sg_opt(dim=-1)
    return tf.expand_dims(tensor, opt.dim, name=opt.name)


@tf.sg_sugar_func
def sg_squeeze(tensor, opt):
    opt += tf.sg_opt(dim=-1)
    return tf.squeeze(tensor, [opt.dim], name=opt.name)


@tf.sg_sugar_func
def sg_flatten(tensor, opt):
    dim = np.prod(tensor.get_shape().as_list()[1:])
    return tf.reshape(tensor, [-1, dim], name=opt.name)


@tf.sg_sugar_func
def sg_reshape(tensor, opt):
    assert opt.shape is not None, 'shape is mandatory.'
    return tf.reshape(tensor, opt.shape, name=opt.name)


@tf.sg_sugar_func
def sg_transpose(tensor, opt):
    assert opt.perm is not None, 'perm is mandatory'
    return tf.transpose(tensor, opt.perm, name=opt.name)


@tf.sg_sugar_func
def sg_argmax(tensor, opt):
    opt += tf.sg_opt(dim=tensor.get_shape().ndims-1)
    return tf.argmax(tensor, opt.dim, opt.name)


@tf.sg_sugar_func
def sg_concat(tensor, opt):
    assert opt.target is not None, 'target is mantory.'
    opt += tf.sg_opt(dim=tensor.get_shape().ndims-1)
    target = opt.target if isinstance(opt.target, (tuple, list)) else [opt.target]
    return tf.concat(opt.dim, [tensor] + target, name=opt.name)


@tf.sg_sugar_func
def sg_one_hot(tensor, opt):
    assert opt.depth is not None, 'depth is mandatory.'
    return tf.one_hot(tensor, opt.depth, name=opt.name)


#
# reduce functions
#

@tf.sg_sugar_func
def sg_sum(tensor, opt):
    return tf.reduce_sum(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_mean(tensor, opt):
    return tf.reduce_mean(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_prod(tensor, opt):
    return tf.reduce_prod(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_min(tensor, opt):
    return tf.reduce_min(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_max(tensor, opt):
    return tf.reduce_max(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_all(tensor, opt):
    return tf.reduce_all(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_any(tensor, opt):
    return tf.reduce_any(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


#
# complicated transform function ( layer related )
#


@tf.sg_sugar_func
def sg_pool(tensor, opt):
    # default stride and pad
    opt += tf.sg_opt(stride=(1, 2, 2, 1), pad='VALID')

    # shape stride
    opt.stride = opt.stride if isinstance(opt.stride, (list, tuple)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # shape size
    opt += tf.sg_opt(size=opt.stride)
    opt.size = opt.size if isinstance(opt.size, (list, tuple)) else [1, opt.size, opt.size, 1]
    opt.size = [1, opt.size[0], opt.size[1], 1] if len(opt.size) == 2 else opt.size

    if opt.avg:
        out = tf.nn.avg_pool(tensor, opt.size, opt.stride, opt.pad)
    else:
        out = tf.nn.max_pool(tensor, opt.size, opt.stride, opt.pad)

    return tf.identity(out, name=opt.name)


@tf.sg_sugar_func
def sg_pool1d(tensor, opt):
    # default stride and pad
    opt += tf.sg_opt(stride=2, pad='VALID')
    opt += tf.sg_opt(size=opt.stride)

    if opt.avg:
        out = tf.nn.avg_pool(tensor.sg_expand_dims(dim=2),
                             (1, opt.size, 1, 1), (1, opt.stride, 1, 1), opt.pad)
    else:
        out = tf.nn.max_pool(tensor.sg_expand_dims(dim=2),
                             (1, opt.size, 1, 1), (1, opt.stride, 1, 1), opt.pad)

    return tf.identity(out.sg_squeeze(dim=2), name=opt.name)

