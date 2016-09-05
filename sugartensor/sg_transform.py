# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

__author__ = 'njkim@jamonglab.com'


#
# transform layers
#

@tf.sg_sugar_func
def sg_cast(self, opt):
    assert opt.dtype is not None, 'dtype is mandatory.'
    return tf.cast(self, opt.dtype)


@tf.sg_sugar_func
def sg_float(self, opt):
    return tf.cast(self, tf.sg_floatx)


@tf.sg_sugar_func
def sg_int(self, opt):
    return tf.cast(self, tf.sg_intx)


@tf.sg_sugar_func
def sg_expand_dims(self, opt):
    opt += tf.sg_opt(dim=-1)
    return tf.expand_dims(self, opt.dim)


@tf.sg_sugar_func
def sg_squeeze(self, opt):
    opt += tf.sg_opt(dim=-1)
    return tf.squeeze(self, [opt.dim])


@tf.sg_sugar_func
def sg_flatten(self, opt):
    dim = np.prod(self.get_shape().as_list()[1:])
    return tf.reshape(self, [-1, dim])


@tf.sg_sugar_func
def sg_reshape(self, opt):
    assert opt.shape is not None, 'shape is mandatory.'
    return tf.reshape(self, opt.shape)


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

    return out
