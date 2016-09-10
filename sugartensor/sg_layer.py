# -*- coding: utf-8 -*-
import sugartensor as tf
import sg_initializer as init


__author__ = 'buriburisuri@gmail.com'


#
# neural network layers
#

@tf.sg_layer_func
def sg_dense(tensor, opt):
    # parameter initialize
    w = init.he_uniform('W', (opt.in_dim, opt.dim))
    if opt.bias:
        b = init.constant('b', opt.dim)

    # apply transform
    out = tf.matmul(tensor, w) + (b if opt.bias else 0)

    return out


@tf.sg_layer_func
def sg_conv(tensor, opt):
    # default options
    opt += tf.sg_opt(size=(3, 3), stride=(1, 1, 1, 1), pad='SAME')
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]
    opt.stride = opt.stride if isinstance(opt.stride, (tuple, list)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # parameter initialize
    w = init.he_uniform('W', (opt.size[0], opt.size[1], opt.in_dim, opt.dim))
    if opt.bias:
        b = init.constant('b', opt.dim)

    # apply convolution
    out = tf.nn.conv2d(tensor, w, strides=opt.stride, padding=opt.pad) + (b if opt.bias else 0)

    return out


@tf.sg_layer_func
def sg_aconv(tensor, opt):
    # default options
    opt += tf.sg_opt(size=(3, 3), rate=2, pad='VALID')
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]

    # parameter initialize
    w = init.he_uniform('W', (opt.size[0], opt.size[1], opt.in_dim, opt.dim))
    if opt.bias:
        b = init.constant('b', opt.dim)

    # apply convolution
    out = tf.nn.atrous_conv2d(tensor, w, rate=opt.rate, padding=opt.pad) + (b if opt.bias else 0)

    return out


@tf.sg_layer_func
def sg_upconv(tensor, opt):
    # default options
    opt += tf.sg_opt(size=(3, 3), stride=(1, 2, 2, 1), pad='SAME')
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]
    opt.stride = opt.stride if isinstance(opt.stride, (tuple, list)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # parameter initialize
    w = init.he_uniform('W', (opt.size[0], opt.size[1], opt.dim, opt.in_dim))
    if opt.bias:
        b = init.constant('b', opt.dim)

    # tedious shape handling for conv2d_transpose
    shape = tensor.get_shape().as_list()
    out_shape = [tf.shape(tensor)[0], shape[1] * opt.stride[1], shape[2] * opt.stride[2], opt.dim]

    # apply convolution
    out = tf.nn.conv2d_transpose(tensor, w, output_shape=tf.pack(out_shape),
                                 strides=opt.stride, padding=opt.pad) + (b if opt.bias else 0)
    # reset shape is needed because conv2d_transpose() erase all shape information.
    out.set_shape([None, out_shape[1], out_shape[2], opt.dim])

    return out


@tf.sg_layer_func
def sg_rnn(tensor, opt):
    # parameter initialize
    w = init.orthogonal('W', (opt.in_dim, opt.dim))
    u = init.identity('U', opt.dim)
    if opt.bias:
        b = init.constant('b', opt.dim)

    # initial state
    init_h = opt.init_state if opt.init_state \
        else tf.zeros((tensor.get_shape().as_list()[0], opt.dim), dtype=tf.sg_floatx)

    # permute dimension for scan loop
    xx = tf.transpose(tensor, [1, 0, 2])

    # step func
    def step(h, x):
        # apply transform
        y = tf.matmul(x, w) + tf.matmul(h, u) + (b if opt.bias else 0)
        return _apply_layer_normalization(y, opt)

    # loop by scan
    out = tf.scan(step, xx, init_h)

    # recover dimension
    out = tf.transpose(out, [1, 0, 2])

    # last sequence only
    if opt.last_only:
        out = out[:, tensor.get_shape().as_list()[1]-1, :]

    return out


@tf.sg_layer_func
def sg_bypass(tensor, opt):
    return tensor


def _apply_layer_normalization(tensor, opt):

    # apply layer normalization
    if opt.ln:
        # offset, scale parameter
        beta = init.constant('beta', tensor.get_shape().as_list()[-1])
        gamma = init.constant('gamma', tensor.get_shape().as_list()[-1], value=1)

        # calc layer mean, variance for final axis
        mean, variance = tf.nn.moments(tensor, axes=[len(tensor.get_shape()) - 1])

        # apply layer normalization ( explicit broadcasting needed )
        broadcast_shape = [-1] + [1] * (len(tensor.get_shape()) - 1)
        tensor = (tensor - tf.reshape(mean, broadcast_shape)) \
                 / tf.reshape(tf.sqrt(variance + tf.sg_eps), broadcast_shape)

        # apply parameter
        tensor = gamma * tensor + beta

    return tensor