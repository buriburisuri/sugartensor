# -*- coding: utf-8 -*-
import sugartensor as tf
import sg_initializer as init
from functools import wraps


__author__ = 'buriburisuri@gmail.com'


#
# neural network layers
#


@tf.sg_layer_func
def sg_bypass(tensor, opt):
    return tensor


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
def sg_conv1d(tensor, opt):
    # default options
    opt += tf.sg_opt(size=2, stride=1, pad='SAME')

    # parameter initialize
    w = init.he_uniform('W', (opt.size, opt.in_dim, opt.dim))
    if opt.bias:
        b = init.constant('b', opt.dim)

    # apply convolution
    out = tf.nn.conv1d(tensor, w, stride=opt.stride, padding=opt.pad) + (b if opt.bias else 0)

    return out


@tf.sg_layer_func
def sg_conv1d_dilated(tensor, opt):

    import math

    # default options
    opt += tf.sg_opt(size=2, rate=1, pad='SAME')

    # parameter initialize
    w = init.he_uniform('W', (opt.size, opt.in_dim, opt.dim))
    if opt.bias:
        b = init.constant('b', opt.dim)

    # reshaping for dilated convolution
    # time_len, channel = tf.shape(tensor)[1], tf.shape(tensor)[2]
    time_len, channel = tensor.get_shape().as_list()[1:]
    padded_len = int(math.ceil(1.0 * time_len / opt.rate) + 1) * opt.rate
    padded = tf.pad(tensor, [[0, 0], [0, padded_len-time_len], [0, 0]])
    padded = padded.sg_reshape(shape=[-1, opt.rate, channel]).sg_transpose(perm=(1, 0, 2))

    # apply convolution
    conv_out = tf.nn.conv1d(padded, w, stride=1, padding='SAME') + (b if opt.bias else 0)

    # recover to original shape
    out = conv_out.sg_transpose(perm=(1, 0, 2)).sg_reshape(shape=[-1, padded_len, channel])

    # cropping output
    out_len = time_len - opt.rate * opt.size + opt.rate if opt.pad == 'VALID' else time_len
    out = out[:, :out_len:, :]
    # set shape is needed.
    out.set_shape([None, out_len, channel])

    return out


@tf.sg_layer_func
def sg_aconv(tensor, opt):
    # default options
    opt += tf.sg_opt(size=(3, 3), rate=2, pad='SAME')
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


#
# RNN layers
#

@tf.sg_layer_func
def sg_rnn(tensor, opt):

    # parameter initialize
    w = init.orthogonal('W', (opt.in_dim, opt.dim))
    u = init.identity('U', opt.dim)
    if opt.bias:
        b = init.constant('b', opt.dim)

    # layer normalization parameters
    if opt.ln:
        # offset, scale parameter
        beta = init.constant('beta', opt.dim)
        gamma = init.constant('gamma', opt.dim, value=1)

    # initial state
    init_h = opt.init_state if opt.init_state \
        else tf.zeros((tensor.get_shape().as_list()[0], opt.dim), dtype=tf.sg_floatx)

    # permute dimension for scan loop
    xx = tf.transpose(tensor, [1, 0, 2])

    # step func
    def step(h, x):

        # layer normalization
        def ln(xx, opt):
            if opt.ln:
                # calc layer mean, variance for final axis
                mean, variance = tf.nn.moments(xx, axes=[len(xx.get_shape()) - 1])

                # apply layer normalization ( explicit broadcasting needed )
                broadcast_shape = [-1] + [1] * (len(xx.get_shape()) - 1)
                xx = (xx - tf.reshape(mean, broadcast_shape)) \
                         / tf.reshape(tf.sqrt(variance + tf.sg_eps), broadcast_shape)

                # apply parameter
                return gamma * xx + beta

        # apply transform
        y = ln(tf.matmul(x, w) + tf.matmul(h, u) + (b if opt.bias else 0), opt)

        return y

    # loop by scan
    out = tf.scan(step, xx, init_h)

    # recover dimension
    out = tf.transpose(out, [1, 0, 2])

    # last sequence only
    if opt.last_only:
        out = out[:, tensor.get_shape().as_list()[1]-1, :]

    return out


@tf.sg_layer_func
def sg_gru(tensor, opt):

    # parameter initialize
    w_z = init.orthogonal('W_z', (opt.in_dim, opt.dim))
    u_z = init.identity('U_z', opt.dim)
    w_r = init.orthogonal('W_r', (opt.in_dim, opt.dim))
    u_r = init.identity('U_r', opt.dim)
    w_h = init.orthogonal('W_h', (opt.in_dim, opt.dim))
    u_h = init.identity('U_h', opt.dim)
    if opt.bias:
        b_z = init.constant('b_z', opt.dim)
        b_r = init.constant('b_r', opt.dim)
        b_h = init.constant('b_h', opt.dim)

    # layer normalization parameters
    if opt.ln:
        # offset, scale parameter
        beta = init.constant('beta', opt.dim)
        gamma = init.constant('gamma', opt.dim, value=1)

    # initial state
    init_h = opt.init_state if opt.init_state \
        else tf.zeros((tensor.get_shape().as_list()[0], opt.dim), dtype=tf.sg_floatx)

    # permute dimension for scan loop
    xx = tf.transpose(tensor, [1, 0, 2])

    # step func
    def step(h, x):

        # layer normalization
        def ln(xx, opt):
            if opt.ln:
                # calc layer mean, variance for final axis
                mean, variance = tf.nn.moments(xx, axes=[len(xx.get_shape()) - 1])

                # apply layer normalization ( explicit broadcasting needed )
                broadcast_shape = [-1] + [1] * (len(xx.get_shape()) - 1)
                xx = (xx - tf.reshape(mean, broadcast_shape)) \
                         / tf.reshape(tf.sqrt(variance + tf.sg_eps), broadcast_shape)

                # apply parameter
                return gamma * xx + beta

        # update gate
        z = tf.sigmoid(ln(tf.matmul(x, w_z) + tf.matmul(h, u_z) + (b_z if opt.bias else 0), opt))
        # reset gate
        r = tf.sigmoid(ln(tf.matmul(x, w_r) + tf.matmul(h, u_r) + (b_r if opt.bias else 0), opt))
        # h_hat
        hh = tf.sigmoid(ln(tf.matmul(x, w_h) + tf.matmul(r*h, u_h) + (b_h if opt.bias else 0), opt))
        # final output
        y = (1. - z) * h + z * hh

        return y

    # loop by scan
    out = tf.scan(step, xx, init_h)

    # recover dimension
    out = tf.transpose(out, [1, 0, 2])

    # last sequence only
    if opt.last_only:
        out = out[:, tensor.get_shape().as_list()[1]-1, :]

    return out

