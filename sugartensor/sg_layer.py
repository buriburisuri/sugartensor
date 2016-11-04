# -*- coding: utf-8 -*-
import sugartensor as tf
import sg_initializer as init


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

def sg_emb(**kwargs):
    opt = tf.sg_opt(kwargs)
    assert opt.name is not None, 'name is mandatory.'

    import sg_initializer as init

    if opt.emb is None:
        # initialize embedding matrix
        assert opt.voca_size is not None, 'voca_size is mandatory.'
        assert opt.dim is not None, 'dim is mandatory.'
        w = init.he_uniform(opt.name, (opt.voca_size-1, opt.dim))
    else:
        # use given embedding matrix
        w = init.external(opt.name, value=opt.emb)

    # 1st row should be zero and not be updated by backprop because of zero padding.
    emb = tf.concat(0, [tf.zeros((1, opt.dim), dtype=tf.sg_floatx), w])

    return emb


@tf.sg_layer_func
def sg_rnn(tensor, opt):

    # step function
    def step(h, x):
        # layer normalization
        ln = lambda v: _ln(v, gamma, beta) if opt.ln else v
        # simple rnn
        y = ln(tf.matmul(tensor[:, i, :], w) + tf.matmul(h, u) + (b if opt.bias else 0))
        return y

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

    # do rnn loop
    h, out = init_h, []
    for i in range(tensor.get_shape().as_list()[1]):
        # apply step func
        h = step(h, tensor[:, i, :])
        # save result
        out.append(h)

    # merge tensor
    out = tf.concat(1, out)

    return out


@tf.sg_layer_func
def sg_gru(tensor, opt):

    # step func
    def step(h, x):
        # layer normalization
        ln = lambda v: _ln(v, gamma, beta) if opt.ln else v
        # update gate
        z = tf.sigmoid(ln(tf.matmul(x, w_z) + tf.matmul(h, u_z) + (b_z if opt.bias else 0)))
        # reset gate
        r = tf.sigmoid(ln(tf.matmul(x, w_r) + tf.matmul(h, u_r) + (b_r if opt.bias else 0)))
        # h_hat
        hh = tf.tanh(ln(tf.matmul(x, w_h) + tf.matmul(r*h, u_h) + (b_h if opt.bias else 0)))
        # final output
        y = (1. - z) * h + z * hh
        return y

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

    # do rnn loop
    h, out = init_h, []
    for i in range(tensor.get_shape().as_list()[1]):
        # apply step function
        h = step(h, tensor[:, i, :])
        # save result
        out.append(h)

    # merge tensor
    out = tf.concat(1, out)

    return out


@tf.sg_layer_func
def sg_lstm(tensor, opt):

    # step func
    def step(h, c, x):
        # layer normalization
        ln = lambda v: _ln(v, gamma, beta) if opt.ln else v
        # input gate
        i = tf.sigmoid(ln(tf.matmul(x, w_i) + tf.matmul(h, u_i) + (b_i if opt.bias else 0)))
        # forget gate
        f = tf.sigmoid(ln(tf.matmul(x, w_f) + tf.matmul(h, u_f) + (b_f if opt.bias else 0)))
        # out gate
        o = tf.sigmoid(ln(tf.matmul(x, w_o) + tf.matmul(h, u_o) + (b_o if opt.bias else 0)))
        # cell gate
        g = tf.tanh(ln(tf.matmul(x, w_g) + tf.matmul(h, u_g) + (b_g if opt.bias else 0)))
        # cell update
        cell = f * c + i * g
        # output
        y = o * tf.tanh(cell)
        return y, cell

    # parameter initialize
    w_i = init.orthogonal('W_i', (opt.in_dim, opt.dim))
    u_i = init.identity('U_i', opt.dim)
    w_f = init.orthogonal('W_f', (opt.in_dim, opt.dim))
    u_f = init.identity('U_f', opt.dim)
    w_o = init.orthogonal('W_o', (opt.in_dim, opt.dim))
    u_o = init.identity('U_o', opt.dim)
    w_g = init.orthogonal('W_g', (opt.in_dim, opt.dim))
    u_g = init.identity('U_g', opt.dim)
    if opt.bias:
        b_i = init.constant('b_z', opt.dim)
        b_f = init.constant('b_r', opt.dim)
        b_o = init.constant('b_h', opt.dim, value=1)
        b_g = init.constant('b_h', opt.dim)

    # layer normalization parameters
    if opt.ln:
        # offset, scale parameter
        beta = init.constant('beta', opt.dim)
        gamma = init.constant('gamma', opt.dim, value=1)

    # initial state
    init_h = opt.init_state if opt.init_state \
        else tf.zeros((tensor.get_shape().as_list()[0], opt.dim), dtype=tf.sg_floatx)

    # do rnn loop
    h, c, out = init_h, init_h, []
    for i in range(tensor.get_shape().as_list()[1]):
        # apply step function
        h, c = step(h, c, tensor[:, i, :])
        # save result
        out.append(h)

    # merge tensor
    out = tf.concat(1, out)

    return out


# layer normalization
def _ln(x, gamma, beta):

    # calc layer mean, variance for final axis
    mean, variance = tf.nn.moments(x, axes=[len(x.get_shape()) - 1], keep_dims=True)

    # apply layer normalization
    x = (x - mean) / tf.sqrt(variance + tf.sg_eps)

    # apply parameter
    return gamma * x + beta
