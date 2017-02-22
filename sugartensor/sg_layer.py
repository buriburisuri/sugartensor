from __future__ import absolute_import
import sugartensor as tf


__author__ = 'buriburisuri@gmail.com'


#
# neural network layers
#


# noinspection PyUnusedLocal
@tf.sg_layer_func
def sg_bypass(tensor, opt):
    r"""Returns the input tensor itself.
    
    Args:
      tensor: A `Tensor` (automatically passed by decorator).
      opt:
        bn: Boolean. If True, batch normalization is applied.
        ln: Boolean. If True, layer normalization is applied.
        dout: A float of range [0, 100). A dropout rate. Default is 0.
        act: A name of activation function. e.g., `sigmoid`, `tanh`, etc.

    Returns:
      The same tensor as `tensor`.
    """
    return tensor


@tf.sg_layer_func
def sg_dense(tensor, opt):
    r"""Applies a full connection.
    
    Args:
      tensor: A 2-D tensor (automatically passed by decorator).
      opt:
        in_dim: An `integer`. The size of input dimension.
        dim: An `integer`. The size of output dimension.
        bias: Boolean. If True, biases are added.
      
    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # parameter initialize
    w = tf.sg_initializer.he_uniform('W', (opt.in_dim, opt.dim))
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # apply transform
    out = tf.matmul(tensor, w) + b

    return out


@tf.sg_layer_func
def sg_conv(tensor, opt):
    r"""Applies a 2-D convolution.
    
    Args:
      tensor: A 4-D `Tensor` (automatically passed by decorator).
      opt:
        size: A tuple/list of positive integers of length 2 representing `[kernel height, kernel width]`.
          Can be an integer if both values are the same.
          If not specified, (3, 3) is set implicitly.
        stride: A tuple/list of positive integers of length 2 or 4 representing stride dimensions.
          If the length is 2, i.e., (a, b), the stride is `[1, a, b, 1]`.
          If the length is 4, i.e., (a, b, c, d), the stride is `[a, b, c, d]`.
          Can be an integer. If the length is a, the stride is `[1, a, a, 1]`.
          Default value is [1, 1, 1, 1].
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        pad: Either `SAME` (Default) or `VALID`. 
        bias: Boolean. If True, biases are added.

    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # default options
    opt += tf.sg_opt(size=(3, 3), stride=(1, 1, 1, 1), pad='SAME')
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]
    opt.stride = opt.stride if isinstance(opt.stride, (tuple, list)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # parameter initialize
    w = tf.sg_initializer.he_uniform('W', (opt.size[0], opt.size[1], opt.in_dim, opt.dim))
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # apply convolution
    out = tf.nn.conv2d(tensor, w, strides=opt.stride, padding=opt.pad) + b

    return out


@tf.sg_layer_func
def sg_conv1d(tensor, opt):
    r"""Applies a 1-D convolution.
    
    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        size: A positive `integer` representing `[kernel width]`.
          If not specified, 2 is set implicitly.
        stride: A positive `integer`. The number of entries by which
          the filter is moved right at each step.
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        pad: Either `SAME` (Default) or `VALID`.
        bias: Boolean. If True, biases are added.
      
    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # default options
    opt += tf.sg_opt(size=2, stride=1, pad='SAME')

    # parameter tf.sg_initializer
    w = tf.sg_initializer.he_uniform('W', (opt.size, opt.in_dim, opt.dim))
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # apply convolution
    out = tf.nn.conv1d(tensor, w, stride=opt.stride, padding=opt.pad) + b

    return out


@tf.sg_layer_func
def sg_aconv(tensor, opt):
    r"""Applies a 2-D atrous (or dilated) convolution.
    
    Args:
      tensor: A 4-D `Tensor` (automatically passed by decorator).
      opt:
        size: A tuple/list of positive integers of length 2 representing `[kernel height, kernel width]`.
          Can be an integer if both values are the same.
          If not specified, (3, 3) is set automatically.
        rate: A positive integer. The stride with which we sample input values across
          the `height` and `width` dimensions. Default is 2.
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        pad: Either `SAME` (Default) or `VALID`.
        bias: Boolean. If True, biases are added.
            
    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # default options
    opt += tf.sg_opt(size=(3, 3), rate=2, pad='SAME')
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]

    # parameter tf.sg_initializer
    w = tf.sg_initializer.he_uniform('W', (opt.size[0], opt.size[1], opt.in_dim, opt.dim))
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # apply convolution
    out = tf.nn.atrous_conv2d(tensor, w, rate=opt.rate, padding=opt.pad) + b

    return out


@tf.sg_layer_func
def sg_aconv1d(tensor, opt):
    r"""Applies 1-D atrous (or dilated) convolution.
    
    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        causal: Boolean. If True, zeros are padded before the time axis such that
          each activation unit doesn't have receptive neurons beyond the equivalent time step.
        size: A positive `integer` representing `[kernel width]`. As a default it is set to 2
          if causal is True, 3 otherwise. 
        rate: A positive `integer`. The stride with which we sample input values across
          the `height` and `width` dimensions. Default is 1.
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        pad: Either `SAME` (Default) or `VALID`.
        bias: Boolean. If True, biases are added.
            
    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # default options
    opt += tf.sg_opt(size=(2 if opt.causal else 3), rate=1, pad='SAME')

    # parameter tf.sg_initializer
    w = tf.sg_initializer.he_uniform('W', (1, opt.size, opt.in_dim, opt.dim))
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    if opt.causal:
        # pre-padding for causality
        if opt.pad == 'SAME':
            pad_len = (opt.size - 1) * opt.rate  # padding size
            x = tf.pad(tensor, [[0, 0], [pad_len, 0], [0, 0]]).sg_expand_dims(axis=1)
        else:
            x = tensor.sg_expand_dims(axis=1)
        # apply 2d convolution
        out = tf.nn.atrous_conv2d(x, w, rate=opt.rate, padding='VALID') + b
    else:
        # apply 2d convolution
        out = tf.nn.atrous_conv2d(tensor.sg_expand_dims(axis=1),
                                  w, rate=opt.rate, padding=opt.pad) + b
    # reduce dimension
    # noinspection PyUnresolvedReferences
    out = out.sg_squeeze(axis=1)

    return out


@tf.sg_layer_func
def sg_upconv(tensor, opt):
    r"""Applies a up convolution (or convolution transpose).
    
    Args:
      tensor: A 4-D `Tensor` (automatically passed by decorator).
      opt:
        size: A tuple/list of integers of length 2 representing `[kernel height, kernel width]`.
          Can be an integer if both values are the same.
          If not specified, (4, 4) is set implicitly.
          Default value is [1, 2, 2, 1].
        stride: A tuple/list of integers of length 2 or 4 representing stride dimensions.
          If the length is 2, i.e., (a, b), the stride is `[1, a, b, 1]`.
          If the length is 4, i.e., (a, b, c, d), the stride is `[a, b, c, d]`.
          Can be an integer. If the length is a, the stride is `[1, a, a, 1]`.
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        pad: Either `SAME` (Default) or `VALID`. 
        bias: Boolean. If True, biases are added.
            
    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # default options
    opt += tf.sg_opt(size=(4, 4), stride=(1, 2, 2, 1), pad='SAME')
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]
    opt.stride = opt.stride if isinstance(opt.stride, (tuple, list)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # parameter tf.sg_initializer
    w = tf.sg_initializer.he_uniform('W', (opt.size[0], opt.size[1], opt.dim, opt.in_dim))
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # tedious shape handling for conv2d_transpose
    shape = tensor.get_shape().as_list()
    out_shape = [tf.shape(tensor)[0], shape[1] * opt.stride[1], shape[2] * opt.stride[2], opt.dim]

    # apply convolution
    out = tf.nn.conv2d_transpose(tensor, w, output_shape=tf.stack(out_shape),
                                 strides=opt.stride, padding=opt.pad) + b
    # reset shape is needed because conv2d_transpose() erase all shape information.
    # noinspection PyUnresolvedReferences
    out.set_shape([None, out_shape[1], out_shape[2], opt.dim])

    return out


@tf.sg_layer_func
def sg_upconv1d(tensor, opt):
    r"""Applies 1-D a up convolution (or convolution transpose).

    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        size:  A positive `integer` representing `[kernel width]`. As a default it is set to 4
        stride: A positive `integer` representing stride dimension. As a default it is set to 2
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        pad: Either `SAME` (Default) or `VALID`.
        bias: Boolean. If True, biases are added.

    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # default options
    opt += tf.sg_opt(size=4, stride=2, pad='SAME')
    opt.size = [opt.size, 1]
    opt.stride = [1, opt.stride, 1, 1]

    # parameter tf.sg_initializer
    w = tf.sg_initializer.he_uniform('W', (opt.size[0], opt.size[1], opt.dim, opt.in_dim))
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # make 4-D tensor
    tensor = tensor.sg_expand_dims(axis=2)

    # tedious shape handling for conv2d_transpose
    shape = tensor.get_shape().as_list()
    out_shape = [tf.shape(tensor)[0], shape[1] * opt.stride[1], shape[2] * opt.stride[2], opt.dim]

    # apply convolution
    out = tf.nn.conv2d_transpose(tensor, w, output_shape=tf.stack(out_shape),
                                 strides=opt.stride, padding=opt.pad) + b
    # reset shape is needed because conv2d_transpose() erase all shape information.
    # noinspection PyUnresolvedReferences
    out.set_shape([None, out_shape[1], out_shape[2], opt.dim])

    # squeeze
    out = out.sq_squeeze(dim=2)

    return out


@tf.sg_layer_func
def sg_espcn(tensor, opt):
    r"""Applies a 2-D efficient sub pixel convolution.
       (see [Shi et al. 2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)

    Args:
      tensor: A 4-D `Tensor` (automatically passed by decorator).
      opt:
        size: A tuple/list of positive integers of length 2 representing `[kernel height, kernel width]`.
          Can be an integer if both values are the same.
          If not specified, (3, 3) is set implicitly.
        stride: A tuple/list of positive integers of length 2 or 4 representing stride dimensions.
          If the length is 2, i.e., (a, b), the stride is `[1, a, b, 1]`.
          If the length is 4, i.e., (a, b, c, d), the stride is `[a, b, c, d]`.
          Can be an integer. If the length is a, the stride is `[1, a, a, 1]`.
          Default value is [1, 1, 1, 1].
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        pad: Either `SAME` (Default) or `VALID`.
        bias: Boolean. If True, biases are added.
        factor: factor to multiply shape by. Default is 2.

    Returns:
      A `Tensor` with the same type as `tensor`.
    """
    # default options
    opt += tf.sg_opt(size=(3, 3), stride=(1, 1, 1, 1), pad='SAME', factor=2)
    opt.size = opt.size if isinstance(opt.size, (tuple, list)) else [opt.size, opt.size]
    opt.stride = opt.stride if isinstance(opt.stride, (tuple, list)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # parameter initialize
    w = tf.sg_initializer.he_uniform('W', (opt.size[0], opt.size[1], opt.in_dim, opt.dim * opt.factor * opt.factor))
    b = tf.sg_initializer.constant('b', opt.dim) if opt.bias else 0

    # apply convolution
    out = tf.nn.conv2d(tensor, w, strides=opt.stride, padding=opt.pad) + b

    # apply periodic shuffle
    out = out.sg_periodic_shuffle(factor=opt.factor)

    return out


#
# RNN layers
#

def sg_emb(**kwargs):
    r"""Returns a look-up table for embedding.
    
    kwargs:
      name: A name for the layer.
      emb: A 2-D array (optional). 
        If None, the resulting tensor should have the shape of 
        `[vocabulary size, embedding dimension size]`.
        Note that its first row is filled with 0's associated with padding.
      in_dim: A positive `integer`. The size of input dimension.
      dim: A positive `integer`. The size of output dimension.
      voca_size: A positive integer. The size of vocabulary.
      
    Returns:
      A 2-D `Tensor` of float32.
    """
    opt = tf.sg_opt(kwargs)
    assert opt.name is not None, 'name is mandatory.'

    if opt.emb is None:
        # initialize embedding matrix
        assert opt.voca_size is not None, 'voca_size is mandatory.'
        assert opt.dim is not None, 'dim is mandatory.'
        w = tf.sg_initializer.he_uniform(opt.name, (opt.voca_size - 1, opt.dim))
    else:
        # use given embedding matrix
        w = tf.sg_initializer.external(opt.name, value=opt.emb)

    # 1st row should be zero and not be updated by backprop because of zero padding.
    emb = tf.concat([tf.zeros((1, opt.dim), dtype=tf.sg_floatx), w], 0)

    return emb


# layer normalization for rnn
def _ln_rnn(x, gamma, beta):
    r"""Applies layer normalization.
    Normalizes the last dimension of the tensor `x`.
    
    Args:
      x: A `Tensor`.
      gamma: A constant `Tensor`. Scale parameter. Default is 1.
      beta: A constant `Tensor`. Offset parameter. Default is 0.
    
    Returns:
      A `Tensor` with the same shape as `x`.
    """
    # calc layer mean, variance for final axis
    mean, variance = tf.nn.moments(x, axes=[len(x.get_shape()) - 1], keep_dims=True)

    # apply layer normalization
    x = (x - mean) / tf.sqrt(variance + tf.sg_eps)

    # apply parameter
    return gamma * x + beta


@tf.sg_rnn_layer_func
def sg_rnn(tensor, opt):
    r"""Applies a simple rnn.
    
    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        bias: Boolean. If True, biases are added.
        ln: Boolean. If True, layer normalization is applied.   
        init_state: A 2-D `Tensor`. If None, the initial state is set to zeros.
        last_only: Boolean. If True, the outputs in the last time step are returned.
    
    Returns:
      A `Tensor`. If last_only is True, the output tensor has shape [batch size, dim].
      Otherwise, [batch size, time steps, dim].
    """
    # layer normalization
    # noinspection PyPep8
    ln = lambda v: _ln_rnn(v, gamma, beta) if opt.ln else v

    # step function
    def step(hh, x):
        # simple rnn
        y = ln(tf.matmul(x, w) + tf.matmul(hh, u) + (b if opt.bias else 0))
        return y

    # parameter initialize
    w = tf.sg_initializer.orthogonal('W', (opt.in_dim, opt.dim))
    u = tf.sg_initializer.identity('U', opt.dim)
    if opt.bias:
        b = tf.sg_initializer.constant('b', opt.dim)

    # layer normalization parameters
    if opt.ln:
        # offset, scale parameter
        beta = tf.sg_initializer.constant('beta', opt.dim)
        gamma = tf.sg_initializer.constant('gamma', opt.dim, value=1)

    # initial state
    init_h = opt.init_state if opt.init_state is not None \
        else tf.zeros((tensor.get_shape().as_list()[0], opt.dim), dtype=tf.sg_floatx)

    # do rnn loop
    h, out = init_h, []
    for i in range(tensor.get_shape().as_list()[1]):
        # apply step func
        h = step(h, tensor[:, i, :])
        # save result
        out.append(h.sg_expand_dims(axis=1))

    # merge tensor
    if opt.last_only:
        out = out[-1].sg_squeeze(axis=1)
    else:
        out = tf.concat(out, 1)

    return out


@tf.sg_rnn_layer_func
def sg_gru(tensor, opt):
    r"""Applies a GRU.
    
    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        bias: Boolean. If True, biases are added.
        ln: Boolean. If True, layer normalization is applied.   
        init_state: A 2-D `Tensor`. If None, the initial state is set to zeros.
        last_only: Boolean. If True, the outputs in the last time step are returned.
    
    Returns:
      A `Tensor`. If last_only is True, the output tensor has shape [batch size, dim].
      Otherwise, [batch size, time steps, dim].
    """

    # layer normalization
    # noinspection PyPep8
    ln = lambda v: _ln_rnn(v, gamma, beta) if opt.ln else v

    # step func
    def step(hh, x):
        # update gate
        z = tf.sigmoid(ln(tf.matmul(x, w_z) + tf.matmul(hh, u_z) + (b_z if opt.bias else 0)))
        # reset gate
        r = tf.sigmoid(ln(tf.matmul(x, w_r) + tf.matmul(hh, u_r) + (b_r if opt.bias else 0)))
        # h_hat
        hh = tf.tanh(ln(tf.matmul(x, w_h) + tf.matmul(r * hh, u_h) + (b_h if opt.bias else 0)))
        # final output
        y = (1. - z) * hh + z * hh
        return y

    # parameter initialize
    w_z = tf.sg_initializer.orthogonal('W_z', (opt.in_dim, opt.dim))
    u_z = tf.sg_initializer.identity('U_z', opt.dim)
    w_r = tf.sg_initializer.orthogonal('W_r', (opt.in_dim, opt.dim))
    u_r = tf.sg_initializer.identity('U_r', opt.dim)
    w_h = tf.sg_initializer.orthogonal('W_h', (opt.in_dim, opt.dim))
    u_h = tf.sg_initializer.identity('U_h', opt.dim)
    if opt.bias:
        b_z = tf.sg_initializer.constant('b_z', opt.dim)
        b_r = tf.sg_initializer.constant('b_r', opt.dim)
        b_h = tf.sg_initializer.constant('b_h', opt.dim)

    # layer normalization parameters
    if opt.ln:
        # offset, scale parameter
        beta = tf.sg_initializer.constant('beta', opt.dim)
        gamma = tf.sg_initializer.constant('gamma', opt.dim, value=1)

    # initial state
    init_h = opt.init_state if opt.init_state is not None \
        else tf.zeros((tensor.get_shape().as_list()[0], opt.dim), dtype=tf.sg_floatx)

    # do rnn loop
    h, out = init_h, []
    for i in range(tensor.get_shape().as_list()[1]):
        # apply step function
        h = step(h, tensor[:, i, :])
        # save result
        # noinspection PyUnresolvedReferences
        out.append(h.sg_expand_dims(axis=1))

    # merge tensor
    if opt.last_only:
        out = out[-1].sg_squeeze(axis=1)
    else:
        out = tf.concat(out, 1)

    return out


@tf.sg_rnn_layer_func
def sg_lstm(tensor, opt):
    r"""Applies an LSTM.

    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        in_dim: A positive `integer`. The size of input dimension.
        dim: A positive `integer`. The size of output dimension.
        bias: Boolean. If True, biases are added.
        ln: Boolean. If True, layer normalization is applied.   
        init_state: A 2-D `Tensor`. If None, the initial state is set to zeros.
        last_only: Boolean. If True, the outputs in the last time step are returned.
    
    Returns:
      A `Tensor`. If last_only is True, the output tensor has shape [batch size, dim].
      Otherwise, [batch size, time steps, dim].
    """
    # layer normalization
    # noinspection PyPep8
    ln = lambda v: _ln_rnn(v, gamma, beta) if opt.ln else v

    # step func
    def step(hh, cc, x):
        # forget gate
        f = tf.sigmoid(ln(tf.matmul(x, w_f) + tf.matmul(hh, u_f) + (b_f if opt.bias else 0)))
        # input gate
        ii = tf.sigmoid(ln(tf.matmul(x, w_i) + tf.matmul(hh, u_i) + (b_i if opt.bias else 0)))
        # new cell value
        c_new = tf.tanh(ln(tf.matmul(x, w_c) + tf.matmul(hh, u_c) + (b_c if opt.bias else 0)))
        # out gate
        o = tf.sigmoid(ln(tf.matmul(x, w_o) + tf.matmul(hh, u_o) + (b_o if opt.bias else 0)))
        # cell update
        cell = f * cc + ii * c_new
        # final output
        y = o * tf.tanh(cell)
        return y, cell

    # parameter initialize
    w_i = tf.sg_initializer.orthogonal('W_i', (opt.in_dim, opt.dim))
    u_i = tf.sg_initializer.identity('U_i', opt.dim)
    w_f = tf.sg_initializer.orthogonal('W_f', (opt.in_dim, opt.dim))
    u_f = tf.sg_initializer.identity('U_f', opt.dim)
    w_o = tf.sg_initializer.orthogonal('W_o', (opt.in_dim, opt.dim))
    u_o = tf.sg_initializer.identity('U_o', opt.dim)
    w_c = tf.sg_initializer.orthogonal('W_c', (opt.in_dim, opt.dim))
    u_c = tf.sg_initializer.identity('U_c', opt.dim)
    if opt.bias:
        b_i = tf.sg_initializer.constant('b_i', opt.dim)
        b_f = tf.sg_initializer.constant('b_f', opt.dim)
        b_o = tf.sg_initializer.constant('b_o', opt.dim, value=1)
        b_c = tf.sg_initializer.constant('b_c', opt.dim)

    # layer normalization parameters
    if opt.ln:
        # offset, scale parameter
        beta = tf.sg_initializer.constant('beta', opt.dim)
        gamma = tf.sg_initializer.constant('gamma', opt.dim, value=1)

    # initial state
    init_h = opt.init_state if opt.init_state is not None \
        else tf.zeros((tensor.get_shape().as_list()[0], opt.dim), dtype=tf.sg_floatx)

    # do rnn loop
    h, c, out = init_h, init_h, []
    for i in range(tensor.get_shape().as_list()[1]):
        # apply step function
        h, c = step(h, c, tensor[:, i, :])
        # save result
        out.append(h.sg_expand_dims(axis=1))

    # merge tensor
    if opt.last_only:
        out = out[-1].sg_squeeze(axis=1)
    else:
        out = tf.concat(out, 1)

    return out
