from __future__ import absolute_import
import sugartensor as tf


__author__ = 'buriburisuri@gmail.com'


#
# VGG nets
#


@tf.sg_sugar_func
def sg_vgg_16(tensor, opt):
    r"""Applies vgg 16 model.
    
    Note that the fc layers in the original architecture 
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.
    
    Args:
      tensor: A `Tensor`
      opt:
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        bn: True or False(default). If True, batch normal will be applied
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network
    
    Returns:
      A `Tensor`.
    """
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True, act='relu', bn=False)

    # convolution layers
    with tf.sg_context(name=opt.name, act=opt.act, bn=opt.bn, reuse=opt.reuse):
        conv = (tensor
                .sg_conv(dim=64, name='conv1_1')
                .sg_conv(name='conv1_2')
                .sg_pool()
                .sg_conv(dim=128, name='conv2_1')
                .sg_conv(name='conv2_2')
                .sg_pool()
                .sg_conv(dim=256, name='conv3_1')
                .sg_conv(name='conv3_2')
                .sg_conv(name='conv3_3')
                .sg_pool()
                .sg_conv(dim=512, name='conv4_1')
                .sg_conv(name='conv4_2')
                .sg_conv(name='conv4_3')
                .sg_pool()
                .sg_conv(dim=512, name='conv5_1')
                .sg_conv(name='conv5_2')
                .sg_conv(name='conv5_3')
                .sg_pool())

    # fully convolution layers
    with tf.sg_context(name=opt.name, act=opt.act, pad='VALID', bn=opt.bn, reuse=opt.reuse):
        fc = (conv
              .sg_conv(dim=4096, size=7, dout=opt.dout, name='fc1')
              .sg_conv(dim=4096, size=1, dout=opt.dout, name='fc2')
              .sg_conv(dim=opt.num_class, size=1, act='linear', name='fc3'))

    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(axis=(1, 2))
        else:
            return fc


@tf.sg_sugar_func
def sg_vgg_19(tensor, opt):
    r"""Applies vgg 19 model.
    
    Note that the fc layers in the original architecture 
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.
    
    Args:
      tensor: A `Tensor`.
      opt:
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        bn: True or False(default). If True, batch normal will be applied
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network
    
    Returns:
      A `Tensor`.
    """
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True, act='relu')

    # convolution layers
    with tf.sg_context(name=opt.name, act=opt.act, bn=opt.bn, reuse=opt.reuse):
        conv = (tensor
                .sg_conv(dim=64, name='conv1_1')
                .sg_conv(name='conv1_2')
                .sg_pool()
                .sg_conv(dim=128, name='conv2_1')
                .sg_conv(name='conv2_2')
                .sg_pool()
                .sg_conv(dim=256, name='conv3_1')
                .sg_conv(name='conv3_2')
                .sg_conv(name='conv3_3')
                .sg_conv(name='conv3_4')
                .sg_pool()
                .sg_conv(dim=512, name='conv4_1')
                .sg_conv(name='conv4_2')
                .sg_conv(name='conv4_3')
                .sg_conv(name='conv4_4')
                .sg_pool()
                .sg_conv(dim=512, name='conv5_1')
                .sg_conv(name='conv5_2')
                .sg_conv(name='conv5_3')
                .sg_conv(name='conv5_4')
                .sg_pool())

    # fully convolution layers
    with tf.sg_context(name=opt.name, act=opt.act, pad='VALID', bn=opt.bn, reuse=opt.reuse):
        fc = (conv
              .sg_conv(dim=4096, size=7, dout=opt.dout, name='fc1')
              .sg_conv(dim=4096, size=1, dout=opt.dout, name='fc2')
              .sg_conv(dim=opt.num_class, size=1, act='linear', name='fc3'))

    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(axis=(1, 2))
        else:
            return fc


#
# Residual nets ( version 2 )
#


@tf.sg_sugar_func
def sg_resnet_50(x, opt):
    r"""Applies residual net 50 model.
    
    Note that the fc layers in the original architecture 
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.
    
    Args:
      x: A `Tensor`.
      opt:
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network
    
    Returns:
      A `Tensor`. 
    """
    return _resnet_graph(x, opt, [3, 4, 6, 3])


@tf.sg_sugar_func
def sg_resnet_101(x, opt):
    r"""Applies residual net 101 model.
    
    Note that the fc layers in the original architecture 
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.
    
    Args:
      x: A `Tensor`.
      opt:
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network

    Returns:
      A `Tensor`. 
    """
    return _resnet_graph(x, opt, [3, 4, 23, 3])


@tf.sg_sugar_func
def sg_resnet_152(x, opt):
    r"""Applies residual net 152 model.
    
    Note that the fc layers in the original architecture 
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.
    
    Args:
      x: A `Tensor`.
      opt:
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network

    Returns:
      A `Tensor`. 
    """
    return _resnet_graph(x, opt, [3, 8, 36, 3])


@tf.sg_sugar_func
def sg_resnet_200(x, opt):
    r"""Applies residual net 200 model.
    
    Note that the fc layers in the original architecture 
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.
    
    Args:
      x: A `Tensor`.
      opt:
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network

    Returns:
      A `Tensor`. 
    """
    return _resnet_graph(x, opt, [3, 24, 36, 3])


@tf.sg_sugar_func
def sg_densenet_121(x, opt):
    r"""Applies dense net 121 model.

    Note that the fc layers in the original architecture
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.

    Args:
      x: A `Tensor`.
      opt:
        k: An integer. The Growth rate of densenet. Default is 32.
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network

    Returns:
      A `Tensor`.
    """
    return _densenet_graph(x, opt + tf.sg_opt(k=32), [6, 12, 24, 16])


@tf.sg_sugar_func
def sg_densenet_169(x, opt):
    r"""Applies dense net 169 model.

    Note that the fc layers in the original architecture
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.

    Args:
      x: A `Tensor`.
      opt:
        k: An integer. The Growth rate of densenet. Default is 32.
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network

    Returns:
      A `Tensor`.
    """
    return _densenet_graph(x, opt + tf.sg_opt(k=32), [6, 12, 32, 32])


@tf.sg_sugar_func
def sg_densenet_201(x, opt):
    r"""Applies dense net 201 model.

    Note that the fc layers in the original architecture
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.

    Args:
      x: A `Tensor`.
      opt:
        k: An integer. The Growth rate of densenet. Default is 32.
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network

    Returns:
      A `Tensor`.
    """
    return _densenet_graph(x, opt + tf.sg_opt(k=32), [6, 12, 48, 32])


@tf.sg_sugar_func
def sg_densenet_161(x, opt):
    r"""Applies dense net 161 model.

    Note that the fc layers in the original architecture
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.

    Args:
      x: A `Tensor`.
      opt:
        k: An integer. The Growth rate of densenet. Default is 48.
        num_class: An integer. Number of class. Default is 1000.
        conv_only: Boolean. If True, fc layers are not applied. Default is False.
        squeeze: Boolean. If True (default), the dimensions with size 1 in the final outputs will be removed.
        act: String. 'relu' (default). the activation function name
        reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
        name: String(Optional). If provided, used as the scope name of this network

    Returns:
      A `Tensor`.
    """
    return _densenet_graph(x, opt + tf.sg_opt(k=48), [6, 12, 36, 24])


#
# Residual net utility
#

@tf.sg_sugar_func
def sg_resnet_layer(x, opt):
    r"""Applies basic architecture of residual net.
    
    Note that the fc layers in the original architecture 
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.
    
    Args:
      x: A `Tensor`.
      opt:
          dim: An integer. Dimension for this resnet layer
          num: Number of times to repeat
          act: String. 'relu' (default). the activation function name
          reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
          name: String. (optional) Used as convolution layer prefix

    Returns:
      A `Tensor`. 
    """
    assert opt.dim is not None, 'dim is mandatory.'
    assert opt.num is not None, 'num is mandatory.'

    # default stride
    opt += tf.sg_opt(stride=1, act='relu')

    # format convolutional layer name
    def cname(index):
        return opt.name if opt.name is None else opt.name + '_%d' % index

    with tf.sg_context(bias=False, reuse=opt.reuse):
        # 1st block
        out = (x
               .sg_bypass(act=opt.act, bn=True, name=cname(0))
               .sg_conv(dim=opt.dim, size=1, act=opt.act, stride=opt.stride, bn=True, name=cname(1))
               .sg_conv(dim=opt.dim, size=3, act=opt.act, bn=True, name=cname(2))
               .sg_conv(dim=opt.dim*4, size=1, name=cname(3)))
        out += x.sg_conv(dim=opt.dim * 4, size=1, stride=opt.stride, name=cname(4))

        # middle blocks
        for i in range(1, opt.num):
            out_new = (out
                       .sg_bypass(act=opt.act, name=cname(4 * i + 1))
                       .sg_conv(dim=opt.dim, size=1, act=opt.act, bn=True, name=cname(4 * i + 2))
                       .sg_conv(dim=opt.dim, size=3, act=opt.act, bn=True, name=cname(4 * i + 3))
                       .sg_conv(dim=opt.dim*4, size=1, name=cname(4 * i + 4)))
            out += out_new

    return out


# construct residual network graphs
def _resnet_graph(x, opt, nums):

    # default option
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True, act='relu')

    # convolution layers ( residual net v2 arch )
    with tf.sg_context(name=opt.name):
        conv = (x
                .sg_conv(dim=64, size=7, stride=2, bias=False, reuse=opt.reuse, name='conv1')
                .sg_pool(size=3, stride=2, pad='SAME')
                .sg_resnet_layer(dim=64, num=nums[0], stride=1, act=opt.act, reuse=opt.reuse, name='conv2')
                .sg_resnet_layer(dim=128, num=nums[1], stride=2, act=opt.act, reuse=opt.reuse, name='conv3')
                .sg_resnet_layer(dim=256, num=nums[2], stride=2, act=opt.act, reuse=opt.reuse, name='conv4')
                .sg_resnet_layer(dim=512, num=nums[3], stride=2, act=opt.act, reuse=opt.reuse, name='conv5')
                .sg_bypass(act=opt.act, bn=True, reuse=opt.reuse, name='final_act')  # final activation
                .sg_pool(size=7, stride=1, avg=True))  # global average pool

        # fully convolution layers
        fc = conv.sg_conv(dim=opt.num_class, size=1, act='linear', bn=False, reuse=opt.reuse, name='fc')

    # return selectively
    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(axis=(1, 2))
        else:
            return fc


#
# Dense net utility
#

@tf.sg_sugar_func
def sg_densenet_layer(x, opt):
    r"""Applies basic architecture of densenet layer.

    Note that the fc layers in the original architecture
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.

    Args:
      x: A `Tensor`.
      opt:
          dim: An integer. Dimension for this resnet layer
          num: Number of times to repeat
          act: String. 'relu' (default). the activation function name
          trans: Boolean. If True(default), transition layer will be applied.
          reuse: Boolean(Optional). If True, all variables will be loaded from previous network.
          name: String. (optional) Used as convolution layer prefix

    Returns:
      A `Tensor`.
    """
    assert opt.dim is not None, 'dim is mandatory.'
    assert opt.num is not None, 'num is mandatory.'

    # default stride
    opt += tf.sg_opt(stride=1, act='relu', trans=True)

    # format convolutional layer name
    def cname(index):
        return opt.name if opt.name is None else opt.name + '_%d' % index

    # dense layer
    with tf.sg_context(bias=False, reuse=opt.reuse):
        out = x
        for i in range(opt.num):
            # dense block
            out_new = (out
                       .sg_bypass(act=opt.act, bn=True, name=cname(3 * i + 1))
                       .sg_conv(dim=opt.dim // 4, size=1, act=opt.act, bn=True, name=cname(3 * i + 2))
                       .sg_conv(dim=opt.dim, size=3, name=cname(3 * i + 3)))
            out = tf.concat([out_new, out], 3)

        # transition layer
        if opt.trans:
            out = (out
                   .sg_bypass(act=opt.act, bn=True, name=cname(3 * i + 4))
                   .sg_conv(size=1, name=cname(3 * i + 5))
                   .sg_pool(avg=True))

    return out


# construct dense network graphs
def _densenet_graph(x, opt, nums):

    # default option
    opt += tf.sg_opt(num_class=1000, k=32, conv_only=False, squeeze=True, act='relu')

    # convolution layers ( dense net arch )
    with tf.sg_context(name=opt.name):
        conv = (x
                .sg_conv(dim=opt.k, size=7, stride=2, bias=False, reuse=opt.reuse, name='conv1')
                .sg_pool(size=3, stride=2, pad='SAME')
                .sg_densenet_layer(dim=opt.k, num=nums[0], act=opt.act, reuse=opt.reuse, name='conv2')
                .sg_densenet_layer(dim=opt.k, num=nums[1], act=opt.act, reuse=opt.reuse, name='conv3')
                .sg_densenet_layer(dim=opt.k, num=nums[2], act=opt.act, reuse=opt.reuse, name='conv4')
                .sg_densenet_layer(dim=opt.k, num=nums[3], act=opt.act, trans=False, reuse=opt.reuse, name='conv5')
                .sg_bypass(act=opt.act, bn=True, reuse=opt.reuse, name='final_act')  # final activation
                .sg_pool(size=7, stride=1, avg=True))  # global average pool

        # fully convolution layers
        fc = conv.sg_conv(dim=opt.num_class, size=1, act='linear', bn=False, reuse=opt.reuse, name='fc')

    # return selectively
    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(axis=(1, 2))
        else:
            return fc
