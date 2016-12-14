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
    
    Returns:
      A `Tensor`.
    """
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True)

    # convolution layers
    with tf.sg_context(name='conv', act='relu'):
        conv = (tensor
                .sg_conv(dim=64)
                .sg_conv()
                .sg_pool()
                .sg_conv(dim=128)
                .sg_conv()
                .sg_pool()
                .sg_conv(dim=256)
                .sg_conv()
                .sg_conv()
                .sg_pool()
                .sg_conv(dim=512)
                .sg_conv()
                .sg_conv()
                .sg_pool()
                .sg_conv(dim=512)
                .sg_conv()
                .sg_conv()
                .sg_pool())

    # fully convolution layers
    with tf.sg_context(name='fc', act='relu', pad='VALID'):
        fc = (conv
              .sg_conv(dim=4096, size=7, dout=opt.dout)
              .sg_conv(dim=4096, size=1, dout=opt.dout)
              .sg_conv(dim=opt.num_class, size=1, act='linear'))

    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(dim=(1, 2))
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
    
    Returns:
      A `Tensor`.
    """
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True)

    # convolution layers
    with tf.sg_context(name='conv', act='relu'):
        conv = (tensor
                .sg_conv(dim=64)
                .sg_conv()
                .sg_pool()
                .sg_conv(dim=128)
                .sg_conv()
                .sg_pool()
                .sg_conv(dim=256)
                .sg_conv()
                .sg_conv()
                .sg_conv()
                .sg_pool()
                .sg_conv(dim=512)
                .sg_conv()
                .sg_conv()
                .sg_conv()
                .sg_pool()
                .sg_conv(dim=512)
                .sg_conv()
                .sg_conv()
                .sg_conv()
                .sg_pool())

    # fully convolution layers
    with tf.sg_context(name='fc', act='relu', pad='VALID'):
        fc = (conv
              .sg_conv(dim=4096, size=7, dout=opt.dout)
              .sg_conv(dim=4096, size=1, dout=opt.dout)
              .sg_conv(dim=opt.num_class, size=1, act='linear'))

    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(dim=(1, 2))
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
    
    Returns:
      A `Tensor`. 
    """
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True)

    # convolution layers ( residual net v2 arch )
    conv = (x
            .sg_conv(dim=64, size=7, stride=2)
            .sg_pool(size=3, stride=2, pad='SAME')
            .sg_resnet_layer(dim=64, num=3, stride=1)
            .sg_resnet_layer(dim=128, num=4, stride=2)
            .sg_resnet_layer(dim=256, num=6, stride=2)
            .sg_resnet_layer(dim=512, num=3, stride=2)
            .sg_bypass(act='relu', bn=True)
            .sg_pool(size=7, stride=1, avg=True))  # global average pool

    # fully convolution layers
    fc = (conv
          .sg_conv(dim=opt.num_class, size=1, act='linear', bn=False))

    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(dim=(1, 2))
        else:
            return fc


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
    
    Returns:
      A `Tensor`. 
    """
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True)

    # convolution layers ( residual net v2 arch )
    conv = (x
            .sg_conv(dim=64, size=7, stride=2)
            .sg_pool(size=3, stride=2, pad='SAME')
            .sg_resnet_layer(dim=64, num=3, stride=1)
            .sg_resnet_layer(dim=128, num=4, stride=2)
            .sg_resnet_layer(dim=256, num=23, stride=2)
            .sg_resnet_layer(dim=512, num=3, stride=2)
            .sg_bypass(act='relu', bn=True)
            .sg_pool(size=7, stride=1, avg=True))  # global average pool

    # fully convolution layers
    fc = (conv
          .sg_conv(dim=opt.num_class, size=1, act='linear', bn=False))

    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(dim=(1, 2))
        else:
            return fc


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
    
    Returns:
      A `Tensor`. 
    """
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True)

    # convolution layers ( residual net v2 arch )
    conv = (x
            .sg_conv(dim=64, size=7, stride=2)
            .sg_pool(size=3, stride=2, pad='SAME')
            .sg_resnet_layer(dim=64, num=3, stride=1)
            .sg_resnet_layer(dim=128, num=8, stride=2)
            .sg_resnet_layer(dim=256, num=36, stride=2)
            .sg_resnet_layer(dim=512, num=3, stride=2)
            .sg_bypass(act='relu', bn=True)
            .sg_pool(size=7, stride=1, avg=True))  # global average pool

    # fully convolution layers
    fc = (conv
          .sg_conv(dim=opt.num_class, size=1, act='linear', bn=False))

    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(dim=(1, 2))
        else:
            return fc


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
    
    Returns:
      A `Tensor`. 
    """
    opt += tf.sg_opt(num_class=1000, conv_only=False, squeeze=True)

    # convolution layers ( residual net v2 arch )
    conv = (x
            .sg_conv(dim=64, size=7, stride=2)
            .sg_pool(size=3, stride=2, pad='SAME')
            .sg_resnet_layer(dim=64, num=3, stride=1)
            .sg_resnet_layer(dim=128, num=24, stride=2)
            .sg_resnet_layer(dim=256, num=36, stride=2)
            .sg_resnet_layer(dim=512, num=3, stride=2)
            .sg_bypass(act='relu', bn=True)
            .sg_pool(size=7, stride=1, avg=True))  # global average pool

    # fully convolution layers
    fc = (conv
          .sg_conv(dim=opt.num_class, size=1, act='linear', bn=False))

    if opt.conv_only:
        return conv
    else:
        if opt.squeeze:
            return fc.sg_squeeze(dim=(1, 2))
        else:
            return fc


#
# Residual net utility
#

@tf.sg_sugar_func
def sg_resnet_layer(tensor, opt):
    r"""Applies basic architecture of residual net.
    
    Note that the fc layers in the original architecture 
      will be replaced with fully convolutional layers.
      For convenience, We still call them fc layers, though.
    
    Args:
      tensor: A `Tensor`.
      opt:
          dim: An integer. Dimension for this resnet layer
          num: Number of times to repeat

    Returns:
      A `Tensor`. 
    """
    assert opt.dim is not None, 'dim is mandatory.'
    assert opt.num is not None, 'num is mandatory.'

    # default stride
    opt += tf.sg_opt(stride=1)

    # 1st block
    out = (tensor
           .sg_bypass(act='relu', bn=True)
           .sg_conv(dim=opt.dim, size=1, act='relu', bn=True, stride=opt.stride)
           .sg_conv(dim=opt.dim, size=3, act='relu', bn=True)
           .sg_conv(dim=opt.dim*4, size=1, act='linear'))
    out += tensor.sg_conv(dim=opt.dim*4, size=1, stride=opt.stride)

    # middle blocks
    for _ in range(opt.num - 1):
        out_new = (out
                   .sg_bypass(act='relu', bn=True)
                   .sg_conv(dim=opt.dim, size=1, act='relu', bn=True)
                   .sg_conv(dim=opt.dim, size=3, act='relu', bn=True)
                   .sg_conv(dim=opt.dim*4, size=1, act='linear'))
        out += out_new

    return out
