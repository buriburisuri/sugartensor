# -*- coding: utf-8 -*-
import sugartensor as tf

__author__ = 'buriburisuri@gmail.com'


@tf.sg_sugar_func
def sg_vgg_16(x, opt):

    # conv1
    with tf.sg_context(name='conv1', act='relu'):
        y = x.sg_conv(dim=64).sg_pool().sg_conv().sg_pool()

    # conv2
    with tf.sg_context(name='conv2', act='relu'):
        y = y.sg_conv(dim=128).sg_pool().sg_conv().sg_pool()

    # conv3
    with tf.sg_context(name='conv3', act='relu'):
        y = y.sg_conv(dim=256).sg_pool().sg_conv().sg_pool().sg_conv().sg_pool()

    # conv4
    with tf.sg_context(name='conv4', act='relu'):
        y = y.sg_conv(dim=512).sg_pool().sg_conv().sg_pool().sg_conv().sg_pool()

    # conv5
    with tf.sg_context(name='conv5', act='relu'):
        y = y.sg_conv(dim=512).sg_pool().sg_conv().sg_pool().sg_conv().sg_pool()

    # fc6~7
    with tf.sg_context(act='relu', dout=opt.dout):
        y = y.sg_flatten().sg_dense(dim=4096, name='fc6')
        y = y.sg_dense(dim=4096, name='fc7')

    # final fc8
    y = y.sg_dense(dim=10, name='fc8')

    return y
