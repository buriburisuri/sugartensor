# -*- coding: utf-8 -*-
import sugartensor as tf

__author__ = 'buriburisuri@gmail.com'


@tf.sg_sugar_func
def sg_sigmoid(x, opt):
    return tf.nn.sigmoid(x, name=opt.name)


@tf.sg_sugar_func
def sg_tanh(x, opt):
    return tf.nn.tanh(x, name=opt.name)


@tf.sg_sugar_func
def sg_relu(x, opt):
    return tf.nn.relu(x, name=opt.name)


@tf.sg_sugar_func
def sg_relu6(x, opt):
    return tf.nn.relu6(x, name=opt.name)


@tf.sg_sugar_func
def sg_leaky_relu(x, opt):
    return tf.select(tf.greater(x, 0), x, 0.01 * x, name=opt.name)


@tf.sg_sugar_func
def sg_elu(x, opt):
    return tf.nn.elu(x, name=opt.name)


@tf.sg_sugar_func
def sg_softplus(x, opt):
    return tf.nn.softplus(x, name=opt.name)


@tf.sg_sugar_func
def sg_softsign(x, opt):
    return tf.nn.softsign(x, name=opt.name)


@tf.sg_sugar_func
def sg_softmax(x, opt):
    dim = x.get_shape().ndims
    assert dim >= 2, 'softmax needs rank 2 at least'
    if dim == 2:
        return tf.nn.softmax(x, name=opt.name)
    else:
        ori_shape = [-1] + x.get_shape().as_list()[1:]
        new_shape = (-1, ori_shape[-1])
        return tf.reshape(tf.nn.softmax(tf.reshape(x, new_shape), name=opt.name), ori_shape)


@tf.sg_sugar_func
def sg_linear(x, opt):
    return x
