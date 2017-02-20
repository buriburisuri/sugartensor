from __future__ import absolute_import
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
    r""""See [Xu, et al. 2015](https://arxiv.org/pdf/1505.00853v2.pdf)

    Args:
        x: A tensor
        opt:
          name: A name for the operation (optional).
    
    Returns:
      A `Tensor` with the same type and shape as `x`.
    """
    return tf.where(tf.greater(x, 0), x, 0.01 * x, name=opt.name)


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
    return tf.nn.softmax(x, name=opt.name)


# noinspection PyUnusedLocal
@tf.sg_sugar_func
def sg_linear(x, opt):
    return x
