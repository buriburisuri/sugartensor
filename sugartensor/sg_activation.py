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
    r""""See Xu, et al. 2015 `https://arxiv.org/pdf/1505.00853v2.pdf`

    Args:
        x: A tensor
        opt:
          name: name: A name for the operation (optional).
    """
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
    r"""Computes softmax activations of a tensor with 2 dimensions or more.
    Note that the native tensorflow function `tf.nn.softmax` must take a 2-D tensor only as its argument.

    Args:
      x: A `Tensor` with shape `[..., num_classes]`.
         Must be one of the following types: `half`, `float32`, `float64`.
      opt:
        name: name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type and shape as input tensor `x`.

    For example,

    ```
    x = [[[2, -1, 3], [3, 1, -2]]]
    sg_softmax(x) => [[[ 0.26538792  0.01321289  0.72139919]
                       [ 0.87560058  0.11849965  0.00589975]]]
    ```
    """
    return tf.nn.softmax(x, name=opt.name)


# noinspection PyUnusedLocal
@tf.sg_sugar_func
def sg_linear(x, opt):
    return x
