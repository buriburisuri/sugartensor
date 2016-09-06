# -*- coding: utf-8 -*-
import sugartensor as tf

__author__ = 'njkim@jamonglab.com'


@tf.sg_sugar_func
def sg_ce(tensor, opt):
    opt += tf.sg_opt(one_hot=False)
    assert opt.target is not None, 'target is mandatory.'

    if opt.one_hot:
        out = tf.nn.softmax_cross_entropy_with_logits(tensor, opt.target)
    else:
        out = tf.nn.sparse_softmax_cross_entropy_with_logits(tensor, opt.target)

    # add summary
    tf.sg_summary_loss(out)

    return out


@tf.sg_sugar_func
def sg_bce(tensor, opt):
    assert opt.target is not None, 'target is mandatory.'

    out = tf.nn.sigmoid_cross_entropy_with_logits(tensor, opt.target)

    # add summary
    tf.sg_summary_loss(out)

    return out


@tf.sg_sugar_func
def sg_mse(tensor, opt):
    assert opt.target is not None, 'target is mandatory.'

    # squared error
    out = tf.square(tensor - opt.target)

    # add summary
    tf.sg_summary_loss(out)

    return out


@tf.sg_sugar_func
def sg_mae(tensor, opt):
    assert opt.target is not None, 'target is mandatory.'

    # absolute error
    out = tf.abs(tensor - opt.target)

    # add summary
    tf.sg_summary_loss(out)

    return out


@tf.sg_sugar_func
def sg_hinge(tensor, opt):
    assert opt.target is not None, 'target is mandatory.'

    # default margin
    opt += tf.sg_opt('margin', 1)

    # reshape target
    shape = tensor.get_shape().as_list()
    broadcast_shape = [-1] + [1] * (len(shape) - 2) + [shape[-1]]
    target = tf.cast(tf.reshape(opt.target, broadcast_shape), st.sg_floatx)

    # hinge loss
    out = tf.maximum(opt.margin - target * tensor, 0)

    # add summary
    tf.sg_summary_loss(out)

    return out
