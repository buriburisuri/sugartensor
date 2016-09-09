# -*- coding: utf-8 -*-
import sugartensor as tf

__author__ = 'buriburisuri@gmail.com'


#
# evaluation layer
#


@tf.sg_sugar_func
def sg_accuracy(tensor, opt):
    assert opt.target is not None, 'target is mandatory.'
    opt += tf.sg_opt(k=1)

    # # calc accuracy
    out = tf.identity(tf.equal(tensor.sg_argmax(), tf.cast(opt.target, tf.int64)).sg_float(), name='acc')
    # out = tf.identity(tf.nn.in_top_k(tensor, opt.target, opt.k).sg_float(), name='acc')

    return out
