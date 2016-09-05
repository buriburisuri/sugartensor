# -*- coding: utf-8 -*-
import tensorflow as tf
import sugartensor as st

__author__ = 'njkim@jamonglab.com'


#
# evaluation layer
#

def accuracy(self, **kwargs):
    opt = st.opt(kwargs) + st.opt(k=1, one_hot=False)
    assert opt.target is not None, 'target is mandatory.'

    # get top k prediction
    hit = tf.nn.in_top_k(self, opt.target, opt.k)

    # calc accuracy
    out = tf.reduce_sum(tf.cast(hit, st.sg_floatx)) / tf.reduce_sum(tf.cast(tf.ones_like(opt.target), st.sg_floatx))

    # add summary
    st.summary(out, prefix='2.accuracy')

    return out
