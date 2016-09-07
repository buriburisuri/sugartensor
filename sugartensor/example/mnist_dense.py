# -*- coding: utf-8 -*-
import sugartensor as tf


__author__ = 'njkim@jamonglab.com'


if __name__ == '__main__':

    # set log level to debug
    tf.sg_verbosity(10)

    # MNIST input tensor ( with QueueRunner )
    data = tf.sg_data.Mnist()

    # inputs
    x = data.train.image
    y = data.train.label

    # create training graph
    logit = (x.sg_flatten()
             .sg_dense(dim=400, act='relu', bn=True, name='fc_1')
             .sg_dense(dim=200, act='relu', bn=True, name='fc_2')
             .sg_dense(dim=100, name='fc_3'))

    # cross entropy loss with logit ( for training set )
    loss = logit.sg_ce(target=y)

    # accuracy evaluation ( for validation set )
    acc = (tf.sg_reuse(logit, input=data.valid.image).sg_softmax()
           .sg_accuracy(target=data.valid.label, name='val'))

    # train
    tf.sg_train(loss=loss, eval_metric=[acc], ep_size=data.train.num_batch)

