# -*- coding: utf-8 -*-
import sugartensor as tf


__author__ = 'njkim@jamonglab.com'


if __name__ == '__main__':

    # set log level to debug
    tf.sg_verbosity(10)

    # MNIST input tensor ( with QueueRunner )
    mnist = tf.sg_data.Mnist()

    # create training graph
    logit = (tf.sg_port()
             .sg_float().sg_flatten()
             .sg_dense(dim=400, act='relu', bn=True, name='fc_1')
             .sg_dense(dim=200, act='relu', bn=True, name='fc_2')
             .sg_dense(dim=100, name='fc_3'))

    # cross entropy loss with logit ( for training set )
    loss = logit.sg_plug(mnist.train.image).sg_ce(target=mnist.train.label.sg_int())

    # accuracy evaluation ( for validation set )
    acc = (logit.sg_plug(mnist.valid.image).sg_softmax()
           .sg_accuracy(target=mnist.valid.label.sg_int(), name='val'))

    # train
    tf.sg_train(loss=loss, save_dir='asset/train/mnist',
                total_batch=mnist.train.total_batch, eval_metric=[acc])

