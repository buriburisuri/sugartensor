# -*- coding: utf-8 -*-
import sugartensor as tf


__author__ = 'njkim@jamonglab.com'


if __name__ == '__main__':

    # set log level to debug
    tf.sg_verbosity(10)

    # MNIST input tensor ( with QueueRunner )
    mnist = tf.sg_data.Mnist()

    # inputs
    x = mnist.train.image.sg_float()
    y = mnist.train.label.sg_int()

    # create training graph
    logit = (x.sg_flatten()
             .sg_dense(dim=400, act='relu', bn=True)
             .sg_dense(dim=200, act='relu', bn=True)
             .sg_dense(dim=100))

    # loss
    loss = logit.sg_ce(target=y)

    # train
    tf.sg_train(loss=loss, save_dir='asset/train/mnist', total_batch=mnist.train.total_batch)


