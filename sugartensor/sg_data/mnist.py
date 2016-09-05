# -*- coding: utf-8 -*-
import sugartensor as tf
from tensorflow.examples.tutorials.mnist import input_data

__author__ = 'mansour'


# constant sg_data to tensor conversion with queue support
def _data_to_tensor(data_list, batch_size, num_epochs):

    # convert to constant tensor
    const_list = [tf.constant(data) for data in data_list]

    # create queue from constant tensor
    queue_list = tf.train.slice_input_producer(const_list, num_epochs=num_epochs, capacity=128)

    # create batch queue
    return tf.train.batch(queue_list, batch_size, capacity=128, num_threads=4)


class Mnist(object):

    _data_dir = './asset/data/mnist'

    def __init__(self, batch_size=128, reshape=False, one_hot=False, num_epochs=None):

        # load sg_data set
        data_set = input_data.read_data_sets(Mnist._data_dir, reshape=reshape, one_hot=one_hot)

        self.batch_size = batch_size

        # save each sg_data set
        _train = data_set.train
        _valid = data_set.validation
        _test = data_set.test

        # member initialize
        self.train, self.valid, self.test = tf.sg_opt(), tf.sg_opt, tf.sg_opt()

        # convert to tensor queue
        self.train.image, self.train.label = \
            _data_to_tensor([_train.images, _train.labels], batch_size, num_epochs)
        self.valid.image, self.valid.label = \
            _data_to_tensor([_valid.images, _valid.labels], batch_size, num_epochs)
        self.test.image, self.test.label = \
            _data_to_tensor([_test.images, _test.labels], batch_size, num_epochs)

        # calc total batch count
        self.train.total_batch = _train.labels.shape[0] // self.batch_size
        self.valid.total_batch = _valid.labels.shape[0] // self.batch_size
        self.test.total_batch = _test.labels.shape[0] // self.batch_size
