from __future__ import absolute_import
import sugartensor as tf
from tensorflow.examples.tutorials.mnist import input_data


__author__ = 'mansour'


# constant sg_data to tensor conversion with queue support
def _data_to_tensor(data_list, batch_size, name=None):
    r"""Returns batch queues from the whole data. 
    
    Args:
      data_list: A list of ndarrays. Every array must have the same size in the first dimension.
      batch_size: An integer.
      name: A name for the operations (optional).
      
    Returns:
      A list of tensors of `batch_size`.
    """
    # convert to constant tensor
    const_list = [tf.constant(data) for data in data_list]

    # create queue from constant tensor
    queue_list = tf.train.slice_input_producer(const_list, capacity=batch_size*128, name=name)

    # create batch queue
    return tf.train.shuffle_batch(queue_list, batch_size, capacity=batch_size*128,
                                  min_after_dequeue=batch_size*32, name=name)


class Mnist(object):
    r"""Downloads Mnist datasets and puts them in queues.
    """
    _data_dir = './asset/data/mnist'

    def __init__(self, batch_size=128, reshape=False, one_hot=False):

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
            _data_to_tensor([_train.images, _train.labels.astype('int32')], batch_size, name='train')
        self.valid.image, self.valid.label = \
            _data_to_tensor([_valid.images, _valid.labels.astype('int32')], batch_size, name='valid')
        self.test.image, self.test.label = \
            _data_to_tensor([_test.images, _test.labels.astype('int32')], batch_size, name='test')

        # calc total batch count
        self.train.num_batch = _train.labels.shape[0] // batch_size
        self.valid.num_batch = _valid.labels.shape[0] // batch_size
        self.test.num_batch = _test.labels.shape[0] // batch_size
