# -*- coding: utf-8 -*-
import sugartensor as tf
import os
import time

__author__ = 'njkim@jamonglab.com'

#
# summary function
#


def _pretty_name(tensor):
    return ':'.join(tensor.name.split(':')[:-1])


def sg_summary(tensor, prefix=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.scalar_summary('10.avg/' + name, tf.reduce_mean(tensor))
        tf.scalar_summary('20.ratio/' + name,
                          tf.reduce_mean(tf.cast(tf.greater(tensor, 0), tf.sg_floatx)))
        tf.scalar_summary('30.max/' + name, tf.reduce_max(tensor))
        tf.histogram_summary(name, tensor)


def sg_summary_loss(tensor, prefix='10. loss'):
    return sg_summary(tensor, prefix)


def sg_summary_activation(tensor, prefix='40. activation'):
    return sg_summary(tensor, prefix)


def sg_summary_param(tensor, prefix='40.parameters'):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.scalar_summary(name, tf.reduce_mean(tensor))


def sg_summary_image(tensor, prefix=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary_image'):
        tf.image_summary(name, tensor)


#
# logger wrappers
#

# use tensorflow logger
_logger = tf.logging._logger

def _log_prefix():

    # current time
    now = time.time()
    now_tuple = time.localtime(now)
    now_millisecond = int(1e3 * (now % 1.0))

    # current filename and line
    filename, line = tf.logging._GetFileAndLine()
    basename = os.path.basename(filename)

    s = '%02d%02d:%02d:%02d:%02d.%03d:%s:%d] ' % (
        now_tuple[1],  # month
        now_tuple[2],  # day
        now_tuple[3],  # hour
        now_tuple[4],  # min
        now_tuple[5],  # sec
        now_millisecond,
        basename,
        line)

    return s


def sg_verbosity(verbosity=0):
    _logger.setLevel(verbosity)

def sg_debug(msg, *args, **kwargs):
    _logger.debug(_log_prefix() + msg, *args, **kwargs)

def sg_info(msg, *args, **kwargs):
    _logger.info(_log_prefix() + msg, *args, **kwargs)

def sg_warn(msg, *args, **kwargs):
    _logger.warn(_log_prefix() + msg, *args, **kwargs)

def sg_error(msg, *args, **kwargs):
    _logger.error(_log_prefix() + msg, *args, **kwargs)

def sg_fatal(msg, *args, **kwargs):
    _logger.fatal(_log_prefix() + msg, *args, **kwargs)
