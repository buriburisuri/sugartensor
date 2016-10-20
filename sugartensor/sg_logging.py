# -*- coding: utf-8 -*-
import sugartensor as tf
import os
import time
import sys

__author__ = 'buriburisuri@gmail.com'


#
# summary function
#


def _pretty_name(tensor):
    return ':'.join(tensor.name.split(':')[:-1])


def sg_summary_loss(tensor, prefix='10. loss'):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.scalar_summary(name + '/avg', tf.reduce_mean(tensor))
        tf.histogram_summary(name, tensor)


def sg_summary_metric(tensor, prefix='20. metric'):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.scalar_summary(name + '/avg', tf.reduce_mean(tensor))
        tf.histogram_summary(name, tensor)


def sg_summary_activation(tensor, prefix='30. activation'):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.scalar_summary(name + '/norm', tf.global_norm([tensor]))
        tf.scalar_summary(name + '/ratio',
                          tf.reduce_mean(tf.cast(tf.greater(tensor, 0), tf.sg_floatx)))
        tf.histogram_summary(name, tensor)


def sg_summary_param(tensor, prefix='40. parameters'):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.scalar_summary(name + '/norm', tf.global_norm([tensor]))
        tf.histogram_summary(name, tensor)


def sg_summary_gradient(tensor, gradient, prefix='50. gradient'):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.scalar_summary(name + '/norm', tf.global_norm([gradient]))
        tf.histogram_summary(name, gradient)


def sg_summary_image(tensor, prefix=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.image_summary(name, tensor)


def sg_summary_audio(tensor, sample_rate=16000, prefix=None):
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor)
    # summary statistics
    with tf.name_scope('summary'):
        tf.audio_summary(name, tensor, sample_rate)


#
# logger wrappers
#

# use tensorflow logger
_logger = tf.logging._logger


def _log_prefix():

    # Returns (filename, linenumber) for the stack frame.
    def _get_file_line():

        # pylint: disable=protected-access
        f = sys._getframe()
        # pylint: enable=protected-access
        our_file = f.f_code.co_filename
        f = f.f_back
        while f:
            code = f.f_code
            if code.co_filename != our_file:
                return (code.co_filename, f.f_lineno)
            f = f.f_back
        return ('<unknown>', 0)

    # current time
    now = time.time()
    now_tuple = time.localtime(now)
    now_millisecond = int(1e3 * (now % 1.0))

    # current filename and line
    filename, line = _get_file_line()
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
