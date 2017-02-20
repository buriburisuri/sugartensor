from __future__ import absolute_import
import sugartensor as tf
import os
import time
import sys


__author__ = 'buriburisuri@gmail.com'


#
# summary function
#


# noinspection PyTypeChecker
def sg_summary_loss(tensor, prefix='losses', name=None):
    r"""Register `tensor` to summary report as `loss`

    Args:
      tensor: A `Tensor` to log as loss
      prefix: A `string`. A prefix to display in the tensor board web UI.
      name: A `string`. A name to display in the tensor board web UI.

    Returns:
      None
    """
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    try:
        tf.summary.scalar(name, tf.reduce_mean(tensor))
        tf.summary.histogram(name + '-h', tensor)
    except:
        pass


# noinspection PyTypeChecker
def sg_summary_metric(tensor, prefix='metrics', name=None):
    r"""Register `tensor` to summary report as `metric`

    Args:
      tensor: A `Tensor` to log as metric
      prefix: A `string`. A prefix to display in the tensor board web UI.
      name: A `string`. A name to display in the tensor board web UI.

    Returns:
      None
    """
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    try:
        tf.summary.scalar(name, tf.reduce_mean(tensor))
        tf.summary.histogram(name + '-h', tensor)
    except:
        pass


def sg_summary_gradient(tensor, gradient, prefix=None, name=None):
    r"""Register `tensor` to summary report as `gradient`

    Args:
      tensor: A `Tensor` to log as gradient
      gradient: A 0-D `Tensor`. A gradient to log
      prefix: A `string`. A prefix to display in the tensor board web UI.
      name: A `string`. A name to display in the tensor board web UI.

    Returns:
        None
    """
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _full_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    try:
        tf.summary.scalar(name + '/grad', tf.reduce_mean(tf.abs(gradient)))
        tf.summary.histogram(name + '/grad-h', tf.abs(gradient))
    except:
        pass


def sg_summary_activation(tensor, prefix=None, name=None):
    r"""Register `tensor` to summary report as `activation`

    Args:
      tensor: A `Tensor` to log as activation
      prefix: A `string`. A prefix to display in the tensor board web UI.
      name: A `string`. A name to display in the tensor board web UI.

    Returns:
      None
    """
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    try:
        tf.summary.scalar(name + '/ratio',
                          tf.reduce_mean(tf.cast(tf.greater(tensor, 0), tf.sg_floatx)))
        tf.summary.histogram(name + '/ratio-h', tensor)
    except:
        pass


def sg_summary_param(tensor, prefix=None, name=None):
    r"""Register `tensor` to summary report as `parameters`

    Args:
      tensor: A `Tensor` to log as parameters
      prefix: A `string`. A prefix to display in the tensor board web UI.
      name: A `string`. A name to display in the tensor board web UI.

    Returns:
      None
    """
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    try:
        norm = tensor
        tf.summary.scalar(name + '/abs', tf.reduce_mean(tf.abs(tensor)))
        tf.summary.histogram(name + '/abs-h', tf.abs(tensor))
    except:
        pass


def sg_summary_image(tensor, prefix=None, name=None):
    r"""Register `tensor` to summary report as `image`

    Args:
      tensor: A tensor to log as image
      prefix: A `string`. A prefix to display in the tensor board web UI.
      name: A `string`. A name to display in the tensor board web UI.

    Returns:
        None
    """
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _full_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    try:
        tf.summary.image(name + '-im', tensor)
    except:
        pass


def sg_summary_audio(tensor, sample_rate=16000, prefix=None, name=None):
    r"""Register `tensor` to summary report as audio

    Args:
      tensor: A `Tensor` to log as audio
      sample_rate : An int. Sample rate to report. Default is 16000.
      prefix: A `string`. A prefix to display in the tensor board web UI.
      name: A `string`. A name to display in the tensor board web UI.

    Returns:
        None
    """
    # defaults
    prefix = '' if prefix is None else prefix + '/'
    # summary name
    name = prefix + _full_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    try:
        tf.summary.audio(name + '-au', tensor, sample_rate)
    except:
        pass


def _pretty_name(tensor):
    return ''.join(tensor.name.split(':')[:-1]).split('/')[-1]


def _full_name(tensor):
    return ''.join(tensor.name.split(':')[:-1])


#
# logger wrappers
#

# use tensorflow logger
# pylint: disable=protected-access
# noinspection PyProtectedMember
_logger = tf.logging._logger


def _log_prefix():

    # Returns (filename, line number) for the stack frame.
    def _get_file_line():

        # pylint: disable=protected-access
        # noinspection PyProtectedMember
        f = sys._getframe()
        # pylint: enable=protected-access
        our_file = f.f_code.co_filename
        f = f.f_back
        while f:
            code = f.f_code
            if code.co_filename != our_file:
                return code.co_filename, f.f_lineno
            f = f.f_back
        return '<unknown>', 0

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
