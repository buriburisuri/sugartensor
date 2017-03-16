from __future__ import absolute_import
import sugartensor as tf
import logging
import os
import time
import sys
import re
from tensorflow.python.ops import gen_logging_ops


__author__ = 'namju.kim@kakaocorp.com'


#
# summary function
#


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
    _scalar(name, tf.reduce_mean(tensor))
    _histogram(name + '-h', tensor)


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
    _scalar(name, tf.reduce_mean(tensor))
    _histogram(name + '-h', tensor)


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
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    # noinspection PyBroadException
    _scalar(name + '/grad', tf.reduce_mean(tf.abs(gradient)))
    _histogram(name + '/grad-h', tf.abs(gradient))


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
    _scalar(name + '/ratio',
            tf.reduce_mean(tf.cast(tf.greater(tensor, 0), tf.sg_floatx)))
    _histogram(name + '/ratio-h', tensor)


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
    _scalar(name + '/abs', tf.reduce_mean(tf.abs(tensor)))
    _histogram(name + '/abs-h', tf.abs(tensor))


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
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    if not tf.get_variable_scope().reuse:
        tf.summary.image(name + '-im', tensor)


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
    name = prefix + _pretty_name(tensor) if name is None else prefix + name
    # summary statistics
    if not tf.get_variable_scope().reuse:
        tf.summary.audio(name + '-au', tensor, sample_rate)


def _pretty_name(tensor):
    name = ''.join(tensor.name.split(':')[:-1])
    return re.sub(r'gpu_[0-9]+/', '', name)


def _scalar(name, tensor):
    if not tf.get_variable_scope().reuse and not tf.sg_get_context().reuse:
        val = gen_logging_ops._scalar_summary(name, tensor)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, val)


def _histogram(name, tensor):
    if not tf.get_variable_scope().reuse and not tf.sg_get_context().reuse:
        val = gen_logging_ops._histogram_summary(name, tensor)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, val)


#
# logger wrappers
#

_logger = logging.getLogger('SugarTensor')
_logger.addHandler(logging.StreamHandler())


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
    _logger.debug('D ' + _log_prefix() + msg, *args, **kwargs)


def sg_info(msg, *args, **kwargs):
    _logger.info('I ' + _log_prefix() + msg, *args, **kwargs)


def sg_warn(msg, *args, **kwargs):
    _logger.warn('W ' + _log_prefix() + msg, *args, **kwargs)


def sg_error(msg, *args, **kwargs):
    _logger.error('E ' + _log_prefix() + msg, *args, **kwargs)


def sg_fatal(msg, *args, **kwargs):
    _logger.fatal('F ' + _log_prefix() + msg, *args, **kwargs)
