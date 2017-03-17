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
@tf.sg_summary_func(prefix='losses', prettify_name=True)
def sg_summary_loss(tensor, opt):
    r"""Register `tensor` to summary report as `loss`

    Args:
      tensor: A `Tensor` to log as loss
      opt:
          prefix: A `string`. A prefix to display in the tensor board web UI.
          name: A `string`. A name to display in the tensor board web UI.

    Returns:
      None
    """
    tf.summary.scalar(opt.name, tf.reduce_mean(tensor))
    tf.summary.histogram(opt.name + '-h', tensor)


# noinspection PyTypeChecker
@tf.sg_summary_func(prefix='metrics', prettify_name=True)
def sg_summary_metric(tensor, opt):
    r"""Register `tensor` to summary report as `metric`

    Args:
      tensor: A `Tensor` to log as metric
      opt:
          prefix: A `string`. A prefix to display in the tensor board web UI.
          name: A `string`. A name to display in the tensor board web UI.

    Returns:
      None
    """
    tf.summary.scalar(opt.name, tf.reduce_mean(tensor))
    tf.summary.histogram(opt.name + '-h', tensor)


@tf.sg_summary_func()
def sg_summary_gradient(tensor, opt):
    r"""Register `tensor` to summary report as `gradient`

    Args:
      tensor: A `Tensor` to log as gradient
      opt:
          gradient: A 0-D `Tensor`. A gradient to log
          prefix: A `string`. A prefix to display in the tensor board web UI.
          name: A `string`. A name to display in the tensor board web UI.

    Returns:
        None
    """
    assert opt.gradient is not None, 'gradient is mandatory.'

    tf.summary.scalar(opt.name + '/grad', tf.reduce_mean(tf.abs(opt.gradient)))
    tf.summary.histogram(opt.name + '/grad-h', tf.abs(opt.gradient))


@tf.sg_summary_func()
def sg_summary_activation(tensor, opt):
    r"""Register `tensor` to summary report as `activation`

    Args:
      tensor: A `Tensor` to log as activation
      opt:
          prefix: A `string`. A prefix to display in the tensor board web UI.
          name: A `string`. A name to display in the tensor board web UI.

    Returns:
      None
    """
    tf.summary.scalar(opt.name + '/ratio',
                      tf.reduce_mean(tf.cast(tf.greater(tensor, 0), tf.sg_floatx)))
    tf.summary.histogram(opt.name + '/ratio-h', tensor)


@tf.sg_summary_func()
def sg_summary_param(tensor):
    r"""Register `tensor` to summary report as `parameters`

    Args:
      tensor: A `Tensor` to log as parameters
      opt:
          prefix: A `string`. A prefix to display in the tensor board web UI.
          name: A `string`. A name to display in the tensor board web UI.

    Returns:
      None
    """
    tf.summary.scalar(opt.name + '/abs', tf.reduce_mean(tf.abs(tensor)))
    tf.summary.histogram(opt.name + '/abs-h', tf.abs(tensor))


@tf.sg_summary_func()
def sg_summary_image(tensor, opt):
    r"""Register `tensor` to summary report as `image`

    Args:
      tensor: A tensor to log as image
      opt:
          prefix: A `string`. A prefix to display in the tensor board web UI.
          name: A `string`. A name to display in the tensor board web UI.

    Returns:
        None
    """
    tf.summary.image(opt.name + '-im', tensor)


@tf.sg_summary_func()
def sg_summary_audio(tensor):
    r"""Register `tensor` to summary report as audio

    Args:
      tensor: A `Tensor` to log as audio
      opt:
          sample_rate : An int. Sample rate to report. Default is 16000.
          prefix: A `string`. A prefix to display in the tensor board web UI.
          name: A `string`. A name to display in the tensor board web UI.

    Returns:
        None
    """
    opt += tf.sg_opt(sample_rate=16000)
    tf.summary.audio(opt.name + '-au', tensor, opt.sample_rate)


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
