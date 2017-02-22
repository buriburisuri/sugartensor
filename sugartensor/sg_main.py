from __future__ import absolute_import
import types
from functools import wraps
import importlib
from contextlib import contextmanager

import sugartensor as tf


__author__ = 'buriburisuri@gmail.com'


#
# default float, int precision
#


sg_floatx = tf.float32
sg_intx = tf.int32
sg_eps = 1e-8

#
# global step
#

_global_step = tf.Variable(0, name='global_step', trainable=False)


def sg_global_step():
    r"""Gets global step count

    Returns:
      A 0-D `Tensor`.

    """
    global _global_step
    return _global_step

#
# global phase(train or infer) flag
#

_phase = tf.Variable(False, name='phase', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
_phase_train = _phase.assign(True)    # phase set ops ( to train )
_phase_infer = _phase.assign(False)   # phase set ops ( to infer )


def sg_phase():
    r""" Gets current training phase

    Returns:
      A boolean `Tensor`. If True, it is in the training phase, otherwise inference phase.
    """
    global _phase
    return _phase


def sg_set_train(sess):
    r"""Set current phase as training mode

    Args:
      sess: session to work

    Returns:
      None
    """
    sess.run(_phase_train)


def sg_set_infer(sess):
    r"""Sets current phase as inference mode

    Args:
      sess: session to work

    Returns:
      None
    """
    sess.run(_phase_infer)


#
# context helpers
#

_context = tf.sg_opt()


@contextmanager
def sg_context(**kwargs):
    r"""Context helper for computational graph building.
    Makes all elements within the with Block share the parameters.

    For example, in the following example, the default value of parameter `bn` will be set to True
    in the all layers within the with block.
    
    ```
    with tf.sg_context(bn=True):
        ...
        ...
    ```

    Args:
      **kwargs:
        in_dim: An integer. The size of input dimension, which is set to the last one by default.
        dim: An integer. The size of output dimension. Has the same value as in_dim by default.
        bn: Boolean. If True, batch normalization is applied.
        ln: Boolean. If True, layer normalization is applied.
        dout: A float of range [0, 100). A dropout rate. Default is 0..
        bias: Boolean. If True (Default), biases are added.
        name: A name for the layer. By default, the function name is assigned.
        act: A name of activation function. e.g., `sigmoid`, `tanh`, etc.
        reuse: `True` or `None`; if `True`, we go into reuse mode for this `layer` scope
          as well as all sub-scopes; if `None`, we just inherit the parent scope reuse.

    Returns:
      None
    """
    global _context
    # set options when enter
    _context = tf.sg_opt(kwargs)
    if _context.name:
        _context.context_name = _context.name
        _context.name = None
        with tf.variable_scope(_context.context_name):
            yield
    else:
        yield
    # clear options when exit
    _context = tf.sg_opt()


#
# sugar function annotator
#

def sg_sugar_func(func):
    r""" Decorates a function `func` so that it can be a sugar function. 
    Sugar function can be used in a chainable manner.

    Args:
        func: function to decorate

    Returns:
      A sugar function.

    """
    @wraps(func)
    def wrapper(tensor, **kwargs):
        # call sugar function
        out = func(tensor, tf.sg_opt(kwargs))
        # save node info for reuse
        out._sugar = tf.sg_opt(func=func, arg=tf.sg_opt(kwargs)+_context, prev=tensor)
        # inject reuse function
        out.sg_reuse = types.MethodType(sg_reuse, out)
        return out

    return wrapper


#
# layer function annotator
#


def sg_layer_func(func):
    r"""Decorates a function `func` as a sg_layer function.

    Args:
        func: function to decorate
    """
    @wraps(func)
    def wrapper(tensor, **kwargs):
        r"""Manages arguments of `tf.sg_opt`.
        
        Args:
          tensor: A `tensor` (automatically passed by decorator).
          kwargs:
            shape:  A list of integers. The shape of `tensor`. Inferred if not specified.
            in_dim: An integer. The size of input dimension, which is set to the last one by default.
            dim: An integer. The size of output dimension. Has the same value as in_dim by default.
            bn: Boolean. If True, batch normalization is applied.
            ln: Boolean. If True, layer normalization is applied.
            dout: A float of range [0, 100). A dropout rate. Set to 0 by default.
            bias: Boolean. If True, biases are added. As a default, it is set to True 
            name: A name for the layer. As a default, the function name is assigned.
            act: A name of activation function. e.g., `sigmoid`, `tanh`, etc.
            reuse: `True` or `None`; if `True`, we go into reuse mode for this `layer` scope 
              as well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
        """

        from . import sg_initializer as init
        from . import sg_activation

        # kwargs parsing
        opt = tf.sg_opt(kwargs) + _context

        # set default argument
        try:
            shape = tensor.get_shape().as_list()
            # batch normalization off, layer normalization off, dropout off
            opt += tf.sg_opt(shape=shape, in_dim=shape[-1], dim=shape[-1], bn=False, ln=False, dout=0)
            assert not (opt.bn and opt.ln), 'one of batch normalization and layer normalization is available.'

            # disable bias when normalization on
            opt += tf.sg_opt(bias=not (opt.bn or opt.ln))
        finally:
            pass

        # automatic layer naming
        if opt.name is None:

            # layer function name will be used as layer name
            opt.name = func.__name__.replace('sg_', '')

            # find existing layer names
            exist_layers = []
            for t in tf.global_variables():
                scope_name = tf.get_variable_scope().name
                prefix = scope_name + '/' if len(scope_name) > 0 else ''
                i = t.name.rfind(prefix + opt.name)
                if i >= 0:
                    exist_layers.append(t.name[i:].split('/')[-2])
            exist_layers = list(set(exist_layers))

            # layer name numbering
            if len(exist_layers) == 0:
                opt.name += '_1'
            else:
                opt.name += '_%d' % (max([int(n.split('_')[-1]) for n in exist_layers]) + 1)

        with tf.variable_scope(opt.name, reuse=opt.reuse) as scope:

            # call layer function
            out = func(tensor, opt)

            # apply batch normalization
            if opt.bn:
                # offset, scale parameter
                beta = init.constant('beta', opt.dim)
                gamma = init.constant('gamma', opt.dim, value=1)

                # offset, scale parameter
                mean_running = init.constant('mean', opt.dim)
                variance_running = init.constant('variance', opt.dim, value=1)

                # calc batch mean, variance
                mean, variance = tf.nn.moments(out, axes=list(range(len(out.get_shape()) - 1)))

                # add running mean, variance to UPDATE_OP collection
                decay = 0.99
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mean_running.assign(mean_running * decay + mean * (1 - decay)))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, variance_running.assign(variance_running * decay + variance * (1 - decay)))

                # select mean, variance by training phase
                m, v = tf.cond(_phase,
                               lambda: (mean, variance),  # batch mean, variance
                               lambda: (mean_running, variance_running))  # saved mean, variance

                # apply batch normalization
                out = tf.nn.batch_normalization(out, m, v, beta, gamma, tf.sg_eps)

            # apply layer normalization
            if opt.ln:
                # offset, scale parameter
                beta = init.constant('beta', opt.dim)
                gamma = init.constant('gamma', opt.dim, value=1)

                # calc layer mean, variance for final axis
                mean, variance = tf.nn.moments(out, axes=[len(out.get_shape()) - 1], keep_dims=True)

                # apply normalization
                out = (out - mean) / tf.sqrt(variance + tf.sg_eps)
                # apply parameter
                out = gamma * out + beta

            # apply activation
            if opt.act:
                out = getattr(sg_activation, 'sg_' + opt.act.lower())(out)

            # apply dropout
            if opt.dout:
                out = tf.cond(_phase,
                              lambda: tf.nn.dropout(out, 1 - opt.dout),
                              lambda: out)

            # rename tensor
            out = tf.identity(out, 'out')

            # add final output summary
            if not scope.reuse:
                tf.sg_summary_activation(out)

            # save node info for reuse
            out._sugar = tf.sg_opt(func=func, arg=tf.sg_opt(kwargs) + _context,
                                   prev=tensor, is_layer=True, name=opt.name)
            # inject reuse function
            out.sg_reuse = types.MethodType(sg_reuse, out)

        return out

    return wrapper


def sg_rnn_layer_func(func):
    r"""Decorates function as sg_rnn_layer functions.
    Args:
        func: function to decorate
    """
    @wraps(func)
    def wrapper(tensor, **kwargs):
        r"""Manages arguments of `tf.sg_opt`.

        Args:
          tensor: automatically passed by decorator
          kwargs:
              in_dim: An integer. The size of input dimension, which is set to the last one by default.
              dim: An integer. The size of output dimension. Has the same value as in_dim by default.
              ln: Boolean. If True, layer normalization is applied.
              bias: Boolean. If True, biases are added. As a default, it is set to True
              name: A name for the layer. As a default, the function name is assigned.
              reuse: `True` or `None`; if `True`, we go into reuse mode for this `layer` scope
                as well as all sub-scopes; if `None`, we just inherit the parent scope reuse.
        """

        # kwargs parsing
        opt = tf.sg_opt(kwargs) + _context

        # set default argument
        try:
            shape = tensor.get_shape().as_list()
            # dropout off
            opt += tf.sg_opt(shape=shape, in_dim=shape[-1], dim=shape[-1], dout=0)
            # disable bias when normalization on
            opt += tf.sg_opt(bias=not opt.ln)
        finally:
            pass

        # automatic layer naming
        if opt.name is None:

            # layer function name will be used as layer name
            opt.name = func.__name__.replace('sg_', '')

            # find existing layer names
            exist_layers = []
            for t in tf.global_variables():
                scope_name = tf.get_variable_scope().name
                prefix = scope_name + '/' if len(scope_name) > 0 else ''
                i = t.name.rfind(prefix + opt.name)
                if i >= 0:
                    exist_layers.append(t.name[i:].split('/')[-2])
            exist_layers = list(set(exist_layers))

            # layer name numbering
            if len(exist_layers) == 0:
                opt.name += '_1'
            else:
                opt.name += '_%d' % (max([int(n.split('_')[-1]) for n in exist_layers]) + 1)

        with tf.variable_scope(opt.name, reuse=opt.reuse) as scope:

            # call layer function
            out = func(tensor, opt)

            # apply dropout
            if opt.dout:
                out = tf.cond(_phase,
                              lambda: tf.nn.dropout(out, 1 - opt.dout),
                              lambda: out)

            # rename tensor
            out = tf.identity(out, 'out')

            # add final output summary
            if scope.reuse:
                tf.sg_summary_activation(out)

            # save node info for reuse
            out._sugar = tf.sg_opt(func=func, arg=tf.sg_opt(kwargs) + _context,
                                   prev=tensor, is_layer=True, name=opt.name)
            # inject reuse function
            out.sg_reuse = types.MethodType(sg_reuse, out)

        return out

    return wrapper


#
# reuse functions for graph cloning
#


# noinspection PyProtectedMember
def sg_reuse(tensor, **opt):
    r""" Reconstruct computational graph of `tensor` so all the parameters 
    can be reused and replace its input tensor with `opt.input`.

    Args:
      tensor: A `Tensor` (automatically given by chaining).
      **opt:
        input: A `Tensor` that will replace the original input tensor.

    Returns:
      Reconstructed tensor nodes.
    """
    opt = tf.sg_opt(opt)
    assert hasattr(tensor, '_sugar'), 'cannot reuse this node.'
    assert opt.input is not None, 'input is mandatory.'

    # get all nodes in this graph
    nodes, prev = [tensor], tensor._sugar.prev
    while prev is not None:
        nodes = [prev] + nodes
        prev = prev._sugar.prev if hasattr(prev, '_sugar') else None

    # create graph again for this input
    out = opt.input
    for node in nodes[1:]:  # exclude head node
        if node._sugar.is_layer:
            fn = tf.sg_layer_func(node._sugar.func)
            if node._sugar.arg.context_name:
                with tf.variable_scope(node._sugar.arg.context_name):
                    out = fn(out, **(node._sugar.arg + tf.sg_opt(name=node._sugar.name, reuse=True)))
            else:
                out = fn(out, **(node._sugar.arg + tf.sg_opt(name=node._sugar.name, reuse=True)))
        else:
            out = node._sugar.func(out, node._sugar.arg)

    return out


#
# input wrapper function
#

def sg_input(shape=None, dtype=sg_floatx, name=None):
    r"""Creates a placeholder.

    Args:
      shape: A tuple/list of integers. If an integers is given, it will turn to a list.
      dtype: A data type. Default is float32.
      name: A name for the placeholder.

    Returns:
      A wrapped placeholder `Tensor`.
    """
    if shape is None:
        return tf.placeholder(dtype, shape=None, name=name)
    else:
        if not isinstance(shape, (list, tuple)):
            shape = [shape]
        return tf.placeholder(dtype, shape=[None] + list(shape), name=name)


#
# helper function for sugar and layer function injection
#

def sg_inject(path, mod_name):
    r"""Converts all functions in the given Python module to sugar functions
    so that they can be used in a chainable manner.

    Args:
      path: A string. Path to the Python module
      mod_name: A string. The name of the Python module to inject.

    Returns:
      None
    """
    # import module
    import sys
    if path not in list(sys.path):
        sys.path.append(path)
    globals()[mod_name] = importlib.import_module(mod_name)
    # find functions
    for func_name in dir(globals()[mod_name]):
        if isinstance(globals()[mod_name].__dict__.get(func_name), types.FunctionType):
            if not func_name.startswith('_'):
                # inject to tf.Variable type
                exec('tf.Variable.%s = %s.%s' % (func_name, mod_name, func_name))
                # inject to tf.Tensor type
                exec('tf.Tensor.%s = %s.%s' % (func_name, mod_name, func_name))


def sg_inject_func(func):
    r"""Converts the function `func` to a sugar function
    so that it can be used in a chainable manner.

    Args:
      func: A function to inject.

    Returns:
      None
    """
    # inject to tf.Variable type
    exec ('tf.Variable.%s = func' % func.__name__)
    # inject to tf.Tensor type
    exec ('tf.Tensor.%s = func' % func.__name__)


#
# Queue Wrapper Annotator
#

# noinspection PyUnboundLocalVariable
@contextmanager
def sg_queue_context(sess=None):
    r"""Context helper for queue routines.

    Args:
      sess: A session to open queues. If not specified, a new session is created.

    Returns:
      None
    """

    # default session
    sess = tf.get_default_session() if sess is None else sess

    # thread coordinator
    coord = tf.train.Coordinator()
    try:
        # start queue thread
        threads = tf.train.start_queue_runners(sess, coord)
        yield
    finally:
        # stop queue thread
        coord.request_stop()
        # wait thread to exit.
        coord.join(threads)


#
# Command line argument util funcs
#

# noinspection PyProtectedMember
def sg_arg():
    r"""Gets current command line options

    Returns:
      tf.sg_opt instance that is updated with current commandd line options.
    """
    if not tf.app.flags.FLAGS.__dict__['__parsed']:
        tf.app.flags.FLAGS._parse_flags()
    return tf.sg_opt(tf.app.flags.FLAGS.__dict__['__flags'])


def sg_arg_def(**kwargs):
    r"""Defines command line options

    Args:
      **kwargs:
        key: A name for the option.
        value : Default value or a tuple of (default value, description).

    Returns:
      None

    For example, 
    
    ```
    # Either of the following two lines will define `--n_epoch` command line argument and set its default value as 1.
    
    tf.sg_arg_def(n_epoch=1) 
    tf.sg_arg_def(n_epoch=(1, 'total number of epochs'))
    ```
    """
    for k, v in kwargs.items():
        if type(v) is tuple or type(v) is list:
            v, c = v[0], v[1]
        else:
            c = k
        if type(v) is str:
            tf.app.flags.DEFINE_string(k, v, c)
        elif type(v) is int:
            tf.app.flags.DEFINE_integer(k, v, c)
        elif type(v) is float:
            tf.app.flags.DEFINE_float(k, v, c)
        elif type(v) is bool:
            tf.app.flags.DEFINE_bool(k, v, c)
