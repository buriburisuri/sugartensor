# -*- coding: utf-8 -*-
import types
from functools import wraps
import importlib
from contextlib import contextmanager

import sugartensor as tf

__author__ = 'njkim@jamonglab.com'

#
# default float, int precision
#


sg_floatx = tf.float32
sg_intx = tf.int32
sg_eps = 1e-8

#
# global variables
#

# global step
_global_step = tf.Variable(0, name='global_step', trainable=False)

# global learning rate
_learning_rate = tf.Variable(0., name='learning_rate', trainable=False)

# global phase(train or infer) flag
_phase = tf.Variable(False, name='phase', trainable=False, collections=[])

# context options
_context = tf.sg_opt()


def sg_global_step(as_tensor=True):
    global _global_step
    if as_tensor:
        return _global_step
    else:
        return tf.get_default_session().run(_global_step)


def sg_learning_rate(as_tensor=True):
    global _learning_rate
    if as_tensor:
        return _learning_rate
    else:
        return tf.get_default_session().run(_learning_rate)


def sg_phase(phase=None):
    global _phase
    if phase is None:
        return _phase
    elif phase == 'train':
        tf.get_default_session().run(tf.assign(_phase, True))
    elif phase == 'infer':
        tf.get_default_session().run(tf.assign(_phase, False))


def sg_current_context():
    global _context
    return _context


@contextmanager
def sg_context(**kwargs):
    global _context
    # set options when enter
    _context = tf.sg_opt(kwargs)
    yield
    # clear options when exit
    _context = tf.sg_opt()


#
# sugar function annotator
#

def sg_sugar_func(func):
    @wraps(func)
    def wrapper(tensor, **kwargs):
        if hasattr(tensor, '_sugar'):
            # template chaining
            return _sugar_node(func, tf.sg_opt(kwargs), tensor)
        else:
            # call sugar function
            return func(tensor, tf.sg_opt(kwargs))

    return wrapper


#
# layer function annotator
#


def sg_layer_func(func):
    @wraps(func)
    def wrapper(tensor, **kwargs):

        import sg_initializer as init
        import sg_activation

        # template process
        if hasattr(tensor, '_sugar'):
            # template chaining
            return _sugar_node(func, tf.sg_opt(kwargs), tensor, 'layer')

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

        # layer function name will be used as layer name
        opt += tf.sg_opt(name=func.__name__)

        # find existing layer names
        exist_layers = []
        for t in tf.get_collection(tf.GraphKeys.VARIABLES):
            i = t.name.rfind('layers/' + opt.name)
            if i >= 0:
                exist_layers.append(t.name[i:].split('/')[1])
        exist_layers = list(set(exist_layers))

        # layer name numbering
        if len(exist_layers) == 0:
            opt.name += '_1'
        else:
            opt.name += '_%d' % (max([int(n.split('_')[-1]) for n in exist_layers]) + 1)

        # all layer variables start with 'layers/' prefix
        with tf.variable_scope('layers'):

            with tf.variable_scope(opt.name):

                # call layer function
                out = func(tensor, opt)

                # apply batch normalization
                if opt.bn:
                    # offset, scale parameter
                    beta = init.constant('beta', out.get_shape().as_list()[-1])
                    gamma = init.constant('gamma', out.get_shape().as_list()[-1], value=1)

                    # offset, scale parameter
                    mean_running = init.constant('mean', out.get_shape().as_list()[-1])
                    variance_running = init.constant('variance', out.get_shape().as_list()[-1], value=1)

                    # calc batch mean, variance
                    mean, variance = tf.nn.moments(out, axes=range(len(out.get_shape()) - 1))

                    # update running mean, variance
                    def update_running_stat():
                        decay = 0.99
                        update_op = [mean_running.assign(mean_running * decay + mean * (1 - decay)),
                                     variance_running.assign(variance_running * decay + variance * (1 - decay))]
                        with tf.control_dependencies(update_op):
                            return tf.identity(mean), tf.identity(variance)

                    # select mean, variance by training phase
                    m, v = tf.cond(_phase,
                                   update_running_stat,  # updated running stat and batch mean, variance
                                   lambda: (mean_running, variance_running))  # saved mean, variance

                    # apply batch normalization
                    out = tf.nn.batch_normalization(out, m, v, beta, gamma, tf.sg_eps)

                # apply layer normalization
                if opt.ln:
                    # offset, scale parameter
                    beta = init.constant('beta', out.get_shape().as_list()[-1])
                    gamma = init.constant('gamma', out.get_shape().as_list()[-1], value=1)

                    # calc layer mean, variance for final axis
                    mean, variance = tf.nn.moments(out, axes=[len(out.get_shape()) - 1])

                    # apply layer normalization ( explicit broadcasting needed )
                    broadcast_shape = [-1] + [1] * (len(out.get_shape()) - 1)
                    out = (out - tf.reshape(mean, broadcast_shape)) \
                          / tf.reshape(tf.sqrt(variance + tf.sg_eps), broadcast_shape)

                    # apply parameter
                    out = gamma * out + beta

                # apply activation
                if opt.act:
                    out = getattr(sg_activation, 'sg_' + opt.act.lower())(out)
                    # add post-activation summary
                    tf.sg_summary_activation(out)

                # apply dropout
                if opt.dout:
                    out = tf.cond(_phase,
                                  lambda: tf.nn.dropout(out, 1 - opt.dout),
                                  lambda: out)

        return out

    return wrapper


#
# sugar template functions
#

def sg_template():
    return _sugar_node(None, None, None)


def _sugar_build(template, tensor):
    assert hasattr(template, '_sugar'), 'This tensor is not template.'

    # get all nodes in this template graph
    nodes, prev = [template], template._sugar.prev
    while prev is not None:
        nodes = [prev] + nodes
        prev = prev._sugar.prev

    # create real tensor graph
    res = tensor
    for node in nodes[1:]:  # exclude head node
        if node._sugar.type == 'layer':
            fn = sg_layer_func(node._sugar.func)
            res = fn(res, **node._sugar.arg)
        else:
            res = node._sugar.func(res, node._sugar.arg)

    return res


def _sugar_node(func, arg, prev, type=None):
    node = tf.constant(-10, name='template')
    node._sugar = tf.sg_opt(func=func, arg=arg, prev=prev, type=type)
    node.build = types.MethodType(_sugar_build, node)
    return node


#
# input wrapper function
#

def sg_input(shape=None, dtype=sg_floatx, name=None):
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
    # import module
    import sys
    if path not in list(sys.path):
        sys.path.append(path[0])
    globals()[mod_name] = importlib.import_module(mod_name)
    # find functions
    for func_name in dir(globals()[mod_name]):
        if isinstance(globals()[mod_name].__dict__.get(func_name), types.FunctionType):
            if not func_name.startswith('_'):
                # inject to tf.Variable type
                exec ('tf.Variable.%s = types.MethodType(%s.%s, None, tf.Variable)' % (func_name, mod_name, func_name))
                # inject to tf.Tensor type
                exec ('tf.Tensor.%s = types.MethodType(%s.%s, None, tf.Tensor)' % (func_name, mod_name, func_name))

