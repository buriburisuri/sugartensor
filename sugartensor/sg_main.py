# -*- coding: utf-8 -*-
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


def sg_global_step(as_tensor=True):
    global _global_step
    return _global_step

#
# global phase(train or infer) flag
#

_phase = tf.Variable(False, name='phase', trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
_phase_train = _phase.assign(True)
_phase_infer = _phase.assign(False)


def sg_phase():
    global _phase
    return _phase


def sg_set_train(sess):
    sess.run(_phase_train)


def sg_set_infer(sess):
    sess.run(_phase_infer)


#
# context helpers
#

_context = tf.sg_opt()


@contextmanager
def sg_context(**kwargs):
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
    @wraps(func)
    def wrapper(tensor, **kwargs):

        import sg_initializer as init
        import sg_activation

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
            for t in tf.get_collection(tf.GraphKeys.VARIABLES):
                scope_name = tf.get_variable_scope().name
                prefix = scope_name + '/' if len(scope_name) > 0 else ''
                i = t.name.rfind(prefix + 'layers/' + opt.name)
                if i >= 0:
                    exist_layers.append(t.name[i:].split('/')[-2])
            exist_layers = list(set(exist_layers))

            # layer name numbering
            if len(exist_layers) == 0:
                opt.name += '_1'
            else:
                opt.name += '_%d' % (max([int(n.split('_')[-1]) for n in exist_layers]) + 1)

        # all layer variables start with 'layers/' prefix
        with tf.variable_scope('layers', reuse=opt.reuse):

            with tf.variable_scope(opt.name):

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

                # apply normalization parameters
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
                if opt.reuse is None or not opt.reuse:
                    tf.sg_summary_activation(out)

                # save node info for reuse
                out._sugar = tf.sg_opt(func=func, arg=tf.sg_opt(kwargs) + _context,
                                       prev=tensor, is_layer=True, name=opt.name)
                # inject reuse function
                out.sg_reuse = types.MethodType(sg_reuse, out)

        return out

    return wrapper


def sg_rnn_layer_func(func):
    @wraps(func)
    def wrapper(tensor, **kwargs):

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
            for t in tf.get_collection(tf.GraphKeys.VARIABLES):
                scope_name = tf.get_variable_scope().name
                prefix = scope_name + '/' if len(scope_name) > 0 else ''
                i = t.name.rfind(prefix + 'layers/' + opt.name)
                if i >= 0:
                    exist_layers.append(t.name[i:].split('/')[-2])
            exist_layers = list(set(exist_layers))

            # layer name numbering
            if len(exist_layers) == 0:
                opt.name += '_1'
            else:
                opt.name += '_%d' % (max([int(n.split('_')[-1]) for n in exist_layers]) + 1)

        # all layer variables start with 'layers/' prefix
        with tf.variable_scope('layers', reuse=opt.reuse):

            with tf.variable_scope(opt.name):

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
                if opt.reuse is None or not opt.reuse:
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
        sys.path.append(path)
    globals()[mod_name] = importlib.import_module(mod_name)
    # find functions
    for func_name in dir(globals()[mod_name]):
        if isinstance(globals()[mod_name].__dict__.get(func_name), types.FunctionType):
            if not func_name.startswith('_'):
                # inject to tf.Variable type
                exec ('tf.Variable.%s = types.MethodType(%s.%s, None, tf.Variable)' % (func_name, mod_name, func_name))
                # inject to tf.Tensor type
                exec ('tf.Tensor.%s = types.MethodType(%s.%s, None, tf.Tensor)' % (func_name, mod_name, func_name))


def sg_inject_func(func):
    # inject to tf.Variable type
    exec ('tf.Variable.%s = func' % func.__name__)
    # inject to tf.Tensor type
    exec ('tf.Tensor.%s = func' % func.__name__)


#
# Queue Wrapper Annotator
#

@contextmanager
def sg_queue_context(sess=None):

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
    if not tf.app.flags.FLAGS.__dict__['__parsed']:
        tf.app.flags.FLAGS._parse_flags()
    return tf.sg_opt(tf.app.flags.FLAGS.__dict__['__flags'])


def sg_arg_def(**kwargs):
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


