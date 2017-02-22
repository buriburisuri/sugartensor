from __future__ import absolute_import, print_function, unicode_literals
import sugartensor as tf
# noinspection PyPackageRequirements
import numpy as np
import os
import time
from tqdm import tqdm
from functools import wraps
from tensorflow.contrib.tensorboard.plugins import projector


__author__ = 'buriburisuri@gmail.com'


# global learning rate
_learning_rate = tf.Variable(0.001, dtype=tf.sg_floatx, name='learning_rate', trainable=False)


def sg_train(**kwargs):
    r"""Trains the model.

    Args:
      **kwargs:
        optim: A name for optimizer. 'MaxProp' (default), 'AdaMax', 'Adam', or 'sgd'.
        loss: A 0-D `Tensor` containing the value to minimize.
        lr: A Python Scalar (optional). Learning rate. Default is .001.
        beta1: A Python Scalar (optional). Default is .9.
        beta2: A Python Scalar (optional). Default is .99.

        eval_metric: A list of tensors containing the value to evaluate. Default is [].
        early_stop: Boolean. If True (default), the training should stop when the following two conditions are met.
          i. Current loss is less than .95 * previous loss.
          ii. Current learning rate is less than 5e-6.
        lr_reset: Boolean. If True, learning rate is set to opt.lr. when training restarts.
          Otherwise (Default), the value of the stored `_learning_rate` is taken.
        save_dir: A string. The root path to which checkpoint and log files are saved.
          Default is `asset/train`.
        max_ep: A positive integer. Maximum number of epochs. Default is 1000.    
        ep_size: A positive integer. Number of Total batches in an epoch. 
          For proper display of log. Default is 1e5.    

        save_interval: A Python scalar. The interval of saving checkpoint files.
          By default, for every 600 seconds, a checkpoint file is written.
        log_interval: A Python scalar. The interval of recoding logs.
          By default, for every 60 seconds, logging is executed.
        max_keep: A positive integer. Maximum number of recent checkpoints to keep. Default is 5.
        keep_interval: A Python scalar. How often to keep checkpoints. Default is 1 hour.

        category: Scope name or list to train

        tqdm: Boolean. If True (Default), progress bars are shown.
        console_log: Boolean. If True, a series of loss will be shown 
          on the console instead of tensorboard. Default is False.
    """
    opt = tf.sg_opt(kwargs)
    assert opt.loss is not None, 'loss is mandatory.'

    # default training options
    opt += tf.sg_opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, category='')

    # get optimizer
    train_op = sg_optim(opt.loss, optim=opt.optim, lr=_learning_rate,
                        beta1=opt.beta1, beta2=opt.beta2, category=opt.category)

    # define train function
    # noinspection PyUnusedLocal
    @sg_train_func
    def train_func(sess, arg):
        return sess.run([opt.loss] + train_op)[0]

    # run train function
    train_func(**opt)


def sg_init(sess):
    r""" Initializes session variables.
    
    Args:
      sess: Session to initialize. 
    """
    # initialize variables
    sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))


def sg_print(tensor_list):
    r"""Simple tensor printing function for debugging.
    Prints the value, shape, and data type of each tensor in the list.
    
    Args:
      tensor_list: A list/tuple of tensors or a single tensor.
      
    Returns:
      The value of the tensors.
      
    For example,
    
    ```python
    import sugartensor as tf
    a = tf.constant([1.])
    b = tf.constant([2.])
    out = tf.sg_print([a, b])
    # Should print [ 1.] (1,) float32
    #              [ 2.] (1,) float32
    print(out)
    # Should print [array([ 1.], dtype=float32), array([ 2.], dtype=float32)]
    ``` 
    """
    # to list
    if type(tensor_list) is not list and type(tensor_list) is not tuple:
        tensor_list = [tensor_list]

    # evaluate tensor list with queue runner
    with tf.Session() as sess:
        sg_init(sess)
        with tf.sg_queue_context():
            res = sess.run(tensor_list)
            for r in res:
                print(r, r.shape, r.dtype)

    if len(res) == 1:
        return res[0]
    else:
        return res


def sg_restore(sess, save_path, category=''):
    r""" Restores previously saved variables.

    Args:
      sess: A `Session` to use to restore the parameters.
      save_path: Path where parameters were previously saved.
      category: A `String` to filter variables starts with given category.

    Returns:

    """
    # to list
    if not isinstance(category, (tuple, list)):
        category = [category]

    # make variable list to load
    var_list = {}
    for cat in category:
        for t in tf.global_variables():
            if t.name.startswith(cat):
                var_list[t.name[:-2]] = t

    # restore parameters
    saver = tf.train.Saver(var_list)
    saver.restore(sess, save_path)


def sg_optim(loss, **kwargs):
    r"""Applies gradients to variables.

    Args:
        loss: A 0-D `Tensor` containing the value to minimize.
        kwargs:
          optim: A name for optimizer. 'MaxProp' (default), 'AdaMax', 'Adam', or 'sgd'.
          lr: A Python Scalar (optional). Learning rate. Default is .001.
          beta1: A Python Scalar (optional). Default is .9.
          beta2: A Python Scalar (optional). Default is .99.
          category: A string or string list. Specifies the variables that should be trained (optional).
            Only if the name of a trainable variable starts with `category`, it's value is updated.
            Default is '', which means all trainable variables are updated.
    """
    opt = tf.sg_opt(kwargs)

    # default training options
    opt += tf.sg_opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, category='')

    # select optimizer
    if opt.optim == 'MaxProp':
        optim = tf.sg_optimize.MaxPropOptimizer(learning_rate=opt.lr, beta2=opt.beta2)
    elif opt.optim == 'AdaMax':
        optim = tf.sg_optimize.AdaMaxOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2)
    elif opt.optim == 'Adam':
        optim = tf.train.AdamOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2)
    else:
        optim = tf.train.GradientDescentOptimizer(learning_rate=opt.lr)

    # get trainable variables
    if isinstance(opt.category, (tuple, list)):
        var_list = []
        for cat in opt.category:
            var_list.extend([t for t in tf.trainable_variables() if t.name.startswith(cat)])
    else:
        var_list = [t for t in tf.trainable_variables() if t.name.startswith(opt.category)]

    # calc gradient
    gradient = optim.compute_gradients(loss, var_list=var_list)

    # add summary
    for v, g in zip(var_list, gradient):
        # exclude batch normal statics
        if 'mean' not in v.name and 'variance' not in v.name \
                and 'beta' not in v.name and 'gamma' not in v.name:
            tf.sg_summary_gradient(v, g)

    # gradient update op
    grad_op = optim.apply_gradients(gradient, global_step=tf.sg_global_step())

    # extra update ops within category ( for example, batch normal running stat update )
    if isinstance(opt.category, (tuple, list)):
        update_op = []
        for cat in opt.category:
            update_op.extend([t for t in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if t.name.startswith(cat)])
    else:
        update_op = [t for t in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if t.name.startswith(opt.category)]

    return [grad_op] + update_op


def sg_train_func(func):
    r""" Decorates a function `func` as sg_train_func.
    
    Args:
        func: A function to decorate
    """
    @wraps(func)
    def wrapper(**kwargs):
        r""" Manages arguments of `tf.sg_opt`.

        Args:
          **kwargs:
            lr: A Python Scalar (optional). Learning rate. Default is .001.
    
            eval_metric: A list of tensors containing the value to evaluate. Default is [].
            early_stop: Boolean. If True (default), the training should stop when the following two conditions are met.
              i. Current loss is less than .95 * previous loss.
              ii. Current learning rate is less than 5e-6.
            lr_reset: Boolean. If True, learning rate is set to opt.lr. when training restarts.
              Otherwise (Default), the value of the stored `_learning_rate` is taken.
            save_dir: A string. The root path to which checkpoint and log files are saved.
              Default is `asset/train`.
            max_ep: A positive integer. Maximum number of epochs. Default is 1000.    
            ep_size: A positive integer. Number of Total batches in an epoch. 
              For proper display of log. Default is 1e5.    
    
            save_interval: A Python scalar. The interval of saving checkpoint files.
              By default, for every 600 seconds, a checkpoint file is written.
            log_interval: A Python scalar. The interval of recoding logs.
              By default, for every 60 seconds, logging is executed.
            max_keep: A positive integer. Maximum number of recent checkpoints to keep. Default is 5.
            keep_interval: A Python scalar. How often to keep checkpoints. Default is 1 hour.
    
            tqdm: Boolean. If True (Default), progress bars are shown.
            console_log: Boolean. If True, a series of loss will be shown 
              on the console instead of tensorboard. Default is False.
        """
        opt = tf.sg_opt(kwargs)

        # default training options
        opt += tf.sg_opt(lr=0.001,
                         save_dir='asset/train',
                         max_ep=1000, ep_size=100000,
                         save_interval=600, log_interval=60,
                         early_stop=True, lr_reset=False,
                         eval_metric=[],
                         max_keep=5, keep_interval=1,
                         tqdm=True, console_log=False)

        # make directory if not exist
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)

        # find last checkpoint
        last_file = tf.train.latest_checkpoint(opt.save_dir)
        if last_file:
            ep = start_ep = int(last_file.split('-')[1]) + 1
            start_step = int(last_file.split('-')[2])
        else:
            ep = start_ep = 1
            start_step = 0

        # checkpoint saver
        saver = tf.train.Saver(max_to_keep=opt.max_keep,
                               keep_checkpoint_every_n_hours=opt.keep_interval)

        # summary writer
        summary_writer = tf.summary.FileWriter(opt.save_dir, graph=tf.get_default_graph())

        # add learning rate summary
        tf.summary.scalar('learning_r', _learning_rate)

        # add evaluation metric summary
        for m in opt.eval_metric:
            tf.sg_summary_metric(m)

        # summary op
        summary_op = tf.summary.merge_all()

        # create session
        if opt.sess:
            sess = opt.sess
        else:
            # session with multiple GPU support
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # initialize variables
        sg_init(sess)

        # restore last checkpoint
        if last_file:
            saver.restore(sess, last_file)

        # set learning rate
        if start_ep == 1 or opt.lr_reset:
            sess.run(_learning_rate.assign(opt.lr))

        # logging
        tf.sg_info('Training started from epoch[%03d]-step[%d].' % (start_ep, start_step))

        try:
            # start data queue runner
            with tf.sg_queue_context(sess):

                # set session mode to train
                tf.sg_set_train(sess)

                # loss history for learning rate decay
                loss, loss_prev, early_stopped = None, None, False

                # time stamp for saving and logging
                last_saved = last_logged = time.time()

                # epoch loop
                for ep in range(start_ep, opt.max_ep + 1):

                    # show progressbar
                    if opt.tqdm:
                        iterator = tqdm(range(opt.ep_size), desc='train', ncols=70, unit='b', leave=False)
                    else:
                        iterator = range(opt.ep_size)

                    # batch loop
                    for _ in iterator:

                        # call train function
                        batch_loss = func(sess, opt)

                        # loss history update
                        if batch_loss is not None:
                            if loss is None:
                                loss = np.mean(batch_loss)
                            else:
                                loss = loss * 0.9 + np.mean(batch_loss) * 0.1

                        # saving
                        if time.time() - last_saved > opt.save_interval:
                            last_saved = time.time()
                            saver.save(sess, opt.save_dir + '/model-%03d' % ep,
                                       write_meta_graph=False,
                                       global_step=sess.run(tf.sg_global_step()))

                        # logging
                        if time.time() - last_logged > opt.log_interval:
                            last_logged = time.time()

                            # set session mode to infer
                            tf.sg_set_infer(sess)

                            # run evaluation op
                            if len(opt.eval_metric) > 0:
                                sess.run(opt.eval_metric)

                            if opt.console_log:   # console logging
                                # log epoch information
                                tf.sg_info('\tEpoch[%03d:lr=%7.5f:gs=%d] - loss = %s' %
                                           (ep, sess.run(_learning_rate), sess.run(tf.sg_global_step()),
                                            ('NA' if loss is None else '%8.6f' % loss)))
                            else:   # tensorboard logging
                                # run logging op
                                summary_writer.add_summary(sess.run(summary_op),
                                                           global_step=sess.run(tf.sg_global_step()))

                            # learning rate decay
                            if opt.early_stop and loss_prev:
                                # if loss stalling
                                if loss >= 0.95 * loss_prev:
                                    # early stopping
                                    current_lr = sess.run(_learning_rate)
                                    if current_lr < 5e-6:
                                        early_stopped = True
                                        break
                                    else:
                                        # decrease learning rate by half
                                        sess.run(_learning_rate.assign(current_lr / 2.))

                            # update loss history
                            loss_prev = loss

                            # revert session mode to train
                            tf.sg_set_train(sess)

                    # log epoch information
                    if not opt.console_log:
                        tf.sg_info('\tEpoch[%03d:lr=%7.5f:gs=%d] - loss = %s' %
                                   (ep, sess.run(_learning_rate), sess.run(tf.sg_global_step()),
                                    ('NA' if loss is None else '%8.6f' % loss)))

                    if early_stopped:
                        tf.sg_info('\tEarly stopped ( no loss progress ).')
                        break
        finally:
            # save last epoch
            saver.save(sess, opt.save_dir + '/model-%03d' % ep,
                       write_meta_graph=False,
                       global_step=sess.run(tf.sg_global_step()))

            # set session mode to infer
            tf.sg_set_infer(sess)

            # logging
            tf.sg_info('Training finished at epoch[%d]-step[%d].' % (ep, sess.run(tf.sg_global_step())))

            # close session
            if opt.sess is None:
                sess.close()

    return wrapper


# Under construction
# def sg_tsne(tensor, meta_file='metadata.tsv', save_dir='asset/tsne'):
#     r""" Manages arguments of `tf.sg_opt`.
#
#     Args:
#         save_dir: A string. The root path to which checkpoint and log files are saved.
#           Default is `asset/train`.
#     """
#
#     # make directory if not exist
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     # checkpoint saver
#     saver = tf.train.Saver()
#
#     # summary writer
#     summary_writer = tf.summary.FileWriter(save_dir, graph=tf.get_default_graph())
#
#     # embedding visualizer
#     config = projector.ProjectorConfig()
#     emb = config.embeddings.add()
#     emb.tensor_name = tensor.name   # tensor
#     # emb.metadata_path = os.path.join(save_dir, meta_file)   # metadata file
#     projector.visualize_embeddings(summary_writer, config)
#
#     # create session
#     sess = tf.Session()
#     # initialize variables
#     sg_init(sess)
#
#     # save tsne
#     saver.save(sess, save_dir + '/model-tsne')
#
#     # logging
#     tf.sg_info('Tsne saved at %s' % (save_dir + '/model-tsne'))
#
#     # close session
#     sess.close()
