# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import os
import time
import sg_optimize
from tqdm import tqdm
from contextlib import contextmanager


__author__ = 'njkim@jamonglab.com'


def sg_optim(loss, **kwargs):
    opt = tf.sg_opt(kwargs)

    # default training options
    opt += tf.sg_opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, category='')

    # select optimizer
    if opt.optim == 'MaxProp':
        optim = tf.sg_optimize.MaxPropOptimizer(learning_rate=opt.lr, beta2=opt.beta2)
    elif opt.optim == 'AdaMax':
        optim = tf.sg_optimize.AdaMaxOptimizer(learning_rate=opt.lr, beta1=opt.beta1, beta2=opt.beta2)

    # get trainable variables
    var_list = [t for t in tf.trainable_variables() if t.name.encode('utf8').startswith(opt.category)]

    # calc gradient
    gradient = optim.compute_gradients(loss, var_list=var_list)

    # add summary
    for v, g in zip(var_list, gradient):
        tf.sg_summary_gradient(v, g)

    # return train op
    return optim.apply_gradients(gradient, global_step=tf.sg_global_step())


@contextmanager
def sg_train_context(**kwargs):
    opt = tf.sg_opt(kwargs)
    # default training options
    opt += tf.sg_opt(save_dir='asset/train',
                     max_ep=1000, ep_size=100000,
                     save_interval=600, log_interval=60,
                     eval=[])

    # make directory if not exist
    if not os.path.exists(opt.save_dir + '/log'):
        os.makedirs(opt.save_dir + '/log')
    if not os.path.exists(opt.save_dir + '/ckpt'):
        os.makedirs(opt.save_dir + '/ckpt')

    # find last checkpoint
    last_file = tf.train.latest_checkpoint(opt.save_dir + '/ckpt')
    if last_file:
        start_ep = int(last_file.split('-')[1])
        start_step = int(last_file.split('-')[2])
    else:
        start_ep = 1
        start_step = 0

    # checkpoint saver
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    # summary writer
    summary_writer = tf.train.SummaryWriter(opt.save_dir + '/log', graph=tf.get_default_graph())

    # add evaluation metric summary
    for m in opt.eval_metric:
        tf.sg_summary_metric(m)

    # summary op
    summary_op = tf.merge_all_summaries()

    # run as default session
    with tf.Session() as sess:

        # initialize variables
        sess.run(tf.group(tf.initialize_all_variables(),
                          tf.initialize_local_variables()))

        # start data queue runner
        with tf.sg_queue_context(sess):

            # restore last checkpoint
            if last_file:
                saver.restore(sess, last_file)

            # logging
            tf.sg_info('Training started from epoch[%03d]-step[%d].' % (start_ep, start_step))

            # time stamp
            last_saved = last_logged = time.time()

            # loop epoch
            for ep in range(1, opt.max_ep + 1):

                # simple progress bar
                for _ in tqdm(range(opt.ep_size),
                              desc='train', ncols=70, unit='b', leave=False):

                    # set mode to train
                    tf.sg_phase(phase='train')

                    # wait train code
                    yield sess

                    # save parameters
                    if time.time() - last_saved > opt.save_interval:
                        saver.save(sess, opt.save_dir + '/ckpt/model-%03d' % ep,
                                   write_meta_graph=False,
                                   global_step=tf.sg_global_step(as_tensor=False))
                        last_saved = time.time()

                    # set mode to infer
                    tf.sg_phase(phase='infer')

                    # logging summary
                    if time.time() - last_logged > opt.log_interval:

                        # run evaluation operations
                        if len(opt.eval_metric) > 0:
                            sess.run(opt.eval_metric)

                        # logging ops
                        summary_writer.add_summary(sess.run(summary_op),
                                                   global_step=tf.sg_global_step(as_tensor=False))
                        last_logged = time.time()

                # log epoch information
                tf.sg_info('\tEpoch[%03d:gs=%d] finished.' % (ep, tf.sg_global_step(as_tensor=False)))

            # save last epoch
            saver.save(sess, opt.save_dir + '/ckpt/model-%03d' % ep,
                       write_meta_graph=False,
                       global_step=tf.sg_global_step(as_tensor=False))

        # logging
        tf.sg_info('Training finished at epoch[%d]-step[%d].' % (ep, tf.sg_global_step(as_tensor=False)))


def sg_train(**kwargs):
    opt = tf.sg_opt(kwargs)
    assert opt.loss is not None, 'loss is mandatory.'

    # default training options
    opt += tf.sg_opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, category='',
                     save_dir='asset/train',
                     max_ep=1000, ep_size=100000,
                     save_interval=600, log_interval=60,
                     early_stop=True, lr_reset=False,
                     eval_metric=[])

    # get optimizer
    train_op = sg_optim(opt.loss, optim=opt.optim, lr=tf.sg_learning_rate(),
                        beta1=opt.beta1, beta2=opt.beta2, category=opt.category)

    # make directory if not exist
    if not os.path.exists(opt.save_dir + '/log'):
        os.makedirs(opt.save_dir + '/log')
    if not os.path.exists(opt.save_dir + '/ckpt'):
        os.makedirs(opt.save_dir + '/ckpt')

    # find last checkpoint
    last_file = tf.train.latest_checkpoint(opt.save_dir + '/ckpt')
    if last_file:
        start_ep = int(last_file.split('-')[1])
        start_step = int(last_file.split('-')[2])
    else:
        start_ep = 1
        start_step = 0

    # checkpoint saver
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

    # summary writer
    summary_writer = tf.train.SummaryWriter(opt.save_dir + '/log', graph=tf.get_default_graph())

    # add evaluation metric summary
    for m in opt.eval_metric:
        tf.sg_summary_metric(m)

    # summary op
    summary_op = tf.merge_all_summaries()

    # run as default session
    with tf.Session() as sess:

        # initialize variables
        sess.run(tf.group(tf.initialize_all_variables(),
                          tf.initialize_local_variables()))

        # start data queue runner
        with tf.sg_queue_context(sess):

            # restore last checkpoint
            if last_file:
                saver.restore(sess, last_file)

            # set learning rate
            if start_ep == 1 or opt.lr_reset:
                sess.run(tf.sg_learning_rate().assign(opt.lr))

            # logging
            tf.sg_info('Training started from epoch[%03d]-step[%d].' % (start_ep, start_step))

            # loss history for learning rate decay
            loss_prev, loss, early_stopped = None, None, False

            # time stamp
            last_saved = last_logged = time.time()

            for ep in range(1, opt.max_ep + 1):

                # loop each epoch
                for _ in tqdm(range(opt.ep_size),
                              desc='train', ncols=70, unit='b', leave=False):

                    # set mode to train
                    tf.sg_phase(phase='train')

                    # run training steps
                    if loss is None:
                        loss = np.mean(sess.run([train_op, opt.loss])[1])
                    else:
                        loss = loss * 0.9 + np.mean(sess.run([train_op, opt.loss])[1]) * 0.1

                    # save parameters
                    if time.time() - last_saved > opt.save_interval:
                        saver.save(sess, opt.save_dir + '/ckpt/model-%03d' % ep,
                                   write_meta_graph=False,
                                   global_step=tf.sg_global_step(as_tensor=False))
                        last_saved = time.time()

                    # set mode to infer
                    tf.sg_phase(phase='infer')

                    # logging summary
                    if time.time() - last_logged > opt.log_interval:

                        # run evaluation operations
                        if len(opt.eval_metric) > 0:
                            sess.run(opt.eval_metric)

                        # logging ops
                        summary_writer.add_summary(sess.run(summary_op),
                                                   global_step=tf.sg_global_step(as_tensor=False))
                        last_logged = time.time()

                        # learning rate decay
                        if opt.early_stop and loss_prev:
                            # if loss stalling
                            if loss >= 0.95 * loss_prev:
                                # early stopping
                                current_lr = tf.sg_learning_rate(as_tensor=False)
                                if current_lr < 5e-6:
                                    early_stopped = True
                                    break
                                else:
                                    # decrease learning rate by half
                                    sess.run(tf.sg_learning_rate().assign(current_lr / 2.))

                        # update loss history
                        loss_prev = loss

                # log epoch information
                tf.sg_info('\tEpoch[%03d:lr=%7.5f:gs=%d] - loss = %8.6f' %
                           (ep, tf.sg_learning_rate(as_tensor=False), tf.sg_global_step(as_tensor=False), loss))

                if early_stopped:
                    tf.sg_info('\tEarly stopped.')
                    break

            # save last epoch
            saver.save(sess, opt.save_dir + '/ckpt/model-%03d' % ep,
                       write_meta_graph=False,
                       global_step=tf.sg_global_step(as_tensor=False))

        # logging
        tf.sg_info('Training finished at epoch[%d]-step[%d].' % (ep, tf.sg_global_step(as_tensor=False)))
