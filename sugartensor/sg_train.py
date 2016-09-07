# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import os
import time
import sg_optimize
from tqdm import tqdm


__author__ = 'njkim@jamonglab.com'


def sg_train(**kwargs):
    opt = tf.sg_opt(kwargs)
    assert opt.loss is not None, 'loss is mandatory.'

    # default training options
    opt += tf.sg_opt(optim='MaxPropOptimizer', lr=0.001, beta1=0.9, beta2=0.99,
                     save_dir='asset/train',
                     max_ep=500, ep_size=100000000,
                     save_interval=600, log_interval=60,
                     early_stop=True, lr_reset=False,
                     eval_metric=[])

    # make directory if not exist
    if not os.path.exists(opt.save_dir + '/log'):
        os.makedirs(opt.save_dir + '/log')
    if not os.path.exists(opt.save_dir + '/ckpt'):
        os.makedirs(opt.save_dir + '/ckpt')

    # select optimizer
    if opt.optim == 'MaxPropOptimizer':
        optim = tf.sg_optimize.MaxPropOptimizer(learning_rate=tf.sg_learning_rate(), beta2=opt.beta2)
    elif opt.optim == 'AdaMaxOptimizer':
        optim = tf.sg_optimize.AdaMaxOptimizer(learning_rate=tf.sg_learning_rate(), beta1=opt.beta1, beta2=opt.beta2)

    # train op
    train_op = optim.minimize(opt.loss, global_step=tf.sg_global_step())

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

        # summary writer
        summary_writer = tf.train.SummaryWriter(opt.save_dir + '/log', graph=sess.graph)

        # saver
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        last_file = tf.train.latest_checkpoint(opt.save_dir + '/ckpt')
        if last_file:
            saver.restore(sess, last_file)
            start_ep = int(last_file.split('-')[1])
            start_step = int(last_file.split('-')[2])
            # early stopping checking
            if tf.sg_learning_rate(as_tensor=False) < 5e-6:
                tf.sg_info('Early stopped at epoch[%d]-step[%d].' % (start_ep, start_step))
                return
        else:
            start_ep = 1
            start_step = tf.sg_global_step(as_tensor=False)

        # set learning rate
        if start_step == 0 or opt.lr_reset:
            sess.run(tf.sg_learning_rate().assign(opt.lr))

        # logging
        tf.sg_info('Training started from epoch[%03d]-step[%d].' % (start_ep, start_step))

        # start data queue runner
        with tf.sg_queue_context(sess):

            # set mode to train
            tf.sg_phase(phase='train')

            #
            # loop all epochs
            #

            # loss history for learning rate decay
            loss_prev, loss, early_stopped = None, None, False

            # time stamp
            last_saved = last_logged = time.time()

            for ep in range(1, opt.max_ep + 1):

                # loop each epoch
                for _ in tqdm(range(opt.ep_size),
                              desc='train', ncols=70, unit='b', leave=False):

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

                if early_stopped:
                    break

                # log epoch information
                tf.sg_info('\tEpoch[%03d:lr=%7.5f:gs=%d] - loss = %8.6f' %
                           (ep, tf.sg_learning_rate(as_tensor=False), tf.sg_global_step(as_tensor=False), loss))

            # save last epoch
            saver.save(sess, opt.save_dir + '/ckpt/model-%03d' % ep,
                       write_meta_graph=False,
                       global_step=tf.sg_global_step(as_tensor=False))

        # logging
        tf.sg_info('Training finished at epoch[%d]-step[%d].' % (ep, tf.sg_global_step(as_tensor=False)))
