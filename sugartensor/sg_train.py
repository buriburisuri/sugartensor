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
    assert opt.save_dir, 'save_dir is mandatory.'

    # default training options
    opt += tf.sg_opt(optim='MaxPropOptimizer', lr=0.001, beta1=0.9, beta2=0.99,
                     early_stop=True,   # learning rate decay
                     max_ep=500, total_batch=100000000,
                     save_interval=600, log_interval=60)

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

    # summary op
    summary_op = tf.merge_all_summaries()

    # run as default session
    with tf.Session() as sess:

        # initialize variables
        sess.run(tf.initialize_all_variables())

        # summary writer
        summary_writer = tf.train.SummaryWriter(opt.save_dir + '/log', graph=sess.graph)

        # saver
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        last_file = tf.train.latest_checkpoint(opt.save_dir + '/ckpt')
        if last_file:
            saver.restore(sess, last_file)
            start_ep = int(last_file.split('-')[1])
            start_step = int(last_file.split('-')[2])
            # early stopping chekcing
            if tf.sg_learning_rate(as_tensor=False) < 5e-6:
                tf.sg_info('Early stopped at epoch[%d]-step[%d].' % (start_ep, start_step))
                return
        else:
            start_ep = 1
            start_step = tf.sg_global_step(as_tensor=False)

        # start data queue runner
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # logging
        tf.sg_info('Training started from epoch[%03d]-step[%d].' % (start_ep, start_step))

        # set mode to train
        tf.sg_phase(phase='train')

        # set learning rate
        sess.run(tf.sg_learning_rate().assign(opt.lr))

        #
        # loop all epochs
        #

        # loss_history for learning rate decay
        loss_prev, early_stopped = None, False

        # time stamp
        last_saved = last_logged = time.time()

        for ep in range(1, opt.max_ep + 1):

            # loop each epoch
            loss = None
            for _ in tqdm(range(opt.total_batch),
                          desc='train', ncols=70, unit='b', leave=False):

                # run training steps
                loss_prev = loss
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

            if early_stopped:
                tf.sg_info('\tTraining early stopped at epoch[%d]-step[%d].' % (ep, tf.sg_global_step(as_tensor=False)))
                # save last epoch
                saver.save(sess, opt.save_dir + '/ckpt/model-%03d' % ep,
                           write_meta_graph=False,
                           global_step=tf.sg_global_step(as_tensor=False))
                break

            # log epoch information
            tf.sg_info('\tEpoch[%03d:lr=%7.5f] - loss = %8.6f' % (ep, tf.sg_learning_rate(as_tensor=False), loss))

        # weight data queue runner
        coord.request_stop()
        coord.join(threads)

        tf.sg_info('Training finished at epoch[%d]-step[%d].' % (ep, tf.sg_global_step(as_tensor=False)))
