# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import os
import time
from tqdm import tqdm
from functools import wraps


__author__ = 'buriburisuri@gmail.com'


# global learning rate
_learning_rate = tf.Variable(0.001, dtype=tf.sg_floatx, name='learning_rate', trainable=False)


def sg_train(**kwargs):
    opt = tf.sg_opt(kwargs)
    assert opt.loss is not None, 'loss is mandatory.'

    # default training options
    opt += tf.sg_opt(optim='MaxProp', lr=0.001, beta1=0.9, beta2=0.99, category='')

    # get optimizer
    train_op = sg_optim(opt.loss, optim=opt.optim, lr=_learning_rate,
                        beta1=opt.beta1, beta2=opt.beta2, category=opt.category)

    # define train function
    @sg_train_func
    def train_func(sess, arg):
        return sess.run([opt.loss, train_op])[0]

    # run train function
    train_func(**opt)


def sg_init(sess):
    # initialize variables
    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_local_variables()))


def sg_print(tensor):
    with tf.Session() as sess:
        sg_init(sess)
        with tf.sg_queue_context():
            res = sess.run(tensor)
            print res, res.shape, res.dtype
    return res


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

    # gradient update op
    return optim.apply_gradients(gradient, global_step=tf.sg_global_step())


def sg_train_func(func):
    @wraps(func)
    def wrapper(**kwargs):
        opt = tf.sg_opt(kwargs)

        # default training options
        opt += tf.sg_opt(lr=0.001,
                         save_dir='asset/train',
                         max_ep=1000, ep_size=100000,
                         save_interval=600, log_interval=60,
                         early_stop=True, lr_reset=False,
                         eval_metric=[])

        # make directory if not exist
        if not os.path.exists(opt.save_dir + '/log'):
            os.makedirs(opt.save_dir + '/log')
        if not os.path.exists(opt.save_dir + '/ckpt'):
            os.makedirs(opt.save_dir + '/ckpt')

        # find last checkpoint
        last_file = tf.train.latest_checkpoint(opt.save_dir + '/ckpt')
        if last_file:
            ep = start_ep = int(last_file.split('-')[1]) + 1
            start_step = int(last_file.split('-')[2])
        else:
            ep = start_ep = 1
            start_step = 0

        # checkpoint saver
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)

        # summary writer
        summary_writer = tf.train.SummaryWriter(opt.save_dir + '/log', graph=tf.get_default_graph())

        # add learning rate summary
        with tf.name_scope('summary'):
            tf.scalar_summary('60. learning_rate/learning_rate', _learning_rate)

        # add evaluation metric summary
        for m in opt.eval_metric:
            tf.sg_summary_metric(m)

        # summary op
        summary_op = tf.merge_all_summaries()

        # create session
        if opt.sess:
            sess = opt.sess
        else:
            sess = tf.Session()
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

                    # batch loop
                    for _ in tqdm(range(opt.ep_size),
                                  desc='train', ncols=70, unit='b', leave=False):

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
                            saver.save(sess, opt.save_dir + '/ckpt/model-%03d' % ep,
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
                    tf.sg_info('\tEpoch[%03d:lr=%7.5f:gs=%d] - loss = %s' %
                               (ep, sess.run(_learning_rate), sess.run(tf.sg_global_step()),
                                ('NA' if loss is None else '%8.6f' % loss)))

                    if early_stopped:
                        tf.sg_info('\tEarly stopped ( no loss progress ).')
                        break
        finally:
            # save last epoch
            saver.save(sess, opt.save_dir + '/ckpt/model-%03d' % ep,
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
