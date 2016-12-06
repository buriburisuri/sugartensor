from __future__ import absolute_import, print_function, unicode_literals
import sugartensor as tf
# noinspection PyPackageRequirements
import numpy as np
import os
import time
from tqdm import tqdm
from functools import wraps


__author__ = 'buriburisuri@gmail.com'


# global learning rate
_learning_rate = tf.Variable(0.001, dtype=tf.sg_floatx, name='learning_rate', trainable=False)


def sg_train(**kwargs):
    r""" Train function.

    Args:
        **kwargs:
            loss: loss to train

            optim: optimizer name ( default: 'MaxProp'.  other : 'AdaMax', 'Adam', 'sgd')
            lr: learning rate ( default : 0.001)
            beta1: optimizer beta1 ( default :0.9 )
            beta2: optimizer beta2 ( default :0.99 )

            eval_metric: evaluation metric tensor list ( default : [] )
            early_stop: automatic learning rate decay and stop. ( default : True)
            lr_reset: whether reset learning rate when restarting training ( default : False)

            save_dir: checkpoint file save path ( default : 'asset/train')
            max_ep: max epoch number to train ( default : 1000 )
            ep_size: total batch number in a epoch ( default : 100000)

            save_interval: checkpoint saving interval ( default : 600 seconds )
            log_interval: logging interval ( default : 60 seconds )
            max_keep: max checkpoint files to keep ( default : 5 )
            keep_interval: checkpoint file keep interval ( default : 1 hour )

            tqdm: whether show tqdm progress bar or not ( default : True)
            console_log: print loss on the console and do not save report file ( default : False )
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
        return sess.run([opt.loss, train_op])[0]

    # run train function
    train_func(**opt)


def sg_init(sess):
    r""" Initialize session variables
    Args:
        sess: session to init
    """
    # initialize variables
    sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))


def sg_print(tensor_list):
    r""" Simple tensor printing function for debugging
    Args:
        tensor_list: tensors to print
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
                print (r, r.shape, r.dtype)
    return res


def sg_optim(loss, **kwargs):
    r""" Create optimizer

    Args:
        loss: loss to optimize
        **kwargs:
            optim: optimizer name ( default: 'MaxProp'.  other : 'AdaMax', 'Adam', 'sgd')
            lr: learning rate ( default : 0.001)
            beta1: optimizer beta1 ( default :0.9 )
            beta2: optimizer beta2 ( default :0.99 )
    Returns:
        optimizer
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
    var_list = [t for t in tf.trainable_variables() if t.name.startswith(opt.category)]

    # calc gradient
    gradient = optim.compute_gradients(loss, var_list=var_list)

    # add summary
    for v, g in zip(var_list, gradient):
        tf.sg_summary_gradient(v, g)

    # gradient update op
    return optim.apply_gradients(gradient, global_step=tf.sg_global_step())


def sg_train_func(func):
    r""" Decorate function as sg_train_func.
    Args:
        func: function to decorate
    """
    @wraps(func)
    def wrapper(**kwargs):
        r""" Manages arguments of `tf.sg_opt`.

        Args:
            **kwargs:
                lr: learning rate ( default : 0.001)
                save_dir: checkpoint file save path ( default : 'asset/train')
                max_ep: max epoch number to train ( default : 1000 )
                ep_size: total batch number in a epoch ( default : 100000)
                save_interval: checkpoint saving interval ( default : 600 seconds )
                log_interval: logging interval ( default : 60 seconds )
                early_stop: automatic learning rate decay and stop. ( default : True)
                lr_reset: whether reset learning rate when restarting training ( default : False)
                eval_metric: evaluation metric tensor list ( default : [] )
                max_keep: max checkpoint files to keep ( default : 5 )
                keep_interval: checkpoint file keep interval ( default : 1 hour )
                tqdm: whether show tqdm progress bar or not ( default : True)
                console_log: print loss on the console and do not save report file ( default : False )
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
        saver = tf.train.Saver(max_to_keep=opt.max_keep,
                               keep_checkpoint_every_n_hours=opt.keep_interval)

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
