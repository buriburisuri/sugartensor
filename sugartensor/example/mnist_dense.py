# -*- coding: utf-8 -*-
import sugartensor as tf


__author__ = 'njkim@jamonglab.com'


if __name__ == '__main__':

    mnist = tf.sg_data.Mnist()

    x = mnist.train.image.sg_float()
    y = mnist.train.label

    tf.sg_verbosity(10)

    with tf.sg_context(act='relu', bn=True):
        logit = (x.sg_conv(dim=16).sg_pool().sg_conv(dim=16).sg_pool()  # encoding
                 .sg_upconv(dim=16).sg_upconv(dim=16).sg_conv(size=1, dim=1, act='sigmoid', bn=False))  # decoding

    loss = logit.sg_mse(target=x/255)

    tf.sg_summary_image(tf.cast(logit*255, tf.uint8))
    # loss = logit.ce(target=y)

    # # accuracy
    # acc = logit.accuracy(target=y)

    optim = tf.sg_optimize.MaxPropOptimizer().minimize(loss, global_step=tf.sg_global_step())

    summary = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/data/mnist/log')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tf.train.start_queue_runners(sess)
        tf.sg_phase(phase='train')
        while True:
            # print sess.run(logit, {x: img, y: label}).shape
            sess.run(optim)
        tf.sg_phase(phase='infer')
        stat = sess.run(summary)
        writer.add_summary(stat, global_step=tf.sg_global_step(as_tensor=False))



