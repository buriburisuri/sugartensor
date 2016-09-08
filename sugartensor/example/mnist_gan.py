# -*- coding: utf-8 -*-
import sugartensor as tf

# set log level to debug
tf.sg_verbosity(10)

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=32)

#
# inputs
#

# input images
x = data.train.image

# generator labels ( all ones )
y = tf.ones(x.get_shape().as_list()[0], dtype=tf.sg_floatx)

# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])

#
# replay memory
#

rep_mem = tf.FIFOQueue(50000, tf.sg_floatx)

#
# create generator
#

# random uniform seed
z = tf.random_uniform((x.get_shape().as_list()[0], 100))

with tf.sg_context(name='generator', act='relu', bn=True):

    # generator network
    gen = (z.sg_dense(dim=200)
           .sg_dense(dim=400)
           .sg_dense(dim=784, act='sigmoid', bn=False)
           .sg_reshape(shape=(x.get_shape().as_list()[0], 28, 28, 1)))

# add image summary
tf.sg_summary_image(gen)

# add to replay memory
rep_mem_op = rep_mem.enqueue(gen)

#
# create discriminator
#

# create real + fake image input
xx = tf.concat(0, [x, rep_mem.dequeue()])

with tf.sg_context(name='discriminator', act='relu', bn=True):
    disc = (xx.sg_flatten()
            .sg_dense(dim=400).sg_dense(dim=200).sg_dense(dim=100)
            .sg_dense(dim=1, act='linear', bn=False).sg_squeeze())

#
# loss
#

# discriminator loss
loss_disc = disc.sg_bce(target=y_disc)
train_disc = tf.sg_optimize.MaxPropOptimizer(learning_rate=0.0001).minimize(loss_disc, global_step=tf.sg_global_step(),
                var_list=[t for t in tf.all_variables() if t.name.encode('utf8').startswith('discriminator')])

# generator loss
loss_gen = disc.sg_reuse(input=gen).sg_bce(target=y)
train_gen = tf.sg_optimize.MaxPropOptimizer(learning_rate=0.0001).minimize(loss_gen, global_step=tf.sg_global_step(),
                var_list=[t for t in tf.all_variables() if t.name.encode('utf8').startswith('generator')])


#
# training
#

# make directory if not exist
import os
from tqdm import tqdm

if not os.path.exists('asset/train/log'):
    os.makedirs('asset/train/log')
if not os.path.exists('asset/train/ckpt'):
    os.makedirs('asset/train/ckpt')


# session create
with tf.Session() as sess:

    # summary
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('asset/train/log', graph=sess.graph)

    # initialize
    sess.run(tf.group(tf.initialize_all_variables(),
                      tf.initialize_local_variables()))

    # initial replay memory
    for _ in range(100):
        sess.run(rep_mem_op)

    # train loop
    with tf.sg_queue_context():
        for i in tqdm(range(100000000),
                      desc='train', ncols=70, unit='b', leave=False):

            # do training
            sess.run(rep_mem_op)
            sess.run(train_disc)
            sess.run(train_gen)

            # logging ops
            if i % 1000 == 0:
                summary_writer.add_summary(sess.run(summary_op),
                                           global_step=tf.sg_global_step(as_tensor=False))
