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

with tf.sg_context(name='generator', act='relu'):

    # generator network
    gen = (z.sg_dense(dim=200)
           .sg_dense(dim=400)
           .sg_dense(dim=784, act='sigmoid')
           .sg_reshape(shape=(-1, 28, 28, 1)))

# add image summary
tf.sg_summary_image(gen)

# add to replay memory
rep_mem_op = rep_mem.enqueue(gen)

#
# create discriminator
#

# create real + fake image input
xx = tf.concat(0, [x, rep_mem.dequeue()])

with tf.sg_context(name='discriminator', act='relu'):
    disc = (xx.sg_flatten()
            .sg_dense(dim=400).sg_dense(dim=200).sg_dense(dim=100)
            .sg_dense(dim=1, act='linear').sg_squeeze())

#
# loss
#

# discriminator loss
loss_disc = disc.sg_bce(target=y_disc)
train_disc = tf.sg_optim(loss_disc, lr=0.0001, category='discriminator')

# generator loss
loss_gen = disc.sg_reuse(input=gen).sg_bce(target=y)
train_gen = tf.sg_optim(loss_gen, lr=0.0001, category='generator')


#
# training
#

@tf.sg_train_func
def train(sess):
    # alternate training
    sess.run(train_disc)  # training discriminator
    sess.run([train_gen, rep_mem_op])  # training generator

# session create
with tf.Session() as sess:

    # init session
    tf.sg_init(sess)

    # init replay memory
    for _ in range(2000):
        sess.run(rep_mem_op)

    # do training
    train(sess=sess, log_interval=10)

