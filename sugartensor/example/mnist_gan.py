# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np

# set log level to debug
tf.sg_verbosity(10)

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=32)

# input images
x = data.train.image

# generator labels ( all ones )
y = tf.ones(x.get_shape().as_list()[0], dtype=tf.sg_floatx)

# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])

#
# create generator
#

# random uniform seed
z = tf.random_uniform((x.get_shape().as_list()[0], 100))

with tf.sg_context(name='generator', stride=2, act='relu', bn=True):

    # generator network
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(size=4, dim=64)
           .sg_upconv(size=4, dim=1, act='sigmoid', bn=False))

# add image summary
tf.sg_summary_image(gen)

#
# create discriminator
#

# create real + fake image input
xx = tf.concat(0, [x, gen])

with tf.sg_context(name='discriminator', stride=2, act='leaky_relu'):
    disc = (xx.sg_conv(size=4, dim=64)
            .sg_conv(size=4, dim=128)
            .sg_flatten()
            .sg_dense(dim=1024)
            .sg_dense(dim=1, act='linear')
            .sg_squeeze())

#
# loss & train ops
#

loss_disc = disc.sg_bce(target=y_disc)  # discriminator loss
loss_gen = disc.sg_reuse(input=gen).sg_bce(target=y)  # generator loss


train_disc = tf.sg_optim(loss_disc, lr=0.0001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_gen, lr=0.001, category='generator')  # generator train ops


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_disc, train_disc])[0]  # training discriminator
    l_gen = sess.run([loss_gen, train_gen])[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
alt_train(log_interval=10, ep_size=data.train.num_batch, early_stop=False, save_dir='asset/train/gan')

