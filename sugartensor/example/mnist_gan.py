import sugartensor as tf
import numpy as np

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
rand_dim = 50     # total random latent dimension


#
# create generator & discriminator function
#

def generator(tensor):
    # reuse flag
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    with tf.sg_context(name='generator', size=4, stride=2, act='leaky_relu', bn=True, reuse=reuse):

        # generator network
        res = (tensor
               .sg_dense(dim=1024, name='fc1')
               .sg_dense(dim=7*7*128, name='fc2')
               .sg_reshape(shape=(-1, 7, 7, 128))
               .sg_upconv(dim=64, name='conv1')
               .sg_upconv(dim=1, act='sigmoid', bn=False, name='conv2'))

        return res


def discriminator(tensor):
    # reuse flag
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    with tf.sg_context(name='discriminator', size=4, stride=2, act='leaky_relu', bn=True, reuse=reuse):
        res = (tensor
               .sg_conv(dim=64, name='conv1')
               .sg_conv(dim=128, name='conv2')
               .sg_flatten()
               .sg_dense(dim=1024, name='fc1')
               .sg_dense(dim=1, act='linear', bn=False, name='fc2')
               .sg_squeeze())
        return res


#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=32)

# input images
x = data.train.image

# labels for discriminator
y_real = tf.ones(batch_size)
y_fake = tf.zeros(batch_size)

# random gaussian seed
z = tf.random_normal((data.batch_size, rand_dim))


#
# Computational graph
#

# generator
gen = generator(z)

# add image summary
tf.sg_summary_image(x, name='real')
tf.sg_summary_image(gen, name='fake')

# discriminator
disc_real = discriminator(x)
disc_fake = discriminator(gen)


#
# loss & train ops
#

# discriminator loss
loss_d_r = disc_real.sg_bce(target=y_real, name='disc_real')
loss_d_f = disc_fake.sg_bce(target=y_fake, name='disc_fake')
loss_d = (loss_d_r + loss_d_f) / 2


# generator loss
loss_g = disc_fake.sg_bce(target=y_real, name='gen')

# train ops
train_disc = tf.sg_optim(loss_d, lr=0.0001, category='discriminator')  # discriminator train ops
train_gen = tf.sg_optim(loss_g, lr=0.001, category='generator')  # generator train ops


#
# training
#

# def alternate training func
@tf.sg_train_func
def alt_train(sess, opt):
    l_disc = sess.run([loss_d] + train_disc)[0]  # training discriminator
    l_gen = sess.run([loss_g] + train_gen)[0]  # training generator
    return np.mean(l_disc) + np.mean(l_gen)

# do training
alt_train(log_interval=10, max_ep=30, ep_size=data.train.num_batch, early_stop=False,
          save_dir='asset/train/gan')

