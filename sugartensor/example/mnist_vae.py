import sugartensor as tf

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 32   # batch size
num_dim = 50      # latent dimension


#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist(batch_size=32)

# input images
x = data.train.image

#
# Computational graph
#

# encoder network
with tf.sg_context(name='encoder', size=4, stride=2, act='relu'):
    mu = (x
          .sg_conv(dim=64)
          .sg_conv(dim=128)
          .sg_flatten()
          .sg_dense(dim=1024)
          .sg_dense(dim=num_dim, act='linear'))

# re-parameterization trick with random gaussian
z = mu + tf.random_normal(mu.get_shape())

# decoder network
with tf.sg_context(name='decoder', size=4, stride=2, act='relu'):
    xx = (z
          .sg_dense(dim=1024)
          .sg_dense(dim=7*7*128)
          .sg_reshape(shape=(-1, 7, 7, 128))
          .sg_upconv(dim=64)
          .sg_upconv(dim=1, act='sigmoid'))

# add image summary
tf.sg_summary_image(x, name='origin')
tf.sg_summary_image(xx, name='recon')

# loss
loss_recon = xx.sg_mse(target=x, name='recon').sg_mean(axis=[1, 2, 3])
loss_kld = tf.square(mu).sg_sum(axis=1) / (28 * 28)
tf.sg_summary_loss(loss_kld, name='kld')
loss = loss_recon + loss_kld * 0.5


# do training
tf.sg_train(loss=loss, log_interval=10, ep_size=data.train.num_batch, max_ep=30, early_stop=False,
            save_dir='asset/train/vae')

