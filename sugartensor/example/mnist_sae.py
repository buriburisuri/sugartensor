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
    z = (x
         .sg_conv(dim=64)
         .sg_conv(dim=128)
         .sg_flatten()
         .sg_dense(dim=1024)
         .sg_dense(dim=num_dim, act='linear'))

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
loss = xx.sg_mse(target=x)


# do training
tf.sg_train(loss=loss, log_interval=10, ep_size=data.train.num_batch, save_dir='asset/train/sae')

