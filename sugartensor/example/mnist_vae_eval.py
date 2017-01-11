import sugartensor as tf
import matplotlib.pyplot as plt

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 100  # batch size
num_dim = 50      # latent dimension


#
# inputs
#

#
# Computational graph
#

# random gaussian seed
z = tf.random_normal((batch_size, num_dim))

# decoder network
with tf.sg_context(name='decoder', size=4, stride=2, act='relu'):
    gen = (z
           .sg_dense(dim=1024)
           .sg_dense(dim=7*7*128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(dim=64)
           .sg_upconv(dim=1, act='sigmoid')
           .sg_squeeze())

#
# draw samples
#

with tf.Session() as sess:

    tf.sg_init(sess)

    # restore parameters
    tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train/vae'), category='decoder')

    # run generator
    imgs = sess.run(gen)

    # plot result
    _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(imgs[i * 10 + j], 'gray')
            ax[i][j].set_axis_off()

    plt.savefig('asset/train/vae/sample.png', dpi=600)
    tf.sg_info('Sample image saved to "asset/train/vae/sample.png"')
    plt.close()
