import sugartensor as tf
import matplotlib.pyplot as plt

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 100  # batch size
rand_dim = 50     # total random latent dimension


#
# create generator function
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


# random normal seed
z = tf.random_normal((batch_size, rand_dim))

# generator
gen = generator(z).sg_squeeze()


#
# draw samples
#

with tf.Session() as sess:

    tf.sg_init(sess)

    # restore parameters
    tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train/gan'), category='generator')

    # run generator
    imgs = sess.run(gen)

    # plot result
    _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
    for i in range(10):
        for j in range(10):
            ax[i][j].imshow(imgs[i * 10 + j], 'gray')
            ax[i][j].set_axis_off()
    plt.savefig('asset/train/gan/sample.png', dpi=600)
    tf.sg_info('Sample image saved to "asset/train/gan/sample.png"')
    plt.close()
