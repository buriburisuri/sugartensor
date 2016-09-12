# -*- coding: utf-8 -*-
import sugartensor as tf
import numpy as np
import matplotlib.pyplot as plt

# set log level to debug
tf.sg_verbosity(10)

#
# inputs
#

batch_size = 100
num_category = 10

# target_number
target_num = tf.placeholder(dtype=tf.sg_intx, shape=batch_size)
# target continuous variable # 1
target_cval_1 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)
# target continuous variable # 2
target_cval_2 = tf.placeholder(dtype=tf.sg_floatx, shape=batch_size)

# category variables
z = (tf.ones(batch_size, dtype=tf.sg_intx) * target_num).sg_one_hot(depth=num_category)

# continuous variables
z = z.sg_concat(target=[target_cval_1.sg_expand_dims(), target_cval_2.sg_expand_dims()])

# random seed = categorical variable + continuous variable + random uniform
z = z.sg_concat(target=tf.random_uniform((batch_size, 38)))

#
# create generator
#

# generator network
with tf.sg_context(name='generator', stride=2, act='relu', bn=True):
    gen = (z.sg_dense(dim=1024)
           .sg_dense(dim=7 * 7 * 128)
           .sg_reshape(shape=(-1, 7, 7, 128))
           .sg_upconv(size=4, dim=64)
           .sg_upconv(size=4, dim=1, act='sigmoid', bn=False).sg_squeeze())


#
# run generator
#
def run_generator(num, x1, x2, fig_name='sample.png'):
    with tf.Session() as sess:
        tf.sg_init(sess)
        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train/infogan/ckpt'))

        # run generator
        imgs = sess.run(gen, {target_num: num,
                              target_cval_1: x1,
                              target_cval_2: x2})

        # plot result
        _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
        for i in range(10):
            for j in range(10):
                ax[i][j].imshow(imgs[i * 10 + j], 'gray')
                ax[i][j].set_axis_off()
        plt.savefig('asset/train/infogan/' + fig_name, dpi=600)
        tf.sg_info('Sample image saved to "asset/train/infogan/%s"' % fig_name)
        plt.close()


#
# draw sample by categorical division
#

run_generator(np.arange(10).repeat(10), np.ones(100) * 0.5, np.ones(100) * 0.5)

#
# draw sample by continuous division
#

for i in range(10):
    run_generator(np.ones(100) * i,
                  np.linspace(0, 1, 10).repeat(10),
                  np.expand_dims(np.linspace(0, 1, 10), axis=1).repeat(10, axis=1).T.flatten(),
                  fig_name='sample%d.png' % i)
