import sugartensor as tf


__author__ = 'namju.kim@kakaocorp.com'


# set log level to debug
tf.sg_verbosity(10)

# batch size
batch_size = 128


# MNIST input tensor ( batch size should be adjusted for multiple GPUS )
data = tf.sg_data.Mnist(batch_size=batch_size * tf.sg_gpus())

# split inputs for each GPU tower
inputs = tf.split(data.train.image, tf.sg_gpus(), axis=0)
labels = tf.split(data.train.label, tf.sg_gpus(), axis=0)


# simple wrapping function with decorator for parallel training
@tf.sg_parallel
def get_loss(opt):

    # conv layers
    with tf.sg_context(name='convs', act='relu', bn=True):
        conv = (opt.input[opt.gpu_index]
                .sg_conv(dim=16, name='conv1')
                .sg_pool()
                .sg_conv(dim=32, name='conv2')
                .sg_pool()
                .sg_conv(dim=32, name='conv3')
                .sg_pool())

    # fc layers
    with tf.sg_context(name='fcs', act='relu', bn=True):
        logit = (conv
                 .sg_flatten()
                 .sg_dense(dim=256, name='fc1')
                 .sg_dense(dim=10, act='linear', bn=False, name='fc2'))

        # cross entropy loss with logit
        return logit.sg_ce(target=opt.target[opt.gpu_index])

# parallel training ( same as single GPU training )
tf.sg_train(loss=get_loss(input=inputs, target=labels), ep_size=data.train.num_batch,
            log_interval=10, save_dir='asset/train/conv_par')

