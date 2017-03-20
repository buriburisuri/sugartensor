import sugartensor as tf


__author__ = 'namju.kim@kakaocorp.com'


# set log level to debug
tf.sg_verbosity(10)

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist()

# inputs
x = data.train.image
y = data.train.label

# create training graph
with tf.sg_context(regularizer='l1'):
    logit = (x.sg_flatten()
             .sg_dense(dim=400, act='relu', bn=True)
             .sg_dense(dim=200, act='relu', bn=True)
             .sg_dense(dim=10))

# cross entropy loss with logit ( for training set ) + L1 regularizer loss ( = sparsity regularizer )
loss = logit.sg_ce(target=y) + tf.sg_regularizer_loss(scale=1.)

# train
tf.sg_train(loss=loss, ep_size=data.train.num_batch, log_interval=10, save_dir='asset/train/l1_regul')

