import sugartensor as tf

# set log level to debug
tf.sg_verbosity(10)

# MNIST input tensor ( with QueueRunner )
data = tf.sg_data.Mnist()

# inputs
x = data.train.image
y = data.train.label

# create training graph
with tf.sg_context(act='relu', bn=True):
    logit = (x.sg_conv(dim=16).sg_pool()
             .sg_conv(dim=32).sg_pool()
             .sg_conv(dim=32).sg_pool()
             .sg_flatten()
             .sg_dense(dim=256)
             .sg_dense(dim=10, act='linear', bn=False))

# cross entropy loss with logit ( for training set )
loss = logit.sg_ce(target=y)

# accuracy evaluation ( for validation set )
acc = (logit.sg_reuse(input=data.valid.image).sg_softmax()
       .sg_accuracy(target=data.valid.label, name='val'))

# train
tf.sg_train(loss=loss, eval_metric=[acc], ep_size=data.train.num_batch, log_interval=10,
            save_dir='asset/train/conv')

