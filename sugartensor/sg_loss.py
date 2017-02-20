from __future__ import absolute_import
import sugartensor as tf

__author__ = 'buriburisuri@gmail.com'


@tf.sg_sugar_func
def sg_ce(tensor, opt):
    r"""Returns softmax cross entropy loss between `tensor` and `target`.
    
    Args:
      tensor: A `Tensor`. Logits. Unscaled log probabilities.
      opt:
        target: A `Tensor` with the same length in the first dimension as the `tensor`. Labels. 
        one_hot: Boolean. Whether to treat the labels as one-hot encoding. Default is False.
        mask: Boolean. If True, zeros in the target will be excluded from the calculation.
        name: A `string`. A name to display in the tensor board web UI.
      
    Returns:
      A 1-D `Tensor` with the same shape as `tensor`. 
    
    For example, 
    
    ```
    tensor = [[[2, -1, 3], [3, 1, -2]]]
    target = [[2, 1]]
    tensor.sg_ce(target=target) => [[ 0.32656264  2.13284516]]
    ```
    
    For example,
    
    ```
    tensor = [[2, -1, 3], [3, 1, -2]]
    target = [[0, 0, 1], [1, 0, 0]]
    tensor.sg_ce(target=target, one_hot=True) => [ 0.32656264  0.13284527]
    ```
    """
    opt += tf.sg_opt(one_hot=False)
    assert opt.target is not None, 'target is mandatory.'

    if opt.one_hot:
        out = tf.identity(tf.nn.softmax_cross_entropy_with_logits(labels=opt.target, logits=tensor), 'ce')
    else:
        out = tf.identity(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=opt.target, logits=tensor), 'ce')

    # masking loss
    if opt.mask:
        out *= tf.not_equal(opt.target, tf.zeros_like(opt.target)).sg_float()

    # add summary
    tf.sg_summary_loss(out, name=opt.name)

    return out


@tf.sg_sugar_func
def sg_bce(tensor, opt):
    r"""Returns sigmoid cross entropy loss between `tensor` and `target`.
    
    Args:
      tensor: A `Tensor`. Logits. Unscaled log probabilities.
      opt:
        target: A `Tensor` with the same shape and dtype as `tensor`. Labels.
        name: A `string`. A name to display in the tensor board web UI.
      
    Returns:
      A `Tensor` of the same shape as `tensor`
    
    For example, 
    
    ```
    tensor = [[2, -1, 3], [3, 1, -2]]
    target = [[0, 1, 1], [1, 1, 0]]
    tensor.sg_bce(target=target) =>     [[ 2.12692809  1.31326163  0.04858733]
                                         [ 0.04858733  0.31326166  0.12692805]]
    ```
    """
    
    assert opt.target is not None, 'target is mandatory.'

    out = tf.identity(tf.nn.sigmoid_cross_entropy_with_logits(labels=opt.target, logits=tensor), 'bce')

    # add summary
    tf.sg_summary_loss(out, name=opt.name)

    return out


@tf.sg_sugar_func
def sg_mse(tensor, opt):
    r"""Returns squared error between `tensor` and `target`.
    
    Args:
      tensor: A `Tensor`.
      opt:
        target: A `Tensor` with the same shape and dtype as `tensor`.
        name: A `string`. A name to display in the tensor board web UI.
       
    Returns:
      A `Tensor` of the same shape and dtype as `tensor` 
    
    For example,
    
    ```
    tensor = [[34, 11, 40], [13, 30, 42]]
    target = [[34, 10, 41], [14, 31, 40]]
    tensor.sg_mse(target=target) => [[ 0.  1.  1.]
                                     [ 1.  1.  4.]]
    ```
    """
    assert opt.target is not None, 'target is mandatory.'

    # squared error
    out = tf.identity(tf.square(tensor - opt.target), 'mse')

    # add summary
    tf.sg_summary_loss(out, name=opt.name)

    return out


@tf.sg_sugar_func
def sg_mae(tensor, opt):
    r"""Returns absolute error between `tensor` and `target`.
    
    Args:
      tensor: A `Tensor`.
      opt:
        target: A `Tensor` with the same shape and dtype as `tensor`.
        name: A `string`. A name to display in the tensor board web UI.
      
    Returns:
      A `Tensor` of the same shape and dtype as `tensor` 
    
    For example,
    
    ```
    tensor = [[34, 11, 40], [13, 30, 42]]
    target = [[34, 10, 41], [14, 31, 40]]
    tensor.sg_mse(target=target) => [[ 0.  1.  1.]
                                     [ 1.  1.  2.]]
    ```
    """
    assert opt.target is not None, 'target is mandatory.'

    # absolute error
    out = tf.identity(tf.abs(tensor - opt.target), 'mae')

    # add summary
    tf.sg_summary_loss(out, name=opt.name)

    return out


@tf.sg_sugar_func
def sg_hinge(tensor, opt):
    r"""Returns hinge loss between `tensor` and `target`.
    
    Args:
      tensor: A `Tensor`.
      opt:
        target: A `Tensor`. Labels.
        margin: An int. Maximum margin. Default is 1.
        name: A `string`. A name to display in the tensor board web UI.
      
    Returns:
      A `Tensor`.
    
    For example,
    
    ```
    tensor = [[30, 10, 40], [13, 30, 42]]
    target = [[0, 0, 1], [0, 1, 0]]
    tensor.sg_hinge(target=target, one_hot=True) =>     [[ 1.  1.  0.]
                                                         [ 1.  0.  1.]]
    ```
    """
    assert opt.target is not None, 'target is mandatory.'

    # default margin
    opt += tf.sg_opt(margin=1)

    # reshape target
    shape = tensor.get_shape().as_list()
    broadcast_shape = [-1] + [1] * (len(shape) - 2) + [shape[-1]]
    target = tf.cast(tf.reshape(opt.target, broadcast_shape), tf.sg_floatx)
    
    # hinge loss
    out = tf.identity(tf.maximum(opt.margin - target * tensor, 0), 'hinge')

    # add summary
    tf.sg_summary_loss(out, name=opt.name)

    return out


@tf.sg_sugar_func
def sg_ctc(tensor, opt):
    r"""Computes the CTC (Connectionist Temporal Classification) Loss between `tensor` and `target`.

    Args:
      tensor: A 3-D `float Tensor`.
      opt:
        target: A `Tensor` with the same length in the first dimension as the `tensor`. Labels. ( Dense tensor )
        name: A `string`. A name to display in the tensor board web UI.

    Returns:
      A 1-D `Tensor` with the same length in the first dimension of the `tensor`.

    For example,

    ```
    tensor = [[[2., -1., 3.], [3., 1., -2.]], [[1., -1., 2.], [3., 1., -2.]]]
    target = [[2., 1.], [2., 3.]]
    tensor.sg_ctc(target=target) => [ 4.45940781  2.43091154]
    ```
    """
    assert opt.target is not None, 'target is mandatory.'

    # default sequence length
    shape = tf.shape(tensor)
    opt += tf.sg_opt(seq_len=tf.ones((shape[0],), dtype=tf.sg_intx) * shape[1])

    # ctc loss
    out = tf.nn.ctc_loss(tensor, opt.target.sg_to_sparse(), opt.seq_len, time_major=False)
    out = tf.identity(out, 'ctc')

    # add summary
    tf.sg_summary_loss(out, name=opt.name)

    return out
