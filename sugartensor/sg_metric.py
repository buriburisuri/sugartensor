from __future__ import absolute_import
import sugartensor as tf

__author__ = 'buriburisuri@gmail.com'


#
# evaluation layer
#


@tf.sg_sugar_func
def sg_accuracy(tensor, opt):
    r"""Returns accuracy of predictions.
    
    Args:
      tensor: A `Tensor`. Probability distributions or unscaled prediction scores.
      opt:
        target: A 'Tensor`. Labels.
      
    Returns:
      A `Tensor` of the same shape as `tensor`. Each value will be 1 if correct else 0. 
    
    For example,
    
    ```
    tensor = [[20.1, 18, -4.2], [0.04, 21.1, 31.3]]
    target = [[0, 1]]
    tensor.sg_accuracy(target=target) => [[ 1.  0.]]
    ```
    """
    assert opt.target is not None, 'target is mandatory.'
    opt += tf.sg_opt(k=1)

    # # calc accuracy
    out = tf.identity(tf.equal(tensor.sg_argmax(), tf.cast(opt.target, tf.int64)).sg_float(), name='acc')
    # out = tf.identity(tf.nn.in_top_k(tensor, opt.target, opt.k).sg_float(), name='acc')

    return out
