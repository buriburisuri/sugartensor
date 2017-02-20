from __future__ import absolute_import
import sugartensor as tf
# noinspection PyPackageRequirements
import numpy as np


__author__ = 'buriburisuri@gmail.com'


#
# transform sugar functions
#

@tf.sg_sugar_func
def sg_identity(tensor, opt):
    r"""Returns the same tensor

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        name : If provided, it replaces current tensor's name

    Returns:
        A `Tensor`. Has the same content as `tensor`.
    """
    return tf.identity(tensor, name=opt.name)


@tf.sg_sugar_func
def sg_cast(tensor, opt):
    r"""Casts a tensor to a new type.
    
    See `tf.cast()` in tensorflow.

    Args:
      tensor: A `Tensor` or `SparseTensor` (automatically given by chain).
      opt:
        dtype : The destination type.
        name : If provided, it replaces current tensor's name

    Returns:
      A `Tensor` or `SparseTensor` with same shape as `tensor`.
    """
    assert opt.dtype is not None, 'dtype is mandatory.'
    return tf.cast(tensor, opt.dtype, name=opt.name)


@tf.sg_sugar_func
def sg_float(tensor, opt):
    r"""Casts a tensor to floatx.
    
    See `tf.cast()` in tensorflow.

    Args:
      tensor: A `Tensor` or `SparseTensor` (automatically given by chain).
      opt:
        name : If provided, it replaces current tensor's name

    Returns:
      A `Tensor` or `SparseTensor` with same shape as `tensor`.
    """
    return tf.cast(tensor, tf.sg_floatx, name=opt.name)


@tf.sg_sugar_func
def sg_int(tensor, opt):
    r"""Casts a tensor to intx.
    
    See `tf.cast()` in tensorflow.

    Args:
      tensor: A `Tensor` or `SparseTensor` (automatically given by chain).
      opt:
        name: If provided, it replaces current tensor's name.

    Returns:
      A `Tensor` or `SparseTensor` with same shape as `tensor`.
    """
    return tf.cast(tensor, tf.sg_intx, name=opt.name)


@tf.sg_sugar_func
def sg_expand_dims(tensor, opt):
    r"""Inserts a new axis.
    
    See tf.expand_dims() in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis : Dimension to expand. Default is -1.
        name: If provided, it replaces current tensor's name.

    Returns:
        A `Tensor`.
    """
    opt += tf.sg_opt(axis=-1)
    return tf.expand_dims(tensor, opt.axis, name=opt.name)


@tf.sg_sugar_func
def sg_squeeze(tensor, opt):
    r"""Removes axis of size 1 from the shape of a tensor.
    
    See `tf.squeeze()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis : A tuple/list of integers or an integer.
               axis to remove. Default is -1.
        name: If provided, it replaces current tensor's name.

    Returns:
      A `Tensor`.
    """
    opt += tf.sg_opt(axis=[-1])
    opt.axis = opt.axis if isinstance(opt.axis, (tuple, list)) else [opt.axis]
    return tf.squeeze(tensor, opt.axis, name=opt.name)


@tf.sg_sugar_func
def sg_flatten(tensor, opt):
    r"""Reshapes a tensor to `batch_size x -1`.
    
    See `tf.reshape()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        name: If provided, it replaces current tensor's name.

    Returns:
      A 2-D tensor.

    """
    dim = np.prod(tensor.get_shape().as_list()[1:])
    return tf.reshape(tensor, [-1, dim], name=opt.name)


@tf.sg_sugar_func
def sg_reshape(tensor, opt):
    r"""Reshapes a tensor.
    
    See `tf.reshape()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        shape: A tuple/list of integers. The destination shape.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    assert opt.shape is not None, 'shape is mandatory.'
    return tf.reshape(tensor, opt.shape, name=opt.name)


@tf.sg_sugar_func
def sg_transpose(tensor, opt):
    r"""Permutes the dimensions according to `opt.perm`.

    See `tf.transpose()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        perm: A permutation of the dimensions of `tensor`. The target shape.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    assert opt.perm is not None, 'perm is mandatory'
    return tf.transpose(tensor, opt.perm, name=opt.name)


@tf.sg_sugar_func
def sg_argmax(tensor, opt):
    r"""Returns the indices of the maximum values along the specified axis.
    
    See `tf.argmax()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis: Target axis. Default is the last one.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    opt += tf.sg_opt(axis=tensor.get_shape().ndims-1)
    return tf.argmax(tensor, opt.axis, opt.name)


@tf.sg_sugar_func
def sg_argmin(tensor, opt):
    r"""Returns the indices of the minimum values along the specified axis.

    See `tf.argin()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis: Target axis. Default is the last one.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    opt += tf.sg_opt(axis=tensor.get_shape().ndims - 1)
    return tf.argmin(tensor, opt.axis, opt.name)


@tf.sg_sugar_func
def sg_concat(tensor, opt):
    r"""Concatenates tensors along a axis.

    See `tf.concat()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        target: A `Tensor`. Must have the same rank as `tensor`, and
          all dimensions except `opt.dim` must be equal.
        axis : Target axis. Default is the last one.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    assert opt.target is not None, 'target is mandatory.'
    opt += tf.sg_opt(axis=tensor.get_shape().ndims-1)
    target = opt.target if isinstance(opt.target, (tuple, list)) else [opt.target]
    return tf.concat([tensor] + target, opt.axis, name=opt.name)


@tf.sg_sugar_func
def sg_one_hot(tensor, opt):
    r"""Converts a tensor into a one-hot tensor.
    
    See `tf.one_hot()` in tensorflow.

    Args:
      tensor: A `Tensor` ( automatically given by chain )
      opt:
        depth: The number of classes.
        name: If provided, replace current tensor's name.

    Returns:
        A `Tensor`.
    """
    assert opt.depth is not None, 'depth is mandatory.'
    return tf.one_hot(tensor, opt.depth, name=opt.name)


# noinspection PyUnusedLocal
@tf.sg_sugar_func
def sg_to_sparse(tensor, opt):
    r"""Converts a dense tensor into a sparse tensor.
    
    See `tf.SparseTensor()` in tensorflow.

    Args:
      tensor: A `Tensor` with zero-padding (automatically given by chain).
      opt:
        name: If provided, replace current tensor's name.

    Returns:
        A `SparseTensor`.
    """
    indices = tf.where(tf.not_equal(tensor.sg_float(), 0.))
    return tf.SparseTensor(indices=indices,
                           values=tf.gather_nd(tensor, indices) - 1,  # for zero-based index
                           shape=tf.shape(tensor).sg_cast(dtype=tf.int64))


#
# reduce functions
#

@tf.sg_sugar_func
def sg_sum(tensor, opt):
    r"""Computes the sum of elements across axis of a tensor.
    
    See `tf.reduce_sum()` in tensorflow.

    Args:
      tensor: A `Tensor` with zero-padding (automatically given by chain).
      opt:
        axis: A tuple/list of integers or an integer. The axis to reduce.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: If provided, replace current tensor's name.

    Returns:
        A `Tensor`.
    """
    return tf.reduce_sum(tensor, axis=opt.axis, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_mean(tensor, opt):
    r"""Computes the mean of elements across axis of a tensor.
    
    See `tf.reduce_mean()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis : A tuple/list of integers or an integer. The axis to reduce.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    return tf.reduce_mean(tensor, axis=opt.axis, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_prod(tensor, opt):
    r"""Computes the product of elements across axis of a tensor.

    See `tf.reduce_prod()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis : A tuple/list of integers or an integer. The axis to reduce.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    return tf.reduce_prod(tensor, axis=opt.axis, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_min(tensor, opt):
    r"""Computes the minimum of elements across axis of a tensor.

    See `tf.reduce_min()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis : A tuple/list of integers or an integer. The axis to reduce.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    return tf.reduce_min(tensor, axis=opt.axis, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_max(tensor, opt):
    r"""Computes the maximum of elements across axis of a tensor.

    See `tf.reduce_max()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis : A tuple/list of integers or an integer. The axis to reduce.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    return tf.reduce_max(tensor, axis=opt.axis, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_all(tensor, opt):
    r"""Computes the "logical and" of elements across axis of a tensor.
    
    See `tf.reduce_all()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis : A tuple/list of integers or an integer. The axis to reduce.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    return tf.reduce_all(tensor, axis=opt.axis, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_any(tensor, opt):
    r"""Computes the "logical or" of elements across axis of a tensor.

    See `tf.reduce_any()` in tensorflow.

    Args:
      tensor: A `Tensor` (automatically given by chain).
      opt:
        axis : A tuple/list of integers or an integer. The axis to reduce.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`.
    """
    return tf.reduce_any(tensor, axis=opt.axis, keep_dims=opt.keep_dims, name=opt.name)


#
# complicated transform function ( layer related )
#


@tf.sg_sugar_func
def sg_pool(tensor, opt):
    r"""Performs the 2-D pooling on the `tensor`.
    Mostly used with sg_conv().

    Args:
      tensor: A 4-D `Tensor` (automatically given by chain).
      opt:
        size: A tuple or list of integers of length 2 representing `[kernel height, kernel width]`.
          Can be an int if both values are the same.
          If not specified, (2, 2) is set implicitly.
        stride: A tuple or list of integers of length 2 or 4 representing stride dimensions.
          If the length is 2, i.e., (a, b), the stride is `[1, a, b, 1]`.
          If the length is 4, i.e., (a, b, c, d), the stride is `[a, b, c, d]`.
          Can be an int. If the length is an int, i.e., a, the stride is `[1, a, a, 1]`.
          The default value is [1, 1, 1, 1].
        avg: Boolean. If True, average pooling is applied. Otherwise, max pooling.
        name: If provided, replace current tensor's name.

    Returns:
      A `Tensor`. The max pooled output tensor.
    """

    # default stride and pad
    opt += tf.sg_opt(stride=(1, 2, 2, 1), pad='VALID')

    # shape stride
    opt.stride = opt.stride if isinstance(opt.stride, (list, tuple)) else [1, opt.stride, opt.stride, 1]
    opt.stride = [1, opt.stride[0], opt.stride[1], 1] if len(opt.stride) == 2 else opt.stride

    # shape size
    opt += tf.sg_opt(size=opt.stride)
    opt.size = opt.size if isinstance(opt.size, (list, tuple)) else [1, opt.size, opt.size, 1]
    opt.size = [1, opt.size[0], opt.size[1], 1] if len(opt.size) == 2 else opt.size

    if opt.avg:
        out = tf.nn.avg_pool(tensor, opt.size, opt.stride, opt.pad)
    else:
        out = tf.nn.max_pool(tensor, opt.size, opt.stride, opt.pad)

    return tf.identity(out, name=opt.name)


@tf.sg_sugar_func
def sg_pool1d(tensor, opt):
    r"""Performs the 1-D pooling on the `tensor`.
    
    Args:
      tensor: A 3-D `Tensor` (automatically passed by decorator).
      opt:
        size: A positive `integer` representing `[kernel width]`.
          Default is 2.
        stride: A positive `integer`. The number of entries by which
          the filter is moved right at each step. Default is 2.
        avg: Boolean. If True, average pooling is applied. Otherwise, max pooling.
        name: If provided, replace current tensor's name.

    Returns:
        A tensor

    """

    # default stride and pad
    opt += tf.sg_opt(stride=2, pad='VALID')
    opt += tf.sg_opt(size=opt.stride)

    if opt.avg:
        out = tf.nn.avg_pool(tensor.sg_expand_dims(axis=2),
                             (1, opt.size, 1, 1), (1, opt.stride, 1, 1), opt.pad)
    else:
        out = tf.nn.max_pool(tensor.sg_expand_dims(axis=2),
                             (1, opt.size, 1, 1), (1, opt.stride, 1, 1), opt.pad)

    return tf.identity(out.sg_squeeze(axis=2), name=opt.name)


@tf.sg_sugar_func
def sg_lookup(tensor, opt):
    r"""Looks up the `tensor`, which is the embedding matrix.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
          emb: A 2-D `Tensor`. An embedding matrix.
          name: If provided, replace current tensor's name.

    Returns:
        A `Tensor`.

    """
    assert opt.emb is not None, 'emb is mandatory.'
    return tf.nn.embedding_lookup(opt.emb, tensor, name=opt.name)


@tf.sg_sugar_func
def sg_reverse_seq(tensor, opt):
    r"""Reverses variable length slices.

    Before applying the pure tensorflow function tf.reverse_sequence,
      this function calculates sequence lengths by counting non-zeros.

    For example,
    
    ```
    tensor = [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
    tensor.sg_reverse_seq()
    => [[3 2 1 0 0]
        [5 4 0 0 0]]
    ```
        
    Args:
      tensor: A 2-D `Tensor` (automatically given by chain).
      opt:
        axis: Axis to reverse. Default is 1.
        name : If provided, it replaces current tensor's name.

    Returns:
      A `Tensor` with the same shape and type as `tensor`.
    """
    # default sequence dimension
    opt += tf.sg_opt(axis=1)
    seq_len = tf.not_equal(tensor, tf.zeros_like(tensor)).sg_int().sg_sum(axis=opt.axis)
    return tf.reverse_sequence(tensor, seq_len, opt.axis, name=opt.name)


@tf.sg_sugar_func
def sg_periodic_shuffle(tensor, opt):
    r""" Periodic shuffle transformation for SubPixel CNN.
        (see [Shi et al. 2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)
        
    Args:
      tensor: A tensor (automatically given by chain).
      opt:
        factor: factor to multiply shape by. Default is 2.
        name : If provided, it replaces current tensor's name.

    Returns:
        A tensor
    """
    # default factor
    opt += tf.sg_opt(factor=2)

    # get current shape
    batch, row, col, channel = tensor.get_shape().as_list()

    # get target channel num
    channel_target = channel // (opt.factor * opt.factor)
    channel_factor = channel // channel_target

    # intermediate shape for shuffling
    shape_1 = [batch, row, col, channel_factor // opt.factor, channel_factor // opt.factor]
    shape_2 = [batch, row * opt.factor, col * opt.factor, 1]

    # reshape and transpose for periodic shuffling for each channel
    out = []
    for i in range(channel_target):
        out.append((tensor[:, :, :, i*channel_factor:(i+1)*channel_factor])
                   .sg_reshape(shape=shape_1)
                   .sg_transpose(perm=(0, 1, 3, 2, 4))
                   .sg_reshape(shape=shape_2))

    # final output
    out = tf.concat(out, 3)

    return tf.identity(out, name=opt.name)


@tf.sg_sugar_func
def sg_inverse_periodic_shuffle(tensor, opt):
    r"""Inverse periodic shuffle transformation for SubPixel CNN.
        (see [Shi et al. 2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)
        
    Args:
      tensor: A tensor (automatically given by chain).
      opt:
        factor: factor to multiply shape by. Default is 2.
        name : If provided, it replaces current tensor's name.

    Returns:
        A tensor
    """
    # default factor
    opt += tf.sg_opt(factor=2)

    # get current shape
    batch, row, col, channel = tensor.get_shape().as_list()

    # get target shape and channel num
    channel_factor = opt.factor * opt.factor

    # intermediate shape for shuffling
    shape_1 = [batch, row // opt.factor, col // opt.factor, channel_factor // opt.factor, channel_factor // opt.factor]
    shape_2 = [batch, row // opt.factor, col // opt.factor, channel_factor]

    # reshape and transpose for periodic shuffling for each channel
    out = []
    for i in range(channel):
        out.append(tensor[:, :, :, i]
                   .sg_expand_dims()
                   .sg_reshape(shape=shape_1)
                   .sg_transpose(perm=(0, 1, 3, 2, 4))
                   .sg_reshape(shape=shape_2))

    # final output
    out = tf.concat(out, 3)

    return tf.identity(out, name=opt.name)
