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
    r""" Return same tensor

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
           name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.identity(tensor, name=opt.name)


@tf.sg_sugar_func
def sg_cast(tensor, opt):
    r""" Cast tensor's data type by given data type.
    See tf.cast() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dtype : data type to cast
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    assert opt.dtype is not None, 'dtype is mandatory.'
    return tf.cast(tensor, opt.dtype, name=opt.name)


@tf.sg_sugar_func
def sg_float(tensor, opt):
    r""" Cast tensor's data type as floatX data type.
    See tf.cast() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.cast(tensor, tf.sg_floatx, name=opt.name)


@tf.sg_sugar_func
def sg_int(tensor, opt):
    r""" Cast tensor's data type as intX data type.
    See tf.cast() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.cast(tensor, tf.sg_intx, name=opt.name)


@tf.sg_sugar_func
def sg_expand_dims(tensor, opt):
    r""" Expand dimension of tensor.
    See tf.expand_dims() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dim : dimension to expand ( default : -1 )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    opt += tf.sg_opt(dim=-1)
    return tf.expand_dims(tensor, opt.dim, name=opt.name)


@tf.sg_sugar_func
def sg_squeeze(tensor, opt):
    r""" Expand dimension of tensor.
    See tf.squeeze() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dim : dimension or dimension list to squeeze ( default : -1 )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    opt += tf.sg_opt(dim=[-1])
    opt.dim = opt.dim if isinstance(opt.dim, (tuple, list)) else [opt.dim]
    return tf.squeeze(tensor, opt.dim, name=opt.name)


@tf.sg_sugar_func
def sg_flatten(tensor, opt):
    r""" Flatten tensor to 2D ( batch x dim ).
    See tf.reshape() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    dim = np.prod(tensor.get_shape().as_list()[1:])
    return tf.reshape(tensor, [-1, dim], name=opt.name)


@tf.sg_sugar_func
def sg_reshape(tensor, opt):
    r""" Reshape tensor to arbitrary shape.
    See tf.reshape() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            shape : target shape ( list or tuple )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    assert opt.shape is not None, 'shape is mandatory.'
    return tf.reshape(tensor, opt.shape, name=opt.name)


@tf.sg_sugar_func
def sg_transpose(tensor, opt):
    r""" Transpose tensor to arbitrary dimension.
    See tf.transpose() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            perm : permutation list ( target dimension list )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    assert opt.perm is not None, 'perm is mandatory'
    return tf.transpose(tensor, opt.perm, name=opt.name)


@tf.sg_sugar_func
def sg_argmax(tensor, opt):
    r""" Get argmax value of tensor along the given dimension
    See tf.argmax() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dim : target dimension ( default : last dimension )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    opt += tf.sg_opt(dim=tensor.get_shape().ndims-1)
    return tf.argmax(tensor, opt.dim, opt.name)


@tf.sg_sugar_func
def sg_concat(tensor, opt):
    r""" Concat tensor with given target tensor along the given dimension
    See tf.concat() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            target : target tensor
            dim : target dimension ( default : last dimension )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    assert opt.target is not None, 'target is mandatory.'
    opt += tf.sg_opt(dim=tensor.get_shape().ndims-1)
    target = opt.target if isinstance(opt.target, (tuple, list)) else [opt.target]
    return tf.concat(opt.dim, [tensor] + target, name=opt.name)


@tf.sg_sugar_func
def sg_one_hot(tensor, opt):
    r""" Convert tensor into one-hot encoding vector
    See tf.one_hot() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            depth : class number
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    assert opt.depth is not None, 'depth is mandatory.'
    return tf.one_hot(tensor, opt.depth, name=opt.name)


# noinspection PyUnusedLocal
@tf.sg_sugar_func
def sg_to_sparse(tensor, opt):
    r""" Convert dense tensor into sparse tensor
    See tf.SparseTensor() in the tensorflow.

    Args:
        tensor: A tensor pad by zero ( automatically given by chain )
        opt:
            name : if provided, replace current tensor's name

    Returns:
        A tensor

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
    r""" Reduce sum tensor along the given dimensions
    See tf.reduce_sum() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dims : dimension list to reduce
            keep_dims : keep dimension after reducing if True ( default : None )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.reduce_sum(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_mean(tensor, opt):
    r""" Reduce mean tensor along the given dimensions
    See tf.reduce_mean() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dims : dimension list to reduce
            keep_dims : keep dimension after reducing if True ( default : None )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.reduce_mean(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_prod(tensor, opt):
    r""" Reduce prod tensor along the given dimensions
    See tf.reduce_prod() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dims : dimension list to reduce
            keep_dims : keep dimension after reducing if True ( default : None )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.reduce_prod(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_min(tensor, opt):
    r""" Reduce min tensor along the given dimensions
    See tf.reduce_min() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dims : dimension list to reduce
            keep_dims : keep dimension after reducing if True ( default : None )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.reduce_min(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_max(tensor, opt):
    r""" Reduce max tensor along the given dimensions
    See tf.reduce_max() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dims : dimension list to reduce
            keep_dims : keep dimension after reducing if True ( default : None )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.reduce_max(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_all(tensor, opt):
    r""" Reduce all tensor along the given dimensions
    See tf.reduce_all() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dims : dimension list to reduce
            keep_dims : keep dimension after reducing if True ( default : None )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.reduce_all(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


@tf.sg_sugar_func
def sg_any(tensor, opt):
    r""" Reduce any tensor along the given dimensions
    See tf.reduce_any() in the tensorflow.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
            dims : dimension list to reduce
            keep_dims : keep dimension after reducing if True ( default : None )
            name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    return tf.reduce_any(tensor, reduction_indices=opt.dims, keep_dims=opt.keep_dims, name=opt.name)


#
# complicated transform function ( layer related )
#


@tf.sg_sugar_func
def sg_pool(tensor, opt):
    r""" 2D pooling functions used with sg_conv().

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
          size: A tuple or list of integers of length 2 representing `[kernel height, kernel width]`.
            Can be an int if both values are the same.
            If not specified, (2, 2) is set implicitly.
          stride: A tuple or list of integers of length 2 or 4 representing stride dimensions.
            If the length is 2, i.e., (a, b), the stride is `[1, a, b, 1]`.
            If the length is 4, i.e., (a, b, c, d), the stride is `[a, b, c, d]`.
            Can be an int. If the length is an int, i.e., a, the stride is `[1, a, a, 1]`.
            The default value is [1, 1, 1, 1].
          avg : if True, average pooling instead of max pooling
          name : if provided, replace current tensor's name

    Returns:
        A tensor

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
    r""" 1D pooling functions used with sg_conv().

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
          size: Scalar ( default : 2)
          stride: Scaloar ( default : 2)
          avg : if True, average pooling instead of max pooling
          name : if provided, replace current tensor's name

    Returns:
        A tensor

    """

    # default stride and pad
    opt += tf.sg_opt(stride=2, pad='VALID')
    opt += tf.sg_opt(size=opt.stride)

    if opt.avg:
        out = tf.nn.avg_pool(tensor.sg_expand_dims(dim=2),
                             (1, opt.size, 1, 1), (1, opt.stride, 1, 1), opt.pad)
    else:
        out = tf.nn.max_pool(tensor.sg_expand_dims(dim=2),
                             (1, opt.size, 1, 1), (1, opt.stride, 1, 1), opt.pad)

    return tf.identity(out.sg_squeeze(dim=2), name=opt.name)


@tf.sg_sugar_func
def sg_lookup(tensor, opt):
    r""" Look up given embedding matrix with this tensor.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
          emb: Embedding matrix
          name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    assert opt.emb is not None, 'emb is mandatory.'
    return tf.nn.embedding_lookup(opt.emb, tensor, name=opt.name)


@tf.sg_sugar_func
def sg_reverse_seq(tensor, opt):
    r""" Reverse sequence of tensor.

    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
          dim: dimension to reverse ( default : 1 )
          name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    # default sequence dimension
    opt += tf.sg_opt(dim=1)
    seq_len = tf.not_equal(tensor, tf.zeros_like(tensor)).sg_int().sg_sum(dims=opt.dim)
    return tf.reverse_sequence(tensor, seq_len, opt.dim, name=opt.name)


@tf.sg_sugar_func
def sg_periodic_shuffle(tensor, opt):
    r""" Periodic shuffle transform for SubPixel CNN.
        (see : http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)
    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
          factor: factor to multiply shape ( default : 2 )
          name : if provided, replace current tensor's name

    Returns:
        A tensor

    """
    # default factor
    opt += tf.sg_opt(factor=2)

    # get current shape
    batch, row, col, channel = tensor.get_shape().as_list()

    # get target channel num
    channel_target = channel / (opt.factor * opt.factor)
    channel_factor = channel / channel_target

    # intermediate shape for shuffling
    shape_1 = [batch, row, col, channel_factor / opt.factor, channel_factor / opt.factor]
    shape_2 = [batch, row * opt.factor, col * opt.factor, 1]

    # reshape and transpose for periodic shuffling for each channel
    out = []
    for i in range(channel_target):
        out.append((tensor[:, :, :, i*channel_factor:(i+1)*channel_factor])
                   .sg_reshape(shape=shape_1)
                   .sg_transpose(perm=(0, 1, 3, 2, 4))
                   .sg_reshape(shape=shape_2))

    # final output
    out = tf.concat(3, out)

    return tf.identity(out, name=opt.name)


@tf.sg_sugar_func
def sg_inverse_periodic_shuffle(tensor, opt):
    r""" Inverse periodic shuffle transform for SubPixel CNN.
        (see : http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf)
    Args:
        tensor: A tensor ( automatically given by chain )
        opt:
          factor: factor to divide shape ( default : 2 )
          name : if provided, replace current tensor's name

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
    shape_1 = [batch, row / opt.factor, col / opt.factor, channel_factor // opt.factor, channel_factor // opt.factor]
    shape_2 = [batch, row / opt.factor, col / opt.factor, channel_factor]

    # reshape and transpose for periodic shuffling for each channel
    out = []
    for i in range(channel):
        out.append(tensor[:, :, :, i]
                   .sg_expand_dims()
                   .sg_reshape(shape=shape_1)
                   .sg_transpose(perm=(0, 1, 3, 2, 4))
                   .sg_reshape(shape=shape_2))

    # final output
    out = tf.concat(3, out)

    return tf.identity(out, name=opt.name)
