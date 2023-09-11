"""Collection of Ivy neural network layers in functional form."""

# global
from typing import Optional, Tuple, Union, Sequence

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    handle_array_function,
    handle_partial_mixed_function,
    inputs_to_ivy_arrays,
    to_native_arrays_and_back,
    inputs_to_native_shapes,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
    handle_device_shifting,
)
from ivy.utils.exceptions import handle_exceptions

# Extra #
# ------#


def _in_projection(
    q,
    k,
    v,
    w,
    b=None,
):
    """
    Projects query, key and value effeciently, depending on whether we are doing self-
    attention (query is key is value) or cross-attention (key is value) or an attention
    where query, key and value are all diferrent.

    it is only used in
    multi_head_attention layer.
    This helper function is a modified version of https://github.com/pytorch/pytorch/b
    lob/5293dee9208cc0e1e7db2ebdcbaef64908c087c6/torch/nn/functional.py#L4762.
    """
    E = q.shape[-1]
    if k is v:
        if q is k:
            # self-attention
            proj = ivy.linear(q, w, bias=b)
            proj = proj.split(num_or_size_splits=3, axis=-1)
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split(num_or_size_splits=[E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = ivy.linear(q, w_q, bias=b_q)
            kv_proj = ivy.linear(k, w_kv, bias=b_kv)
            kv_proj = kv_proj.split(num_or_size_splits=2, axis=-1)
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.split(num_or_size_splits=3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.split(num_or_size_splits=3)
        return (
            ivy.linear(q, w_q, bias=b_q),
            ivy.linear(k, w_k, bias=b_k),
            ivy.linear(v, w_v, bias=b_v),
        )


# Linear #
@handle_exceptions
@handle_nestable
@handle_partial_mixed_function
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def linear(
    x: Union[ivy.Array, ivy.NativeArray],
    weight: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Apply a linear transformation to the incoming data: y = x * t(weight) + bias.
    The operation also supports batching of the weight matrices. This is useful if a
    batch of different network parameters are to be represented.

    Parameters
    ----------
    x
        The input x to compute linear transformation on.
        *[outer_batch_shape,inner_batch_shape,in_features]*
    weight
        The weight matrix. *[outer_batch_shape,out_features,in_features]*
    bias
        The bias vector, default is ``None``. *[outer_batch_shape,out_features]*
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Result array of the linear transformation.
        *[outer_batch_shape,inner_batch_shape,out_features]*
    
    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([1., 2., 3.])
    >>> w = ivy.array([[1., 0., 0.]])
    >>> y = ivy.linear(x, w)
    >>> print(y)
    ivy.array([1])  
    
    >>> x = ivy.array([[0.666, -0.4269, 1.911]])
    >>> w = ivy.array([[1., 0., 0.], [0., 0., 1.]])
    >>> y = ivy.zeros(2)
    >>> ivy.linear(x, w, out=y)
    >>> print(y)
    ivy.array([[0.666, 1.91 ]])

    >>> x = ivy.array([[1.546, 5.234, 6.487], \
                       [0.157, 5.753, 4.52], \
                       [5.165, 3.159, 7.101]])
    >>> w = ivy.array([[1.545, 2.547, 3.124], \
                       [5.852, 8.753, 6.963]])   
    >>> b = ivy.array([-1., 1.])
    >>> ivy.linear(x, w, bias=b, out=x)
    >>> print(x)
    ivy.array([[ 35. , 101. ],
               [ 28. ,  83.7],
               [ 37.2, 108. ]])
        
    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[1., 2., 3.], \
                                       [4., 5., 6.]]), \
                          b=ivy.array([1.1, 2.2, 3.3]))
    >>> w = ivy.Container(a=ivy.array([[1., 2., 3.], \
                                       [-1., 1., 2.]]), \
                          b=ivy.array([[0., -1., 1.], \
                                       [0., 1., 1.]]))
    >>> b = ivy.Container(a=ivy.array([1., -1.]), b=ivy.array([1., 1.]))
    >>> y = ivy.linear(x, w, bias=b)
    >>> print(y)
    {
        a: ivy.array([[15., 6.],
                      [33., 12.]]),
        b: ivy.array([2.1, 6.5])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([[1.1, 2.2, 3.3], \
                                       [11., 22., 33.]]), \
                          b=ivy.array([[1.245, 0.278, 4.105], \
                                       [7., 13., 17.]]))
    >>> w = ivy.array([[1., 2., 3.], \
                       [4., 5., 6.], \
                       [7., 8., 9.]])
    >>> b = ivy.Container(a=ivy.array([1., 0., -1.]), \
                          b=ivy.array([1., 1., 0.]))
    >>> ivy.linear(x, w, bias=b, out=x)
    >>> print(x)
    {
        a: ivy.array([[16.4, 35.2, 54.],
                      [155., 352., 549.]]),
        b: ivy.array([[15.1, 32., 47.9],
                      [85., 196., 306.]])
    }
    
    """
    outer_batch_shape = list(weight.shape[:-2])
    num_outer_batch_dims = len(outer_batch_shape)
    inner_batch_shape = list(x.shape[num_outer_batch_dims:-1])
    num_inner_batch_dims = len(inner_batch_shape)
    num_out_feats, num_in_feats = list(weight.shape[-2:])

    # OBS x IBS x OF
    y = ivy.matmul(
        x,
        ivy.swapaxes(
            ivy.reshape(
                weight,
                outer_batch_shape
                + [1] * max(num_inner_batch_dims - 1, 0)
                + [num_out_feats, num_in_feats],
            ),
            -1,
            -2,
        ),
    )

    if ivy.exists(bias):
        # OBS x [1]*len(IBS) x OF
        bias_broadcast = ivy.reshape(
            bias, outer_batch_shape + [1] * num_inner_batch_dims + [num_out_feats]
        )

        # OBS x IBS x OF
        y = y + bias_broadcast

    if ivy.exists(out):
        return ivy.inplace_update(out, y)
    return y


linear.mixed_backend_wrappers = {
    "to_add": (
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays", "handle_partial_mixed_function"),
}


# Dropout #


@handle_exceptions
@handle_nestable
@handle_partial_mixed_function
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def dropout(
    x: Union[ivy.Array, ivy.NativeArray],
    prob: float,
    /,
    *,
    scale: bool = True,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    training: bool = True,
    seed: Optional[int] = None,
    noise_shape: Optional[Sequence[int]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Randomly setting a fraction of input tensor to zeroes with probability.

    `prob` at each update during training time to prevent possible overfitting.
    The inputs not set to 0 are scaled up `1 / (1 - prob)` by default, so that
    overall sum is unchanged at training time and inference time.

    Parameters
    ----------
    x
        The input array x to perform dropout on.
    prob
        The probability of zeroing out each array element, float between 0 and 1.
    scale
        Whether to scale the output by `1/(1-prob)`. Default is ``True``.
    dtype
        output array data type. If dtype is None, the output array data type
        must be inferred from x. Default is ``None``.
    training
        Turn on dropout if training, turn off otherwise. Default is ``True``.
    seed
        Set a default seed for random number generating (for reproducibility). Default
        is ``None``.
    noise_shape
        a sequence representing the shape of the binary dropout mask that will be
        multiplied with the input. A shape dimension set to None means that a different
        mask value will be applied to each element of the input across that dimension. A
        dimension set to 1 means the same mask value will be applied to all elements of
        the input across that dimension.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Result array after dropout is performed.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[1., 2., 3.],
    ...                [4., 5., 6.],
    ...                [7., 8., 9.],
    ...                [10., 11., 12.]])
    >>> y = ivy.dropout(x,0.3)
    >>> print(y)
    ivy.array([[ 1.42857146,  2.85714293,  4.28571415],
           [ 0.        ,  7.14285755,  8.5714283 ],
           [10.        , 11.4285717 ,  0.        ],
           [14.2857151 ,  0.        , 17.1428566 ]])


    >>> x = ivy.array([[1.5, 2.6],
    ...                [4.9, 6.6],
    ...                [7.2, 8.7]])
    >>> y = ivy.dropout(x,0.5)
    >>> print(y)
    ivy.array([[ 0.        ,  5.19999981],
               [ 0.        ,  0.        ],
               [ 0.        , 17.39999962]])

    >>> x = ivy.array([[1., 2., 3.],
    ...                [4., 5., 6.],
    ...                [7., 8., 9.],
    ...                [10., 11., 12.]])
    >>> y = ivy.dropout(x,0.3,scale=False)
    >>> print(y)
    ivy.array([[ 1.,  2., 3.],
               [ 4.,  5., 0.],
               [ 7.,  0., 9.],
               [10., 11., 0.]])

    >>> x = ivy.array([[1.5, 2.6],
    ...                [4.9, 6.6],
    ...                [7.2, 8.7]])
    >>> y = ivy.dropout(x,0.5,scale=False)
    >>> print(y)
    ivy.array([[0., 2.6],
               [0., 0. ],
               [0., 8.7]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[1., 2., 3.], [4., 5., 6.]]),
    ...                   b=ivy.array([7., 8., 9.]))
    >>> y = ivy.dropout(x,0.3)
    >>> print(y)
    {
    a: ivy.array([[0., 0., 4.28571415],
                  [5.71428585, 7.14285755, 0.]]),
    b: ivy.array([0., 11.4285717, 12.8571434])
    }

    >>> x = ivy.Container(a=ivy.array([[1.1, 2.2, 3.3], [11., 22., 33.]]),
    ...                   b=ivy.array([[1.245, 0.278, 4.105], [7., 13., 17.]]))
    >>> y = ivy.dropout(x,0.5)
    >>> print(y)
    {
        a: ivy.array([[0., 4.4000001, 6.5999999],
                      [22., 44., 0.]]),
        b: ivy.array([[2.49000001, 0.55599999, 8.21000004],
                      [14., 0., 0.]])
    }

    >>> x = ivy.Container(a=ivy.array([[1., 2., 3.], [4., 5., 6.]]),
    ...                   b=ivy.array([7., 8., 9.]))
    >>> y = ivy.dropout(x,0.3)
    >>> print(y)
    {
        a: ivy.array([[0., 0., 3.],
                      [4., 5., 0.]]),
        b: ivy.array([0., 8., 9.])
    }

    >>> x = ivy.Container(a=ivy.array([[1.1, 2.2, 3.3], [11., 22., 33.]]),
    ...                   b=ivy.array([[1.245, 0.278, 4.105], [7., 13., 17.]]))
    >>> y = ivy.dropout(x,0.5)
    >>> print(y)
    {
        a: ivy.array([[0., 2.2, 3.3],
                      [11., 22., 0.]]),
        b: ivy.array([[1.245, 0.278, 4.105],
                      [7., 0., 0.]])
    }
    """
    if prob == 0 or not training:
        if dtype is not None:
            x = ivy.astype(x, dtype)
        return x if not ivy.exists(out) else ivy.inplace_update(out, x)
    if noise_shape is None:
        noise_shape = x.shape
    else:
        noise_shape = list(noise_shape)
        for i, v in enumerate(noise_shape):
            if v is None:
                noise_shape[i] = x.shape[i]
    mask = ivy.where(
        ivy.random_uniform(shape=noise_shape, device=ivy.dev(x), dtype=dtype, seed=seed)
        < prob,
        0.0,
        1.0,
    )
    x = x * mask
    if scale:
        x = ivy.multiply(x, 1.0 / (1.0 - prob), out=out)
    return x if not ivy.exists(out) else ivy.inplace_update(out, x)


dropout.mixed_backend_wrappers = {
    "to_add": (
        "handle_out_argument",
        "inputs_to_native_arrays",
        "outputs_to_ivy_arrays",
        "handle_device_shifting",
    ),
    "to_skip": ("inputs_to_ivy_arrays", "handle_partial_mixed_function"),
}


# Attention #


@handle_exceptions
@handle_array_like_without_promotion
@handle_array_function
def scaled_dot_product_attention(
    q: Union[ivy.Array, ivy.NativeArray],
    k: Union[ivy.Array, ivy.NativeArray],
    v: Union[ivy.Array, ivy.NativeArray],
    scale: float,
    /,
    *,
    mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Apply scaled dot product attention to inputs x using optional mask.

    Parameters
    ----------
    q
        The queries input array. The shape of queries input array should be in
        *[batch_shape,num_queries,feat_dim]*. The queries input array should have the
        same size as keys and values.
    k
        The keys input array. The shape of keys input array should be in
        *[batch_shape,num_keys,feat_dim]*. The keys input array should have the same
        size as queries and values.
    v
        The values input array. The shape of values input should be in
        *[batch_shape,num_keys,feat_dim]*. The values input array should have the same
        size as queries and keys.
    scale
        The scale float value.
        The scale float value is used to scale the query-key pairs before softmax.
    mask
        The mask input array. The mask to apply to the query-key values. Default is
        None. The shape of mask input should be in *[batch_shape,num_queries,num_keys]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The output following application of scaled dot-product attention.
        The output array is the weighted sum produced by the attention score and value.
        The shape of output array is *[batch_shape,num_queries,feat_dim]* .

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
    >>> k = ivy.array([[[0.6, 1.5], [2.4, 3.3],[4.2, 5.1]]])
    >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1],[4.3, 5.3]]])
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1)
    >>> print(result)
    ivy.array([[[4.04,5.03],[4.3,5.3],[4.3,5.3]]])


    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
    >>> k = ivy.array([[[0.6, 1.5], [2.4, 3.3],[4.2, 5.1]]])
    >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1],[4.3, 5.3]]])
    >>> mask = ivy.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]])
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1, mask=mask)
    >>> print(result)
    ivy.array([[[2.3, 3.23],[2.3, 3.23],[2.3, 3.23]]])

    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]])
    >>> k = ivy.array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
    >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
    >>> out = ivy.zeros(shape=(1, 3, 2))
    >>> ivy.scaled_dot_product_attention(q, k, v, 1, out=out)
    >>> print(out)
    ivy.array([[[4.04, 5.03],[4.3 , 5.3 ],[4.3 , 5.3 ]]])

    >>> q = ivy.native_array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
    >>> k = ivy.native_array([[[0.6, 1.5], [2.4, 3.3],[4.2, 5.1]]])
    >>> v = ivy.native_array([[[0.4, 1.3], [2.2, 3.1],[4.3, 5.3]]])
    >>> mask = ivy.native_array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],[0.0, 0.0, 0.0]]])
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1, mask=mask)
    >>> print(result)
    ivy.array([[[2.3, 3.23],[2.3, 3.23],[2.3, 3.23]]])

    >>> q = ivy.native_array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]])
    >>> k = ivy.native_array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
    >>> v = ivy.native_array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
    >>> out = ivy.zeros(shape=(1, 3, 2))
    >>> ivy.scaled_dot_product_attention(q, k, v, 1, out=out)
    >>> print(out)
    ivy.array([[[4.04, 5.03],[4.3 , 5.3 ],[4.3 , 5.3 ]]])


    With :class:`ivy.Container` input:

    >>> q = ivy.Container(a=ivy.array([[[0.2, 1.], [2.7, 3.], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[1.2, 1.], [2.2, 3.], [4.4, 5.6]]]))
    >>> k = ivy.Container(a=ivy.array([[[4.2, 1.], [2.2, 3.3], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[3.2, 1.], [2.2, 3.6], [4.0, 5.6]]]))
    >>> v = ivy.Container(a=ivy.array([[[5.2, 1.], [2.1, 3.], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]]))
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1)
    >>> print(result)
    {
        a:ivy.array([[[4.27, 5.4],[4.4, 5.6],[4.4, 5.6]]]),
        b:ivy.array([[[4.35, 5.54],[4.4, 5.6],[4.4, 5.6]]])
    }


    >>> q = ivy.Container(a=ivy.array([[[0.2, 1.], [2.7, 3.], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[1.2, 1.], [2.2, 3.], [4.4, 5.6]]]))
    >>> k = ivy.Container(a=ivy.array([[[4.2, 1.], [2.2, 3.3], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[3.2, 1.], [2.2, 3.6], [4.0, 5.6]]]))
    >>> v = ivy.Container(a=ivy.array([[[5.2, 1.], [2.1, 3.], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]]))
    >>> mask =
    ... ivy.Container(a=ivy.array([[[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]]),
    ...               b=ivy.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0,1.0]]]))
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1, mask=mask)
    >>> print(result)
    {
        a: ivy.array([[[4.27, 5.4],
                       [4.4, 5.6],
                       [4.4, 5.6]]]),
        b: ivy.array([[[4.35, 5.54],
                       [4.4, 5.6],
                       [4.4, 5.6]]])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.NativeArray` inputs:

    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
    >>> k = ivy.native_array([[[0.6, 1.5], [2.4, 3.3],[4.2, 5.1]]])
    >>> v = ivy.native_array([[[0.4, 1.3], [2.2, 3.1],[4.3, 5.3]]])
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1)
    >>> print(result)
    ivy.array([[
            [4.04, 5.03],
            [4.3 , 5.3 ],
            [4.3 , 5.3 ]
        ]])

    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]])
    >>> k = ivy.native_array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
    >>> v = ivy.native_array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
    >>> out = ivy.zeros(shape=(1, 3, 2))
    >>> ivy.scaled_dot_product_attention(q, k, v, 1, out=out)
    >>> print(out)
    ivy.array([[[4.04, 5.03],[4.3 , 5.3 ],[4.3 , 5.3 ]]])

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
    >>> k = ivy.Container(a=ivy.array([[[4.2, 1.], [2.2, 3.3], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[3.2, 1.], [2.2, 3.6], [4.0, 5.6]]]))
    >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1)
    >>> print(result)
    {
        a: ivy.array([[[4.14, 5.13],
                       [4.3, 5.3],
                       [4.3, 5.3]]]),
        b: ivy.array([[[4.09, 5.08],
                       [4.3, 5.3],
                       [4.3, 5.3]]])
    }
    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
    >>> k = ivy.Container(a=ivy.array([[[4.2, 1.], [2.2, 3.3],[4.4, 5.6]]]),
    ...                   b=ivy.array([[[3.2, 1.], [2.2, 3.6],[4.0, 5.6]]]))
    >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1],[4.3, 5.3]]])
    >>> mask = ivy.native_array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1)
    >>> print(result)
    {
        a: ivy.array([[[4.14, 5.13],
                    [4.3, 5.3],
                    [4.3, 5.3]]]),
        b: ivy.array([[[4.09, 5.08],
                    [4.3, 5.3],
                    [4.3, 5.3]]])
    }
    """
    # BS x Q x K
    sim = ivy.einsum("... q f, ... k f -> ... q k", q, k) * scale

    if ivy.exists(mask):
        # BS x Q x K
        sim = ivy.where(
            ivy.logical_not(mask),
            -ivy.ones_like(sim) * ivy.finfo(ivy.dtype(sim)).max,
            sim,
        )

    # BS x Q x K
    attn = ivy.softmax(sim, axis=-1)

    # BS x Q x F
    return ivy.einsum("... q k, ... k f -> ... q f", attn, v, out=out)


@handle_array_function
@inputs_to_ivy_arrays
@handle_out_argument
# @handle_array_like_without_promotion
@handle_nestable
@handle_exceptions
def multi_head_attention(
    query: Union[ivy.Array, ivy.NativeArray],
    key: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    value: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    /,
    *,
    num_heads: Optional[int] = 8,
    scale: Optional[float] = None,
    attention_mask: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    in_proj_weights: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    q_proj_weights: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    k_proj_weights: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    v_proj_weights: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out_proj_weights: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    in_proj_bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out_proj_bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    is_causal: Optional[bool] = False,
    return_attention_weights: Optional[bool] = False,
    average_attention_weights: Optional[bool] = True,
    dropout: Optional[float] = 0.0,
    training: Optional[bool] = False,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """
    Apply multi-head attention to inputs x. This is an implementation of multi-headed
    attention as described in the paper "Attention is all you Need" (Vaswani et al.,
    2017). If `query`, `key`, `value` are the same, then this is self-attention. Each
    timestep in `query` attends to the corresponding sequence in `key`, and returns a
    fixed-width vector. This layer first projects `query`, `key` and `value`. These are
    (effectively) a list of tensors of length `num_attention_heads`, where the
    corresponding shapes are `(batch_size, <query dimensions>, key_dim)`, `(batch_size,
    <key/value dimensions>, key_dim)`, `(batch_size, <key/value dimensions>,
    value_dim)`. Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then interpolated
    by these probabilities, then concatenated back to a single tensor. Finally, the
    result tensor with the last dimension as value_dim can take an linear projection and
    return.

    Parameters
    ----------
    query
        query embeddings *[batch_shape,num_queries,query_dim]*.
    key
        key embeddings *[batch_shape,num_queries,key_dim]*.
    value
        value embeddings *[batch_shape,num_queries,value_dim]*.
    num_heads
        The number of attention heads to use.
    scale
        The value by which to scale the query-key similarity measure before softmax.
    attention_mask
        The mask to apply to the query-key values. Default is ``None``.
        *[batch_shape,num_queries,num_keys]*.
    in_proj_weights
        The weights used to project query, key and value *[3*E, E].
    q_proj_weights
        The weights used to project query if in_proj_weights is None *[new_E, E].
    k_proj_weights
        The weights used to project key if in_proj_weights is None *[new_E, E].
    v_proj_weights
        The weights used to project value if in_proj_weights is None *[new_E, E].
    out_proj_weights
        The weights used to project the output.
    in_proj_bias
        The bias used when projecting with query, key and value.
    out_proj_bias
        The bias used when projecting the output.
    is_causal
        If True, Uses a causal attention mask and ignores provided attention_mask.
    return_attention_weights
        If True, returns attention_weights alongside the output
        as a tuple (output, attenion_weights). Defaults to `False`.
    average_attention_weights
        If true, indicates that the returned ``attention_weights`` should be averaged
        across heads. Otherwise, ``attention_weights`` are provided separately per head.
        Note that this flag only has an effect when ``return_attention_weights=True``.
        Default: ``True`` (i.e. average weights across heads)
    dropout
        Specifies the dropout probablity, dropout is applied to attention_weights.
    training
        If True, dropout is used, otherwise dropout is not activated.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The output following application of multi-head attention.
        *[batch_shape,num_queries,out_feat_dim]* if input is batched
        otherwise *[num_queries, out_feat_dim]

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.
    """
    num_dims = query.ndim
    ivy.assertions.check_all(
        num_dims > 1 and num_dims < 4,
        (
            "Number of dimensions should be 2 (for unbatched input) or 3 (for batched"
            f" input), got {num_dims}"
        ),
    )
    if key is None and value is None:
        key = value = query
    if num_dims == 2:
        query, key, value = [ivy.expand_dims(x, axis=0) for x in [query, key, value]]
    if ivy.exists(in_proj_weights):
        q, k, v = _in_projection(query, key, value, w=in_proj_weights, b=in_proj_bias)
    elif all([ivy.exists(x) for x in [q_proj_weights, k_proj_weights, v_proj_weights]]):
        if ivy.exists(in_proj_bias):
            b_q, b_k, b_v = ivy.split(in_proj_bias, num_or_size_splits=3)
        else:
            b_q = b_k = b_v = None
        q, k, v = (
            ivy.linear(query, q_proj_weights, bias=b_q),
            ivy.linear(key, k_proj_weights, bias=b_k),
            ivy.linear(value, v_proj_weights, bias=b_v),
        )
    else:
        q, k, v = query, key, value
    batch_size, q_seq_length, emb_dim = q.shape[0], q.shape[1], q.shape[-1]
    k_seq_length = k.shape[1]
    ivy.assertions.check_true(
        emb_dim % num_heads == 0, "features must be divisible by number of heads"
    )
    dims_per_head = emb_dim // num_heads
    # isolate heads
    q = q.reshape((batch_size, q_seq_length, num_heads, dims_per_head)).permute_dims(
        (0, 2, 1, 3)
    )
    k = k.reshape((batch_size, k_seq_length, num_heads, dims_per_head)).permute_dims(
        (0, 2, 3, 1)
    )
    v = v.reshape((batch_size, k_seq_length, num_heads, dims_per_head)).permute_dims(
        (0, 2, 1, 3)
    )
    # perform bmm
    attn_scores = ivy.matmul(q, k)
    # scale
    scale = 1 / (dims_per_head**0.5) if not scale else scale
    attn_scores *= scale
    # apply attention mask
    if ivy.exists(attention_mask) or is_causal:
        if is_causal:
            # create causal mask
            attention_mask = ivy.tril(ivy.ones((q_seq_length, k_seq_length)))
        attention_mask = attention_mask.astype("bool")
        attn_scores = ivy.where(attention_mask, attn_scores, -ivy.inf)
    # perform softmax
    attn_weights = ivy.softmax(attn_scores, axis=-1)
    # perform dropout
    attn_weights = ivy.dropout(attn_weights, dropout, training=training)
    # bmm with values
    attention_out = ivy.matmul(attn_weights, v)
    attention_out = attention_out.permute_dims((0, 2, 1, 3)).reshape(
        (batch_size, q_seq_length, -1)
    )
    # proj out if out_proj_weight exists
    if ivy.exists(out_proj_weights):
        attention_out = ivy.linear(attention_out, out_proj_weights, bias=out_proj_bias)
    # if input was unbatched, unbatchify the output
    if num_dims == 2:
        attention_out = attention_out.squeeze(axis=0)
    if return_attention_weights:
        if average_attention_weights:
            attn_weights = attn_weights.mean(axis=1)
            if num_dims == 2:
                attn_weights = attn_weights.squeeze(axis=0)
        return attention_out, attn_weights
    else:
        return attention_out


# Convolutions #


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def conv1d(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int]] = 1,
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[ivy.Array] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 1-D convolution given 3-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,w,d_in]* or *[batch_size,d_in,w]*.
    filters
        Convolution filters *[fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
        padding), or a sequence of n (low, high) integer pairs that give the padding to
        apply before and after each spatial dimension.
    data_format
        The ordering of the dimensions in the input, one of "NWC" or "NCW". "NWC"
        corresponds to input with shape (batch_size, width, channels), while "NCW"
        corresponds to input with shape (batch_size, channels, width).
    filter_format
        Either "channel_first" or "channel_last". "channel_first" corresponds to "OIW",
         input data formats, while "channel_last" corresponds to "WIO", "HWIO", "DHWIO".
     x_dilations
        The dilation factor for each dimension of input. (Default value = 1)
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    bias
        Bias array of shape *[d_out]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the convolution operation.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.asarray([[[0.], [3.], [0.]]]) #NWC
    >>> filters = ivy.array([[[0.]], [[1.]], [[0.]]]) #WIO
    >>> result = ivy.conv1d(x, filters, (1,), 'SAME', data_format='NWC',dilations= (1,))
    >>> print(result)
    ivy.array([[[0.], [3.], [0.]]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[[1., 3.], [2., 4.], [5., 7]]])
    >>> filters = ivy.native_array([[[0., 1.], [1., 0.]]])
    >>> result = ivy.conv1d(x, filters, (2,),'VALID')
    >>> print(result)
    ivy.array([[[3., 1.],
    ...         [7., 5.]]])

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a=ivy.array([[[1.2, 3.1, 4.8], [5.9, 2.2, 3.3],
    ...                                 [10.8, 7.6, 4.9], [6.1, 2.2, 9.5]]]),
    ...                   b=ivy.array([[[8.8, 7.7, 6.6], [1.1, 2.2, 3.5]]]))
    >>> filters = ivy.array([[[1., 0., 1.], [0., 1., 0.], [1., 1., 0.]]])
    >>> result  = ivy.conv1d(x, filters, 3, 'VALID')
    >>> print(result)
    {
            a: ivy.array([[[6., 7.9, 1.2],
    ...                    [15.6, 11.7, 6.1]]]),
    ...     b: ivy.array([[[15.4, 14.3, 8.8]]])
    }
    """
    return current_backend(x).conv1d(
        x,
        filters,
        strides,
        padding,
        data_format=data_format,
        filter_format=filter_format,
        x_dilations=x_dilations,
        dilations=dilations,
        bias=bias,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def conv1d_transpose(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "NWC",
    dilations: Union[int, Tuple[int]] = 1,
    bias: Optional[ivy.Array] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 1-D transpose convolution given 3-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,w,d_in]* or *[batch_size,d_in,w]*.
    filters
        Convolution filters *[fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        Either ‘SAME’ (padding so that the output's shape is the same as the
        input's), or ‘VALID’ (padding so that the output's shape is `output_shape`).
    output_shape
        Shape of the output (Default value = None)
    data_format
        The ordering of the dimensions in the input, one of "NWC" or "NCW". "NWC"
        corresponds to input with shape (batch_size, width, channels), while "NCW"
        corresponds to input with shape (batch_size, channels, width).
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    bias
        Bias array of shape *[d_out]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose convolution operation.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 28, 3])
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 6])
    >>> y = ivy.conv1d_transpose(x, filters, 2, 'SAME')
    >>> print(y.shape)
    (1, 56, 6)

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 128, 64])
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[1, 64, 64])
    >>> ivy.conv1d_transpose(x, filters, 1, 'VALID', out=x)
    >>> print(x.shape)
    (1, 128, 64)

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 256, 64])
    >>> y = ivy.zeros_like(x)
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 64, 32])
    >>> ivy.conv1d_transpose(x, filters, [1, 1, 1], 'VALID', out=y)
    >>> print(y.shape)
    (1, 258, 32)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array(
    ...         ivy.random_normal(mean=0, std=1, shape=[1,256,128])
    ... )
    >>> filters = ivy.native_array(
    ...         ivy.random_normal(mean=0, std=1, shape=[3, 128, 32])
    ... )
    >>> y = ivy.conv1d_transpose(x, filters, 2, 'SAME')
    >>> print(y.shape)
    (1, 512, 32)

    With one :class:`ivy.Container` input:

    >>> x = ivy.full((1, 6, 1), 2.7)
    >>> a = ivy.random_normal(mean=0, std=1, shape=[3, 1, 1])
    >>> b = ivy.random_normal(mean=0, std=1, shape=[3, 1, 1])
    >>> filters = ivy.Container(a=a, b=b)
    >>> y = ivy.conv1d_transpose(x, filters, 1, 'VALID', dilations=2)
    >>> print(y.shape)
    {
        a: [1,10,1],
        b: [1,10,1]
    }

    With multiple :class:`ivy.Container` inputs:

    >>> a = ivy.random_normal(mean=0, std=1, shape=[1, 14, 3])
    >>> b = ivy.random_normal(mean=0, std=1, shape=[1, 28, 3])
    >>> c = ivy.random_normal(mean=0, std=1, shape=[3, 3, 6])
    >>> d = ivy.random_normal(mean=0, std=1, shape=[3, 3, 6])
    >>> x = ivy.Container(a=a, b=b)
    >>> filters = ivy.Container(c=c, d=d)
    >>> y = ivy.conv1d_transpose(x, filters, 2, 'SAME')
    >>> print(y.shape)
    {
        a: {
            c: [1,28,6],
            d: [1,28,6]
        },
        b: {
            c: [1,56,6],
            d: [1,56,6]
        }
    }
    """
    return current_backend(x).conv1d_transpose(
        x,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        bias=bias,
        out=out,
    )


@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def conv2d(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[ivy.Array] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 2-D convolution given 4-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,h,w,d_in]* or *[batch_size,d_in,h,w]*.
    filters
        Convolution filters *[fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
        padding), or a sequence of n (low, high) integer pairs that give the padding to
        apply before and after each spatial dimension.
    data_format
        The ordering of the dimensions in the input, one of "NHWC" or "NCHW". "NHWC"
        corresponds to inputs with shape (batch_size, height, width, channels), while
        "NCHW" corresponds to input with shape (batch_size, channels, height, width).
    filter_format
        Either "channel_first" or "channel_last". "channel_first" corresponds to "OIHW",
         input data formats, while "channel_last" corresponds to "HWIO".
     x_dilations
        The dilation factor for each dimension of input. (Default value = 1)
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    bias
        Bias array of shape *[d_out]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the convolution operation.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[[[1.], [2.0],[3.]],
    ...                 [[1.], [2.0],[3.]],
    ...                 [[1.], [2.0],[3.]]]]) #NHWC

    >>> filters = ivy.array([[[[0.]],[[1.]],[[0.]]],
    ...                      [[[0.]],[[1.]], [[0.]]],
    ...                      [[[0.]],[[1.]], [[0.]]]]) #HWIO
    >>> result = ivy.conv2d(x, filters, (1,), 'SAME', data_format='NHWC',
    ... dilations= (1,))
    >>> print(result)
    ivy.array([[
              [[2.],[4.],[6.]],
              [[3.],[6.],[9.]],
              [[2.],[4.],[6.]]
              ]])

    With one :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[[[1.], [2.0],[3.]],
    ...                                 [[1.], [2.0],[3.]],
    ...                                 [[1.], [2.0],[3.]]]]))
    >>> filters = ivy.eye(3, 3).reshape((3, 3, 1, 1)).astype(ivy.float32)
    >>> result = ivy.conv2d(x, filters, (2,), 'SAME', data_format='NHWC',
    ...    dilations= (1,))
    >>> print(result)
    {
        a:ivy.array([[[[3.], [3.]], [[1.], [5.]]]])
    }

    With multiple :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a = ivy.eye(3, 3).reshape((1, 3, 3, 1)),
    ...                   b = ivy.eye(4, 4).reshape((1, 4, 4, 1)),
    ...                   c = ivy.eye(5, 5).reshape((1, 5, 5, 1)))
    >>> filters = ivy.array([[1, 1, 1],
    ...                      [0, 1, 1],
    ...                      [0, 0, 1]], dtype = ivy.float32).reshape((3, 3, 1, 1))
    >>> result = ivy.conv2d(x, filters, (2,), 'SAME')
    >>> print(result)
    {
        a:ivy.array([[[[2.], [0.]], [[1.], [2.]]]]),
        b:ivy.array([[[[3.], [0.]], [[1.], [2.]]]]),
        c:ivy.array([[[[2.], [0.], [0.]],
                      [[1.], [3.], [0.]],
                      [[0.], [1.], [2.]]
                    ]])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a = ivy.eye(3, 3).reshape((1, 3, 3, 1)),
    ...                   b = ivy.eye(5, 5).reshape((1, 5, 5, 1)))
    >>> filters = ivy.array([[2, 0, 1],
    ...                      [1, 3, 1],
    ...                      [0, 1, 1]], dtype = ivy.float32).reshape((3, 3, 1, 1))
    >>> result = ivy.conv2d(x, filters, (2,), 'SAME')
    >>> print(result)
    {
        a:ivy.array([[[[4.],[0.]],[[1.],[5.]]]]),
        b:ivy.array([[[[4.],[0.],[0.]],[[1.],[6.],[0.]],[[0.],[1.],[5.]]]])
    }
    """
    return current_backend(x).conv2d(
        x,
        filters,
        strides,
        padding,
        data_format=data_format,
        filter_format=filter_format,
        x_dilations=x_dilations,
        dilations=dilations,
        bias=bias,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def conv2d_transpose(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    bias: Optional[ivy.Array] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 2-D transpose convolution given 4-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,h,w,d_in]* or *[batch_size,d_in,h,w]*.
    filters
        Convolution filters *[fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        Either ‘SAME’ (padding so that the output's shape is the same as the
        input's), or ‘VALID’ (padding so that the output's shape is `output_shape`).
    output_shape
        Shape of the output (Default value = None)
    data_format
        The ordering of the dimensions in the input, one of "NHWC" or "NCHW". "NHWC"
        corresponds to inputs with shape (batch_size, height, width, channels), while
        "NCHW" corresponds to input with shape (batch_size, channels, height, width).
    filter_format
        Either "channel_first" or "channel_last". "channel_first" corresponds to
        "OIDHW" input data formats, while "channel_last" corresponds to "DHWIO" .
    x_dilations
        The dilation factor for each dimension of input. (Default value = 1)
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    bias
        Bias array of shape *[d_out]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose convolution operation.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 28, 28, 3])
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 6])
    >>> y = ivy.conv2d_transpose(x,filters,2,'SAME')
    >>> print(y.shape)
    (1, 56, 56, 6)

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 128, 128, 64])
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[1, 1, 64, 64])
    >>> ivy.conv2d_transpose(x,filters,1,'VALID',out=x)
    >>> print(x.shape)
    (1, 128, 128, 64)

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 256, 256, 64])
    >>> y = ivy.zeros_like(x)
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 64, 32])
    >>> ivy.conv2d_transpose(x,filters,[1, 1, 1],'VALID',out=y)
    >>> print(y.shape)
    (1, 258, 258, 32)

    With one :class:`ivy.Container` inputs:
    >>> x = ivy.full((1, 6, 6, 1), 2.7)
    >>> a = ivy.random_normal(mean=0, std=1, shape=[3, 3, 1, 1])
    >>> b = ivy.random_normal(mean=0, std=1, shape=[3, 3, 1, 1])
    >>> filters = ivy.Container(a=a, b=b)
    >>> y = ivy.conv2d_transpose(x,filters,1,'VALID',dilations=2)
    >>> print(y.shape)
    {
        a: [1,10,10,1],
        b: [1,10,10,1]
    }

    With multiple :class:`ivy.Container` inputs:
    >>> a = ivy.random_normal(mean=0, std=1, shape=[1, 14, 14, 3])
    >>> b = ivy.random_normal(mean=0, std=1, shape=[1, 28, 28, 3])
    >>> c = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 6])
    >>> d = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 6])
    >>> x = ivy.Container(a=a, b=b)
    >>> filters = ivy.Container(c=c, d=d)
    >>> y = ivy.conv2d_transpose(x,filters,2,'SAME')
    >>> print(y.shape)
    {
        a: {
            c: [1,28,28,6],
            d: [1,28,28,6]
        },
        b: {
            c: [1,56,56,6],
            d: [1,56,56,6]
        }
    }
    """
    return current_backend(x).conv2d_transpose(
        x,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        bias=bias,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def depthwise_conv2d(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int, int]] = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 2-D depthwise convolution given 4-D input ``x`` and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,h,w,d_in]* or *[batch_size,d_in,h,w]*.
    filters
        Convolution filters *[fh,fw,d_in]*. (d_in must be the same as d from x)
    strides
        The stride of the sliding window for each dimension of input.
    padding
        either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
        padding), or a sequence of n (low, high) integer pairs that give the padding to
        apply before and after each spatial dimension.
    data_format
        The ordering of the dimensions in the input, one of "NHWC" or "NCHW". "NHWC"
        corresponds to inputs with shape (batch_size, height, width, channels), while
        "NCHW" corresponds to input with shape (batch_size, channels, height, width).
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the convolution operation.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 28, 28, 3]) #NHWC
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3]) #HWI (I == d_in)
    >>> y = ivy.depthwise_conv2d(x, filters, (1, 1), 'VALID')
    >>> print(y.shape)
    (1, 26, 26, 3)

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 32, 32, 3]) #NHWC
    >>> y = ivy.zeros_like(x)
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[5, 5, 3]) #HWI (I == d_in)
    >>> ivy.depthwise_conv2d(x, filters, [2, 2], 'SAME', out=y)
    >>> print(y.shape)
    (1, 16, 16, 3)

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 64, 64, 32]) #NHWC
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[4, 4, 32]) #HWI (I == d_in)
    >>> ivy.depthwise_conv2d(x, filters, [1, 1], 'VALID', out=x)
    >>> print(x.shape)
    (1, 61, 61, 32)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array(
    ...     ivy.random_normal(mean=0, std=1, shape=[1, 7, 7, 64])
    ... ) #NHWC
    >>> filters = ivy.native_array(
    ...    ivy.random_normal(mean=0, std=1, shape=[3, 3, 64])
    ... ) #HWI (I == d_in)
    >>> y = ivy.depthwise_conv2d(x, filters, [1, 1], 'SAME')
    >>> print(y.shape)
    (1, 7, 7, 64)

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.eye(6, 6).reshape((1, 6, 6, 1)) #NHWC
    >>> a = ivy.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]]).expand_dims(axis=-1)
    >>> b = ivy.array([[1., 1., 1.],
    ...                [1., 1., 1.],
    ...                [1., 1., 1.]]).expand_dims(axis=-1) / 9.0
    >>> filters = ivy.Container(a = a, b = b)
    >>> y = ivy.depthwise_conv2d(x, filters, 1, 'VALID', dilations=2)
    >>> print(y)
    {
        a: ivy.array([[[[-6.],
                        [0.]],
                       [[0.],
                        [-6.]]]]),
        b: ivy.array([[[[0.333],
                        [0.]],
                       [[0.],
                        [0.333]]]])
    }

    With a mix of :class:`ivy.Array`, code:`ivy.NativeArray`
    and :class:`ivy.Container` inputs:

    >>> x = ivy.eye(6, 6).reshape((1, 6, 6, 1)) #NHWC
    >>> y = ivy.native_array(ivy.eye(6, 6).reshape((1, 6, 6, 1)))
    >>> inp = ivy.Container(x = x, y = y)
    >>> filter = ivy.array([[1., 1., 1.],
    ...                     [1., -8., 1.],
    ...                     [1., 1., 1.]]).expand_dims(axis=-1)
    >>> y = ivy.depthwise_conv2d(inp, filter, 1, 'VALID', dilations=2)
    >>> print(y)
    {
        x: ivy.array([[[[-6.],
                        [0.]],
                       [[0.],
                        [-6.]]]]),
        y: ivy.array([[[[-6.],[0.]],[[0.],[-6.]]]])
    }

    """
    return current_backend(x).depthwise_conv2d(
        x,
        filters,
        strides,
        padding,
        data_format=data_format,
        dilations=dilations,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def conv3d(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    strides: Union[int, Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    data_format: str = "NDHWC",
    filter_format: str = "channel_last",
    x_dilations: Union[int, Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[ivy.Array] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 3-D convolution given 5-D input x and filters arrays.

    Parameters
    ----------
    x
        Input volume *[batch_size,d,h,w,d_in]* or *[batch_size,d_in,d,h,w]*.
    filters
        Convolution filters *[fd,fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
        padding), or a sequence of n (low, high) integer pairs that give the padding to
        apply before and after each spatial dimension.
    data_format
        The ordering of the dimensions in the input, one of "NDHWC" or "NCDHW". "NDHWC"
        corresponds to inputs with shape (batch_size, depth, height, width, channels),
        while "NCDHW" corresponds to input with shape (batch_size, channels, depth,
        height, width).
    filter_format
        Either "channel_first" or "channel_last". "channel_first" corresponds 
        to "OIDHW",input data formats, while "channel_last" corresponds to "DHWIO".
     x_dilations
        The dilation factor for each dimension of input. (Default value = 1)    
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    bias
        Bias array of shape *[d_out]*
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the convolution operation.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array\
               ([[[1., 2. ,1.], [1., 2. ,1.], [1., 2. ,1.]],\
                [[1., 2. ,1.], [1., 2. ,1.], [1., 2. ,1.]],\
                [[1., 2. ,1.], [1., 2. ,1.], [1., 2. ,1.]]]).reshape((1, 3, 3, 3, 1))

    >>> filters = ivy.array([[[0.,1.,0.],\
                              [0.,1.,0.],\
                              [0.,1.,0.]]]).reshape((1,3,3,1,1))


    >>> result = ivy.conv3d(x, filters, (1,1,1), 'SAME', data_format = 'NDHWC',\
                            dilations = (1,1,1))

    >>> print(result)
    ivy.array([[[[[2.],[4.],[2.]],[[3.],[6.],[3.]],[[2.],[4.],[2.]]],
                [[[2.],[4.],[2.]],[[3.],[6.],[3.]],[[2.],[4.],[2.]]],
                [[[2.],[4.],[2.]],[[3.],[6.],[3.]],[[2.],[4.],[2.]]]]])

    With one :class:`ivy.Container` input:

    >>> x = ivy.Container(a = ivy.ones((1, 3, 3, 3, 1)).astype(ivy.float32) )

    >>> filters = ivy.ones((3, 3, 3, 1, 1)).astype(ivy.float32)

    >>> result = ivy.conv3d(x, filters, 2, 'SAME')
    >>> print(result)
    {
        a: ivy.array([[[[[8.],[8.]],[[8.],[8.]]],[[[8.],[8.]],[[8.],[8.]]]]])
    }

    With multiple :class:`ivy.Container` input:

    >>> x = ivy.Container( a = ivy.random_normal(mean = 0, std = 1,\
                               shape = [1, 3, 5, 5, 1]),\
                           b = ivy.random_normal(mean = 0, std = 1,\
                               shape = [1, 5, 32 ,32, 3]),\
                           c = ivy.random_normal(mean = 0, std = 1,\
                               shape = [1, 32, 32, 32, 1]))

    >>> filters = ivy.ones((3, 5, 5, 1, 3)).astype(ivy.float32) #DHWIO

    >>> result = ivy.conv3d(x, filters, 1, 'SAME')
    >>> print(result.cont_shapes)
    {
        a: [1,3,5,5,3],
        b: [1,5,32,32,3],
        c: [1,32,32,32,3]
    }
    """
    return current_backend(x).conv3d(
        x,
        filters,
        strides,
        padding,
        data_format=data_format,
        filter_format=filter_format,
        x_dilations=x_dilations,
        dilations=dilations,
        bias=bias,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def conv3d_transpose(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int, int, int]] = 1,
    bias: Optional[ivy.Array] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 3-D transpose convolution given 5-D input x and filters arrays.

    Parameters
    ----------
    x
        Input volume *[batch_size,d,h,w,d_in]* or *[batch_size,d_in,d,h,w]*.
    filters
        Convolution filters *[fd,fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        Either ‘SAME’ (padding so that the output's shape is the same as the
        input's), or ‘VALID’ (padding so that the output's shape is `output_shape`).
    output_shape
        Shape of the output (Default value = None)
    data_format
        The ordering of the dimensions in the input, one of "NDHWC" or "NCDHW". "NDHWC"
        corresponds to inputs with shape (batch_size, depth, height, width, channels),
        while "NCDHW" corresponds to input with shape (batch_size, channels, depth,
        height, width).
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    bias
        Bias array of shape *[d_out]*
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose convolution operation.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3])
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6])
    >>> y = ivy.conv3d_transpose(x, filters, 2, 'SAME')
    >>> print(y.shape)
    (1, 6, 56, 56, 6)

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 7, 256, 256, 64])
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 64, 32])
    >>> y = ivy.conv3d_transpose(x, filters, [1, 1, 1], 'VALID')
    >>> print(y.shape)
    (1, 9, 258, 258, 32)

    With :class:`ivy.Container` inputs:

    >>> a = ivy.random_normal(mean=0, std=1, shape=[1, 3, 14, 14, 3])
    >>> b = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3]))
    >>> c = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6])
    >>> d = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6]))
    >>> x = ivy.Container(a=a, b=b)
    >>> filters = ivy.Container(c=c, d=d)
    >>> y = ivy.conv3d_transpose(x, filters, 2, 'SAME')
    >>> print(y.shape)
    [1, 6, 56, 56, 6]

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.full((1, 6, 6, 6, 1), 2.7)
    >>> a =  ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 1, 1])
    >>> b =  ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 1, 1])
    >>> filters = ivy.Container(a = a, b = b)
    >>> y = ivy.conv3d_transpose(x, filters, 1, 'VALID', dilations=1)
    >>> print(y.shape)
    [1, 8, 8, 8, 1]


    >>> x = ivy.full((1, 6, 6, 6, 1), 1.23)
    >>> a =  ivy.array(ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 1, 1]))
    >>> b =  ivy.array(ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 1, 1]))
    >>> filters = ivy.Container(a = a, b = b)
    >>> y = ivy.conv3d_transpose(x, filters, 1, 'VALID', dilations=1)
    >>> print(y.shape)
    [1, 8, 8, 8, 1]
    """
    return current_backend(x).conv3d_transpose(
        x,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        bias=bias,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def conv_general_dilated(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, int, Sequence[Tuple[int, int]]],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    filter_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 1-D, 2-D, and 3-D convolution given 3-D, 4-D and 5-D input x respectively
    and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,d,h,w,d_in]* or *[batch_size,d_in,d,h,w]*.
    filters
        Convolution filters *[fd,fh,fw,d_in/feature_group_count,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
        padding), or a sequence of n (low, high) integer pairs that give the padding to
        apply before and after each spatial dimension.
    dims
        Either 1, 2, or 3 corresponding to 1-D, 2-D, and 3-D convolution.
    data_format
        Either "channel_first" or "channel_last". "channel_first" corresponds to "NCW",
        "NCHW", "NCDHW" input data formatS for 1-D, 2-D, 3-D convolution respectively,
        while "channel_last" corresponds to "NWC", "NHWC", "NDHWC" respectively.
    filter_format
        Either "channel_first" or "channel_last". "channel_first" corresponds to "OIW",
        "OIHW", "OIDHW" input data formats for 1-D, 2-D, 3-D convolution respectively,
        while "channel_last" corresponds to "WIO", "HWIO", "DHWIO" respectively.
    feature_group_count
         split input into groups, d_in should be divisible by the number of groups.
         (Default value = 1)
    x_dilations
        The dilation factor for each dimension of input. (Default value = 1)
    dilations
        The dilation factor for each dimension of filter. (Default value = 1)
    bias
        Bias array of shape *[d_out]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose convolution operation.
    """
    return current_backend(x).conv_general_dilated(
        x,
        filters,
        strides,
        padding,
        dims=dims,
        data_format=data_format,
        filter_format=filter_format,
        feature_group_count=feature_group_count,
        x_dilations=x_dilations,
        dilations=dilations,
        bias=bias,
        out=out,
    )


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@handle_out_argument
@inputs_to_native_shapes
@to_native_arrays_and_back
@handle_array_function
@handle_device_shifting
def conv_general_transpose(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    dims: int = 2,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "channel_last",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 1-D, 2-D, and 3-D transpose convolution given 3-D, 4-D and 5-D input x
    respectively and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,d,h,w,d_in]* or *[batch_size,d_in,d,h,w]*.
    filters
        Convolution filters *[fd,fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        Either ‘SAME’ (padding so that the output's shape is the same as the
        input's), or ‘VALID’ (padding so that the output's shape is `output_shape`).
    dims
        Either 1, 2, or 3 corresponding to 1-D, 2-D, and 3-D convolution.
    output_shape
        Shape of the output.
    data_format
        Either "channel_first" or "channel_last". "channel_first" corresponds to "NCW",
        "NCHW", "NCDHW" input data formatS for 1-D, 2-D, 3-D convolution respectively,
        while "channel_last" corresponds to "NWC", "NHWC", "NDHWC" respectively.
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    feature_group_count
         split input into groups, d_in should be divisible by the number of groups.
    bias
        Bias array of shape *[d_out]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose convolution operation.
    """
    return current_backend(x).conv_general_transpose(
        x,
        filters,
        strides,
        padding,
        dims=dims,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        feature_group_count=feature_group_count,
        bias=bias,
        out=out,
    )


@handle_exceptions
@handle_array_like_without_promotion
@handle_out_argument
@handle_array_function
@inputs_to_native_shapes
def conv(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, Sequence[Tuple[int, int]]],
    /,
    *,
    transpose: bool = False,
    dims: int = 2,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "channel_last",
    filter_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Compute a 1-D, 2-D, and 3-D transpose or dilated convolution given 3-D, 4-D and 5-D
    input x respectively and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,d,h,w,d_in]* or *[batch_size,d_in,d,h,w]*.
    filters
        Convolution filters *[fd,fh,fw,d_in/feature_group_count,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        either the string ‘SAME’ (padding with zeros evenly), the string ‘VALID’ (no
        padding), or a sequence of n (low, high) integer pairs that give the padding to
        apply before and after each spatial dimension.
    transpose
        True for computing transpose convolution, and False for dilated convolution.
        When True, `x_dilations` must be 1 (the default).
    dims
        Either 1, 2, or 3 corresponding to 1-D, 2-D, and 3-D convolution.
    output_shape
        Shape of the output (Default value = None)
    data_format
        Either "channel_first" or "channel_last". "channel_first" corresponds to "NCW",
        "NCHW", "NCDHW" input data formatS for 1-D, 2-D, 3-D convolution respectively,
        while "channel_last" corresponds to "NWC", "NHWC", "NDHWC" respectively.
    filter_format
        Either "channel_first" or "channel_last". "channel_first" corresponds to "OIW",
        "OIHW", "OIDHW" input data formats for 1-D, 2-D, 3-D convolution respectively,
        while "channel_last" corresponds to "WIO", "HWIO", "DHWIO" respectively.
    feature_group_count
         split input into groups, d_in should be divisible by the number of groups.
         (Default value = 1)
    x_dilations
        The dilation factor for each dimension of input. (Default value = 1)
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    bias
        Bias array of shape *[d_out]*.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose or dilated convolution operation.
    """
    if transpose:
        assert x_dilations == 1, "x_dilations must be 1 for transpose convolutions."
        return conv_general_transpose(
            x,
            filters,
            strides,
            padding,
            dims=dims,
            output_shape=output_shape,
            data_format=data_format,
            dilations=dilations,
            feature_group_count=feature_group_count,
            bias=bias,
            out=out,
        )
    else:
        return conv_general_dilated(
            x,
            filters,
            strides,
            padding,
            dims=dims,
            data_format=data_format,
            filter_format=filter_format,
            feature_group_count=feature_group_count,
            x_dilations=x_dilations,
            dilations=dilations,
            bias=bias,
            out=out,
        )


# LSTM #


@handle_exceptions
@handle_nestable
@handle_array_like_without_promotion
@inputs_to_ivy_arrays
@handle_array_function
def lstm_update(
    x: Union[ivy.Array, ivy.NativeArray],
    init_h: Union[ivy.Array, ivy.NativeArray],
    init_c: Union[ivy.Array, ivy.NativeArray],
    kernel: Union[ivy.Array, ivy.NativeArray],
    recurrent_kernel: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    recurrent_bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Tuple[ivy.Array, ivy.Array]:
    """
    Perform long-short term memory update by unrolling time dimension of input array.

    Parameters
    ----------
    x
        input tensor of LSTM layer *[batch_shape, t, in]*.
    init_h
        initial state tensor for the cell output *[batch_shape, out]*.
    init_c
        initial state tensor for the cell hidden state *[batch_shape, out]*.
    kernel
        weights for cell kernel *[in, 4 x out]*.
    recurrent_kernel
        weights for cell recurrent kernel *[out, 4 x out]*.
    bias
        bias for cell kernel *[4 x out]*. (Default value = None)
    recurrent_bias
        bias for cell recurrent kernel *[4 x out]*. (Default value = None)

    Returns
    -------
    ret
        hidden state for all timesteps *[batch_shape,t,out]* and cell state for last
        timestep *[batch_shape,out]*
    """
    # get shapes
    x_shape = list(x.shape)
    batch_shape = x_shape[:-2]
    timesteps = x_shape[-2]
    input_channels = x_shape[-1]
    x_flat = ivy.reshape(x, (-1, input_channels))

    # input kernel
    Wi = kernel
    Wi_x = ivy.reshape(
        ivy.matmul(x_flat, Wi) + (bias if bias is not None else 0),
        batch_shape + [timesteps, -1],
    )
    Wii_x, Wif_x, Wig_x, Wio_x = ivy.split(Wi_x, num_or_size_splits=4, axis=-1)

    # recurrent kernel
    Wh = recurrent_kernel

    # lstm states
    ht = init_h
    ct = init_c

    # lstm outputs
    hts_list = list()

    # unrolled time dimension with lstm steps
    for Wii_xt, Wif_xt, Wig_xt, Wio_xt in zip(
        ivy.unstack(Wii_x, axis=-2),
        ivy.unstack(Wif_x, axis=-2),
        ivy.unstack(Wig_x, axis=-2),
        ivy.unstack(Wio_x, axis=-2),
    ):
        htm1 = ht
        ctm1 = ct

        Wh_htm1 = ivy.matmul(htm1, Wh) + (
            recurrent_bias if recurrent_bias is not None else 0
        )
        Whi_htm1, Whf_htm1, Whg_htm1, Who_htm1 = ivy.split(
            Wh_htm1, num_or_size_splits=4, axis=-1
        )

        it = ivy.sigmoid(Wii_xt + Whi_htm1)
        ft = ivy.sigmoid(Wif_xt + Whf_htm1)
        gt = ivy.tanh(Wig_xt + Whg_htm1)
        ot = ivy.sigmoid(Wio_xt + Who_htm1)
        ct = ft * ctm1 + it * gt
        ht = ot * ivy.tanh(ct)

        hts_list.append(ivy.expand_dims(ht, axis=-2))

    return ivy.concat(hts_list, axis=-2), ct


# Helpers #


def _handle_padding(x, strides, filters, padding):
    if padding == "SAME":
        if x % strides == 0:
            pad = max(filters - strides, 0)
        else:
            pad = max(filters - (x % strides), 0)
    else:
        pad = 0
    return pad


def _deconv_length(dim_size, stride_size, kernel_size, padding, dilation=1):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    if padding == "SAME":
        dim_size = dim_size * stride_size
    else:
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    return dim_size


def _get_x_data_format(dims: int = 2, data_format: str = "channel_first"):
    if dims == 1:
        if data_format == "channel_first":
            return "NCW"
        else:
            return "NWC"
    if dims == 2:
        if data_format == "channel_first":
            return "NCHW"
        else:
            return "NHWC"
    elif dims == 3:
        if data_format == "channel_first":
            return "NCDHW"
        else:
            return "NDHWC"


def _get_num_padded_values(i, p, n, k, s):
    """
    Get number of padded values in a specific window.

    Parameters
    ----------
    i window index
    p total amount of padding
    n input size
    k kernel size
    s stride

    Returns
    -------
        number of padded values in a particular window represented by i
    """
    current_index = s * i
    left_padding = p // 2
    return max(0, left_padding - current_index) + max(
        0, current_index + k - n - left_padding
    )
