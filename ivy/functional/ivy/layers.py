"""Collection of Ivy neural network layers in functional form."""

# global
from typing import Optional, Tuple, Union, List, Callable

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like,
)
from ivy.exceptions import handle_exceptions


# Extra #
# ------#


# Linear #


@handle_exceptions
@handle_array_like
def linear(
    x: Union[ivy.Array, ivy.NativeArray],
    weight: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Applies a linear transformation to the incoming data: y = x * t(weight) + bias.
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


# Dropout #


@handle_exceptions
@handle_array_like
def dropout(
    x: Union[ivy.Array, ivy.NativeArray],
    prob: float,
    /,
    *,
    scale: bool = True,
    dtype: ivy.Dtype = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Randomly zeroes some elements of the input tensor with probability p using
    samples from a Bernoulli distribution.

    Parameters
    ----------
    x
        The input array x to perform dropout on.
    prob
        The probability of zeroing out each array element.
    scale
        Whether to scale the output by 1/(1-prob), default is ``True``.
    dtype

    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        Result array of the linear transformation. *[N,∗,out_features]*

    """
    x = ivy.where(
        ivy.random_uniform(shape=x.shape, device=ivy.dev(x), dtype=dtype) < prob,
        ivy.zeros_like(x, dtype=dtype),
        x,
    )
    if scale:
        x = ivy.multiply(x, 1 / (1 - prob), out=out)
    if ivy.exists(out):
        return ivy.inplace_update(out, x)
    return x


@handle_exceptions
@to_native_arrays_and_back
@handle_array_like
def dropout1d(
    x: Union[ivy.Array, ivy.NativeArray],
    prob: float,
    /,
    *,
    training: bool = True,
    data_format: str = "NWC",
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Randomly zero out entire channels with probability prob using samples from
     a Bernoulli distribution and the remaining channels are scaled by (1/1-prob).
     In this case, dropout1d performs a channel-wise dropout but assumes
     a channel is a 1D feature map.

    Parameters
    ----------
    x
        a 2D or 3D input array. Should have a floating-point data type.
    prob
        probability of a channel to be zero-ed.
    training
        controls whether dropout1d is performed during training or ignored
        during testing.
    data_format
        "NWC" or "NCW". Defaults to "NWC".
    out
        optional output array, for writing the result to.
        It must have a shape that the inputs broadcast to.

    Returns
    -------
    ret
        an array with some channels zero-ed and the rest of channels are
         scaled by (1/1-prob).
    """
    return current_backend(x).dropout1d(
        x, prob, training=training, data_format=data_format, out=out
    )


# Attention #


@handle_exceptions
@handle_array_like
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
    """Applies scaled dot product attention to inputs x using optional mask.

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

    Functional Examples
    -------------------

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

    With :class:`ivy.NativeArray` input:

    >>> q = ivy.native_array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]])
    >>> k = ivy.native_array([[[0.6, 1.5], [2.4, 3.3],[4.2, 5.1]]])
    >>> v = ivy.native_array([[[0.4, 1.3], [2.2, 3.1],[4.3, 5.3]]])
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1)
    >>> print(result)
    ivy.array([[[4.04,5.03],[4.3,5.3],[4.3,5.3]]])

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
    {a:ivy.array([[[4.27,5.4],[4.4,5.6],[4.4,5.6]]]),b:ivy.array([[[4.35,5.54],[4.4,5.6],[4.4,5.6]]])}


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


    Instance Method Examples
    ------------------------

    With :class:`ivy.Array` input:

    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]])
    >>> k = ivy.array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
    >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
    >>> mask = ivy.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    >>> result = ivy.scaled_dot_product_attention(q, k, v, 1, mask=mask)
    >>> print(result)
    ivy.array([[[2.3, 3.23],[2.3, 3.23],[2.3, 3.23]]])

    >>> q = ivy.array([[[0.2, 1.], [2.2, 3.], [4.4, 5.6]]])
    >>> k = ivy.array([[[0.6, 1.5], [2.4, 3.3], [4.2, 5.1]]])
    >>> v = ivy.array([[[0.4, 1.3], [2.2, 3.1], [4.3, 5.3]]])
    >>> mask = ivy.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    >>> out = ivy.zeros(shape=(1, 3, 2))
    >>> ivy.scaled_dot_product_attention(q, k, v, 1, mask=mask, out=out)
    >>> print(out)
    ivy.array([[[2.3, 3.23],[2.3, 3.23],[2.3, 3.23]]])

    With :class:`ivy.Container` input:

    >>> q = ivy.Container(a=ivy.array([[[0.2, 1.], [2.7, 3.], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[1.2, 1.], [2.2, 3.], [4.4, 5.6]]]))
    >>> k = ivy.Container(a=ivy.array([[[4.2, 1.], [2.2, 3.3],[4.4, 5.6]]]),
    ...                   b=ivy.array([[[3.2, 1.], [2.2, 3.6], [4.0, 5.6]]]))
    >>> v = ivy.Container(a=ivy.array([[[5.2, 1.], [2.1, 3.],[4.4, 5.6]]]),
    ...                   b=ivy.array([[[0.2, 1.], [2.2, 3.],[4.4, 5.6]]]))
    >>> mask =
    ... ivy.Container(a=ivy.array([[[1.0, 1.0, 1.0],[1.0, 1.0, 1.0],[1.0, 1.0,1.0]]]),
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


@handle_exceptions
@handle_array_like
def multi_head_attention(
    x: Union[ivy.Array, ivy.NativeArray],
    scale: float,
    num_heads: int,
    /,
    *,
    context: Union[ivy.Array, ivy.NativeArray] = None,
    mask: Union[ivy.Array, ivy.NativeArray] = None,
    to_q_fn: Callable = None,
    to_kv_fn: Callable = None,
    to_out_fn: Callable = None,
    to_q_v=None,
    to_kv_v=None,
    to_out_v=None,
    out: Optional[ivy.Array] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Applies multi-head attention to inputs x.

    Parameters
    ----------
    x
        The array to determine the queries from *[batch_shape,num_queries,query_dim]*.
    scale
        The value by which to scale the query-key similarity measure before softmax.
    num_heads
        The number of attention heads to use.
    context
        The array to determine the keys and values from. Default is ``None``.
        *[batch_shape,num_keys,cont_feat_dim]*.
    mask
        The mask to apply to the query-key values. Default is ``None``.
        *[batch_shape,num_queries,num_keys]*
    to_q_fn
        The function to compute queries from input x, returning queries
        *[batch_shape,num_queries,numheads×head_dim]*. (Default value = None)
    to_kv_fn
        The function to compute keys and values from the context. (Default value = None)
    to_out_fn
        The function to compute the output from the scaled dot-product attention.
        (Default value = None)
    to_q_v
        The variables for function to_q_fn. Default is ``None``.
    to_kv_v
        The variables for function to_kv_fn. Default is ``None``.
    to_out_v
        The variables for function to_out_fn. Default is ``None``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The output following application of multi-head attention.
        *[batch_shape,num_queries,out_feat_dim]*

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.array([[[0.2, 1.],
    ...                 [2.2, 3.],
    ...                 [4.4, 5.6]]])
    >>> context = ivy.array([[[0.2, 1., 1.1, 4.2],
    ...                       [2.2, 3., 0.9, 3.6],
    ...                       [4.4, 5.6, 2.2, 0.4]]])
    >>> result = ivy.multi_head_attention(x, 1, 2, context=context)
    >>> print(result)
    ivy.array([[[1.5678761 , 0.65441847],
    ...         [2.18969631, 0.40131447],
    ...         [2.19991851, 0.40000153]]])

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[[0.2, 1.],
    ...                        [2.2, 3.],
    ...                        [4.4, 5.6]]])
    >>> context = ivy.native_array([[[0.2, 1., 1.1, 4.2],
    ...                              [2.2, 3., 0.9, 3.6],
    ...                              [4.4, 5.6, 2.2, 0.4]]])
    >>> result = ivy.multi_head_attention(x, 1, 2, context=context)
    >>> print(result)
    ivy.array([[[1.5678761 , 0.65441847],
    ...         [2.18969631, 0.40131447],
    ...         [2.19991851, 0.40000153]]])

    With :class:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([[[0.2, 1.1], [2.2, 3.4], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[1.4, 0.3], [1.2, 3.9], [0.4, 3.7]]]))
    >>> context = ivy.Container(a=ivy.array([[[0.2, 1.8, 1.1, 4.2],
    ...                                       [2.2, 3.3, 0.9, 3.6],
    ...                                       [4.4, 5.6, 2.2, 0.4]]]),
    ...                         b=ivy.array([[[1.4, 0.3, 4.4, 5.6],
    ...                                       [1.2, 3.9, 4.2, 5.1],
    ...                                       [0.4, 3.7, 4.3, 5.3]]]))
    >>> result = ivy.multi_head_attention(x, 1, 2, context=context)
    >>> print(result)
    {
        a: ivy.array([[[1.5678761, 0.68589532],
                       [2.18969631, 0.40129396],
                       [2.19991851, 0.40000817]]]),
        b: ivy.array([[[4.31219625, 5.25698996],
                       [4.31022024, 5.16286421],
                       [4.30296469, 5.16460133]]])
    }

    With a mix of :class:`ivy.Container` and :class:`ivy.Array` inputs:

    >>> x = ivy.Container(a=ivy.array([[[0.2, 1.1], [2.2, 3.4], [4.4, 5.6]]]),
    ...                   b=ivy.array([[[1.4, 0.3], [1.2, 3.9], [0.4, 3.7]]]))
    >>> context = ivy.array([[[0.2, 1., 1.1, 4.2],
    ...                       [2.2, 3., 0.9, 3.6],
    ...                       [4.4, 5.6, 2.2, 0.4]]])
    >>> result = ivy.multi_head_attention(x, 1, 2, context=context)
    >>> print(result)
    {
        a: ivy.array([[[1.5678761, 0.59497029],
                       [2.18969631, 0.40046397],
                       [2.19991851, 0.40000153]]]),
        b: ivy.array([[[2.14009905, 1.81691194],
                       [2.10732293, 0.40012637],
                       [1.73519301, 0.40021262]]])
    }

    With a mix of :class:`ivy.Array` and :class:`ivy.Container` inputs:

    >>> x = ivy.array([[[0.2, 1.],
    ...                 [2.2, 3.],
    ...                 [4.4, 5.6]]])
    >>> context = ivy.Container(a=ivy.array([[[0.2, 1.8, 1.1, 4.2],
    ...                                       [2.2, 3.3, 0.9, 3.6],
    ...                                       [4.4, 5.6, 2.2, 0.4]]]),
    ...                         b=ivy.array([[[1.4, 0.3, 4.4, 5.6],
    ...                                       [1.2, 3.9, 4.2, 5.1],
    ...                                       [0.4, 3.7, 4.3, 5.3]]]))
    >>> result = ivy.multi_head_attention(x, 1, 2, context=context)
    >>> print(result)
    {
        a: ivy.array([[[1.5678761, 0.7615059],
                       [2.18969631, 0.40326414],
                       [2.19991851, 0.40000817]]]),
        b: ivy.array([[[4.30141067, 5.19610119],
                       [4.32028484, 5.1708746],
                       [4.34100914, 5.14920235]]])
    }

    With :class:`ivy.Array` inputs and :class:`ivy.Array` mask:

    >>> x = ivy.array([[[0.2, 1.],
    ...                 [2.2, 3.],
    ...                 [4.4, 5.6]]])
    >>> context = ivy.array([[[0.2, 1., 1.1, 4.2],
    ...                       [2.2, 3., 0.9, 3.6],
    ...                       [4.4, 5.6, 2.2, 0.4]]])
    >>> mask = ivy.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    >>> result = ivy.multi_head_attention(x, 1, 2, context=context, mask=mask)
    >>> print(result)
    ivy.array([[[1.40000009, 2.73333335],
    ...         [1.40000009, 2.73333335],
    ...         [1.40000009, 2.73333335]]])

    With :class:`ivy.Array` inputs and lambda to_q_fn and to_kv_fn functions specified:

    >>> x = ivy.array([[[0.2, 1.],
    ...                 [2.2, 3.],
    ...                 [4.4, 5.6]]])
    >>> context = ivy.array([[[0.2, 1., 1.1, 4.2],
    ...                       [2.2, 3., 0.9, 3.6],
    ...                       [4.4, 5.6, 2.2, 0.4]]])
    >>> to_q_fn = lambda n, v: n
    >>> to_kv_fn = lambda n, v: ivy.split(n, num_or_size_splits=2, axis=-1)
    >>> result = layers.multi_head_attention(x, 1, 2, context=context,
    ...                                      to_q_fn=to_q_fn, to_kv_fn=to_kv_fn)
    >>> print(result)
    ivy.array([[[1.5678761 , 0.65441847],
    ...         [2.18969631, 0.40131447],
    ...         [2.19991851, 0.40000153]]])


    """
    # BS x Q x (HxF)
    q = to_q_fn(x, v=to_q_v) if ivy.exists(to_q_fn) else x

    # BS x K x CF
    context = ivy.default(context, x)

    # BS x K x (2xHxF)    or    BS x K x (HxF),  BS x K x (HxF)

    if ivy.exists(to_kv_fn):
        kv = to_kv_fn(context, v=to_kv_v)
    else:
        kv = ivy.split(context, num_or_size_splits=2, axis=-1)

    # BS x K x (HxF),  BS x K x (HxF)
    if isinstance(kv, (tuple, list)):
        k, v = kv
    else:
        k, v = ivy.split(kv, num_or_size_splits=2, axis=-1)

    # BS x H x Q x F,  BS x H x K x F,  BS x H x K x F
    q, k, v = map(
        lambda t: ivy.einops_rearrange(t, "... n (h f) -> ... h n f", h=num_heads),
        (q, k, v),
    )

    # BS x H x Q x K
    if ivy.exists(mask):
        mask = ivy.einops_repeat(mask, "... q k -> ... h q k", h=num_heads)

    # BS x H x Q x F
    sdpa = ivy.scaled_dot_product_attention(q, k, v, scale, mask=mask)

    # BS x Q x (HxF)
    sdpa = ivy.einops_rearrange(sdpa, "... h q f -> ... q (h f)")

    # BS x Q x OF
    ret = to_out_fn(sdpa, v=to_out_v) if ivy.exists(to_out_fn) else sdpa
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


# Convolutions #


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def conv1d(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: int,
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    dilations: int = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 1-D convolution given 3-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,w,d_in]*.
    filters
        Convolution filters *[fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    data_format
        NWC" or "NCW". Defaults to "NWC".
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the convolution operation.

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
        dilations=dilations,
        out=out,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def conv1d_transpose(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: int,
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "NWC",
    dilations: int = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 1-D transpose convolution given 3-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,w,d_in]*.
    filters
        Convolution filters *[fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    output_shape
        Shape of the output (Default value = None)
    data_format
        NWC" or "NCW". Defaults to "NWC".
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose convolution operation.

    """
    return current_backend(x).conv1d_transpose(
        x,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        out=out,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_array_like
def conv2d(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 2-D convolution given 4-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,h,w,d_in]*.
    filters
        Convolution filters *[fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    data_format
        NHWC" or "NCHW". Defaults to "NHWC".
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
        dilations=dilations,
        out=out,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def conv2d_transpose(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "NHWC",
    dilations: Union[int, Tuple[int], Tuple[int, int]] = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 2-D transpose convolution given 4-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,h,w,d_in]*.
    filters
        Convolution filters *[fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    output_shape
        Shape of the output (Default value = None)
    data_format
        NHWC" or "NCHW". Defaults to "NHWC".
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose convolution operation.

    """
    return current_backend(x).conv2d_transpose(
        x,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        out=out,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def depthwise_conv2d(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: Union[str, List[int]],
    /,
    *,
    data_format: str = "NHWC",
    dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Computes a 2-D depthwise convolution given 4-D input ``x`` and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,h,w,d]*.
    filters
        Convolution filters *[fh,fw,d_in]*. (d_in must be the same as d from x)
    strides
        The stride of the sliding window for each dimension of input.
    padding
        "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    data_format
        "NHWC" or "NCHW". Defaults to "NHWC".
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


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def conv3d(
    x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    filters: Union[ivy.Array, ivy.NativeArray, ivy.Container],
    strides: Union[int, Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    dilations: Optional[Union[int, Tuple[int, int, int]]] = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 3-D convolution given 5-D input x and filters arrays.

    Parameters
    ----------
    x
        Input volume *[batch_size,d,h,w,d_in]*.
    filters
        Convolution filters *[fd,fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    data_format
        NDHWC" or "NCDHW". Defaults to "NDHWC".
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
    >>> print(result.shapes)
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
        dilations=dilations,
        out=out,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def conv3d_transpose(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, List[int]],
    /,
    *,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "NDHWC",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 3-D transpose convolution given 5-D input x and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,d,h,w,d_in]*.
    filters
        Convolution filters *[fd,fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    output_shape
        Shape of the output (Default value = None)
    data_format
        "NDHWC" or "NCDHW". Defaults to "NDHWC".
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the transpose convolution operation.

    Functional Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3])
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6])
    >>> y = ivy.conv3d_transpose(x, filters, 2, 'SAME')
    >>> print(y.shape)
    (1, 6, 56, 56, 6)

    With :class:`ivy.NativeArray` input:

    >>> x = ivy.native_array(
    ...    ivy.random_normal(mean=0, std=1, shape=[1, 7, 256, 256, 64])
    ... )
    >>> filters = ivy.native_array(
    ...    ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 64, 32])
    ... )
    >>> y = ivy.conv3d_transpose(x, filters, [1, 1, 1], 'VALID')
    >>> print(y.shape)
    (1, 9, 258, 258, 32)

    With :class:`ivy.Container` inputs:

    >>> x = ivy.Container(a = ivy.random_normal(
    ...                       mean=0, std=1, shape=[1, 3, 28, 28, 3]
    ...                       ),
    b = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3]))
    >>> filters = ivy.Container(c = ivy.random_normal(
    ...                             mean=0, std=1, shape=[3, 3, 3, 3, 6]
    ...                             ),
    d = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6]))
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


    With a mix of :class:`ivy.Array`, :class:`ivy.NativeArray`
    and :class:`ivy.Container` inputs:

    >>> x = ivy.full((1, 6, 6, 6, 1), 1.23)
    >>> a =  ivy.native_array(ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 1, 1]))
    >>> b =  ivy.native_array(ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 1, 1]))
    >>> filters = ivy.Container(a = a, b = b)
    >>> y = ivy.conv3d_transpose(x, filters, 1, 'VALID', dilations=1)
    >>> print(y.shape)
    [1, 8, 8, 8, 1]

    Instance Method Examples
    ------------------------

    Using :class:`ivy.Array` instance method:

    >>> x = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3])
    >>> filters = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 6])
    >>> y = x.conv3d_transpose(filters, 2, 'SAME')
    >>> print(y.shape)
    (1, 6, 56, 56, 6)

    Using :class:`ivy.Container` instance method:

    >>> x = ivy.Container(a = ivy.random_normal(
    ...                            mean=0, std=1, shape=[1, 3, 28, 28, 3]
    ...                          ),
    b = ivy.random_normal(mean=0, std=1, shape=[1, 3, 28, 28, 3]))

    >>> filters = ivy.Container(c = ivy.random_normal(
    ...                                 mean=0, std=1, shape=[3, 3, 3, 3, 3]
    ...                             ),
    d = ivy.random_normal(mean=0, std=1, shape=[3, 3, 3, 3, 3]))

    >>> y = x.conv3d_transpose(filters, 2, "SAME")
    >>> print(y.shape)
    (1, 6, 56, 56, 3)
    """
    return current_backend(x).conv3d_transpose(
        x,
        filters,
        strides,
        padding,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        out=out,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def conv_general_dilated(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, List[int]],
    /,
    *,
    dims: int = 2,
    data_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 1-D, 2-D, and 3-D convolution given 3-D, 4-D and 5-D
    input x respectively and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,d,h,w,d_in]*.
    filters
        Convolution filters *[fd,fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    dims
        Shape of input.
    data_format
        "channel_first" or "channel_last" Defaults to "channel_last"
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
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
        feature_group_count=feature_group_count,
        x_dilations=x_dilations,
        dilations=dilations,
        out=out,
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
@handle_array_like
def conv_general_transpose(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, List[int]],
    /,
    *,
    dims: int = 2,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "channel_last",
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    feature_group_count: int = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes a 1-D, 2-D, and 3-D transpose convolution given 3-D, 4-D and 5-D
    input x respectively and filters arrays.

    Parameters
    ----------
    x
        Input image *[batch_size,d,h,w,d_in]*.
    filters
        Convolution filters *[fd,fh,fw,d_in,d_out]*.
    strides
        The stride of the sliding window for each dimension of input.
    padding
        "SAME" or "VALID" indicating the algorithm, or list indicating the per-dimension
        paddings.
    dims
        Shape of input.
    data_format
        "channel_first" or "channel_last" Defaults to "channel_last"
    dilations
        The dilation factor for each dimension of input. (Default value = 1)
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
        out=out,
    )


@handle_out_argument
@handle_exceptions
@handle_array_like
def conv(
    x: Union[ivy.Array, ivy.NativeArray],
    filters: Union[ivy.Array, ivy.NativeArray],
    strides: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    padding: Union[str, List[int]],
    /,
    *,
    transpose: bool = False,
    dims: int = 2,
    output_shape: Optional[Union[ivy.Shape, ivy.NativeShape]] = None,
    data_format: str = "channel_last",
    feature_group_count: int = 1,
    x_dilations: Union[int, Tuple[int], Tuple[int, int]] = 1,
    dilations: Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]] = 1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
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
            feature_group_count=feature_group_count,
            x_dilations=x_dilations,
            dilations=dilations,
            out=out,
        )


# LSTM #


@to_native_arrays_and_back
@handle_exceptions
@handle_array_like
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
    """Perform long-short term memory update by unrolling time dimension of input array.

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


def handle_padding(x, strides, filters, padding):
    if padding == "SAME":
        if x % strides == 0:
            pad = max(filters - strides, 0)
        else:
            pad = max(filters - (x % strides), 0)
    else:
        pad = 0
    return pad


def deconv_length(dim_size, stride_size, kernel_size, padding, dilation=1):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    if padding == "VALID":
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == "SAME":
        dim_size = dim_size * stride_size
    return dim_size


def get_x_data_format(dims: int = 2, data_format: str = "channel_first"):
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


@to_native_arrays_and_back
@handle_out_argument
@handle_exceptions
@handle_array_like
def fft(
    x: Union[ivy.Array, ivy.NativeArray],
    dim: int,
    /,
    *,
    norm: Optional[str] = "backward",
    n: Optional[Union[int, Tuple[int]]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    r"""Computes the one dimensional discrete Fourier transform given input at least
    1-D input x.

    Parameters
    ----------
    x
        Input volume *[...,d_in,...]*,
        where d_in indicates the dimension that needs FFT.
    dim
        The dimension along which to take the one dimensional FFT.
    norm
        Optional argument, "backward", "ortho" or "forward". Defaults to be "backward".
        "backward" indicates no normalization.
        "ortho" indicates normalization by $\frac{1}{\sqrt{n}}$.
        "forward" indicates normalization by $\frac{1}{n}$.
    n
        Optional argument indicating the sequence length, if given, the input would be
        padded with zero or truncated to length n before performing FFT.
        Should be a integer greater than 1.
    out
        Optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The result of the FFT operation.

    Examples
    --------
    >>> ivy.fft(np.exp(2j * np.pi * np.arange(8) / 8), 0)
    ivy.array([-3.44509285e-16+1.14423775e-17j,  8.00000000e+00-8.11483250e-16j,
            2.33486982e-16+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j,
            9.95799250e-17+2.33486982e-16j,  0.00000000e+00+7.66951701e-17j,
            1.14423775e-17+1.22464680e-16j,  0.00000000e+00+1.22464680e-16j])
    >>> ivy.fft(np.exp(2j * np.pi * np.arange(8) / 8), 0, n=16)
    ivy.array([-3.44509285e-16+1.14423775e-17j,  1.00000000e+00+5.02733949e+00j,
        8.00000000e+00-8.11483250e-16j,  1.00000000e+00-5.02733949e+00j,
        2.33486982e-16+1.22464680e-16j,  1.00000000e+00-1.49660576e+00j,
        0.00000000e+00+1.22464680e-16j,  1.00000000e+00-6.68178638e-01j,
        9.95799250e-17+2.33486982e-16j,  1.00000000e+00-1.98912367e-01j,
        0.00000000e+00+7.66951701e-17j,  1.00000000e+00+1.98912367e-01j,
        1.14423775e-17+1.22464680e-16j,  1.00000000e+00+6.68178638e-01j,
        0.00000000e+00+1.22464680e-16j,  1.00000000e+00+1.49660576e+00j])
    >>> ivy.fft(np.exp(2j * np.pi * np.arange(8) / 8), 0, norm="ortho")
    ivy.array([-1.21802426e-16+4.04549134e-18j,  2.82842712e+00-2.86902654e-16j,
        8.25501143e-17+4.32978028e-17j,  0.00000000e+00+4.32978028e-17j,
        3.52068201e-17+8.25501143e-17j,  0.00000000e+00+2.71158374e-17j,
        4.04549134e-18+4.32978028e-17j,  0.00000000e+00+4.32978028e-17j])
    """
    return current_backend(x).fft(x, dim, norm=norm, n=n, out=out)
