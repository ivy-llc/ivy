from typing import Union, Optional, Tuple, Sequence
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_math_ops
import ivy
from ivy import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)
from .. import backend_version


def histogram(
    a: tf.Tensor,
    /,
    *,
    bins: Optional[Union[int, tf.Tensor]] = None,
    axis: Optional[int] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[tf.DType] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[tf.Tensor] = None,
    density: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor]:
    min_a = tf.reduce_min(a)
    max_a = tf.reduce_max(a)
    if isinstance(bins, tf.Tensor) and range:
        raise ivy.exceptions.IvyException(
            "Must choose between specifying bins and range or bin edges directly"
        )
    if range:
        if isinstance(bins, int):
            bins = tf.cast(
                tf.linspace(start=range[0], stop=range[1], num=bins + 1), dtype=a.dtype
            )
    elif isinstance(bins, int):
        range = (min_a, max_a)
        bins = tf.cast(
            tf.linspace(start=range[0], stop=range[1], num=bins + 1), dtype=a.dtype
        )
    if tf.shape(bins)[0] < 2:
        raise ivy.exceptions.IvyException("bins must have at least 1 bin (size > 1)")
    if min_a < bins[0] and not extend_lower_interval:
        raise ivy.exceptions.IvyException(
            "Values of x outside of the intervals cause errors in tensorflow backend. "
            "Consider using extend_lower_interval to deal with this."
        )
    if max_a > bins[-1] and not extend_upper_interval:
        raise ivy.exceptions.IvyException(
            "Values of x outside of the intervals cause errors in tensorflow backend. "
            "Consider using extend_upper_interval to deal with this."
        )
    ret = tfp.stats.histogram(
        x=a,
        edges=bins,
        axis=axis,
        weights=weights,
        extend_lower_interval=extend_lower_interval,
        extend_upper_interval=extend_upper_interval,
        dtype=dtype,
        name="histogram",
    )
    if density:
        pass
    # TODO: Tensorflow native dtype argument is not working
    if dtype:
        ret = tf.cast(ret, dtype)
        bins = tf.cast(bins, dtype)
    # TODO: weird error when returning bins: return ret, bins
    return ret


@with_supported_dtypes(
    {
        "2.13.0 and below": (
            "float",
            "complex",
        )
    },
    backend_version,
)
def median(
    input: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tfp.stats.percentile(
        input,
        50.0,
        axis=axis,
        interpolation="midpoint",
        keepdims=keepdims,
    )


def nanmean(
    a: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    np_math_ops.enable_numpy_methods_on_tensor()
    return tf.experimental.numpy.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype)


def quantile(
    a: Union[tf.Tensor, tf.Variable],
    q: Union[tf.Tensor, float],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    interpolation: str = "linear",
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis

    result = tfp.stats.percentile(
        a,
        tf.math.multiply(q, 100),
        axis=axis,
        interpolation=interpolation,
        keepdims=keepdims,
    )
    return result


def corrcoef(
    x: tf.Tensor,
    /,
    *,
    y: tf.Tensor,
    rowvar: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> tf.Tensor:
    if y is None:
        xarr = x
    else:
        axis = 0 if rowvar else 1
        xarr = tf.concat([x, y], axis=axis)

    if rowvar:
        mean_t = tf.reduce_mean(xarr, axis=1, keepdims=True)
        cov_t = ((xarr - mean_t) @ tf.transpose(xarr - mean_t)) / (x.shape[1] - 1)
    else:
        mean_t = tf.reduce_mean(xarr, axis=0, keepdims=True)
        cov_t = (tf.transpose(xarr - mean_t) @ (xarr - mean_t)) / (x.shape[1] - 1)

    cov2_t = tf.linalg.diag(1 / tf.sqrt(tf.linalg.diag_part(cov_t)))
    cor = cov2_t @ cov_t @ cov2_t
    return cor


def _nanmedian_helper(input, axis=None, keepdims=False):
    """
    The approach to Handle Nans in single dimensional plus multi-dimensional inputs are
    composed on two-parts.

    PART 1:  In this part, you have axis=None, it means we have to work on
    flattened data, we don't need to work on different axis.there are two cases here

    Case 1: which is if our input data does contain all the Nans or not,
    if our input have just Nans (means no numbers) then we'll not use
    temp[~tf.math.is_nan(temp)] function with our input because it will remove all Nans
    and we get empty tensor and this raise an error when it sent to percentile function,
    in this case we need to keep this input but just we flatten the input and percentile
    function returns nan if it find nan in median and here all the input is nan then we
    get our result.

    Case 2: if we have a number (0.4, 0.3, 0. ,1., 2., .....) with nans then we use this
    function temp[~tf.math.is_nan(temp)], it will return a tensor by extracting the nans
    and just keeping the values, but remember the returned tensor will be flattened and
    axis=None work on flattene inputs, so in this case we are also on same page :)

    for example: [[12.0 ,4.0 ,ivy.nan], [ivy.nan, ivy.nan,2.2]] => returned:
    [12.0 ,4.0, 2.2] now this will be our new input in percentile function.

    PART 2: In this case you have to do more work because we now don't allow to work
    directly on flattened data, Here are two cases also.

    CASE 1: we need to consider axis parameter here, but percentile axis does work
    differently and we don't have median function in tensorflow yet, so we need to make
    our input data compatible to the axis, then we compute nanmedian along that specific
    axis. we transpose the input data according to our axis, axis can be (0,), (1,),
    (0,1), (0,1,2) and input can be multi-dimensional, so we need to take care of edge
    cases before making it compatible.

    CASE 2: Here the main Nan handling part comes, you can only use 1D inputs here so we
    have to flatten the input then we have jump parameter which is use to say how many
    iterations we want to make because we have to calculate the row-wise median along
    axis=None now, so we slice out some data from the flattened input and then we use
    that 1D Input to remove the nans and use it in our percentile.

    For example: input = [[ivy.nan, 3, ivy.nan, 7],[4, ivy.nan,6, 9]], axis=1

    flatten data -> [[nan  3. nan  7.  4. nan  6.  9.]]
    num_jumps -> 2 because we have to slice out this in (1, 4) and (1,4),
    then it works same as PART 1 CASE 1 AND CASE 2.
    now for first slice we get -> 5.0 and for second we get -> 6.0, these calculated
    along axis=1 now we append the data into result, so to make the shape of result
    compatible with the numpy output, we reshaped it.

    the result which we get from our _nanmedian_helper = [5., 6.]
    """

    dtype = input.dtype
    temp = tf.cast(input, tf.float64)
    num_dim = tf.rank(temp)
    keepdim_shape = tf.shape(temp)
    q = 50.0

    # PART 1
    if axis is None:
        # PART 1 CASE 1
        if tf.reduce_all(tf.math.is_nan(temp)):
            temp = tf.reshape(temp, shape=(1, -1))
        else:
            # PART 1 CASE 2
            temp = temp[~tf.math.is_nan(temp)]

        ret = tfp.stats.percentile(
            temp,
            q,
            axis=axis,
            interpolation="midpoint",
            keepdims=keepdims,
        )
        if dtype in [tf.int32, tf.int64, tf.float64]:
            ret = tf.cast(ret, dtype=tf.float64)
        elif dtype in [tf.float16, tf.bfloat16]:
            ret = tf.cast(ret, dtype=tf.float16)
        else:
            ret = tf.cast(ret, dtype=tf.float32)
        return ret

    axis = [axis] if isinstance(axis, int) else list(axis)
    # PART 2 CASE 1
    for i in axis:
        keepdim_shape = tf.tensor_scatter_nd_update(keepdim_shape, [[i]], [1])
    axis = [num_dim + x if x < 0 else x for x in axis]
    axis.sort()
    dimension = tf.size(temp.shape)
    while tf.size(axis) > 0:
        axis1 = axis[0]
        for axis2 in range(axis1 + 1, dimension):
            temp = tf.transpose(
                temp,
                perm=tf.tensor_scatter_nd_update(
                    tf.range(tf.rank(temp)), [[axis1], [axis2]], [axis2, axis1]
                ),
            )
            axis1 = axis2
        axis = [x - 1 for x in axis]
        axis.pop(0)
        dimension = dimension - 1
    temp = tf.reshape(
        temp, shape=tf.concat([tf.shape(temp)[: (dimension - len(axis))], [-1]], axis=0)
    )

    tensor = tf.reshape(temp, shape=(1, -1))
    shape = temp.shape
    dim = temp.ndim
    slice_size = shape[len(shape) - 1]
    num_jumps = 1
    result = []

    if slice_size == 1:
        if dim == 2 and input.shape[0] == 1:
            return tensor
        if dim > 2 and input.shape[0] == 1:
            return tf.reshape(tensor, shape=input.shape)

        tensor = tf.reshape(tensor, shape=shape[:-1])
        return tensor
    # PART 2 CASE 2
    i = dim
    while i > 1:
        num_jumps *= shape[len(shape) - i]
        i -= 1

    for i in range(num_jumps):
        start = i * slice_size
        end = (i + 1) * slice_size
        arr = tensor[:, start:end]
        if tf.reduce_all(tf.math.is_nan(arr)):
            arr = tf.reshape(arr, shape=(1, -1))
        else:
            arr = arr[~tf.math.is_nan(arr)]

        ret = tfp.stats.percentile(
            arr, q, axis=None, interpolation="midpoint", keepdims=keepdims
        )
        if keepdims:
            ret = tf.squeeze(ret)

        result.append(ret)

    result = tf.reshape(result, shape=shape[:-1])

    if keepdims:
        keepdim_shape = tuple(keepdim_shape)
        result = tf.reshape(result, shape=keepdim_shape)

    if dtype in [tf.int32, tf.int64, tf.float64]:
        result = tf.cast(result, dtype=tf.float64)
    elif dtype in [tf.float16, tf.bfloat16]:
        result = tf.cast(result, dtype=tf.float16)
    else:
        result = tf.cast(result, dtype=tf.float32)

    return result


def nanmedian(
    input: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if overwrite_input:
        copied_input = tf.identity(input)
        return _nanmedian_helper(copied_input, axis, keepdims)

    else:
        result = _nanmedian_helper(input, axis, keepdims)
        return result


@with_supported_device_and_dtypes(
    {
        "2.13.0 and below": {
            "cpu": (
                "int64",
                "int32",
                "float32",
                "float64",
            ),
            "gpu": (
                "int64",
                "int32",
                "float32",
                "float64",
            ),
        }
    },
    backend_version,
)
def bincount(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    weights: Optional[Union[tf.Tensor, tf.Variable]] = None,
    minlength: int = 0,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.bincount(
        x.numpy().tolist(),
        weights=weights,
        minlength=minlength,
        dtype=x.dtype if weights is None else weights.dtype,
    )


@with_supported_device_and_dtypes(
    {
        "2.13.0 and below": {
            "cpu": ("float32", "float64"),
            "gpu": ("bfloat16", "float16", "float32", "float64"),
        }
    },
    backend_version,
)
def igamma(
    a: tf.Tensor, /, *, x: tf.Tensor, out: Optional[tf.Tensor] = None
) -> tf.Tensor:
    return tf.math.igamma(a, x)


@with_unsupported_dtypes({"2.13.0 and below": ("float16", "bfloat16")}, backend_version)
def cov(
    x1: tf.Tensor,
    x2: tf.Tensor = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[tf.Tensor] = None,
    aweights: Optional[tf.Tensor] = None,
    dtype: Optional[type] = None,
) -> tf.Tensor:
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    if len(tf.shape(x1)) > 2:
        raise ValueError("x1 has more than 2 dimensions")

    if x2 is not None:
        if len(tf.shape(x2)) > 2:
            raise ValueError("x2 has more than 2 dimensions")

    if dtype is None:
        if x2 is None:
            dtype = tf.experimental.numpy.result_type(x1, tf.float64)
        else:
            dtype = tf.experimental.numpy.result_type(x1, x2, tf.float64)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    X = tf.experimental.numpy.array(x1, ndmin=2, dtype=dtype)
    if not rowVar and tf.shape(X)[0] != 1:
        X = tf.transpose(X)

    if x2 is not None:
        x2 = tf.experimental.numpy.array(x2, copy=False, ndmin=2, dtype=dtype)
        if not rowVar and tf.shape(x2)[0] != 1:
            x2 = tf.transpose(x2)

        X = tf.concat([X, x2], axis=0)

    w = None
    if fweights is not None:
        fweights = tf.cast(fweights, dtype=tf.float64)

        if not tf.reduce_all(fweights == tf.round(fweights)):
            raise TypeError("fweights must be integer")
        if len(tf.shape(fweights)) > 1:
            raise RuntimeError("fweights must be 1 dimensional")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if tf.experimental.numpy.any(fweights < 0):
            raise ValueError("fweights cannot be negative")

        w = fweights

    if aweights is not None:
        aweights = tf.cast(aweights, dtype=tf.float64)

        if len(tf.shape(aweights)) > 1:
            raise RuntimeError("aweights must be 1 dimensional")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if tf.experimental.numpy.any(aweights < 0):
            raise ValueError("aweights cannot be negative")

        if w is None:
            w = aweights
        else:
            w = w * aweights

    avg, w_sum = tf.experimental.numpy.average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]

    if w is None:
        fact = tf.shape(X)[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum

    if fact <= 0:
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = tf.transpose(X)
    else:
        X_T = tf.transpose(X * w)

    fact = tf.cast(fact, tf.as_dtype(dtype))
    c = tf.matmul(X, tf.math.conj(X_T))
    return tf.math.truediv(c, fact)


def cummax(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    if x.dtype in (tf.bool, tf.float16):
        x = tf.cast(x, tf.float64)
    elif x.dtype in (tf.int16, tf.int8, tf.uint8):
        x = tf.cast(x, tf.int64)
    elif x.dtype in (tf.complex128, tf.complex64):
        x = tf.cast(tf.math.real(x), tf.float64)

    if exclusive or reverse:
        if exclusive and reverse:
            x, indices = __find_cummax(
                tf.experimental.numpy.flip(x, axis=axis), axis=axis
            )
            x, indices = tf.experimental.numpy.swapaxes(
                x, axis, -1
            ), tf.experimental.numpy.swapaxes(indices, axis, -1)
            x, indices = tf.experimental.numpy.concatenate(
                (tf.experimental.numpy.zeros_like(x[..., -1:]), x[..., :-1]), -1
            ), tf.experimental.numpy.concatenate(
                (
                    tf.experimental.numpy.zeros_like(indices[..., -1:]),
                    indices[..., :-1],
                ),
                -1,
            )
            x, indices = tf.experimental.numpy.swapaxes(
                x, axis, -1
            ), tf.experimental.numpy.swapaxes(indices, axis, -1)
            res, indices = tf.experimental.numpy.flip(
                x, axis=axis
            ), tf.experimental.numpy.flip(indices, axis=axis)
        elif exclusive:
            x = tf.experimental.numpy.swapaxes(x, axis, -1)
            x = tf.experimental.numpy.concatenate(
                (tf.experimental.numpy.zeros_like(x[..., -1:]), x[..., :-1]), -1
            )
            x = tf.experimental.numpy.swapaxes(x, axis, -1)
            res, indices = __find_cummax(x, axis=axis)
        elif reverse:
            x = tf.experimental.numpy.flip(x, axis=axis)
            x, indices = __find_cummax(x, axis=axis)
            res, indices = tf.experimental.numpy.flip(
                x, axis=axis
            ), tf.experimental.numpy.flip(indices, axis=axis)
        return res, indices

    return __find_cummax(x, axis=axis)


def __find_cummax(x: tf.Tensor, axis: int = 0) -> Tuple[tf.Tensor, tf.Tensor]:
    values, indices = [], []
    if (
        isinstance(x[0], tf.Tensor)
        and isinstance(x[0].numpy().tolist(), list)
        and len(x[0].numpy().tolist()) >= 1
    ):
        if axis >= 1:
            for ret1 in x:
                value, indice = __find_cummax(ret1, axis=axis - 1)
                indices.append(indice)
                values.append(value)
        else:
            x_list = x.numpy()
            z_list = __get_index(x_list.tolist())
            indices, values, n1 = x_list.copy(), x_list.copy(), {}
            indices.fill(0)
            values.fill(0)
            z_list = sorted(z_list, key=lambda i: i[1])
            for y, y_index in z_list:
                multi_index = y_index
                if tuple(multi_index[1:]) not in n1:
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices[y_index] = multi_index[0]
                    values[y_index] = y
                elif (
                    y
                    >= x_list[
                        tuple([n1[tuple(multi_index[1:])]] + list(multi_index[1:]))
                    ]
                ):
                    n1[tuple(multi_index[1:])] = multi_index[0]
                    indices[y_index] = multi_index[0]
                    values[y_index] = y
                else:
                    indices[y_index] = n1[tuple(multi_index[1:])]
                    values[y_index] = x_list[
                        tuple([n1[tuple(multi_index[1:])]] + list(multi_index[1:]))
                    ]
    else:
        x_indices = tf.convert_to_tensor(list(range(0, x.shape[0])), dtype=x.dtype)
        values, indices = tf.scan(
            lambda a, b: (
                a
                if a > b
                or tf.experimental.numpy.where(x[0].numpy() == b[0].numpy()) == 0
                else b
            ),
            (x, x_indices),
        )

    return tf.convert_to_tensor(values, dtype=x.dtype), tf.cast(
        tf.convert_to_tensor(indices), dtype=tf.int64
    )


def __get_index(lst, indices=None, prefix=None):
    if indices is None:
        indices = []
    if prefix is None:
        prefix = []

    if isinstance(lst, list):
        for i, sub_lst in enumerate(lst):
            sub_indices = prefix + [i]
            __get_index(sub_lst, indices, sub_indices)
    else:
        indices.append((lst, tuple(prefix)))
    return indices


@with_unsupported_dtypes(
    {"2.13.0 and below": ("bfloat16", "complex")},
    backend_version,
)
def cummin(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.as_native_dtype(dtype)
    if reverse:
        x = tf.reverse(x, axis=[axis])
    x_unstacked = tf.unstack(x, axis=axis)
    cummin_x_unstacked = []
    cummin_x_unstacked.append(x_unstacked[0])
    for i, x_sub in enumerate(x_unstacked[1:]):
        cummin_x_sub = tf.minimum(cummin_x_unstacked[i], x_sub)
        cummin_x_unstacked.append(cummin_x_sub)
    cummin_x = tf.stack(cummin_x_unstacked, axis=axis)
    if reverse:
        cummin_x = tf.reverse(cummin_x, axis=[axis])
    if dtype is None:
        return cummin_x
    else:
        return tf.cast(cummin_x, dtype)
