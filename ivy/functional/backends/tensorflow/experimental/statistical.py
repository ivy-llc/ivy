from typing import Union, Optional, Tuple, Sequence
import tensorflow as tf

from tensorflow.python.ops.numpy_ops import np_math_ops
import ivy
from ivy import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)
from .. import backend_version

# from ivy.functional.backends.paddle.experimental.statistical import to_positive_axis
from copy import deepcopy


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
    # TODO: Implement in pure tensorflow
    pass


@with_supported_dtypes(
    {
        "2.15.0 and below": (
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
    pass
    # TODO: Implement in pure tensorflow


@with_supported_dtypes(
    {
        "2.15.0 and below": (
            "float",
            "complex",
        )
    },
    backend_version,
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


def nanmin(
    a: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    initial: Optional[Union[int, float, complex]] = None,
    where: Optional[Union[tf.Tensor, tf.Variable]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    axis = tuple(axis) if isinstance(axis, list) else axis
    nan_mask = tf.math.is_nan(a)
    if where is not None:
        nan_mask = tf.math.logical_or(nan_mask, tf.math.logical_not(where))

    masked_tensor = tf.where(nan_mask, tf.constant(float("inf"), dtype=a.dtype), a)

    if axis is None:
        result = tf.math.reduce_min(masked_tensor, keepdims=keepdims)
    else:
        result = tf.math.reduce_min(masked_tensor, axis=axis, keepdims=keepdims)
    if initial is not None:
        result = tf.minimum(result, initial)
    return result


def _infer_dtype(dtype: tf.DType):
    default_dtype = ivy.infer_default_dtype(dtype)
    if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
        return default_dtype
    return dtype


def nanprod(
    a: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[tf.DType] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
    initial: Optional[Union[int, float, complex]] = None,
    where: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    np_math_ops.enable_numpy_methods_on_tensor()
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(a.dtype)
    if initial is None:
        initial = 1
    axis = tuple(axis) if isinstance(axis, list) else axis
    return (
        tf.experimental.numpy.nanprod(a, axis=axis, keepdims=keepdims, dtype=dtype)
        * initial
    )


def _validate_quantile(q):
    if tf.experimental.numpy.ndim(q) == 1 and tf.size(q) < 10:
        for i in range(tf.size(q)):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (tf.math.reduce_all(q >= 0) and tf.math.reduce_all(q <= 1)):
            return False
    return True


def to_positive_axis(axis, ndim):
    if not isinstance(axis, (list, tuple)):
        axis = [axis]

    if len(axis) == 0:
        raise ValueError("Axis can't be empty!")

    if len(set(axis)) != len(axis):
        raise ValueError("Duplicated axis!")

    for i in range(len(axis)):
        if not (isinstance(axis[i], int) and (ndim > axis[i] >= -ndim)):
            raise ValueError("Axis must be int in range [-rank(x), rank(x))")
        if axis[i] < 0:
            axis[i] += ndim
    return axis


def _handle_axis(a, q, fn, keepdims=False, axis=None):
    nd = tf.experimental.numpy.ndim(a)
    axis_arg = deepcopy(axis)
    if axis is not None:
        axis = to_positive_axis(axis, nd)

        if len(axis) == 1:
            axis_arg = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)

            for i, s in enumerate(sorted(keep)):
                a = tf.experimental.numpy.moveaxis(a, s, i)
            a = tf.reshape(
                a,
                [
                    *a.shape[:nkeep],
                    -1,
                ],
            )
            axis_arg = -1

    ret = fn(a, q, axis=axis_arg)

    if keepdims:
        if axis is None:
            index_ret = (None,) * nd
        else:
            index_ret = tuple(None if i in axis else slice(None) for i in range(nd))
        ret = ret[(Ellipsis,) + index_ret]

    return ret


def _quantile(a, q, axis=None):
    ret_dtype = a.dtype
    if tf.experimental.numpy.ndim(q) > 1:
        raise ValueError("q argument must be a scalar or 1-dimensional!")
    if axis is None:
        axis = 0
        a = tf.reshape(a, [-1])
    elif axis != 0:
        a = tf.experimental.numpy.moveaxis(a, axis, 0)
        axis = 0

    n = a.shape[axis]

    indices = q * (n - 1)

    a = tf.sort(a, axis)

    indices_below = tf.cast(tf.math.floor(indices), dtype=tf.int32)
    indices_upper = tf.cast(tf.math.ceil(indices), dtype=tf.int32)

    weights = indices - tf.cast(indices_below, dtype=ret_dtype)

    indices_below = tf.clip_by_value(indices_below, 0, n - 1)
    indices_upper = tf.clip_by_value(indices_upper, 0, n - 1)
    tensor_upper = tf.gather(a, indices_upper, axis=axis)
    tensor_below = tf.gather(a, indices_below, axis=axis)

    pred = weights <= 0.5
    out = tf.where(pred, tensor_below, tensor_upper)

    return tf.cast(out, ret_dtype)


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
    pass
    # TODO: Implement in pure tensorflow


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


def nanmedian(
    input: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    pass
    # TODO: Implement in pure tensorflow


@with_supported_device_and_dtypes(
    {
        "2.15.0 and below": {
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
        x,
        weights=weights,
        minlength=minlength,
        dtype=x.dtype if weights is None else weights.dtype,
    )


@with_supported_device_and_dtypes(
    {
        "2.15.0 and below": {
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


@with_unsupported_dtypes({"2.15.0 and below": ("float16", "bfloat16")}, backend_version)
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


@with_unsupported_dtypes(
    {"2.15.0 and below": ("bool",)},
    backend_version,
)
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
    if x.dtype in (tf.complex128, tf.complex64):
        x = tf.math.real(x)

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
    {"2.15.0 and below": ("bfloat16", "bool", "complex")},
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
