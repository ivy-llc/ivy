# global
from typing import Optional, Union, Tuple, Sequence
import paddle
import ivy.functional.backends.paddle as paddle_backend
import ivy
from copy import deepcopy

# local
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from ivy.utils.exceptions import IvyNotImplementedException
from . import backend_version


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def median(
    input: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    # keepdims is set to True because in versions up to 2.5.1
    # there was a problem when the axis was defined and it was the
    # only axis in the tensor so it needs to be handled manually

    ret_dtype = input.dtype
    if input.dtype not in [paddle.int32, paddle.int64, paddle.float32, paddle.float64]:
        if paddle.is_complex(input):
            ret = paddle.complex(
                paddle.median(input.real(), axis=axis, keepdim=True),
                paddle.median(input.imag(), axis=axis, keepdim=True),
            )
        else:
            ret = paddle.median(input.cast("float32"), axis=axis, keepdim=True)
    else:
        ret = paddle.median(input, axis=axis, keepdim=True)
    if not keepdims:
        ret = paddle_backend.squeeze(ret, axis=axis)
    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == input.ndim:
            axis = None
    if (input.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


def nanmean(
    a: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ret_dtype = dtype if dtype is not None else a.dtype
    a = a.cast(
        ret_dtype
    )  # this is necessary to match other FWs behaviour which cast before calculation
    if a.dtype not in [paddle.int64, paddle.float32, paddle.float64]:
        if paddle.is_complex(a):
            ret = paddle.complex(
                paddle.nanmean(a.real(), axis=axis, keepdim=keepdims),
                paddle.nanmean(a.imag(), axis=axis, keepdim=keepdims),
            )
        else:
            ret = paddle.nanmean(a.cast("float32"), axis=axis, keepdim=keepdims)
    else:
        ret = paddle.nanmean(a, axis=axis, keepdim=keepdims)

    # The following code is to simulate other frameworks
    # output shapes behaviour since min output dim is 1 in paddle
    if isinstance(axis, Sequence):
        if len(axis) == a.ndim:
            axis = None
    if (a.ndim == 1 or axis is None) and not keepdims:
        ret = ret.squeeze()
    return ret.astype(ret_dtype)


def _validate_quantile(q):
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (paddle.all(0 <= q) and paddle.all(q <= 1)):
            return False
    return True


def _to_positive_axis(axis, ndim):
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


def _handle_axis(a, q, fn, keepdims=False, axis=None, interpolation="nearest"):
    nd = a.ndim
    axis_arg = deepcopy(axis)
    if axis is not None:
        axis = _to_positive_axis(axis, nd)

        if len(axis) == 1:
            axis_arg = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)

            for i, s in enumerate(sorted(keep)):
                a = a.moveaxis(s, i)
            a = a.reshape(
                a.shape[:nkeep]
                + [
                    -1,
                ]
            )
            axis_arg = -1

    ret = fn(a, q, axis=axis_arg, interpolation=interpolation)

    if keepdims:
        if axis is None:
            index_ret = (None,) * nd
        else:
            index_ret = tuple(None if i in axis else slice(None) for i in range(nd))
        ret = ret[(Ellipsis,) + index_ret]

    return ret


def _quantile(a, q, axis=None, interpolation="nearest"):
    ret_dtype = a.dtype
    if q.ndim > 2:
        raise ValueError("q argument must be a scalar or 1-dimensional!")
    if axis is None:
        axis = 0
        a = paddle.flatten(a)

    n = a.shape[axis]
    if axis != 0:
        a = paddle.moveaxis(a, axis, 0)

    indices = []
    for q_num in q:
        index = q_num * (n - 1)
        indices.append(index)

    a = paddle.sort(a, 0)
    outputs = []

    for index in indices:
        if interpolation == "lower":
            index = paddle.floor(index)
        elif interpolation == "higher":
            index = paddle.ceil(index)
        elif interpolation == "nearest":
            index = paddle.round(index)
        elif interpolation == "midpoint":
            index_floor = paddle.floor(index)
            index_ceil = paddle.ceil(index)
            index = (index_ceil + index_floor) / 2

        indices_below = paddle.floor(index).astype(paddle.int32)
        indices_upper = paddle.ceil(index).astype(paddle.int32)

        if interpolation == "nearest_jax":
            weights = index - indices_below.astype(paddle.float64)

            indices_below = paddle.clip(indices_below, 0, n - 1)
            indices_upper = paddle.clip(indices_upper, 0, n - 1)
            tensor_upper = paddle.gather(a, indices_upper, axis=0)
            tensor_below = paddle.gather(a, indices_below, axis=0)

            pred = weights <= 0.5
            out = paddle.where(pred, tensor_below, tensor_upper)
        else:
            tensor_upper = paddle.gather(a, indices_upper, axis=0)
            tensor_below = paddle.gather(a, indices_below, axis=0)
            weights = index - indices_below.astype(paddle.float64)
            out = paddle.lerp(
                tensor_below.astype(paddle.float64),
                tensor_upper.astype(paddle.float64),
                weights.astype(paddle.float64),
            )
        outputs.append(out)
    return paddle.concat(outputs, axis=0).astype(ret_dtype)


def _compute_quantile_wrapper(
    x,
    q,
    axis=None,
    keepdims=False,
    interpolation="linear",
):
    if not _validate_quantile(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if interpolation not in [
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
        "nearest_jax",
    ]:
        raise ValueError(
            "Interpolation must be 'linear', 'lower', 'higher', 'midpoint' or 'nearest'"
        )
    return _handle_axis(
        x,
        q,
        _quantile,
        keepdims=keepdims,
        axis=axis,
        interpolation=interpolation,
    )


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "float16",
                "bfloat16",
                "complex64",
                "complex128",
            )
        }
    },
    backend_version,
)
def quantile(
    a: paddle.Tensor,
    q: Union[paddle.Tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    # added the nearest_jax mode to enable jax-like calculations for method="nearest"
    return _compute_quantile_wrapper(
        x=a,
        q=q,
        axis=axis,
        keepdims=keepdims,
        interpolation=interpolation,
    )


def corrcoef(
    x: paddle.Tensor,
    /,
    *,
    y: Optional[paddle.Tensor] = None,
    rowvar: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def histogram(
    a: paddle.Tensor,
    /,
    *,
    bins: Optional[Union[int, paddle.Tensor]] = None,
    axis: Optional[int] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[paddle.Tensor] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[paddle.Tensor] = None,
    density: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor]:
    if range is None:
        min_range = 0
        max_range = 0
    else:
        min_range = range[0]
        max_range = range[1]
    return paddle.histogram(a, bins=bins, min=min_range, max=max_range)


def nanmedian(
    input: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    overwrite_input: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if input.dtype not in [paddle.int32, paddle.int64, paddle.float32, paddle.float64]:
        if dtype is None:
            dtype = input.dtype
        input = input.cast("float32")
        paddle.nanmedian(x=input, axis=axis, keepdim=keepdims).cast(dtype)
    return paddle.nanmedian(x=input, axis=axis, keepdim=keepdims).cast(dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "float16",
                "bool",
            )
        }
    },
    backend_version,
)
def unravel_index(
    indices: paddle.Tensor,
    shape: Tuple[int],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if indices.ndim == 0:
        indices = indices.unsqueeze(0)
    coord = []
    indices = indices
    for dim in reversed(shape):
        coord.append((indices % dim).astype("int32"))
        indices = paddle.floor(indices / dim)

    return tuple(reversed(coord))


@with_unsupported_device_and_dtypes(
    {
        "2.5.1 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "float16",
                "float32",
                "float64",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def bincount(
    x: paddle.Tensor,
    /,
    *,
    weights: Optional[paddle.Tensor] = None,
    minlength: int = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.bincount(x, weights=weights, minlength=minlength).cast(
        x.dtype if weights is None else weights.dtype
    )


def igamma(
    a: paddle.Tensor,
    /,
    *,
    x: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    results = []
    ret_dtype = a.dtype if out is None else out.dtype
    if paddle.float16 in [a.dtype, x.dtype]:
        a = a.astype("float32")
        x = x.astype("float32")

    for ai, xi in zip(a.flatten(), x.flatten()):
        ai = ai.astype("float64")
        xi = xi.astype("float64")

        def integrand(t):
            return paddle.exp(-t) * paddle.pow(t, ai - 1)

        intervals = paddle.linspace(0, xi, 10001).astype("float64")
        interval_width = xi / 10000
        values = integrand(intervals)
        integral = paddle.multiply((values[:-1] + values[1:]) / 2, interval_width)
        result = paddle.divide(paddle.sum(integral), paddle.exp(paddle.lgamma(ai)))
        results.append(result)

    return paddle.to_tensor(results, dtype=ret_dtype).reshape(a.shape)


def cov(
    x1: paddle.Tensor,
    x2: paddle.Tensor = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[paddle.Tensor] = None,
    aweights: Optional[paddle.Tensor] = None,
    dtype: Optional[paddle.dtype] = None,
) -> paddle.Tensor:
    if fweights is not None:
        fweights = fweights.astype("float64")

    if aweights is not None:
        aweights = aweights.astype("float64")

    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be an integer")

    if len(x1.shape) > 2:
        raise ValueError("x1 has more than 2 dimensions")

    if x2 is not None:
        if len(x2.shape) > 2:
            raise ValueError("x2 has more than 2 dimensions")

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    if dtype is None:
        x1 = x1.astype("float64")
        if x2 is not None:
            x2 = x2.astype("float64")
    else:
        x1 = x1.astype(dtype)
        if x2 is not None:
            x2 = x2.astype(dtype)

    X = x1
    if not rowVar and X.shape[0] != 1:
        X = paddle.transpose(X, perm=tuple(range(len(X.shape) - 1, -1, -1)))

    if x2 is not None:
        if not rowVar and x2.shape[0] != 1:
            x2 = paddle.transpose(x2, perm=tuple(range(len(x2.shape) - 1, -1, -1)))
        if len(x2.shape) > 1:
            X = paddle.concat([X, x2], axis=0)
        else:
            X = paddle.stack([X, x2], axis=0)

    if not rowVar:
        X = paddle.transpose(X, perm=tuple(range(len(X.shape) - 1, -1, -1)))

    return paddle.linalg.cov(
        X, rowvar=rowVar, ddof=ddof, fweights=fweights, aweights=aweights
    )


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def cummax(
    x: paddle.Tensor,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    if x.dtype in (paddle.bool, paddle.float16):
        x = paddle.cast(x, "float64")
    elif x.dtype in (paddle.int16, paddle.int8, paddle.uint8):
        x = paddle.cast(x, "int64")
    elif x.dtype in (paddle.complex128, paddle.complex64):
        x = paddle.cast(paddle.real(x), "float64")

    if not (exclusive or reverse):
        return __find_cummax(x, axis=axis)

    elif exclusive and reverse:
        x, indices = __find_cummax(ivy.flip(x, axis=(axis,)), axis=axis)
        x, indices = ivy.swapaxes(x, axis, -1), ivy.swapaxes(indices, axis, -1)
        x = ivy.concat((ivy.zeros_like(x[..., -1:]), x[..., :-1]), axis=-1)
        indices = ivy.concat(
            (ivy.zeros_like(indices[..., -1:]), indices[..., :-1]), axis=-1
        )
        x, indices = ivy.swapaxes(x, axis, -1), ivy.swapaxes(indices, axis, -1)
        return ivy.flip(x, axis=(axis,)), ivy.flip(indices, axis=(axis,))

    elif exclusive:
        x = ivy.swapaxes(x, axis, -1)
        x = ivy.concat((ivy.zeros_like(x[..., -1:]), x[..., :-1]), axis=-1)
        x = ivy.swapaxes(x, axis, -1)
        x, indices = __find_cummax(x, axis=axis)

        return x, indices

    else:
        x, indices = __find_cummax(ivy.flip(x, axis=(axis,)), axis=axis)
        return ivy.flip(x, axis=axis), ivy.flip(indices, axis=axis)


def __find_cummax(
    x: paddle.Tensor, axis: int = 0, dtype: Optional[paddle.dtype] = None
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    indices = []
    values = []
    x_dtype = x.dtype if dtype is None else dtype
    if (
        isinstance(x.tolist()[0], list)
        and len(x[0].shape) >= 1
        and (
            (type(x[0]) == paddle.Tensor)
            or (type(x[0]) == ivy.data_classes.array.array.Array)
        )
    ):
        if axis >= 1:
            if not isinstance(x, list):
                x = x.tolist()
            for ret1 in x:
                value, indice = __find_cummax(
                    paddle.to_tensor(ret1, dtype=x_dtype), axis=axis - 1, dtype=x_dtype
                )
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
        if not isinstance(x, list):
            x = x.tolist()
        n = 0
        for idx, y in enumerate(x):
            if x[n] > y:
                values.append(x[n])
            elif x[n] <= y or idx == 0:
                n = idx
                values.append(y)
            indices.append(n)

    if type(x) == paddle.Tensor:
        return paddle.to_tensor(values, dtype=x.dtype), paddle.to_tensor(
            indices, dtype="int64"
        )
    else:
        return ivy.array(values, dtype=x_dtype), ivy.array(indices, dtype="int64")


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


@with_unsupported_device_and_dtypes(
    {"2.5.1 and below": {"cpu": ("uint8", "int8", "int16")}},
    backend_version,
)
def cummin(
    x: paddle.Tensor,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    dtype = dtype if dtype is not None else x.dtype
    if reverse:
        x = paddle.flip(x, axis=[axis])
    x_unstacked = paddle.unbind(x, axis=axis)
    cummin_x_unstacked = []
    cummin_x_unstacked.append(x_unstacked[0])
    for i, x_sub in enumerate(x_unstacked[1:]):
        cummin_x_sub = paddle.minimum(cummin_x_unstacked[i], x_sub)
        cummin_x_unstacked.append(cummin_x_sub)
    cummin_x = paddle.stack(cummin_x_unstacked, axis=axis)
    if reverse:
        cummin_x = paddle.flip(cummin_x, axis=[axis])
    return cummin_x.cast(dtype)
