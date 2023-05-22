# global
from typing import Optional, Union, Tuple, Sequence
import paddle
from ivy.utils.exceptions import IvyNotImplementedException

# local
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from . import backend_version


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "uint16",
                "bfloat16",
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
    if input.ndim == 0:
        input = input.unsqueeze(0)
        return paddle.median(x=input, axis=axis).squeeze()
    elif input.ndim == 1:
        return paddle.median(x=input) if keepdims else paddle.median(x=input).squeeze()

    return paddle.median(x=input, axis=axis, keepdim=keepdims)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": ("uint16", "bfloat16", "float16", "complex64", "complex128")
        }
    },
    backend_version,
)
def nanmean(
    a: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if a.dtype not in [paddle.int64, paddle.float32, paddle.float64]:
        if dtype is None:
            dtype = a.dtype
        a = a.cast("float32")
        paddle.nanmean(x=a, axis=axis, keepdim=keepdims).cast(dtype)
    return paddle.nanmean(x=a, axis=axis, keepdim=keepdims).cast(dtype)


def _compute_quantile(
    x, q, axis=None, keepdim=False, ignore_nan=False, interpolation="linear"
):
    # Validate x
    if not isinstance(x, paddle.Tensor):
        raise TypeError("input x should be a Tensor.")
    ret_dtype = x.dtype
    # Validate q
    if isinstance(q, (int, float)):
        q = [q]
    elif isinstance(q, (list, tuple)):
        if len(q) <= 0:
            raise ValueError("q should not be empty")
    elif isinstance(q, paddle.Tensor):
        q = q.tolist()
    else:
        raise TypeError("Type of q should be int, float, list or tuple.")

    # Validate axis
    dims = len(x.shape)
    out_shape = list(x.shape)
    if axis is None:
        x = paddle.flatten(x)
        axis = 0
        out_shape = [1] * dims
    else:
        if isinstance(axis, (list, tuple)):
            if len(axis) <= 0:
                raise ValueError("axis should not be empty")
            axis_src, axis_dst = [], []
            for axis_single in axis:
                if not isinstance(axis_single, int) or not (
                    axis_single < dims and axis_single >= -dims
                ):
                    raise ValueError(
                        "Axis should be None, int, or a list, element should in "
                        "range [-rank(x), rank(x))."
                    )
                if axis_single < 0:
                    axis_single = axis_single + dims
                axis_src.append(axis_single)
                out_shape[axis_single] = 1
            axis_dst = list(range(-len(axis), 0))
            x = paddle.moveaxis(x, axis_src, axis_dst)
            x = paddle.flatten(x, axis_dst[0], axis_dst[-1])
            axis = axis_dst[0]
        else:
            if not isinstance(axis, int) or not (axis < dims and axis >= -dims):
                raise ValueError(
                    "Axis should be None, int, or a list, element should in "
                    "range [-rank(x), rank(x))."
                )
            if axis < 0:
                axis += dims
            out_shape[axis] = 1

    mask = x.isnan()
    valid_counts = mask.logical_not().sum(axis=axis, keepdim=True, dtype="float64")

    indices = []

    for q_num in q:
        if q_num < 0 or q_num > 1:
            raise ValueError("q should be in range [0, 1]")
        if paddle.in_dynamic_mode():
            q_num = paddle.to_tensor(q_num, dtype="float64")
        if ignore_nan:
            indices.append(q_num * (valid_counts - 1))
        else:
            index = q_num * (valid_counts - 1)
            last_index = x.shape[axis] - 1
            nums = paddle.full_like(index, fill_value=last_index)
            index = paddle.where(mask.any(axis=axis, keepdim=True), nums, index)
            indices.append(index)

    sorted_tensor = paddle.sort(x, axis)

    outputs = []

    for index in indices:
        if interpolation not in ["linear", "lower", "higher", "midpoint", "nearest"]:
            raise ValueError(
                "interpolation must be 'linear', 'lower', 'higher', 'midpoint', "
                "or 'nearest'"
            )
        if interpolation == "lower":
            index = paddle.floor(index)
        elif interpolation == "higher":
            index = paddle.ceil(index)
        elif interpolation == "nearest":
            index = paddle.round(index)
        elif interpolation == "midpoint":
            index_floor = paddle.floor(index)
            index_ceil = paddle.ceil(index)
            index = (index_floor + index_ceil) / 2

        indices_below = paddle.floor(index).astype(paddle.int32)
        indices_upper = paddle.ceil(index).astype(paddle.int32)
        tensor_upper = paddle.take_along_axis(sorted_tensor, indices_upper, axis=axis)
        tensor_below = paddle.take_along_axis(sorted_tensor, indices_below, axis=axis)
        weights = index - indices_below.astype("float64")
        out = paddle.lerp(
            tensor_below.astype("float64"),
            tensor_upper.astype("float64"),
            weights,
        )
        if not keepdim:
            out = paddle.squeeze(out, axis=axis)
        else:
            out = out.reshape(out_shape)
        outputs.append(out)

    if len(q) > 1:
        outputs = paddle.stack(outputs, 0)
    else:
        outputs = outputs[0]

    return outputs.astype(ret_dtype)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "uint16",
                "bfloat16",
                "float16",
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
    return _compute_quantile(
        x=a,
        q=q,
        axis=axis,
        keepdim=keepdims,
        interpolation=interpolation,
        ignore_nan=False,
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
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "uint16",
                "bfloat16",
                "float16",
                "complex64",
                "complex128",
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


def cov_paddle(
    x1: paddle.Tensor,
    x2: paddle.Tensor = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[paddle.Tensor] = None,
    aweights: Optional[paddle.Tensor] = None,
    dtype: Optional[type] = None,
) -> paddle.Tensor:
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

    X = x1.astype("float64")
    if not rowVar and X.shape[0] != 1:
        X = paddle.transpose(X, perm=tuple(range(len(X.shape)-1, -1, -1)))

    x2 = x2.astype(("float64"))
    if x2 is not None:
        if not rowVar and x2.shape[0] != 1:
            x2 = paddle.transpose(x2, perm=tuple(range(len(X.shape)-1, -1, -1)))
        X = paddle.concat([X, x2], axis=0)

    w = None
    if fweights is not None:
        fweights = fweights.astype("float64")
        if not paddle.all(fweights == paddle.round(fweights)):
            raise TypeError("fweights must be an integer")
        if len(fweights.shape) > 1:
            raise RuntimeError("fweights must be 1-dimensional")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if paddle.any(fweights < 0):
            raise ValueError("fweights cannot be negative")

        w = fweights

    if aweights is not None:
        aweights = aweights.astype("float64")
        if len(aweights.shape) > 1:
            raise RuntimeError("aweights must be 1-dimensional")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if paddle.any(aweights < 0):
            raise ValueError("aweights cannot be negative")

        if w is None:
            w = aweights
        else:
            w = w * aweights


    # print('w: ',w)
    # Calculate weighted average
    # print('before avg: ',X)
    temp_avg = paddle.multiply(X, w)
    avg = temp_avg.sum(axis=1)/paddle.sum(w)

    if w is not None:
        w_sum = paddle.sum(w)
    # print('avg', avg)
    # print('w_sum', w_sum)
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * paddle.sum(w * aweights) / w_sum

    if fact <= 0:
        fact = 0.0

    X -= avg.unsqueeze(1)
    # print('X: ', X)
    if w is None:
        X_T = paddle.transpose(X, perm=tuple(range(len(X.shape)-1, -1, -1)))
    else:
        X_temp = X * w
        X_T = paddle.transpose(X_temp, perm=tuple(range(len(X_temp.shape)-1, -1, -1)))
    # print('X_T: ', X_T)
    c = paddle.matmul(X, X_T)
    # print('c: ', c)
    result = paddle.divide(c, paddle.to_tensor(fact, dtype="float64"))
    return result
