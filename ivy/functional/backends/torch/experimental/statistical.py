# global
from typing import Optional, Union, Tuple, Sequence
import torch

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
import ivy
from ..statistical import _infer_dtype
from copy import deepcopy


@with_unsupported_dtypes(
    {
        "2.0.1 and below": (
            "uint8",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "bfloat16",
        )
    },
    backend_version,
)
def histogram(
    a: torch.Tensor,
    /,
    *,
    bins: Optional[Union[int, torch.Tensor]] = None,
    axis: Optional[int] = None,
    extend_lower_interval: Optional[bool] = False,
    extend_upper_interval: Optional[bool] = False,
    dtype: Optional[torch.dtype] = None,
    range: Optional[Tuple[float]] = None,
    weights: Optional[torch.Tensor] = None,
    density: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    min_a = torch.min(a)
    max_a = torch.max(a)
    if isinstance(bins, torch.Tensor) and range:
        raise ivy.exceptions.IvyException(
            "Must choose between specifying bins and range or bin edges directly"
        )
    if range:
        bins = torch.linspace(
            start=range[0], end=range[1], steps=bins + 1, dtype=a.dtype
        )
        range = None
    elif isinstance(bins, int):
        range = (min_a, max_a)
        bins = torch.linspace(
            start=range[0], end=range[1], steps=bins + 1, dtype=a.dtype
        )
        range = None
    if bins.size()[0] < 2:
        raise ivy.exceptions.IvyException("bins must have at least 1 bin (size > 1)")
    bins_out = bins.clone()
    if extend_lower_interval and min_a < bins[0]:
        bins.data[0] = min_a
    if extend_upper_interval and max_a > bins[-1]:
        bins.data[-1] = max_a
    if a.ndim > 0 and axis is not None:
        inverted_shape_dims = list(torch.flip(torch.arange(a.ndim), dims=[0]))
        if isinstance(axis, int):
            axis = [axis]
        shape_axes = 1
        for dimension in axis:
            inverted_shape_dims.remove(dimension)
            inverted_shape_dims.append(dimension)
            shape_axes *= a.shape[dimension]
        a_along_axis_1d = (
            a.permute(inverted_shape_dims).flatten().reshape((-1, shape_axes))
        )
        if weights is None:
            ret = []
            for a_1d in a_along_axis_1d:
                ret_1d = torch.histogram(
                    a_1d,
                    bins=bins,
                    # TODO: waiting tensorflow version support to density
                    # density=density,
                )[0]
                ret.append(ret_1d.tolist())
        else:
            weights_along_axis_1d = (
                weights.permute(inverted_shape_dims).flatten().reshape((-1, shape_axes))
            )
            ret = []
            for a_1d, weights_1d in zip(a_along_axis_1d, weights_along_axis_1d):
                ret_1d = torch.histogram(
                    a_1d,
                    bins=bins,
                    weight=weights_1d,
                    # TODO: waiting tensorflow version support to density
                    # density=density,
                )[0]
                ret.append(ret_1d.tolist())
        out_shape = list(a.shape)
        for dimension in sorted(axis, reverse=True):
            del out_shape[dimension]
        out_shape.insert(0, len(bins) - 1)
        ret = torch.tensor(ret)
        ret = ret.flatten()
        index = torch.zeros(len(out_shape), dtype=int)
        ret_shaped = torch.zeros(out_shape)
        dim = 0
        i = 0
        if index.tolist() == (torch.tensor(out_shape) - 1).tolist():
            ret_shaped.data[tuple(index)] = ret[i]
        while index.tolist() != (torch.tensor(out_shape) - 1).tolist():
            ret_shaped.data[tuple(index)] = ret[i]
            dim_full_flag = False
            while index[dim] == out_shape[dim] - 1:
                index[dim] = 0
                dim += 1
                dim_full_flag = True
            index[dim] += 1
            i += 1
            if dim_full_flag:
                dim = 0
        if index.tolist() == (torch.tensor(out_shape) - 1).tolist():
            ret_shaped.data[tuple(index)] = ret[i]
        ret = ret_shaped
    else:
        ret = torch.histogram(
            a, bins=bins, range=range, weight=weights, density=density
        )[0]
    dtype = ivy.as_native_dtype(dtype)
    if dtype:
        ret = ret.type(dtype)
        bins_out = bins_out.type(dtype)
    # TODO: weird error when returning bins: return ret, bins_out
    return ret


histogram.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bool")}, backend_version)
def median(
    input: torch.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if isinstance(axis, tuple):
        if len(axis) == 1:
            axis = axis[0]
    ret = quantile(
        input,
        0.5,
        axis=axis,
        keepdims=keepdims,
        interpolation="midpoint",
    )
    if input.dtype in [torch.int64, torch.float64]:
        ret = torch.asarray(ret, dtype=torch.float64)
    elif input.dtype in [torch.float16, torch.bfloat16]:
        ret = torch.asarray(ret, dtype=input.dtype)
    else:
        ret = torch.asarray(ret, dtype=torch.float32)
    return ret


median.support_native_out = False


def nanmean(
    a: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nanmean(a, dim=axis, keepdim=keepdims, dtype=dtype, out=out)


nanmean.support_native_out = True


def nanprod(
    a: torch.Tensor,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[torch.Tensor] = None,
    initial: Optional[Union[int, float, complex, ivy.Container]] = None,
    where: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    a = torch.nan_to_num(a, nan=1.0)
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(a.dtype)
    if axis == ():
        return a.type(dtype)
    if axis is None:
        return torch.prod(input=a, dtype=dtype, out=out)
    if isinstance(axis, tuple) or isinstance(axis, list):
        for i in axis:
            a = torch.prod(a, dim=i, keepdim=keepdims, dtype=dtype, out=out)
        return a
    return torch.prod(a, dim=axis, keepdim=keepdims, dtype=dtype, out=out)


nanprod.support_native_out = True


def _validate_quantile(q):
    if q.ndim == 1 and torch.numel(q) < 10:
        for i in range(torch.numel(q)):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (torch.all(0 <= q) and torch.all(q <= 1)):
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


def _handle_axis(a, q, fn, keepdims=False, axis=None):
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
                a = torch.moveaxis(a, s, i)
            a = a.view(
                [
                    *a.shape[:nkeep],
                    -1,
                ]
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
    if q.ndim > 2:
        raise ValueError("q argument must be a scalar or 1-dimensional!")
    if axis is None:
        axis = 0
        a = a.flatten()

    n = a.shape[axis]
    if axis != 0:
        a = torch.moveaxis(a, axis, 0)

    indices = []
    for q_num in q:
        index = q_num * (n - 1)
        indices.append(index)

    a = torch.sort(a, 0)[0]
    outputs = []

    for index in indices:
        indices_below = torch.floor(index).to(torch.int64)
        indices_upper = torch.ceil(index).to(torch.int64)

        weights = index - indices_below.to(torch.float64)

        indices_below = torch.clip(indices_below, 0, n - 1)
        indices_upper = torch.clip(indices_upper, 0, n - 1)
        tensor_upper = torch.index_select(a, 0, indices_upper)
        tensor_below = torch.index_select(a, 0, indices_below)

        pred = weights <= 0.5
        out = torch.where(pred, tensor_below, tensor_upper)
        outputs.append(out)
    return torch.concat(outputs, dim=0).to(ret_dtype)


def _compute_quantile_wrapper(
    x, q, axis=None, keepdims=False, interpolation="linear", out=None
):
    if not _validate_quantile(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    if interpolation in [
        "linear",
        "lower",
        "higher",
        "midpoint",
        "nearest",
        "nearest_jax",
    ]:
        if interpolation == "nearest_jax":
            return _handle_axis(x, q, _quantile, keepdims=keepdims, axis=axis)
        else:
            return torch.quantile(
                x, q, dim=axis, keepdim=keepdims, interpolation=interpolation, out=out
            )
    else:
        raise ValueError(
            "Interpolation must be 'linear', 'lower', 'higher', 'midpoint' or 'nearest'"
        )


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, backend_version)
def quantile(
    a: torch.Tensor,
    q: Union[torch.Tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    interpolation: str = "linear",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # added the nearest_jax mode to enable jax-like calculations for method="nearest"
    return _compute_quantile_wrapper(
        a,
        q,
        axis=axis,
        keepdims=keepdims,
        interpolation=interpolation,
        out=out,
    )


quantile.support_native_out = True


def corrcoef(
    x: torch.Tensor,
    /,
    *,
    y: Optional[torch.Tensor] = None,
    rowvar: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if y is None:
        xarr = x
    else:
        axis = 0 if rowvar else 1
        xarr = torch.concat([x, y], dim=axis)
        xarr = xarr.T if not rowvar else xarr

    return torch.corrcoef(xarr)


def _nanmedian(input, axis, keepdims):
    dtype = input.dtype
    temp = input.to(torch.float64)
    num_dim = len(temp.size())
    keepdim_shape = list(temp.size())
    q = 0.5

    axis = [axis] if isinstance(axis, int) else list(axis)

    for i in axis:
        keepdim_shape[i] = 1
    axis = [num_dim + x if x < 0 else x for x in axis]
    axis.sort()
    dimension = len(temp.size())
    while len(axis) > 0:
        axis1 = axis[0]
        for axis2 in range(axis1 + 1, dimension):
            temp = torch.transpose(temp, axis1, axis2)
            axis1 = axis2
        axis = [x - 1 for x in axis]
        axis.pop(0)
        dimension = dimension - 1
    temp = torch.flatten(temp, start_dim=dimension - len(axis))
    ret = torch.nanquantile(temp, q, dim=-1, keepdim=keepdims, interpolation="midpoint")
    if keepdims:
        keepdim_shape = tuple(keepdim_shape)
        ret = ret.reshape(keepdim_shape)

    if dtype in [torch.int32, torch.int64, torch.float64]:
        ret = torch.asarray(ret, dtype=torch.float64)
    elif dtype in [torch.float16, torch.bfloat16]:
        ret = torch.asarray(ret, dtype=torch.float16)
    else:
        ret = torch.asarray(ret, dtype=torch.float32)

    return ret


@with_unsupported_dtypes({"2.0.1 and below": ("bfloat16", "float16")}, backend_version)
def nanmedian(
    input: torch.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: bool = False,
    overwrite_input: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if overwrite_input:
        copied_input = input.clone()
        if axis is None:
            copied_input = copied_input.flatten()

            ret = torch.nanquantile(
                copied_input.double(),
                0.5,
                dim=-1,
                keepdim=keepdims,
                interpolation="midpoint",
            )

            if input.dtype in [torch.int32, torch.int64, torch.float64]:
                ret = ret.to(torch.float64)
            elif input.dtype in [torch.float16, torch.bfloat16]:
                ret = ret.to(torch.float16)
            else:
                ret = ret.to(torch.float32)
            return ret

        return _nanmedian(copied_input, axis, keepdims)

    else:
        if axis is None:
            input = input.flatten()

            ret = torch.nanquantile(
                input.double(), 0.5, dim=-1, keepdim=keepdims, interpolation="midpoint"
            )

            if input.dtype in [torch.int32, torch.int64, torch.float64]:
                ret = ret.to(torch.float64)
            elif input.dtype in [torch.float16, torch.bfloat16]:
                ret = ret.to(torch.float16)
            else:
                ret = ret.to(torch.float32)
            return ret

        return _nanmedian(input, axis, keepdims)


nanmedian.support_native_out = True


def bincount(
    x: torch.Tensor,
    /,
    *,
    weights: Optional[torch.Tensor] = None,
    minlength: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        ret = torch.bincount(x, minlength=minlength)
        ret = ret.to(x.dtype)
    else:
        ret = torch.bincount(x, weights=weights, minlength=minlength)
        ret = ret.to(weights.dtype)
    return ret


bincount.support_native_out = False


def igamma(
    a: torch.Tensor,
    /,
    *,
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.special.gammainc(a, x, out=out)


igamma.support_native_out = True


@with_unsupported_dtypes({"2.0.1 and below": ("float16", "bfloat16")}, backend_version)
def cov(
    x1: torch.Tensor,
    x2: torch.Tensor = None,
    /,
    *,
    rowVar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    fweights: Optional[torch.Tensor] = None,
    aweights: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # dtype casts separately
    if fweights is not None:
        fweights = fweights.type(torch.int64)
    if aweights is not None:
        aweights = aweights.type(torch.float64)

    if x1.dim() > 2:
        raise ValueError("x1 has more than 2 dimensions")

    if x2 is not None:
        if x2.dim() > 2:
            raise ValueError("x2 has more than 2 dimensions")

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    if dtype is None:
        x1 = x1.type(torch.float64)
        if x2 is not None:
            x2 = x2.type(torch.float64)
    else:
        x1 = x1.type(dtype)
        if x2 is not None:
            x2 = x2.type(dtype)

    X = x1
    if not rowVar and len(x1.shape) != 1:
        X = torch.t(x1)

    if x2 is not None:
        if not rowVar and len(x2.shape) != 1:
            x2 = torch.t(x2)
        X = torch.vstack((X, x2))

    return torch.cov(X, correction=ddof, fweights=fweights, aweights=aweights)


cov.support_native_out = False


@with_unsupported_dtypes(
    {"2.0.1 and below": ("uint8", "bfloat16", "float16")},
    backend_version,
)
def cummax(
    x: torch.Tensor,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dtype in (torch.bool, torch.float16):
        x = x.to(dtype=torch.float64)
    elif x.dtype in (torch.int16, torch.int8, torch.uint8):
        x = x.to(dtype=torch.int64)
    elif x.dtype in (torch.complex64, torch.complex128):
        x = x.real.to(dtype=torch.float64)

    if exclusive or reverse:
        if exclusive and reverse:
            x1, x2 = torch.cummax(torch.flip(x, dims=(axis,)), axis)
            x1, x2 = torch.transpose(x1, axis, -1), torch.transpose(x2, axis, -1)
            x1, x2 = torch.concat(
                (torch.zeros_like(x1[..., -1:]), x1[..., :-1]), -1
            ), torch.concat((torch.zeros_like(x2[..., -1:]), x2[..., :-1]), -1)
            x1, x2 = torch.transpose(x1, axis, -1), torch.transpose(x2, axis, -1)
            res1, res2 = torch.flip(x1, dims=(axis,)), torch.flip(x2, dims=(axis,))
        elif exclusive:
            x = torch.transpose(x, axis, -1)
            x = torch.cat((torch.zeros_like(x[..., -1:]), x[..., :-1]), -1)
            x1, x2 = torch.cummax(x, -1)
            res1, res2 = torch.transpose(x1, axis, -1), torch.transpose(x2, axis, -1)
        else:
            x1, x2 = torch.cummax(torch.flip(x, dims=(axis,)), axis)
            res1, res2 = torch.flip(x1, dims=(axis,)), torch.flip(x2, dims=(axis,))
        return res1, res2

    return torch.cummax(x, axis, out=out)


@with_unsupported_dtypes(
    {
        "2.0.1 and below": ("uint8", "float16", "bfloat16"),
        "1.12.1 and above": ("uint8", "float16"),
    },
    backend_version,
)
def cummin(
    x: torch.Tensor,
    /,
    *,
    axis: int = 0,
    exclusive: bool = False,
    reverse: bool = False,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    if not (reverse):
        ret = torch.cummin(x, axis)[0]
    else:
        ret = torch.cummin(torch.flip(x, dims=(axis,)), axis)[0]
        ret = torch.flip(ret, (axis,))
    if ivy.exists(out):
        return ivy.inplace_update(out, ret.to(dtype))
    return ret.to(dtype)
