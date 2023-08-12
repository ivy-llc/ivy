# local

import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_jax_dtype,
)
from ivy.functional.frontends.jax.numpy import promote_types_of_jax_inputs

import warnings


@to_ivy_arrays_and_back
def einsum(
    subscripts,
    *operands,
    out=None,
    optimize="optimal",
    precision=None,
    preferred_element_type=None,
    _use_xeinsum=False,
    _dot_general=None,
):
    return ivy.einsum(subscripts, *operands, out=out)


@handle_jax_dtype
@to_ivy_arrays_and_back
def mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a) else a.dtype
    ret = ivy.mean(a, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ivy.astype(ret, ivy.as_ivy_dtype(dtype), copy=False)


@handle_jax_dtype
@to_ivy_arrays_and_back
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a) else a.dtype
    ret = ivy.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ivy.astype(ret, ivy.as_ivy_dtype(dtype), copy=False)


@to_ivy_arrays_and_back
def argmin(a, axis=None, out=None, keepdims=None):
    return ivy.argmin(a, axis=axis, out=out, keepdims=keepdims)


@to_ivy_arrays_and_back
def bincount(x, weights=None, minlength=0, *, length=None):
    x_list = [int(x[i]) for i in range(x.shape[0])]
    max_val = int(ivy.max(ivy.array(x_list)))
    ret = [x_list.count(i) for i in range(0, max_val + 1)]
    ret = ivy.array(ret)
    ret = ivy.astype(ret, ivy.as_ivy_dtype(ivy.int64))
    return ret


@handle_jax_dtype
@to_ivy_arrays_and_back
def cumprod(a, axis=None, dtype=None, out=None):
    if dtype is None:
        dtype = ivy.as_ivy_dtype(a.dtype)
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_jax_dtype
@to_ivy_arrays_and_back
def cumsum(a, axis=0, dtype=None, out=None):
    if dtype is None:
        dtype = ivy.uint8
    return ivy.cumsum(a, axis, dtype=dtype, out=out)


cumproduct = cumprod


@handle_jax_dtype
@to_ivy_arrays_and_back
def sum(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=None,
    promote_integers=True,
):
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a.dtype) else ivy.as_ivy_dtype(a.dtype)

    # TODO: promote_integers is only supported from JAX v0.4.10
    if dtype is None and promote_integers:
        if ivy.is_bool_dtype(dtype):
            dtype = ivy.default_int_dtype()
        elif ivy.is_uint_dtype(dtype):
            if ivy.dtype_bits(dtype) < ivy.dtype_bits(ivy.default_uint_dtype()):
                dtype = ivy.default_uint_dtype()
        elif ivy.is_int_dtype(dtype):
            if ivy.dtype_bits(dtype) < ivy.dtype_bits(ivy.default_int_dtype()):
                dtype = ivy.default_int_dtype()

    if initial:
        if axis is None:
            a = ivy.reshape(a, (1, -1))
            axis = 0
        s = list(ivy.shape(a))
        s[axis] = 1
        header = ivy.full(s, initial)
        a = ivy.concat([a, header], axis=axis)

    ret = ivy.sum(a, axis=axis, keepdims=keepdims, out=out)

    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ivy.astype(ret, ivy.as_ivy_dtype(dtype))


@to_ivy_arrays_and_back
def min(a, axis=None, out=None, keepdims=False, where=None):
    ret = ivy.min(a, axis=axis, out=out, keepdims=keepdims)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


amin = min


@to_ivy_arrays_and_back
def max(a, axis=None, out=None, keepdims=False, where=None):
    ret = ivy.max(a, axis=axis, out=out, keepdims=keepdims)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


amax = max


@to_ivy_arrays_and_back
def average(a, axis=None, weights=None, returned=False, keepdims=False):
    # canonicalize_axis to ensure axis or the values in axis > 0
    if isinstance(axis, (tuple, list)):
        a_ndim = len(ivy.shape(a))
        new_axis = [0] * len(axis)
        for i, v in enumerate(axis):
            if not -a_ndim <= v < a_ndim:
                raise ValueError(
                    f"axis {v} is out of bounds for array of dimension {a_ndim}"
                )
            new_axis[i] = v + a_ndim if v < 0 else v
        axis = tuple(new_axis)

    if weights is None:
        ret = ivy.mean(a, axis=axis, keepdims=keepdims)
        if axis is None:
            fill_value = int(a.size) if ivy.is_int_dtype(ret) else float(a.size)
            weights_sum = ivy.full(shape=(), fill_value=fill_value, dtype=ret.dtype)
        else:
            if isinstance(axis, tuple):
                # prod with axis has dtype Sequence[int]
                fill_value = 1
                for d in axis:
                    fill_value *= a.shape[d]
            else:
                fill_value = a.shape[axis]
            fill_value = int(fill_value) if ivy.is_int_dtype(ret) else float(fill_value)
            weights_sum = ivy.full_like(ret, fill_value=fill_value)
    else:
        a = ivy.asarray(a, copy=False)
        weights = ivy.asarray(weights, copy=False)
        a, weights = promote_types_of_jax_inputs(a, weights)

        a_shape = ivy.shape(a)
        a_ndim = len(a_shape)
        weights_shape = ivy.shape(weights)

        # Make sure the dimensions work out
        if a_shape != weights_shape:
            if len(weights_shape) != 1:
                raise ValueError(
                    "1D weights expected when shapes of a and weights differ."
                )
            if axis is None:
                raise ValueError(
                    "Axis must be specified when shapes of a and weights differ."
                )
            elif isinstance(axis, tuple):
                raise ValueError(
                    "Single axis expected when shapes of a and weights differ"
                )
            elif weights.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis."
                )

            weights = ivy.broadcast_to(
                weights, shape=(a_ndim - 1) * (1,) + weights_shape
            )
            weights = ivy.moveaxis(weights, -1, axis)

        weights_sum = ivy.sum(weights, axis=axis)
        ret = ivy.sum(a * weights, axis=axis, keepdims=keepdims) / weights_sum

    if returned:
        if ret.shape != weights_sum.shape:
            weights_sum = ivy.broadcast_to(weights_sum, shape=ret.shape)
        return ret, weights_sum

    return ret


@to_ivy_arrays_and_back
def nanmax(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    nan_mask = ivy.isnan(a)
    a = ivy.where(ivy.logical_not(nan_mask), a, a.full_like(-ivy.inf))
    where_mask = None
    if initial is not None:
        if ivy.is_array(where):
            a = ivy.where(where, a, a.full_like(initial))
            where_mask = ivy.all(ivy.logical_not(where), axis=axis, keepdims=keepdims)
        s = ivy.shape(a, as_array=True)
        if axis is not None:
            if isinstance(axis, (tuple, list)) or ivy.is_array(axis):
                # introducing the initial in one dimension is enough
                ax = axis[0] % len(s)
            else:
                ax = axis % len(s)
            s[ax] = 1
        header = ivy.full(ivy.Shape(s.to_list()), initial, dtype=ivy.dtype(a))
        if axis:
            if isinstance(axis, (tuple, list)) or ivy.is_array(axis):
                a = ivy.concat([a, header], axis=axis[0])
            else:
                a = ivy.concat([a, header], axis=axis)
        else:
            a = ivy.concat([a, header], axis=0)
    res = ivy.max(a, axis=axis, keepdims=keepdims, out=out)
    if nan_mask is not None:
        nan_mask = ivy.all(nan_mask, axis=axis, keepdims=keepdims, out=out)
        if ivy.any(nan_mask):
            res = ivy.where(
                ivy.logical_not(nan_mask),
                res,
                initial if initial is not None else ivy.nan,
                out=out,
            )
    if where_mask is not None and ivy.any(where_mask):
        res = ivy.where(ivy.logical_not(where_mask), res, ivy.nan, out=out)
    return res


@to_ivy_arrays_and_back
def nanmin(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    nan_mask = ivy.isnan(a)
    a = ivy.where(ivy.logical_not(nan_mask), a, a.full_like(+ivy.inf))
    where_mask = None
    if initial is not None:
        if ivy.is_array(where):
            a = ivy.where(where, a, a.full_like(initial))
            where_mask = ivy.all(ivy.logical_not(where), axis=axis, keepdims=keepdims)
        s = ivy.shape(a, as_array=True)
        if axis is not None:
            if isinstance(axis, (tuple, list)) or ivy.is_array(axis):
                # introducing the initial in one dimension is enough
                ax = axis[0] % len(s)
            else:
                ax = axis % len(s)
            s[ax] = 1
        header = ivy.full(ivy.Shape(s.to_list()), initial, dtype=ivy.dtype(a))
        if axis:
            if isinstance(axis, (tuple, list)) or ivy.is_array(axis):
                a = ivy.concat([a, header], axis=axis[0])
            else:
                a = ivy.concat([a, header], axis=axis)
        else:
            a = ivy.concat([a, header], axis=0)
    res = ivy.min(a, axis=axis, keepdims=keepdims, out=out)
    if nan_mask is not None:
        nan_mask = ivy.all(nan_mask, axis=axis, keepdims=keepdims, out=out)
        if ivy.any(nan_mask):
            res = ivy.where(
                ivy.logical_not(nan_mask),
                res,
                initial if initial is not None else ivy.nan,
                out=out,
            )
    if where_mask is not None and ivy.any(where_mask):
        res = ivy.where(ivy.logical_not(where_mask), res, ivy.nan, out=out)
    return res


@handle_jax_dtype
@to_ivy_arrays_and_back
def nanstd(
    a, /, *, axis=None, dtype=None, out=None, ddof=0, keepdims=False, where=True
):
    a = ivy.nan_to_num(a)
    axis = tuple(axis) if isinstance(axis, list) else axis

    if dtype:
        a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))

    ret = ivy.std(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret


@handle_jax_dtype
@to_ivy_arrays_and_back
def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    is_nan = ivy.isnan(a)
    if dtype is None:
        dtype = "float16" if ivy.is_int_dtype(a) else a.dtype
    if ivy.any(is_nan):
        a = [i for i in a if ivy.isnan(i) is False]

    if dtype:
        a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))

    ret = ivy.var(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    all_nan = ivy.isnan(ret)
    if ivy.all(all_nan):
        ret = ivy.astype(ret, ivy.array([float("inf")]))
    return ret


@handle_jax_dtype
@to_ivy_arrays_and_back
def nancumsum(a, axis=None, dtype=None, out=None):
    a = ivy.where(ivy.isnan(a), ivy.zeros_like(a), a)
    return ivy.cumsum(a, axis=axis, dtype=dtype, out=out)


@handle_jax_dtype
@to_ivy_arrays_and_back
def nancumprod(a, axis=None, dtype=None, out=None):
    a = ivy.where(ivy.isnan(a), ivy.zeros_like(a), a)
    return ivy.cumprod(a, axis=axis, dtype=dtype, out=out)


@handle_jax_dtype
@with_unsupported_dtypes({"0.4.14 and below": ("bfloat16",)}, "jax")
@to_ivy_arrays_and_back
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float32" if ivy.is_int_dtype(a) else a.dtype
    std_a = ivy.std(a, axis=axis, correction=ddof, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        where = ivy.array(where, dtype=ivy.bool)
        std_a = ivy.where(
            where, std_a, ivy.default(out, ivy.zeros_like(std_a)), out=out
        )
    return ivy.astype(std_a, ivy.as_ivy_dtype(dtype), copy=False)


@to_ivy_arrays_and_back
def corrcoef(x, y=None, rowvar=True):
    return ivy.corrcoef(x, y=y, rowvar=rowvar)


@to_ivy_arrays_and_back
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    return ivy.median(a, axis=axis, out=out, keepdims=keepdims)


@to_ivy_arrays_and_back
def ptp(a, axis=None, out=None, keepdims=False):
    x = ivy.max(a, axis=axis, keepdims=keepdims)
    y = ivy.min(a, axis=axis, keepdims=keepdims)
    return ivy.subtract(x, y)


@handle_jax_dtype
@to_ivy_arrays_and_back
def nanmean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=None):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype is None:
        dtype = "float64" if ivy.is_int_dtype(a) else a.dtype
    if ivy.is_array(where):
        where1 = ivy.array(where, dtype=ivy.bool)
        a = ivy.where(where1, a, ivy.full_like(a, ivy.nan))
    nan_mask1 = ivy.isnan(a)
    not_nan_mask1 = ~ivy.isnan(a)
    b1 = ivy.where(ivy.logical_not(nan_mask1), a, ivy.zeros_like(a))
    array_sum1 = ivy.sum(b1, axis=axis, dtype=dtype, keepdims=keepdims, out=out)
    not_nan_mask_count1 = ivy.sum(
        not_nan_mask1, axis=axis, dtype=dtype, keepdims=keepdims, out=out
    )
    count_zero_handel = ivy.where(
        not_nan_mask_count1 != 0,
        not_nan_mask_count1,
        ivy.full_like(not_nan_mask_count1, ivy.nan),
    )
    return ivy.divide(array_sum1, count_zero_handel)


@to_ivy_arrays_and_back
def nanmedian(
    a,
    /,
    *,
    axis=None,
    keepdims=False,
    out=None,
    overwrite_input=False,
):
    return ivy.nanmedian(
        a, axis=axis, keepdims=keepdims, out=out, overwrite_input=overwrite_input
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes({"0.4.14 and below": ("float16", "bfloat16")}, "jax")
def correlate(a, v, mode="valid", precision=None):
    if ivy.get_num_dims(a) != 1 or ivy.get_num_dims(v) != 1:
        raise ValueError("correlate() only support 1-dimensional inputs.")
    if a.shape[0] == 0 or v.shape[0] == 0:
        raise ValueError(
            f"correlate: inputs cannot be empty, got shapes {a.shape} and {v.shape}."
        )
    if v.shape[0] > a.shape[0]:
        need_flip = True
        a, v = v, a
    else:
        need_flip = False

    out_order = slice(None)

    if mode == "valid":
        padding = [(0, 0)]
    elif mode == "same":
        padding = [(v.shape[0] // 2, v.shape[0] - v.shape[0] // 2 - 1)]
    elif mode == "full":
        padding = [(v.shape[0] - 1, v.shape[0] - 1)]
    else:
        raise ValueError("mode must be one of ['full', 'same', 'valid']")

    result = ivy.conv_general_dilated(
        a[None, None, :],
        v[:, None, None],
        (1,),
        padding,
        dims=1,
        data_format="channel_first",
    )
    return ivy.flip(result[0, 0, out_order]) if need_flip else result[0, 0, out_order]


@to_ivy_arrays_and_back
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    return ivy.cov(
        m, y, rowVar=rowvar, bias=bias, ddof=ddof, fweights=fweights, aweights=aweights
    )


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {"0.4.14 and below": ("complex64", "complex128", "bfloat16", "bool", "float16")},
    "jax",
)
def quantile(
    a,
    q,
    /,
    *,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    interpolation=None,
):
    if method == "nearest":
        return ivy.quantile(
            a, q, axis=axis, keepdims=keepdims, interpolation="nearest_jax", out=out
        )
    return ivy.quantile(
        a, q, axis=axis, keepdims=keepdims, interpolation=method, out=out
    )


@to_ivy_arrays_and_back
def nanquantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method='linear',
    keepdims=False,
    interpolation=None):
    if interpolation:
        warnings.warn('interpolation is deprecated name for the method keyword argument since version 1.22.0 in JAX') 
        method=interpolation   
    return ivy.nanquantile(
        a, q, axis=axis, keepdims=keepdims, interpolation=method, out=out
    )
