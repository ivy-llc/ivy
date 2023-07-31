# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _minimum(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.minimum(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def amin(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    out_dtype = ivy.dtype(a)
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
                s[ax] = 1
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
    if where_mask is not None and ivy.any(where_mask):
        res = ivy.where(ivy.logical_not(where_mask), res, initial, out=out)
    return ivy.astype(res, out_dtype, out=out, copy=False)


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def amax(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    out_dtype = ivy.dtype(a)
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
                s[ax] = 1
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
    if where_mask is not None and ivy.any(where_mask):
        res = ivy.where(ivy.logical_not(where_mask), res, initial, out=out)
    return ivy.astype(res, out_dtype, out=out, copy=False)


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanmin(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    out_dtype = ivy.dtype(a)
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
                s[ax] = 1
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
    return ivy.astype(res, out_dtype, out=out, copy=False)


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _maximum(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.maximum(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanmax(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    out_dtype = ivy.dtype(a)
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
                s[ax] = 1
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
    return ivy.astype(res, out_dtype, out=out, copy=False)


@handle_numpy_out
@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def _fmax(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.fmax(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def _fmin(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
#Test with out
    ret = ivy.fmin(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret
