# global
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
)


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def minimum(
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


@to_ivy_arrays_and_back
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
        res = ivy.where(ivy.logical_not(where_mask), res, ivy.nan, out=out)
    return res


@to_ivy_arrays_and_back
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
        res = ivy.where(ivy.logical_not(where_mask), res, ivy.nan, out=out)
    return res


@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
def maximum(
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
