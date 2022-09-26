# global
import ivy


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
    if dtype:
        x1 = ivy.astype(ivy.array(x1), ivy.as_ivy_dtype(dtype))
        x2 = ivy.astype(ivy.array(x2), ivy.as_ivy_dtype(dtype))
    ret = ivy.minimum(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


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

    if initial is not None:
        s = ivy.shape(a, as_array=True)
        ax = axis

        if ivy.is_array(where):
            a = ivy.where(where, a, ivy.default(out, ivy.zeros_like(a)), out=out)
        if axis is None:
            ax = 0
        if ivy.get_num_dims(s) < 2:
            header = ivy.array([initial])
        else:
            initial_shape = s.__setitem__(ax, 1)
            header = ivy.full(ivy.Shape(tuple(initial_shape)), initial)

        a = ivy.concat([a, header], axis=axis)

    return ivy.min(a, axis=axis, keepdims=keepdims, out=out)


def nanmin(
    a,
    axis=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
    bound_max = ivy.finfo(a).max
    # select only the specified elements
    if ivy.is_array(where):
        a = ivy.where(where, a, a.full_like(bound_max))
    nan_mask = ivy.isnan(a)
    x = ivy.where(ivy.logical_not(nan_mask), a, a.full_like(bound_max))
    if initial is not None:
        s = ivy.shape(x, as_array=True)
        if axis is not None:
            s[axis] = 1
        header = ivy.full(ivy.Shape(s.to_list()), initial)
        if axis:
            x = ivy.concat([x, header], axis=axis)
        else:
            x = ivy.concat([x, header])
    return ivy.min(x, axis=axis, out=out, keepdims=keepdims)
