# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_jax_dtype,
)
from ivy.functional.frontends.jax.numpy import promote_types_of_jax_inputs


@to_ivy_arrays_and_back
def clip(a, a_min=None, a_max=None, out=None):
    ivy.utils.assertions.check_all_or_any_fn(
        a_min,
        a_max,
        fn=ivy.exists,
        type="any",
        limit=[1, 2],
        message="at most one of a_min or a_max can be None",
    )
    a = ivy.array(a)
    if a_min is None:
        a, a_max = promote_types_of_jax_inputs(a, a_max)
        return ivy.minimum(a, a_max, out=out)
    if a_max is None:
        a, a_min = promote_types_of_jax_inputs(a, a_min)
        return ivy.maximum(a, a_min, out=out)
    return ivy.clip(a, a_min, a_max, out=out)


@handle_jax_dtype
@to_ivy_arrays_and_back
def concatenate(arrays, axis=0, dtype=None):
    ret = ivy.concat(arrays, axis=axis)
    if dtype:
        ret = ivy.array(ret, dtype=dtype)
    return ret


@to_ivy_arrays_and_back
def reshape(a, newshape, order="C"):
    return ivy.reshape(a, shape=newshape, order=order)


@to_ivy_arrays_and_back
def moveaxis(a, source, destination):
    return ivy.moveaxis(a, source, destination)


@to_ivy_arrays_and_back
def flipud(m):
    return ivy.flipud(m, out=None)


@to_ivy_arrays_and_back
def transpose(a, axes=None):
    if not axes:
        axes = list(range(len(a.shape)))[::-1]
    if type(axes) is int:
        axes = [axes]
    if (len(a.shape) == 0 and not axes) or (len(a.shape) == 1 and axes[0] == 0):
        return a
    return ivy.permute_dims(a, axes, out=None)


@to_ivy_arrays_and_back
def flip(m, axis=None):
    return ivy.flip(m, axis=axis)


@to_ivy_arrays_and_back
def fliplr(m):
    return ivy.fliplr(m)


@to_ivy_arrays_and_back
def expand_dims(a, axis):
    return ivy.expand_dims(a, axis=axis)


@to_ivy_arrays_and_back
def stack(arrays, axis=0, out=None, dtype=None):
    if dtype:
        return ivy.astype(
            ivy.stack(arrays, axis=axis, out=out), ivy.as_ivy_dtype(dtype)
        )
    return ivy.stack(arrays, axis=axis, out=out)


@to_ivy_arrays_and_back
def take(
    a,
    indices,
    axis=None,
    out=None,
    mode=None,
    unique_indices=False,
    indices_are_sorted=False,
    fill_value=None,
):
    return ivy.take_along_axis(a, indices, axis, out=out)


@to_ivy_arrays_and_back
def broadcast_to(arr, shape):
    return ivy.broadcast_to(arr, shape)


@to_ivy_arrays_and_back
def append(arr, values, axis=None):
    if axis is None:
        return ivy.concat((ivy.flatten(arr), ivy.flatten(values)), axis=0)
    else:
        return ivy.concat((arr, values), axis=axis)


@to_ivy_arrays_and_back
def swapaxes(a, axis1, axis2):
    return ivy.swapaxes(a, axis1, axis2)


@to_ivy_arrays_and_back
def atleast_3d(*arys):
    return ivy.atleast_3d(*arys)


@to_ivy_arrays_and_back
def atleast_1d(*arys):
    return ivy.atleast_1d(*arys)


@to_ivy_arrays_and_back
def atleast_2d(*arys):
    return ivy.atleast_2d(*arys)


@to_ivy_arrays_and_back
def squeeze(a, axis=None):
    return ivy.squeeze(a, axis)


@to_ivy_arrays_and_back
def dsplit(ary, indices_or_sections):
    return ivy.dsplit(ary, indices_or_sections)


@to_ivy_arrays_and_back
def vsplit(ary, indices_or_sections):
    return ivy.vsplit(ary, indices_or_sections)


@to_ivy_arrays_and_back
def hsplit(ary, indices_or_sections):
    return ivy.hsplit(ary, indices_or_sections)


@to_ivy_arrays_and_back
def bartlett(M):
    if M == 1:
        return ivy.ones(1)
    elif M % 2 == 0:
        return ivy.concat(
            (ivy.linspace(0, 1, M // 2 + 1), ivy.linspace(1, 0, M // 2 + 1)[1:])
        )
    else:
        return ivy.concat(
            (ivy.linspace(0, 1, (M + 1) // 2), ivy.linspace(1, 0, (M + 1) // 2 - 1)[1:])
        )


@to_ivy_arrays_and_back
def blackman(M):
    if M < 1:
        return ivy.array([])

    if M == 1:
        return ivy.ones(1)

    alpha = 0.16
    a0 = (1 - alpha) / 2
    a1 = 1 / 2
    a2 = alpha / 2

    n = ivy.arange(0, M)
    ret = (
        a0
        - a1 * ivy.cos(2 * ivy.pi * n / (M - 1))
        + a2 * ivy.cos(4 * ivy.pi * n / (M - 1))
    )

    return ret


@to_ivy_arrays_and_back
def block(arrays):
    concatenated = ivy.concat(arrays, axis=1)
    num_rows = arrays[0].shape[0]
    num_cols = sum(arr.shape[1] for arr in arrays)
    ret = ivy.reshape(concatenated, (num_rows, num_cols))
    return ret
