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
def repeat(a, repeats, axis=None, *, total_repeat_length=None):
    return ivy.repeat(a, repeats, axis=axis)


@to_ivy_arrays_and_back
def reshape(a, newshape, order="C"):
    return ivy.reshape(a, shape=newshape, order=order)


@to_ivy_arrays_and_back
def ravel(a, order="C"):
    return ivy.reshape(a, shape=(-1,), order=order)


@to_ivy_arrays_and_back
def resize(a, new_shape):
    a = ivy.array(a)
    resized_a = ivy.reshape(a, new_shape)
    return resized_a


@to_ivy_arrays_and_back
def moveaxis(a, source, destination):
    return ivy.moveaxis(a, source, destination)


@to_ivy_arrays_and_back
def flipud(m):
    return ivy.flipud(m, out=None)


@to_ivy_arrays_and_back
def transpose(a, axes=None):
    if ivy.isscalar(a):
        return ivy.array(a)
    elif a.ndim == 1:
        return a
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
    return ivy.gather(a, indices, axis=axis, out=out)


@to_ivy_arrays_and_back
def broadcast_arrays(*args):
    return ivy.broadcast_arrays(*args)


@to_ivy_arrays_and_back
def broadcast_shapes(*shapes):
    return ivy.broadcast_shapes(*shapes)


@to_ivy_arrays_and_back
def broadcast_to(array, shape):
    return ivy.broadcast_to(array, shape)


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
def tril(m, k=0):
    return ivy.tril(m, k=k)


@to_ivy_arrays_and_back
def block(arrays):
    return arrays
    

@to_ivy_arrays_and_back
def squeeze(a, axis=None):
    return ivy.squeeze(a, axis=axis)


@to_ivy_arrays_and_back
def rot90(m, k=1, axes=(0, 1)):
    return ivy.rot90(m, k=k, axes=axes)


@to_ivy_arrays_and_back
def split(ary, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, (list, tuple, ivy.Array)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[ary.shape[axis]])
            .astype(ivy.int8)
            .to_list()
        )
    return ivy.split(
        ary, num_or_size_splits=indices_or_sections, axis=axis, with_remainder=False
    )


@to_ivy_arrays_and_back
def array_split(ary, indices_or_sections, axis=0):
    return ivy.split(
        ary, num_or_size_splits=indices_or_sections, axis=axis, with_remainder=True
    )


@to_ivy_arrays_and_back
def tile(A, reps):
    return ivy.tile(A, reps)


@to_ivy_arrays_and_back
def dsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, ivy.Array)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[ary.shape[2]])
            .astype(ivy.int8)
            .to_list()
        )
    return ivy.dsplit(ary, indices_or_sections)


@to_ivy_arrays_and_back
def dstack(tup, dtype=None):
    return ivy.dstack(tup)


@to_ivy_arrays_and_back
def vsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, ivy.Array)):
        indices_or_sections = (
            ivy.diff(indices_or_sections, prepend=[0], append=[ary.shape[0]])
            .astype(ivy.int8)
            .to_list()
        )
    return ivy.vsplit(ary, indices_or_sections)


@to_ivy_arrays_and_back
def hsplit(ary, indices_or_sections):
    if isinstance(indices_or_sections, (list, tuple, ivy.Array)):
        if ary.ndim == 1:
            indices_or_sections = (
                ivy.diff(indices_or_sections, prepend=[0], append=[ary.shape[0]])
                .astype(ivy.int8)
                .to_list()
            )
        else:
            indices_or_sections = (
                ivy.diff(indices_or_sections, prepend=[0], append=[ary.shape[1]])
                .astype(ivy.int8)
                .to_list()
            )
    return ivy.hsplit(ary, indices_or_sections)


@to_ivy_arrays_and_back
def roll(a, shift, axis=None):
    return ivy.roll(a, shift, axis=axis)


@to_ivy_arrays_and_back
def row_stack(tup):
    if len(ivy.shape(tup[0])) == 1:
        xs = []
        for t in tup:
            xs += [ivy.reshape(t, (1, ivy.shape(t)[0]))]
        return ivy.concat(xs, axis=0)
    return ivy.concat(tup, axis=0)


@to_ivy_arrays_and_back
def column_stack(tup):
    if len(ivy.shape(tup[0])) == 1:
        ys = []
        for t in tup:
            ys += [ivy.reshape(t, (ivy.shape(t)[0], 1))]
        return ivy.concat(ys, axis=1)
    return ivy.concat(tup, axis=1)


@to_ivy_arrays_and_back
def pad(array, pad_width, mode="constant", **kwargs):
    return ivy.pad(array, pad_width, mode=mode, **kwargs)


def hamming(M):
    if M <= 1:
        return ivy.ones([M], dtype=ivy.float64)
    n = ivy.arange(M)
    ret = 0.54 - 0.46 * ivy.cos(2.0 * ivy.pi * n / (M - 1))
    return ret


@to_ivy_arrays_and_back
def hanning(M):
    if M <= 1:
        return ivy.ones([M], dtype=ivy.float64)
    n = ivy.arange(M)
    ret = 0.5 * (1 - ivy.cos(2.0 * ivy.pi * n / (M - 1)))
    return ret


@to_ivy_arrays_and_back
def kaiser(M, beta):
    if M <= 1:
        return ivy.ones([M], dtype=ivy.float64)
    n = ivy.arange(M)
    alpha = 0.5 * (M - 1)
    ret = ivy.i0(beta * ivy.sqrt(1 - ((n - alpha) / alpha) ** 2)) / ivy.i0(beta)
    return ret


@handle_jax_dtype
@to_ivy_arrays_and_back
def tri(N, M=None, k=0, dtype="float64"):
    if M is None:
        M = N
    ones = ivy.ones((N, M), dtype=dtype)
    return ivy.tril(ones, k=k)


@to_ivy_arrays_and_back
def blackman(M):
    if M < 1:
        return ivy.array([])
    if M == 1:
        return ivy.ones((1,))
    n = ivy.arange(0, M)
    alpha = 0.16
    a0 = (1 - alpha) / 2
    a1 = 1 / 2
    a2 = alpha / 2
    ret = (
        a0
        - a1 * ivy.cos(2 * ivy.pi * n / (M - 1))
        + a2 * ivy.cos(4 * ivy.pi * n / (M - 1))
    )
    return ret
