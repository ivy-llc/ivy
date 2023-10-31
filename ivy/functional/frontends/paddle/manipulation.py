# global
import ivy
from ivy.functional.frontends.paddle.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_supported_dtypes,
    with_supported_device_and_dtypes,
)


@with_unsupported_dtypes({"2.5.2 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def abs(x, name=None):
    return ivy.abs(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def broadcast_to(x, shape, name=None):
    return ivy.broadcast_to(x, shape)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "bool",
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "uint8",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def cast(x, dtype):
    return ivy.astype(x, dtype)


@with_unsupported_dtypes({"2.5.2 and below": ("int8", "int16")}, "paddle")
@to_ivy_arrays_and_back
def concat(x, axis, name=None):
    return ivy.concat(x, axis=axis)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def expand(x, shape, name=None):
    return ivy.expand(x, shape)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int8", "uint8", "int16", "float16")},
    "paddle",
)
@to_ivy_arrays_and_back
def flip(x, axis, name=None):
    return ivy.flip(x, axis=axis)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def gather(params, indices, axis=-1, batch_dims=0, name=None):
    return ivy.gather(params, indices, axis=axis, batch_dims=batch_dims)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int8", "uint8", "int16", "uint16", "float16", "bfloat16")},
    "paddle",
)
@to_ivy_arrays_and_back
def gather_nd(x, index, name=None):
    return ivy.gather_nd(x, index)


@to_ivy_arrays_and_back
def put_along_axis(arr, indices, values, axis, reduce="assign"):
    result = ivy.put_along_axis(arr, indices, values, axis)
    return result


@with_supported_dtypes(
    {"2.5.2 and below": ("int32", "int64", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def repeat_interleave(x, repeats, axis=None, name=None):
    return ivy.repeat(x, repeats, axis=axis)


@to_ivy_arrays_and_back
def reshape(x, shape, name=None):
    return ivy.reshape(x, shape)


@with_supported_dtypes(
    {
        "2.5.0 and below": (
            "float32",
            "float64",
            "int32",
            "int64",
            "complex64",
            "complex128",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def roll(x, shifts, axis=None, name=None):
    return ivy.roll(x, shifts, axis=axis)


@with_supported_device_and_dtypes(
    {
        "2.5.2 and above": {
            "cpu": (
                "bool",
                "int32",
                "int64",
                "float32",
                "float64",
            ),
            "gpu": ("float16",),
        },
    },
    "paddle",
)
@to_ivy_arrays_and_back
def rot90(x, k=1, axes=(0, 1), name=None):
    return ivy.rot90(x, k=k, axes=axes)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def split(x, num_or_sections, axis=0, name=None):
    return ivy.split(x, num_or_size_splits=num_or_sections, axis=axis)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("float16", "bfloat16", "int8", "int16")},
    "paddle",
)
@to_ivy_arrays_and_back
def squeeze(x, axis=None, name=None):
    return ivy.squeeze(x, axis=axis)


@to_ivy_arrays_and_back
def stack(x, axis=0, name=None):
    return ivy.stack(x, axis=axis)


@with_supported_dtypes(
    {
        "2.5. and below": (
            "int64",
            "float64",
            "complex128",
            "float32",
            "complex64",
            "int32",
            "uint8",
            "bfloat16",
            "bool",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def strided_slice(x, axes, starts, ends, strides):
    shape = list(x.shape)
    slices = [slice(None) for i in range(len(shape))]
    if axes:
        for i in axes:
            starts[i] = starts[i] if (starts[i]) > 0 else -starts[i] - 1
            ends[i] = ends[i] if (ends[i]) > 0 else -ends[i] - 1
            slices[i] = slice(starts[i], ends[i], strides[i])
    else:
        for i, start, end, stride in enumerate(zip(starts, ends, strides)):
            start = start if start > 0 else -start - 1
            end = end if end > 0 else -end - 1
            slices[i] = slice(start, end, stride)
    slices = tuple(slices)
    return x[slices]


def take_along_axis(arr, indices, axis):
    return ivy.take_along_axis(arr, indices, axis)


@with_unsupported_dtypes(
    {"2.5.2 and below": ("int8", "uint8", "int16", "float16")},
    "paddle",
)
@to_ivy_arrays_and_back
def tile(x, repeat_times, name=None):
    return ivy.tile(x, repeats=repeat_times)


@to_ivy_arrays_and_back
def tolist(x):
    return ivy.to_list(x)


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "int32", "int64", "float16", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def unbind(input, axis=0):
    shape = list(input.shape)
    num_splits = shape[axis]
    shape.pop(axis)
    return tuple([x.reshape(tuple(shape)) for x in split(input, num_splits, axis=axis)])


@with_supported_dtypes(
    {"2.5.2 and below": ("bool", "int32", "int64", "float16", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def unique_consecutive(x, axis=0):
    return ivy.unique_consecutive(x, axis=axis)


@with_supported_dtypes(
    {
        "2.5.2 and below": (
            "float32",
            "float64",
            "int32",
            "int64",
        )
    },
    "paddle",
)
@to_ivy_arrays_and_back
def unstack(x, axis=0, name=None):
    return ivy.unstack(x, axis=axis)


absolute = abs
