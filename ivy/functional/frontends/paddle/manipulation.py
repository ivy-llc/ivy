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


@with_unsupported_dtypes({"2.6.0 and below": ("float16", "bfloat16")}, "paddle")
@to_ivy_arrays_and_back
def abs(x, name=None):
    return ivy.abs(x)


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def broadcast_to(x, shape, name=None):
    return ivy.broadcast_to(x, shape)


@with_supported_dtypes(
    {
        "2.6.0 and below": (
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


@with_unsupported_dtypes({"2.6.0 and below": ("int8", "int16")}, "paddle")
@to_ivy_arrays_and_back
def concat(x, axis, name=None):
    return ivy.concat(x, axis=axis)


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def expand(x, shape, name=None):
    return ivy.expand(x, shape)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("int8", "uint8", "int16", "float16")},
    "paddle",
)
@to_ivy_arrays_and_back
def flip(x, axis, name=None):
    return ivy.flip(x, axis=axis)


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "float32", "float64", "int32", "int64")},
    "paddle",
)
@to_ivy_arrays_and_back
def gather(params, indices, axis=-1, batch_dims=0, name=None):
    return ivy.gather(params, indices, axis=axis, batch_dims=batch_dims)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("int8", "uint8", "int16", "uint16", "float16", "bfloat16")},
    "paddle",
)
@to_ivy_arrays_and_back
def gather_nd(x, index, name=None):
    return ivy.gather_nd(x, index)


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "int32", "int64", "float16", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def index_add(x, index, axis, value, *, name=None):
    x = ivy.swapaxes(x, axis, 0)
    value = ivy.swapaxes(value, axis, 0)
    _to_adds = []
    index = sorted(zip(ivy.to_list(index), range(len(index))), key=(lambda i: i[0]))
    while index:
        _curr_idx = index[0][0]
        while len(_to_adds) < _curr_idx:
            _to_adds.append(ivy.zeros_like(value[0]))
        _to_add_cum = ivy.get_item(value, index[0][1])
        while (len(index)) > 1 and (index[0][0] == index[1][0]):
            _to_add_cum = _to_add_cum + ivy.get_item(value, index.pop(1)[1])
        index.pop(0)
        _to_adds.append(_to_add_cum)
    while len(_to_adds) < x.shape[0]:
        _to_adds.append(ivy.zeros_like(value[0]))
    _to_adds = ivy.stack(_to_adds)
    if len(x.shape) < 2:
        # Added this line due to the paddle backend treating scalars as 1-d arrays
        _to_adds = ivy.flatten(_to_adds)

    ret = ivy.add(x, _to_adds)
    ret = ivy.swapaxes(ret, axis, 0)
    return ret


@with_supported_dtypes({"2.5.1 and below": ("int32", "int64")}, "paddle")
@to_ivy_arrays_and_back
def one_hot(x, num_classes, name=None):
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    one_hot_tensor = ivy.one_hot(x, num_classes)
    return one_hot_tensor.astype(ivy.float32)


@to_ivy_arrays_and_back
def put_along_axis(arr, indices, values, axis, reduce="assign"):
    result = ivy.put_along_axis(arr, indices, values, axis)
    return result


@with_supported_dtypes(
    {"2.6.0 and below": ("int32", "int64", "float32", "float64")},
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
        "2.6.0 and above": {
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
    {"2.6.0 and below": ("int16", "complex64", "complex128")},
    "paddle",
)
@to_ivy_arrays_and_back
def split(x, num_or_sections, axis=0, name=None):
    return ivy.split(x, num_or_size_splits=num_or_sections, axis=axis)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("float16", "bfloat16", "int8", "int16")},
    "paddle",
)
@to_ivy_arrays_and_back
def squeeze(x, axis=None, name=None):
    return ivy.squeeze(x, axis=axis)


@to_ivy_arrays_and_back
def stack(x, axis=0, name=None):
    return ivy.stack(x, axis=axis)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def take_along_axis(arr, indices, axis):
    return ivy.take_along_axis(arr, indices, axis)


@with_unsupported_dtypes(
    {"2.6.0 and below": ("int8", "uint8", "int16", "float16", "bfloat16")},
    "paddle",
)
@to_ivy_arrays_and_back
def tile(x, repeat_times, name=None):
    return ivy.tile(x, repeats=repeat_times)


@to_ivy_arrays_and_back
def tolist(x):
    return ivy.to_list(x)


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "int32", "int64", "float16", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def unbind(input, axis=0):
    shape = list(input.shape)
    num_splits = shape[axis]
    shape.pop(axis)
    return tuple(x.reshape(tuple(shape)) for x in split(input, num_splits, axis=axis))


@with_supported_dtypes(
    {"2.6.0 and below": ("bool", "int32", "int64", "float16", "float32", "float64")},
    "paddle",
)
@to_ivy_arrays_and_back
def unique_consecutive(x, axis=0):
    return ivy.unique_consecutive(x, axis=axis)


@with_supported_dtypes(
    {
        "2.6.0 and below": (
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
