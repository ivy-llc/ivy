# global
import logging

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import (
    to_ivy_arrays_and_back,
)
from ivy.functional.frontends.numpy.func_wrapper import from_zero_dim_arrays_to_scalar
from ivy.func_wrapper import (
    with_supported_device_and_dtypes,
    with_unsupported_dtypes,
)


@to_ivy_arrays_and_back
@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
            "float16",
            "bfloat16",
        )
    },
    "jax",
)
def argmax(a, axis=None, out=None, keepdims=False):
    return ivy.argmax(a, axis=axis, keepdims=keepdims, out=out, dtype=ivy.int64)


# argmin
@to_ivy_arrays_and_back
@with_supported_device_and_dtypes(
    {
        "0.4.20 and below": {
            "cpu": (
                "int16",
                "int32",
                "int64",
                "float32",
                "float64",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
            )
        }
    },
    "jax",
)
def argmin(a, axis=None, out=None, keepdims=None):
    if a is not None:
        if isinstance(a, list):
            if all(isinstance(elem, ivy.Array) for elem in a):
                if len(a) == 1:
                    a = a[0]
                else:
                    return [
                        ivy.argmin(
                            ivy.to_native_arrays(elem),
                            axis=axis,
                            out=out,
                            keepdims=keepdims,
                        )
                        for elem in a
                    ]
            else:
                raise ValueError(
                    "Input 'a' must be an Ivy array or a list of Ivy arrays."
                )

        if not isinstance(a, ivy.Array):
            raise TypeError("Input 'a' must be an array.")

        if a.size == 0:
            raise ValueError("Input 'a' must not be empty.")

        return ivy.argmin(a, axis=axis, out=out, keepdims=keepdims)
    else:
        raise ValueError("argmin takes at least 1 argument.")


@to_ivy_arrays_and_back
def argsort(a, axis=-1, kind="stable", order=None):
    if kind != "stable":
        logging.warning(
            "'kind' argument to argsort is ignored; only 'stable' sorts are supported."
        )
    if order is not None:
        raise ivy.utils.exceptions.IvyError(
            "'order' argument to argsort is not supported."
        )

    return ivy.argsort(a, axis=axis)


@to_ivy_arrays_and_back
def argwhere(a, /, *, size=None, fill_value=None):
    if size is None and fill_value is None:
        return ivy.argwhere(a)

    result = ivy.matrix_transpose(
        ivy.vstack(ivy.nonzero(a, size=size, fill_value=fill_value))
    )
    num_of_dimensions = a.ndim

    if num_of_dimensions == 0:
        return result[:0].reshape(result.shape[0], 0)

    return result.reshape(result.shape[0], num_of_dimensions)


@with_unsupported_dtypes(
    {
        "0.4.24 and below": (
            "uint8",
            "int8",
            "bool",
        )
    },
    "jax",
)
@to_ivy_arrays_and_back
def count_nonzero(a, axis=None, keepdims=False):
    return ivy.astype(ivy.count_nonzero(a, axis=axis, keepdims=keepdims), "int64")


@to_ivy_arrays_and_back
def extract(condition, arr):
    if condition.dtype is not bool:
        condition = condition != 0
    return arr[condition]


@to_ivy_arrays_and_back
def flatnonzero(a):
    return ivy.nonzero(ivy.reshape(a, (-1,)))


@to_ivy_arrays_and_back
def lexsort(keys, /, *, axis=-1):
    return ivy.lexsort(keys, axis=axis)


@to_ivy_arrays_and_back
def msort(a):
    return ivy.msort(a)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanargmax(a, /, *, axis=None, out=None, keepdims=False):
    if out is not None:
        raise NotImplementedError(
            "The 'out' argument to jnp.nanargmax is not supported."
        )
    nan_mask = ivy.isnan(a)
    if not ivy.any(nan_mask):
        return ivy.argmax(a, axis=axis, keepdims=keepdims)

    a = ivy.where(nan_mask, -ivy.inf, a)
    res = ivy.argmax(a, axis=axis, keepdims=keepdims)
    return ivy.where(ivy.all(nan_mask, axis=axis, keepdims=keepdims), -1, res)


@to_ivy_arrays_and_back
@from_zero_dim_arrays_to_scalar
def nanargmin(a, /, *, axis=None, out=None, keepdims=None):
    if out is not None:
        raise NotImplementedError(
            "The 'out' argument to jnp.nanargmax is not supported."
        )
    nan_mask = ivy.isnan(a)
    if not ivy.any(nan_mask):
        return ivy.argmin(a, axis=axis, keepdims=keepdims)

    a = ivy.where(nan_mask, ivy.inf, a)
    res = ivy.argmin(a, axis=axis, keepdims=keepdims)
    return ivy.where(ivy.all(nan_mask, axis=axis, keepdims=keepdims), -1, res)


@to_ivy_arrays_and_back
def nonzero(a, *, size=None, fill_value=None):
    return ivy.nonzero(a, size=size, fill_value=fill_value)


@to_ivy_arrays_and_back
def searchsorted(a, v, side="left", sorter=None, *, method="scan"):
    return ivy.searchsorted(a, v, side=side, sorter=sorter, ret_dtype="int32")


@to_ivy_arrays_and_back
def sort(a, axis=-1, kind="quicksort", order=None):
    # todo: handle case where order is not None
    return ivy.sort(a, axis=axis)


@to_ivy_arrays_and_back
def sort_complex(a):
    return ivy.sort(a)


@to_ivy_arrays_and_back
def unique(
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    *,
    size=None,
    fill_value=None,
):
    uniques = list(ivy.unique_all(ar, axis=axis))
    if size is not None:
        fill_value = fill_value if fill_value is not None else 1  # default fill_value 1
        pad_len = size - len(uniques[0])
        if pad_len > 0:
            # padding
            num_dims = len(uniques[0].shape) - 1
            padding = [(0, 0)] * num_dims + [(0, pad_len)]
            uniques[0] = ivy.pad(uniques[0], padding, constant_values=fill_value)
            # padding the indices and counts with zeros
            for i in range(1, len(uniques)):
                if i == 2:
                    continue
                uniques[i] = ivy.pad(uniques[i], padding[-1], constant_values=0)
        else:
            for i in range(len(uniques)):
                uniques[i] = uniques[i][..., :size]
    # constructing a list of bools for indexing
    bools = [return_index, return_inverse, return_counts]
    # indexing each element whose condition is True except for the values
    uniques = [uniques[0]] + [uni for idx, uni in enumerate(uniques[1:]) if bools[idx]]
    return uniques[0] if len(uniques) == 1 else uniques


@to_ivy_arrays_and_back
def where(condition, x=None, y=None, *, size=None, fill_value=0):
    if x is None and y is None:
        return nonzero(condition, size=size, fill_value=fill_value)
    if x is not None and y is not None:
        return ivy.where(condition, x, y)
    else:
        raise ValueError("Both x and y should be given.")
