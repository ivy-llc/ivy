# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
    to_ivy_dtype,
)
from ivy.functional.frontends.tensorflow.tensor import EagerTensor


@to_ivy_arrays_and_back
def argsort(values, axis=-1, direction="ASCENDING", stable=False, name=None):
    if direction == "DESCENDING":
        descending = True
    else:
        descending = False
    return ivy.argsort(values, axis=axis, descending=descending, stable=stable).astype(
        "int32"
    )


@to_ivy_arrays_and_back
def clip_by_value(t, clip_value_min, clip_value_max):
    ivy.assertions.check_all_or_any_fn(
        clip_value_min,
        clip_value_max,
        fn=ivy.exists,
        type="all",
        message="clip_value_min and clip_value_max must exist",
    )
    t = ivy.array(t)
    return ivy.clip(t, clip_value_min, clip_value_max)


@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
@handle_tf_dtype
@to_ivy_arrays_and_back
def eye(num_rows, num_columns=None, batch_shape=None, dtype=ivy.float32, name=None):
    return ivy.eye(num_rows, num_columns, batch_shape=batch_shape, dtype=dtype)


@to_ivy_arrays_and_back
def fill(dims, value, name=None):
    return ivy.full(dims, value)


@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
@handle_tf_dtype
@to_ivy_arrays_and_back
def ones(shape, dtype=ivy.float32, name=None):
    return ivy.ones(shape, dtype=dtype)


@handle_tf_dtype
@to_ivy_arrays_and_back
def zeros_like(input, dtype=None, name=None):
    return ivy.zeros_like(input, dtype=dtype)


@handle_tf_dtype
def constant(value, dtype=None, shape=None, name=None):
    if shape is not None:
        value = ivy.reshape(value, shape=shape)
    if dtype is not None:
        return EagerTensor(ivy.astype(value, dtype))
    return EagerTensor(value)


@handle_tf_dtype
def convert_to_tensor(value, dtype, dtype_hint, name=None):
    if dtype:
        return EagerTensor(ivy.astype(value, dtype))
    elif dtype_hint:
        return EagerTensor(ivy.astype(value, dtype_hint))
    return EagerTensor(value)


@to_ivy_arrays_and_back
def einsum(equation, *inputs, **kwargs):
    return ivy.einsum(equation, *inputs)


@to_ivy_arrays_and_back
def rank(input, **kwargs):
    return ivy.astype(ivy.array(input.ndim), ivy.int32)


@handle_tf_dtype
@to_ivy_arrays_and_back
def ones_like(input, dtype=None, name=None):
    return ivy.ones_like(input, dtype=dtype)


@handle_tf_dtype
@to_ivy_arrays_and_back
def zeros(shape, dtype=ivy.float32, name=None):
    return ivy.zeros(shape=shape, dtype=dtype)


@to_ivy_arrays_and_back
def expand_dims(input, axis, name=None):
    return ivy.expand_dims(input, axis=axis)


@to_ivy_arrays_and_back
def squeeze(input, axis=None, name=None):
    return ivy.squeeze(input, axis=axis)


@to_ivy_arrays_and_back
def concat(values, axis, name=None):
    return ivy.concat(values, axis=axis)


@to_ivy_arrays_and_back
def shape(input, out_type=ivy.int32, name=None):
    out_type = to_ivy_dtype(out_type)
    if out_type in ["int32", "int64"]:
        return ivy.array(ivy.shape(input), dtype=out_type)
    else:
        return ivy.array(ivy.shape(input), dtype="int64")


@with_unsupported_dtypes({"2.10.0 and below": ("float16", "bfloat16")}, "tensorflow")
@handle_tf_dtype
@to_ivy_arrays_and_back
def range(start, limit=None, delta=1, /, *, dtype=None, name=None):
    return ivy.arange(start, limit, delta, dtype=dtype)


@to_ivy_arrays_and_back
def sort(values, axis=-1, direction="ASCENDING", name=None):
    descending = True
    if direction == "ASCENDING":
        descending = False
    else:
        ivy.assertions.check_equal(
            direction,
            "DESCENDING",
            message="Argument `direction` should be one of 'ASCENDING' or 'DESCENDING'",
        )
    return ivy.sort(values, axis=axis, descending=descending)


@to_ivy_arrays_and_back
def searchsorted(sorted_sequence, values, side="left", out_type="int32"):
    out_type = to_ivy_dtype(out_type)
    if out_type not in ["int32", "int64"]:
        out_type = "int64"
    return ivy.searchsorted(sorted_sequence, values, side=side, ret_dtype=out_type)


@to_ivy_arrays_and_back
def stack(values, axis=0, name="stack"):
    return ivy.stack(values, axis=axis)


@to_ivy_arrays_and_back
def gather(params, indices, axis=None, batch_dims=0, name=None):
    return ivy.gather(params, indices, axis=axis, batch_dims=batch_dims)


@to_ivy_arrays_and_back
def gather_nd(params, indices, batch_dims=0, name=None):
    return ivy.gather_nd(params, indices, batch_dims=batch_dims)
