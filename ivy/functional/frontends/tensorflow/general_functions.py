# global
from typing import Any

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.tensorflow.func_wrapper import to_ivy_arrays_and_back
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
@to_ivy_arrays_and_back
def eye(num_rows, num_columns=None, batch_shape=None, dtype=ivy.float32, name=None):
    return ivy.eye(num_rows, num_columns, batch_shape=batch_shape, dtype=dtype)


@with_unsupported_dtypes({"2.9.0 and below": ("float16", "bfloat16")}, "tensorflow")
@to_ivy_arrays_and_back
def ones(shape, dtype=ivy.float32, name=None):
    return ivy.ones(shape, dtype=dtype)


@to_ivy_arrays_and_back
def zeros_like(input, dtype=None, name=None):
    return ivy.zeros_like(input, dtype=dtype)


def constant(
    value: Any,
    dtype: Any = None,
    shape: Any = None,
) -> EagerTensor:
    if shape:
        value = ivy.reshape(ivy.array(value, dtype=dtype), shape=shape)
        return EagerTensor(value)
    return EagerTensor(ivy.array(value, dtype=dtype))


def convert_to_tensor(
    value: Any,
    dtype: Any = None,
    dtype_hint: Any = None,
) -> Any:
    if dtype:
        return EagerTensor(ivy.array(value, dtype=dtype))
    elif dtype_hint:
        return EagerTensor(ivy.array(value, dtype=dtype_hint))
    return EagerTensor(ivy.array(value))


@to_ivy_arrays_and_back
def einsum(equation, *inputs, **kwargs):
    return ivy.einsum(equation, *inputs)


@to_ivy_arrays_and_back
def rank(input, **kwargs):
    return ivy.astype(ivy.array(input.ndim), ivy.int32)


@to_ivy_arrays_and_back
def ones_like(input, dtype=None, name=None):
    return ivy.ones_like(input, dtype=dtype)


@to_ivy_arrays_and_back
def expand_dims(input, axis, name=None):
    return ivy.expand_dims(input, axis=axis)
