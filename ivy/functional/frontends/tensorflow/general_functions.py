# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.frontends.tensorflow.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_tf_dtype,
    to_ivy_dtype,
)
from ivy.functional.frontends.tensorflow.tensor import EagerTensor
import ivy.functional.frontends.tensorflow as tf_frontend


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
def convert_to_tensor(value, dtype=None, dtype_hint=None, name=None):
    if dtype:
        return tf_frontend.cast(value, dtype)
    elif dtype_hint:
        return tf_frontend.cast(value, dtype_hint)
    if hasattr(value, "ivy_array"):
        return EagerTensor(value.ivy_array)
    return EagerTensor(value)


@to_ivy_arrays_and_back
def einsum(equation, *inputs, **kwargs):
    return ivy.einsum(equation, *inputs)


@to_ivy_arrays_and_back
def reshape(tensor, shape, name=None):
    shape = shape.to_list() if ivy.is_array(shape) else shape
    return ivy.reshape(tensor, shape=shape)


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
def matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
    name=None,
):
    return ivy.matmul(a, b)


@to_ivy_arrays_and_back
def shape(input, out_type=ivy.int32, name=None):
    out_type = to_ivy_dtype(out_type)
    if out_type in ["int32", "int64"]:
        return ivy.array(ivy.shape(input), dtype=out_type)
    else:
        return ivy.array(ivy.shape(input), dtype="int64")


@to_ivy_arrays_and_back
def shape_n(input, out_type=ivy.int32, name=None):
    out_type = to_ivy_dtype(out_type)
    if out_type in ["int32", "int64"]:
        return [ivy.array(ivy.shape(i), dtype=out_type) for i in input]
    else:
        return [ivy.array(ivy.shape(i), dtype="int64") for i in input]


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
def identity(input, name=None):
    return ivy.copy_array(input)


def stack(values, axis=0, name="stack"):
    return ivy.stack(values, axis=axis)


@to_ivy_arrays_and_back
def is_tensor(x, name=None):
    return ivy.is_array(x)


@to_ivy_arrays_and_back
def gather(params, indices, axis=None, batch_dims=0, name=None):
    return ivy.gather(params, indices, axis=axis, batch_dims=batch_dims)


@to_ivy_arrays_and_back
def gather_nd(params, indices, batch_dims=0, name=None):
    return ivy.gather_nd(params, indices, batch_dims=batch_dims)


@to_ivy_arrays_and_back
def pad(tensor, paddings, mode="CONSTANT", constant_values=0, name=None):
    paddings = paddings.to_list() if ivy.is_array(paddings) else paddings
    return ivy.pad(tensor, paddings, mode=mode.lower(), constant_values=constant_values)


@to_ivy_arrays_and_back
def transpose(a, perm=None, conjugate=False, name="transpose"):
    # handle conjugate when ivy supports complex numbers
    if perm is not None:
        return ivy.permute_dims(a, axes=perm)
    n = a.ndim
    perm = ivy.arange(n - 1, -1, -1)
    return ivy.permute_dims(a, axes=perm)


@to_ivy_arrays_and_back
def strided_slice(
    input_,
    begin,
    end,
    strides=None,
    begin_mask=0,
    end_mask=0,
    ellipsis_mask=0,
    new_axis_mask=0,
    shrink_axis_mask=0,
    var=None,
    name=None,
):
    def num_to_bit_list(number):
        return list(map(int, "{:0{size}b}".format(number, size=len(input_.shape))))

    begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask = list(
        map(
            num_to_bit_list,
            [begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask],
        )
    )

    full_slice = ()
    need_ellipsis = False
    if len(input_.shape) - len(begin.shape) > 0:
        need_ellipsis = True
    for i, _ in enumerate(begin.shape):
        if need_ellipsis and ellipsis_mask[i]:
            full_slice += (...,)
            need_ellipsis = False
        else:
            if new_axis_mask[i]:
                full_slice += (ivy.newaxis,)
            else:
                if not begin_mask[i] or shrink_axis_mask[i]:
                    begin_i = int(begin[i])
                else:
                    begin_i = None
                if shrink_axis_mask[i]:
                    end_i = begin_i + 1
                elif end_mask[i]:
                    end_i = None
                else:
                    end_i = int(end[i])
                full_slice += (slice(begin_i, end_i, int(strides[i])),)
    return input_[full_slice]


@to_ivy_arrays_and_back
def slice(input_, begin, size, name=None):
    return strided_slice(input_, begin, begin + size, [1] * len(size))


@to_ivy_arrays_and_back
def linspace(start, stop, num, name=None, axis=0):
    return ivy.linspace(start, stop, num, axis=axis)


@to_ivy_arrays_and_back
def realdiv(x, y, name=None):
    return ivy.divide(x, y)


@with_unsupported_dtypes({"2.9.0 and below": ("uint16",)}, "tensorflow")
@to_ivy_arrays_and_back
def tile(input, multiples, name=None):
    return ivy.tile(input, multiples)


@to_ivy_arrays_and_back
def one_hot(
    indices: ivy.array,
    depth: int,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    device=None,
    out=None,
):
    return ivy.one_hot(indices, depth)


@to_ivy_arrays_and_back
def where(condition: ivy.array, x=None, y=None, name=None):
    if x is None and y is None:
        return ivy.argwhere(condition)
    else:
        return ivy.where(condition, x, y)
