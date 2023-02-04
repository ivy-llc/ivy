# global
from typing import Any
import itertools
import string

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from ivy.functional.frontends.jax.numpy import can_cast


@to_ivy_arrays_and_back
def abs(x):
    return ivy.abs(x)


@to_ivy_arrays_and_back
def acos(x):
    return ivy.acos(x)


@to_ivy_arrays_and_back
def add(x, y):
    return ivy.add(x, y)


@to_ivy_arrays_and_back
def argmax(operand, axis, index_dtype):
    return ivy.astype(ivy.argmax(operand, axis=axis), index_dtype)


@to_ivy_arrays_and_back
def argmin(operand, axis, index_dtype):
    return ivy.astype(ivy.argmin(operand, axis=axis), index_dtype)


@to_ivy_arrays_and_back
def asin(x):
    return ivy.asin(x)


@to_ivy_arrays_and_back
def atan(x):
    return ivy.atan(x)


@to_ivy_arrays_and_back
def atan2(x, y):
    return ivy.atan2(x, y)


@to_ivy_arrays_and_back
def bitwise_and(x, y):
    return ivy.bitwise_and(x, y)


@to_ivy_arrays_and_back
def bitwise_not(x):
    return ivy.bitwise_invert(x)


@to_ivy_arrays_and_back
def bitwise_or(x, y):
    return ivy.bitwise_or(x, y)


@to_ivy_arrays_and_back
def bitwise_xor(x, y):
    return ivy.bitwise_xor(x, y)


@to_ivy_arrays_and_back
def broadcast(operand, sizes):
    ret = ivy.zeros(tuple(sizes) + tuple(ivy.shape(operand)), dtype=ivy.dtype(operand))
    return ret + operand


@to_ivy_arrays_and_back
def ceil(x):
    return ivy.ceil(x)


@to_ivy_arrays_and_back
def clamp(min, x, max):
    return ivy.clip(x, min, max)


@to_ivy_arrays_and_back
def concatenate(operands, dimension):
    return ivy.concat(operands, axis=dimension)


def _format_rhs(rhs, dims):
    if not isinstance(dims, int):
        dim_nums = dims
        dims = len(dim_nums[0]) - 2
        if dim_nums[1][-1] == "O":
            dims = -1
    if dims == 1:
        return ivy.permute_dims(rhs, axes=(2, 1, 0))
    elif dims == 2:
        return ivy.permute_dims(rhs, axes=(2, 3, 1, 0))
    elif dims == 3:
        return ivy.permute_dims(rhs, axes=(2, 3, 4, 1, 0))
    else:
        return rhs


@to_ivy_arrays_and_back
def conv(
    lhs, rhs, window_strides, padding, precision=None, preferred_element_type=None
):
    if preferred_element_type:
        lhs = ivy.astype(lhs, preferred_element_type)
        rhs = ivy.astype(rhs, preferred_element_type)
    dims = len(lhs.shape) - 2
    rhs = _format_rhs(rhs, dims)
    return ivy.conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        dims=dims,
        data_format="channel_first",
    )


def _get_general_df(data_format):
    if data_format is None:
        return "channel_first"
    if data_format[1] == "C":
        return "channel_first"
    if data_format[-1] == "C":
        return "channel_last"


@to_ivy_arrays_and_back
def conv_transpose(
    lhs,
    rhs,
    strides,
    padding,
    rhs_dilation=None,
    dimension_numbers=None,
    transpose_kernel=False,
    precision=None,
    preferred_element_type=None,
):
    if preferred_element_type:
        lhs = ivy.astype(lhs, preferred_element_type)
        rhs = ivy.astype(rhs, preferred_element_type)
    if dimension_numbers[1][-1] == "O":
        rhs = ivy.swapaxes(rhs, -1, -2)
    else:
        rhs = ivy.swapaxes(rhs, 0, 1)
    return ivy.conv_general_transpose(
        lhs,
        _format_rhs(rhs, dimension_numbers),
        strides,
        padding,
        dims=len(lhs.shape) - 2,
        data_format=_get_general_df(dimension_numbers[0]),
        dilations=1 if rhs_dilation is None else rhs_dilation,
    )


@to_ivy_arrays_and_back
def conv_general_dilated(
    lhs,
    rhs,
    window_strides,
    padding,
    lhs_dilation=None,
    rhs_dilation=None,
    dimension_numbers=None,
    feature_group_count=1,
    batch_group_count=1,
    precision=None,
    preferred_element_type=None,
):
    # TODO: add support for batch_group_count
    if preferred_element_type:
        lhs = ivy.astype(lhs, preferred_element_type)
        rhs = ivy.astype(rhs, preferred_element_type)
    return ivy.conv_general_dilated(
        lhs,
        _format_rhs(rhs, dimension_numbers),
        window_strides,
        padding,
        dims=len(lhs.shape) - 2,
        data_format=_get_general_df(dimension_numbers[0]),
        x_dilations=1 if lhs_dilation is None else lhs_dilation,
        dilations=1 if rhs_dilation is None else rhs_dilation,
        feature_group_count=feature_group_count,
    )


@to_ivy_arrays_and_back
def convert_element_type(operand, new_dtype):
    assert can_cast(ivy.dtype(operand), new_dtype), "Cannot cast from {} to {}".format(
        ivy.dtype(operand), new_dtype
    )
    return ivy.astype(operand, new_dtype, copy=False)


@to_ivy_arrays_and_back
def cos(x):
    return ivy.cos(x)


@to_ivy_arrays_and_back
def cosh(x):
    return ivy.cosh(x)


@to_ivy_arrays_and_back
def cumprod(operand, axis=None, reverse=False):
    dtype = ivy.dtype(operand)
    return ivy.cumprod(operand, axis=axis, reverse=reverse).astype(dtype)


@to_ivy_arrays_and_back
def cumsum(operand, axis=None, reverse=False):
    if reverse:
        return ivy.flip(ivy.cumsum(ivy.flip(operand), axis=axis, dtype=operand.dtype))
    return ivy.cumsum(operand, axis=axis, dtype=operand.dtype)


@to_ivy_arrays_and_back
def div(x, y):
    return ivy.astype(ivy.divide(x, y), x.dtype)


@to_ivy_arrays_and_back
def dot(lhs, rhs, precision=None, preferred_element_type=None):
    ret = ivy.matmul(lhs, rhs)
    if preferred_element_type:
        ret = ivy.astype(ret, preferred_element_type, copy=False)
    return ret


@to_ivy_arrays_and_back
def dot_general(
    lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None
):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    assert len(lhs.shape) == len(rhs.shape)
    ivy.assertions.check_less(
        len(lhs.shape), 52, "number of dimensions greater than 52 is not supported"
    )
    new_id = itertools.count()
    lhs_axis_ids = [next(new_id) for _ in lhs.shape]
    rhs_axis_ids = [next(new_id) for _ in rhs.shape]
    lhs_out_axis_ids = lhs_axis_ids[:]
    rhs_out_axis_ids = rhs_axis_ids[:]
    for lhs_axis, rhs_axis in zip(lhs_contracting, rhs_contracting):
        shared_id = next(new_id)
        lhs_axis_ids[lhs_axis] = shared_id
        rhs_axis_ids[rhs_axis] = shared_id
        lhs_out_axis_ids[lhs_axis] = None
        rhs_out_axis_ids[rhs_axis] = None
    batch_ids = []
    for lhs_axis, rhs_axis in zip(lhs_batch, rhs_batch):
        shared_id = next(new_id)
        lhs_axis_ids[lhs_axis] = shared_id
        rhs_axis_ids[rhs_axis] = shared_id
        lhs_out_axis_ids[lhs_axis] = None
        rhs_out_axis_ids[rhs_axis] = None
        batch_ids.append(shared_id)
    out_axis_ids = list(
        filter(lambda x: x is not None, batch_ids + lhs_out_axis_ids + rhs_out_axis_ids)
    )
    char_list = [*string.ascii_letters]
    lhs_axis_ids = "".join(str(char_list[i]) for i in lhs_axis_ids)
    rhs_axis_ids = "".join(str(char_list[i]) for i in rhs_axis_ids)
    out_axis_ids = "".join(str(char_list[i]) for i in out_axis_ids)
    equ_str = f"{lhs_axis_ids},{rhs_axis_ids}->{out_axis_ids}"
    ret = ivy.einsum(equ_str, lhs, rhs)
    if preferred_element_type:
        ret = ivy.astype(ret, preferred_element_type, copy=False)
    return ret


@to_ivy_arrays_and_back
def eq(x, y):
    return ivy.equal(x, y)


@to_ivy_arrays_and_back
def erf(x):
    return ivy.erf(x)


@to_ivy_arrays_and_back
def exp(x):
    return ivy.exp(x)


@to_ivy_arrays_and_back
def expand_dims(array, dimensions):
    return ivy.expand_dims(array, axis=dimensions)


@to_ivy_arrays_and_back
def expm1(x):
    return ivy.expm1(x)


@to_ivy_arrays_and_back
def full(shape, fill_value, dtype=None):
    return ivy.full(shape, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def full_like(x, fill_value, dtype=None, shape=None):
    if shape is None:
        return ivy.full_like(x, fill_value, dtype=dtype)
    return ivy.full(shape, fill_value, dtype=dtype)


@to_ivy_arrays_and_back
def ge(x, y):
    return ivy.greater_equal(x, y)


@to_ivy_arrays_and_back
def gt(x, y):
    return ivy.greater(x, y)


@to_ivy_arrays_and_back
def le(x, y):
    return ivy.less_equal(x, y)


@to_ivy_arrays_and_back
def log(x):
    return ivy.log(x)


@to_ivy_arrays_and_back
def log1p(x):
    return ivy.log1p(x)


@to_ivy_arrays_and_back
def lt(x, y):
    return ivy.less(x, y)


@to_ivy_arrays_and_back
def max(x: Any, y: Any):
    return ivy.maximum(x, y)


@to_ivy_arrays_and_back
def min(x, y):
    return ivy.minimum(x, y)


@to_ivy_arrays_and_back
def mul(x, y):
    return ivy.multiply(x, y)


@to_ivy_arrays_and_back
def ne(x, y):
    return ivy.not_equal(x, y)


@to_ivy_arrays_and_back
def neg(x):
    return ivy.negative(x)


@to_ivy_arrays_and_back
def pow(x, y):
    return ivy.pow(x, y)


@to_ivy_arrays_and_back
def reciprocal(x):
    return ivy.reciprocal(x)


@to_ivy_arrays_and_back
def rem(x, y):
    return ivy.remainder(ivy.abs(x), ivy.abs(y)) * ivy.sign(x)


@to_ivy_arrays_and_back
def reshape(operand, new_sizes, dimensions=None):
    if dimensions:
        operand = ivy.permute_dims(operand, dimensions)
    return ivy.reshape(operand, new_sizes)


@to_ivy_arrays_and_back
def rev(operand, dimensions):
    return ivy.flip(operand, axis=dimensions)


@to_ivy_arrays_and_back
def round(x, rounding_method=1):
    if rounding_method == 0:
        ret = ivy.where(
            ivy.less(x, 0),
            ivy.ceil(x) - (ivy.ceil(x) - ivy.floor(x)),
            ivy.ceil(x),
        )
    elif rounding_method == 1:
        ret = ivy.ceil(x)
        ret = ivy.where(ivy.remainder(ret, 2) == 0, ret, ret - 1)
    return ivy.where(ivy.abs(x - ivy.floor(x) - 0.5) < 1e-7, ret, ivy.round(x))


@to_ivy_arrays_and_back
def rsqrt(x):
    return ivy.reciprocal(ivy.sqrt(x))


@to_ivy_arrays_and_back
def shift_left(x, y):
    return ivy.bitwise_left_shift(x, y)


@to_ivy_arrays_and_back
def sign(x):
    return ivy.sign(x)


@to_ivy_arrays_and_back
def sin(x):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def sinh(x):
    return ivy.sinh(x)


@to_ivy_arrays_and_back
def sort(operand, dimension=-1, is_stable=True, num_keys=1):
    return ivy.sort(operand, axis=dimension, stable=is_stable)


@to_ivy_arrays_and_back
def sqrt(x):
    return ivy.sqrt(x)


@to_ivy_arrays_and_back
def square(x):
    return ivy.square(x)


@to_ivy_arrays_and_back
def sub(x, y):
    return ivy.subtract(x, y)


@to_ivy_arrays_and_back
def tan(x):
    return ivy.tan(x)


@to_ivy_arrays_and_back
def transpose(operand, permutation):
    return ivy.permute_dims(operand, permutation)


@to_ivy_arrays_and_back
def shift_right_logical(x, y):
    return ivy.bitwise_right_shift(x, y)


@to_ivy_arrays_and_back
def asinh(x):
    return ivy.asinh(x)


@to_ivy_arrays_and_back
def atanh(x):
    return ivy.atanh(x)


@to_ivy_arrays_and_back
def select(pred, on_true, on_false):
    return ivy.where(pred, on_true, on_false)
