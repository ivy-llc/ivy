# global
from typing import Any
import itertools
import string
import builtins

# local
import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back
from ivy.func_wrapper import with_unsupported_dtypes, frontend_outputs_to_ivy_arrays

_slice = builtins.slice


@to_ivy_arrays_and_back
def imag(x):
    return ivy.imag(x)


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


@to_ivy_arrays_and_back
def conv(
    lhs, rhs, window_strides, padding, precision=None, preferred_element_type=None
):
    if preferred_element_type:
        lhs = ivy.astype(lhs, preferred_element_type)
        rhs = ivy.astype(rhs, preferred_element_type)
    dims = len(lhs.shape) - 2
    return ivy.conv_general_dilated(
        lhs,
        rhs,
        window_strides,
        padding,
        dims=dims,
        data_format="channel_first",
        filter_format="channel_first",
    )


def _dimension_numbers(dimension_numbers, lhs_len, transp=False):
    if dimension_numbers is None:
        if transp:
            iota = (0, lhs_len - 1, *range(1, lhs_len - 1))
            iotb = (lhs_len - 1, lhs_len - 2, *range(0, lhs_len - 2))
            return iota, iotb, iota
        else:
            iota = tuple(range(lhs_len))
            return iota, iota, iota
    elif isinstance(dimension_numbers[0], (tuple, list)):
        return dimension_numbers
    else:
        lhs_spec, rhs_spec, out_spec = dimension_numbers

        def getperm(spec, charpair):
            spatial = (i for i, c in enumerate(spec) if c not in charpair)
            if spec is not rhs_spec:
                spatial = sorted(spatial, key=lambda i: rhs_spec.index(spec[i]))
            return (spec.index(charpair[0]), spec.index(charpair[1])) + tuple(spatial)

        charpairs = ("N", "C"), ("O", "I"), ("N", "C")
        lhs_spec, rhs_spec, out_spec = map(getperm, dimension_numbers, charpairs)
        return lhs_spec, rhs_spec, out_spec


def _argsort_tuple(the_tuple):
    return tuple([i for i, _ in sorted(enumerate(the_tuple), key=lambda x: x[1])])


def _conv_transpose_padding(k, s, padding):
    if padding == "SAME":
        pad_len = k + s - 2
        if s > k - 1:
            pad_a = k - 1
        else:
            pad_a = int(ivy.to_scalar(ivy.ceil(pad_len / 2)))
    elif padding == "VALID":
        pad_len = k + s - 2 + ivy.to_scalar(ivy.maximum(k - s, 0))
        pad_a = k - 1
    else:
        raise ValueError("Padding mode must be `SAME` or `VALID`.")
    pad_b = pad_len - pad_a
    return pad_a, pad_b


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
    # TODO: add support for transpose_kernel
    if preferred_element_type:
        lhs = ivy.astype(lhs, preferred_element_type)
        rhs = ivy.astype(rhs, preferred_element_type)
    dims = len(lhs.shape) - 2
    dim_nums = _dimension_numbers(dimension_numbers, dims + 2, transp=True)
    rhs_spec = tuple([dim_nums[1][i] for i in (*range(2, dims + 2), 1, 0)])
    rhs_dilation = 1 if rhs_dilation is None else rhs_dilation
    if isinstance(padding, str):
        k_sdims = [rhs.shape[i] for i in rhs_spec[:-2]]
        effective_k_size = map(lambda k, r: (k - 1) * r + 1, k_sdims, rhs_dilation)
        padding = [
            _conv_transpose_padding(k, s, padding)
            for k, s in zip(effective_k_size, strides)
        ]
    return ivy.permute_dims(
        ivy.conv_general_dilated(
            ivy.permute_dims(lhs, axes=dim_nums[0]),
            ivy.permute_dims(rhs, axes=rhs_spec),
            1,
            padding,
            dilations=rhs_dilation,
            x_dilations=strides,
            dims=dims,
            data_format="channel_first",
        ),
        axes=_argsort_tuple(dim_nums[2]),
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
    dims = len(lhs.shape) - 2
    dim_nums = _dimension_numbers(dimension_numbers, dims + 2)
    rhs_spec = tuple([dim_nums[1][i] for i in (*range(2, dims + 2), 1, 0)])
    return ivy.permute_dims(
        ivy.conv_general_dilated(
            ivy.permute_dims(lhs, axes=dim_nums[0]),
            ivy.permute_dims(rhs, axes=rhs_spec),
            window_strides,
            padding,
            dims=dims,
            data_format="channel_first",
            x_dilations=1 if lhs_dilation is None else lhs_dilation,
            dilations=1 if rhs_dilation is None else rhs_dilation,
            feature_group_count=feature_group_count,
        ),
        axes=_argsort_tuple(dim_nums[2]),
    )


@to_ivy_arrays_and_back
def convert_element_type(operand, new_dtype):
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
def batch_matmul(lhs, rhs, precision=None):
    if lhs.ndim < 2 or rhs.ndim < 2:
        raise ValueError(
            "Arguments to batch_matmul must be at least 2D, got {}, {}".format(
                lhs.ndim, rhs.ndim
            )
        )
    if lhs.ndim != rhs.ndim:
        raise ValueError(
            "Arguments to batch_matmul must have same ndim, got {}, {}".format(
                lhs.ndim, rhs.ndim
            )
        )
    return ivy.matmul(lhs, rhs).astype(lhs.dtype)


@with_unsupported_dtypes({"0.4.5 and below": ("bool",)}, "jax")
@to_ivy_arrays_and_back
def dot_general(
    lhs, rhs, dimension_numbers, precision=None, preferred_element_type=None
):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    ivy.utils.assertions.check_less(
        len(lhs.shape),
        52,
        "number of dimensions greater than 52 is not supported",
        as_array=False,
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


@with_unsupported_dtypes({"0.4.5 and below": ("complex",)}, "jax")
@to_ivy_arrays_and_back
def ge(x, y):
    return ivy.greater_equal(x, y)


@with_unsupported_dtypes({"0.4.5 and below": ("complex",)}, "jax")
@to_ivy_arrays_and_back
def gt(x, y):
    return ivy.greater(x, y)


@with_unsupported_dtypes({"0.4.5 and below": ("complex",)}, "jax")
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
def pad(operand, padding_value, padding_config):
    return ivy.pad(
        operand, padding_config, mode="dilated", constant_values=padding_value
    )


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
    return ivy.sign(x, np_variant=False)


@to_ivy_arrays_and_back
def sin(x):
    return ivy.sin(x)


@to_ivy_arrays_and_back
def sinh(x):
    return ivy.sinh(x)


@to_ivy_arrays_and_back
def slice(operand, start_indices, limit_indices, strides=None):
    strides = [1] * len(operand.shape) if strides is None else strides

    full_slice = ()
    for i, _ in enumerate(operand.shape):
        strides_i = int(strides[i])
        start_i = int(start_indices[i])
        limit_i = int(limit_indices[i])
        full_slice += (_slice(start_i, limit_i, strides_i),)
    return operand[full_slice]


@to_ivy_arrays_and_back
def slice_in_dim(operand, start_index, limit_index, stride=1, axis=0):
    start_indices = [0] * operand.ndim
    limit_indices = list(operand.shape)
    strides = [1] * operand.ndim

    len_axis = operand.shape[axis]
    start_index_int = start_index if start_index is not None else 0
    limit_index_int = limit_index if limit_index is not None else len_axis

    if start_index_int < 0:
        start_index_int = start_index_int + len_axis
    if limit_index_int < 0:
        limit_index_int = limit_index_int + len_axis

    axis = int(axis)
    start_indices[axis] = start_index_int
    limit_indices[axis] = limit_index_int
    strides[axis] = int(stride)
    return slice(operand, start_indices, limit_indices, strides)


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


# top_k
@to_ivy_arrays_and_back
def top_k(operand, k):
    values, indices = ivy.top_k(operand, k, axis=-1)
    indices = ivy.astype(indices, ivy.int32, copy=False)
    return [values, indices]


@to_ivy_arrays_and_back
def reduce_window(
    operand,
    init_value,
    computation,
    window_dimensions,
    window_strides,
    padding,
    base_dilation=None,
    window_dilation=None,
):
    computation = frontend_outputs_to_ivy_arrays(computation)
    return ivy.reduce_window(
        operand,
        init_value,
        computation,
        window_dimensions,
        window_strides=window_strides,
        padding=padding,
        base_dilation=base_dilation,
        window_dilation=window_dilation,
    )


@to_ivy_arrays_and_back
def squeeze(array, dimensions):
    return ivy.squeeze(array, axis=dimensions)


@to_ivy_arrays_and_back
def real(x):
    return ivy.real(x)


@to_ivy_arrays_and_back
def nextafter(x1, x2):
    return ivy.nextafter(x1, x2)


@to_ivy_arrays_and_back
def conj(x):
    return ivy.conj(x)


@to_ivy_arrays_and_back
def is_finite(x):
    return ivy.isfinite(x)


@to_ivy_arrays_and_back
def cbrt(x):
    return ivy.pow(x, 1 / 3)


@to_ivy_arrays_and_back
def erfc(x):
    value = ivy.erf(x)
    value = (1.0 - value) if value is not None else None
    return value
