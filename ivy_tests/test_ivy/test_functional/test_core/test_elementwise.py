"""
Collection of tests for elementwise functions
"""

# global
import pytest
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# abs
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_abs(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype in ['uint16', 'uint32', 'uint64']:
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'abs',
        x=np.asarray(x, dtype=dtype))


# acosh
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_acosh(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'acosh',
        x=np.asarray(x, dtype=dtype))


# acos
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_acos(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'acos',
        x=np.asarray(x, dtype=dtype))


# add
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_add(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     if dtype == 'float16':
#         return # numpy array api doesnt support float16
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'add',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# asin
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_asin(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'asin',
        x=np.asarray(x, dtype=dtype))


# asinh
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_asinh(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'asinh',
        x=np.asarray(x, dtype=dtype))


# atan
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_atan(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'atan',
        x=np.asarray(x, dtype=dtype))


# atan2
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs, 2),
       as_variable=helpers.list_of_length(st.booleans(), 2),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=helpers.list_of_length(st.booleans(), 2),
       container=helpers.list_of_length(st.booleans(), 2),
       instance_method=st.booleans())
def test_atan2(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and 'float16' in dtype:
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'atan2',
        x1=np.asarray(x[0], dtype=dtype[0]), x2=np.asarray(x[1], dtype=dtype[1]))


# atanh
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_atanh(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'atanh',
        x=np.asarray(x, dtype=dtype))


# bitwise_and
# @given(dtype_and_x=helpers.dtype_and_values(ivy.int_dtype_strs + ('bool',)),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_bitwise_and(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'bitwise_and',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# bitwise_left_shift
# @given(dtype_and_x=helpers.dtype_and_values(ivy.int_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_bitwise_left_shift(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'bitwise_left_shift',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# bitwise_invert
@given(dtype_and_x=helpers.dtype_and_values(ivy.int_dtype_strs + ('bool',)),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_bitwise_invert(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if dtype in ['uint16', 'uint32', 'uint64']:
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'bitwise_invert',
        x=np.asarray(x, dtype=dtype))


# bitwise_or
# @given(dtype_and_x=helpers.dtype_and_values(ivy.int_dtype_strs + ('bool',)),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_bitwise_or(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'bitwise_or',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# bitwise_right_shift
# @given(dtype_and_x=helpers.dtype_and_values(ivy.int_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_bitwise_right_shift(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'bitwise_right_shift',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# bitwise_xor
# @given(dtype_and_x=helpers.dtype_and_values(ivy.int_dtype_strs + ('bool',)),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_bitwise_xor(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'bitwise_xor',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# ceil
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=st.booleans(),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 1),
#        native_array=st.booleans(),
#        container=st.booleans(),
#        instance_method=st.booleans())
# def test_ceil(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     if fw == 'torch' and dtype == 'float16':
#         return
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'ceil',
#         x=np.asarray(x, dtype=dtype))


# cos
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_cos(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'cos',
        x=np.asarray(x, dtype=dtype))


# cosh
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_cosh(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'cosh',
        x=np.asarray(x, dtype=dtype))


# divide
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_divide(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'divide',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# equal
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_equal(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'equal',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# exp
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_exp(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'exp',
        x=np.asarray(x, dtype=dtype))


# expm1
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_expm1(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'expm1',
        x=np.asarray(x, dtype=dtype))


# floor
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=st.booleans(),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 1),
#        native_array=st.booleans(),
#        container=st.booleans(),
#        instance_method=st.booleans())
# def test_floor(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     if fw in  ['torch', 'numpy'] and dtype == 'float16':
#         return
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'floor',
#         x=np.asarray(x, dtype=dtype))


# floor_divide
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_floor_divide(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'floor_divide',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# greater
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_greater(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'greater',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# greater_equal
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_greater_equal(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'greater_equal',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# isfinite
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_isfinite(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if dtype in ivy.invalid_dtype_strs:
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'isfinite',
        x=np.asarray(x, dtype=dtype))


# isinf
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_isinf(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if dtype in ivy.invalid_dtype_strs:
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'isinf',
        x=np.asarray(x, dtype=dtype))


# isnan
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_isnan(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if dtype in ivy.invalid_dtype_strs:
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'isnan',
        x=np.asarray(x, dtype=dtype))


# less
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_less(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'less',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# less_equal
# @given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_less_equal(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     if dtype in ivy.invalid_dtype_strs:
#         return
#     dtype = [dtype, dtype]
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'less_equal',
#         x1=np.asarray(x, dtype=dtype[0]), x2=np.asarray(x, dtype=dtype[1]))


# log
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_log(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'log',
        x=np.asarray(x, dtype=dtype))


# log1p
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_log1p(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'log1p',
        x=np.asarray(x, dtype=dtype))


# log2
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_log2(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'log2',
        x=np.asarray(x, dtype=dtype))


# log10
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_log10(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'log10',
        x=np.asarray(x, dtype=dtype))


# logaddexp
@given(dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs, 2),
       as_variable=helpers.list_of_length(st.booleans(), 2),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 2),
       native_array=helpers.list_of_length(st.booleans(), 2),
       container=helpers.list_of_length(st.booleans(), 2),
       instance_method=st.booleans())
def test_logaddexp(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    if fw == 'torch' and 'float16' in dtype:
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'logaddexp',
        x1=np.asarray(x[0], dtype=dtype[0]), x2=np.asarray(x[1], dtype=dtype[1]))


# logical_and
# @given(dtype_and_x=helpers.dtype_and_values(('bool',), 2),
#        as_variable=helpers.list_of_length(st.booleans(), 2),
#        with_out=st.booleans(),
#        num_positional_args=st.integers(0, 2),
#        native_array=helpers.list_of_length(st.booleans(), 2),
#        container=helpers.list_of_length(st.booleans(), 2),
#        instance_method=st.booleans())
# def test_logical_and(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
#     dtype, x = dtype_and_x
#     helpers.test_array_function(
#         dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'logical_and',
#         x1=np.asarray(x[0], dtype=dtype[0]), x2=np.asarray(x[1], dtype=dtype[1]))


# logical_not
@given(dtype_and_x=helpers.dtype_and_values(('bool',)),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 2),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans())
def test_logical_not(dtype_and_x, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw):
    dtype, x = dtype_and_x
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'logical_not',
        x=np.asarray(x, dtype=dtype))


# logical_or
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_logical_or(with_out, native_array):
    x1 = ivy.array([0, 1, 1], dtype='bool')
    x2 = ivy.array([0, 1, 1], dtype='bool')
    out = ivy.array([0, 0, 0], dtype='bool')
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.logical_or(x1, x2, out=out)
    else:
        ret = ivy.logical_or(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# logical_xor
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_logical_xor(with_out, native_array):
    x1 = ivy.array([0, 1, 1], dtype='bool')
    x2 = ivy.array([0, 1, 1], dtype='bool')
    out = ivy.array([0, 0, 0], dtype='bool')
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.logical_xor(x1, x2, out=out)
    else:
        ret = ivy.logical_xor(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# multiply
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_multiply(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x1 = ivy.variable(x1)
        x2 = ivy.variable(x2)
        out = ivy.variable(out)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.multiply(x1, x2, out=out)
    else:
        ret = ivy.multiply(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# negative
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_negative(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.negative(x, out=out)
    else:
        ret = ivy.negative(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# not_equal
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_not_equal(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([0, 0, 0], dtype='bool')
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x1 = ivy.variable(x1)
        x2 = ivy.variable(x2)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.not_equal(x1, x2, out=out)
    else:
        ret = ivy.not_equal(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# positive
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_positive(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.positive(x, out=out)
    else:
        ret = ivy.positive(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# pow
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_pow(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x1 = ivy.variable(x1)
        x2 = ivy.variable(x2)
        out = ivy.variable(out)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.pow(x1, x2, out=out)
    else:
        ret = ivy.pow(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# remainder
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_remainder(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'torch' and dtype == 'bfloat16':
        pytest.skip("torch remainder doesnt support bfloat16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x1 = ivy.variable(x1)
        out = ivy.variable(out)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.remainder(x1, x2, out=out)
    else:
        ret = ivy.remainder(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# round
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_round(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'torch' and dtype == 'float16':
        pytest.skip("torch round doesnt allow float16")
    if ivy.current_framework_str() == 'tensorflow' and dtype == 'bfloat16':
        pytest.skip("tf round doesnt allow bfloat16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.round(x, out=out)
    else:
        ret = ivy.round(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# sign
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_sign(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.sign(x, out=out)
    else:
        ret = ivy.sign(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# sin
@given(dtype=st.sampled_from(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_sin(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'sin',
        x=np.asarray(x, dtype=dtype))


# sinh
@given(dtype=st.sampled_from(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_sinh(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'sinh',
        x=np.asarray(x, dtype=dtype))


# square
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_square(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.square(x, out=out)
    else:
        ret = ivy.square(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# sqrt
@given(dtype=st.sampled_from(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_sqrt(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'sqrt',
        x=np.asarray(x, dtype=dtype))


# subtract
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_subtract(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x1 = ivy.variable(x1)
        x2 = ivy.variable(x2)
        out = ivy.variable(out)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.subtract(x1, x2, out=out)
    else:
        ret = ivy.subtract(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# tan
@given(dtype=st.sampled_from(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_tan(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'tan',
        x=np.asarray(x, dtype=dtype))


# tanh
@given(dtype=st.sampled_from(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_tanh(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'tanh',
        x=np.asarray(x, dtype=dtype))


# trunc
@pytest.mark.parametrize(
    "dtype", ivy.numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_trunc(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'torch' and dtype == 'float16':
        pytest.skip("torch trunc doesnt allow float16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.trunc(x, out=out)
    else:
        ret = ivy.trunc(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# Extra #
# ------#


# erf
@given(dtype=st.sampled_from(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_erf(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'erf',
        x=np.asarray(x, dtype=dtype))

# add tests for minimum, maximum
