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
@given(dtype_and_x=helpers.numeric_dtype_and_values(),
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
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_acosh(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'acosh',
        x=np.asarray(x, dtype=dtype))


# acos
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_acos(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'acos',
        x=np.asarray(x, dtype=dtype))


# add
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_add(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'numpy' and dtype == 'float16':
        pytest.skip("numpy array api doesnt support float16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    y = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        y = ivy.variable(y)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        y = y.data
        out = out.data
    if with_out:
        ret = ivy.add(x, y, out=out)
    else:
        ret = ivy.add(x, y)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# asin
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_asin(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'asin',
        x=np.asarray(x, dtype=dtype))


# asinh
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_asinh(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'asinh',
        x=np.asarray(x, dtype=dtype))


# atan
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_atan(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'atan',
        x=np.asarray(x, dtype=dtype))


# atan2
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_atan2(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'atan2',
        x1=np.asarray(x, dtype=dtype), x2=np.asarray(x, dtype=dtype))


# atanh
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_atanh(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'atanh',
        x=np.asarray(x, dtype=dtype))


# bitwise_and
@pytest.mark.parametrize(
    "dtype", ivy.all_int_dtype_strs + ('bool',))
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_bitwise_and(dtype, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([0, 1, 1], dtype=dtype)
    x2 = ivy.array([0, 1, 1], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.bitwise_and(x1, x2, out=out)
    else:
        ret = ivy.bitwise_and(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# bitwise_left_shift
@pytest.mark.parametrize(
    "dtype", ivy.all_int_dtype_strs)
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_bitwise_left_shift(dtype, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([0, 1, 1], dtype=dtype)
    x2 = ivy.array([0, 1, 1], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.bitwise_left_shift(x1, x2, out=out)
    else:
        ret = ivy.bitwise_left_shift(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# bitwise_invert
@pytest.mark.parametrize(
    "dtype", ivy.all_int_dtype_strs + ('bool',))
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_bitwise_invert(dtype, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([0, 1, 1], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.bitwise_invert(x, out=out)
    else:
        ret = ivy.bitwise_invert(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# bitwise_or
@pytest.mark.parametrize(
    "dtype", ivy.all_int_dtype_strs + ('bool',))
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_bitwise_or(dtype, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([0, 1, 1], dtype=dtype)
    x2 = ivy.array([0, 1, 1], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.bitwise_or(x1, x2, out=out)
    else:
        ret = ivy.bitwise_or(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# bitwise_right_shift
@pytest.mark.parametrize(
    "dtype", ivy.all_int_dtype_strs)
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_bitwise_right_shift(dtype, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([0, 1, 1], dtype=dtype)
    x2 = ivy.array([0, 1, 1], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.bitwise_right_shift(x1, x2, out=out)
    else:
        ret = ivy.bitwise_right_shift(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# bitwise_xor
@pytest.mark.parametrize(
    "dtype", ivy.all_int_dtype_strs + ('bool',))
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_bitwise_xor(dtype, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([0, 1, 1], dtype=dtype)
    x2 = ivy.array([0, 1, 1], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.bitwise_xor(x1, x2, out=out)
    else:
        ret = ivy.bitwise_xor(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# ceil
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_ceil(dtype, as_variable, with_out, native_array):
    # rest tests out argument
    if ivy.current_framework_str() == 'torch' and dtype == 'float16':
        pytest.skip("torch ceil doesnt allow float16")
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
        ret = ivy.ceil(x, out=out)
    else:
        ret = ivy.ceil(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# cos
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_cos(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'cos',
        x=np.asarray(x, dtype=dtype))


# cosh
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_cosh(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'cosh',
        x=np.asarray(x, dtype=dtype))


# divide
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_divide(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='float64')
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
        ret = ivy.divide(x1, x2, out=out)
    else:
        ret = ivy.divide(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# equal
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_equal(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='bool')
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
        ret = ivy.equal(x1, x2, out=out)
    else:
        ret = ivy.equal(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# exp
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_exp(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'exp',
        x=np.asarray(x, dtype=dtype))


# expm1
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_expm1(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'expm1',
        x=np.asarray(x, dtype=dtype))


# floor
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_floor(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() in ['torch', 'numpy'] and dtype == 'float16':
        pytest.skip("torch and numpy array api dont allow float16")
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
        ret = ivy.floor(x, out=out)
    else:
        ret = ivy.floor(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# floor_divide
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_floor_divide(dtype, as_variable, with_out, native_array):
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
        ret = ivy.floor_divide(x1, x2, out=out)
    else:
        ret = ivy.floor_divide(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# greater
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_greater(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='bool')
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
        ret = ivy.greater(x1, x2, out=out)
    else:
        ret = ivy.greater(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# greater_equal
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_greater_equal(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='bool')
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
        ret = ivy.greater_equal(x1, x2, out=out)
    else:
        ret = ivy.greater_equal(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# isfinite
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_isfinite(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='bool')
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.isfinite(x, out=out)
    else:
        ret = ivy.isfinite(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# isinf
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_isinf(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='bool')
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.isinf(x, out=out)
    else:
        ret = ivy.isinf(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# isnan
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_isnan(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='bool')
    if as_variable:
        if not ivy.is_float_dtype(dtype):
            pytest.skip("only floating point variables are supported")
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.isnan(x, out=out)
    else:
        ret = ivy.isnan(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# less
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_less(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='bool')
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
        ret = ivy.less(x1, x2, out=out)
    else:
        ret = ivy.less(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# less_equal
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_less_equal(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype='bool')
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
        ret = ivy.less_equal(x1, x2, out=out)
    else:
        ret = ivy.less_equal(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# log
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_log(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'log',
        x=np.asarray(x, dtype=dtype))


# log1p
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_log1p(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'log1p',
        x=np.asarray(x, dtype=dtype))


# log2
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_log2(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'log2',
        x=np.asarray(x, dtype=dtype))


# log10
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_log10(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'log10',
        x=np.asarray(x, dtype=dtype))


# logaddexp
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
       as_variable=st.booleans(),
       with_out=st.booleans(),
       num_positional_args=st.integers(0, 1),
       native_array=st.booleans(),
       container=st.booleans(),
       instance_method=st.booleans(),
       x=st.lists(st.floats()))
def test_logaddexp(dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, x):
    if fw == 'torch' and dtype == 'float16':
        return
    helpers.test_array_function(
        dtype, as_variable, with_out, num_positional_args, native_array, container, instance_method, fw, 'logaddexp',
        x1=np.asarray(x, dtype=dtype), x2=np.asarray(x, dtype=dtype))


# logical_and
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_logical_and(with_out, native_array):
    x1 = ivy.array([0, 1, 1], dtype='bool')
    x2 = ivy.array([0, 1, 1], dtype='bool')
    out = ivy.array([0, 0, 0], dtype='bool')
    if native_array:
        x1 = x1.data
        x2 = x2.data
        out = out.data
    if with_out:
        ret = ivy.logical_and(x1, x2, out=out)
    else:
        ret = ivy.logical_and(x1, x2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# logical_not
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_logical_not(with_out, native_array):
    x = ivy.array([0, 1, 1], dtype='bool')
    out = ivy.array([0, 0, 0], dtype='bool')
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.logical_not(x, out=out)
    else:
        ret = ivy.logical_not(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


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
    "dtype", ivy.all_numeric_dtype_strs)
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
    "dtype", ivy.all_numeric_dtype_strs)
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
    "dtype", ivy.all_numeric_dtype_strs)
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
    "dtype", ivy.all_numeric_dtype_strs)
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
    "dtype", ivy.all_numeric_dtype_strs)
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
    "dtype", ivy.all_numeric_dtype_strs)
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
    "dtype", ivy.all_numeric_dtype_strs)
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
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
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
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
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
    "dtype", ivy.all_numeric_dtype_strs)
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
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
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
    "dtype", ivy.all_numeric_dtype_strs)
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
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
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
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
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
    "dtype", ivy.all_numeric_dtype_strs)
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
@given(dtype=helpers.sample(ivy_np.valid_float_dtype_strs),
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
