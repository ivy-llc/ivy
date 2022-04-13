"""
Collection of tests for elementwise functions
"""

# global
import pytest

# local
import ivy


# abs
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_abs(dtype, as_variable, with_out, native_array):
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
        ret = ivy.abs(x, out=out)
    else:
        ret = ivy.abs(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# acosh
@pytest.mark.parametrize(
    "dtype", ivy.float_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_acosh(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'torch' and dtype == 'float16':
        pytest.skip("torch acosh doesnt allow float16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.acosh(x, out=out)
    else:
        ret = ivy.acosh(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# acos
@pytest.mark.parametrize(
    "dtype", ivy.float_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_acos(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'torch' and dtype == 'float16':
        pytest.skip("torch acos doesnt allow float16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([0.5, 0.8, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.acos(x, out=out)
    else:
        ret = ivy.acos(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# add
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
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
        out = ivy.variable(out)
    if native_array:
        x = x.data
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
@pytest.mark.parametrize(
    "dtype", ivy.float_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_asin(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'torch' and dtype == 'float16':
        pytest.skip("torch asin doesnt allow float16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([0.5, 0.8, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.asin(x, out=out)
    else:
        ret = ivy.asin(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# asinh
@pytest.mark.parametrize(
    "dtype", ivy.float_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_asinh(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'torch' and dtype == 'float16':
        pytest.skip("torch asinh doesnt allow float16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([0.5, 0.8, 4], dtype=dtype)
    out = ivy.array([2, 3, 4], dtype=dtype)
    if as_variable:
        if with_out:
            pytest.skip("variables do not support out argument")
        x = ivy.variable(x)
        out = ivy.variable(out)
    if native_array:
        x = x.data
        out = out.data
    if with_out:
        ret = ivy.asinh(x, out=out)
    else:
        ret = ivy.asinh(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)