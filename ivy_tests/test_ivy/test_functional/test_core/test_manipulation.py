"""
Collection of tests for sorting functions
"""

# global
import pytest

# local
import ivy


# concat
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_concat(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([2, 3, 4], dtype=dtype)
    x2 = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
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
        ret = ivy.concat([x1, x2], out=out)
    else:
        ret = ivy.concat([x1, x2])
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# expand_dims
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_expand_dims(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array([[2, 3, 4]], dtype=dtype)
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
        ret = ivy.expand_dims(x, out=out)
    else:
        ret = ivy.expand_dims(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# flip
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_flip(dtype, as_variable, with_out, native_array):
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
        ret = ivy.flip(x, out=out)
    else:
        ret = ivy.flip(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# permute_dims
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_permute_dims(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([[0, 1], [2, 3]], dtype=dtype)
    out = ivy.array([[0, 1], [2, 3]], dtype=dtype)
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
        ret = ivy.permute_dims(x, (1, 0), out=out)
    else:
        ret = ivy.permute_dims(x, (1, 0))
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# reshape
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_reshape(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3, 4], dtype=dtype)
    out = ivy.array([[2, 3], [4, 5]], dtype=dtype)
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
        ret = ivy.reshape(x, (2, 2), out=out)
    else:
        ret = ivy.reshape(x, (2, 2))
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# roll
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_roll(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([[1, 2], [3, 4]], dtype=dtype)
    out = ivy.array([[2, 3], [4, 5]], dtype=dtype)
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
        ret = ivy.roll(x, 2, out=out)
    else:
        ret = ivy.roll(x, 2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# squeeze
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_squeeze(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([[[1, 2], [3, 4]]], dtype=dtype)
    out = ivy.array([[2, 3], [4, 5]], dtype=dtype)
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
        ret = ivy.squeeze(x, 0, out=out)
    else:
        ret = ivy.squeeze(x, 0)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# stack
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_stack(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x1 = ivy.array([1, 2, 3, 4], dtype=dtype)
    x2 = ivy.array([1, 2, 3, 4], dtype=dtype)
    out = ivy.array([[2, 3, 4, 5], [4, 5, 6, 7]], dtype=dtype)
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
        ret = ivy.stack([x1, x2], 0, out=out)
    else:
        ret = ivy.stack([x1, x2], 0)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# Extra #
# ------#


# repeat
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_repeat(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'tensorflow' and dtype == 'uint16':
        pytest.skip("tf repeat doesnt allow uint16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
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
        ret = ivy.repeat(x, 2, out=out)
    else:
        ret = ivy.repeat(x, 2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# tile
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_tile(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'tensorflow' and dtype == 'uint16':
        pytest.skip("tf tile doesnt allow uint16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
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
        ret = ivy.tile(x, 2, out=out)
    else:
        ret = ivy.tile(x, 2)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# constant_pad
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_constant_pad(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
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
        ret = ivy.constant_pad(x, [[2, 1]], out=out)
    else:
        ret = ivy.constant_pad(x, [[2, 1]])
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# zero_pad
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_zero_pad(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 2, 3], dtype=dtype)
    out = ivy.array([2, 3, 4, 5, 6, 7], dtype=dtype)
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
        ret = ivy.zero_pad(x, [[2, 1]], out=out)
    else:
        ret = ivy.zero_pad(x, [[2, 1]])
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# swapaxes
@pytest.mark.parametrize(
    "dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_swapaxes(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([[1, 2], [3, 4]], dtype=dtype)
    out = ivy.array([[2, 3], [4, 5]], dtype=dtype)
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
        ret = ivy.swapaxes(x, 0, 1, out=out)
    else:
        ret = ivy.swapaxes(x, 0, 1)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# clip
@pytest.mark.parametrize(
    "dtype", ivy.all_numeric_dtype_strs)
@pytest.mark.parametrize(
    "as_variable", [True, False])
@pytest.mark.parametrize(
    "with_out", [True, False])
@pytest.mark.parametrize(
    "native_array", [True, False])
def test_clip(dtype, as_variable, with_out, native_array):
    if ivy.current_framework_str() == 'torch' and dtype == 'float16':
        pytest.skip("torch clamp doesnt allow float16")
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([[1, 2], [3, 4]], dtype=dtype)
    out = ivy.array([[2, 3], [4, 5]], dtype=dtype)
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
        ret = ivy.clip(x, 1, 3, out=out)
    else:
        ret = ivy.clip(x, 1, 3)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)
