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