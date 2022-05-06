"""Collection of tests for statstical functions."""
# global
import pytest
import numpy as np
from hypothesis import given, assume, strategies as st
# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# min
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_min(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    assume(x)
    assume(dtype not in ivy.invalid_dtype_strs)
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "min",
        x=np.asarray(x, dtype=dtype),
    )


# max
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_numeric_dtype_strs),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_max(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x = dtype_and_x
    assume(x)
    assume(dtype not in ivy.invalid_dtype_strs)
    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "max",
        x=np.asarray(x, dtype=dtype),
    )


# mean
# @given(
#     dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtype_strs),
#     as_variable=st.booleans(),
#     with_out=st.booleans(),
#     num_positional_args=st.integers(0, 1),
#     native_array=st.booleans(),
#     container=st.booleans(),
#     instance_method=st.booleans(),
# )
# def test_mean(
#     dtype_and_x,
#     as_variable,
#     with_out,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     fw,
# ):
#     dtype, x = dtype_and_x
#     assume(x)
#     helpers.test_array_function(
#         dtype,
#         as_variable,
#         with_out,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "mean",
#         x=np.asarray(x, dtype=dtype),
#     )


# var
@pytest.mark.parametrize("dtype", ivy.float_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_var(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([1, 10, 25], dtype=dtype)
    out = ivy.array(98, dtype=dtype)
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
        ret = ivy.var(x, out=out)
    else:
        ret = ivy.var(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# prod
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_prod(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs or (
        ivy.backend == "torch" and (dtype == "bfloat16" or dtype == "float16")
    ):
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array(24, dtype=dtype)
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
        ret = ivy.prod(x, out=out)
    else:
        ret = ivy.prod(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# sum
@pytest.mark.parametrize("dtype", ivy.all_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_sum(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array(9, dtype=dtype)
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
        ret = ivy.sum(x, out=out)
    else:
        ret = ivy.sum(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# std
@pytest.mark.parametrize("dtype", ivy.float_dtype_strs)
@pytest.mark.parametrize("as_variable", [True, False])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("native_array", [True, False])
def test_std(dtype, as_variable, with_out, native_array):
    if dtype in ivy.invalid_dtype_strs:
        pytest.skip("invalid dtype")
    x = ivy.array([2, 3, 4], dtype=dtype)
    out = ivy.array(9, dtype=dtype)
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
        ret = ivy.std(x, out=out)
    else:
        ret = ivy.std(x)
    if with_out:
        if not native_array:
            assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is (out if native_array else out.data)


# einsum
@pytest.mark.parametrize(
    "eq_n_op_n_shp",
    [
        ("ii", (np.arange(25).reshape(5, 5),), ()),
        ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
        ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,)),
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("with_out", [True, False])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_einsum(eq_n_op_n_shp, dtype, with_out, tensor_fn, device, call):
    # smoke test
    eq, operands, true_shape = eq_n_op_n_shp
    operands = [tensor_fn(op, dtype, device) for op in operands]
    if with_out:
        out = ivy.zeros(true_shape, dtype=dtype)
        ret = ivy.einsum(eq, *operands, out=out)
    else:
        ret = ivy.einsum(eq, *operands)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_shape
    # value test
    assert np.allclose(
        call(ivy.einsum, eq, *operands),
        ivy.functional.backends.numpy.einsum(
            eq, *[ivy.to_numpy(op) for op in operands]
        ),
    )
    # out test
    if with_out:
        assert ret is out
        if ivy.current_framework_str() in ["tensorflow", "jax"]:
            # these frameworks do not support native inplace updates
            return
        assert ret.data is out.data
