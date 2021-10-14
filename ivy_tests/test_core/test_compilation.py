"""
Collection of tests for templated general functions
"""

# global
import time
import pytest
import numpy as np
from numbers import Number

# local
import ivy
import ivy.numpy
import ivy_tests.helpers as helpers


# Tests #
# ------#


# functions to compile
def _fn_1(x):
    return x**2


def _fn_2(x):
    return (x + 10)**0.5 - 5


def _fn_3(x):
    time.sleep(1)
    return ivy.reduce_mean(ivy.reduce_sum(x, keepdims=True), keepdims=True)


@pytest.mark.parametrize(
    "x", [[1], [[0.0, 1.0], [2.0, 3.0]]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_compile_native(x, dtype_str, tensor_fn, dev_str, call):
    if ivy.wrapped_mode():
        # Wrapped mode does not yet support function compilation
        pytest.skip()
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()

    # function 1
    x = tensor_fn(x, dtype_str, dev_str)
    comp_fn = ivy.compile_native(_fn_1)
    # type test
    assert callable(comp_fn)
    # value test
    non_compiled_return = _fn_1(x)
    compiled_return = comp_fn(x)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))

    # function 2
    x = tensor_fn(x, dtype_str, dev_str)
    comp_fn = ivy.compile_native(_fn_2)
    # type test
    assert callable(comp_fn)
    # value test
    non_compiled_return = _fn_2(x)
    compiled_return = comp_fn(x)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))


@pytest.mark.parametrize(
    "x", [[1]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_compile_ivy(x, dtype_str, tensor_fn, dev_str, call):
    if ivy.wrapped_mode():
        # Wrapped mode does not yet support function compilation
        pytest.skip()
    # smoke test
    if (isinstance(x, Number) or len(x) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()

    # function 3
    x = tensor_fn(x, dtype_str, dev_str)
    comp_fn = ivy.compile_ivy(_fn_3, x)
    # type test
    assert callable(comp_fn)
    # value test
    start_time = time.perf_counter()
    non_compiled_return = _fn_3(x)
    non_comp_time_taken = time.perf_counter() - start_time
    start_time = time.perf_counter()
    compiled_return = comp_fn(x)
    comp_time_taken = time.perf_counter() - start_time
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))
    assert comp_time_taken < non_comp_time_taken
