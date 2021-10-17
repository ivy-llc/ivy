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
    for _ in range(100000):
        pass
    return x**2


def _fn_2(x):
    for _ in range(100000):
        pass
    return (x + 10)**0.5 - 5


def _fn_3(x):
    time.sleep(0.05)
    return ivy.reduce_mean(ivy.reduce_sum(x, keepdims=True), keepdims=True)

def _fn_4(x):
    y = ivy.reduce_mean(x)
    z = ivy.reduce_sum(x)
    f = ivy.reduce_var(y)
    time.sleep(0.05)
    k = ivy.cos(z)
    m = ivy.sin(f)
    o = ivy.tan(y)
    return ivy.concatenate([k, m, o], -1)


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


'''
@pytest.mark.parametrize(
    "x_raw", [[1]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_compile_ivy_inplace(x_raw, dtype_str, tensor_fn, dev_str, call):
    if ivy.wrapped_mode():
        # Wrapped mode does not yet support function compilation
        pytest.skip()
    if call is not helpers.torch_call:
        # currently only supported by PyTorch
        pytest.skip()
    # smoke test
    if (isinstance(x_raw, Number) or len(x_raw) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()

    # function 1
    x = tensor_fn(x_raw, dtype_str, dev_str)
    comp_fn = ivy.compile_ivy(_fn_1, x)
    # type test
    assert callable(comp_fn)
    # value test
    start_time = time.perf_counter()
    non_compiled_return = _fn_1(x)
    non_comp_time_taken = time.perf_counter() - start_time
    start_time = time.perf_counter()
    compiled_return = comp_fn(x)
    comp_time_taken = time.perf_counter() - start_time
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))
    assert comp_time_taken < non_comp_time_taken

    # function 2
    x = tensor_fn(x_raw, dtype_str, dev_str)
    comp_fn = ivy.compile_ivy(_fn_2, x)
    # type test
    assert callable(comp_fn)
    # value test
    start_time = time.perf_counter()
    non_compiled_return = _fn_2(x)
    non_comp_time_taken = time.perf_counter() - start_time
    start_time = time.perf_counter()
    compiled_return = comp_fn(x)
    comp_time_taken = time.perf_counter() - start_time
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))
    assert comp_time_taken < non_comp_time_taken
'''


@pytest.mark.parametrize(
    "x_raw", [[1]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
def test_compile_ivy(x_raw, dtype_str, tensor_fn, dev_str, call):
    if ivy.wrapped_mode():
        # Wrapped mode does not yet support function compilation
        pytest.skip()
    if call is not helpers.torch_call:
        # currently only supported by PyTorch
        pytest.skip()
    # smoke test
    if (isinstance(x_raw, Number) or len(x_raw) == 0) and tensor_fn == helpers.var_fn and call is helpers.mx_call:
        # mxnet does not support 0-dimensional variables
        pytest.skip()

    # function 3
    x = tensor_fn(x_raw, dtype_str, dev_str)
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

    # function 4
    '''
    x = tensor_fn(x_raw, dtype_str, dev_str)
    comp_fn = ivy.compile_ivy(_fn_4, x)
    # type test
    assert callable(comp_fn)
    # value test
    start_time = time.perf_counter()
    non_compiled_return = _fn_4(x)
    non_comp_time_taken = time.perf_counter() - start_time
    start_time = time.perf_counter()
    compiled_return = comp_fn(x)
    comp_time_taken = time.perf_counter() - start_time
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))
    assert comp_time_taken < non_comp_time_taken
    '''
