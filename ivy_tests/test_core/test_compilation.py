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
def _fn_1(x, with_non_compiled: bool = False):
    for _ in range(100000):
        pass
    if with_non_compiled:
        (x + 3) * 4  # ops not to be compiled into the graph
    return x**2


def _fn_2(x, with_non_compiled: bool = False):
    for _ in range(100000):
        pass
    if with_non_compiled:
        (x + 3) * 4  # ops not to be compiled into the graph
    return (x + 10)**0.5 - 5


def _fn_3(x, with_non_compiled: bool = False, with_internal_gen: bool = False):
    if with_internal_gen:
        x += ivy.array([1.])
    time.sleep(0.05)
    if with_non_compiled:
        (x + 3) * 4  # ops not to be compiled into the graph
    return ivy.reduce_mean(ivy.reduce_sum(x, keepdims=True), keepdims=True)


def _fn_4(x, with_non_compiled: bool = False, with_internal_gen: bool = False):
    if with_internal_gen:
        x += ivy.array([1.])
    y = ivy.reduce_mean(x)
    z = ivy.reduce_sum(x)
    f = ivy.reduce_var(y)
    time.sleep(0.05)
    k = ivy.cos(z)
    m = ivy.sin(f)
    o = ivy.tan(y)
    if with_non_compiled:
        (x + 3) * 4  # ops not to be compiled into the graph
    return ivy.concatenate([k, m, o], -1)


def _wide_fn(x, with_non_compiled: bool = False, with_internal_gen: bool = False):
    if with_internal_gen:
        x += ivy.array([1.])
    if with_non_compiled:
        (x + 3) * 4  # ops not to be compiled into the graph
    graph_width = 10
    xs = [x]*graph_width
    for i in range(5):
        xs = [x_ + j for x_, j in zip(xs, range(graph_width))]
    return ivy.concatenate(xs, 0)


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
    comp_fn = ivy.compile_native(_fn_1)
    # type test
    assert callable(comp_fn)
    # value test
    x = tensor_fn(x, dtype_str, dev_str)
    non_compiled_return = _fn_1(x)
    x = tensor_fn(x, dtype_str, dev_str)
    compiled_return = comp_fn(x)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))

    # function 2
    comp_fn = ivy.compile_native(_fn_2)
    # type test
    assert callable(comp_fn)
    # value test
    x = tensor_fn(x, dtype_str, dev_str)
    non_compiled_return = _fn_2(x)
    x = tensor_fn(x, dtype_str, dev_str)
    compiled_return = comp_fn(x)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize(
    "x_raw", [[1]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
@pytest.mark.parametrize(
    "with_non_compiled", [True, False])
def test_compile_ivy_inplace(x_raw, dtype_str, tensor_fn, with_non_compiled, dev_str, call):
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
    comp_fn = ivy.compile_ivy(_fn_1, x, with_non_compiled)
    # type test
    assert callable(comp_fn)
    # value test
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    non_compiled_return = _fn_1(x, with_non_compiled)
    non_comp_time_taken = time.perf_counter() - start_time
    assert len(comp_fn.__self__._param_dict) == 2
    assert comp_fn.__self__.params_all_empty()
    assert len(list(comp_fn.__self__._functions)) == 1
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    compiled_return = comp_fn(x, with_non_compiled)
    comp_time_taken = time.perf_counter() - start_time
    assert len(comp_fn.__self__._param_dict) == 2
    assert comp_fn.__self__.params_all_empty()
    assert len(list(comp_fn.__self__._functions)) == 1
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))
    assert comp_time_taken < non_comp_time_taken

    # function 2
    x = tensor_fn(x_raw, dtype_str, dev_str)
    comp_fn = ivy.compile_ivy(_fn_2, x, with_non_compiled)
    # type test
    assert callable(comp_fn)
    # value test
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    non_compiled_return = _fn_2(x, with_non_compiled)
    non_comp_time_taken = time.perf_counter() - start_time
    assert len(comp_fn.__self__._param_dict) == 4
    assert comp_fn.__self__.params_all_empty()
    assert len(list(comp_fn.__self__._functions)) == 3
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    compiled_return = comp_fn(x, with_non_compiled)
    comp_time_taken = time.perf_counter() - start_time
    assert len(comp_fn.__self__._param_dict) == 4
    assert comp_fn.__self__.params_all_empty()
    assert len(list(comp_fn.__self__._functions)) == 3
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))
    assert comp_time_taken < non_comp_time_taken


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize(
    "x_raw", [[1]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
@pytest.mark.parametrize(
    "with_non_compiled", [True, False])
@pytest.mark.parametrize(
    "with_internal_gen", [True, False])
def test_compile_ivy(x_raw, dtype_str, tensor_fn, with_non_compiled, with_internal_gen, dev_str, call):
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
    comp_fn = ivy.compile_ivy(_fn_3, x, with_non_compiled, with_internal_gen)
    # type test
    assert callable(comp_fn)
    # value test
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    non_compiled_return = _fn_3(x, with_non_compiled, with_internal_gen)
    non_comp_time_taken = time.perf_counter() - start_time
    assert len(comp_fn.__self__._param_dict) == 3 + (1 if with_internal_gen else 0)
    assert comp_fn.__self__.params_all_empty()
    assert len(list(comp_fn.__self__._functions)) == 2 + (1 if with_internal_gen else 0)
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    compiled_return = comp_fn(x, with_non_compiled, with_internal_gen)
    comp_time_taken = time.perf_counter() - start_time
    assert len(comp_fn.__self__._param_dict) == 3 + (1 if with_internal_gen else 0)
    assert comp_fn.__self__.params_all_empty()
    assert len(list(comp_fn.__self__._functions)) == 2 + (1 if with_internal_gen else 0)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))
    assert comp_time_taken < non_comp_time_taken

    # function 4
    x = tensor_fn(x_raw, dtype_str, dev_str)
    comp_fn = ivy.compile_ivy(_fn_4, x, with_non_compiled, with_internal_gen)
    # type test
    assert callable(comp_fn)
    # value test
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    non_compiled_return = _fn_4(x, with_non_compiled, with_internal_gen)
    non_comp_time_taken = time.perf_counter() - start_time
    assert len(comp_fn.__self__._param_dict) == 11 + (1 if with_internal_gen else 0)
    assert comp_fn.__self__.params_all_empty()
    assert len(list(comp_fn.__self__._functions)) == 10 + (1 if with_internal_gen else 0)
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    compiled_return = comp_fn(x, with_non_compiled, with_internal_gen)
    comp_time_taken = time.perf_counter() - start_time
    assert len(comp_fn.__self__._param_dict) == 11 + (1 if with_internal_gen else 0)
    assert comp_fn.__self__.params_all_empty()
    assert len(list(comp_fn.__self__._functions)) == 10 + (1 if with_internal_gen else 0)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(compiled_return))
    assert comp_time_taken < non_comp_time_taken


# noinspection PyUnresolvedReferences
@pytest.mark.parametrize(
    "x_raw", [[0]])
@pytest.mark.parametrize(
    "dtype_str", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array])
@pytest.mark.parametrize(
    "with_non_compiled", [True, False])
@pytest.mark.parametrize(
    "with_internal_gen", [True, False])
def test_compile_ivy_multiproc(x_raw, dtype_str, tensor_fn, with_non_compiled, with_internal_gen, dev_str, call):
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

    # non-compiled
    x = tensor_fn(x_raw, dtype_str, dev_str)
    non_compiled_return = _wide_fn(x, with_non_compiled, with_internal_gen)

    # compiled single threaded
    st_comp_fn = ivy.compile_ivy(_wide_fn, x, with_non_compiled, with_internal_gen, num_workers=1)
    assert callable(st_comp_fn)
    assert len(st_comp_fn.__self__._param_dict) == 52 + (1 if with_internal_gen else 0)
    assert st_comp_fn.__self__.params_all_empty()
    assert len(list(st_comp_fn.__self__._functions)) == 51 + (1 if with_internal_gen else 0)
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    st_return = st_comp_fn(x, with_non_compiled, with_internal_gen)
    st_time_taken = time.perf_counter() - start_time
    assert len(st_comp_fn.__self__._param_dict) == 52 + (1 if with_internal_gen else 0)
    assert st_comp_fn.__self__.params_all_empty()
    assert len(list(st_comp_fn.__self__._functions)) == 51 + (1 if with_internal_gen else 0)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(st_return))

    # compiled multi-processing
    multi_comp_fn = ivy.compile_ivy(_wide_fn, x, with_non_compiled, with_internal_gen, num_workers=ivy.num_cpu_cores())
    assert callable(multi_comp_fn)
    assert len(multi_comp_fn.__self__._param_dict) == 52 + (1 if with_internal_gen else 0)
    assert multi_comp_fn.__self__.params_all_empty()
    assert len(list(multi_comp_fn.__self__._functions)) == 51 + (1 if with_internal_gen else 0)
    start_time = time.perf_counter()
    x = tensor_fn(x_raw, dtype_str, dev_str)
    multi_return = multi_comp_fn(x, with_non_compiled, with_internal_gen)
    multi_time_taken = time.perf_counter() - start_time
    assert len(multi_comp_fn.__self__._param_dict) == 52 + (1 if with_internal_gen else 0)
    assert multi_comp_fn.__self__.params_all_empty()
    assert len(list(multi_comp_fn.__self__._functions)) == 51 + (1 if with_internal_gen else 0)
    assert np.allclose(ivy.to_numpy(non_compiled_return), ivy.to_numpy(multi_return))

    # assert multi-processing was faster
    # ToDo: try to improve multi-processing and make this more efficient, perhaps by removing costly manager dict
    # assert multi_time_taken < st_time_taken
